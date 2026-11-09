Incremental decode EPLB
=======================

Activation
----------

Select this runtime with ``--enable-eplb --eplb-algorithm incremental``.
The initial integration targets DeepSeek V4.1 Flash with DeepEP low-latency
decode, optionally DSPARK. For EP32, the additional required options are::

    --ep-num-redundant-experts 64
    --disable-shared-experts-fusion
    --moe-a2a-backend deepep
    --deepep-mode auto
    --num-continuous-decode-steps 1

``--ep-dispatch-algorithm static`` is resolved automatically. Decode CUDA
graphs must remain enabled. ``low_latency`` is also supported as the DeepEP
mode. The spare count is exactly two slots per EP rank, so it changes with EP
size. The runtime rejects windowed async EPLB, external placement files,
distribution recorders, elastic EP, LoRA, fused shared-expert storage, PDMux,
two/single-batch overlap, prefill-only servers, and online model-weight updates.

Policy constants
----------------

Planner tuning values are grouped at the top of ``config.py``. They are source
constants, not server flags; use identical values on every EP rank.

* ``CPU_PLANNER_THREADS = 4`` sets the OpenMP threads per EP process.
  Planning parallelizes across layers; decisions within a layer are deterministic.
* ``SPLIT_TARGET = 0.6`` and ``MERGE_TARGET = 0.4`` are relative to mean rank
  token load. Growth targets ``ceil(expert_load / (SPLIT_TARGET * mean))``
  owners; shrinking uses the analogous merge target only if it reduces the
  current owner count. Available slots and copy budgets can delay reaching it.
* ``MAX_COPIES_PER_LAYER_PER_STEP = 2`` caps new destination copies across the
  entire EP group, for each layer and target decode step. Resident-copy reuse
  costs zero. Split placement and exchange repair share this budget.
* ``MAX_OWNER_MOVES_PER_LAYER_PER_STEP = 32`` caps repair-phase relocations;
  a swap uses two. It does not cap split/merge membership changes.
* ``EXCHANGE_GAIN_DIVISOR = 1000`` sets the repair gain floor to approximately
  mean rank load squared divided by this value. Increasing it permits smaller
  gains and can increase placement churn.

``--ep-num-redundant-experts`` controls total extra physical storage slots,
not copies per step or owners per expert. This implementation requires two
extra slots per rank and reserves at least one inactive slot for safe staging.
Home slots can be reused; owners are not restricted to the extra slots.

Lifecycle
---------

At each target decode boundary, snapshot the preceding decode's counts with
the placement that produced them. Activate the completed generation and
enqueue the decode graph, then plan the next placement on the background worker. Expert splits are sized against mean rank load,
with split/merge hysteresis and a bounded number of copies per layer.
Inactive home slots and spare slots belong to the same placement space.
Resident experts are reused without another copy.

The CPU planner reads pinned snapshots of the authoritative GPU state and
writes reusable pinned output buffers. It releases the Python GIL while planning.
The outputs return to the planner stream for the existing GPU publication and
routing kernels; there is no separate CPU placement generation or GPU planner
fallback. The CPU extension is compiled lazily with a C++ compiler and OpenMP.
Use ``OMP_WAIT_POLICY=PASSIVE`` to avoid idle OpenMP workers spinning between
steps. CPU synchronization stays on the worker, after decode has been enqueued.

The worker consumes the publication, transfers only into reserved inactive
slots, and stages adoption after every EP rank completes the generation.
The next boundary installs the prepared routing. If a submission is late,
that boundary still waits: this implementation does not promise completely
hidden transfer latency. A transfer failure prevents route publication and
is propagated to the inference thread.

Prefill does not split, collect counts, or advance the planner. It follows
the current primary-owner map because a reused home slot may no longer
contain its original expert. Draft construction and routing remain outside
the target placement context. Separate shared modules are excluded by the
same predicate used for allocation and weight discovery.

Transport integration
---------------------

The planner, placement, routing, and diagnostics are shared. Transport is
injected by the model-runner adapter, without another command-line flag.

* ``init_incremental_eplb(..., transport=None)`` preserves the original peer copy
  engine and peer-memory count/completion exchange. It requires a supported
  symmetric-memory fabric with directly accessible peer mappings. This is
  a fabric-specific adapter choice; ordinary multi-node CUDA clusters must
  not assume they provide that fabric.
* ``transport_factory=lambda: MooncakeTransport.create(gpu_id=...)`` registers local arenas
  and reads source regions directly into reserved execution slots. Remote
  addresses are used only by Mooncake, not dereferenced by CUDA. Count and
  generation collectives run on the worker using a dedicated EP Gloo group.
  The default upstream model-runner adapter selects this path.

Mooncake registration and one-sided read handling follow SGLang PR #45282,
revision ``776f60a2af54d7d07472709f70d738d9286b564a``. Its staging-layer
installation loop is not used here. The engine must support GPU memory
registration and ``batch_transfer_sync_read`` on the selected fabric.
Registration/transfer errors are fatal; there is no silent fallback.

For the Mooncake NVLink/MNNVL path, set ``MOONCAKE_PROTOCOL=nvlink`` and
``MC_FORCE_MNNVL=1``. Unset ``MC_FORCE_HCA``, ``MC_FORCE_TCP``,
``MC_INTRANODE_NVLINK`` and ``MC_INTRA_NVLINK``; merely setting these to zero
may still select them in Mooncake. The explicit transport override is required
because topology discovery can otherwise choose RDMA despite the protocol
argument. Expert arenas use SGLang's shared Mooncake NVLink memory-pool helper;
failure to create that pool aborts initialization. Pool lifetime follows the
transport and its registered arenas. Cross-node support still depends on the
installed Mooncake build and fabric; local NVLink success is not sufficient.

Diagnostics
-----------

Set ``SGLANG_INCREMENTAL_EPLB_DEBUG_DIR`` to enable diagnostics. It creates
``balance/`` for per-rank expert residency, owners, tokens and GPU phases,
and ``transfers/`` for moved expert IDs, source/destination slots, bytes and
publication/copy timings. With a directory set, diagnostics record every target
decode replay, including its actual batch size. An empty directory setting
(the default) disables instrumentation. Sampling and filtering belong in external
test tooling; the serving runtime has no sampling-window or alternate-routing
options.

Individual grouped-GEMM timings are available when the backend exposes the
instrumented two-GEMM wrappers. Otherwise they are null, not zero; the total
expert-compute interval remains available. Per-expert GEMM latency cannot be
inferred by dividing a grouped kernel's duration by its token counts.
Instrumented runs are diagnostics, not throughput baselines.

Validation status
-----------------

CPU tests cover option rejection before model loading, placement isolation,
transfer ordering, separate worker collectives, draft buffer capacity and
bounded diagnostics. Device-side invariant checks remain fail-stop: continuing
with invalid ownership could read or overwrite the wrong expert weights.
Count exchanges include generation checks on the Mooncake path; a missing
rank still relies on the process-group timeout. Peer-memory kernels require
lockstep EP boundaries; reused generations fail immediately and missing
generations fail after a bounded device wait (30 seconds at the reported clock).

JIT builds use PyTorch's extension cache; Mooncake does not compile peer-memory
exchange code. Peer transport requires hardware validation on each peer-mapping fabric.
The GPU-planner implementation of both transports passed EP32 DeepSeek V4.1 Flash serving tests on PPU,
including DSPARK, real prefill and decode, and full-batch server-log timing.
Mooncake NVLink/MNNVL additionally passed a 32-rank exact-weight transfer and
routing test covering publication, home-slot reuse and deregistration. These
transport results apply to the tested PPU fabric and installed Mooncake build; NVIDIA
cross-node validation remains outstanding. Model accuracy was measured with
the GPU planner and peer transport; the CPU planner and Mooncake adapter still
require their own full-model accuracy validation. CPU planning passed exact
placement and 32-rank peer-copy checks and one PPU serving comparison before
this build/interface cleanup; those results do not replace testing the cleaned
runtime.
The DeepEP build must expose
``cumulative_local_expert_recv_stats`` in ``low_latency_dispatch``; changing
weight transport does not remove this requirement.

Draft expert placement is isolated from the target independently of EPLB.
DSPARK switches between target and draft DeepEP layouts and cleans the incoming
layout on each switch, whether EPLB is enabled or disabled. An unchanged layout
needs no extra clean. Include layout-switch costs in performance measurements.
The planner accepts at most 2**24 routed assignments per layer per generation
(for example, EP64 x 4096 dispatch tokens/rank x top-64). Larger totals fail
before placement changes; threshold and squared-load products fit int64.

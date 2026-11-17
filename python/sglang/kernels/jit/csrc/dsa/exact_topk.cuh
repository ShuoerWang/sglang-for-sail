#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace sglang::dsa {
namespace {

using device::AlignedVector;
using device::kWarpSize;
namespace warp = device::warp;

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
constexpr uint32_t kMaxScoreTopK = 8192;

struct TopKRadixBin {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

/// Refine the k-th largest key using the counts of this radix digit.
template <uint32_t kBins>
SGL_DEVICE void find_topk_radix_bin(
    const uint32_t* hist,
    const uint32_t total_active,
    const uint32_t topk_remain,
    uint32_t* warp_sum,
    TopKRadixBin& match) {
  constexpr uint32_t kThreads = kBins < kBlockSize ? kWarpSize : kBlockSize;
  constexpr uint32_t kItems = (kBins + kThreads - 1) / kThreads;
  const auto tx = threadIdx.x;
  if (tx < kThreads) {
    const auto lane = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    uint32_t counts[kItems];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const auto bin = tx * kItems + i;
      counts[i] = bin < kBins ? hist[bin] : 0u;
      local_sum += counts[i];
    }
    const auto warp_inc = warp::inclusive_sum(local_sum, lane);
    uint32_t prefix = warp_inc - local_sum;
    if constexpr (kThreads == kBlockSize) {
      if (lane == kWarpSize - 1) warp_sum[warp_id] = warp_inc;
      __syncthreads();
      prefix += warp::reduce_sum(lane < warp_id ? warp_sum[lane] : 0u);
    }
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix += counts[i];
      const auto above = total_active - prefix;
      if (above < topk_remain && above + counts[i] >= topk_remain) {
        match = {tx * kItems + i, above, counts[i]};
      }
    }
  }
  __syncthreads();
}

// Ragged selection with positions written in ascending order. The radix
// passes only keep counts; the output pass can consume an arbitrarily large
// group of equal keys without staging the indices in shared memory.
template <typename Score>
struct OrderedTopK {
  static constexpr bool kBF16 = std::is_same_v<Score, uint16_t>;
  static constexpr uint32_t kVecSize = 16 / sizeof(Score);
  static constexpr uint32_t kBins = kBF16 ? 8192 : 4096;
  using Key = std::conditional_t<kBF16, uint16_t, uint32_t>;
  using KeyVec = AlignedVector<Key, kVecSize>;
  using Mask = std::conditional_t<kBF16, uint16_t, uint8_t>;
  static constexpr uint32_t kMaskCapacity = kBins * sizeof(uint32_t) / sizeof(Mask);

  struct Smem {
    union {
      alignas(16) uint32_t histogram[kBins];
      Mask masks[kMaskCapacity];
    };
    TopKRadixBin match;
    uint32_t warp_counts[kNumWarps];
    uint32_t warp_ties[kNumWarps];
  };

  struct Row {
    const Score* scores;
    uint32_t length;
    int32_t misalignment;

    SGL_DEVICE int32_t start(uint32_t group) const {
      return static_cast<int32_t>(group * kVecSize) - misalignment;
    }

    SGL_DEVICE uint32_t groups() const {
      return (length + misalignment + kVecSize - 1) / kVecSize;
    }

    SGL_DEVICE static Key key(Score score) {
      if constexpr (kBF16) {
        return (score & 0x8000u) ? (score ^ 0xffffu) : (score | 0x8000u);
      } else {
        auto bits = __float_as_uint(score);
        if ((bits & 0x7fffffffu) == 0) bits = 0;
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
      }
    }

    SGL_DEVICE uint32_t load(uint32_t group, KeyVec& keys) const {
      const auto begin = start(group);
      if (begin >= 0 && static_cast<uint32_t>(begin) + kVecSize <= length) {
        AlignedVector<Score, kVecSize> values;
        values.load(scores + begin);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) keys[j] = key(values[j]);
        return (1u << kVecSize) - 1;
      }
      uint32_t mask = 0;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto index = static_cast<uint32_t>(begin) + j;
        const bool valid = index < length;
        keys[j] = valid ? key(scores[index]) : 0;
        mask |= static_cast<uint32_t>(valid) << j;
      }
      return mask;
    }
  };

  template <uint32_t kShift, uint32_t kDigitBins, typename Visit>
  SGL_DEVICE static void refine(
      Smem& smem, const Visit& visit, uint32_t& lower, uint32_t& active, uint32_t& remaining) {
    for (uint32_t i = threadIdx.x; i < kDigitBins; i += kBlockSize) smem.histogram[i] = 0;
    __syncthreads();
    visit([&](uint32_t key) {
      const auto digit = (key - lower) >> kShift;
      if (digit < kDigitBins) atomicAdd(smem.histogram + digit, 1u);
    });
    __syncthreads();
    find_topk_radix_bin<kDigitBins>(
        smem.histogram, active, remaining, smem.warp_counts, smem.match);
    lower |= smem.match.bin << kShift;
    active = smem.match.equal_count;
    remaining -= smem.match.above_count;
  }

  struct Cutoff {
    uint32_t key;
    uint32_t equal_quota;
    bool include_equal;
  };

  template <typename Visit>
  SGL_DEVICE static Cutoff find_cutoff(Smem& smem, const Visit& visit, uint32_t length, uint32_t topk) {
    uint32_t threshold = 0, active = length, remaining = topk;
    if constexpr (kBF16) {
      refine<3, 8192>(smem, visit, threshold, active, remaining);
      if (remaining != active) refine<0, 8>(smem, visit, threshold, active, remaining);
    } else {
      refine<20, 4096>(smem, visit, threshold, active, remaining);
      if (remaining != active) refine<8, 4096>(smem, visit, threshold, active, remaining);
      if (remaining != active) refine<0, 256>(smem, visit, threshold, active, remaining);
    }
    const bool include_equal = remaining == active;
    return {threshold, include_equal ? 0u : remaining, include_equal};
  }

  SGL_DEVICE static Mask selection_mask(
      const KeyVec& keys, uint32_t valid, uint32_t threshold, bool include_equal) {
    uint32_t above = 0, equal = 0;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      above |= static_cast<uint32_t>(keys[j] > threshold || (include_equal && keys[j] == threshold)) << j;
      equal |= static_cast<uint32_t>(!include_equal && keys[j] == threshold) << j;
    }
    return (above & valid) | ((equal & valid) << kVecSize);
  }

  template <uint32_t kItems>
  SGL_DEVICE static void forward(
      const Score* scores, uint32_t length, uint32_t topk, int32_t offset, int32_t* out, Smem& smem) {
    constexpr uint32_t kVectorMask = (1u << kVecSize) - 1;
    const Row row{scores, length, static_cast<int32_t>((reinterpret_cast<uintptr_t>(scores) & 15) / sizeof(Score))};
    const auto thread = threadIdx.x;
    const auto lane = thread % kWarpSize;
    const auto warp_id = thread / kWarpSize;
    const auto groups = row.groups();
    const auto stripes = (groups + kBlockSize - 1) / kBlockSize;
    const auto first = warp_id * stripes * kWarpSize + lane;
    const auto end = min(first - lane + stripes * kWarpSize, groups);

    // Warps own consecutive spans so their prefix counts also resolve ties
    // in input order. Short rows keep their score keys across all radix passes.
    KeyVec keys[kItems == 0 ? 1 : kItems];
    uint32_t valid[kItems == 0 ? 1 : kItems];
    if constexpr (kItems != 0) {
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        const auto group = first + i * kWarpSize;
        if (group < end) {
          valid[i] = row.load(group, keys[i]);
        } else {
          keys[i].fill(0);
          valid[i] = 0;
        }
      }
    }

    const auto visit = [&](const auto& count) {
      const auto count_keys = [&](const KeyVec& vector, uint32_t mask) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (mask & (1u << j)) count(vector[j]);
        }
      };
      if constexpr (kItems != 0) {
#pragma unroll
        for (uint32_t i = 0; i < kItems; ++i) count_keys(keys[i], valid[i]);
      } else {
        for (uint32_t group = thread; group < groups; group += kBlockSize) {
          KeyVec vector;
          const auto mask = row.load(group, vector);
          count_keys(vector, mask);
        }
      }
    };

    const auto [threshold, quota, include_equal] = find_cutoff(smem, visit, length, topk);

    Mask local_masks[kItems == 0 ? 1 : kItems];
    const bool cache_masks = groups <= kMaskCapacity;
    const auto load_mask = [&](uint32_t group) {
      KeyVec vector;
      const auto mask = row.load(group, vector);
      return selection_mask(vector, mask, threshold, include_equal);
    };
    uint32_t above = 0, equal = 0;
    const auto count = [&](uint32_t mask) {
      above += __popc(mask & kVectorMask);
      equal += __popc(mask >> kVecSize);
    };
    if constexpr (kItems != 0) {
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        local_masks[i] = selection_mask(keys[i], valid[i], threshold, include_equal);
        count(local_masks[i]);
      }
    } else {
      for (uint32_t group = first; group < end; group += kWarpSize) {
        const auto mask = load_mask(group);
        if (cache_masks) smem.masks[group] = mask;
        count(mask);
      }
    }
    above = warp::reduce_sum(above);
    equal = warp::reduce_sum(equal);
    if (lane == 0) {
      smem.warp_counts[warp_id] = above;
      smem.warp_ties[warp_id] = equal;
    }
    __syncthreads();

    uint32_t earlier_equal = warp::reduce_sum(lane < warp_id ? smem.warp_ties[lane] : 0u);
    uint32_t output = warp::reduce_sum(lane < warp_id ? smem.warp_counts[lane] : 0u) + min(quota, earlier_equal);
    const auto emit = [&](uint32_t mask, uint32_t group) {
      auto tied = mask >> kVecSize;
      auto selected = mask & kVectorMask;
      const auto n_equal = static_cast<uint32_t>(__popc(tied));
      const auto eq_scan = warp::inclusive_sum(n_equal, lane);
      const auto eq_before = earlier_equal + eq_scan - n_equal;
      const auto take = min(n_equal, quota - min(quota, eq_before));
      if (take == n_equal) {
        selected |= tied;
      } else {
        for (uint32_t i = 0; i < take; ++i) {
          selected |= tied & -tied;
          tied &= tied - 1;
        }
      }
      const auto n_selected = static_cast<uint32_t>(__popc(selected));
      const auto scan = warp::inclusive_sum(n_selected, lane);
      auto dest = output + scan - n_selected;
      const auto start = row.start(group);
      while (selected != 0) {
        out[dest++] = offset + (start + __ffs(selected) - 1);
        selected &= selected - 1;
      }
      output += warp::broadcast(scan, kWarpSize - 1);
      earlier_equal += warp::broadcast(eq_scan, kWarpSize - 1);
    };
    if constexpr (kItems != 0) {
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (i >= stripes) break;
        emit(local_masks[i], first + i * kWarpSize);
      }
    } else {
      for (uint32_t stripe = first - lane; stripe < end; stripe += kWarpSize) {
        const auto group = stripe + lane;
        const auto mask = group < end ? (cache_masks ? smem.masks[group] : load_mask(group)) : 0u;
        emit(mask, group);
      }
    }
  }
};

template <typename Score>
struct ExactTopKParams {
  const Score* __restrict__ scores;
  const int32_t* __restrict__ lengths;
  const int32_t* __restrict__ offsets;
  int32_t* __restrict__ out_indices;
  int64_t score_stride;
  uint32_t topk;
  uint32_t force_last;
};

template <bool kUsePDL, typename Score, uint32_t kResidentItems = 0>
__global__ __launch_bounds__(kBlockSize, kResidentItems == 0 ? 2 : 1) void exact_topk_kernel(
    const __grid_constant__ ExactTopKParams<Score> params) {
  device::enable_smem_spilling();
  const auto row = blockIdx.x;
  const auto length = static_cast<uint32_t>(params.lengths[row]);
  const auto offset = params.offsets ? params.offsets[row] : 0;
  const auto topk = params.topk;
  auto* out = params.out_indices + row * static_cast<int64_t>(topk);

  device::PDLWaitPrimary<kUsePDL>();
  if (length <= topk) {
    for (uint32_t i = threadIdx.x; i < topk; i += kBlockSize) {
      out[i] = i < length ? static_cast<int32_t>(i) + offset : -1;
    }
    return;
  }

  const auto prefix_length = length - params.force_last;
  const auto prefix_topk = topk - params.force_last;
  if (prefix_topk != 0) {
    using Selection = OrderedTopK<Score>;
    constexpr uint32_t kVecSize = Selection::kVecSize;
    __shared__ typename Selection::Smem smem;
    const auto* scores = params.scores + row * params.score_stride;
    if constexpr (kResidentItems != 0) {
      Selection::template forward<kResidentItems>(scores, prefix_length, prefix_topk, offset, out, smem);
    } else {
      const auto misalignment = (reinterpret_cast<uintptr_t>(scores) & 15) / sizeof(Score);
      const auto groups = (prefix_length + misalignment + kVecSize - 1) / kVecSize;
      if (groups <= kBlockSize) {
        Selection::template forward<1>(scores, prefix_length, prefix_topk, offset, out, smem);
      } else if (groups <= 2 * kBlockSize) {
        Selection::template forward<2>(scores, prefix_length, prefix_topk, offset, out, smem);
      } else if (groups <= 4 * kBlockSize) {
        Selection::template forward<4>(scores, prefix_length, prefix_topk, offset, out, smem);
      } else {
        Selection::template forward<0>(scores, prefix_length, prefix_topk, offset, out, smem);
      }
    }
  }
  for (uint32_t i = threadIdx.x; i < params.force_last; i += kBlockSize) {
    out[prefix_topk + i] = static_cast<int32_t>(prefix_length + i) + offset;
  }
}

}  // namespace

template <bool kUsePDL>
struct ExactTopKKernel {
  static void run(
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView out_indices,
      tvm::ffi::Optional<tvm::ffi::TensorView> offsets,
      int64_t force_last) {
    if (scores.dtype().code == kDLBfloat) {
      run_typed<uint16_t, bf16_t>(scores, lengths, out_indices, offsets, force_last);
    } else {
      run_typed<float, float>(scores, lengths, out_indices, offsets, force_last);
    }
  }

 private:
  template <typename Score, typename DType>
  static void run_typed(
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView out_indices,
      tvm::ffi::Optional<tvm::ffi::TensorView> offsets,
      int64_t force_last) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"max_seq_len"};
    auto stride = SymbolicSize{"score_stride"};
    auto k = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({rows, width}).with_strides({stride, 1}).with_dtype<DType>().with_device(device).verify(scores);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows, k}).with_dtype<int32_t>().with_device(device).verify(out_indices);
    const int32_t* offsets_ptr = nullptr;
    if (offsets.has_value()) {
      TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(offsets.value());
      offsets_ptr = static_cast<const int32_t*>(offsets.value().data_ptr());
    }

    const auto topk = k.unwrap();
    RuntimeCheck(topk > 0 && topk <= INT_MAX, "k must be a positive int32");
    RuntimeCheck(force_last >= 0 && force_last <= topk, "force_last must be in [0, k]");
    RuntimeCheck(topk - force_last <= kMaxScoreTopK, "k - force_last must be at most 8192");
    RuntimeCheck(width.unwrap() <= INT_MAX - 16, "score width exceeds the exact top-k index range");
    if (rows.unwrap() == 0) return;

    const ExactTopKParams<Score> params{
        .scores = static_cast<const Score*>(scores.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .offsets = offsets_ptr,
        .out_indices = static_cast<int32_t*>(out_indices.data_ptr()),
        .score_stride = stride.unwrap(),
        .topk = static_cast<uint32_t>(topk),
        .force_last = static_cast<uint32_t>(force_last),
    };
    auto launch = LaunchKernel(static_cast<uint32_t>(rows.unwrap()), kBlockSize, device.unwrap());
    launch.config({.use_pdl = kUsePDL});
    constexpr uint32_t kVecSize = OrderedTopK<Score>::kVecSize;
    // Reserve one partial vector for an unaligned row. Keeping short launches
    // separate from the streaming kernel reduces the live register ranges.
    const auto max_prefix = std::max<int64_t>(0, width.unwrap() - force_last) + kVecSize - 1;
    if (max_prefix <= kBlockSize * kVecSize) {
      launch.launch(exact_topk_kernel<kUsePDL, Score, 1>, params);
    } else if (max_prefix <= 2 * kBlockSize * kVecSize) {
      launch.launch(exact_topk_kernel<kUsePDL, Score, 2>, params);
    } else if (max_prefix <= 4 * kBlockSize * kVecSize) {
      launch.launch(exact_topk_kernel<kUsePDL, Score, 4>, params);
    } else {
      launch.launch(exact_topk_kernel<kUsePDL, Score>, params);
    }
  }
};

}  // namespace sglang::dsa

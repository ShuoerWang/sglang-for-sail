# openai-protocol 1.0.0 local compatibility patch

Source: the published openai-protocol 1.0.0 crate, repository https://github.com/lightseekorg/smg, Apache-2.0. Original source copied from the existing Cargo registry used to build this gateway.

Only functional delta: add a serde-flattened `other` map to ChatCompletionRequest, matching CompletionRequest. This preserves cache_salt and other engine extensions when a validated request is serialized to P and D. No protocol version/API upgrade or unrelated DP field changes. Regression coverage is in the gateway PD router tests.

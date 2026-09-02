function(ninfer_internal_includes target)
  # The gfx906 compat include tree must come first so that #include
  # <cuda_runtime.h> and friends resolve to the HIP-mapping shims.
  target_include_directories(${target} BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/src/compat/gfx906/include)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

# Compiles a target's .cu sources as HIP (the llama.cpp-gfx906 pattern: CUDA
# sources build directly under hip-clang). Call after the target's source list
# is final. gfx906 has no separable-compilation requirement; the archives are
# plain static libraries.
function(ninfer_cuda_archive target)
  get_target_property(_ninfer_hip_sources ${target} SOURCES)
  list(FILTER _ninfer_hip_sources INCLUDE REGEX "\\.cu$")
  if(_ninfer_hip_sources)
    set_source_files_properties(${_ninfer_hip_sources} PROPERTIES LANGUAGE HIP)
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--gpu-max-threads-per-block=1024>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(ninfer_core STATIC
  core/arena.cu
  core/device.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/paged_kv_cache.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp
  runtime/engine/request_memory.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC hip::host Threads::Threads)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# The NVFP4 TMA archive (warp-specialized SM120 kernels) is deleted on gfx906:
# the hardware has no TMA, no NVFP4, and no tensor cores. Its launchers are
# stubbed in ops/gfx906_stubs.cpp.

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/bidirectional_gqa_attention.cu
  ops/launcher/swa.cu
  ops/launcher/gqa_attention_decode.cu
  ops/launcher/gqa_attention_prefill.cu
  ops/launcher/kv_cache_append_prefix.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/vision_attention.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_tiled_gfx906.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_small_t.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_tiled_gfx906.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/linear/bf16/bf16_gemv.cu
  ops/linear/bf16/bf16_gemm_mma.cu
  ops/linear/bf16/bf16_small_t.cu
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_gemv.cu
  ops/linear/fp8/fp8_small_t.cu
  ops/linear/fp8/fp8_a16_mma.cu
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_gemv.cu
  ops/linear/nvfp4/nvfp4_small_t.cu
  ops/linear/nvfp4/nvfp4_w4a4.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  ops/linear/q4/q4_small_t_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/gfx906/q4_tiled_gfx906.cu
  ops/linear/gfx906/q5_tiled_gfx906.cu
  ops/linear/gfx906/w8_tiled_gfx906.cu
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_tiled_gfx906.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ops/linear/w8/w8_small_t.cu
  ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ops/linear/w8/w8_dispatch.cpp
  ops/linear_pair/w8/w8_pair_decode.cu
  ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_gemm_splitk.cu
  ops/linear_pair/w8/w8_pair_plan.cpp
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/common/allreduce.cu
  ops/linear/linear.cpp
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/bidirectional_gqa_attention.cpp
  ops/wrapper/swa.cpp
  ops/wrapper/gqa_attention.cpp
  ops/wrapper/kv_cache_append_prefix.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/vision_attention.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp)

# gfx906 has no FP8/NVFP4 units and the 27B target is dense, so the fp8/,
# nvfp4/, and sparse_moe/ op families are deleted from the build (pattern:
# ninfer-3090's w4a4 source filtering). Their entry points become throwing
# stubs so the retained dispatch layers still link.
get_target_property(ninfer_ops_sources ninfer_ops SOURCES)
list(FILTER ninfer_ops_sources EXCLUDE REGEX "(fp8|nvfp4|sparse_moe)/")
set_property(TARGET ninfer_ops PROPERTY SOURCES ${ninfer_ops_sources})
target_sources(ninfer_ops PRIVATE ops/gfx906_stubs.cpp)

ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)

if(NINFER_ENABLE_MEDIA)
  add_library(ninfer_media_decode STATIC
    media/decode/decode.cpp)
  ninfer_internal_includes(ninfer_media_decode)
  target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)
else()
  # FFmpeg-free build: media decode entry points throw at runtime.
  add_library(ninfer_media_decode STATIC
    media/decode/decode_stub.cpp)
  ninfer_internal_includes(ninfer_media_decode)
endif()

if(NINFER_BUILD_MEDIA_ACQUIRE)
  if(NINFER_ENABLE_MEDIA)
    # Product-only path/data/HTTP acquisition. No target package links this library.
    add_library(ninfer_media_acquire STATIC
      product/media_acquire/acquire.cpp)
    ninfer_internal_includes(ninfer_media_acquire)
    target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
  else()
    # libcurl-free build: media acquisition entry points throw at runtime.
    add_library(ninfer_media_acquire STATIC
      product/media_acquire/acquire_stub.cpp)
    ninfer_internal_includes(ninfer_media_acquire)
  endif()
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(ninfer_product_load_progress STATIC
    product/load_progress/load_progress.cpp)
  ninfer_internal_includes(ninfer_product_load_progress)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/contract/sampling.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)
add_subdirectory(targets/qwen3_6_27b)
add_subdirectory(targets/qwen3_6_35b_a3b)

target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_artifact
    ninfer_core
    ninfer_ops
    ninfer_text
    ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/openai_schema.cpp
    serve/responses_schema.cpp
    serve/anthropic_schema.cpp
    serve/response_store.cpp
    serve/tool_call_parser.cpp
    serve/translate.cpp
    serve/console_log.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/responses_http.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire hip::host)
endif()

#pragma once

#include "common.cuh"

// LittleBit linear layer (llama-graph.cpp build_lb_mm), up to 9 nodes:
//   y = mul_mat(U, pad(mul_mat(V, x * g))) + mul_mat(U_R, pad(mul_mat(V_R, x * g_R)))
// For few tokens each node is a tiny kernel, so run the chain as 2 kernels instead.
// Activations stay F32 (no Q8_1), so the result is at least as exact as the unfused chain.

// larger batches use the unfused path (MMQ is fast there)
#define GGML_CUDA_LB_MAX_TOKENS 8
// consecutive projections with the same input and shape (e.g. ffn_gate + ffn_up) share the 2 kernels
#define GGML_CUDA_LB_MAX_PROJ 2

struct ggml_cuda_lb_proj {
    const ggml_tensor * v[2] = { nullptr, nullptr };
    const ggml_tensor * u[2] = { nullptr, nullptr };
    const ggml_tensor * g[2] = { nullptr, nullptr };
    ggml_tensor * dst = nullptr; // ADD (residual) or the last MUL_MAT
};

struct ggml_cuda_lb_match {
    int n_nodes = 0;
    int n_proj  = 0;
    int n_paths = 0;
    const ggml_tensor * x = nullptr;
    ggml_cuda_lb_proj proj[GGML_CUDA_LB_MAX_PROJ];
    int outputs[8]; // node indices used outside the fused range
    int n_outputs = 0;
};

// Whether to use the fused kernels on a device of compute capability cc. Default: AMD only.
// On an NVIDIA RTX A1000 (Ampere) the fused path measured about 2x slower than the unfused chain
// for generation (GeniusDevam/littlebit-gguf#1), so other vendors keep the unfused chain until
// the kernels are tuned there. GGML_CUDA_LB_FUSION=1 forces it on, =0 off, on any device.
// GGML_CUDA_DISABLE_FUSION still turns off all fusion, this one included.
bool ggml_cuda_lb_fusion_enabled(int cc);

bool ggml_cuda_lb_match_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_lb_match & m);

void ggml_cuda_lb_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_lb_match & m);

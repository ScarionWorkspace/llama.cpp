#pragma once

#include "common.cuh"

// LittleBit linear layer (llama-graph.cpp build_lb_mm), up to 9 nodes:
//   y = mul_mat(U, pad(mul_mat(V, x * g))) + mul_mat(U_R, pad(mul_mat(V_R, x * g_R)))
// For few tokens each node is a tiny kernel, so run the chain as 2 kernels instead.
// Activations stay F32 (no Q8_1), so the result is at least as exact as the unfused chain.

struct ggml_cuda_lb_match {
    int n_nodes = 0;
    const ggml_tensor * x = nullptr;
    const ggml_tensor * v[2] = { nullptr, nullptr };
    const ggml_tensor * u[2] = { nullptr, nullptr };
    const ggml_tensor * g[2] = { nullptr, nullptr };
    ggml_tensor * dst = nullptr; // ADD (residual) or the last MUL_MAT
};

// larger batches use the unfused path (MMQ is fast there)
#define GGML_CUDA_LB_MAX_TOKENS 8

bool ggml_cuda_lb_match_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_lb_match & m);

void ggml_cuda_lb_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_lb_match & m);

#include "littlebit.cuh"

#include "ggml-impl.h"

// Each lane handles 4 consecutive elements. Stage A splits the long reduction over LB_A_WARPS warps
// per block (LB_A_ROWS rows per block share the activation loads), stage B uses one warp per row.
#define LB_A_ROWS  4
#define LB_A_WARPS 8
#define LB_B_WARPS 4

// signs of row elements j .. j+3 (j % 4 == 0), bit 1 -> +1, bit 0 -> -1; d is the block scale
static __device__ __forceinline__ float4 lb_q1_0_signs4(const block_q1_0 * row, const int j, float & d) {
    const block_q1_0 & b = row[j / QK1_0];
    d = __half2float(b.d);
    const int k = j % QK1_0;
    const int q = b.qs[k / 8] >> (k % 8);
    return make_float4(q & 1 ? 1.0f : -1.0f, q & 2 ? 1.0f : -1.0f, q & 4 ? 1.0f : -1.0f, q & 8 ? 1.0f : -1.0f);
}

static __device__ __forceinline__ float lb_dot4(const float4 a, const float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// t[p][c][i] = sum_j V_p[i, j] * g_p[j] * x[c][j], rows r .. r_pad-1 are zero (replaces the PAD node)
template <int ncols>
static __global__ void lb_stage_a(
        const block_q1_0 * __restrict__ v0, const block_q1_0 * __restrict__ v1,
        const float * __restrict__ g0, const float * __restrict__ g1,
        const float * __restrict__ x, float * __restrict__ t,
        const int n_in, const int r, const int r_pad, const int64_t x_stride, const int64_t v_row_blocks) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float partial[LB_A_WARPS][LB_A_ROWS][ncols];

    const int row0 = blockIdx.x * LB_A_ROWS; // r_pad % LB_A_ROWS == 0: all rows of a block are in one path
    const int path = row0 / r_pad;
    const int i0   = row0 % r_pad;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const block_q1_0 * vp = path == 0 ? v0 : v1;
    const float * g = path == 0 ? g0 : g1;

    float acc[LB_A_ROWS][ncols] = { { 0.0f } };
    for (int j = 4*(warp*warp_size + lane); j < n_in; j += 4*warp_size*LB_A_WARPS) {
        const float4 g4 = *(const float4 *) (g + j);
        float4 xg[ncols];
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float4 x4 = *(const float4 *) (x + c*x_stride + j);
            xg[c] = make_float4(x4.x*g4.x, x4.y*g4.y, x4.z*g4.z, x4.w*g4.w);
        }
#pragma unroll
        for (int rr = 0; rr < LB_A_ROWS; ++rr) {
            if (i0 + rr >= r) {
                break;
            }
            float d;
            const float4 s4 = lb_q1_0_signs4(vp + (i0 + rr)*v_row_blocks, j, d);
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                acc[rr][c] += d * lb_dot4(s4, xg[c]);
            }
        }
    }

#pragma unroll
    for (int rr = 0; rr < LB_A_ROWS; ++rr) {
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float sum = warp_reduce_sum<warp_size>(acc[rr][c]);
            if (lane == 0) {
                partial[warp][rr][c] = sum;
            }
        }
    }
    __syncthreads();

    const int k = warp*warp_size + lane;
    if (k < LB_A_ROWS*ncols) {
        const int rr = k / ncols;
        const int c  = k % ncols;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < LB_A_WARPS; ++w) {
            sum += partial[w][rr][c];
        }
        t[(int64_t) path*ncols*r_pad + c*r_pad + i0 + rr] = i0 + rr < r ? sum : 0.0f;
    }
}

// y[c][o] = sum_p sum_k U_p[o, k] * t[p][c][k] (the ADD of the residual path is done here)
template <int ncols>
static __global__ void lb_stage_b(
        const block_q1_0 * __restrict__ u0, const block_q1_0 * __restrict__ u1,
        const float * __restrict__ t, float * __restrict__ y,
        const int n_out, const int r_pad, const int64_t y_stride, const int64_t u_row_blocks) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int o    = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (o >= n_out) {
        return;
    }

    float acc[ncols] = { 0.0f };
    const int n_paths = u1 ? 2 : 1;
    for (int p = 0; p < n_paths; ++p) {
        const block_q1_0 * urow = (p == 0 ? u0 : u1) + o*u_row_blocks;
        const float * tp = t + (int64_t) p * ncols * r_pad;
        for (int k = 4*lane; k < r_pad; k += 4*warp_size) {
            float d;
            const float4 s4 = lb_q1_0_signs4(urow, k, d);
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                acc[c] += d * lb_dot4(s4, *(const float4 *) (tp + c*r_pad + k));
            }
        }
    }

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float sum = warp_reduce_sum<warp_size>(acc[c]);
        if (lane == 0) {
            y[c*y_stride + o] = sum;
        }
    }
}

// MUL(x, g) -> MUL_MAT(v, .) [-> PAD] -> MUL_MAT(u, .), as built by build_lb_mm in llama-graph.cpp
static bool lb_match_path(const ggml_cgraph * cgraph, int idx, int & n, const ggml_tensor *& x,
                          const ggml_tensor *& v, const ggml_tensor *& u, const ggml_tensor *& g) {
    if (idx + 3 > cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[idx];
    const ggml_tensor * mmv = cgraph->nodes[idx + 1];
    if (mul->op != GGML_OP_MUL || mmv->op != GGML_OP_MUL_MAT || mmv->src[1] != mul) {
        return false;
    }
    int k = idx + 2;
    const ggml_tensor * pad = nullptr;
    if (cgraph->nodes[k]->op == GGML_OP_PAD) {
        pad = cgraph->nodes[k];
        if (pad->src[0] != mmv) {
            return false;
        }
        const int32_t * pp = (const int32_t *) pad->op_params;
        if (pp[0] != 0 || pp[2] != 0 || pp[3] != 0 || pp[4] != 0 || pp[5] != 0 || pp[6] != 0 || pp[7] != 0 || pp[8] != 0) {
            return false;
        }
        k++;
        if (k >= cgraph->n_nodes) {
            return false;
        }
    }
    const ggml_tensor * mmu = cgraph->nodes[k];
    if (mmu->op != GGML_OP_MUL_MAT || mmu->src[1] != (pad ? pad : mmv)) {
        return false;
    }

    x = mul->src[0];
    g = mul->src[1];
    v = mmv->src[0];
    u = mmu->src[0];

    if (v->type != GGML_TYPE_Q1_0 || u->type != GGML_TYPE_Q1_0 || x->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(v) || !ggml_is_contiguous(u) || !ggml_is_contiguous(g)) {
        return false;
    }
    if (ggml_nrows(g) != 1 || g->ne[0] != x->ne[0] || x->ne[2] != 1 || x->ne[3] != 1 || x->ne[1] > GGML_CUDA_LB_MAX_TOKENS) {
        return false;
    }
    if (v->ne[0] != x->ne[0] || u->ne[0] < v->ne[1] || u->ne[0] % QK1_0 != 0 || v->ne[2] != 1 || u->ne[2] != 1) {
        return false;
    }
    if (!pad && u->ne[0] != v->ne[1]) {
        return false;
    }
    if (mmu->type != GGML_TYPE_F32 || mmv->type != GGML_TYPE_F32) {
        return false;
    }

    n = k - idx + 1;
    return true;
}

bool ggml_cuda_lb_match_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_lb_match & m) {
    int n0 = 0;
    if (!lb_match_path(cgraph, node_idx, n0, m.x, m.v[0], m.u[0], m.g[0])) {
        return false;
    }
    const ggml_tensor * mmu0 = cgraph->nodes[node_idx + n0 - 1];

    int n1 = 0;
    const ggml_tensor * x1 = nullptr;
    int count = n0;
    m.dst = (ggml_tensor *) mmu0;
    if (lb_match_path(cgraph, node_idx + n0, n1, x1, m.v[1], m.u[1], m.g[1]) && x1 == m.x &&
            node_idx + n0 + n1 < cgraph->n_nodes) {
        const ggml_tensor * mmu1 = cgraph->nodes[node_idx + n0 + n1 - 1];
        ggml_tensor * add = cgraph->nodes[node_idx + n0 + n1];
        const bool same_shape = m.v[1]->ne[1] == m.v[0]->ne[1] && m.u[1]->ne[0] == m.u[0]->ne[0] &&
                                m.u[1]->ne[1] == m.u[0]->ne[1];
        if (add->op == GGML_OP_ADD && same_shape &&
                ((add->src[0] == mmu0 && add->src[1] == mmu1) || (add->src[0] == mmu1 && add->src[1] == mmu0)) &&
                add->type == GGML_TYPE_F32 && ggml_is_contiguous(add)) {
            count = n0 + n1 + 1;
            m.dst = add;
        } else {
            m.v[1] = m.u[1] = m.g[1] = nullptr;
        }
    } else {
        m.v[1] = m.u[1] = m.g[1] = nullptr;
    }
    if (!ggml_is_contiguous(m.dst)) {
        return false;
    }

    enum ggml_op ops[16];
    for (int i = 0; i < count; ++i) {
        ops[i] = cgraph->nodes[node_idx + i]->op;
    }
    const int out = node_idx + count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, count, ops, &out, 1)) {
        return false;
    }
    m.n_nodes = count;
    return true;
}

template <int ncols>
static void lb_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_lb_match & m, cudaStream_t stream) {
    const int n_in  = m.x->ne[0];
    const int r     = m.v[0]->ne[1];
    const int r_pad = m.u[0]->ne[0];
    const int n_out = m.u[0]->ne[1];
    const int n_paths = m.v[1] ? 2 : 1;

    ggml_cuda_pool_alloc<float> t(ctx.pool(), (size_t) n_paths * ncols * r_pad);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    GGML_ASSERT(r_pad % LB_A_ROWS == 0 && LB_A_ROWS*ncols <= LB_A_WARPS*warp_size);

    lb_stage_a<ncols><<<n_paths * r_pad / LB_A_ROWS, dim3(warp_size, LB_A_WARPS, 1), 0, stream>>>(
        (const block_q1_0 *) m.v[0]->data, m.v[1] ? (const block_q1_0 *) m.v[1]->data : nullptr,
        (const float *) m.g[0]->data, m.g[1] ? (const float *) m.g[1]->data : nullptr,
        (const float *) m.x->data, t.get(), n_in, r, r_pad,
        m.x->nb[1] / sizeof(float), m.v[0]->nb[1] / sizeof(block_q1_0));

    lb_stage_b<ncols><<<(n_out + LB_B_WARPS - 1) / LB_B_WARPS, dim3(warp_size, LB_B_WARPS, 1), 0, stream>>>(
        (const block_q1_0 *) m.u[0]->data, m.u[1] ? (const block_q1_0 *) m.u[1]->data : nullptr,
        t.get(), (float *) m.dst->data, n_out, r_pad,
        m.dst->nb[1] / sizeof(float), m.u[0]->nb[1] / sizeof(block_q1_0));
}

void ggml_cuda_lb_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_lb_match & m) {
    cudaStream_t stream = ctx.stream();
    switch (m.x->ne[1]) {
        case 1: lb_launch<1>(ctx, m, stream); break;
        case 2: lb_launch<2>(ctx, m, stream); break;
        case 3: lb_launch<3>(ctx, m, stream); break;
        case 4: lb_launch<4>(ctx, m, stream); break;
        case 5: lb_launch<5>(ctx, m, stream); break;
        case 6: lb_launch<6>(ctx, m, stream); break;
        case 7: lb_launch<7>(ctx, m, stream); break;
        case 8: lb_launch<8>(ctx, m, stream); break;
        default: GGML_ABORT("unsupported number of tokens for LittleBit fusion: %lld", (long long) m.x->ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

#include "littlebit.cuh"

#include "ggml-impl.h"

// Each lane handles LB_CHUNK = 32 consecutive elements (a quarter Q1_0 block), so a warp covers 32 chunks per step
// and more loads are in flight. Stage A splits the reduction over LB_A_WARPS warps per block (LB_A_ROWS rows per
// block share the activation loads), stage B uses LB_B_ROWS rows per warp that share the latent loads.
#define LB_CHUNK   32
#define LB_A_ROWS  4
#define LB_A_WARPS 4
#define LB_B_ROWS  2
#define LB_B_WARPS 4

// sign bits of row elements j .. j+31 (j % 32 == 0), bit 1 -> +1; d is the block scale
static __device__ __forceinline__ uint32_t lb_q1_0_bits32(const block_q1_0 * row, const int j, float & d) {
    const block_q1_0 & b = row[j / QK1_0];
    d = __half2float(b.d);
    // qs is only 2-byte aligned inside the 18-byte block
    const uint16_t * q16 = (const uint16_t *) (b.qs + (j % QK1_0) / 8);
    return (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);
}

// sum over 4 elements of (bit ? a : -a), bits in the low 4 bits of q
static __device__ __forceinline__ float lb_signed_sum4(const uint32_t q, const float4 a) {
    return (q & 1 ? a.x : -a.x) + (q & 2 ? a.y : -a.y) + (q & 4 ? a.z : -a.z) + (q & 8 ? a.w : -a.w);
}

// path q = proj*n_paths + p
struct lb_ptrs {
    const block_q1_0 * v[2*GGML_CUDA_LB_MAX_PROJ];
    const block_q1_0 * u[2*GGML_CUDA_LB_MAX_PROJ];
    const float      * g[2*GGML_CUDA_LB_MAX_PROJ];
    float            * dst[GGML_CUDA_LB_MAX_PROJ];
};

// t[q][c][i] = sum_j V_q[i, j] * g_q[j] * x[c][j], rows r .. r_pad-1 are zero (replaces the PAD node)
template <int ncols>
static __global__ void lb_stage_a(const lb_ptrs ptrs, const float * __restrict__ x, float * __restrict__ t,
        const int n_in, const int r, const int r_pad, const int64_t x_stride, const int64_t v_row_blocks) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float partial[LB_A_WARPS][LB_A_ROWS][ncols];

    const int row0 = blockIdx.x * LB_A_ROWS; // r_pad % LB_A_ROWS == 0: all rows of a block are in one path
    const int path = row0 / r_pad;
    const int i0   = row0 % r_pad;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const block_q1_0 * vp = ptrs.v[path];
    const float * g = ptrs.g[path];

    float acc[LB_A_ROWS][ncols] = { { 0.0f } };
    for (int j = LB_CHUNK*(warp*warp_size + lane); j < n_in; j += LB_CHUNK*warp_size*LB_A_WARPS) {
        uint32_t bits[LB_A_ROWS];
        float d[LB_A_ROWS];
#pragma unroll
        for (int rr = 0; rr < LB_A_ROWS; ++rr) {
            bits[rr] = i0 + rr < r ? lb_q1_0_bits32(vp + (i0 + rr)*v_row_blocks, j, d[rr]) : 0;
            d[rr]    = i0 + rr < r ? d[rr] : 0.0f;
        }
        float part[LB_A_ROWS][ncols] = { { 0.0f } };
#pragma unroll
        for (int e = 0; e < LB_CHUNK; e += 4) {
            const float4 g4 = *(const float4 *) (g + j + e);
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                const float4 x4 = *(const float4 *) (x + c*x_stride + j + e);
                const float4 xg = make_float4(x4.x*g4.x, x4.y*g4.y, x4.z*g4.z, x4.w*g4.w);
#pragma unroll
                for (int rr = 0; rr < LB_A_ROWS; ++rr) {
                    part[rr][c] += lb_signed_sum4(bits[rr] >> e, xg);
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < LB_A_ROWS; ++rr) {
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                acc[rr][c] += d[rr] * part[rr][c];
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
// n_out % LB_B_ROWS == 0, so all rows of a warp belong to one projection
template <int ncols>
static __global__ void lb_stage_b(const lb_ptrs ptrs, const float * __restrict__ t,
        const int n_proj, const int n_paths, const int n_out, const int r_pad, const int64_t y_stride, const int64_t u_row_blocks) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * LB_B_ROWS;
    const int lane = threadIdx.x;
    if (row0 >= n_proj*n_out) {
        return;
    }
    const int proj = row0 / n_out;
    const int o0   = row0 % n_out;

    float acc[LB_B_ROWS][ncols] = { { 0.0f } };
    for (int p = 0; p < n_paths; ++p) {
        const int q = proj*n_paths + p;
        const block_q1_0 * up = ptrs.u[q] + o0*u_row_blocks;
        const float * tp = t + (int64_t) q * ncols * r_pad;
        for (int k = LB_CHUNK*lane; k < r_pad; k += LB_CHUNK*warp_size) {
            uint32_t bits[LB_B_ROWS];
            float d[LB_B_ROWS];
#pragma unroll
            for (int rr = 0; rr < LB_B_ROWS; ++rr) {
                bits[rr] = lb_q1_0_bits32(up + rr*u_row_blocks, k, d[rr]);
            }
            float part[LB_B_ROWS][ncols] = { { 0.0f } };
#pragma unroll
            for (int e = 0; e < LB_CHUNK; e += 4) {
#pragma unroll
                for (int c = 0; c < ncols; ++c) {
                    const float4 t4 = *(const float4 *) (tp + c*r_pad + k + e);
#pragma unroll
                    for (int rr = 0; rr < LB_B_ROWS; ++rr) {
                        part[rr][c] += lb_signed_sum4(bits[rr] >> e, t4);
                    }
                }
            }
#pragma unroll
            for (int rr = 0; rr < LB_B_ROWS; ++rr) {
#pragma unroll
                for (int c = 0; c < ncols; ++c) {
                    acc[rr][c] += d[rr] * part[rr][c];
                }
            }
        }
    }

#pragma unroll
    for (int rr = 0; rr < LB_B_ROWS; ++rr) {
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float sum = warp_reduce_sum<warp_size>(acc[rr][c]);
            if (lane == 0) {
                ptrs.dst[proj][c*y_stride + o0 + rr] = sum;
            }
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
    if (v->ne[0] != x->ne[0] || u->ne[0] < v->ne[1] || u->ne[0] % QK1_0 != 0 || v->ne[2] != 1 || u->ne[2] != 1 ||
            u->ne[1] % LB_B_ROWS != 0) {
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

// one projection starting at node idx: main path, optional residual path + ADD
static bool lb_match_proj(const ggml_cgraph * cgraph, int idx, int & count, const ggml_tensor *& x, ggml_cuda_lb_proj & pr) {
    int n0 = 0;
    if (!lb_match_path(cgraph, idx, n0, x, pr.v[0], pr.u[0], pr.g[0])) {
        return false;
    }
    const ggml_tensor * mmu0 = cgraph->nodes[idx + n0 - 1];
    count  = n0;
    pr.dst = (ggml_tensor *) mmu0;
    pr.v[1] = pr.u[1] = pr.g[1] = nullptr;

    int n1 = 0;
    const ggml_tensor * x1 = nullptr;
    ggml_cuda_lb_proj res;
    if (lb_match_path(cgraph, idx + n0, n1, x1, res.v[0], res.u[0], res.g[0]) && x1 == x && idx + n0 + n1 < cgraph->n_nodes) {
        const ggml_tensor * mmu1 = cgraph->nodes[idx + n0 + n1 - 1];
        ggml_tensor * add = cgraph->nodes[idx + n0 + n1];
        const bool same_shape = res.v[0]->ne[1] == pr.v[0]->ne[1] && res.u[0]->ne[0] == pr.u[0]->ne[0] &&
                                res.u[0]->ne[1] == pr.u[0]->ne[1];
        if (add->op == GGML_OP_ADD && same_shape &&
                ((add->src[0] == mmu0 && add->src[1] == mmu1) || (add->src[0] == mmu1 && add->src[1] == mmu0)) &&
                add->type == GGML_TYPE_F32) {
            pr.v[1] = res.v[0];
            pr.u[1] = res.u[0];
            pr.g[1] = res.g[0];
            pr.dst  = add;
            count   = n0 + n1 + 1;
        }
    }
    return ggml_is_contiguous(pr.dst);
}

static bool lb_same_shape(const ggml_cuda_lb_proj & a, const ggml_cuda_lb_proj & b) {
    return (a.v[1] != nullptr) == (b.v[1] != nullptr) && a.v[0]->ne[1] == b.v[0]->ne[1] &&
           a.u[0]->ne[0] == b.u[0]->ne[0] && a.u[0]->ne[1] == b.u[0]->ne[1];
}

bool ggml_cuda_lb_match_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_lb_match & m) {
    int count = 0;
    if (!lb_match_proj(cgraph, node_idx, count, m.x, m.proj[0])) {
        return false;
    }
    m.n_proj  = 1;
    m.n_paths = m.proj[0].v[1] ? 2 : 1;
    m.n_outputs = 0;
    m.outputs[m.n_outputs++] = node_idx + count - 1;

    // append following projections with the same input, skipping view nodes (e.g. a RESHAPE of the previous result)
    while (m.n_proj < GGML_CUDA_LB_MAX_PROJ) {
        int j = node_idx + count;
        int views[2];
        int n_views = 0;
        while (j < cgraph->n_nodes && n_views < 2 && (cgraph->nodes[j]->op == GGML_OP_RESHAPE || cgraph->nodes[j]->op == GGML_OP_VIEW)) {
            views[n_views++] = j++;
        }
        int n = 0;
        const ggml_tensor * x = nullptr;
        ggml_cuda_lb_proj pr;
        if (j >= cgraph->n_nodes || !lb_match_proj(cgraph, j, n, x, pr) || x != m.x || !lb_same_shape(pr, m.proj[0])) {
            break;
        }
        for (int k = 0; k < n_views; ++k) {
            m.outputs[m.n_outputs++] = views[k];
        }
        m.proj[m.n_proj++] = pr;
        count = j + n - node_idx;
        m.outputs[m.n_outputs++] = node_idx + count - 1;
    }

    enum ggml_op ops[32];
    if (count > 31) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        ops[i] = cgraph->nodes[node_idx + i]->op;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, count, ops, m.outputs, m.n_outputs)) {
        return false;
    }
    m.n_nodes = count;
    return true;
}

template <int ncols>
static void lb_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_lb_match & m, cudaStream_t stream) {
    const ggml_cuda_lb_proj & p0 = m.proj[0];
    const int n_in  = m.x->ne[0];
    const int r     = p0.v[0]->ne[1];
    const int r_pad = p0.u[0]->ne[0];
    const int n_out = p0.u[0]->ne[1];
    const int n_q   = m.n_proj * m.n_paths;

    lb_ptrs ptrs = {};
    for (int k = 0; k < m.n_proj; ++k) {
        for (int p = 0; p < m.n_paths; ++p) {
            ptrs.v[k*m.n_paths + p] = (const block_q1_0 *) m.proj[k].v[p]->data;
            ptrs.u[k*m.n_paths + p] = (const block_q1_0 *) m.proj[k].u[p]->data;
            ptrs.g[k*m.n_paths + p] = (const float *) m.proj[k].g[p]->data;
        }
        ptrs.dst[k] = (float *) m.proj[k].dst->data;
    }

    ggml_cuda_pool_alloc<float> t(ctx.pool(), (size_t) n_q * ncols * r_pad);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    GGML_ASSERT(r_pad % LB_A_ROWS == 0 && LB_A_ROWS*ncols <= LB_A_WARPS*warp_size && n_out % LB_B_ROWS == 0);

    lb_stage_a<ncols><<<n_q * r_pad / LB_A_ROWS, dim3(warp_size, LB_A_WARPS, 1), 0, stream>>>(
        ptrs, (const float *) m.x->data, t.get(), n_in, r, r_pad,
        m.x->nb[1] / sizeof(float), p0.v[0]->nb[1] / sizeof(block_q1_0));

    constexpr int rows_b = LB_B_WARPS * LB_B_ROWS;
    lb_stage_b<ncols><<<(m.n_proj*n_out + rows_b - 1) / rows_b, dim3(warp_size, LB_B_WARPS, 1), 0, stream>>>(
        ptrs, t.get(), m.n_proj, m.n_paths, n_out, r_pad,
        p0.dst->nb[1] / sizeof(float), p0.u[0]->nb[1] / sizeof(block_q1_0));
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

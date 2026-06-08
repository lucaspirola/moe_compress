// Grouped symmetric Gram accumulation for MoE input-covariance calibration.
//
// Computes, per expert e, the in-place accumulation
//     cov[e] += X_e^T @ X_e         (X_e = the rows of x_sorted routed to e)
//     counts[e] += (offsets[e+1] - offsets[e])
// where x_sorted is the expert-contiguous token matrix and offsets is its
// per-expert prefix-sum. This is the calibration "input_covariance" Gram
// Sigma_in = sum_t x_t x_t^T, kept as a RAW (unnormalized) fp32 sum.
//
// DIMENSION-AGNOSTIC: E (experts), d (= d_in for this matrix group), and R
// (total routed rows) are runtime kernel arguments, never compile-time
// constants -- the same kernel serves any MoE architecture and any matrix
// group (gate/up with d_in = hidden_size; down with d_in = intermediate_size).
//
// fp32 CUDA-core FMA accumulation (NOT tf32 tensor cores) to keep Sigma_in
// numerically exact for the downstream eigendecomposition/EoRA contract.
//
// Implementation: shared-memory-tiled X_e^T @ X_e with symmetry (strictly-lower
// output tiles early-return; upper tiles write both [i,j] and the mirror [j,i];
// diagonal tiles write [i,j] only). Each output element is written by exactly
// one thread, so the in-place `+=` needs no atomics. A further optimization
// (per-thread register micro-tiling / vectorized loads) could raise fp32
// throughput if the op is ever shown to dominate the forward; deferred until
// measured (current form is correct, graph-safe, and temp-buffer-free).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace calib {

// Shared-memory-tiled  cov[e] += X_e^T @ X_e  with symmetry.
//
// Output is tiled TILE x TILE. The grid is the full (nTiles, nTiles, E) square
// but strictly-lower output tiles (bj < bi) early-return, halving the compute;
// an upper tile (bj > bi) writes both C[i,j] and its mirror C[j,i] (the mirror
// element's own tile is a skipped lower tile, so there is no double-write and
// no atomics). Diagonal tiles (bj == bi) write only C[i,j], covering the whole
// block including its sub-diagonal.
//
// As[i_local][k_local] stages the X^T tile, Bs[k_local][j_local] the X tile;
// acc += sum_k As[i][k] * Bs[k][j] accumulates in fp32. Each output element is
// written by exactly one thread.
template <int TILE>
__global__ void gram_grouped_accum_kernel(
    float* __restrict__ cov,            // [E, d, d]
    int64_t* __restrict__ counts,       // [E]
    const float* __restrict__ x_sorted, // [R, d]
    const int64_t* __restrict__ offsets,// [E+1]
    int d) {
  const int e = blockIdx.z;
  const int bi = blockIdx.y;  // output row tile
  const int bj = blockIdx.x;  // output col tile
  if (bj < bi) return;        // symmetry: skip strictly-lower tiles

  const int64_t lo = offsets[e];
  const int64_t hi = offsets[e + 1];
  const int n_e = static_cast<int>(hi - lo);

  const int ty = threadIdx.y;  // i_local
  const int tx = threadIdx.x;  // j_local

  __shared__ float As[TILE][TILE];  // [i_local][k_local]
  __shared__ float Bs[TILE][TILE];  // [k_local][j_local]

  const int i = bi * TILE + ty;
  const int j = bj * TILE + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < n_e; k0 += TILE) {
    const int ii = bi * TILE + ty;
    const int ka = k0 + tx;
    As[ty][tx] = (ii < d && ka < n_e)
                     ? x_sorted[(lo + ka) * (int64_t)d + ii]
                     : 0.0f;
    const int jj = bj * TILE + tx;
    const int kb = k0 + ty;
    Bs[ty][tx] = (jj < d && kb < n_e)
                     ? x_sorted[(lo + kb) * (int64_t)d + jj]
                     : 0.0f;
    __syncthreads();
#pragma unroll
    for (int kl = 0; kl < TILE; ++kl) {
      acc += As[ty][kl] * Bs[kl][tx];
    }
    __syncthreads();
  }

  if (i < d && j < d) {
    cov[((int64_t)e * d + i) * d + j] += acc;
    if (bj > bi) {
      cov[((int64_t)e * d + j) * d + i] += acc;  // mirror (Gram is symmetric)
    }
  }

  if (bi == 0 && bj == 0 && ty == 0 && tx == 0) {
    counts[e] += (hi - lo);
  }
}

// In-place accumulate. Mutates `cov` and `counts`.
void gram_grouped_accum(
    at::Tensor cov,       // [E, d, d] float32 cuda  (mutated)
    at::Tensor counts,    // [E]       int64   cuda  (mutated)
    at::Tensor x_sorted,  // [R, d]    float32 cuda
    at::Tensor offsets) { // [E+1]     int64   cuda
  TORCH_CHECK(cov.is_cuda() && counts.is_cuda() && x_sorted.is_cuda() &&
                  offsets.is_cuda(),
              "gram_grouped_accum: all tensors must be CUDA");
  TORCH_CHECK(cov.scalar_type() == at::kFloat, "cov must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(x_sorted.scalar_type() == at::kFloat, "x_sorted must be float32");
  TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
  TORCH_CHECK(cov.dim() == 3 && cov.size(1) == cov.size(2),
              "cov must be [E, d, d]");
  TORCH_CHECK(cov.is_contiguous() && x_sorted.is_contiguous() &&
                  offsets.is_contiguous() && counts.is_contiguous(),
              "gram_grouped_accum: tensors must be contiguous");

  const int E = static_cast<int>(cov.size(0));
  const int d = static_cast<int>(cov.size(1));
  const int64_t R = x_sorted.size(0);
  TORCH_CHECK(x_sorted.size(1) == d, "x_sorted last dim must equal d");
  TORCH_CHECK(offsets.numel() == E + 1, "offsets must have E+1 elements");
  TORCH_CHECK(counts.numel() == E, "counts must have E elements");
  if (E == 0 || d == 0) return;
  (void)R;

  constexpr int TILE = 16;
  const int n_tiles = (d + TILE - 1) / TILE;
  const at::cuda::CUDAGuard guard(cov.device());
  dim3 block(TILE, TILE);
  dim3 grid(static_cast<unsigned>(n_tiles), static_cast<unsigned>(n_tiles),
            static_cast<unsigned>(E));
  auto stream = at::cuda::getCurrentCUDAStream();
  gram_grouped_accum_kernel<TILE><<<grid, block, 0, stream>>>(
      cov.data_ptr<float>(), counts.data_ptr<int64_t>(),
      x_sorted.data_ptr<float>(), offsets.data_ptr<int64_t>(), d);
}

// Meta (fake-tensor) impl: a no-op. The op mutates in place and returns
// nothing, so under fake-tensor tracing (torch.compile / export) there is no
// shape to infer -- the Meta kernel just must exist so tracing never launches
// the real CUDA kernel.
void gram_grouped_accum_meta(at::Tensor /*cov*/, at::Tensor /*counts*/,
                             at::Tensor /*x_sorted*/, at::Tensor /*offsets*/) {}

}  // namespace calib

// Op registration. Mutating-tensor schema (a!)/(b!) declares the in-place
// writes so Dynamo/Inductor keep the call alive (no DCE) and treat it as a
// side-effecting node -- the native equivalent of the Python ops'
// mutates_args.
TORCH_LIBRARY(calib_gram, m) {
  m.def(
      "gram_grouped_accum(Tensor(a!) cov, Tensor(b!) counts, "
      "Tensor x_sorted, Tensor offsets) -> ()");
}
TORCH_LIBRARY_IMPL(calib_gram, CUDA, m) {
  m.impl("gram_grouped_accum", &calib::gram_grouped_accum);
}
TORCH_LIBRARY_IMPL(calib_gram, Meta, m) {
  m.impl("gram_grouped_accum", &calib::gram_grouped_accum_meta);
}

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
// v1 is correctness-first: one thread per (e, i, j) output element, looping
// the expert's token segment with naive global loads. Each output element is
// written by exactly one thread, so the in-place `+=` needs no atomics. A
// shared-memory-tiled + symmetric (upper-triangle + mirror) variant is a
// later optimization that must preserve this numerical contract.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace calib {

__global__ void gram_grouped_accum_kernel(
    float* __restrict__ cov,            // [E, d, d]
    int64_t* __restrict__ counts,       // [E]
    const float* __restrict__ x_sorted, // [R, d]
    const int64_t* __restrict__ offsets,// [E+1]
    int E, int d, int64_t R) {
  const int e = blockIdx.z;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
  const int j = blockIdx.x * blockDim.x + threadIdx.x;  // output col
  if (e >= E || i >= d || j >= d) return;

  const int64_t lo = offsets[e];
  const int64_t hi = offsets[e + 1];

  float acc = 0.0f;
  for (int64_t k = lo; k < hi; ++k) {
    const float* row = x_sorted + k * (int64_t)d;
    acc += row[i] * row[j];
  }
  cov[((int64_t)e * d + i) * d + j] += acc;

  // One thread per expert advances the token count.
  if (i == 0 && j == 0) {
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

  const at::cuda::CUDAGuard guard(cov.device());
  dim3 block(16, 16);
  dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y,
            static_cast<unsigned>(E));
  auto stream = at::cuda::getCurrentCUDAStream();
  gram_grouped_accum_kernel<<<grid, block, 0, stream>>>(
      cov.data_ptr<float>(), counts.data_ptr<int64_t>(),
      x_sorted.data_ptr<float>(), offsets.data_ptr<int64_t>(), E, d, R);
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

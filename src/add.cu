#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

constexpr int CUDA_NUM_THREADS = 128;

constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GET_BLOCKS(const int N) {
  return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
           MAXIMUM_NUM_BLOCKS),
           // Use at least 1 block, since CUDA does not allow empty block
           1);
}

// define the kernel function:
template <typename T>
__global__ void sum(T *a, T *b, T *c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    c[i] = a[i] + b[i];
  }
}

// define the kernel calling code:
template <typename T>
void AddGPUKernel(T *in_a, T *in_b, T *out_c, int N, cudaStream_t stream) {
  sum<T>
      <<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(in_a, in_b, out_c, N);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instansiate the kernel template for T=float:
template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c, int N, cudaStream_t stream);
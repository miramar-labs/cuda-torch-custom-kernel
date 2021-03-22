#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// declare templates for front (cpp) and back (cuda) sides of function:
template <typename T>
void AddGPUKernel(T *in_a, T *in_b, T *out_c, int N, cudaStream_t stream);

template <typename T>
void AddGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
  int N = in_a.numel();
  if (N != in_b.numel())
    throw std::invalid_argument("Size mismatch A.numel(): " + std::to_string(in_a.numel())
          + ", B.numel(): " + std::to_string(in_b.numel()));

  out_c.resize_({N});

  // call the kernel function...
  AddGPUKernel<T>(in_a.data_ptr<T>(), in_b.data_ptr<T>(),
          out_c.data_ptr<T>(), N, at::cuda::getCurrentCUDAStream());
}

// instantiate the CPP template for T=float:
template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

// declare the extension module with the AddGPU function:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.doc() = "pybind11 example plugin";
  m.def("AddGPU", &AddGPU<float>);
}
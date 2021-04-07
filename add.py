import torch

from CustomPyTorchCUDAKernelBackend import AddGPU

def add_gpu(a, b):
    assert isinstance(a, torch.cuda.FloatTensor) 
    assert isinstance(b, torch.cuda.FloatTensor)
    assert a.numel() == b.numel()

    c = a.new()
    AddGPU(a, b, c)
    return c
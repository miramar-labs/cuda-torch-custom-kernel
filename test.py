import torch
import CustomPyTorchCUDAKernel as CustCUDA

if torch.cuda.is_available():
    a = torch.cuda.FloatTensor(4)
    b = torch.cuda.FloatTensor(4)
    a.normal_()
    b.normal_()
    c = CustCUDA.add_gpu(a, b)
    print(a, b, c)
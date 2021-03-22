import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

setup(
    name='CustomPyTorchCUDAKernel',
    version='1.0.0',
    install_requires=['torch'],
    packages=['CustomPyTorchCUDAKernel'],
    package_dir={'CustomPyTorchCUDAKernel': './'},
    ext_modules=[
        CUDAExtension(
            name='CustomPyTorchCUDAKernelBackend',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Aaron Cody',
    author_email='aaron@aaroncody.com',
    description='Custom PyTorch CUDA Extension',
    keywords='PyTorch CUDA Extension',
    url='https://github.com/miramar-labs/cuda-torch-custom-kernel',
    zip_safe=False,
)

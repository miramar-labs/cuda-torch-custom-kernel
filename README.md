# Custom CUDA Kernel for PyTorch

This codebase demonstrates how to:
- implement a simple CUDA kernel function
- interface it to PyTorch using pybind11 / ATen

Shortly I will add more code to this extension module that:
- demonstrates some common CUDA/GPU optimization techniques
- implements a custom layer of a PyTorch neural network

To Build:

- Create a Python virtualenv based on Python 3.8 and activate it.

        pushd cuda-torch-custom-kernel
        bash ../pyvenv.sh 3.8.0 .
        source ./venv3.8.0/bin/activate

- Install requirements:

        pip3 install -r requirements.txt

- Build extension module:

        python3 setup.py install --force

From a terminal window, launch JupyterLAB and load up test.ipynb - run that...

        jupyter lab
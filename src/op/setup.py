from telnetlib import EC
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
import os 

import torch

if torch.cuda.is_available() == False:
    raise Exception("No GPU found. Compile on a node with available GPU")

cuda_bin = os.environ.get("CUDADIR") 

if cuda_bin is None:
    raise Exception("CUDADIR environment variable is not set appropriately")

cuda_bin = cuda_bin + r"/bin"

sys.path.append(cuda_bin)

setup(
    name="op",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "op",
            [
                "calculate_force.cpp", 
                "kernel/calculateForce.cu",
            ],
        ),
        CUDAExtension(
            "op_grad",
            [
                "calculate_force_grad.cpp",
                "kernel/calculateForceGrad.cu"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

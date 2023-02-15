from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#import sys
#sys.path.append("/home/husiyu/tools/cuda-11/bin")

setup(
    name="op",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "op",
            [
                "calculate_force.cpp", 
                "kernel/calculateForce.cu",
                "calculate_force_grad.cpp",
                "kernel/calculateForceGrad.cu",
                "calculate_virial_force.cpp", 
                "kernel/calculateVirial.cu",
                "calculate_virial_force_grad.cpp", 
                "kernel/calculateVirialGrad.cu",
                "register_op.cpp",
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

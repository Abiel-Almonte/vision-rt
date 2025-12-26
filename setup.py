# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="_visionrt",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="_visionrt",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vision_rt",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="vision_rt",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_data={"vision_rt": ["*.so", "*.pyd", "*.pyi"]},
    zip_safe=False,
)
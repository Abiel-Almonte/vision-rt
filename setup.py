# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os 

venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    cudnn_include = os.path.join(venv_path, 'lib/python3.11/site-packages/nvidia/cudnn/include')
    cudnn_lib = os.path.join(venv_path, 'lib/python3.11/site-packages/nvidia/cudnn/lib')
else:
    raise RuntimeError("VIRTUAL_ENV not set")

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
            include_dirs=["csrc", cudnn_include],
            library_dirs=[cudnn_lib],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="nubzuki-softgroup",
    version="0.1.0",
    description="Challenge-local SoftGroup integration for Nubzuki instance segmentation",
    packages=find_packages(include=("softgroup", "softgroup.*")),
    ext_modules=[
        CUDAExtension(
            name="softgroup.ops.ops",
            sources=[
                "softgroup/ops/src/softgroup_api.cpp",
                "softgroup/ops/src/softgroup_ops.cpp",
                "softgroup/ops/src/cuda.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

from setuptools import Extension, setup
import numpy as np

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()

setup(
        name="RobustPolyfit",
        version="0.0.8",
        author="Jonathan Parkinson",
        description="Outlier-robust regression for polynomials 1 <= degree <= 3",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jlparki/outlier_robust_polyfit",
        ext_modules=[Extension("RobustPolyfit", ["src/RobustPolyfit.c"])],
        include_dirs=[np.get_include()],
        install_requires=["numpy", "scipy", "cython"]
)


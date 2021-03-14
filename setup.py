from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()

setup(
        name="RobustPolyfit",
        version="0.0.2",
        packages=find_packages(),
        author="Jonathan Parkinson",
        description="Outlier-robust regression for polynomials 1 <= degree <= 3",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jlparki/outlier_robust_polyfit",
        ext_modules=cythonize(["src/RobustPolyfit/__init__.pyx"]),
        include_dirs=np.get_include(),
        install_requires=["numpy", "scipy", "cython"]
)

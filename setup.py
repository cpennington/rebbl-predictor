from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("_rebbl_predictor.pyx", language_level="3")
)

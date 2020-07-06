import pybind11
from setuptools import setup, find_packages, Extension

polar_prefix = ""
polar_include = ""

ext_modules = [
    Extension(
        '_polar_direct',
        ['direct/module.cpp', 'direct/Context.cpp', 'direct/Graph.cpp'],
        include_dirs=[pybind11.get_include(),
        # The following 2 entries are in case we do in-tree build
        polar_prefix + "/export",
        polar_include,
        polar_prefix + "/include"],
        library_dirs=[ polar_prefix + "/lib"],
        libraries=['athena'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='PolarAI',
    version='0.0.1',
    zip_safe=False,
    packages=find_packages(),
    ext_modules=ext_modules,
    requires=['pybind11'],
    package_data={'': [polar_prefix + "/lib/*.so*"]},
    include_package_data=True
)

import os
from setuptools import setup

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('seymour.common', ['seymour/common/common.pyx']),
    Extension('seymour.genome', ['seymour/genome/genome.pyx']),
    Extension('seymour.ga', ['seymour/ga/ga.pyx']),
    Extension('seymour.net', ['seymour/net/net.pyx'])
]
    
setup(
    name = "seymour",
    version = "0.0.1",
    author = "Liam Schumm",
    author_email = "lschumm@protonmail.com",
    description = "A genetic algorithm solver library, primarily used for the solving of fixed-size deep neural nets.",
    url = "http://github.com/lschumm/seymour",
    packages=['seymour'],
    install_requires=['numpy'],
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)

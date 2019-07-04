import os
from setuptools import setup

setup(
    name = "seymour",
    version = "0.0.1",
    author = "Liam Schumm",
    author_email = "lschumm@protonmail.com",
    description = "A genetic algorithm solver library, primarily used for the solving of fixed-size deep neural nets.",
    url = "http://github.com/lschumm/seymour",
    packages=['seymour'],
    install_requires=['numpy'],
)

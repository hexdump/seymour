#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import os

setup(
    name='seymour',
    version='0.0.1a1',
    description='Seymour is a library for quickly prototyping genetic algorithms to solve machine learning problems using models which don\'t have traditional optimization techniques.',
    packages=find_packages(exclude=['tests']),
)

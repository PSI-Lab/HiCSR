#!/usr/bin/env python

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
              "numpy==1.16.3",
              "pyyaml==5.3",
              "matplotlib==3.1.3",
              "tensorboard==2.0.0",
              "torch==1.4.0",
              "pillow==8.3.2"
              ]

setup(
    name='HiCSR',
    version='1.0.0',
    description='Hi-C enhancement python package',
    url='https://github.com/PSI-Lab/HiCSR',
    author='Michael Dimmick',
    author_email='mdimmick@psi.toronto.edu',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=REQUIRED_PACKAGES
)


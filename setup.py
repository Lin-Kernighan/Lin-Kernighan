# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='lin_kernighan',
    version='0.1.0',
    description='Implementation of LKH Search',
    long_description=readme,
    author='Pleshcheev Dmitriy, Dmitriy Yampolskiy',
    author_email='plescheev.da@phystech.edu',
    url='https://github.com/Lin-Kernighan/Lin-Kernighan',
    license=license,
    install_requires=required,
    python_requires='>=3.8',
    packages=find_packages()
)

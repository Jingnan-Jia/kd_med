# -*- coding: utf-8 -*-
# @Time    : 7/27/21 5:23 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kd_med", # Replace with your own username
    version="0.0.12",
    author="Jingnan",
    author_email="jiajingnan2222@gmail.com",
    description="A plug-in for knowledge distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jingnan-jia/kd_med",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['googledrivedownloader', 'torch'],

)

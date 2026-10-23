#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import fnmatch
from setuptools import setup, find_packages


# Path to the directory that contains this setup.py file.
base_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(base_dir, "molscrub", "_version.py")) as f:
    version = re.search(r'__version__ = "(.+)"', f.read()).group(1)


def find_files(directory):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*"):
            matches.append(os.path.join(root, filename))
    return matches


setup(
    name="molscrub",
    author="Forli Lab",
    version=version,
    author_email="forli@scripps.edu",
    url="https://github.com/forlilab/molscrub",
    description="Enumerate states of small organic molecules: 3D coordinates, tautomers, pH-based adjustments",
    long_description=open(os.path.join(base_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    scripts=[
        "scripts/scrub.py",
    ],
    package_data={"molscrub": ["data/*"]},
    data_files=[("", ["README.md", "LICENSE"]), ("scripts", find_files("scripts"))],
    include_package_data=True,
    zip_safe=False,
    install_requires=["rdkit>=2022.03.1", "rich", "numpy", "joblib", "pandas"],
    python_requires=">=3.9",
    license="GPL-v3",
)

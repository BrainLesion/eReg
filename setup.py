#!/usr/bin/env python

"""The setup script."""


import sys, re
from setuptools import setup, find_packages

try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except Exception as error:
    readme = "No README information found."
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % ("README.md", error))

try:
    filepath = "eReg/version.py"
    version_file = open(filepath)
    (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())

except Exception as error:
    __version__ = "0.0.1"
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % (filepath, error))

requirements = [
    "black",
    "numpy==1.22.0",
    "pyyaml",
    "pytest",
    "pytest-cov",
    "requests",
    "SimpleITK!=2.0.*",
    "SimpleITK!=2.2.1",  # https://github.com/mlcommons/GaNDLF/issues/536
    "scikit-image",
    "setuptools",
    "tqdm",
]

if __name__ == "__main__":
    setup(
        name="eReg",
        version=__version__,
        author="FETS-AI",
        author_email="admin@fets.ai",
        python_requires=">=3.8",
        packages=find_packages(),
        entry_points={
            "console_scripts": [
                "register=eReg.cli.run:main",
            ],
        },
        classifiers=[
            "Development Status :: 1 - Planning",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
        ],
        description=("Template python project."),
        install_requires=requirements,
        license="Apache-2.0",
        long_description=readme,
        long_description_content_type="text/markdown",
        include_package_data=True,
        keywords="machine learning",
        zip_safe=False,
    )

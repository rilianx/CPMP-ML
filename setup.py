import os
import pathlib

from setuptools import find_packages
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = "2.1.0"

setup(
    name= "CPMP_ML",
    description= "Library to Container Pre Marshalling Problem (CPMP)",
    long_description_content_type= "text/markdown",
    long_description= README,
    version= VERSION,
    url= "https://github.com/rilianx/CPMP-ML.git",
    author= "Rilianx Team",
    install_requires= [
        "numpy",
        "tensorflow==2.18.0"
    ],
    python_requires= ">=3.11",
    packages= find_packages(
        include=("cpmp_ml", "cpmp_ml.*"),
        exclude=["views", "views.*", "gui", "gui.*"]
    ),
    package_data={
        "cpmp_ml": ["optimizer/GreedyV3/x86_64_feg", "optimizer/GreedyV3/arm64_feg"]
    },
    include_package_data=True
)

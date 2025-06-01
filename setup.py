from setuptools import find_packages
from setuptools import setup

setup(
    name='scots',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.9',
)


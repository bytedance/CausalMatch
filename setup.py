from setuptools import setup, find_packages
import os
import re

def readme():
    with open('README.md') as f:
        return f.read()


with open(os.path.join(os.path.dirname(__file__), "causalmatch", "_version.py")) as file:
    for line in file:
        m = re.fullmatch("__version__ = '([^']+)'\n", line)
        if m:
            version = m.group(1)

dependencies = [
    'pandas>=1.3.4, <=2.2.0',
    'numpy>=1.3.0, <=1.26.4'
]

setup(
    name    = "causalmatch",
    version = '0.0.1',
    author  = "xiaoyuzhou",
    author_email="xiaoyuzhou@bytedance.com",
    description="Propensity score matching and coarsened exact matching.",
    long_description=readme(),
    packages=find_packages(),
    install_requires=dependencies,
    python_requires='>=3.6'
)

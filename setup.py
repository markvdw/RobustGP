from setuptools import setup, find_packages

requirements = [
    'numpy>=1.18.1',
    'scipy>=1.4.1',
    'matplotlib>=3.1.3',
    'json_tricks',
    'jug'
]

setup(
    name='robustgp',
    version='0.0.2',
    packages=find_packages(),
    install_requires=requirements,
)

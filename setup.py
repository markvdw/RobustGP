from setuptools import setup, find_packages

requirements = [
    'numpy>=1.18.1',
    'tensorflow>=2.2.0',
    'scipy>=1.4.1',
    'matplotlib>=3.3.1',
    'json_tricks',
    'jug'
]

setup(
    name='robustgp',
    version='0.0.2',
    packages=find_packages(),
    install_requires=requirements,
)

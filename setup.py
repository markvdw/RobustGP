from setuptools import setup, find_packages

requirements = [
    'numpy>=1.16.4',
    'tensorflow>=2.0.0',
    'scipy>=1.3.0',
    'matplotlib>=3.1.0',
    'observations',
    'jupytext',
    'papermill',
    'json_tricks',
    'jug'
]

setup(
    name='inducing_init',
    version='0.0.2',
    packages=find_packages(),
    install_requires=requirements,
)

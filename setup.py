from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name = 'MLProject-Hotel-Reservation',
    version = '0.1',
    author = 'zackiferdinansyah144@gmail.com',
    packages = find_packages(),
    install_requires = requirements
)
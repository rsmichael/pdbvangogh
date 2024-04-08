from setuptools import setup, find_packages

setup(
    name='pdbvangogh',
    version='0.1',
    description='pdbvangogh implements style transfer on PDB structures with artistic backgrounds',
    packages=find_packages(where='src/'),
    package_dir={'': 'src/'}, 
    install_requires=[
       'pytest==8.1.1',
       'tensorflow==2.15.0',
       'tensorflow-hub==0.12.0'

    ],
)
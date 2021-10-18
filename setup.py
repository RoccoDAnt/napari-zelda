from setuptools import setup, find_packages


use_scm={"write_to": "napari_zelda/_version.py"}

setup(
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True
    )

from setuptools import setup, find_packages

use_scm={"write_to": "napari_zelda/_version.py"}

setup(
    name='napari-zelda',
    version='0.1.2',
    author="Rocco D'Antuono, Giuseppina Pisignano",
    description="ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari",
    download_url = 'https://github.com/RoccoDAnt/napari-zelda/archive/refs/tags/v0.1.2.tar.gz',
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    include_package_data=True
    )

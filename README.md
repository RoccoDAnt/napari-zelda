# napari-zelda

[![License](https://img.shields.io/pypi/l/napari-zelda.svg?color=green)](https://github.com/RoccoDAnt/napari-zelda/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-zelda.svg?color=green)](https://pypi.org/project/napari-zelda)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-zelda.svg?color=green)](https://python.org)
[![tests](https://github.com/RoccoDAnt/napari-zelda/workflows/tests/badge.svg)](https://github.com/RoccoDAnt/napari-zelda/actions)
[![codecov](https://codecov.io/gh/RoccoDAnt/napari-zelda/branch/master/graph/badge.svg)](https://codecov.io/gh/RoccoDAnt/napari-zelda)

## ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari
###### Authors: Rocco D'Antuono, Giuseppina Pisignano
----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## What you can do with ZELDA plugin for napari
1. Segment objects such as cells and organelles in 2D/3D.

2. Segment two populations in 2D/3D (e.g. cells and organelles, nuclei and nuclear spots, tissue structures and cells) establishing the "Parent-Child" relation: count how many mitochondria are contained in each cell, how many spots localize in every nucleus, how many cells are within a tissue compartment.

  Example: cell cytoplasms (parent objects) and mitochondria (child objects)
  ![](/docs/2D-AF488.png) <br> **Actin** | ![](/docs/2D-MT.png) <br> **Mitochondria**| ![](/docs/2D-AF488_MT.png) <br> **Merge**
  ------ | ------| -----
  ![](/docs/2D-AF488_parents.png) <br> **Parent cell cytoplasms** | ![](/docs/2D-MT_children.png) <br> **Children mitochondria**|  ![](/docs/2D-MT_childrenbyParent.png) <br> **Children labelled by Parents**


3. Plot results within napari interface.

    ![](/docs/Plot_hist_Area.png) <br> **Histogram** | ![](/docs/Plot_scatter_Area-EqDiam.png) <br> **Scatterplot**|
    ------ | ------|
4. Customize an image analysis workflow in graphical mode (no scripting knowledge required).
  ![](/docs/CustomProtocol.png) <br> **Custom image analysis workflow** |
  ------ |


## Installation

**Option A.** You can install `napari-zelda` via [pip]. For the best experience, create a conda environment, and use napari 0.4.7, using the following instructions:

    conda create -y -n napari-env python==3.8  
    conda activate napari-env  
    pip install "napari[all]"  
    pip install napari==0.4.7  
    pip install napari-zelda  


**Option B.** Alternatively, clone the repository and install locally via [pip]:

    pip install -e .

**Option C.** Another option is to use the napari interface to install it (up to napari version 0.4.10 but the best experience is with napari 0.4.7):
1. Plugins / Install/Uninstall Package(s)
![](/docs/Clipboard_ZELDA_Plugin_install_in_napari.png)

2. Choose ZELDA
![](/docs/Clipboard_ZELDA_Plugin_install_ZELDA_in_napari_Arrow.png)

3. ZELDA is installed
![](/docs/Clipboard_ZELDA_Plugin_installed_ZELDA_in_napari_Arrow.png)

4. Launch ZELDA
![](/docs/Clipboard_ZELDA_Launch_ZELDA.png)


## Contributing

Contributions are welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-zelda" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

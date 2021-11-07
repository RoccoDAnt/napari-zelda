# napari-zelda

[![License](https://img.shields.io/pypi/l/napari-zelda.svg?color=green)](https://github.com/RoccoDAnt/napari-zelda/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-zelda.svg?color=green)](https://pypi.org/project/napari-zelda)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-zelda.svg?color=green)](https://python.org)
[![tests](https://github.com/RoccoDAnt/napari-zelda/workflows/tests/badge.svg)](https://github.com/RoccoDAnt/napari-zelda/actions)
[![codecov](https://codecov.io/gh/RoccoDAnt/napari-zelda/branch/master/graph/badge.svg)](https://codecov.io/gh/RoccoDAnt/napari-zelda)

## ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari
##### Authors: Rocco D'Antuono, Giuseppina Pisignano
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
  ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488.png) <br> **Actin** | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT.png) <br> **Mitochondria**| ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488_MT.png) <br> **Merge**
  ------ | ------| -----
  ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488_parents.png) <br> **Parent cell cytoplasms** | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT_children.png) <br> **Children mitochondria**| ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT_childrenbyParent.png) <br> **Children labelled by Parents**


3. Plot results within napari interface.

    ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Plot_hist_Area.png) <br> **Histogram** | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Plot_scatter_Area-EqDiam.png) <br> **Scatterplot**|
    ------ | ------|

4. Customize an image analysis workflow in graphical mode (no scripting knowledge required).

  | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/CustomProtocol.png) <br> **Custom image analysis workflow** |
  ------ |

5. Import and Export Protocols (image analysis workflows) in graphical mode (share with the community!).

    | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_Import_and_Export_Protocols.png) <br> **Import and Export of ZELDA Protocols** |
    ------ |

## Installation

**Option A.** You can install `napari-zelda` via [pip]. For the best experience, create a conda environment and use napari!=0.4.11, using the following instructions:

    conda create -y -n napari-env python==3.8  
    conda activate napari-env  
    pip install "napari[all]"  
    pip install napari-zelda  


**Option B.** Alternatively, clone the repository and install locally via [pip]:

    pip install -e .

**Option C.** Another option is to use the napari interface to install it (make sure napari!=0.4.11):
1. Plugins / Install/Uninstall Package(s)
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_install_in_napari.png)

2. Choose ZELDA
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_install_ZELDA_in_napari_Arrow.png)

3. ZELDA is installed
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_installed_ZELDA_in_napari_Arrow.png)

4. Launch ZELDA
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Launch_ZELDA.png)


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

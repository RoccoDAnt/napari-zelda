# napari-zelda

[![License](https://img.shields.io/pypi/l/napari-zelda.svg?color=green)](https://github.com/RoccoDAnt/napari-zelda/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-zelda.svg?color=green)](https://pypi.org/project/napari-zelda)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-zelda.svg?color=green)](https://python.org)
[![tests](https://github.com/RoccoDAnt/napari-zelda/workflows/tests/badge.svg)](https://github.com/RoccoDAnt/napari-zelda/actions)
[![codecov](https://codecov.io/gh/RoccoDAnt/napari-zelda/branch/master/graph/badge.svg)](https://codecov.io/gh/RoccoDAnt/napari-zelda)

## ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari
#### Authors: Rocco D'Antuono, Giuseppina Pisignano

###### Article: Front. Comput. Sci., 04 January 2022 | https://doi.org/10.3389/fcomp.2021.796117

###### Examples of 2D and 3D data sets: [https://doi.org/10.5281/zenodo.5651284](https://zenodo.org/record/5651284#.YYgn_WDP2Ch)
----------------------------------

## What you can do with ZELDA plugin for napari
The plugin can be used to analyze 2D/3D image data sets.  
Multidimensional images (each channel corresponding to a napari layer) can be used to:

1. Segment objects such as cells and organelles in 2D/3D.

2. Segment two populations in 2D/3D (e.g. cells and organelles, nuclei and nuclear spots, tissue structures and cells) establishing the "Parent-Child" relation: count how many mitochondria are contained in each cell, how many spots localize in every nucleus, how many cells are within a tissue compartment.

  Example: cell cytoplasms (parent objects) and mitochondria (child objects)
  ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488.png) <br> **Actin** | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT.png) <br> **Mitochondria**| ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488_MT.png) <br> **Merge**
  ------ | ------| -----
  ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-AF488_parents.png) <br> **Parent cell cytoplasms** | ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT_children.png) <br> **Children mitochondria**| ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/2D-MT_childrenbyParent.png) <br> **Children labelled by Parents**

The images shown above are available in the [**docs**](https://github.com/RoccoDAnt/napari-zelda/tree/main/docs) folder of this repository and were segmented using ZELDA with the following parameters:


   | **Parent objects** | **GB: sigma=2.0-> Th_parents=60.0-> DistMap-> Maxima: min_dist=10** |
   | -----|  ----|
   | **Children objects** | **GB: sigma=0.3-> Th_children=450.0 -> DistMap-> Maxima: min_dist=2**|

For small monitors it may be convenient to float the protocol panel

  |![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin-set_panel_to_float.png) <br> **Float a panel in napari** |
  ------ |

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

**Option A.** The easiest option is to use the napari interface to install ZELDA (make sure napari!=0.4.11):
1. Plugins / Install/Uninstall Package(s)

  ![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_install_in_napari.png)

2. Choose ZELDA
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_install_ZELDA_in_napari_Arrow.png)

3. ZELDA is installed
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Plugin_installed_ZELDA_in_napari_Arrow.png)

4. Launch ZELDA
![](https://raw.githubusercontent.com/RoccoDAnt/napari-zelda/main/docs/Clipboard_ZELDA_Launch_ZELDA.png)


**Option B.** You can install `napari-zelda` also via [pip]. For the best experience, create a conda environment and use napari!=0.4.11, using the following instructions:

    conda create -y -n napari-env python==3.8  
    conda activate napari-env
    conda install napari pyqt  
    pip install napari-zelda  


**Option C.** Alternatively, clone the repository and install locally via [pip]:

    pip install -e .

**Option D.** Get the latest code with [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [pip]:

    conda create -y -n napari-env python=3.8 git
    conda activate napari-env
    conda install napari pyqt
    pip install git+https://github.com/RoccoDAnt/napari-zelda.git


## Specifications

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

The GUI has been developed using [magicgui](https://github.com/napari/magicgui) widgets, while the image analysis and processing include functions from [scikit-image](https://scikit-image.org/), [SciPy](https://scipy.org/), and [NumPy](https://numpy.org/). Results are handled with [pandas](https://pandas.pydata.org/) and [datatable](https://datatable.readthedocs.io/en/latest/). Plots are obtained with [matplotlib](https://matplotlib.org/).  
<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->


## Contributing

Contributions are welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

Users can add new protocol steps to their local installation using [magicgui](https://github.com/napari/magicgui) widgets.
Code can be added at the end of napari_zelda.py file:

>###Add here new functionalities for ZELDA ###
>
>###@magicgui(layout="vertical")
>
>###def new_functionality_widget(viewer: 'napari.Viewer'):
>
>###...
>
>###
>
>###End###



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

[file an issue]: https://github.com/RoccoDAnt/napari-zelda/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

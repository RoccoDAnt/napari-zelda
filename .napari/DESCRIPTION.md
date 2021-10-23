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

**Option A.** You can install `napari-zelda` via [pip]. For the best experience, create a conda environment and use napari!=0.4.11, using the following instructions:

    conda create -y -n napari-env python==3.8  
    conda activate napari-env  
    pip install "napari[all]"  
    pip install napari!=0.4.11  
    pip install napari-zelda  


**Option B.** Alternatively, clone the repository and install locally via [pip]:

    pip install -e .

**Option C.** Another option is to use the napari interface to install it (make sure napari!=0.4.11):
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


<!-- This file is designed to provide you with a starting template for documenting
the functionality of your plugin. Its content will be rendered on your plugin's
napari hub page.

The sections below are given as a guide for the flow of information only, and
are in no way prescriptive. You should feel free to merge, remove, add and
rename sections at will to make this document work best for your plugin.

# Description

This should be a detailed description of the context of your plugin and its
intended purpose.

If you have videos or screenshots of your plugin in action, you should include them
here as well, to make them front and center for new users.

You should use absolute links to these assets, so that we can easily display them
on the hub. The easiest way to include a video is to use a GIF, for example hosted
on imgur. You can then reference this GIF as an image.

![Example GIF hosted on Imgur](https://i.imgur.com/A5phCX4.gif)

Note that GIFs larger than 5MB won't be rendered by GitHub - we will however,
render them on the napari hub.

The other alternative, if you prefer to keep a video, is to use GitHub's video
embedding feature.

1. Push your `DESCRIPTION.md` to GitHub on your repository (this can also be done
as part of a Pull Request)
2. Edit `.napari/DESCRIPTION.md` **on GitHub**.
3. Drag and drop your video into its desired location. It will be uploaded and
hosted on GitHub for you, but will not be placed in your repository.
4. We will take the resolved link to the video and render it on the hub.

Here is an example of an mp4 video embedded this way.

https://user-images.githubusercontent.com/17995243/120088305-6c093380-c132-11eb-822d-620e81eb5f0e.mp4

# Intended Audience & Supported Data

This section should describe the target audience for this plugin (any knowledge,
skills and experience required), as well as a description of the types of data
supported by this plugin.

Try to make the data description as explicit as possible, so that users know the
format your plugin expects. This applies both to reader plugins reading file formats
and to function/dock widget plugins accepting layers and/or layer data.
For example, if you know your plugin only works with 3D integer data in "tyx" order,
make sure to mention this.

If you know of researchers, groups or labs using your plugin, or if it has been cited
anywhere, feel free to also include this information here.

# Quickstart

This section should go through step-by-step examples of how your plugin should be used.
Where your plugin provides multiple dock widgets or functions, you should split these
out into separate subsections for easy browsing. Include screenshots and videos
wherever possible to elucidate your descriptions.

Ideally, this section should start with minimal examples for those who just want a
quick overview of the plugin's functionality, but you should definitely link out to
more complex and in-depth tutorials highlighting any intricacies of your plugin, and
more detailed documentation if you have it.

# Additional Install Steps (uncommon)
We will be providing installation instructions on the hub, which will be sufficient
for the majority of plugins. They will include instructions to pip install, and
to install via napari itself.

Most plugins can be installed out-of-the-box by just specifying the package requirements
over in `setup.cfg`. However, if your plugin has any more complex dependencies, or
requires any additional preparation before (or after) installation, you should add
this information here.

# Getting Help

This section should point users to your preferred support tools, whether this be raising
an issue on GitHub, asking a question on image.sc, or using some other method of contact.
If you distinguish between usage support and bug/feature support, you should state that
here.

# How to Cite

Many plugins may be used in the course of published (or publishable) research, as well as
during conference talks and other public facing events. If you'd like to be cited in
a particular format, or have a DOI you'd like used, you should provide that information here. -->

The developer has not yet provided a napari-hub specific description.

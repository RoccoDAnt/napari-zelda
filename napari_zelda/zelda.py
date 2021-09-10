"""
ZELDA: a 3D Image Segmentation and Parent to Child relation plugin for microscopy image analysis in napari
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from napari.layers import Image, Labels, Layer, Points
from magicgui import magicgui

@magicgui(
         threshold={'widget_type': 'Slider', "max": 5000, 'min':0},
         call_button="Apply")
def threshold(layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        #print(napari.types.ImageData)
        th=layer.data.mean(-1)>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+'of '+str(layer.name))

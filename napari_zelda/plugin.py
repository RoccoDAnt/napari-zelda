from napari_plugin_engine import napari_hook_implementation
from .napari_zelda import launch_ZELDA
import napari
from napari import Viewer


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return launch_ZELDA()

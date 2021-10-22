
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

def napari_experimental_provide_dock_widget():
    from .napari_zelda import launch_ZELDA
    return launch_ZELDA

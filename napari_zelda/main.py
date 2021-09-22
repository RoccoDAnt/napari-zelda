
import napari
from .napari_zelda import launch_ZELDA
from napari import Viewer

def main():
    #viewer = napari.Viewer()
    #viewer.window.add_dock_widget(launch_ZELDA, area='bottom')
    napari.run()

if __name__ == '__main__':
    main()

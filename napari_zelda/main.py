
import napari
from .napari_zelda import zelda

def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(threshold, area='bottom')
    napari.run()

if __name__ == '__main__':
    main()

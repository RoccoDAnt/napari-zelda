from napari_zelda import napari_experimental_provide_dock_widget


def test_import_and_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget_factory = napari_experimental_provide_dock_widget()
    widget = widget_factory()
    viewer.window.add_dock_widget(widget)
    assert len(viewer.window._dock_widgets) > 0

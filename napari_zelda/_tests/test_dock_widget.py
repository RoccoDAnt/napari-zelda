from npe2 import PluginManifest
import napari_zelda


def test_npe2_manifest():
    manifest = PluginManifest.from_distribution("napari-zelda")
    assert manifest.name == "napari-zelda"
    assert len(manifest.contributions.widgets) > 0


def test_import_and_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-zelda",
        widget_name="ZELDA",
    )
    assert len(viewer.window._dock_widgets) > 0
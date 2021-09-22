"""
ZELDA: a 3D Image Segmentation and Parent to Child relation plugin for microscopy image analysis in napari
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QGridLayout
from napari.layers import Image, Labels, Layer, Points
from magicgui import magicgui, magic_factory
import napari
from napari import Viewer
from magicgui.widgets import SpinBox, FileEdit, Slider, FloatSlider, Label, Container, MainWindow, ComboBox, TextEdit
import skimage.filters
from skimage.feature import peak_local_max
from skimage.transform import rotate
from skimage.segmentation import watershed
#from skimage.morphology import watershed
from skimage import measure
import numpy as np
import pandas as pd
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


@magicgui(
         labels=False,
         threshold={'widget_type': 'FloatSlider', "max": 5000.0, 'min':0.0},
         call_button="Apply",
         persist=True
         )
def threshold_one_pop(viewer: 'napari.Viewer', layer: Image, label: str='Threshold', threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magicgui(labels=False,
         threshold={'widget_type': 'FloatSlider', "max": 5000.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_parent(viewer: 'napari.Viewer',layer: Image, label: str='Threshold Parent Population', threshold: int = 1)-> napari.types.ImageData:
    if layer:
        #print(napari.types.ImageData)
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magicgui(labels=False,
         threshold={'widget_type': 'FloatSlider', "max": 5000.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_children(viewer: 'napari.Viewer',layer: Image, label: str='Threshold Children Population', threshold: int = 1)-> napari.types.ImageData:
    if layer:
        #print(napari.types.ImageData)
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))


@magicgui(labels=False,
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_one_pop(viewer: 'napari.Viewer',layer: Image, label: str='Gaussian Blur', sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    #sigma.changed.connect(set_label)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))
        #return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)

@magicgui(labels=False,
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_parent_pop(viewer: 'napari.Viewer',layer: Image, label: str='Gaussian Blur: Parent Pop', sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    #sigma.changed.connect(set_label)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))
        #return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)

@magicgui(labels=False,
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_children_pop(viewer: 'napari.Viewer',layer: Image, label: str='Gaussian Blur: Children Pop', sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    #sigma.changed.connect(set_label)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))
        #return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)

@magicgui(labels=False, call_button="Get DistanceMap", persist=True)
def distance_map_one_pop(viewer: 'napari.Viewer',layer: Image,  label: str='Distance map')-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(call_button="Get DistanceMap for Parent", persist=True)
def distance_map_parent_pop(viewer: 'napari.Viewer',layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(call_button="Get DistanceMap for Children", persist=True)
def distance_map_children_pop(viewer: 'napari.Viewer',layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(call_button="Show seeds", persist=True)
def show_seeds_one_pop(viewer: 'napari.Viewer', DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(call_button="Show seeds", persist=True)
def show_seeds_parent_pop(viewer: 'napari.Viewer', DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(call_button="Show seeds", persist=True)
def show_seeds_children_pop(viewer: 'napari.Viewer', DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(call_button="Watershed", persist=True)
def watershed_one_pop(viewer: 'napari.Viewer',DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(call_button="Watershed", persist=True)
def watershed_parent_pop(viewer: 'napari.Viewer',DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled Parent objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(call_button="Watershed", persist=True)
def watershed_children_pop(viewer: 'napari.Viewer',DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled Children objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(call_button="Measure objects",
          save_log={'widget_type':'CheckBox','name':'Save_Log','text':'Save Log'},
          save_to_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def measure_one_pop(labels: Image, original: Image, save_log, save_to_path):
    properties=measure.regionprops_table(labels.data, original.data,
               properties= ['area', 'mean_intensity','equivalent_diameter'])
    prop={'Area': properties['area'],'Equivalent_diameter': properties['equivalent_diameter'],'MFI': properties['mean_intensity']}
    prop_df=pd.DataFrame(prop)
    prop_df.to_csv(str(save_to_path)+'\Results.csv')

    log=Label(name='Log:', tooltip=None,)
    log.value="-> Th="+str(threshold_one_pop.threshold.value)+"-> GB: sigma="+str(gaussian_blur_one_pop.sigma.value)+"-> DistMap"
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_one_pop.min_dist.value) + " -> Found n="+str(len(prop_df))+ " objects"
    measure_one_pop.insert(4,log)

    if save_log == True:
        Log_file = open(str(save_to_path)+'\Log_ZELDA.txt','w')
        Log_file.write(log.value)
        Log_file.close()

@magicgui(call_button="Relate and Measure",
          save_to_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def relate_and_measure(viewer: 'napari.Viewer', Parents_labels: Image, Children_labels: Image, Original_to_measure: Image, save_to_path):
    properties=measure.regionprops_table(Children_labels.data, Original_to_measure.data,
               properties= ['label','area', 'mean_intensity','equivalent_diameter'])
    binary_ch=Children_labels.data>0
    corresponding_parents=Parents_labels.data*binary_ch
    viewer.add_image(corresponding_parents, rgb=False, name='Labelled objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

    properties_CorrespondingParent=measure.regionprops_table(Children_labels.data, Parents_labels.data, properties=['min_intensity'])
    prop={'Parent_label': properties_CorrespondingParent['min_intensity'],'Area': properties['area'],'Equivalent_diameter': properties['equivalent_diameter'],'MFI': properties['mean_intensity']}
    prop_df=pd.DataFrame(prop)
    prop_df.to_csv(str(save_to_path)+'\Results_parents-children.csv')

    log=Label(name='Log:', tooltip=None,)
    log.value="-> Th_parents="+str(threshold_parent_pop.threshold.value)+"-> GB: sigma="+str(gaussian_blur_one_pop.sigma.value)+"-> DistMap"
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_one_pop.min_dist.value) + " -> Found n="+str(len(prop_df))+ " objects"
    measure_one_pop.insert(4,log)

    if save_log == True:
        Log_file = open(r''+str(save_to_path)+'\Log_ZELDA.txt','w')
        Log_file.write(log.value)
        Log_file.close()

@magicgui(layout="vertical",
          table_path={'widget_type': 'FileEdit', 'value':'Documents\properties.csv', 'mode':'r','filter':'*.csv'},
          plot_h={'widget_type':'CheckBox','name':'Histogram','text':'Histogram'},
          plot_s={'widget_type':'CheckBox','name':'Scatterplot','text':'Scatterplot'},
          save_plots={'widget_type':'CheckBox','name':'Save_plots','text':'Save plots'},
          saveTo_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          histogram={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
          scatterplot_X={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
          scatterplot_Y={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
          persist=True,
          call_button="Re-plot",
          result_widget=False
          )
def results_widget(viewer: 'napari.Viewer',
                   table_path,
                   plot_h,
                   plot_s,
                   save_plots,
                   saveTo_path,
                   histogram: str='Area',
                   scatterplot_X: str='Area',
                   scatterplot_Y: str='MFI'
                   ):
    table=pd.read_csv(table_path)

    if plot_h== True:
        plot_widget_histogram = FigureCanvas(Figure(figsize=(2, 1.5), dpi=150))
        ax = plot_widget_histogram.figure.subplots()
        ax.set(xlim=(0, 10*np.median(table[str(histogram)])), ylim=(0,len(table)))
        ax.set_title('Histogram of '+histogram, color='gray')
        ax.set_xlabel(str(histogram))
        ax.set_ylabel('Counts')
        ax.hist(data=table, x=str(histogram), color='blue', bins=50)
        ax.tick_params(labelright=False, right=False,labeltop=False,top=False,colors='black')
        plot_widget_histogram.figure.set_tight_layout('tight')

        viewer.window.add_dock_widget(plot_widget_histogram ,name='Plot results',area='bottom')
        if save_plots== True:
            plot_widget_histogram.print_tiff(str(saveTo_path)+'\Histogram of '+histogram+'.tiff')
    if plot_s== True:
        plot_widget_scattering = FigureCanvas(Figure(figsize=(2, 1.5), dpi=150))
        ax = plot_widget_scattering.figure.subplots()
        ax.set(xlim=(0, 10*np.median(table[str(scatterplot_X)])), ylim=(0, 10*np.median(table[str(scatterplot_Y)])))
        ax.set_title('Scatterplot of '+str(scatterplot_X)+' vs '+str(scatterplot_Y), color='gray')
        ax.set_xlabel(str(scatterplot_X))
        ax.set_ylabel(str(scatterplot_Y))
        ax.scatter(data=table, x=scatterplot_X, y=scatterplot_Y, color='blue')
        ax.tick_params(labelright=False, right=False,labeltop=False,top=False,colors='black')
        plot_widget_scattering.figure.set_tight_layout('tight')
        viewer.window.add_dock_widget(plot_widget_scattering,name='Plot results',area='bottom')
        if save_plots== True:
            plot_widget_scattering.print_tiff(str(saveTo_path)+'\Scatterplot of '+str(scatterplot_X)+' vs '+str(scatterplot_Y)+'.tiff')


@magic_factory(
               auto_call=False,
               call_button=True,
               dropdown={"choices": ['Segment a single population',
                                     'Segment two populations and relate',
                                     'Data Plotter']},
               protocol_descriptions={'widget_type': 'TextEdit', 'value':''},
               labels=False
                )
def launch_ZELDA(
        viewer: 'napari.Viewer',
        protocol_descriptions: str='ZELDA plugin for napari.\nChoose a Protocol and Start',
        dropdown: str= 'Segment a single population',
        ):
        dock_widgets=MainWindow(name='ZELDA protocol', annotation=None, label=None, tooltip=None, visible=True,
                               enabled=True, gui_only=False, backend_kwargs={}, layout='vertical', widgets=(), labels=True)
        viewer.window.add_dock_widget(dock_widgets, name='ZELDA: Protocol')

        launch_ZELDA.protocol_descriptions.value=('"ZELDA: a 3D Image Segmentation and Parent to Child relation plugin for microscopy image analysis in napari".\n\n'
                                +'\nPROTOCOL DESCRIPTIONS \n\n- "Segment a single population".\n'
                                +'Suggested steps: \n1. Gaussian Blur \n2. Threshold \n3. Distance map \n4. Show seeds \n5. Watershed \n6. Measure objects \n7. Plot data'
                                +'\n\n- "Segment two populations and relate".\nThe protocol allows to segment in parallel two populations.'
                                +'\nThe larger objects, called Parents, may contain the smallest ones called "Children". \nSuggested steps:'
                                +'\n1. Gaussian Blur \n2. Threshold \n3. Distance map \n4. Show seeds \n5. Watershed \n6. Relate and Measure objects \n7. Plot data using Data Plotter protocol'
                                +'\n\n- "Data Plotter" protocol.\n1. Load a result table \n2. Use the Histogram/Scatter tool to explore the data \n3. Save the plots')

        if dropdown == 'Segment a single population':
            single_pop_protocol=Container(name='Single Population', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            single_pop_protocol.insert(0, gaussian_blur_one_pop)
            single_pop_protocol.insert(1, threshold_one_pop)
            single_pop_protocol.insert(2, distance_map_one_pop)
            single_pop_protocol.insert(3, show_seeds_one_pop)
            single_pop_protocol.insert(4, watershed_one_pop)
            single_pop_protocol.insert(5, measure_one_pop)
            single_pop_protocol.insert(6,results_widget)
            dock_widgets.insert(0,single_pop_protocol)
            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Segment two populations and relate':
            parent_pop_protocol=Container(name='Parent Population', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='horizontal', labels=False)
            parent_pop_protocol.insert(0, gaussian_blur_parent_pop)
            parent_pop_protocol.insert(1, threshold_parent)
            parent_pop_protocol.insert(2, distance_map_parent_pop)
            parent_pop_protocol.insert(3, show_seeds_parent_pop)
            parent_pop_protocol.insert(4, watershed_parent_pop)


            children_pop_protocol=Container(name='Children Population', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            children_pop_protocol.insert(0, gaussian_blur_children_pop)
            children_pop_protocol.insert(1, threshold_children)
            children_pop_protocol.insert(2, distance_map_children_pop)
            children_pop_protocol.insert(3, show_seeds_children_pop)
            children_pop_protocol.insert(4, watershed_children_pop)

            dock_widgets.insert(0,parent_pop_protocol)
            dock_widgets.insert(1,children_pop_protocol)
            viewer.window.add_dock_widget(relate_and_measure, name='ZELDA: Relate and Measure',area='bottom')

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Data Plotter':
            data_plotter_protocol=Container(name='Results plotter', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            data_plotter_protocol.insert(0,results_widget)
            dock_widgets.insert(0,data_plotter_protocol)
            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

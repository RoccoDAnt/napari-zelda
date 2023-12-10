"""
ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QGridLayout, QGroupBox
from napari.layers import Image, Labels, Layer, Points
from magicgui import magicgui, magic_factory
import napari
from napari import Viewer
try:
    from napari.settings import SETTINGS
except ImportError:
    print("Warning: import of napari.settings failed - 'save window geometry' option will not be used")
from magicgui.widgets import SpinBox, FileEdit, Slider, FloatSlider, Label, Container, MainWindow, ComboBox, TextEdit, PushButton, ProgressBar, Select
import skimage
import skimage.morphology
import skimage.filters
from skimage.feature import peak_local_max
from skimage.transform import rotate
from skimage.segmentation import watershed
from skimage import measure
import numpy as np
import pandas as pd
#import datatable as dt #if using datatable instead of pandas
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import json
import os
import inspect

def mock():
    return

path=os.path.abspath(inspect.getfile(mock))
prot_path=os.path.dirname(os.path.dirname(path))

protocols_file=open(os.path.join(prot_path,'napari_zelda','protocols_dict.json'), "rb")

protocols_json = json.load(protocols_file)
protocols=list()
for i in range(0,len(protocols_json['Protocols'])):
    protocols.append(protocols_json['Protocols'][i]['name'])

protocols_file.seek(0)
protocols_file.close()

corresponding_widgets={
                        "Threshold": "threshold_one_pop",
                        "GaussianBlur": "gaussian_blur_one_pop",
                        "DistanceMap": "distance_map_one_pop",
                        "ShowSeeds":"show_seeds_one_pop",
                        "Watershed":"watershed_one_pop",
                        "Measure": "measure_one_pop",
                        "Plot": "results_widget",
                        "Image Calibration": "image_calibration",
                        "Morphological Operations":"morphological_operations",
                        "Filter by Area":"filterByArea_widget",
                        "Median Filter":"median_filter"
                        }

protocols_description=open(os.path.join(prot_path,'napari_zelda','protocols_description.txt'), 'r').read()

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Threshold"},
         Otsu={'widget_type':'CheckBox','name':'Otsu_threshold'},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True
         )
def threshold_one_pop(viewer: 'napari.Viewer', label, Otsu, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        if Otsu==True:
            threshold = skimage.filters.threshold_otsu(np.array(layer.data))
        th=layer.data>threshold
        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Threshold - Parents"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_parents(viewer: 'napari.Viewer', label, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Threshold - Children"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_children(viewer: 'napari.Viewer', label, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))


@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Gaussian Blur"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_one_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, scale=layer.scale, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Gaussian Blur - Parents"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_parent_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, scale=layer.scale, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Gaussian Blur - Children"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_children_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, scale=layer.scale, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Distance Map"}, call_button="Get DistanceMap", persist=True)
def distance_map_one_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, scale=layer.scale, name='DistMap of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Distance Map - Parents"}, call_button="Get DistanceMap", persist=True)
def distance_map_parent_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, scale=layer.scale, name='DistMap of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Distance Map - Children"}, call_button="Get DistanceMap", persist=True)
def distance_map_children_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, scale=layer.scale, name='DistMap of '+str(layer.name))

@magicgui(label={'widget_type':'Label', 'label':"Show seeds"}, call_button="Show seeds", persist=True)
def show_seeds_one_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist, exclude_border=False)
        points = np.array(coords)
        viewer.add_points(points*DistanceMap.scale, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Show seeds - Parents"}, call_button="Show seeds", persist=True)
def show_seeds_parent_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist, exclude_border=False)
        points = np.array(coords)
        viewer.add_points(points*DistanceMap.scale, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Show seeds - Children"}, call_button="Show seeds", persist=True)
def show_seeds_children_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist, exclude_border=False)
        points = np.array(coords)
        viewer.add_points(points*DistanceMap.scale, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Segment"}, call_button="Watershed", persist=True)
def watershed_one_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        seeds=np.array((np.reciprocal(DistanceMap.scale)*seeds.data).astype(int))
        mask[tuple((seeds).T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, scale=DistanceMap.scale, rgb=False, name='Labelled objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Segment Parents"}, call_button="Watershed", persist=True)
def watershed_parent_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        seeds=np.array((np.reciprocal(DistanceMap.scale)*seeds.data).astype(int))
        mask[tuple((seeds).T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, scale=DistanceMap.scale, rgb=False, name='Labelled Parent objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Segment Children"}, call_button="Watershed", persist=True)
def watershed_children_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        seeds=np.array((np.reciprocal(DistanceMap.scale)*seeds.data).astype(int))
        mask[tuple((seeds).T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, scale=DistanceMap.scale, rgb=False, name='Labelled Children objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Measure segmented objects"}, call_button="Measure objects",
          save_log={'widget_type':'CheckBox','name':'Save_Log','text':'Save Log'},
          save_to={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def measure_one_pop( label, labels: Image, original: Image, save_log, save_to):
    voxel_size=np.prod(original.scale)
    properties=measure.regionprops_table(labels.data, original.data, properties= ['label','area', 'intensity_mean','intensity_min','intensity_max','equivalent_diameter','axis_major_length','axis_minor_length','centroid','centroid_weighted','extent','solidity'])
    prop={'Label': properties['label'], 'Area': properties['area']*original.scale[-1]*original.scale[-2], 'Volume': properties['area']*voxel_size,'Equivalent_diameter': properties['equivalent_diameter']*original.scale[-1],'MFI': properties['intensity_mean'],
    'Min_Intensity': properties['intensity_min'], 'Max_Intensity': properties['intensity_max'],'MajorAxis_Length': properties['axis_major_length']*original.scale[-1],
    'MinorAxis_Length': properties['axis_minor_length']*original.scale[-1],
    'Extent': properties['extent'],
    'Solidity': properties['solidity']
    }
    if  len(original.data.shape)==2:
        prop['Centroid_X']= properties['centroid-1']*original.scale[-1]
        prop['Centroid_Y']= properties['centroid-0']*original.scale[-2]

        prop['Weighted_Centroid_X']= properties['centroid_weighted-1']*original.scale[-1]
        prop['Weighted_Centroid_Y']= properties['centroid_weighted-0']*original.scale[-2]

        additional_properties_2D=measure.regionprops_table(labels.data, original.data, properties= ['orientation','perimeter'])
        prop['Orientation']= additional_properties_2D['orientation']
        prop['Perimeter']= additional_properties_2D['perimeter']*original.scale[-1]

    if len(original.data.shape)==3:
        prop['Centroid_X']= properties['centroid-2']*original.scale[-1]
        prop['Centroid_Y']= properties['centroid-1']*original.scale[-2]
        prop['Centroid_Z']= properties['centroid-0']*original.scale[-3]

        prop['Weighted_Centroid_X']= properties['centroid_weighted-2']*original.scale[-1]
        prop['Weighted_Centroid_Y']= properties['centroid_weighted-1']*original.scale[-2]
        prop['Weighted_Centroid_Z']= properties['centroid_weighted-0']*original.scale[-3]

    prop_df=pd.DataFrame(prop)
    #prop_df=dt.Frame(prop) #datatable instead of pandas
    prop_df.to_csv(str(save_to)+'\Results.csv')

    log=Label(name='Log', tooltip=None)
    log.value="-> GB: sigma="+str(gaussian_blur_one_pop.sigma.value)+"-> Th="+str(threshold_one_pop.threshold.value)+"-> DistMap"
    #log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_one_pop.min_dist.value) + " -> Found n="+str(prop_df.nrows)+ " objects" #if using datatable
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_one_pop.min_dist.value) + " -> Found n="+str(len(prop_df))+ " objects"
    #measure_one_pop.insert(4,log)


    if save_log == True:
        Log_file = open(str(save_to)+'\Log_ZELDA_single_population.txt','w')
        Log_file.write(log.value)
        Log_file.close()

@magicgui(label={'widget_type':'Label', 'label':"Relate Parent-to-Child and Measure"}, call_button="Relate and Measure",
          save_to_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def relate_and_measure(viewer: 'napari.Viewer', label, Parents_labels: Image, Children_labels: Image, Original_to_measure: Image, save_to_path):
    properties=measure.regionprops_table(Children_labels.data, Original_to_measure.data, properties= ['label','area','intensity_mean','intensity_min','intensity_max','equivalent_diameter','axis_major_length','axis_minor_length','centroid','centroid_weighted','extent','solidity']
    )
    binary_ch=Children_labels.data>0
    corresponding_parents=Parents_labels.data*binary_ch
    viewer.add_image(corresponding_parents, scale=Parents_labels.scale, rgb=False, name='Labelled children objects by parent', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')
    voxel_size=np.prod(Original_to_measure.scale)
    properties_CorrespondingParent=measure.regionprops_table(Children_labels.data, Parents_labels.data, properties=['intensity_max'])
    prop={'Parent_label': properties_CorrespondingParent['intensity_max'].astype(float), 'Label': properties['label'], 'Area': properties['area']*Original_to_measure.scale[-1]*Original_to_measure.scale[-2],
    'Volume': properties['area']*voxel_size,
    'Equivalent_diameter': properties['equivalent_diameter']*Original_to_measure.scale[-1],'MFI': properties['intensity_mean'],'Min_Intensity': properties['intensity_min'], 'Max_Intensity': properties['intensity_max'],
    'MajorAxis_Length': properties['axis_major_length']*Original_to_measure.scale[-1],
    'MinorAxis_Length': properties['axis_minor_length']*Original_to_measure.scale[-1],
    'Extent': properties['extent'],
    'Solidity': properties['solidity']
    }
    if  len(Original_to_measure.data.shape)==2:
        prop['Centroid_X']= properties['centroid-1']*Original_to_measure.scale[-1]
        prop['Centroid_Y']= properties['centroid-0']*Original_to_measure.scale[-2]

        prop['Weighted_Centroid_X']= properties['centroid_weighted-1']*Original_to_measure.scale[-1]
        prop['Weighted_Centroid_Y']= properties['centroid_weighted-0']*Original_to_measure.scale[-2]

        additional_properties_2D_twoPop=measure.regionprops_table(Children_labels.data, Original_to_measure.data, properties= ['orientation','perimeter'])
        prop['Orientation']= additional_properties_2D_twoPop['orientation']
        prop['Perimeter']= additional_properties_2D_twoPop['perimeter']*Original_to_measure.scale[-1]

    if len(Original_to_measure.data.shape)==3:
        prop['Centroid_X']= properties['centroid-2']*Original_to_measure.scale[-1]
        prop['Centroid_Y']= properties['centroid-1']*Original_to_measure.scale[-2]
        prop['Centroid_Z']= properties['centroid-0']*Original_to_measure.scale[-3]

        prop['Weighted_Centroid_X']= properties['centroid_weighted-2']*Original_to_measure.scale[-1]
        prop['Weighted_Centroid_Y']= properties['centroid_weighted-1']*Original_to_measure.scale[-2]
        prop['Weighted_Centroid_Z']= properties['centroid_weighted-0']*Original_to_measure.scale[-3]

    prop_df=pd.DataFrame(prop)
    #prop_df=dt.Frame(prop) #datatable instead of pandas
    prop_df.to_csv(str(save_to_path)+'\Results_parents-children.csv')

    log=Label(name='Log', tooltip=None)
    log.value="-> GB: sigma="+str(gaussian_blur_parent_pop.sigma.value)+"-> Th_parents="+str(threshold_parents.threshold.value)+"-> DistMap"
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_parent_pop.min_dist.value) + " -> Found n="+str( np.max(prop_df['Parent_label'].to_numpy()) )+ " objects"
    log.value=log.value+"\n-> GB: sigma="+str(gaussian_blur_children_pop.sigma.value)+"-> Th_children="+str(threshold_children.threshold.value)+"-> DistMap"
    #log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_children_pop.min_dist.value) + " -> Found n="+str(prop_df.nrows)+ " objects" #if using datatable
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_children_pop.min_dist.value) + " -> Found n="+str(len(prop_df))+ " objects"
    measure_one_pop.insert(4,log)

    #if save_log == True:
    Log_file = open(r''+str(save_to_path)+'\Log_ZELDA_Parents_Children.txt','w')
    Log_file.write(log.value)
    Log_file.close()

@magicgui(label={'widget_type':'Label', 'label':"Plot results and save graphs"}, layout="vertical",
          table_path={'widget_type': 'FileEdit', 'value':'Documents\properties.csv', 'mode':'r','filter':'*.csv'},
          plot_h={'widget_type':'CheckBox','name':'Histogram','text':'Histogram'},
          plot_s={'widget_type':'CheckBox','name':'Scatterplot','text':'Scatterplot'},
          save_plots={'widget_type':'CheckBox','name':'Save_plots','text':'Save plots'},
          saveTo_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          histogram={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter','Min_Intensity','Max_Intensity','MajorAxis_Length','MinorAxis_Length','Parent_label','Weighted_Centroid_X','Weighted_Centroid_Y','Weighted_Centroid_Z','Centroid_X','Centroid_Y','Centroid_Z','Extent','Solidity','Orientation','Perimeter')},
          scatterplot_X={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter','Min_Intensity','Max_Intensity','MajorAxis_Length','MinorAxis_Length','Parent_label','Weighted_Centroid_X','Weighted_Centroid_Y','Weighted_Centroid_Z','Centroid_X','Centroid_Y','Centroid_Z','Extent','Solidity','Orientation','Perimeter')},
          scatterplot_Y={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter','Min_Intensity','Max_Intensity','MajorAxis_Length','MinorAxis_Length','Parent_label','Weighted_Centroid_X','Weighted_Centroid_Y','Weighted_Centroid_Z','Centroid_X','Centroid_Y','Centroid_Z','Extent','Solidity','Orientation','Perimeter')},
          persist=True,
          call_button="Plot",
          result_widget=False
          )
def results_widget(viewer: 'napari.Viewer',
                   label,
                   table_path,
                   plot_h,
                   plot_s,
                   save_plots,
                   saveTo_path,
                   histogram: str='Area',
                   scatterplot_X: str='Area',
                   scatterplot_Y: str='MFI'
                   ):
    table=pd.read_csv(table_path) #if using pandas

#    table = dt.fread(table_path) #if using datatable
    if plot_h== True:
        hist_data=table[str(histogram)][table[str(histogram)]!=np.inf][table[str(histogram)]!=np.NINF].dropna().to_numpy() #if using pandas
        plot_widget_histogram = FigureCanvas(Figure(figsize=(2, 1.5), dpi=150))
        ax = plot_widget_histogram.figure.subplots()
#        ax.set(xlim=(0, 5*np.median(table[str(histogram)])), ylim=(0, table.nrows)) #if using datatable
        ax.set(xlim=(0, 5*np.median(hist_data)), ylim=(0, len(hist_data)))
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
        scatter_data_x=table[str(scatterplot_X)][table[str(scatterplot_X)]!=np.inf][table[str(scatterplot_X)]!=np.NINF][table[str(scatterplot_Y)]!=np.inf][table[str(scatterplot_Y)]!=np.NINF].dropna().to_numpy() #if using pandas
        scatter_data_y=table[str(scatterplot_Y)][table[str(scatterplot_Y)]!=np.inf][table[str(scatterplot_Y)]!=np.NINF][table[str(scatterplot_X)]!=np.inf][table[str(scatterplot_X)]!=np.NINF].dropna().to_numpy() #if using pandas
        plot_widget_scattering = FigureCanvas(Figure(figsize=(2, 1.5), dpi=150))
        ax = plot_widget_scattering.figure.subplots()
#        ax.set(xlim=(0, 5*np.median(table[str(scatterplot_X)])), ylim=(0, 5*np.median(table[str(scatterplot_Y)]))) #if using datatable
        ax.set(xlim=(0, 5*np.median(scatter_data_x)), ylim=(0, 5*np.median(scatter_data_y)))
        ax.set_title('Scatterplot of '+str(scatterplot_X)+' vs '+str(scatterplot_Y), color='gray')
        ax.set_xlabel(str(scatterplot_X))
        ax.set_ylabel(str(scatterplot_Y))
        ax.scatter(x=scatter_data_x, y=scatter_data_y, color='blue')
        ax.tick_params(labelright=False, right=False,labeltop=False,top=False,colors='black')
        plot_widget_scattering.figure.set_tight_layout('tight')
        viewer.window.add_dock_widget(plot_widget_scattering,name='Plot results',area='bottom')
        if save_plots== True:
            plot_widget_scattering.print_tiff(str(saveTo_path)+'\Scatterplot of '+str(scatterplot_X)+' vs '+str(scatterplot_Y)+'.tiff')

@magicgui(
         xy={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001, 'label':'pixel size (um)'},
         z={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001 , 'label':'z (um)'},
         label={'widget_type':'Label', 'label':"Image calibration"},
         call_button="Apply"
         )
def image_calibration(viewer: 'napari.Viewer', label, layer: Image, xy: float = 1.0000, z: float = 1.0000)-> napari.types.ImageData:
    if layer:
        scale=[z, xy, xy]
        layer.scale=scale[-layer.ndim:]

@magicgui(
         xy={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001, 'label':'pixel size (um)'},
         z={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001 , 'label':'z (um)'},
         label={'widget_type':'Label', 'label':"Image calibration parents"},
         call_button="Apply"
         )
def image_calibration_parents(viewer: 'napari.Viewer', label, layer: Image, xy: float = 1.0000, z: float = 1.0000)-> napari.types.ImageData:
    if layer:
        scale=[z, xy, xy]
        layer.scale=scale[-layer.ndim:]

@magicgui(
         xy={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001, 'label':'pixel size (um)'},
         z={'widget_type': 'FloatSpinBox', "max": 1000000.0, 'min':0.0, 'step':0.0001 , 'label':'z (um)'},
         label={'widget_type':'Label', 'label':"Image calibration children"},
         call_button="Apply"
         )
def image_calibration_children(viewer: 'napari.Viewer', label, layer: Image, xy: float = 1.0000, z: float = 1.0000)-> napari.types.ImageData:
    if layer:
        scale=[z, xy, xy]
        layer.scale=scale[-layer.ndim:]


@magicgui(label={'widget_type':'Label', 'label':"Morphological Operations"},
          Operation={'widget_type':'ComboBox', 'label':"Morphological Operation", 'choices':['Erosion','Dilation','Opening','Closing']},
          call_button="Process",
          persist=True)
def morphological_operations(viewer: 'napari.Viewer', label, Operation, Original: Image, element_size: int=1)-> napari.types.ImageData:
    if Original:
        if  len(Original.data.shape)==2:
            selem = skimage.morphology.disk(element_size)
        elif  len(Original.data.shape)==3:
            selem = skimage.morphology.cube(element_size)
        if Operation == 'Erosion':
            morph_processed = skimage.morphology.erosion(Original.data, selem)
        elif Operation == 'Dilation':
            morph_processed = skimage.morphology.dilation(Original.data, selem)
        elif Operation == 'Opening':
            morph_processed = skimage.morphology.opening(Original.data, selem)
        elif Operation == 'Closing':
            morph_processed = skimage.morphology.closing(Original.data, selem)
        viewer.add_image(morph_processed, scale=Original.scale, rgb=False, name=''+str(Original.name)+'_'+str(Operation)+' of '+str(element_size)+'', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')


@magicgui(label={'widget_type':'Label', 'label':"Import/Export Protocols"}, layout="vertical",
          Import_protocols_from={'widget_type': 'FileEdit', 'value':str(os.path.join(prot_path,'napari_zelda','protocols_dict.json')), 'mode':'r','filter':'*.json'},
          Export_protocols_to={'widget_type': 'FileEdit', 'value':'Documents\ZELDA\exported_protocols_dict.json', 'mode':'w', 'filter':'*.json'},
          persist=True,
          call_button="Import list",
          result_widget=False
          )
def protocol_exchange_widget(viewer: 'napari.Viewer', label, Import_protocols_from, Export_protocols_to):
    existing_protocols_file=open(os.path.join(prot_path,'napari_zelda','protocols_dict.json'), "rb")
    existing_protocols_json = json.load(existing_protocols_file)

    existing_protocols=list()
    for i in range(0,len(existing_protocols_json['Protocols'])):
        existing_protocols.append(existing_protocols_json['Protocols'][i]['name'])
    existing_protocols_file.seek(0)
    existing_protocols_file.close()
    ProtocolList=Select(label='Selected_protocols', choices=existing_protocols)

    SaveProtFile=PushButton(name='Append Protocol to File', annotation=None, label=None, tooltip='Save the selected Protocol in the new file', visible=True, enabled=True, gui_only=False, text='Save Protocols', value=0)

    Log=Label(value='', visible=False)

    ExpProt_container = Container()
    ExpProt_container.show()

    ExpProt_container.insert(0, ProtocolList)
    ExpProt_container.insert(1, SaveProtFile)
    ExpProt_container.insert(2, Log)


    protocol_exchange_widget.insert(4, ExpProt_container)

    SaveProtFile.changed.connect(save_protocols_to_file)


def save_protocols_to_file(self):
        imported_protocols_file=open(os.path.abspath(str(protocol_exchange_widget.Import_protocols_from.value)), "rb")
        imported_protocols_json = json.load(imported_protocols_file)
        imported_protocols_file.seek(0)
        imported_protocols_file.close()


        existing_protocols_file=open(os.path.join(prot_path,'napari_zelda','protocols_dict.json'), "rb")
        export_protocols_json = json.load(existing_protocols_file)
        existing_protocols_file.seek(0)
        existing_protocols_file.close()

        if os.path.exists(protocol_exchange_widget.Export_protocols_to.value) == False:
            export_protocols_file=open(os.path.abspath(protocol_exchange_widget.Export_protocols_to.value), "w+")

            for i in range(0,len(export_protocols_json['Protocols'])-1):
                del export_protocols_json['Protocols'][0]
        elif os.path.exists(protocol_exchange_widget.Export_protocols_to.value) == True:
            export_protocols_file=open(os.path.abspath(protocol_exchange_widget.Export_protocols_to.value), "r+")

        for i in range(0,len(imported_protocols_json['Protocols'])):
            new_json_entry=imported_protocols_json['Protocols'][i]
            export_protocols_json["Protocols"].append(new_json_entry)
        del export_protocols_json['Protocols'][0]

        json.dump(export_protocols_json, export_protocols_file, indent = 4)
        export_protocols_file.seek(0)
        export_protocols_file.close()

        #protocol_exchange_widget.Log.visible=True
        #protocol_exchange_widget.Log.value = 'Protocols exported'


@magic_factory(
               auto_call=False,
               call_button=True,
               dropdown={"choices": protocols},
               textbox={'widget_type': 'TextEdit', 'value': protocols_description, 'label':'ZELDA'},
               labels=False
                )
def launch_ZELDA(
        viewer: 'napari.Viewer',
        textbox,
        #protocols: list= protocols,
        dropdown: str= 'Segment a single population'
        ):

        try:
            SETTINGS.application.save_window_geometry = "False"
        except:
            pass

        minusculeWidget_maxWidth=100
        smallWidget_maxWidth=120
        mediumWidget_maxWidth=150
        bigWidget_maxWidth=220
        hugeWidget_maxWidth=290

        widgetHeight_small=125
        widgetHeight_big=265

        image_calibration.native.setMaximumWidth(bigWidget_maxWidth)
        gaussian_blur_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        threshold_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        distance_map_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_one_pop.native.setMaximumWidth(bigWidget_maxWidth)
        watershed_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        measure_one_pop.native.setMaximumWidth(bigWidget_maxWidth)
        results_widget.native.setMaximumWidth(hugeWidget_maxWidth)

        image_calibration.native.setMaximumHeight(widgetHeight_big)
        gaussian_blur_one_pop.native.setMaximumHeight(widgetHeight_big)
        threshold_one_pop.native.setMaximumHeight(widgetHeight_big)
        distance_map_one_pop.native.setMaximumHeight(widgetHeight_big)
        show_seeds_one_pop.native.setMaximumHeight(widgetHeight_big)
        watershed_one_pop.native.setMaximumHeight(widgetHeight_big)
        measure_one_pop.native.setMaximumHeight(widgetHeight_big)
        results_widget.native.setMaximumHeight(widgetHeight_big)

        image_calibration_parents.native.setMaximumWidth(bigWidget_maxWidth)
        gaussian_blur_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        threshold_parents.native.setMaximumWidth(mediumWidget_maxWidth)
        distance_map_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_parent_pop.native.setMaximumWidth(hugeWidget_maxWidth)
        watershed_parent_pop.native.setMaximumWidth(bigWidget_maxWidth)
        image_calibration_children.native.setMaximumWidth(bigWidget_maxWidth)
        gaussian_blur_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        threshold_children.native.setMaximumWidth(mediumWidget_maxWidth)
        distance_map_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_children_pop.native.setMaximumWidth(hugeWidget_maxWidth)
        watershed_children_pop.native.setMaximumWidth(bigWidget_maxWidth)
        relate_and_measure.native.setMaximumWidth(hugeWidget_maxWidth)

        image_calibration_parents.native.setMaximumHeight(widgetHeight_small)
        gaussian_blur_parent_pop.native.setMaximumHeight(widgetHeight_small)
        threshold_parents.native.setMaximumHeight(widgetHeight_small)
        distance_map_parent_pop.native.setMaximumHeight(widgetHeight_small)
        show_seeds_parent_pop.native.setMaximumHeight(widgetHeight_small)
        watershed_parent_pop.native.setMaximumHeight(widgetHeight_small)
        image_calibration_children.native.setMaximumHeight(widgetHeight_small)
        gaussian_blur_children_pop.native.setMaximumHeight(widgetHeight_small)
        threshold_children.native.setMaximumHeight(widgetHeight_small)
        distance_map_children_pop.native.setMaximumHeight(widgetHeight_small)
        show_seeds_children_pop.native.setMaximumHeight(widgetHeight_small)
        watershed_children_pop.native.setMaximumHeight(widgetHeight_small)
        relate_and_measure.native.setMaximumHeight(widgetHeight_big)

        dock_widgets=MainWindow(name='ZELDA protocol', annotation=None, label=None, tooltip=None, visible=True,
                               enabled=True, gui_only=False, backend_kwargs={}, layout='horizontal', widgets=(), labels=True)
        viewer.window.add_dock_widget(dock_widgets, name=str(dropdown), area='bottom')
        if dropdown == 'Segment a single population':
            single_pop_protocol=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            single_pop_protocol.insert(0, image_calibration)
            single_pop_protocol.insert(1, gaussian_blur_one_pop)
            single_pop_protocol.insert(2, threshold_one_pop)
            single_pop_protocol.insert(3, distance_map_one_pop)
            single_pop_protocol.insert(4, show_seeds_one_pop)
            single_pop_protocol.insert(5, watershed_one_pop)
            single_pop_protocol.insert(6, measure_one_pop)
            single_pop_protocol.insert(7, results_widget)

            dock_widgets.insert(0,single_pop_protocol)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Segment two populations and relate':
            parent_pop_protocol=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='horizontal', labels=False)
            parent_pop_protocol.insert(0, image_calibration_parents)
            parent_pop_protocol.insert(1, gaussian_blur_parent_pop)
            parent_pop_protocol.insert(2, threshold_parents)
            parent_pop_protocol.insert(3, distance_map_parent_pop)
            parent_pop_protocol.insert(4, show_seeds_parent_pop)
            parent_pop_protocol.insert(5, watershed_parent_pop)

            children_pop_protocol=Container(name=' ', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            children_pop_protocol.insert(0, image_calibration_children)
            children_pop_protocol.insert(1, gaussian_blur_children_pop)
            children_pop_protocol.insert(2, threshold_children)
            children_pop_protocol.insert(3, distance_map_children_pop)
            children_pop_protocol.insert(4, show_seeds_children_pop)
            children_pop_protocol.insert(5, watershed_children_pop)

            parent_children_container=Container(name='Segmentation', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='vertical', labels=False)
            parent_children_container.insert(0,parent_pop_protocol)
            parent_children_container.insert(1,children_pop_protocol)

            relate_and_measure_container=Container(name='  ', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='vertical', labels=False)
            relate_and_measure_container.insert(0,relate_and_measure)


            dock_widgets.insert(0, parent_children_container)
            dock_widgets.insert(1, relate_and_measure_container)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Data Plotter':
            data_plotter_protocol=Container(name='Results plotter', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            data_plotter_protocol.insert(0,results_widget)
            dock_widgets.insert(0,data_plotter_protocol)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Design a New Protocol':
            new_protocol=Container(name='New Protocol', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            new_protocol.insert(0, new_protocol_widget)
            dock_widgets.insert(0, new_protocol)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Import and Export Protocols':
            new_protocol=Container(name='Import and Export Protocols', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            new_protocol.insert(0, protocol_exchange_widget)
            dock_widgets.insert(0, new_protocol)
            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if (protocols.index(dropdown)>4):
            custom_panel=Container(name='Custom Protocol: "'+dropdown+'"', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='horizontal', labels=False)

            steps_types = ['Threshold', 'GaussianBlur', 'DistanceMap', 'ShowSeeds', 'Watershed', 'Measure', 'Plot','Image Calibration','Morphological Operations']
            available_protocols=len(protocols)
            choosen_protocol=protocols.index(dropdown)

            for k in range(0, len(protocols_json['Protocols'][choosen_protocol]['steps'])):
                step_toAdd=corresponding_widgets[protocols_json['Protocols'][choosen_protocol]['steps'][k]['step_name']]
                custom_panel.insert(k, globals() [step_toAdd])

            dock_widgets.insert(0,custom_panel)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'
        viewer.add_image(skimage.data.astronaut(), name='napari-zelda-astronaut-test', rgb=True)
        viewer.layers.remove('napari-zelda-astronaut-test')

#add Custom protocols
@magicgui(layout="vertical",
          np_name={'widget_type': 'LineEdit', 'value':'MyNewProtocol','tooltip':'Name of the new protocol', 'label':'Name'},
          np_steps={'widget_type': 'SpinBox', 'value':3, 'max':30, 'label': 'Steps:'},
          Log={'widget_type': 'Label', 'value':'', 'visible': False},
          persist=True,
          call_button="Design"
          )
def new_protocol_widget(viewer: 'napari.Viewer',
                   np_name,
                   np_steps,
                   Log
                   ):
                   steps_types = ['Threshold', 'GaussianBlur', 'DistanceMap', 'ShowSeeds', 'Watershed', 'Measure', 'Plot','Image Calibration','Morphological Operations','Filter by Area', 'Median Filter']
                   np_container=Container()
                   for k in range(0, np_steps):
                       np_container.insert(k, ComboBox(choices=steps_types, value=steps_types[0], label='Select step '+str(k+1)+':', name='step_'+str(k)+'', tooltip='Choose a function for this step of the custom protocol'))

                   save_button=PushButton(name='Save Protocol', annotation=None, label=None, tooltip='Save current Protocol', visible=True, enabled=True, gui_only=False, text='Save', value=0)
                   save_button.changed.connect(save_protocol)

                   np_container.insert(np_steps,save_button)
                   np_container.show()
                   new_protocol_widget.insert(3, np_container)
                   new_protocol_widget.call_button.visible=False
                   new_protocol_widget.np_steps.visible=False


def save_protocol(self):
        np_container = new_protocol_widget[3]
        line=new_protocol_widget.np_name.value+'\n'
        protocols_history=open(os.path.join(prot_path,'napari_zelda','protocols_history.txt'),'a')
        protocols_history.write(line)
        protocols_history.close()
        #add to json
        listed_steps={}
        np_json_entry ={"name": new_protocol_widget.np_name.value,
                        "widget": str(new_protocol_widget.np_name.value)+'_protocol_widget',
                        "steps": listed_steps
                        }
        np_json_entry["steps"]=[{ "step_number": j+1, "step_name": str(np_container[j].value) } for j in range(0, (new_protocol_widget.np_steps.value))]

        protocols_file=open(os.path.join(prot_path,'napari_zelda','protocols_dict.json'), "r+")
        protocols_json = json.load(protocols_file)
        protocols_json["Protocols"].append(np_json_entry)
        protocols_file.seek(0)
        json.dump(protocols_json, protocols_file, indent = 4)
        new_protocol_widget.Log.value = '"'+new_protocol_widget.np_name.value+'" saved to the database'
        new_protocol_widget.Log.visible=True
        protocols_file.close()

@magicgui(label={'widget_type':'Label', 'label':"Filter labelled object by area"},
          layout='vertical',
          table_path={'widget_type': 'FileEdit', 'value':'Documents\properties.csv', 'mode':'r','filter':'*.csv'},
          Area_Min={"widget_type": "Slider", "min": 1, "max": 1000000000},
          Area_Max={"widget_type": "Slider", "min": 10, "max": 1000000000},
          persist=True,
          call_button="Apply filter",
          result_widget=False
          )
def filterByArea_widget(viewer: 'napari.Viewer',
                   label,
                   table_path,
                   layer: Image,
                   Area_Min: int = 10,
                   Area_Max: int = 10000
                   )-> napari.types.ImageData:
    table=pd.read_csv(table_path)
    #table = dt.fread(table_path)
    if layer is not None:
        filteredAreaValues = np.array(table['Label'][table['Area']>Area_Min][table['Area']<Area_Max])
        mask=np.isin(layer.data, filteredAreaValues)
        viewer.add_image(layer.data*np.array(mask), scale=layer.scale, name='FilteredByArea_'+str(Area_Min)+'-'+str(Area_Max), opacity=0.6, blending='opaque', colormap='inferno')

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':"Median Filter"},
         element_size={'widget_type': 'IntSlider', "max": 15, 'min':0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def median_filter(viewer: 'napari.Viewer', label, layer: Image, element_size: int = 1, mode="nearest")-> napari.types.ImageData:
    if layer:
        if  len(layer.data.shape)==2:
            selem = skimage.morphology.disk(element_size)
        elif  len(layer.data.shape)==3:
            selem = skimage.morphology.cube(element_size)
        median=skimage.filters.median(layer.data, footprint=selem, mode=mode)
        viewer.add_image(median, scale=layer.scale, name='Median radius='+str(element_size)+' of '+str(layer.name))

### Add here new functionalities for ZELDA ###
### @magicgui(layout="vertical")
### def new_functionality_widget(viewer: 'napari.Viewer'):
###                   ...
###

### End ###

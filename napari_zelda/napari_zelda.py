"""
ZELDA: a 3D Image Segmentation and Parent-Child relation plugin for microscopy image analysis in napari
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QGridLayout, QGroupBox
from napari.layers import Image, Labels, Layer, Points
from magicgui import magicgui, magic_factory
import napari
from napari import Viewer
from napari.settings import SETTINGS
from magicgui.widgets import SpinBox, FileEdit, Slider, FloatSlider, Label, Container, MainWindow, ComboBox, TextEdit, PushButton, ProgressBar
import skimage.filters
from skimage.feature import peak_local_max
from skimage.transform import rotate
from skimage.segmentation import watershed
from skimage import measure
import numpy as np
import pandas as pd
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

protocols_file=open(prot_path+'\\napari_zelda'+'\protocols_dict.json', "rb")

protocols_json = json.load(protocols_file)
protocols=list()
for i in range(0,len(protocols_json['Protocols'])):
    protocols.append(protocols_json['Protocols'][i]['name'])

protocols_file.seek(0)

corresponding_widgets={
                        "Threshold": "threshold_one_pop",
                        "GaussianBlur": "gaussian_blur_one_pop",
                        "DistanceMap": "distance_map_one_pop",
                        "ShowSeeds":"show_seeds_one_pop",
                        "Watershed":"watershed_one_pop",
                        "Measure": "measure_one_pop",
                        "Plot": "results_widget",
                        }

protocols_description=open(prot_path+'\\napari_zelda'+'\protocols_description.txt', 'r').read()

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Threshold"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True
         )
def threshold_one_pop(viewer: 'napari.Viewer', label, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))


@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Threshold to Parents"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_parents(viewer: 'napari.Viewer', label, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Threshold to Children"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True)
def threshold_children(viewer: 'napari.Viewer', label, layer: Image, threshold: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, name='Threshold th='+str(threshold)+' of '+str(layer.name))


@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Gaussian Blur"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_one_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    self.native.setMaximumWidth(50)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Gaussian Blur to Parents"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_parent_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    self.native.setMaximumWidth(50)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False,
         label={'widget_type':'Label', 'value':" apply Gaussian Blur to Children"},
         sigma={'widget_type': 'FloatSlider', "max": 10.0, 'min':0.0},
         mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
         call_button="Apply",
         persist=True)
def gaussian_blur_children_pop(viewer: 'napari.Viewer', label, layer: Image, sigma: float = 1.0, mode="nearest")-> napari.types.ImageData:
    self.native.setMaximumWidth(50)
    if layer:
        gb=skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode, preserve_range=True)
        viewer.add_image(gb, name='GaussianBlur sigma='+str(sigma)+' of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Get Distance Map of"}, call_button="Get DistanceMap", persist=True)
def distance_map_one_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Get Distance Map of Parents"}, call_button="Get DistanceMap", persist=True)
def distance_map_parent_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(labels=False, label={'widget_type':'Label', 'value':"Get Distance Map of Children"}, call_button="Get DistanceMap", persist=True)
def distance_map_children_pop(viewer: 'napari.Viewer', label, layer: Image)-> napari.types.ImageData:
    if layer:
        img=layer.data*255
        dist_map=ndimage.distance_transform_edt(img)
        viewer.add_image(dist_map, name='DistMap of '+str(layer.name))

@magicgui(label={'widget_type':'Label', 'label':"Show seeds"}, call_button="Show seeds", persist=True)
def show_seeds_one_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Show seeds for Parents"}, call_button="Show seeds", persist=True)
def show_seeds_parent_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Show seeds for Children"}, call_button="Show seeds", persist=True)
def show_seeds_children_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, mask: Image, min_dist: int=1)-> napari.types.ImageData:
    if DistanceMap:
        coords = skimage.feature.peak_local_max(DistanceMap.data, labels=mask.data, min_distance=min_dist)
        points = np.array(coords)
        viewer.add_points(points, name='Maxima at dist_min='+str(min_dist)+' of '+str(DistanceMap.name), size=3)

@magicgui(label={'widget_type':'Label', 'label':"Segment with Watershed"}, call_button="Watershed", persist=True)
def watershed_one_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Segment Parents"}, call_button="Watershed", persist=True)
def watershed_parent_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled Parent objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Segment Children"}, call_button="Watershed", persist=True)
def watershed_children_pop(viewer: 'napari.Viewer', label, DistanceMap: Image, binary: Image, seeds: Points)-> napari.types.ImageData:
    if DistanceMap:
        mask = np.zeros(DistanceMap.data.shape, dtype=bool)
        mask[tuple(seeds.data.T)] = True
        markers, _ = ndimage.label(mask)
        labels = skimage.segmentation.watershed(-DistanceMap.data, markers, mask=binary.data)
        viewer.add_image(labels, rgb=False, name='Labelled Children objects', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

@magicgui(label={'widget_type':'Label', 'label':"Measure segmented objects"}, call_button="Measure objects",
          save_log={'widget_type':'CheckBox','name':'Save_Log','text':'Save Log'},
          save_to_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def measure_one_pop( label, labels: Image, original: Image, save_log, save_to_path):
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
        Log_file = open(str(save_to_path)+'\Log_ZELDA_single_population.txt','w')
        Log_file.write(log.value)
        Log_file.close()

@magicgui(label={'widget_type':'Label', 'label':"Relate Parent-to-Child and Measure"}, call_button="Relate and Measure",
          save_to_path={'widget_type': 'FileEdit', 'value':'\Documents', 'mode':'d','tooltip':'Save results to this folder path'},
          persist=True
            )
def relate_and_measure(viewer: 'napari.Viewer', label, Parents_labels: Image, Children_labels: Image, Original_to_measure: Image, save_to_path):
    properties=measure.regionprops_table(Children_labels.data, Original_to_measure.data,
               properties= ['label','area', 'mean_intensity','equivalent_diameter'])
    binary_ch=Children_labels.data>0
    corresponding_parents=Parents_labels.data*binary_ch
    viewer.add_image(corresponding_parents, rgb=False, name='Labelled children objects by parent', opacity=0.6, rendering='mip', blending='additive', colormap='inferno')

    properties_CorrespondingParent=measure.regionprops_table(Children_labels.data, Parents_labels.data, properties=['max_intensity'])
    prop={'Parent_label': properties_CorrespondingParent['max_intensity'],'Area': properties['area'],'Equivalent_diameter': properties['equivalent_diameter'],'MFI': properties['mean_intensity']}
    prop_df=pd.DataFrame(prop)
    prop_df.to_csv(str(save_to_path)+'\Results_parents-children.csv')

    log=Label(name='Log:', tooltip=None,)
    log.value="-> Th_parents="+str(threshold_parents.threshold.value)+"-> GB: sigma="+str(gaussian_blur_parent_pop.sigma.value)+"-> DistMap"
    log.value=log.value+"-> Maxima: min_dist=" + str(show_seeds_parent_pop.min_dist.value) + " -> Found n="+str(np.max(prop_df['Parent_label']))+ " objects"
    log.value=log.value+"\n-> Th_children="+str(threshold_children.threshold.value)+"-> GB: sigma="+str(gaussian_blur_children_pop.sigma.value)+"-> DistMap"
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
          histogram={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
          scatterplot_X={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
          scatterplot_Y={'widget_type':'ComboBox','choices':('Area','MFI','Equivalent_diameter')},
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
               dropdown={"choices": protocols},
               textbox={'widget_type': 'TextEdit', 'value': protocols_description, 'label':'ZELDA'},
               labels=False
                )
def launch_ZELDA(
        viewer: 'napari.Viewer',
        textbox,
        protocols: list= protocols,
        dropdown: str= 'Segment a single population'
        ):

        SETTINGS.application.save_window_geometry = "True"
        minusculeWidget_maxWidth=100
        smallWidget_maxWidth=120
        mediumWidget_maxWidth=150
        bigWidget_maxWidth=220
        hugeWidget_maxWidth=290

        gaussian_blur_one_pop.native.setMaximumWidth(minusculeWidget_maxWidth)
        threshold_one_pop.native.setMaximumWidth(minusculeWidget_maxWidth)
        distance_map_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        watershed_one_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        measure_one_pop.native.setMaximumWidth(bigWidget_maxWidth)
        results_widget.native.setMaximumWidth(hugeWidget_maxWidth)

        gaussian_blur_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        threshold_parents.native.setMaximumWidth(smallWidget_maxWidth)
        distance_map_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        watershed_parent_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        gaussian_blur_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        threshold_children.native.setMaximumWidth(smallWidget_maxWidth)
        distance_map_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        show_seeds_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        watershed_children_pop.native.setMaximumWidth(mediumWidget_maxWidth)
        relate_and_measure.native.setMaximumWidth(hugeWidget_maxWidth)

        dock_widgets=MainWindow(name='ZELDA protocol', annotation=None, label=None, tooltip=None, visible=True,
                               enabled=True, gui_only=False, backend_kwargs={}, layout='vertical', widgets=(), labels=True)
        viewer.window.add_dock_widget(dock_widgets, name=str(dropdown))
        if dropdown == 'Segment a single population':
            single_pop_protocol=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            single_pop_protocol.insert(0, gaussian_blur_one_pop)
            single_pop_protocol.insert(1, threshold_one_pop)
            single_pop_protocol.insert(2, distance_map_one_pop)
            single_pop_protocol.insert(3, show_seeds_one_pop)
            single_pop_protocol.insert(4, watershed_one_pop)
            single_pop_protocol.insert(5, measure_one_pop)
            single_pop_protocol.insert(6, results_widget)

            dock_widgets.insert(0,single_pop_protocol)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'

        if dropdown == 'Segment two populations and relate':
            parent_pop_protocol=Container(name='Parent Population', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='horizontal', labels=False)
            parent_pop_protocol.insert(0, gaussian_blur_parent_pop)
            parent_pop_protocol.insert(1, threshold_parents)
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

            viewer.window.add_dock_widget(relate_and_measure, name='ZELDA: Relate and Measure', area='bottom')

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

        if (protocols.index(dropdown)>3):
            custom_panel=Container(name='Custom Protocol: "'+dropdown+'"', annotation=None, label=None, visible=True, enabled=True,
                                         gui_only=False, layout='horizontal', labels=False)

            steps_types = ['Threshold', 'GaussianBlur', 'DistanceMap','Measure','Plot']
            available_protocols=len(protocols)
            choosen_protocol=protocols.index(dropdown)

            for k in range(0, len(protocols_json['Protocols'][choosen_protocol]['steps'])):
                step_toAdd=corresponding_widgets[protocols_json['Protocols'][choosen_protocol]['steps'][k]['step_name']]
                custom_panel.insert(k, globals() [step_toAdd])

            dock_widgets.insert(0,custom_panel)

            launch_ZELDA._call_button.text = 'Restart with the selected Protocol'


#add Custom protocols
@magicgui(layout="vertical",
          np_name={'widget_type': 'LineEdit', 'value':'MyNewProtocol','tooltip':'Name of the new protocol', 'label':'Name'},
          np_steps={'widget_type': 'SpinBox', 'value':3, 'max':10, 'label': 'Steps:'},
          Log={'widget_type': 'Label', 'value':'', 'visible': False},
          persist=True,
          call_button="Design"
          )
def new_protocol_widget(viewer: 'napari.Viewer',
                   np_name,
                   np_steps,
                   Log
                   ):
                   steps_types = ['Threshold', 'GaussianBlur', 'DistanceMap', 'ShowSeeds', 'Watershed', 'Measure', 'Plot']
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
        protocols_history=open(prot_path+'\\napari_zelda'+'\protocols_history.txt','a')

        protocols_history.write(line)
        protocols_history.close()
        #add to json
        listed_steps={}
        np_json_entry ={"name": new_protocol_widget.np_name.value,
                        "widget": str(new_protocol_widget.np_name.value)+'_protocol_widget',
                        "steps": listed_steps
                        }
        np_json_entry["steps"]=[{ "step_number": j+1, "step_name": str(np_container[j].value) } for j in range(0, (new_protocol_widget.np_steps.value))]

        protocols_file=open(prot_path+'\\napari_zelda'+'\protocols_dict.json', "r+")

        protocols_json = json.load(protocols_file)
        protocols_json["Protocols"].append(np_json_entry)
        protocols_file.seek(0)
        json.dump(protocols_json, protocols_file, indent = 4)
        new_protocol_widget.Log.value = '"'+new_protocol_widget.np_name.value+'" saved to the database'
        new_protocol_widget.Log.visible=True


### End ###

### import everthing we need
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
import mahotas as mh
from skimage.measure import label, regionprops, regionprops_table
import natsort  # package for naturally sort name of image sequence
from tqdm import tqdm
import pandas as pd
import cv2
import trackpy as tp # important package for particle tracking
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable


### set functions
#loading images
def load_im(target_dir, file_name):
    im = cv2.imread(os.path.join(target_dir, file_name))
    im = cv2.bitwise_not(im)
    im = cv2.cvtColor(im[:,:], cv2.COLOR_BGR2GRAY)
    
    return im

#watershed segmentation
def watershed_segmentation(im):
    locmax = mh.regmax(im)
    seeds, nr_nuclei = mh.label(locmax)
    T = mh.thresholding.otsu(np.uint8(im))
    dist = mh.distance(np.uint8(im) > T)
    dist = dist.max() - dist
    dist -= dist.min()
    dist = dist/float(dist.ptp()) * 255
    dist = dist.astype(np.uint8)
    nuclei, lines = mh.cwatershed(dist, seeds, return_lines=True)
    
    #parameters extraction-1
    nuclei_without_border = mh.labeled.remove_bordering(nuclei)
    nuclei_new = label(nuclei_without_border)
    
    return nuclei_new, lines

#extract coordinates of centroids
def extract_centroids(nuclei):
    #use scikit image func
    props=regionprops(nuclei)
    centroids_y=[props[i].centroid[0].astype('float32') for i in range(len(props))]
    centroids_x=[props[i].centroid[1].astype('float32') for i in range(len(props))]
    labels=[np.int32(props[i].label) for i in range(len(props))]
    return centroids_y, centroids_x, labels

#extract the cells correctly tracked by the algorithm
def extract_cell_tracking_result(nuclei_list, track_result_filtered, check_number):
    #set 3D storage
    nuclei_track_result = np.zeros((nuclei_list[0].shape[0], nuclei_list[0].shape[1], check_number))
    
    #extract particle num satisfying check number
    frame_num_list=np.unique(track_result_filtered['frame'])
    
    for frame_num in tqdm(frame_num_list):
        #extract information of the certain frame
        track_result_certain_frame=track_result_filtered[track_result_filtered["frame"]==frame_num]
        
        #set target
        target_label=np.asarray(track_result_certain_frame["label"])
        target_particle=np.asarray(track_result_certain_frame["particle"])
        target_nuclei=list(nuclei_list[frame_num].flatten())
        
        #create temporary zero matrix
        zero_map=np.zeros((nuclei_list[0].shape[0]*nuclei_list[0].shape[1]))
        
        #set function
        #replace_func = {label: new for label, new in zip(target_label_sort, new_label)} 
        replace_func = {label: new for label, new in zip(target_label, target_particle)} 

        #replace label values to anatomical values
        replace_result=np.asarray(list(map(replace_func.get, target_nuclei)))
        result_index=np.where(replace_result!=None)[0]

        #project the result to zero map
        zero_map[result_index]=replace_result[result_index]

        #merge result of each frame to 3D storage
        nuclei_track_result[:,:,frame_num]=zero_map.reshape((nuclei_list[0].shape[0], nuclei_list[0].shape[1]))
        
    return nuclei_track_result

#function for visualization
def result_visualization_mod(nuclei_list, track_result_filtered, change_rate_result, num):
    start_point = num-1
    #set 3D storage
    result_map = np.zeros((nuclei_list[:,:,0].shape[0], nuclei_list[:,:,0].shape[1], len(change_rate_result)))
    
    #extract particle num satisfying check number
    frame_num_list=np.unique(track_result_filtered['frame'])[start_point:]
    
    for i in tqdm(range(len(frame_num_list))):
        frame_num = frame_num_list[i]
        #extract information of the certain frame
        track_result_certain_frame=track_result_filtered[track_result_filtered["frame"]==frame_num]        
        #set target
        #target_label=np.asarray(track_result_certain_frame["label"])
        #target_particle=np.asarray(track_result_certain_frame["particle"])
        new_label=np.unique(nuclei_list[:,:, frame_num])[1:]
        target_nuclei=list(nuclei_list[:,:, frame_num].flatten())
        target_change_rate_result=change_rate_result[i]
        
        #set zero map
        zero_map=np.zeros((nuclei_list[:,:,0].shape[0]*nuclei_list[:,:,0].shape[1]))    

        #set function
        replace_func = {new: change_rate for new, change_rate in target_change_rate_result} 

        #replace label values to anatomical values
        replace_result=np.asarray(list(map(replace_func.get, target_nuclei)))
        result_index=np.where(replace_result!=None)[0]

        #project the result to zero map
        zero_map[result_index]=replace_result[result_index]

        #save result
        result_map[:,:,i]=zero_map.reshape((nuclei_list[:,:,0].shape[0], nuclei_list[:,:,0].shape[1]))
        
    return result_map
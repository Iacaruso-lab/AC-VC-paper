import mat73
import os
from scipy.io import loadmat
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy
import importlib
#from skimage import io
# import pims
from matplotlib import cm
import imageio
import seaborn as sns
import statsmodels.stats.multitest
from tqdm import tqdm
from matplotlib import gridspec
#import svglib
from pathlib import Path
import nrrd
import cv2 as cv2
import math

# allen sdk is needed
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache

import ccf_streamlines.projection as ccfproj

#useful
#https://allensdk.readthedocs.io/en/stable/_static/examples/nb/mouse_connectivity.html
from analysis_utils import *

def getProjectionIntensity(final_ids, df,ops,mcc,rsp,structure_tree, Ac_exp):
    
    outputPath = os.path.join(ops['dataPath'], 'allen_connectivity_analysis')

    target_areas = ['VISp','VISpl', 'VISpor','VISli', 'VISl', 'VISal', 'VISrl','VISa', 'VISam', 'VISpm']
    areas = ['V1','P','POR','LI', 'LM', 'AL','RL','A','AM', 'PM'] #for making figures
    mcc = MouseConnectivityCache(manifest_file=Path(outputPath) / 'manifest.json', resolution=25) #25 for analysisw
    rsp = mcc.get_reference_space()

    idx = np.array([np.nonzero(np.array(Ac_exp['id']) == final_ids[j])[0] for j in range(len(final_ids))])
    df = Ac_exp.iloc[np.squeeze(idx)]

    intensity_byArea_all = np.zeros((len(df), len(target_areas)))

    for e in tqdm(range(len(df))):
        exp_ID = df['id'].iloc[e]
        inj_vol = df['injection_volume'].iloc[e]

        exp_data, exp_info = mcc.get_projection_density(exp_ID)

        exp_data_norm = exp_data/inj_vol

        shape_template = exp_data.shape
        x_midpoint = shape_template[2] // 2
        right_mask = np.zeros(shape_template, dtype=bool)
        right_mask[:, :, x_midpoint:] = 1

        left_mask = np.zeros(shape_template, dtype=bool)
        left_mask[:, :, 0:x_midpoint] = 1

        if df['injection_z'].iloc[e] < 5000:
            side_mask = left_mask
        else:
            side_mask = right_mask

        for ar in range(len(target_areas)):
            try:
                structure = structure_tree.get_structures_by_acronym([target_areas[ar]])
            except KeyError:
                print(f'{acronym} does not exist - need to check naming')
                continue
            if 315 in structure[0]['structure_id_path']:
                structure_id = structure[0]['id']
                mask = rsp.make_structure_mask([structure_id], direct_only=False)
                mask_thisRegion = mask*side_mask
                intensity_thisRegion = np.mean(exp_data_norm[np.where(mask_thisRegion)])

                intensity_byArea_all[e,ar] = intensity_thisRegion
                
    return intensity_byArea_all

def plotProjectionStrength(df,intensity_byArea_all, ops):
    areas = ops['areas']

    norm_intensity_all = np.zeros_like(intensity_byArea_all)
    for e in range(len(intensity_byArea_all)):
        intensity_this = intensity_byArea_all[e,:]
        norm_intensity = (intensity_this - min(intensity_this))/(max(intensity_this) - min(intensity_this))
        norm_intensity_all[e,:] = norm_intensity

    thresh = 8000
    anterior, posterior, shapes = [],[],[]
    for e in range(len(intensity_byArea_all)):
        if df['injection_x'].iloc[e] < 8000:
            anterior.append(norm_intensity_all[e,:])
        else:
            posterior.append(norm_intensity_all[e,:])

        if 'Emx' in df['specimen_name'].iloc[e]:
            shape = 's'
        elif'Cux' in df['specimen_name'].iloc[e]:
            shape = 'D'
        elif'Rorb' in df['specimen_name'].iloc[e]:
            shape = '^'
        elif'Rbp' in df['specimen_name'].iloc[e]:
            shape = 'v'
        else:
            shape = 'o'

        shapes.append(shape)        

    anterior = np.array(anterior)
    posterior = np.array(posterior)

    color_anterior = '#0850D4'
    color_posterior = '#DB0F04'

    fig = plt.figure(figsize=(ops['mm']*100,ops['mm']*80), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    avg_anterior = np.mean(anterior,0)
    avg_posterior = np.mean(posterior,0)
    plt.plot(np.arange(0,len(ops['areas'])), avg_anterior, c=color_anterior, linewidth=2, label = 'Anterior')
    plt.plot(np.arange(0,len(ops['areas'])), avg_posterior, c=color_posterior,linewidth=2, label = 'Posterior')
    # plt.scatter(np.arange(0,len(areas)), avg_posterior, c=color_posterior,s=10)

    for i in range(anterior.shape[0]):
        plt.plot(np.arange(0,len(ops['areas'])), anterior[i,:], c=color_anterior, linewidth=0.25)
    for i in range(posterior.shape[0]):
        plt.plot(np.arange(0,len(ops['areas'])), posterior[i,:], c=color_posterior, linewidth=0.25)
    myPlotSettings_splitAxis(fig,ax, 'Normalised projection strength', '', '', mySize=12)
    plt.xticks(np.arange(0,len(areas)), ops['areas'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    plt.yticks([0,0.2, 0.4, 0.6, 0.8,1], ['0','0.2', '0.4', '0.6', '0.8','1'])
    
    
    ###############################################
    location = np.array(df['injection_x'])
    location = abs(location - 8600)
    fig = plt.figure(figsize=(ops['mm']*260, ops['mm']*100), constrained_layout=True)
    for ar in range(len(areas)):
        intensity = norm_intensity_all[:,ar]

        lm = doLinearRegression_withCI(location, intensity)

        ax = fig.add_subplot(2,5,ar+1)
        for i in range(len(location)):   
            plt.scatter(location[i], intensity[i], c= ops['myColorsDict']['HVA_colors'][areas[ar]], marker = shapes[i], alpha = 0.5, s=8, edgecolor='k', linewidth=0.15)       
        plt.fill_between(lm['x_vals'], lm['ci_upper'], lm['ci_lower'], facecolor = ops['myColorsDict']['HVA_colors'][areas[ar]],alpha = 0.2)
        plt.plot(lm['x_vals'], lm['y_vals'],c =ops['myColorsDict']['HVA_colors'][areas[ar]],linewidth = 1)
        plt.text(0, 1.1,'p: ' + str(np.round(lm['pVal_corr'],4)), fontsize=12)
        myPlotSettings_splitAxis(fig,ax, '', '',areas[ar], mySize=15)
        ax.tick_params(axis='y', pad=1)   
        ax.tick_params(axis='x', pad=1)
        plt.ylim([-0.1,1.1])
        plt.xlim([-20,1020])
        plt.xticks([0,500,1000])
        plt.yticks([0,0.5, 1], ['0', '0.5', '1'])
        
        
def plotProjectionIntensity_onCortex(df, ops, final_ids):


    filesPath = os.path.join(ops['dataPath'], 'allen_connectivity_analysis','flatmap_stuff')

    cortex_mask = imageio.volread(os.path.join(filesPath, 'cortex_mask_10um.tiff'))

    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file=os.path.join(filesPath,"flatmap_butterfly.nrrd"),
        labels_file=os.path.join(filesPath,"labelDescription_ITKSNAPColor.txt"),
    )

    # We get the left hemisphere region boundaries with the default arguments
    bf_left_boundaries = bf_boundary_finder.region_boundaries()

    # And we can get the right hemisphere boundaries that match up with
    # our projection if we specify the same configuration
    bf_right_boundaries = bf_boundary_finder.region_boundaries(
        # we want the right hemisphere boundaries, but located in the right place
        # to plot both hemispheres at the same time
        hemisphere='right_for_both',

        # we also want the hemispheres to be adjacent
        view_space_for_other_hemisphere='flatmap_butterfly',
    )

    proj_top = ccfproj.Isocortex2dProjector(
        # Specify our view lookup file
        os.path.join(filesPath, "flatmap_butterfly.h5"),
        # Specify our streamline file
        os.path.join(filesPath, "surface_paths_10_v3.h5"),
        # Specify that we want to project both hemispheres
        hemisphere="both",
        # The top view contains space for the right hemisphere, but is empty.
        # Therefore, we tell the projector to put both hemispheres side-by-side
        view_space_for_other_hemisphere="flatmap_butterfly",
    )
    
    ###############################################################
    mm = ops['mm']

    inj_pos_flatmap = np.zeros((len(final_ids),2))
    for e in range(len(final_ids)):
        exp_ID = final_ids[e]
        path = os.path.join(ops['dataPath'],'allen_connectivity_analysis', 'experiment_' + str(exp_ID))
        injection_projection_max = imageio.imread(os.path.join(path,'InjectionLocation_projected.tiff'))

        roi = np.nonzero(injection_projection_max)
        x_max = np.max(roi[0])
        x_min = np.min(roi[0])
        x = int(x_min + (x_max-x_min)/2)
        y_max = np.max(roi[1])
        y_min = np.min(roi[1])
        y = int(y_min + (y_max-y_min)/2)

        if x > 1176: #i.e. if injection was on the right side
            x =  injection_projection_max.shape[0]-x

        inj_pos_flatmap[e,:] = [x,y]


    height, width = injection_projection_max.shape
    mask = np.empty((width, height)); mask[:] = np.nan
    target_areas = ['VISp','VISpl', 'VISpor','VISli', 'VISl', 'VISal', 'VISrl','VISa', 'VISam', 'VISpm']
    for i in range(len(target_areas)):
        contours = bf_left_boundaries[target_areas[i]].astype(int)
        # mask[contours[:,0], contours[:,1]] = 1
        # that = contours.T
        cv2.fillPoly(mask, [contours], color=1)  # Fill with 1s

    thresh = 8000
    anterior, posterior = [],[]
    inj_pos_ant, inj_pos_post = [],[]
    inj_vol_ant, inj_vol_post = [],[]
    for e in range(len(final_ids)):
        if df['injection_x'].iloc[e] < 8000:
            anterior.append(final_ids[e])
            inj_pos_ant.append(inj_pos_flatmap[e,:])
            inj_vol_ant.append(df['injection_volume'].iloc[e])
        else:
            posterior.append(final_ids[e])
            inj_pos_post.append(inj_pos_flatmap[e,:])
            inj_vol_post.append(df['injection_volume'].iloc[e])

    inj_pos_ant = np.array(inj_pos_ant)
    inj_pos_post = np.array(inj_pos_post)
    anterior = np.array(anterior)
    posterior = np.array(posterior)

    for i in range(len(anterior)):
        exp_ID = anterior[i]
        path = os.path.join(ops['dataPath'],'allen_connectivity_analysis', 'experiment_' + str(exp_ID))
        projection_max = imageio.imread(os.path.join(path,'maxIntProjection.tiff'))

        proj = np.rot90(projection_max,3)

        projection_masked = proj*mask

        proj_mask_norm = (projection_masked - np.nanmin(projection_masked))/(np.nanmax(projection_masked) - np.nanmin(projection_masked))

        if i ==0:
            im_ant = proj_mask_norm
        else:
            im_ant = im_ant + proj_mask_norm

    im_ant =im_ant/len(anterior)

    for i in range(len(posterior)):
        exp_ID = posterior[i]
        path = os.path.join(ops['dataPath'],'allen_connectivity_analysis', 'experiment_' + str(exp_ID))
        projection_max = imageio.imread(os.path.join(path,'maxIntProjection.tiff'))

        proj = np.rot90(projection_max,3)

        projection_masked = proj*mask

        proj_mask_norm = (projection_masked - np.nanmin(projection_masked))/(np.nanmax(projection_masked) - np.nanmin(projection_masked))

        if i ==0:
            im_post = proj_mask_norm
        else:
            im_post = im_post +  proj_mask_norm

    im_post = im_post/len(posterior)

    def sphere_radius(volume):
        return ((3 * volume) / (4 * math.pi)) ** (1/3)

    radius_ant = sphere_radius(np.array(inj_vol_ant))*100 #radius is in mm, and each pixel is 10 um, so this puts in in pixels
    radius_post = sphere_radius(np.array(inj_vol_post))*100 #radius is in mm, and each pixel is 10 um, so this puts in in pixels

    mm = ops['mm']

    plotInjArea = 1
    fig =plt.figure(figsize=(mm*120,mm*80), constrained_layout=True)
    gs = gridspec.GridSpec(1,2, figure=fig, hspace=0.1, wspace=0.05,left=0.16, right=0.95, bottom=0.1, top=0.95)

    ax = fig.add_subplot(gs[0,0])
    plt.imshow(im_ant, cmap = 'PuBu', alpha = 1,vmin=0, vmax=0.6)
    cbar = plt.colorbar(ticks = [0,0.2,0.4,0.6],fraction=0.04, pad=0.04)
    cbar.ax.set_yticklabels(['0','0.2','0.4','0.6'],fontsize=6)
    # plt.colorbar(fraction=0.04, pad =0.02)
    for k, boundary_coords in bf_left_boundaries.items():
        ax.plot(*boundary_coords.T, c="gray", lw=1)
    for k, boundary_coords in bf_right_boundaries.items():
        ax.plot(*boundary_coords.T, c="gray", lw=1)
    if plotInjArea:
        for j in range(len(radius_ant)):
            theta = np.linspace(0, 2*np.pi, 300)  # Angles from 0 to 2π
            x = radius_ant[j] * np.cos(theta) + inj_pos_ant[j,0]
            y = radius_ant[j] * np.sin(theta) + inj_pos_ant[j,1]

            plt.plot(x, y, color='k', linewidth=0.5)
            plt.fill(x, y, color='lightblue', alpha=0.5)  # Fill the circle with color
    else:
        plt.scatter(inj_pos_ant[:,0],inj_pos_ant[:,1], s=50, c='b')
    # plt.gca().invert_yaxis()  # Flip only the display
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    plt.xlim(150, 900)  # Restrict x-axis
    plt.ylim(700, 1300)  # Restrict y-axis (inverted for image coordinates)
    ax.invert_yaxis()  # Flip only the display
    plt.xticks([],[])
    plt.yticks([],[])
    # plt.title('Anterior')

    ax = fig.add_subplot(gs[0,1])
    plt.imshow(im_post, cmap = 'OrRd', alpha = 1,vmin=0, vmax=0.6)
    cbar = plt.colorbar(ticks = [0,0.2,0.4,0.6],fraction=0.04, pad=0.04)
    cbar.ax.set_yticklabels(['0','0.2','0.4','0.6'],fontsize=6)
    for k, boundary_coords in bf_left_boundaries.items():
        ax.plot(*boundary_coords.T, c="gray", lw=1)
    for k, boundary_coords in bf_right_boundaries.items():
        ax.plot(*boundary_coords.T, c="gray", lw=1)

    if plotInjArea:
        for j in range(len(radius_post)):
            theta = np.linspace(0, 2*np.pi, 300)  # Angles from 0 to 2π
            x = radius_post[j] * np.cos(theta) + inj_pos_post[j,0]
            y = radius_post[j] * np.sin(theta) + inj_pos_post[j,1]

            plt.plot(x, y, color='k', linewidth=0.5)
            plt.fill(x, y, color='orange', alpha=0.2)  # Fill the circle with color
    else:  
        plt.scatter(inj_pos_post[:,0],inj_pos_post[:,1], s=50, c='orange')
    # plt.gca().invert_yaxis()  # Flip only the display
    plt.xlim(150, 900)  # Restrict x-axis
    plt.ylim(700, 1300)  # Restrict y-axis (inverted for image coordinates)
    ax.invert_yaxis()  # Flip only the display
    plt.xticks([],[])
    plt.yticks([],[])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    # plt.title('Posterior')

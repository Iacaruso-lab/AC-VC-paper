import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import imageio
from skimage import feature,filters
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.colors as mcolors
import pandas as pd
from tqdm import tqdm

import ccf_streamlines.projection as ccfproj

from analysis_utils import *

def plotInjectionPercentages(animals, regionNames, ops):


    perc_summary = np.zeros((len(animals), len(regionNames)))
    names = []
    for a in range(len(animals)):

        path = os.path.join(ops['dataPath'],'data_byAnimal', 'A'+ str(animals[a]), 'Segmented_injections')

        regions = pd.read_csv(os.path.join(path,'region_0.csv'))

        region_perc = []
        for r in range(len(regionNames)):
            perc = []
            for i in range(len(regions)):
                names.append(regions['structure_name'][i])
                if regionNames[r] in regions['structure_name'][i]:
                    perc.append(regions['percentage_of_total'][i])

            perc = np.sum(np.array(perc))
            region_perc.append(perc)

        perc_summary[a,:] = np.array(region_perc)   
    
    plotNames = ['AUDp', 'AUDv','AUDpo', 'TEa','AUDd', 'Ect','Hipp.CA1','VIS']
    # plotNames = regionNames
    fig = plt.figure(figsize=(ops['mm']*40,ops['mm']*45),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    xVals = np.arange(0,len(regionNames))
    for a in range(len(animals)):
        plt.scatter(xVals, perc_summary[a,:], c= 'k', s= 3)
        plt.plot(xVals, perc_summary[a,:], linewidth = 0.2, c = 'gray')

    meanVal = np.nanmean(perc_summary,0)
    stdVal = np.nanstd(perc_summary,0)

    plt.scatter(xVals, meanVal, c= 'r', s= 10)
    plt.plot(xVals, meanVal, linewidth = 1, c = 'r')
    plt.ylim([-5, 102])
    myPlotSettings_splitAxis(fig, ax, 'Percentage of expression volume (%)', '', '', mySize=6)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    plt.xticks(xVals, plotNames, rotation =45, horizontalalignment='right')

    animalList = np.repeat(animals, len(plotNames))
    regionList = np.tile(plotNames, len(animals))
    perc = []
    for a in range(len(animals)):
        for r in range(len(plotNames)):
            perc.append(perc_summary[a,r])

    injection_summary = {'animals': animalList,
                         'region': regionList,
                         'injection_percentage': np.array(perc)}   

    #injection_df = pd.DataFrame(injection_summary)
    #injection_df.to_csv(os.path.join(outputPath, 'Ac_injection_summary.csv')) 

   # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\segmentedInjections_scatter.svg'))

def plotInjectionlocation(animals,ops):
    #sys.path.insert(0, 'C:\\Users\\egeaa\\Documents\\GitHub\ccf_streamlines\\') #this package needs to be downloaded first. Change paths if necessary

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
        view_space_for_other_hemisphere='flatmap_butterfly')

    proj_top = ccfproj.Isocortex2dProjector(
        # Specify our view lookup file
        os.path.join(filesPath, "flatmap_butterfly.h5"),
        # Specify our streamline file
        os.path.join(filesPath, "surface_paths_10_v3.h5"),
        # Specify that we want to project both hemispheres
        hemisphere="both",
        # The top view contains space for the right hemisphere, but is empty.
        # Therefore, we tell the projector to put both hemispheres side-by-side
        view_space_for_other_hemisphere="flatmap_butterfly")
    
    ######################################################
    fig = plt.figure(figsize=(ops['mm']*100,ops['mm']*100))
    ax = fig.add_subplot(1,1,1)
    blank = np.empty((1360,2352));blank[:] = np.nan
    plt.imshow(blank)
    for k, boundary_coords in bf_left_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)
    for k, boundary_coords in bf_right_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)

    animals = [107, 109, 112, 113,128, 131, 132, 149, 151, 153, 154, 166, 168, 170,171, 178] #frequencies only
    # animals = [112, 128, 131, 132, 149, 151, 153, 154, 166, 168, 170, 171,178] #locations only

    cmap = cm.get_cmap('turbo', len(animals))  # Get 6 evenly spaced colors        
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(animals))]
    for a in range(len(animals)):    

        outputPath = os.path.join(ops['dataPath'],'data_byAnimal','A' + str(animals[a]),'Segmented_injections')

        contours0 = np.load(os.path.join(outputPath,'segmented_injection_contours.npy'))
        first = np.expand_dims(contours0[0,:],0)
        contours = np.append(contours0, first,axis=0)

        # plt.fill(contours[:,0], contours[:,1], color='#BD1828', alpha =0.1)
        # plt.plot(contours[:,0], contours[:,1], c= '#BD1828', linewidth=0.5)
        plt.fill(contours[:,0], contours[:,1], color='gray', alpha =0.1)
        plt.plot(contours[:,0], contours[:,1], c= 'k', linewidth=0.25)
        # plt.fill(contours[:,0], contours[:,1], color=colors[a], alpha =0.5)
        # plt.plot(contours[:,0], contours[:,1], c= colors[a], linewidth=0.5)
        centre_x = int(min(contours0[:,0]) + (max(contours0[:,0]) - min(contours0[:,0]))/2)
        centre_y = int(min(contours0[:,1]) + (max(contours0[:,1]) - min(contours0[:,1]))/2)

        plt.scatter(centre_x, centre_y, c= 'k', s=10)
        # centroid[a,:] = [centre_x, centre_y]

        plt.xlim(100, 540)  # Restrict x-axis
        plt.ylim(650, 1070)  # Restrict y-axis (inverted for image coordinates)
        # plt.title(str(animals[a]))
        plt.gca().invert_yaxis()  # Flip only the display
        for spine in ax.spines.values():
            spine.set_linewidth(1)  # Adjust thickness
        plt.xticks([],[])
        plt.yticks([],[])

    ##########################################################
    centroid = np.zeros((len(animals),2))        
    for a in range(len(animals)):    

        outputPath = os.path.join(ops['dataPath'],'data_byAnimal','A' + str(animals[a]),'Segmented_injections')
        contours0 = np.load(os.path.join(outputPath,'segmented_injection_contours.npy'))

        centre_x = int(min(contours0[:,0]) + (max(contours0[:,0]) - min(contours0[:,0]))/2)
        centre_y = int(min(contours0[:,1]) + (max(contours0[:,1]) - min(contours0[:,1]))/2)

        # plt.scatter(centre_x, centre_y, c= colors[a], label = str(animals[a]))
        centroid[a,:] = [centre_x, centre_y]

    anteriorThresh = 875
    ventralThresh = 330

    ventralAnimals, dorsalAnimals, anteriorAnimals, posteriorAnimals = [], [], [],[]
    for a in range(len(animals)):
        if centroid[a,1] < anteriorThresh:
            anteriorAnimals.append(animals[a])
        else:
            posteriorAnimals.append(animals[a])

        if centroid[a,0] < ventralThresh:
            ventralAnimals.append(animals[a])
        else:
            dorsalAnimals.append(animals[a])

    pos_df = {'anteriorAnimals' : anteriorAnimals,
              'posteriorAnimals' : posteriorAnimals,
              'ventralAnimals' : ventralAnimals,
              'dorsalAnimals' :dorsalAnimals, 
              'allAnimals': animals,
              'centroid_DV': centroid[:,0],
              'centroid_AP': centroid[:,1]}   

    fig = plt.figure(figsize=(ops['mm']*100,ops['mm']*100))
    ax = fig.add_subplot(1,1,1)
    blank = np.empty((1360,2352));blank[:] = np.nan
    plt.imshow(blank)
    for k, boundary_coords in bf_left_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)
    for k, boundary_coords in bf_right_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)

    color_ventral = 'darkorange'
    color_dorsal = 'green'
    for a in range(len(animals)):    

        outputPath = os.path.join(ops['dataPath'],'data_byAnimal','A' + str(animals[a]),'Segmented_injections')

        contours0 = np.load(os.path.join(outputPath,'segmented_injection_contours.npy'))
        first = np.expand_dims(contours0[0,:],0)
        contours = np.append(contours0, first,axis=0)

        # plt.fill(contours[:,0], contours[:,1], color='#BD1828', alpha =0.1)
        # plt.plot(contours[:,0], contours[:,1], c= '#BD1828', linewidth=0.5)
        if animals[a] in ventralAnimals:
            color = color_ventral
        elif animals[a] in dorsalAnimals:
            color = color_dorsal

        plt.fill(contours[:,0], contours[:,1], color=color, alpha =0.2)
        plt.plot(contours[:,0], contours[:,1], c= color, linewidth=0.2)


        plt.xlim(100, 540)  # Restrict x-axis
        plt.ylim(650, 1070)  # Restrict y-axis (inverted for image coordinates)
        # plt.title(str(animals[a]))
        plt.gca().invert_yaxis()  # Flip only the display
        for spine in ax.spines.values():
            spine.set_linewidth(1)  # Adjust thickness
        plt.xticks([],[])
        plt.yticks([],[])

    for a in range(len(animals)):    
        centre_x, centre_y = centroid[a,:]
        plt.scatter(centre_x, centre_y, c= 'k', s=10)

    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\segmentedInjections_onFlatmap_colored_DV.svg'))


    fig = plt.figure(figsize=(ops['mm']*100,ops['mm']*100))
    ax = fig.add_subplot(1,1,1)
    blank = np.empty((1360,2352));blank[:] = np.nan
    plt.imshow(blank)
    for k, boundary_coords in bf_left_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)
    for k, boundary_coords in bf_right_boundaries.items():
        plt.plot(*boundary_coords.T, c="gray", lw=1)

    color_anterior = 'blue'
    color_posterior = 'red'
    for a in range(len(animals)):    

        outputPath = os.path.join(ops['dataPath'],'data_byAnimal','A' + str(animals[a]),'Segmented_injections')

        contours0 = np.load(os.path.join(outputPath,'segmented_injection_contours.npy'))
        first = np.expand_dims(contours0[0,:],0)
        contours = np.append(contours0, first,axis=0)

        # plt.fill(contours[:,0], contours[:,1], color='#BD1828', alpha =0.1)
        # plt.plot(contours[:,0], contours[:,1], c= '#BD1828', linewidth=0.5)
        if animals[a] in anteriorAnimals:
            color = color_anterior
        elif animals[a] in posteriorAnimals:
            color = color_posterior

        plt.fill(contours[:,0], contours[:,1], color=color, alpha =0.2)
        plt.plot(contours[:,0], contours[:,1], c= color, linewidth=0.1)
        # centre_x, centre_y = centroid[a,:]
        # plt.scatter(centre_x, centre_y, c= color, s=50)


        plt.xlim(100, 540)  # Restrict x-axis
        plt.ylim(650, 1070)  # Restrict y-axis (inverted for image coordinates)
        # plt.title(str(animals[a]))
        plt.gca().invert_yaxis()  # Flip only the display
        for spine in ax.spines.values():
            spine.set_linewidth(1)  # Adjust thickness
        plt.xticks([],[])
        plt.yticks([],[])

    for a in range(len(animals)):    

        centre_x, centre_y = centroid[a,:]
        plt.scatter(centre_x, centre_y, c= 'k', s=10)

    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\segmentedInjections_onFlatmap_colored_AP.svg'))


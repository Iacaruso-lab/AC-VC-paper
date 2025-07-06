
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import imageio
from matplotlib import gridspec

from analysis_utils import *

def plotProportionCentre_onMap_red(fig,ax, ref,ref2, map_v1,df, ops,cmap='OrRd', b=300):
    
    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']

    leftBorder = 1.6666
    
    centre_tuned = np.nonzero(np.array(df['aziPeak']) < leftBorder)[0]    
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 

    binned_prop_map_centre = makeProportions_bySpatialBin_v3(df,binned_map, centre_tuned, thresh = 5, mask='none', V1_mask=[])
    
    binned_values_map_smooth = smooth_spatialBins(binned_prop_map_centre, spatialBin =b, nSmoothBins=1)

    def get_midPoint(x, a, b, c, d):
        return c + (x - a) * (d - c) / (b - a)
    
    vmax = 0.6 #np.nanmax(binned_values_map_smooth)
    vmin = 0

    plt.imshow(ref2)
    # plt.imshow(ref)
    pad = np.empty((13,513));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_values_map_smooth),0)
    binned_map_adj = binned_map_adj[:,:-40]
    pad = np.empty((398,37));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_map_adj),1)
    plt.imshow(binned_map_adj,cmap=cmap, vmin =vmin, vmax=vmax,alpha = 0.95)
    plt.yticks([],[])
    plt.xticks([],[])
    plt.axis('off')
    cbar = plt.colorbar(ticks = [0,0.3, 0.6],fraction=0.038, pad=0.04)
    cbar.ax.set_yticklabels(['0', '30', '60'], fontsize=15)

    
def plotBestElevation_red_onMap(fig, df, ref, ref2,map_V1, b=250):
    
    data = np.array(df['elevPeak'])
    df['elevPeak_inv'] = abs(np.nanmax(data)- data)   #flip it around so that max is top led location
    
    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']
       
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
    binned_values_map = makeMeanValue_bySpatialBin_v2(df, binned_map,thresh =5,  varName = 'elevPeak_inv', mask ='', V1_mask = map_V1)
    
    binned_values_map_smooth = smooth_spatialBins(binned_values_map, spatialBin =b, nSmoothBins=1)

    cmap = 'coolwarm'
    colors = sns.color_palette(cmap, n_colors =100, as_cmap = True)
        
    ax = fig.add_subplot(1,1,1)
    plt.imshow(ref2)
    pad = np.empty((13,513));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_values_map_smooth),0)
    binned_map_adj = binned_map_adj[:,:-40]
    pad = np.empty((398,37));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_map_adj),1)

    plt.imshow(binned_map_adj,cmap=colors,vmin =2-(30/18), vmax=2+(30/18), alpha =0.95)
    plt.yticks([],[])
    plt.xticks([],[])
    plt.axis('off')
    cbar = plt.colorbar(ticks = [2-(30/18),2,2+(30/18)],fraction=0.038, pad=0.04)
    cbar.ax.set_yticklabels(['-30', '0', '30'],fontsize=15)
    
    
def plotAzimuthDistance(df,peak,gaussFit, df_red, ops, eng,nShuffles=100):
         
    contraIdx = np.nonzero(peak > 6)[0]
    idx = np.intersect1d(gaussFit, contraIdx)
    
    outOnes = np.nonzero(np.array(df['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df)), outOnes)
    
    idx0 = np.intersect1d(idx,inOnes)
    
    data0 = peak[idx0] -6
    df0 = df.iloc[idx0]
    
    data1 = np.array(df_red['aziPeak'])  #flip it around so that max is top led location
    peakAzi_byArea_red = []
    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(df_red['area']) == ops['areas'][ar])[0]
        
        azi_thisArea = df_red.iloc[idx_thisArea]['aziPeak']
    
        peakAzi_byArea_red.append(np.nanmean(azi_thisArea))
        
    #get grean peak by session   
    #green
    aziShuffle = np.random.choice(np.arange(len(data0)), 100)
    peak_azi_bySession = []
    peak_azi_bySession_sh = []
    sessionIdx= np.unique(np.array(df0['sessionIdx']))
    for s in range(len(sessionIdx)):
        idx_thisSession = np.nonzero(np.array(df0['sessionIdx']) == sessionIdx[s])[0]
              
      
        peak_azi = data0[idx_thisSession]
        
        if len(idx_thisSession) < 10:
            peak_azi_bySession.append(np.nan)
        else:

            peak_azi_bySession.append(np.nanmean(peak_azi))
        
        sh = np.zeros((len(idx_thisSession), nShuffles))
        for n in range(nShuffles):
            idx = np.random.choice(np.arange(len(data0)), len(idx_thisSession))
            sh[:,n] = data0[idx]
        
        peak_azi_bySession_sh.append(sh)
      
    sessionRef = makeSessionReference(df0)
    #alternative shuffling: 
    aziShuffles_green_all = []
    for n in range(1000):
        aziShuffle = np.random.choice(np.arange(len(data0)), 100)
        aziShuffles_green_all.append(data0[aziShuffle])
    
    
    peakAzi_byArea = []
    peakAzi_byArea_sh = []

    for ar in range(len(ops['areas'])):  
        idx = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        peak_bySession_this = np.array([peak_azi_bySession[idx[i]] for i in range(len(idx))])
        peak_bySession_this_clean = peak_bySession_this[np.nonzero(np.isnan(peak_bySession_this) < 0.5)[0]]

        peakAzi_byArea.append(peak_bySession_this_clean)
        
        peak_bySession_this_sh = np.array([peak_azi_bySession_sh[idx[i]] for i in range(len(idx))])

        peakAzi_byArea_sh.append(peak_bySession_this_sh)
        
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_azi_bySession)) <0.5)[0]
    thisIdx = np.intersect1d(notV1,notNan)
   
    df_forTest = pd.DataFrame({'peakAzi_bySession': np.array(peak_azi_bySession)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'azi': np.array(sessionRef['seshAzi'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)


    formula = 'peakAzi_bySession ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
    
    
    formula = 'peakAzi_bySession ~ azi + (1|animal)'                 
    savePath = os.path.join(ops['outputPath'], 'LMM_green_aud.mat')
     
     #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 
      
    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')


    #Now same for red
    peak_azi_bySession_red = []
  
    sessionIdx= np.unique(np.array(df_red['sessionIdx']))
    for s in range(len(sessionIdx)):
        idx_thisSession = np.nonzero(np.array(df_red['sessionIdx']) == sessionIdx[s])[0]
              
        peak_azi = data1[idx_thisSession]
        
        peak_azi_bySession_red.append(np.nanmedian(peak_azi))
           
    sessionRef = makeSessionReference(df_red)
    peak_azi_bySession_red = np.array(peak_azi_bySession_red)
    
    peakAzi_byArea_red = []
    for ar in range(len(ops['areas'])):  
        idx = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        peak_bySession_this = peak_azi_bySession_red[idx]        
        peak_bySession_this_clean = peak_bySession_this[np.nonzero(np.isnan(peak_bySession_this) < 0.5)[0]]

        peakAzi_byArea_red.append(peak_bySession_this_clean)
            
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_azi_bySession_red)) <0.5)[0]
    thisIdx =notNan
    thisIdx = np.intersect1d(notV1,notNan)

    df_forTest = pd.DataFrame({'peakAzi_bySession': np.array(peak_azi_bySession_red)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)


    formula = 'peakAzi_bySession ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
    fig = plt.figure(figsize=(ops['mm']*80, ops['mm']*80), constrained_layout =True)
    
    ax = fig.add_subplot(1,1,1)
    for ar in range(len(ops['areas'])):
        median_azi = np.nanmean(peakAzi_byArea_red[ar])
        
        plt.plot([ar-0.3, ar+0.3], [median_azi,median_azi] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(peakAzi_byArea_red[ar])) 
        plt.scatter(xVals_scatter, np.array(peakAzi_byArea_red[ar]), s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,alpha=0.3,zorder = 1)
        
    myPlotSettings_splitAxis(fig, ax, 'Best visual azimuth (deg)', '', str(p_LMM), mySize=15)
    plt.xticks(np.arange(0, len(ops['areas'])), ops['areas'], rotation =90)
    plt.yticks([0,2,4,6], ['0', '36', '72', '108'])
    plt.ylim([-0.1,6.2])
    ax.tick_params(axis='y', pad=1)   

    ##% Distance between green and red
    fig = plt.figure(figsize=(ops['mm']*80, ops['mm']*80), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    for ar in range(1,len(ops['areas'])):
        median_red = np.nanmedian(peakAzi_byArea_red[ar])
        
        vals_thisArea = peakAzi_byArea[ar]
        
        notNan = np.nonzero(np.isnan(np.array(vals_thisArea)) <0.5)[0]
        vals_thisArea = np.array(vals_thisArea)[notNan]
        
        
        distance = [np.median(abs(vals_thisArea[i] - median_red)) for i in range(len(vals_thisArea))]
        median_distance = np.nanmedian(distance)
        
        distances_sh = []
        for i in range(len(aziShuffles_green_all)):
            distances_sh.append(np.nanmedian(abs(aziShuffles_green_all[i] - median_red)))
        
        upper = np.percentile(np.array(distances_sh), 95)
        lower = np.percentile(np.array(distances_sh), 5)

        vals_thisArea_sh = peakAzi_byArea_sh[ar]
        vals_sh = []
        for i in range(len(vals_thisArea_sh)):
            vals_sh.append(np.median([abs(vals_thisArea_sh[i][:,n] - median_red) for n in range(nShuffles)]))
                    
        vals_sh = np.array(vals_sh)[notNan]
        median_sh = np.nanmedian(np.array(vals_sh))
        
        plt.plot([ar-0.25, ar+0.25], [median_distance,median_distance] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(distance)) 
        plt.scatter(xVals_scatter, np.array(distance), s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,alpha=0.3,zorder = 1)
        
        plt.plot([ar-0.25, ar+0.25], [median_sh,median_sh] , linewidth = 2, c = 'silver',zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(distance)) 
        
        U,p = stats.mannwhitneyu(distance, vals_sh)
        adj_p = statsmodels.stats.multitest.multipletests(np.repeat(p,9), method='fdr_bh')[1][0]
        
        if adj_p < 0.05 and adj_p > 0.01:
            plt.text(ar-0.2, 2, '*', fontsize=15)
        elif adj_p < 0.01 and adj_p > 0.001:
             plt.text(ar-0.2, 2, '**', fontsize=15)
        elif adj_p < 0.001:
             plt.text(ar-0.2, 2, '***', fontsize=15)
             
    myPlotSettings_splitAxis(fig,ax, 'Azimuth distance (deg)', '', '',mySize=15)
    plt.xticks(np.arange(1,len(ops['areas'])), ops['areas'][1::], rotation =90, fontsize=15)
    deg_per_N = 18
    yPos = np.array([0,35,70])
    yPos0 = yPos/deg_per_N
    plt.yticks(yPos0, ['0', '35', '70'])
    # plt.ylim([-0.1, 3.9])
    plt.ylim([-0.1, yPos0[-1]])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    for tick in ax.get_xticklabels():
        tick.set_fontsize(5) 



def plotAzimuth_acrossMod_matchedFOV(df,peak,gaussFit, df_red,ops,eng):
    
    outOnes = np.nonzero(np.array(df['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df)), outOnes)
    
    idx0 = np.intersect1d(inOnes,gaussFit)
    
    contraIdx = np.nonzero(peak >= 6)[0]
    idx = np.intersect1d(idx0,contraIdx)
    
    df_green = df.iloc[idx]
    peak_green = peak[idx]
    df_green['aziPeak_fit'] = peak_green-6
    
    outOnes = np.nonzero(np.array(df_red['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df_red)), outOnes)
    
    df_red = df_red.iloc[inOnes]
    
    sessionRef_0 = makeSessionReference(df_green)
    sessionRef_1 = makeSessionReference(df_red)
    
    df_red['aziPeak_fit'] = df_red['aziPeak']
    df_red['elevPeak_fit'] = abs(df_red['elevPeak'] - max(df_red['elevPeak'])) #inverting it for plotting purposes

    #first do azimuth only
    sessionRef_0 = makeSessionReference(df_green)
    sessionRef_1 = makeSessionReference(df_red)
    
    sessionIdx_0 = np.unique(np.array(df_green['sessionIdx']))
    sessionIdx_1 = np.unique(np.array(df_red['sessionIdx']))
    commonOnes = np.intersect1d(sessionIdx_0,sessionIdx_1)
    
    seshAreas_common = []
    for i in range(len(commonOnes)):
        rel_idx = np.nonzero(sessionRef_0['seshIdx'] == commonOnes[i])[0]      
        seshAreas_common.append(sessionRef_0['seshAreas'][rel_idx[0]])

    areas_green, fitAzis_green,areas_red, fitAzis_red,fitElevs_red, sessionIdx, animal_g= [],[],[],[],[],[],[]
    sessionX, sessionY, sessionElev, sessionAzi = [], [], [], []
    for s in range(len(commonOnes)):  
        sessionIdx.append(commonOnes[s])

        #green
        idx_thisSession_g = np.nonzero(np.array(df_green['sessionIdx']) == commonOnes[s])[0]
        
        df_green_this = df_green.iloc[idx_thisSession_g]
        
        fitAzis_green.append(np.nanmedian(np.array(df_green_this['aziPeak_fit'])))
    
        theseAreas = np.array(df_green_this['area'])
        areas1, counts = np.unique(theseAreas, return_counts=True)
                   
        if len(areas1) > 0:   
            areas_green.append(areas1[np.argmax(counts)])                   
        else:
            areas_green.append('OUT')
            
        animal_g.append(np.array(df_green_this['animal'])[0])
        if s ==0:
            df_green_commonOnes_a1 = df_green_this
        else:
            df_green_commonOnes_a1 = pd.concat([df_green_commonOnes_a1, df_green_this])
        
        #red
        idx_thisSession_r = np.nonzero(np.array(df_red['sessionIdx']) == commonOnes[s])[0]
        
        df_red_this = df_red.iloc[idx_thisSession_r]
        
        fitAzis_red.append(np.nanmedian(np.array(df_red_this['aziPeak_fit'])))
        fitElevs_red.append(np.nanmedian(np.array(df_red_this['elevPeak_fit'])))

        
        theseAreas = np.array(df_red_this['area'])
        areas1, counts = np.unique(theseAreas, return_counts=True)
                   
        if len(areas1) > 0:   
            areas_red.append(areas1[np.argmax(counts)])                   
        else:
            areas_red.append('OUT')
            
        if s ==0:
            df_red_commonOnes_a1 = df_red_this
        else:
            df_red_commonOnes_a1 = pd.concat([df_red_commonOnes_a1, df_red_this])
        
    
    inV1 = np.nonzero(np.array(areas_green) == 'V1')[0]
     
    
    combDict = {'fitAzis_green': np.array(fitAzis_green)[inV1],
                # 'meanElevs_green': np.array(meanElevs_green),
                     'area_green': np.array(areas_green)[inV1],
                     'sessionIdx': np.array(sessionIdx)[inV1],
                     'animal': np.array(animal_g)[inV1],
                     'fitAzis_red': np.array(fitAzis_red)[inV1],
                     'fitElevs_red': np.array(fitElevs_red)[inV1],
                     'area_red': np.array(areas_red)[inV1]}    
    combDF = pd.DataFrame(data= combDict)
    #
    # run LMM 
    df_path= os.path.join(ops['outputPath'],'df_bySession_greenVsRed_forLMM.csv')
    combDF.to_csv(df_path)
    formula = 'fitAzis_green ~ 1 + fitAzis_red + (1|animal)'

    savePath = os.path.join(ops['outputPath'], 'LMM_greenVsRed_bySession.mat')
    
    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
        
    azimuths = [ '0', '36', '72', '108']
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['fitAzis_red'][0][0]
    slope_p = res['fitAzis_red'][0][1]
    xVals = np.arange(0,6.1,0.1)
    yVals = intercept + slope*xVals
     
    #
    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*80,ops['mm']*80), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.plot([0,6], [0,6], color = 'gray', linewidth = 0.3)
    plt.scatter(np.array(combDict['fitAzis_red']), np.array(combDict['fitAzis_green']), c= 'k', s =1)
    x_axis = 'fitAzis_red'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(combDF[x_axis]), max(combDF[x_axis]), len(fitLine))
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'gray',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 1, linestyle ='dashed') 
    myPlotSettings_splitAxis(fig, ax, 'Best sound azimuth, \n Ac boutons (\u00B0)', 'Best visual azimuth, \n V1 neurons (\u00B0)','', mySize=15)
    plt.text(3,1,'p: ' + str(np.round(slope_p,3)),fontsize=15)
    plt.xticks([0,2,4,6], azimuths)
    plt.yticks([0,2,4,6], azimuths)
    plt.ylim([0,6])
    plt.xlim([0,6])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   

    #%% NOw correlate it with proportion of centre-tuned boutons
    outOnes = np.nonzero(np.array(df['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df)), outOnes)
    
    idx0 = np.intersect1d(inOnes,gaussFit)
    
    df_gaussFit = df.iloc[idx0]
    peak_gauss = peak[idx0]
    
    leftBorder = 4.4
    rightBorder = 7.6
 
    left_tuned = np.nonzero(peak_gauss < leftBorder)[0]
    right_tuned = np.nonzero(peak_gauss > rightBorder)[0]
    centre_tuned0 = np.setdiff1d(np.arange(0,len(peak_gauss)), left_tuned)
    centre_tuned1 = np.setdiff1d(np.arange(0,len(peak_gauss)), right_tuned)
    centre_tuned = np.intersect1d(centre_tuned0, centre_tuned1)
    
    #shuffle
    nShuffles = 1000
    peak_gauss_sh = peak_gauss.copy(); np.random.shuffle(peak_gauss_sh)
    seshIdx_unique = np.unique(df_gaussFit['sessionIdx'])
    prop_left = np.empty(len(seshIdx_unique));prop_left[:] = np.nan
    prop_right = np.empty(len(seshIdx_unique));prop_right[:] = np.nan
    prop_centre = np.empty(len(seshIdx_unique));prop_centre[:] = np.nan
 
    for s in range(len(seshIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df_gaussFit['sessionIdx']) == seshIdx_unique[s])[0]
        
        if len(idx_thisSession) <10:
            continue
        left_thisSesh = np.intersect1d(idx_thisSession, left_tuned)
        right_thisSesh = np.intersect1d(idx_thisSession, right_tuned)
        centre_thisSesh = np.intersect1d(idx_thisSession, centre_tuned)
        
        
        prop_left[s] = len(left_thisSesh)/len(idx_thisSession)
        prop_right[s] = len(right_thisSesh)/len(idx_thisSession)
        prop_centre[s] = len(centre_thisSesh)/len(idx_thisSession)
 
    df_green = df_gaussFit
    sessionRef_0 = makeSessionReference(df_green)
    sessionRef_1 = makeSessionReference(df_red)
    
    sessionIdx_0 = np.unique(np.array(df_green['sessionIdx']))
    sessionIdx_1 = np.unique(np.array(df_red['sessionIdx']))
    commonOnes = np.intersect1d(sessionIdx_0,sessionIdx_1)
    
    commonOnes_idx = np.squeeze(np.array([np.nonzero(np.array(sessionRef_0['seshIdx']) == commonOnes[i])[0] for i in range(len(commonOnes))]))
    propCentre_green = prop_centre[commonOnes_idx]
    
    seshAreas_common = []
    for i in range(len(commonOnes)):
        rel_idx = np.nonzero(sessionRef_0['seshIdx'] == commonOnes[i])[0]      
        seshAreas_common.append(sessionRef_0['seshAreas'][rel_idx[0]])

    areas_green, areas_red, fitAzis_red,fitElevs_red, sessionIdx, animal_g= [],[],[],[],[],[]
    sessionX, sessionY, sessionElev, sessionAzi = [], [], [], []
    for s in range(len(commonOnes)):  
        sessionIdx.append(commonOnes[s])

        #green
        idx_thisSession_g = np.nonzero(np.array(df_green['sessionIdx']) == commonOnes[s])[0]
        
        df_green_this = df_green.iloc[idx_thisSession_g]
        
        theseAreas = np.array(df_green_this['area'])
        areas1, counts = np.unique(theseAreas, return_counts=True)
                   
        if len(areas1) > 0:   
            areas_green.append(areas1[np.argmax(counts)])                   
        else:
            areas_green.append('OUT')
            
        animal_g.append(np.array(df_green_this['animal'])[0])
        if s ==0:
            df_green_commonOnes_a1 = df_green_this
        else:
            df_green_commonOnes_a1 = pd.concat([df_green_commonOnes_a1, df_green_this])
        
        #red
        idx_thisSession_r = np.nonzero(np.array(df_red['sessionIdx']) == commonOnes[s])[0]
        
        df_red_this = df_red.iloc[idx_thisSession_r]
        
        fitAzis_red.append(np.nanmedian(np.array(df_red_this['aziPeak_fit'])))
        fitElevs_red.append(np.nanmedian(np.array(df_red_this['elevPeak_fit'])))

        
        theseAreas = np.array(df_red_this['area'])
        areas1, counts = np.unique(theseAreas, return_counts=True)
                   
        if len(areas1) > 0:   
            areas_red.append(areas1[np.argmax(counts)])                   
        else:
            areas_red.append('OUT')
            
        if s ==0:
            df_red_commonOnes_a1 = df_red_this
        else:
            df_red_commonOnes_a1 = pd.concat([df_red_commonOnes_a1, df_red_this])
        
    
    inV1 = np.nonzero(np.array(areas_green) == 'V1')[0]
     
    
    combDict = {'propCentre': np.array(propCentre_green)[inV1],
                # 'meanElevs_green': np.array(meanElevs_green),
                     'area_green': np.array(areas_green)[inV1],
                     'sessionIdx': np.array(sessionIdx)[inV1],
                     'animal': np.array(animal_g)[inV1],
                     'fitAzis_red': np.array(fitAzis_red)[inV1],
                     'fitElevs_red': np.array(fitElevs_red)[inV1],
                     'area_red': np.array(areas_red)[inV1]}    
    combDF = pd.DataFrame(data= combDict)
    azimuths = [ '0', '36', '72', '108']
    
    #Proportion, so doing spearman correlation
    r_spearman,p_spearman = scipy.stats.spearmanr(combDict['fitAzis_red'], combDict['propCentre'])

    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*80,ops['mm']*80), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(combDict['fitAzis_red']), np.array(combDict['propCentre']), c= 'k', s =1)
    myPlotSettings_splitAxis(fig, ax, 'Percentage centre-\ntuned AC-boutons (%)', 'Best visual azimuth, \n V1 neurons (\u00B0)','', mySize=15)
    plt.text(4,0.45,'r= ' + str(np.round(r_spearman,3)) + '\np= ' + str(np.round(p_spearman,3)), fontsize=15)
    plt.xticks([0,2,4,6], azimuths)
    plt.yticks([0,0.2,0.4,0.6], ['0', '20', '40', '60'])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   

    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestAzimuth_vs_propCentre_acrossMod_matchedFOV.svg'))

   
def plotPropCentre_againstAzi_spatialBins(ref,map_V1,df, df_red,ops, b =300, mask='none', propCentre_green=1, propCentre_red =1):

    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']
        
    if propCentre_green:
        leftBorder = 4.4
        rightBorder = 7.6
           
        # b = 300
        left_tuned = np.nonzero(np.array(df['peak']) < leftBorder)[0]
        right_tuned = np.nonzero(np.array(df['peak']) > rightBorder)[0]
        centre_tuned0 = np.setdiff1d(np.arange(0,len(np.array(df['peak']))), left_tuned)
        centre_tuned1 = np.setdiff1d(np.arange(0,len(np.array(df['peak']))), right_tuned)
        centre_tuned = np.intersect1d(centre_tuned0, centre_tuned1)
        
        binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
        
        binned_prop_map_centre = makeProportions_bySpatialBin_v3(df,binned_map, centre_tuned, thresh = 5, mask = mask, V1_mask=map_V1)
    
        bins_unique = np.unique(binned_map)
        
        binValues_green = getBinValues(binned_map, binned_prop_map_centre, ops['map_colors'], ops['colors_LUT'])
        y_title = 'Best sound azimuth \nAC-boutons (%)'

    else:
        contraTuned = np.nonzero(np.array(df['peak']) >=6)[0]
        df = df.iloc[contraTuned]
        df['peak'] = df['peak'] -6
        binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
        binned_values_map = makeMeanValue_bySpatialBin_v2(df, binned_map,thresh =5,  varName = 'peak', mask = mask, V1_mask = map_V1)
        bins_unique = np.unique(binned_map)
    
        binValues_green = getBinValues(binned_map, binned_values_map, ops['map_colors'], ops['colors_LUT'])
        y_title = 'Best sound azimuth \nAC-boutons (%)'

    

    df_red = df_red[~df_red['x'].isnull()]
    df_red = df_red[~df_red['y'].isnull()]
    df_red = df_red[df_red['x'] != 0]
    df_red = df_red[df_red['y'] != 0]
    df_red = df_red[df_red['area'] != 'OUT']
    
    if propCentre_red:
        leftBorder = 1.6666
    
        centre_tuned = np.nonzero(np.array(df_red['aziPeak']) < leftBorder)[0]
       
        binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
    
        binned_prop_map_centre_red = makeProportions_bySpatialBin_v3(df_red,binned_map, centre_tuned, thresh = 5, mask='none', V1_mask=[])
       
        binValues_red = getBinValues(binned_map, binned_prop_map_centre_red, ops['map_colors'], ops['colors_LUT'])
        x_title = 'Percentage centre-tuned \nVC-neurons (%)'
    #against real azimuth
    else:
        binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
        binned_values_map = makeMeanValue_bySpatialBin_v2(df_red, binned_map,thresh =5,  varName = 'aziPeak', mask = mask, V1_mask = map_V1)
        
        binValues_red = getBinValues(binned_map, binned_values_map, ops['map_colors'], ops['colors_LUT'])
        x_title = 'Best visual azimuth \nVC-neurons (%)'

        
    #against proportion centre-tuned
    vals_green, vals_red, valArea = [],[],[]
    for i in range(len(binValues_red)):
        if not np.isnan(binValues_red['values'][i]) and not np.isnan(binValues_green['values'][i]) and not binValues_green['binArea'][i] =='OUT':
            vals_green.append(binValues_green['values'][i])
            vals_red.append(binValues_red['values'][i])
            valArea.append(binValues_green['binArea'][i])

    vals_green = np.array(vals_green)
    vals_red = np.array(vals_red)
    if mask == 'V1':
        title = 'V1 spatial bins only'
    elif mask == 'HVAs':
        title = 'HVA spatial bins only'
    else:
        title = 'All spatial bins'
        
    areaColors = ops['myColorsDict']['HVA_colors']
    colors = np.array([areaColors[valArea[j]] for j in range(len(valArea))])
   
    #%%
    r,p = scipy.stats.spearmanr(vals_red, vals_green)

    fig = plt.figure(figsize=(ops['mm']*30,ops['mm']*30), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(vals_red, vals_green, s =5, facecolors =colors,alpha =0.5, linewidth=0)
    plt.text(0.1,0.4, 'r: ' + str(np.round(r,3)) + '\np: ' + str(np.round(p,3)), fontsize=5)
    myPlotSettings_splitAxis(fig, ax, y_title, x_title, '',mySize=6)
    if propCentre_green:
        plt.yticks([0,0.25, 0.5], ['0','25','50'])
    else:
        plt.yticks([0,2, 4,6], ['0','36','72','108'])
    if propCentre_red:
        plt.xticks([0,0.4, 0.8], ['0','40','80'])
    else:
        plt.xticks([0,2,4,6], ['0','36','72','108'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    

def plotElevationDistance(df,maps,peak, df_red, ops, eng,nShuffles=100):
    
    noArea_idx = np.nonzero(np.array(df['area']) == 'OUT')[0]
    
    badRoiPosition1 = np.nonzero(np.array(df['x']) ==0)[0]
    badRoiPosition2 = np.nonzero(np.array(df['y']) ==0)[0]
    badRoiPosition = np.unique(np.concatenate((badRoiPosition1,badRoiPosition2),0))
    
    noArea_idx = np.unique(np.concatenate((noArea_idx,badRoiPosition),0))
           

    elevPeak = getElevation_greenAud(df, maps, peak, onlyPeakSide = 1)
    # contraIdx = np.nonzero(np.array(df_fit_1d_green_aud_full_elev['gaussian_peak']) > 6)[0]

    includeIdx_green_elev = np.setdiff1d(np.arange(0,len(df)), noArea_idx)

    df_green_elev = df.iloc[includeIdx_green_elev]
    df_green_elev['elevPeak'] = elevPeak[includeIdx_green_elev]
    
    
    data0 = elevPeak[includeIdx_green_elev]
    df0= df_green_elev
        
    data1 = abs(np.nanmax(np.array(df_red['elevPeak']))- np.array(df_red['elevPeak']))   #flip it around so that max is top led location
    peakElev_byArea_red = []
    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(df_red['area']) == ops['areas'][ar])[0]
        
        elev_thisArea = df_red.iloc[idx_thisArea]['elevPeak']
    
        peakElev_byArea_red.append(np.nanmean(elev_thisArea))
        
    #get grean peak by session   
    #green
    peak_elev_bySession = []
    peak_elev_bySession_sh = []
    sessionIdx= np.unique(np.array(df0['sessionIdx']))
    for s in range(len(sessionIdx)):
        idx_thisSession = np.nonzero(np.array(df0['sessionIdx']) == sessionIdx[s])[0]
              
      
        peak_elev = data0[idx_thisSession]
        
        if len(idx_thisSession) < 10:
            peak_elev_bySession.append(np.nan)
        else:

            peak_elev_bySession.append(np.nanmean(peak_elev))
        
        sh = np.zeros((len(idx_thisSession), nShuffles))
        for n in range(nShuffles):
            idx = np.random.choice(np.arange(len(data0)), len(idx_thisSession))
            sh[:,n] = data0[idx]
        
        peak_elev_bySession_sh.append(sh)
      
    sessionRef = makeSessionReference(df0)
    # peak_azi_bySession = np.array(peak_azi_bySession)
    
    peakElev_byArea = []
    peakElev_byArea_sh = []

    for ar in range(len(ops['areas'])):  
        idx = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        peak_bySession_this = np.array([peak_elev_bySession[idx[i]] for i in range(len(idx))])
        peak_bySession_this_clean = peak_bySession_this[np.nonzero(np.isnan(peak_bySession_this) < 0.5)[0]]

        peakElev_byArea.append(peak_bySession_this_clean)
        
        peak_bySession_this_sh = np.array([peak_elev_bySession_sh[idx[i]] for i in range(len(idx))])
        # peak_bySession_this_clean = peak_bySession_this_sh[np.nonzero(np.isnan(peak_bySession_this_sh) < 0.5)[0]]

        peakElev_byArea_sh.append(peak_bySession_this_sh)
    
        
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_elev_bySession)) <0.5)[0]
    thisIdx = np.intersect1d(notV1,notNan)
    seshMapGood = np.nonzero(np.array(sessionRef['seshMapGood']) == 1)[0]
    thisIdx = np.intersect1d(thisIdx, seshMapGood)

    df_forTest = pd.DataFrame({'peakElev_bySession': np.array(peak_elev_bySession)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'elev': np.array(sessionRef['seshElev'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)


    formula = 'peakElev_bySession ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
    
    
    formula = 'peakElev_bySession ~ elev + (1|animal)'                 
    savePath = os.path.join(ops['outputPath'], 'LMM_green_aud.mat')
     
     #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 
      
    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')


    #Now same for red
    peak_elev_bySession_red = []
  
    sessionIdx= np.unique(np.array(df_red['sessionIdx']))
    for s in range(len(sessionIdx)):
        idx_thisSession = np.nonzero(np.array(df_red['sessionIdx']) == sessionIdx[s])[0]
              
        peak_elev = data1[idx_thisSession]
        
        peak_elev_bySession_red.append(np.nanmedian(peak_elev))
           
    sessionRef = makeSessionReference(df_red)
    peak_elev_bySession_red = np.array(peak_elev_bySession_red)
    
    peakElev_byArea_red = []
    for ar in range(len(ops['areas'])):  
        idx = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        peak_bySession_this = peak_elev_bySession_red[idx]
        # notNan = np.nonzero(np.isnan(np.array(peak_bySession_this)) <0.5)[0]
        
        peak_bySession_this_clean = peak_bySession_this[np.nonzero(np.isnan(peak_bySession_this) < 0.5)[0]]

        peakElev_byArea_red.append(peak_bySession_this_clean)
            
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_elev_bySession_red)) <0.5)[0]
    thisIdx =notNan
    thisIdx = np.intersect1d(notV1,notNan)

    df_forTest = pd.DataFrame({'peakElev_bySession': np.array(peak_elev_bySession_red)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)


    formula = 'peakElev_bySession ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
    #%%
    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout =True)
    
    ax = fig.add_subplot(1,1,1)
    for ar in range(len(ops['areas'])):
        median_elev = np.nanmean(peakElev_byArea_red[ar])
        
        plt.plot([ar-0.25, ar+0.25], [median_elev,median_elev] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(peakElev_byArea_red[ar])) 
        plt.scatter(xVals_scatter, np.array(peakElev_byArea_red[ar]), s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,alpha=0.3,zorder = 1)
        
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (deg)', '', str(p_LMM), mySize=15)
    plt.xticks(np.arange(0, len(ops['areas'])), ops['areas'], rotation =90)
    plt.yticks([-0.222,2,4.222], ['-40','0', '40'])
    plt.ylim([-0.222,4.222])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    ##% Distance between green and red
    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    for ar in range(1,len(ops['areas'])):
        median_red = np.nanmedian(peakElev_byArea_red[ar])
        
        vals_thisArea = peakElev_byArea[ar]
        
        notNan = np.nonzero(np.isnan(np.array(vals_thisArea)) <0.5)[0]
        vals_thisArea = np.array(vals_thisArea)[notNan]
        
        
        distance = [np.median(abs(vals_thisArea[i] - median_red)) for i in range(len(vals_thisArea))]
        median_distance = np.nanmedian(distance)
        
        vals_thisArea_sh = peakElev_byArea_sh[ar]
        vals_sh = []
        for i in range(len(vals_thisArea_sh)):
            vals_sh.append(np.median([abs(vals_thisArea_sh[i][:,n] - median_red) for n in range(nShuffles)]))
                    
        vals_sh = np.array(vals_sh)[notNan]
        median_sh = np.nanmedian(np.array(vals_sh))
        
        plt.plot([ar-0.25, ar+0.25], [median_distance,median_distance] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(distance)) 
        plt.scatter(xVals_scatter, np.array(distance), s = 20, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,alpha=0.3,zorder = 1)
        
        plt.plot([ar-0.25, ar+0.25], [median_sh,median_sh] , linewidth = 2, c = 'silver',zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(distance)) 
        # plt.scatter(xVals_scatter, np.array(vals_sh), s = 5, facecolors = 'white' , edgecolors = 'lightgray', linewidths =0.5,zorder = 1)
        
        U,p = stats.mannwhitneyu(distance, vals_sh)
        adj_p = statsmodels.stats.multitest.multipletests(np.repeat(p,10), method='fdr_bh')[1][0]
        # print(str(p))
        if adj_p < 0.05 and adj_p > 0.01:
            plt.text(ar-0.2, 2, '*', fontsize=15)
        elif adj_p < 0.01 and adj_p > 0.001:
             plt.text(ar-0.2, 2, '**', fontsize=15)
        elif adj_p < 0.001:
             plt.text(ar-0.4, 2, '***', fontsize=15)
 
        
    myPlotSettings_splitAxis(fig,ax, 'Elevation distance (deg)', '', '',mySize=15)
    plt.xticks(np.arange(1,len(ops['areas'])), ops['areas'][1::], rotation =90)
    deg_per_N = 18
    yPos = np.array([0,20,40])
    yPos0 = yPos/deg_per_N
    plt.yticks(yPos0, ['0', '20', '40'])
    plt.ylim([-0.1, yPos0[-1]])
    ax.tick_params(axis='y', pad=1) 
    ax.tick_params(axis='x', pad=1) 
    for tick in ax.get_xticklabels():
        tick.set_fontsize(5) 

    #%%
    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    vals_green, vals_red = [],[]
    for ar in range(1,len(ops['areas'])):
        median_red = np.nanmedian(peakElev_byArea_red[ar])
        print(str(median_red))
        vals_thisArea = peakElev_byArea[ar]
        
        notNan = np.nonzero(np.isnan(np.array(vals_thisArea)) <0.5)[0]
        vals_thisArea = np.array(vals_thisArea)[notNan]
       
        plt.scatter(np.repeat(median_red,len(vals_thisArea)), vals_thisArea, s = 3, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.25,alpha=0.3, zorder = 1)
        plt.scatter(median_red, np.median(vals_thisArea), s = 20, facecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =1,zorder = 2)
    
        vals_green.append(np.median(vals_thisArea))
        vals_red.append(median_red)     

    vals_green = np.array(vals_green)
    vals_red = np.array(vals_red)
    
    df_forTest = pd.DataFrame({'median_green': vals_green, 
                               'median_red': vals_red})

    formula = 'median_green~ median_red'

    df_path= os.path.join(ops['outputPath'],'df_bySession_green_freq_forLMM.csv')
    df_forTest.to_csv(df_path)
    
    savePath = os.path.join(ops['outputPath'], 'LMM_freq_green_aud.mat')
    
    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')

    lm = doLinearRegression(vals_red, vals_green)
    x_axis = 'median_red'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(df_forTest[x_axis]), max(df_forTest[x_axis]), len(fitLine))   
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'silver',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 0.5, linestyle='dashed')

    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation, AC-boutons', 'Best visual elevation, VC-neurons', '',mySize=15)

    plt.yticks([2-(20/18),2,2+(20/18),2+(40/18)], ['-20','0', '20', '40'])
    plt.ylim([2-(20/18),2+(40/18)])
    plt.xticks([2-(20/18),2,2+(20/18),2+(40/18)], ['-20','0', '20', '40'])
    plt.xlim([2-(20/18),2+(40/18)])
    plt.plot([0,4],[0,4], color='gray', linewidth=0.25)
    # plt.plot(lm['x_vals'], lm['y_vals'], c = 'b')
    plt.text(2-(20/18),3.5, 'r: ' + str(np.round(lm['corr'],3)) + '\np: ' + str(np.round(res[x_axis][0][1],3)),fontsize=15)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    return vals_green, vals_red


def plotElevation_spatialBins_acrossMod(ref,df, maps, peak,df_red,ops, b =300, mask ='none'):
    
    elev = getElevation_greenAud(df, maps, peak)
    
    df['peak'] = elev
    
    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']
    
    mapsPath =  'Z:\\home\\shared\\Alex_analysis_camp\\retinotopyMaps\\'
    map_V1 = imageio.imread(os.path.join(mapsPath,'Reference_map_allen_V1Marked.png'))
        
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
    binned_values_map = makeMeanValue_bySpatialBin_v2(df, binned_map,thresh =5,  varName = 'peak', mask =mask, V1_mask = map_V1)

    bins_unique = np.unique(binned_map)
    binValues_green = getBinValues(binned_map, binned_values_map, ops['map_colors'], ops['colors_LUT'])

    #now red
    df_red = df_red[~df_red['x'].isnull()]
    df_red = df_red[~df_red['y'].isnull()]
    df_red = df_red[df_red['x'] != 0]
    df_red = df_red[df_red['y'] != 0]
    df_red = df_red[df_red['area'] != 'OUT']
    
    data = np.array(df_red['elevPeak'])
    df_red['elevPeak_inv'] = abs(np.nanmax(data)- data)   #flip it around so that max is top led location
    
    mapsPath =  'Z:\\home\\shared\\Alex_analysis_camp\\retinotopyMaps\\'
    map_V1 = imageio.imread(os.path.join(mapsPath,'Reference_map_allen_V1Marked.png'))
        
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
    binned_values_map = makeMeanValue_bySpatialBin_v2(df_red, binned_map,thresh =5,  varName = 'elevPeak_inv', mask = mask, V1_mask = map_V1)
    
    binValues_red = getBinValues(binned_map, binned_values_map, ops['map_colors'], ops['colors_LUT'])
    
    vals_green, vals_red, valArea = [],[],[]
    for i in range(len(binValues_red)):
        if not np.isnan(binValues_red['values'][i]) and not np.isnan(binValues_green['values'][i]):
            vals_green.append(binValues_green['values'][i])
            vals_red.append(binValues_red['values'][i])
            valArea.append(binValues_green['binArea'][i])

    vals_green = np.array(vals_green)
    vals_red = np.array(vals_red)
    
    areaColors = ops['myColorsDict']['HVA_colors']
    colors = np.array([areaColors[valArea[j]] for j in range(len(valArea))])

    fig = plt.figure(figsize=(ops['mm']*100,ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(vals_red, vals_green, s=25,facecolors =colors,alpha =0.5, linewidth=0)
    lm = doLinearRegression_withCI(vals_red, vals_green)
    plt.plot(lm['x_vals'], lm['y_vals'],c = 'k',linestyle='dashed',linewidth = 2)
    plt.fill_between(lm['x_vals'], lm['ci_lower'], lm['ci_upper'], facecolor = 'silver',alpha = 0.3)
    plt.text(1,3.4, 'r: ' + str(np.round(lm['corr'],3)) + '\np: ' + str(np.round(lm['pVal_corr'],3)), fontsize=15)
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation, AC-boutons', 'Best visual elevation, VC-neurons', '',mySize=15)
    plt.yticks([2-(20/18),2,2+(20/18),2+(40/18)], ['-20','0', '20', '40'])
    plt.xticks([2-(20/18),2,2+(20/18),2+(40/18)], ['-20','0', '20', '40'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
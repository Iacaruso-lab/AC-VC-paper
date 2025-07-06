import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.io import loadmat
from scipy.stats import friedmanchisquare, wilcoxon                
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import imageio
from matplotlib import gridspec

from analysis_utils import *


def plotAzimuthDistribution(df, peak,fig,ax):
    if np.max(peak) > 7:
        azimuths = ['-108','-90','-72','-54','-36','-18','0','18','36','54','72','90','108']
    else:
        azimuths = ['-108','-72','-36','0','36','72','108']

    bins_peak = np.arange(0,len(azimuths), 1)

    hist_all, bins = np.histogram(peak,bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = '#C8C6C6',  histtype ='stepfilled',alpha = 0.4)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = 'k', histtype ='step', linewidth = 0.5)
    plt.xlim([min(bins_peak),max(bins_peak)])
    plt.xticks([0,6,12],['-108','0','108'])           

    plt.ylim([0,0.15])
    plt.yticks([0,0.05, 0.1, 0.15], ['0','5','10','15'])
    plt.xlim([-0.1, 12.1])
    plt.text(0,0.14, 'n: ' + str(len(peak)), fontsize=15)     
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Sound azimuth (\u00b0)', '', mySize=15)
    ax.spines['bottom'].set_bounds(0,12)
    ax.tick_params(axis='x', pad=1)   
    ax.tick_params(axis='y', pad=1)   
    
def plotAzimuthDistribution_byArea(fig, gs, df, gaussFit,peak, ops):
    peaks_byArea, peaks_collapsed_byArea = [], []
    cnt = 0
    bins_peak = np.arange(0,len(ops['azimuths']), 1.2)

    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(df['area']) == ops['areas'][ar])[0]
        
        gaussIdx_thisArea = np.intersect1d(gaussFit, idx_thisArea)
        # gaussIdx_thisArea = np.intersect1d(gaussIdx_thisArea,a1_idx)
        
        peaks_this = peak[gaussIdx_thisArea]
        peaks_all =  peak[gaussFit]  
       #  peaks_all =  param_gauss[np.intersect1d(gaussFit, a1_idx),1]  
       
        t, p_ks = stats.kstest(peaks_this, peaks_all)
    
        peaks_this_collapsed = abs(peaks_this - np.round(max(peaks_all))/2)
        peaks_byArea.append(peaks_this)
        peaks_collapsed_byArea.append(peaks_this_collapsed)

        
        hist_thisArea, bins = np.histogram(peaks_this,bins_peak)
        hist_thisArea_norm = hist_thisArea/np.sum(hist_thisArea)
        median_thisArea = np.nanmedian(peaks_this)
                    
        hist_all, bins = np.histogram(peaks_all,bins_peak)
        hist_all_norm = hist_all/np.sum(hist_all)
        
        if np.mod(cnt,2) ==0:
            k = 0
        else:
            k=1
        ax = fig.add_subplot(gs[int(np.floor(cnt/2)), k])       
        #option 2
        plt.hist(bins[:-1],bins,weights = hist_thisArea_norm, color = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],histtype='step',linewidth = 2, alpha = 1,label = 'n: ' + str(len(gaussIdx_thisArea)))           
        plt.hist(bins[:-1],bins,weights = hist_all_norm, color = '#C8C6C6', histtype='stepfilled', alpha=1)
        plt.xlim([min(bins_peak),max(bins_peak)])
        
        if len(ops['azimuths']) ==13:
            plt.xticks([0,6,12],['-108', '0', '108'])           
            plt.ylim([0,0.3])
            plt.yticks([0, 0.3], ['0', '30'])
            plt.xlim([-0.2, 12.2])
             
            if ops['areas'][ar] == 'POR':
                plt.text(0.5, 0.75, 'POR', ha='center', fontsize=15,  weight='normal', transform=plt.gca().transAxes, color=ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])
            else:
                plt.text(5.8, 0.23, ops['areas'][ar], horizontalalignment ='center', weight='normal',color=ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])
            if cnt ==8:
                myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Sound azimuth (\u00b0)', '', mySize=15)
            elif cnt == 9:
                myPlotSettings_splitAxis(fig, ax, '', '','', mySize=15)
            else:
                ax.spines["bottom"].set_visible(False)
                plt.xticks([],[])
                myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=15)
            if k==1:
                plt.yticks([0, 0.3], ['', ''])
                
            ax.tick_params(axis='both', length=2)  # Change tick length for both axes
            ax.tick_params(axis='y', pad=1)   
            ax.tick_params(axis='x', pad=1)   

        cnt +=1  
        

def plotProportionCentre_onMap(fig, ref,ref2, df, ops, b=300):
    
    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']

    mapsPath =  'Z:\\home\\shared\\Alex_analysis_camp\\retinotopyMaps\\'
    map_V1 = imageio.imread(os.path.join(mapsPath,'Reference_map_allen_V1Marked.png'))
    
    # for b in binSize:
    leftBorder = 4.4 # 6 = 0 degrees. -30 deg  is 4.4, because 1 space is 18 deg
    rightBorder = 7.6
       
    # b =300
    left_tuned = np.nonzero(np.array(df['peak']) < leftBorder)[0]
    right_tuned = np.nonzero(np.array(df['peak']) > rightBorder)[0]
    centre_tuned0 = np.setdiff1d(np.arange(0,len(np.array(df['peak']))), left_tuned)
    centre_tuned1 = np.setdiff1d(np.arange(0,len(np.array(df['peak']))), right_tuned)
    centre_tuned = np.intersect1d(centre_tuned0, centre_tuned1)
    
    lateral_tuned = np.setdiff1d(np.arange(0,len(df)), centre_tuned)
    
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 

    binned_prop_map_centre = makeProportions_bySpatialBin_v3(df,binned_map, centre_tuned, thresh = 5, mask='none', V1_mask=[])
    
    binned_values_map_smooth = smooth_spatialBins(binned_prop_map_centre, spatialBin =b, nSmoothBins=1)

    def get_midPoint(x, a, b, c, d):
        return c + (x - a) * (d - c) / (b - a)
    
    #chance = 0.18118882788254953
    vmax = 0.32 #np.nanmax(binned_prop_map_left)
    vmin = 0
    cmap = 'OrRd'
    
    ax = fig.add_subplot(1,1,1)
    plt.imshow(ref2)
    pad = np.empty((13,513));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_values_map_smooth),0)
    binned_map_adj = binned_map_adj[:,:-40]
    pad = np.empty((398,37));pad[:] = np.nan
    binned_map_adj = np.concatenate((pad,binned_map_adj),1)

    plt.imshow(binned_map_adj,cmap=cmap, vmin =vmin, vmax=vmax,alpha = 0.95)
    plt.yticks([],[])
    plt.xticks([],[])
    plt.axis('off')
    plt.title('Percentage centre-tuned boutons (%)')
    cbar = plt.colorbar(ticks = [0,0.16, 0.32],fraction=0.038, pad=0.04)
    cbar.ax.set_yticklabels(['0', '16', '32'], fontsize=15)
    
    
def plotProportionCentre_bySession(df,gaussFit,peak, eng, ops, injectionSubset = []):
    leftBorder = 4.4 #
    rightBorder = 7.6
    
    outOnes = np.nonzero(np.array(df['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df)), outOnes)
    
    idx = np.intersect1d(inOnes,gaussFit)
    
    ventral_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['ventralAnimals'] for i in range(len(df))]))[0]
    dorsal_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['dorsalAnimals'] for i in range(len(df))]))[0]
    anterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['anteriorAnimals'] for i in range(len(df))]))[0]
    posterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['posteriorAnimals'] for i in range(len(df))]))[0]
    
    if len(injectionSubset) > 0:
        if injectionSubset == 'ventral':
            idx = np.intersect1d(ventral_idx, idx)
        elif injectionSubset == 'dorsal':
            idx = np.intersect1d(dorsal_idx, idx)
        elif injectionSubset == 'anterior':
            idx = np.intersect1d(anterior_idx, idx)
        elif injectionSubset == 'posterior':
            idx = np.intersect1d(posterior_idx, idx)
            
    peak_gauss = peak[idx]
    df_gaussFit = df.iloc[idx]
  
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

    sessionRef = makeSessionReference(df_gaussFit)   
    
    inj_DV, inj_AP= [],[]
    for j in range(len(sessionRef['seshAnimal'])):
        if sessionRef['seshAnimal'][j] in ops['ventralAnimals']:
            inj_DV.append('Ventral')
        elif sessionRef['seshAnimal'][j] in ops['dorsalAnimals']:
            inj_DV.append('Dorsal')
            
        if sessionRef['seshAnimal'][j] in ops['anteriorAnimals']:
            inj_AP.append('Anterior')
        elif sessionRef['seshAnimal'][j] in ops['posteriorAnimals']:
            inj_AP.append('Posterior')

    prop_left_byArea, prop_right_byArea, prop_centre_byArea = [],[],[]
    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        prop_this = np.array(prop_left[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_left_byArea.append(prop_this)
        
        prop_this = np.array(prop_right[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_right_byArea.append(prop_this)
        
        prop_this = np.array(prop_centre[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_centre_byArea.append(prop_this)
        
    notNanIdx = np.nonzero(np.isnan(np.array(prop_centre)) < 0.5)[0]  
        
    df_props_forTest = pd.DataFrame({'proportion_centre': np.array(prop_centre[notNanIdx]),
                                     'proportion_left': np.array(prop_left[notNanIdx]),
                                     'proportion_right': np.array(prop_right[notNanIdx]),
                            'area': np.array(sessionRef['seshAreas'])[notNanIdx],
                            'stream': np.array(sessionRef['seshStream'])[notNanIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[notNanIdx],
                            'Inj_DV': np.array(sessionRef['pos_DV'])[notNanIdx],
                            'Inj_AP': np.array(sessionRef['pos_AP'])[notNanIdx],
                            'prop_ventral': np.array(sessionRef['prop_ventral'])[notNanIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_prop_forTest.csv')

    df_props_forTest.to_csv(df_path)

    prop_left_areaShuffle = np.zeros((len(ops['areas']), nShuffles))
    prop_centre_areaShuffle = np.zeros((len(ops['areas']), nShuffles))
    prop_right_areaShuffle = np.zeros((len(ops['areas']), nShuffles))

    for n in range(nShuffles):
        areas_bySession = np.array(sessionRef['seshAreas'])
        np.random.shuffle(areas_bySession)
        for ar in range(len(ops['areas'])):  
            idx = np.nonzero(areas_bySession  == ops['areas'][ar])[0]
            prop_left_areaShuffle[ar,n] = np.nanmedian(prop_left[idx])
            prop_centre_areaShuffle[ar,n] = np.nanmedian(prop_centre[idx])
            prop_right_areaShuffle[ar,n] = np.nanmedian(prop_right[idx])

    N = 200
    left_sh, centre_sh, right_sh = [], [],[]
    for n in range(nShuffles):
        rand = np.random.choice(np.arange(0,len(df_gaussFit)), N, replace =True)
        
        left_sh.append(len(np.intersect1d(left_tuned, rand))/N)
        centre_sh.append(len(np.intersect1d(centre_tuned, rand))/N)
        right_sh.append(len(np.intersect1d(right_tuned, rand))/N)
            
    left_sh = np.mean(left_sh)
    centre_sh = np.mean(centre_sh)
    right_sh = np.mean(right_sh)
   
    
    #centre tuned
    #%%
    fig = plt.figure(figsize = (ops['mm']*100,ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    formula = 'proportion_centre ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)

    propCentre_median_byArea = []
    for ar in range(len(ops['areas'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_centre_byArea[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_centre_byArea[ar]),np.nanmedian(prop_centre_byArea[ar])], linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],alpha=1,zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_centre_byArea[ar]), s = 10, facecolors ='white' , edgecolor = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 1,linewidth=0.5, alpha =0.3)

        propCentre_median_byArea.append(np.nanmedian(prop_centre_byArea[ar]))
       
        if p_LMM < 0.05:
            p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(prop_centre_byArea, multiComp = 'fdr')
            # p_mannWhitney
            cnt = 0
            for c in range(len(compIdx)):
                if p_mannWhitney[c] < 0.05:
                    pos = compIdx[c].split('_')
                    plt.hlines(0.52+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.35)
                    cnt += 0.02
        
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', '', 'Centre, p: ' + str(p_LMM), mySize=15)  
    # myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=5)  
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation = 90)
    plt.ylim([-0.02, 0.8])
    plt.yticks([0,0.2,0.4,0.6,0.8], ['0','20', '40','60', '80'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\percentageCentre_byArea.svg'))


    #%% Also plot the other proportions for the supplementals
    fig = plt.figure(figsize = (ops['mm']*200,ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,2,1)
    formula = 'proportion_left ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
           
   
    for ar in range(len(ops['areas'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_left_byArea[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_left_byArea[ar]),np.nanmedian(prop_left_byArea[ar])], linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],alpha=1,zorder = 2) 
        plt.scatter(xVals_scatter, np.array(prop_left_byArea[ar]), s = 10, facecolors ='white' , edgecolor = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 1,linewidth=0.5, alpha =0.3)

        if p_LMM < 0.05:
            p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(prop_left_byArea, multiComp = 'fdr')
            # p_mannWhitney
            cnt = 0
            for c in range(len(compIdx)):
                if p_mannWhitney[c] < 0.05:
                    pos = compIdx[c].split('_')
                    plt.hlines(0.52+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.35)
                    cnt += 0.02      
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', '',  'Ipsi, p: ' + str(p_LMM), mySize=15)  
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation = 90)
    plt.ylim([-0.02, 1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1], ['0','20', '40','60','80','100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    
    ax = fig.add_subplot(1,2,2)
    formula = 'proportion_right ~ area + (1|animal)'                 
    p_LMM = eng.linearMixedModel_fromPython_anova(df_path, formula, nargout=1)
           
    for ar in range(len(ops['areas'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_right_byArea[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_right_byArea[ar]),np.nanmedian(prop_right_byArea[ar])], linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],alpha=1,zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_right_byArea[ar]), s = 10, facecolors ='white' , edgecolor = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 1,linewidth=0.5, alpha =0.3)

        
        if p_LMM < 0.05:
            p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(prop_right_byArea, multiComp = 'fdr')
            # p_mannWhitney
            cnt = 0
            for c in range(len(compIdx)):
                if p_mannWhitney[c] < 0.05:
                    pos = compIdx[c].split('_')
                    plt.hlines(0.52+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.35)
                    cnt += 0.02
      
    myPlotSettings_splitAxis(fig, ax, '', '',  'Contra, p: ' + str(p_LMM), mySize=15)  
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation = 90)
    plt.ylim([-0.02, 1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1], ['0','20', '40','60','80','100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    
    # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\percentageLeftRight_byArea.svg'))
   
    return propCentre_median_byArea


def plotProportionCentre_byStream(fig, df,peak,gaussFit,eng, ops):
    leftBorder = 4.4
    rightBorder = 7.6
    
    outOnes = np.nonzero(np.array(df['area']) == 'OUT')[0]
    inOnes = np.setdiff1d(np.arange(0, len(df)), outOnes)
    
    idx = np.intersect1d(inOnes,gaussFit)

    peak_gauss = peak[idx]
    df_gaussFit = df.iloc[idx]
  
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

    sessionRef = makeSessionReference(df_gaussFit)
            
    prop_left_byGroup, prop_right_byGroup, prop_centre_byGroup = [],[],[]
    for ar in range(len(ops['groups'])):
        idx_thisArea = np.nonzero(np.array(sessionRef['seshStream']) == ops['groups'][ar])[0]
        
        prop_this = np.array(prop_left[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_left_byGroup.append(prop_this)
        
        prop_this = np.array(prop_right[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_right_byGroup.append(prop_this)
        
        prop_this = np.array(prop_centre[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_centre_byGroup.append(prop_this)
                  
       
    notNanIdx = np.nonzero(np.isnan(np.array(prop_centre)) < 0.5)[0]  
    notV1 = np.nonzero(np.array(sessionRef['seshStream']) != 'V1')[0]
    thisIdx = np.intersect1d(notV1,notNanIdx)

    df_props_forTest = pd.DataFrame({'proportion_centre': np.array(prop_centre[thisIdx]),
                                     'proportion_left': np.array(prop_left[thisIdx]),
                                     'proportion_right': np.array(prop_right[thisIdx]),
                                     'area': np.array(sessionRef['seshAreas'])[thisIdx],
                                     'stream': np.array(sessionRef['seshStream'])[thisIdx],
                                     'animal':  np.array(sessionRef['seshAnimal'])[thisIdx],
                                     'Inj_DV': np.array(sessionRef['pos_DV'])[thisIdx],
                                     'Inj_AP': np.array(sessionRef['pos_AP'])[thisIdx],
                                     'prop_ventral': np.array(sessionRef['prop_ventral'])[thisIdx]})    
        
    #plot it
    df_path = os.path.join(ops['outputPath'], 'df_prop_forTest.csv')
    df_props_forTest.to_csv(df_path)
    
    #% Centre tuned
    fig = plt.figure(figsize=(ops['mm']*60, ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
   
    formula = 'proportion_centre ~ stream + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
                
    for ar in range(1,len(ops['groups'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.06,size = len(prop_centre_byGroup[ar])) 
        plt.plot([ar-0.2,ar+0.2], [np.nanmedian(prop_centre_byGroup[ar]),np.nanmedian(prop_centre_byGroup[ar])], linewidth = 2, c = ops['colors_groups'][ar],zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_centre_byGroup[ar]), s = 10, facecolors = 'white' , edgecolors = ops['colors_groups'][ar], linewidths =0.5,zorder = 1, alpha =0.3)
               
    myPlotSettings_splitAxis(fig, ax, 'Percentage centre-tuned boutons (%)', '', 'p: ' + str(np.round(p_LMM,3)), mySize=15)  
    # plt.xticks(np.arange(1,len(ops['groups'])), ['Ventral','Dorsal' ], rotation = 45, horizontalalignment='right')
    plt.xticks(np.arange(1,len(ops['groups'])), ['Ventral','Dorsal' ])
    plt.ylim([-0.02, 0.8])
    plt.yticks([0,0.2,0.4, 0.6, 0.8], ['0', '20', '40', '60', '80'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

  
    #left tuned
    fig = plt.figure(figsize = (ops['mm']*100,ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,2,1)
    formula = 'proportion_left ~ stream + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
           
    for ar in range(1,len(ops['groups'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_left_byGroup[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_left_byGroup[ar]),np.nanmedian(prop_left_byGroup[ar])], linewidth = 2, c = ops['colors_groups'][ar],zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_left_byGroup[ar]), s = 10, facecolors = 'white' , edgecolors =  ops['colors_groups'][ar], linewidths =0.5,zorder = 1,alpha=0.3)
           
    p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(prop_left_byGroup, multiComp = 'fdr')
    cnt = 0
    for c in range(len(compIdx)):
        if p_mannWhitney[c] < 0.05:
            pos = compIdx[c].split('_')
            plt.hlines(0.9+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.5)
            cnt += 0.02    
        
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', '', 'Ipsi, p: ' + str(np.round(p_LMM,3)), mySize=15)  
    plt.xticks(np.arange(1,len(ops['groups'])), ['Ventral','Dorsal' ], rotation = 0, horizontalalignment='center')
    plt.ylim([-0.05, 1])
    plt.yticks([0,0.5,1], ['0', '50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    
    ax = fig.add_subplot(1,2,2)
    formula = 'proportion_right ~ stream + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
           
    for ar in range(1,len(ops['groups'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_right_byGroup[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_right_byGroup[ar]),np.nanmedian(prop_right_byGroup[ar])], linewidth = 2, c = ops['colors_groups'][ar],zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_right_byGroup[ar]), s = 10, facecolors = 'white' , edgecolors = ops['colors_groups'][ar], linewidths =0.5,zorder = 1, alpha=0.3)
    
    p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(prop_right_byGroup, multiComp = 'fdr')
    cnt = 0
    for c in range(len(compIdx)):
        if p_mannWhitney[c] < 0.05:
            pos = compIdx[c].split('_')
            plt.hlines(0.9+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.5)
            cnt += 0.02    
        
    myPlotSettings_splitAxis(fig, ax, '', '', 'Contra, p: ' + str(np.round(p_LMM,3)), mySize=15)  
    plt.xticks(np.arange(1,len(ops['groups'])), ['Ventral','Dorsal' ], rotation = 0, horizontalalignment='center')
    plt.ylim([-0.05, 1])
    plt.yticks([0,0.5,1], ['0', '50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    #---------------------------------------------------------------------------------------------------------
    t = []
    for i in range(len(df_gaussFit)):
        if df_gaussFit['area'].iloc[i] in ops['dorsal']:
            t.append('Dorsal')
        elif df_gaussFit['area'].iloc[i] in ops['ventral']:
            t.append('Ventral')
        elif df_gaussFit['area'].iloc[i] == 'V1':
            t.append('V1')
        else:
            t.append('')

    df_gaussFit['streamIdx'] = t
    
    #% ------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(100*ops['mm'], 100*ops['mm']), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    bins_peak = np.arange(0,len(ops['azimuths']), 1)

    for ar in range(1,len(ops['groups'])):
        idx_thisArea = np.nonzero(np.array(df_gaussFit['streamIdx']) == ops['groups'][ar])[0]
        
        peaks_this = peak_gauss[idx_thisArea]
        peaks_all =  peak_gauss

        hist_thisArea, bins = np.histogram(peaks_this,bins_peak)
        hist_thisArea_norm = hist_thisArea/np.sum(hist_thisArea)
                    
        hist_all, bins = np.histogram(peaks_all,bins_peak)
        hist_all_norm = hist_all/np.sum(hist_all)
        
       
        plt.hist(bins[:-1],bins,weights = hist_thisArea_norm, color = ops['colors_groups'][ar],histtype='stepfilled', alpha = 0.1,label = 'n: ' + str(len(peaks_this)))           
        plt.hist(bins[:-1],bins,weights = hist_thisArea_norm, color =ops['colors_groups'][ar],histtype='step',linewidth = 0.75, alpha = 1)           
        plt.xlim([min(bins_peak),max(bins_peak)])
        if len(ops['azimuths']) ==13:
            plt.xticks([0,6,12],['-108', '0', '108'])           
            plt.ylim([0,0.15])
            plt.yticks([0,0.05, 0.1,0.15], ['0','5','10','15'])
            plt.xlim([-0.2, 12.2])
           
            myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Sound azimuth (\u00b0)', '', mySize=15)
          
        ax.tick_params(axis='both', length=2)  # Change tick length for both axes
        ax.tick_params(axis='y', pad=1)  
        ax.tick_params(axis='x', pad=1)  

        
def exploreInjectionLocation_azimuths(df0, peak0, ops,eng):
    #%%
    animals_thisDataset = df0['animal'].unique()
    
    ventralAn = np.intersect1d(animals_thisDataset, ops['ventralAnimals'])        
    dorsalAn = np.intersect1d(animals_thisDataset, ops['dorsalAnimals'])  
   
    anteriorAn = np.intersect1d(animals_thisDataset,ops['anteriorAnimals'])         
    posteriorAn = np.intersect1d(animals_thisDataset,ops['posteriorAnimals'])    
    
    t = []
    for i in range(len(df0)):
        if df0['area'].iloc[i] in ops['dorsal']:
            t.append('Dorsal')
        elif df0['area'].iloc[i] in ops['ventral']:
            t.append('Ventral')
        elif df0['area'].iloc[i] == 'V1':
            t.append('V1')
        else:
            t.append('')                        
    df0['streamIdx'] = t
    
    
    animals = df0['animal'].unique()
    post_idx = np.zeros(0,); ant_idx = np.zeros(0,)
    dorsal_idx = np.zeros(0,); ventral_idx = np.zeros(0,)
    post_idx_V1= np.zeros(0,); ant_idx_V1 = np.zeros(0,)
    dorsal_idx_V1 = np.zeros(0,); ventral_idx_V1 = np.zeros(0,)
    post_idx_dorsalStream= np.zeros(0,); ant_idx_dorsalStream = np.zeros(0,)
    dorsal_idx_dorsalStream = np.zeros(0,); ventral_idx_dorsalStream = np.zeros(0,)
    post_idx_ventralStream= np.zeros(0,); ant_idx_ventralStream = np.zeros(0,)
    dorsal_idx_ventralStream = np.zeros(0,); ventral_idx_ventralStream = np.zeros(0,)
    for a in range(len(animals)):
        idx_this = np.nonzero(np.array(df0['animal']) == animals[a])[0]
        
        idx_V1 = np.nonzero(np.array(df0['streamIdx']) == 'V1')[0]
        idx_ventral = np.nonzero(np.array(df0['streamIdx']) == 'Ventral')[0]
        idx_dorsal = np.nonzero(np.array(df0['streamIdx']) == 'Dorsal')[0]

        idx_this_V1 = np.intersect1d(idx_this,idx_V1)
        idx_this_dorsal = np.intersect1d(idx_this,idx_dorsal)
        idx_this_ventral = np.intersect1d(idx_this,idx_ventral)

        if animals[a] in posteriorAn:
            post_idx = np.concatenate((post_idx,idx_this),0)
            post_idx_V1 = np.concatenate((post_idx_V1,idx_this_V1),0)
            post_idx_dorsalStream = np.concatenate((post_idx_dorsalStream,idx_this_dorsal),0)
            post_idx_ventralStream = np.concatenate((post_idx_ventralStream,idx_this_ventral),0)
        elif animals[a] in anteriorAn:
            ant_idx = np.concatenate((ant_idx,idx_this),0)
            ant_idx_V1 = np.concatenate((ant_idx_V1,idx_this_V1),0)
            ant_idx_dorsalStream = np.concatenate((ant_idx_dorsalStream,idx_this_dorsal),0)
            ant_idx_ventralStream = np.concatenate((ant_idx_ventralStream,idx_this_ventral),0)
                    
        if animals[a] in ventralAn:
            ventral_idx = np.concatenate((ventral_idx,idx_this),0)
            ventral_idx_V1 = np.concatenate((ventral_idx_V1,idx_this_V1),0)
            ventral_idx_dorsalStream = np.concatenate((ventral_idx_dorsalStream,idx_this_dorsal),0)
            ventral_idx_ventralStream = np.concatenate((ventral_idx_ventralStream,idx_this_ventral),0)
        elif animals[a] in dorsalAn:
            dorsal_idx = np.concatenate((dorsal_idx,idx_this),0)
            dorsal_idx_V1 = np.concatenate((dorsal_idx_V1,idx_this_V1),0)
            dorsal_idx_dorsalStream = np.concatenate((dorsal_idx_dorsalStream,idx_this_dorsal),0)
            dorsal_idx_ventralStream = np.concatenate((dorsal_idx_ventralStream,idx_this_ventral),0)
               
    
    #%
    bins_peak = np.arange(0,len(ops['azimuths']), 1)

    #%%   
    #Anterior vs posterior
    animalGroups = [anteriorAn, posteriorAn]
    color_anterior = 'blue'
    color_posterior = 'red'
    
    fig = plt.figure(figsize=(ops['mm']*30, ops['mm']*38), constrained_layout=True) 
    ax = fig.add_subplot(1,1,1) 
    
    hist_all, bins = np.histogram(peak0[ant_idx.astype(int)],bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = color_anterior,  histtype ='step',linewidth= 1.3, alpha = 0.8, label = 'Anterior Inj.')

    hist_all, bins = np.histogram(peak0[post_idx.astype(int)],bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = color_posterior,  histtype ='step',linewidth= 1.3, alpha = 0.8, label = 'Posterior Inj.')

    plt.xticks([0,6,12],['-108','0','108'])   
    plt.xlim([-0.1, 12.1])        
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Sound azimuth (\u00b0)', '', mySize=6)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    plt.yticks([0,0.05, 0.1, 0.15], ['0', '5','10','15'])
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestAzimuth_injectionLocation_AP_nice.svg'))

    
    #Dorsal vs ventral
    animalGroups = [ventralAn, dorsalAn]
    color_dorsal = 'green'
    color_ventral = 'darkorange'
    
    fig = plt.figure(figsize=(ops['mm']*30, ops['mm']*38), constrained_layout=True) 
    ax = fig.add_subplot(1,1,1) 
    
    hist_all, bins = np.histogram(peak0[ventral_idx.astype(int)],bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = color_ventral,  histtype ='step',linewidth= 1.3, alpha = 0.8, label = 'Ventral Inj.')

    hist_all, bins = np.histogram(peak0[dorsal_idx.astype(int)],bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = color_dorsal,  histtype ='step',linewidth= 1.3, alpha = 0.8, label = 'Dorsal Inj.')
    
    plt.xticks([0,6,12],['-108','0','108'])   
    plt.xlim([-0.1, 12.1])        
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Sound azimuth (\u00b0)', '', mySize=6)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    plt.yticks([0,0.05, 0.1, 0.15], ['0', '5','10','15'])
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestAzimuth_injectionLocation_DV_nice.svg'))

    #%% Percentage centre-tuned
    leftBorder = 4.4
    rightBorder = 7.4
    
    left_tuned = np.nonzero(peak0 < leftBorder)[0]
    right_tuned = np.nonzero(peak0 > rightBorder)[0]
    centre_tuned0 = np.setdiff1d(np.arange(0,len(peak0)), left_tuned)
    centre_tuned1 = np.setdiff1d(np.arange(0,len(peak0)), right_tuned)
    centre_tuned = np.intersect1d(centre_tuned0, centre_tuned1)
    
    #shuffle
    nShuffles = 1000
    peak0_sh = peak0.copy(); np.random.shuffle(peak0_sh)
    
    seshIdx_unique = np.unique(df0['sessionIdx'])
    prop_left = np.empty(len(seshIdx_unique));prop_left[:] = np.nan
    prop_right = np.empty(len(seshIdx_unique));prop_right[:] = np.nan
    prop_centre = np.empty(len(seshIdx_unique));prop_centre[:] = np.nan

    for s in range(len(seshIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df0['sessionIdx']) == seshIdx_unique[s])[0]
        
        if len(idx_thisSession) <10:
            continue
        left_thisSesh = np.intersect1d(idx_thisSession, left_tuned)
        right_thisSesh = np.intersect1d(idx_thisSession, right_tuned)
        centre_thisSesh = np.intersect1d(idx_thisSession, centre_tuned)
        
        
        prop_left[s] = len(left_thisSesh)/len(idx_thisSession)
        prop_right[s] = len(right_thisSesh)/len(idx_thisSession)
        prop_centre[s] = len(centre_thisSesh)/len(idx_thisSession)

    sessionRef = makeSessionReference(df0)   
    
    nSessions = len(sessionRef['seshAnimal'])
    dorsal_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in dorsalAn for i in range(nSessions)]))[0]
    ventral_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in ventralAn for i in range(nSessions)]))[0]
    posterior_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in posteriorAn for i in range(nSessions)]))[0]
    anterior_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in anteriorAn for i in range(nSessions)]))[0]

    v1_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] =='V1' for i in range(nSessions)]))[0]
    dorsalAreas_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] in ops['dorsal'] for i in range(nSessions)]))[0]
    ventralAreas_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] in ops['ventral'] for i in range(nSessions)]))[0]

    def excludeNans(data):
        notNan = np.nonzero(np.isnan(data) < 0.5)[0]
        return data[notNan]

    
    #%%
    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(prop_centre)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_forTest = pd.DataFrame({'prop_centre': np.array(prop_centre)[these],
                               'area': np.array(sessionRef['seshAreas'])[these], 
                               'animal':  np.array(sessionRef['seshAnimal'])[these], 
                               'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                               'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                               'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})
            
    df_forTest['Inj_DV'] = df_forTest['Inj_DV'] - min(df_forTest['Inj_DV'])  
    df_forTest['Inj_AP'] = abs(df_forTest['Inj_AP'] - max(df_forTest['Inj_AP'])) 

    df_path= os.path.join(ops['outputPath'],'df_forLMM.csv')
    df_forTest.to_csv(df_path)
    formula = 'prop_centre ~ 1 + Inj_DV + (1|animal)'
    # formula = 'meanElevs_green ~ 1 + fitElevs_red + (1|animal)'

    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    
    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_DV'][0][0]
    slope_p = res['Inj_DV'][0][1]
    xVals = np.arange(0,max(df_forTest['Inj_DV']),1)
    yVals = intercept + slope*xVals
     
    r_spearman,p_spearman = scipy.stats.spearmanr(df_forTest['Inj_DV'], df_forTest['prop_centre'])

    #
    fig = plt.figure(figsize =(ops['mm']*37,ops['mm']*35), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_forTest['Inj_DV']), np.array(df_forTest['prop_centre']), c= 'k', s =1)
    x_axis = 'Inj_DV'
    myPlotSettings_splitAxis(fig, ax, 'Percentage centre-\ntuned boutons (%)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(70,0.65,'r: ' + str(np.round(r_spearman,4)) + '\np: ' + str(np.round(p_spearman,4)))
    plt.xticks([0,40,80,120], ['0', '400', '800', '1200'])
    plt.yticks([0,0.2, 0.4, 0.6, 0.8],['0','20','40','60','80'])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   
        
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\propCentre_bySession_againstDV_pos.svg'))

    
    formula = 'prop_centre ~ 1 + Inj_AP + (1|animal)'
    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    r_spearman,p_spearman = scipy.stats.spearmanr(df_forTest['Inj_AP'], df_forTest['prop_centre'])

    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_AP'][0][0]
    slope_p = res['Inj_AP'][0][1]
    xVals = np.arange(0,max(df_forTest['Inj_AP']),1)
    yVals = intercept + slope*xVals
     
    #
    fig = plt.figure(figsize =(ops['mm']*37,ops['mm']*35), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_forTest['Inj_AP']), np.array(df_forTest['prop_centre']), c= 'k', s =1) 
    myPlotSettings_splitAxis(fig, ax, 'Percentage centre-\ntuned boutons (%)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(55,0.65,'r: ' + str(np.round(r_spearman,4)) + '\np: ' + str(np.round(p_spearman,4)))
    plt.xticks([0,50,100], ['0', '500', '1000'])
    plt.yticks([0,0.2, 0.4, 0.6, 0.8],['0','20','40','60','80'])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   
    
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\propCentre_bySession_againstAP_pos.svg'))


def plotElevationDistributions(df,maps,peak,eng,ops,injectionSubset=[]):
    
    noArea_idx = np.nonzero(np.array(df['area']) == 'OUT')[0]
    
    badRoiPosition1 = np.nonzero(np.array(df['x']) ==0)[0]
    badRoiPosition2 = np.nonzero(np.array(df['y']) ==0)[0]
    badRoiPosition = np.unique(np.concatenate((badRoiPosition1,badRoiPosition2),0))
    
    noArea_idx = np.unique(np.concatenate((noArea_idx,badRoiPosition),0))
           
    elevPeak = getElevation_greenAud(df, maps, peak, onlyPeakSide = 1)
    includeIdx_green_elev = np.setdiff1d(np.arange(0,len(df)), noArea_idx)
 
    ventral_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['ventralAnimals'] for i in range(len(df))]))[0]
    dorsal_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['dorsalAnimals'] for i in range(len(df))]))[0]
    anterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['anteriorAnimals'] for i in range(len(df))]))[0]
    posterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['posteriorAnimals'] for i in range(len(df))]))[0]
    
    if len(injectionSubset) > 0:
        if injectionSubset == 'ventral':
            includeIdx_green_elev = np.intersect1d(ventral_idx,  includeIdx_green_elev)
        elif injectionSubset == 'dorsal':
            includeIdx_green_elev = np.intersect1d(dorsal_idx,  includeIdx_green_elev)
        elif injectionSubset == 'anterior':
            includeIdx_green_elev= np.intersect1d(anterior_idx,  includeIdx_green_elev)
        elif injectionSubset == 'posterior':
            includeIdx_green_elev = np.intersect1d(posterior_idx,  includeIdx_green_elev)

    df_green_elev = df.iloc[includeIdx_green_elev]
    df_green_elev['elevPeak'] = elevPeak[includeIdx_green_elev]
    
    data0 = elevPeak[includeIdx_green_elev]
    df0= df_green_elev
    
    #% PLot all together
    fig = plt.figure(figsize=(ops['mm']*60, ops['mm']*60), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    bins_peak = np.array([0,2,4,6])

    hist_all, bins = np.histogram(data0,bins_peak)
    hist_all_norm = hist_all/np.sum(hist_all)
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = '#C8C7C7',  histtype ='stepfilled',alpha = 0.4, orientation='horizontal')
    plt.hist(bins[:-1],bins,weights = hist_all_norm, color = 'k', histtype ='step', linewidth = 0.75, orientation='horizontal')
    plt.ylim([min(bins_peak)-0.1,max(bins_peak)])
    plt.yticks([1, 3, 5],['-36','0','36'])           

    plt.xlim([0,0.5])
    plt.xticks([0,0.25,0.5],['0','25','50'])           
     
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (deg)', 'Percentage of boutons (%)', '', mySize=15)
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   
    
    ##PLot it divided by area
    fig = plt.figure(figsize=(30*ops['mm'], 59*ops['mm']), constrained_layout=False)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3,left=0.16, right=0.95, bottom=0.1, top=0.95)

    
    peaks_byArea =[]
    cnt = 0
    bins_peak = np.array([0,2,4,6])

    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(df0['area']) == ops['areas'][ar])[0]
        
        peaks_this = data0[idx_thisArea]
        peaks_all =  data0
      
        peaks_byArea.append(peaks_this)

        hist_thisArea, bins = np.histogram(peaks_this,bins_peak)
        hist_thisArea_norm = hist_thisArea/np.sum(hist_thisArea)
                    
        hist_all, bins = np.histogram(peaks_all,bins_peak)
        hist_all_norm = hist_all/np.sum(hist_all)
        
        if np.mod(cnt,2) ==0:
            k = 0
        else:
            k=1
        ax = fig.add_subplot(gs[int(np.floor(cnt/2)), k])
       
        #option 2
        plt.hist(bins[:-1],bins,weights = hist_thisArea_norm, color = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],
                 histtype='step',linewidth = 1, alpha = 1,orientation='horizontal')           
        plt.hist(bins[:-1],bins,weights = hist_all_norm, color = '#C8C7C7', histtype='stepfilled', alpha=0.6, orientation='horizontal')
      
        if ops['areas'][ar] == 'POR':
            plt.text(0.45, 0.5, 'POR',  horizontalalignment ='center',weight='normal',color=ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], fontsize=6)
        else:
            plt.text(0.5, 0.5, ops['areas'][ar], horizontalalignment ='center', weight='normal',color=ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], fontsize=6)
       
        plt.ylim([min(bins_peak)-0.1,max(bins_peak)+0.1])
        plt.yticks([1, 3, 5],['-36','0','36'])   
        plt.xlim([0, 0.6])
        
        if cnt ==8 or cnt ==9:
            plt.xticks([0,0.3,0.6],['0','30', '60'])
        else:
            ax.spines["bottom"].set_visible(False)
            plt.xticks([], [])


        myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=6)
        if k==1:
            plt.yticks([1, 3,5], ['', '', ''])
        
        ax.tick_params(axis='both', length=2)  # Change tick length for both axes
        ax.tick_params(axis='y', pad=1)   
        ax.tick_params(axis='x', pad=1)   
        
        cnt +=1  
    
    
def plotElevation_byArea(df,maps,peak,eng,ops,nShuffles, injectionSubset=[]):
    
    noArea_idx = np.nonzero(np.array(df['area']) == 'OUT')[0]
    
    badRoiPosition1 = np.nonzero(np.array(df['x']) ==0)[0]
    badRoiPosition2 = np.nonzero(np.array(df['y']) ==0)[0]
    badRoiPosition = np.unique(np.concatenate((badRoiPosition1,badRoiPosition2),0))
    
    noArea_idx = np.unique(np.concatenate((noArea_idx,badRoiPosition),0))
           
    elevPeak = getElevation_greenAud(df, maps, peak, onlyPeakSide = 1)

    includeIdx_green_elev = np.setdiff1d(np.arange(0,len(df)), noArea_idx)
 
    ventral_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['ventralAnimals'] for i in range(len(df))]))[0]
    dorsal_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['dorsalAnimals'] for i in range(len(df))]))[0]
    anterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['anteriorAnimals'] for i in range(len(df))]))[0]
    posterior_idx =np.nonzero(np.array([df['animal'].iloc[i] in ops['posteriorAnimals'] for i in range(len(df))]))[0]
    
    if len(injectionSubset) > 0:
        if injectionSubset == 'ventral':
            includeIdx_green_elev = np.intersect1d(ventral_idx,  includeIdx_green_elev)
        elif injectionSubset == 'dorsal':
            includeIdx_green_elev = np.intersect1d(dorsal_idx,  includeIdx_green_elev)
        elif injectionSubset == 'anterior':
            includeIdx_green_elev= np.intersect1d(anterior_idx,  includeIdx_green_elev)
        elif injectionSubset == 'posterior':
            includeIdx_green_elev = np.intersect1d(posterior_idx,  includeIdx_green_elev)

    df_green_elev = df.iloc[includeIdx_green_elev]
    df_green_elev['elevPeak'] = elevPeak[includeIdx_green_elev]
    
    
    data0 = elevPeak[includeIdx_green_elev]
    df0= df_green_elev
    
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
    inj_DV, inj_AP= [],[]
    for j in range(len(sessionRef['seshAnimal'])):
        if sessionRef['seshAnimal'][j] in ops['ventralAnimals']:
            inj_DV.append('Ventral')
        elif sessionRef['seshAnimal'][j] in ops['dorsalAnimals']:
            inj_DV.append('Dorsal')
            
        if sessionRef['seshAnimal'][j] in ops['anteriorAnimals']:
            inj_AP.append('Anterior')
        elif sessionRef['seshAnimal'][j] in ops['posteriorAnimals']:
            inj_AP.append('Posterior')

    # peak_azi_bySession = np.array(peak_azi_bySession)
    peakElev_byArea = []
    peakElev_byArea_sh = []

    for ar in range(len(ops['areas'])):  
        idx = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        peak_bySession_this = np.array([peak_elev_bySession[idx[i]] for i in range(len(idx))])
        peak_bySession_this_clean = peak_bySession_this[np.nonzero(np.isnan(peak_bySession_this) < 0.5)[0]]

        peakElev_byArea.append(peak_bySession_this_clean)
        peak_bySession_this_sh = np.array([peak_elev_bySession_sh[idx[i]] for i in range(len(idx))])
        peakElev_byArea_sh.append(peak_bySession_this_sh)
    
    
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_elev_bySession)) <0.5)[0]
    thisIdx = np.intersect1d(notV1,notNan)
   
    df_forTest = pd.DataFrame({'peakElev_bySession': np.array(peak_elev_bySession)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'elev': np.array(sessionRef['seshElev'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx],
                            'Inj_DV': np.array(inj_DV)[thisIdx],
                            'Inj_AP': np.array(inj_AP)[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')
    df_forTest.to_csv(df_path)

    formula = 'peakElev_bySession ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)

    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout =True)  
    ax = fig.add_subplot(1,1,1)
    for ar in range(len(ops['areas'])):
        median_elev0 = np.array([np.nanmedian(peakElev_byArea[ar][i]) for i in range(len(peakElev_byArea[ar]))])
        median_elev = np.nanmedian(median_elev0)
        plt.plot([ar-0.25, ar+0.25], [median_elev,median_elev] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(median_elev0)) 
        plt.scatter(xVals_scatter, np.array(median_elev0), s = 10, facecolors = 'white' , edgecolors =  ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,alpha=0.3,zorder =1)
        
        
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (deg)', '', 'p: ' + str(np.round(p_LMM,3)), mySize=15)
    plt.xticks(np.arange(0, len(ops['areas'])), ops['areas'], rotation =90)
    plt.ylim([0.888888, 3.66666])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
    if p_LMM < 0.05:
        p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(peakElev_byArea, multiComp = 'fdr')
        cnt = 0
        for c in range(len(compIdx)):
            if p_mannWhitney[c] < 0.05:
                pos = compIdx[c].split('_')
                plt.hlines(3.2+cnt, int(pos[0]), int(pos[1]), color = 'k', linewidth =0.5)
                cnt += 0.1
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)  
    # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestElevation_byArea.svg'))

    
    
def plotElevation_byStream(df, peak, maps, eng, ops):
    
    noArea_idx = np.nonzero(np.array(df['area']) == 'OUT')[0]
        
    badRoiPosition1 = np.nonzero(np.array(df['x']) ==0)[0]
    badRoiPosition2 = np.nonzero(np.array(df['y']) ==0)[0]
    badRoiPosition = np.unique(np.concatenate((badRoiPosition1,badRoiPosition2),0))
    
    noArea_idx = np.unique(np.concatenate((noArea_idx,badRoiPosition),0))
           
    elevPeak = getElevation_greenAud(df, maps, peak, onlyPeakSide =1)

    includeIdx_green_elev = np.setdiff1d(np.arange(0,len(df)), noArea_idx)

    df_green_elev = df.iloc[includeIdx_green_elev]
    df_green_elev['elevPeak'] = elevPeak[includeIdx_green_elev]
    
    data0 = elevPeak[includeIdx_green_elev]
    df0= df_green_elev
    
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
        
    sessionRef = makeSessionReference(df0)   
 
    meanElev_byGroup = []
    for ar in range(len(ops['groups'])):
        idx_thisArea = np.nonzero(np.array(sessionRef['seshStream']) == ops['groups'][ar])[0]
        
        med_this = np.array([peak_elev_bySession[idx_thisArea[i]] for i in range(len(idx_thisArea))])
        idx =np.nonzero(np.isnan(med_this) < 0.05)[0]
        med_this = med_this[idx]
        meanElev_byGroup.append(med_this)
        
    notNan = np.nonzero(np.isnan(np.array(peak_elev_bySession)) <0.5)[0]
    thisIdx =notNan
    df_forTest = pd.DataFrame({'peakElev_bySession': np.array(peak_elev_bySession)[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'elev': np.array(sessionRef['seshElev'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx],
                            'Inj_DV': np.array(sessionRef['pos_DV'])[thisIdx],
                            'Inj_AP': np.array(sessionRef['pos_AP'])[thisIdx],
                            'prop_ventral': np.array(sessionRef['prop_ventral'])[thisIdx]})
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)
        
    formula = 'peakElev_bySession ~ stream + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)

    #%%
    fig = plt.figure(figsize=(ops['mm']*80, ops['mm']*80), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)    
    for ar in range(1,len(ops['groups'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.1,size = len(meanElev_byGroup[ar])) 
        plt.plot([ar-0.3,ar+0.3], [np.nanmedian(meanElev_byGroup[ar]),np.nanmedian(meanElev_byGroup[ar])], linewidth = 2, c = ops['colors_groups'][ar],zorder = 2)
        plt.scatter(xVals_scatter, np.array(meanElev_byGroup[ar]), s = 10, facecolors = 'white' , edgecolors = ops['colors_groups'][ar], linewidths =0.5,alpha =0.3,zorder = 1)
                   
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (deg)', '', 'p: ' + str(np.round(p_LMM,3)), mySize=15)  
    plt.xticks(np.arange(1,len(ops['groups'])), ['Ventral','Dorsal' ])
    plt.ylim([0.888888, 2+(30/18)])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
    ax.tick_params(axis='x', pad=1)  
    ax.tick_params(axis='y', pad=1)   

    #%% Plot distribution by stream
    t = []
    for i in range(len(df0)):
        if df0['area'].iloc[i] in ops['dorsal']:
            t.append('Dorsal')
        elif df0['area'].iloc[i] in ops['ventral']:
            t.append('Ventral')
        elif df0['area'].iloc[i] == 'V1':
            t.append('V1')
        else:
            t.append('')

    df0['streamIdx'] = t
    
    groups = ['Ventral', 'Dorsal']
    elev_byStream =[]
    for g in range(len(groups)):
        these = np.nonzero(np.array(df0['streamIdx']) == groups[g])[0]
        
        elev_byStream.append(data0[these])
        
    fig = plt.figure(figsize=(ops['mm']*80, ops['mm']*80), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    bins_peak = np.array([0,2,4,6])

    colors = [ops['myColorsDict']['HVA_colors']['ventral'],ops['myColorsDict']['HVA_colors']['dorsal']]
    for g in range(len(groups)):
        hist_all, bins = np.histogram(elev_byStream[g],bins_peak)
        hist_all_norm = hist_all/np.sum(hist_all)
        plt.hist(bins[:-1],bins,weights = hist_all_norm, color = colors[g], histtype ='stepfilled',alpha = 0.1, orientation='horizontal')
        plt.hist(bins[:-1],bins,weights = hist_all_norm, color = colors[g], histtype ='step', linewidth = 0.75, orientation='horizontal')
        
    plt.ylim([min(bins_peak)-0.1,max(bins_peak)])
    plt.yticks([1, 3, 5],['-36','0','36'])           

    plt.xlim([0,0.5])
    plt.xticks([0,0.25,0.5],['0','25','50'])           
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (deg)', 'Percentage of boutons (%)', '', mySize=15)
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   

def plotBestElevation_onMap(fig, df, maps, peak, ref, ref2, map_V1, b=250):
  
    elev = getElevation_greenAud(df, maps, peak)
    
    df['peak'] = elev
    
    df = df[~df['x'].isnull()]
    df = df[~df['y'].isnull()]
    df = df[df['x'] != 0]
    df = df[df['y'] != 0]
    df = df[df['area'] != 'OUT']
    
    binned_map = makeSpatialBinnedMap(ref,spatialBin =b) 
    binned_values_map = makeMeanValue_bySpatialBin_v2(df, binned_map,thresh =5,  varName = 'peak', mask = 'none', V1_mask = map_V1)
    
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

    plt.imshow(binned_map_adj,cmap=colors,vmin =1.166, vmax=2.8333, alpha =0.95)
   
    plt.yticks([],[])
    plt.xticks([],[])
    plt.axis('off')
    # if 'freq' in dataType:
    cbar = plt.colorbar(ticks = [1.166,2,2.8333],fraction=0.038, pad=0.04)
    cbar.ax.set_yticklabels(['-15', '0', '15'],fontsize=15)

def exploreInjectionLocation_elevation(df, peak,maps, ops,eng):
    
    nShuffles =100
    noArea_idx = np.nonzero(np.array(df['area']) == 'OUT')[0]
    
    badRoiPosition1 = np.nonzero(np.array(df['x']) ==0)[0]
    badRoiPosition2 = np.nonzero(np.array(df['y']) ==0)[0]
    badRoiPosition = np.unique(np.concatenate((badRoiPosition1,badRoiPosition2),0))
    
    noArea_idx = np.unique(np.concatenate((noArea_idx,badRoiPosition),0))
           
    elevPeak = getElevation_greenAud(df, maps, peak, onlyPeakSide = 1)

    includeIdx_green_elev = np.setdiff1d(np.arange(0,len(df)), noArea_idx)

    df_green_elev = df.iloc[includeIdx_green_elev]
    df_green_elev['elevPeak'] = elevPeak[includeIdx_green_elev]
    
    peak0 = elevPeak[includeIdx_green_elev]
    df0= df_green_elev
    
    
    animals_thisDataset = df0['animal'].unique()
    
    ventralAn = np.intersect1d(animals_thisDataset, ops['ventralAnimals'])        #[109,113,128,149,154,166,168]
    dorsalAn = np.intersect1d(animals_thisDataset, ops['dorsalAnimals'])  #[107,112,131,132,151,153,170,171,178]
   
    anteriorAn = np.intersect1d(animals_thisDataset,ops['anteriorAnimals'])         #[113,128,151,154,170,178]
    posteriorAn = np.intersect1d(animals_thisDataset,ops['posteriorAnimals'])     #[107,109,112,131,132,149,153,166,168,171]
    
    t = []
    for i in range(len(df0)):
        if df0['area'].iloc[i] in ops['dorsal']:
            t.append('Dorsal')
        elif df0['area'].iloc[i] in ops['ventral']:
            t.append('Ventral')
        elif df0['area'].iloc[i] == 'V1':
            t.append('V1')
        else:
            t.append('')                        
    df0['streamIdx'] = t
    
    animals = df0['animal'].unique()
    post_idx = np.zeros(0,); ant_idx = np.zeros(0,)
    dorsal_idx = np.zeros(0,); ventral_idx = np.zeros(0,)
    post_idx_V1= np.zeros(0,); ant_idx_V1 = np.zeros(0,)
    dorsal_idx_V1 = np.zeros(0,); ventral_idx_V1 = np.zeros(0,)
    post_idx_dorsalStream= np.zeros(0,); ant_idx_dorsalStream = np.zeros(0,)
    dorsal_idx_dorsalStream = np.zeros(0,); ventral_idx_dorsalStream = np.zeros(0,)
    post_idx_ventralStream= np.zeros(0,); ant_idx_ventralStream = np.zeros(0,)
    dorsal_idx_ventralStream = np.zeros(0,); ventral_idx_ventralStream = np.zeros(0,)
    for a in range(len(animals)):
        idx_this = np.nonzero(np.array(df0['animal']) == animals[a])[0]
        
        idx_V1 = np.nonzero(np.array(df0['streamIdx']) == 'V1')[0]
        idx_ventral = np.nonzero(np.array(df0['streamIdx']) == 'Ventral')[0]
        idx_dorsal = np.nonzero(np.array(df0['streamIdx']) == 'Dorsal')[0]

        idx_this_V1 = np.intersect1d(idx_this,idx_V1)
        idx_this_dorsal = np.intersect1d(idx_this,idx_dorsal)
        idx_this_ventral = np.intersect1d(idx_this,idx_ventral)

        if animals[a] in posteriorAn:
            post_idx = np.concatenate((post_idx,idx_this),0)
            post_idx_V1 = np.concatenate((post_idx_V1,idx_this_V1),0)
            post_idx_dorsalStream = np.concatenate((post_idx_dorsalStream,idx_this_dorsal),0)
            post_idx_ventralStream = np.concatenate((post_idx_ventralStream,idx_this_ventral),0)
        elif animals[a] in anteriorAn:
            ant_idx = np.concatenate((ant_idx,idx_this),0)
            ant_idx_V1 = np.concatenate((ant_idx_V1,idx_this_V1),0)
            ant_idx_dorsalStream = np.concatenate((ant_idx_dorsalStream,idx_this_dorsal),0)
            ant_idx_ventralStream = np.concatenate((ant_idx_ventralStream,idx_this_ventral),0)
                    
        if animals[a] in ventralAn:
            ventral_idx = np.concatenate((ventral_idx,idx_this),0)
            ventral_idx_V1 = np.concatenate((ventral_idx_V1,idx_this_V1),0)
            ventral_idx_dorsalStream = np.concatenate((ventral_idx_dorsalStream,idx_this_dorsal),0)
            ventral_idx_ventralStream = np.concatenate((ventral_idx_ventralStream,idx_this_ventral),0)
        elif animals[a] in dorsalAn:
            dorsal_idx = np.concatenate((dorsal_idx,idx_this),0)
            dorsal_idx_V1 = np.concatenate((dorsal_idx_V1,idx_this_V1),0)
            dorsal_idx_dorsalStream = np.concatenate((dorsal_idx_dorsalStream,idx_this_dorsal),0)
            dorsal_idx_ventralStream = np.concatenate((dorsal_idx_ventralStream,idx_this_ventral),0)
    
    #%%
    peak_elev_bySession = []
    sessionIdx= np.unique(np.array(df0['sessionIdx']))
    for s in range(len(sessionIdx)):
        idx_thisSession = np.nonzero(np.array(df0['sessionIdx']) == sessionIdx[s])[0]
              
      
        peak_elev = peak0[idx_thisSession]
        
        if len(idx_thisSession) < 10:
            peak_elev_bySession.append(np.nan)
        else:
            peak_elev_bySession.append(np.nanmean(peak_elev))
        
        sh = np.zeros((len(idx_thisSession), nShuffles))
        for n in range(nShuffles):
            idx = np.random.choice(np.arange(len(peak0)), len(idx_thisSession))
            sh[:,n] = peak0[idx]
       

    sessionRef = makeSessionReference(df0)   
    
    nSessions = len(sessionRef['seshAnimal'])
    dorsal_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in dorsalAn for i in range(nSessions)]))[0]
    ventral_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in ventralAn for i in range(nSessions)]))[0]
    posterior_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in posteriorAn for i in range(nSessions)]))[0]
    anterior_idx = np.nonzero(np.array([sessionRef['seshAnimal'][i] in anteriorAn for i in range(nSessions)]))[0]

    v1_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] =='V1' for i in range(nSessions)]))[0]
    dorsalAreas_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] in ops['dorsal'] for i in range(nSessions)]))[0]
    ventralAreas_idx = np.nonzero(np.array([sessionRef['seshAreas'][i] in ops['ventral'] for i in range(nSessions)]))[0]

    def excludeNans(data):
        notNan = np.nonzero(np.isnan(data) < 0.5)[0]
        return data[notNan]
    
    peak_elev_bySession = np.array(peak_elev_bySession)

    bestElev_dorsalInj_all = excludeNans(peak_elev_bySession[dorsal_idx]); 
    bestElev_ventralInj_all = excludeNans(peak_elev_bySession[ventral_idx])
    bestElev_anteriorInj_all = excludeNans(peak_elev_bySession[anterior_idx])
    bestElev_posteriorInj_all = excludeNans(peak_elev_bySession[posterior_idx])
    
    #%%
    color_anterior = 'blue'
    color_posterior = 'red'
    color_dorsal = 'green'
    color_ventral = 'darkorange'
    
    #%% for paper, just all
    fig = plt.figure(figsize=(ops['mm']*29,ops['mm']*31), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    #all
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(bestElev_ventralInj_all)) 
    plt.plot([-0.25,+0.25], [np.nanmedian(bestElev_ventralInj_all),np.nanmedian(bestElev_ventralInj_all)], linewidth = 2, c = color_ventral,alpha=1,zorder = 2)
    plt.scatter(xVals_scatter, np.array(bestElev_ventralInj_all), s = 10, facecolors ='white' , edgecolor = color_ventral,zorder = 1,linewidth=0.5, alpha =0.3)
    xVals_scatter = np.random.normal(loc =1,scale =0.05,size = len(bestElev_dorsalInj_all)) 
    plt.plot([1-0.25,1+0.25], [np.nanmedian(bestElev_dorsalInj_all),np.nanmedian(bestElev_dorsalInj_all)], linewidth = 2, c = color_dorsal,alpha=1,zorder = 2)
    plt.scatter(xVals_scatter, np.array(bestElev_dorsalInj_all), s = 10, facecolors ='white' , edgecolor = color_dorsal,zorder = 1,linewidth=0.5, alpha =0.3)
    U, p = stats.mannwhitneyu(bestElev_dorsalInj_all, bestElev_ventralInj_all)
    plt.text(0.3, 3.5, 'p=' + str(np.round(p,3)), fontsize=6)
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (\u00b0)', '', '', mySize=6)
    plt.xticks([0,1], ['Ventral', 'Dorsal'])
    plt.ylim([0.888888, 3.66666])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestElevation_DV_medians.svg'))

    
    fig = plt.figure(figsize=(ops['mm']*29,ops['mm']*31), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    #all
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(bestElev_posteriorInj_all)) 
    plt.plot([0-0.25,0+0.25], [np.nanmedian(bestElev_posteriorInj_all),np.nanmedian(bestElev_posteriorInj_all)], linewidth = 2, c = color_posterior,alpha=1,zorder = 2)
    plt.scatter(xVals_scatter, np.array(bestElev_posteriorInj_all), s = 10, facecolors ='white' , edgecolor = color_posterior,zorder = 1,linewidth=0.5, alpha =0.3)
    xVals_scatter = np.random.normal(loc =1,scale =0.05,size = len(bestElev_anteriorInj_all)) 
    plt.plot([1-0.25,1+0.25], [np.nanmedian(bestElev_anteriorInj_all),np.nanmedian(bestElev_anteriorInj_all)], linewidth = 2, c = color_anterior,alpha=1,zorder = 2)
    plt.scatter(xVals_scatter, np.array(bestElev_anteriorInj_all), s = 10, facecolors ='white' , edgecolor = color_anterior,zorder = 1,linewidth=0.5, alpha =0.3)
    U, p = stats.mannwhitneyu(bestElev_anteriorInj_all, bestElev_posteriorInj_all)
    plt.text(0.3, 3.5, 'p=' + str(np.round(p,3)), fontsize=6)
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (\u00b0)', '', '', mySize=6)
    plt.xticks([0,1], ['Posterior', 'Anterior'])
    plt.ylim([0.888888, 3.66666])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestElevation_AP_medians.svg'))

    #%% scatter against position
    
    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(peak_elev_bySession)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_forTest = pd.DataFrame({'peakElev': np.array(peak_elev_bySession)[these],
                               'area': np.array(sessionRef['seshAreas'])[these], 
                               'animal':  np.array(sessionRef['seshAnimal'])[these], 
                               'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                               'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                               'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})
            
    df_forTest['Inj_DV'] = df_forTest['Inj_DV'] - min(df_forTest['Inj_DV'])  
    df_forTest['Inj_AP'] = abs(df_forTest['Inj_AP'] - max(df_forTest['Inj_AP'])) 

    df_path= os.path.join(ops['outputPath'],'df_forLMM.csv')
    df_forTest.to_csv(df_path)
    formula = 'peakElev ~ 1 + Inj_DV + (1|animal)'

    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_DV'][0][0]
    slope_p = res['Inj_DV'][0][1]
    xVals = np.arange(0,max(df_forTest['Inj_DV']),1)
    yVals = intercept + slope*xVals
     
    r_spearman,p_spearman = scipy.stats.spearmanr(df_forTest['Inj_DV'], df_forTest['peakElev'])

    #
    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*36,ops['mm']*35), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_forTest['Inj_DV']), np.array(df_forTest['peakElev']), c= 'k', s =1)
    x_axis = 'Inj_DV'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(df_forTest[x_axis]), max(df_forTest[x_axis]), len(fitLine))
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'gray',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 1, linestyle ='dashed') 
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (\u00b0)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(70,3.1,'r: ' + str(np.round(r_spearman,4)) + '\np: ' + str(np.round(p_spearman,4)))
    plt.xticks([0,40,80,120], ['0', '400', '800', '1200'])
    # plt.yticks([0,0.2, 0.4, 0.6, 0.8],['0','20','40','60','80'])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1) 
    plt.ylim([0.888888, 3.66666])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
        
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\peakElev_bySession_againstDV_pos.svg'))

    
    formula = 'peakElev ~ 1 + Inj_AP + (1|animal)'

    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_AP'][0][0]
    slope_p = res['Inj_AP'][0][1]
    xVals = np.arange(0,max(df_forTest['Inj_AP']),1)
    yVals = intercept + slope*xVals
     
    r_spearman,p_spearman = scipy.stats.spearmanr(df_forTest['Inj_AP'], df_forTest['peakElev'])
    #
    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*36,ops['mm']*35), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_forTest['Inj_AP']), np.array(df_forTest['peakElev']), c= 'k', s =1)
    x_axis = 'Inj_AP'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(df_forTest[x_axis]), max(df_forTest[x_axis]), len(fitLine))
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'gray',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 1, linestyle ='dashed') 
    myPlotSettings_splitAxis(fig, ax, 'Best sound elevation (\u00b0)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(1,3.1,'r: ' + str(np.round(r_spearman,4)) + '\np: ' + str(np.round(p_spearman,4)))
    plt.xticks([0,50,100], ['0', '500', '1000'])
    # plt.yticks([0,0.2, 0.4, 0.6, 0.8],['0','20','40','60','80'])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1) 
    plt.ylim([0.888888, 3.66666])
    plt.yticks([2-(20/18),2-(10/18),2, 2 + (10/18), 2+(20/18), 2+(30/18)], ['-20', '-10', '0', '10', '20', '30'])
        
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\peakElev_bySession_againstAP_pos.svg'))
 
    
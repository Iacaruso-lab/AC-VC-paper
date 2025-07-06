# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:56:21 2025

@author: egeaa
"""
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import stats
import statsmodels.stats.multitest
from sklearn.neighbors import KernelDensity
import os
#import pims
from tqdm import tqdm
#import sys
import pandas as pd
import seaborn as sns
#import imageio
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from analysis_utils import *

#%% ----------------------------------------------------------------------------------
def quantifySignificance_frequencies(df,eng,ops):
    # df = df_freq_peak
    areas = ops['areas']
    #%
    nSDs = 1
    ####
    #get responsive ones 
    green_aud_resp_idx = np.load(os.path.join(ops['dataPath'],'frequencies_dataset','responsive_idx_freq_boutons.npy')) 
    df_green_aud_resp = df.iloc[green_aud_resp_idx]

    green_aud_prop_resp = makeProportions_bySession_v2(df_green_aud_resp, df) #includes responsive to both
    green_aud_prop_resp_median = np.nanmedian(green_aud_prop_resp)
    
    ######
    #divide responsive sessions by area
    areas_green_aud = asignAreaToSession(df, policy='mostRois')
    green_aud_resp_byArea = divideSessionsByArea(green_aud_prop_resp, areas, areas_green_aud)
    
    ##########
    #get selective
    green_aud_selective_freq = np.load(os.path.join(ops['dataPath'],'frequencies_dataset', 'selective_freq_idx_boutons.npy'))
    green_aud_selective_vol = np.load(os.path.join(ops['dataPath'],'frequencies_dataset', 'selective_vol_idx_boutons.npy'))

    df_sel_freq = df.iloc[green_aud_selective_freq]
    df_sel_vol = df.iloc[green_aud_selective_vol]
   
    ##########
    #make the proportions selective
    green_aud_prop_sel_freq = makeProportions_bySession_v2(df_sel_freq, df_green_aud_resp)
    green_aud_prop_sel_median_freq = np.nanmedian(green_aud_prop_sel_freq)

    green_aud_prop_sel_vol = makeProportions_bySession_v2(df_sel_vol, df_green_aud_resp)
    green_aud_prop_sel_median_vol = np.nanmedian(green_aud_prop_sel_vol)

    ############
    #assign area to session
    areas_green_aud = asignAreaToSession(df, policy='mostRois')
    green_aud_sel_byArea_freq = divideSessionsByArea(green_aud_prop_sel_freq, areas, areas_green_aud)
    green_aud_sel_byArea_vol = divideSessionsByArea(green_aud_prop_sel_vol, areas, areas_green_aud)
  
    # Save for LMM
    sessionRef = makeSessionReference(df_green_aud_resp)

    notOut = np.nonzero(np.array(areas_green_aud['areas']) != 'OUT')[0]
    df_props_forTest = pd.DataFrame({'proportion_resp': np.array(green_aud_prop_resp)[notOut],
                                     'proportion_sel_freq': np.array(green_aud_prop_sel_freq)[notOut],
                                      'proportion_sel_vol': np.array(green_aud_prop_sel_vol)[notOut],
                                    'area': np.array(areas_green_aud['areas'])[notOut], 
                                    'animal':  np.array(areas_green_aud['animals'])[notOut],
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[notOut],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[notOut],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[notOut]})#
    df_path = os.path.join(ops['outputPath'], 'prop_freq_forLMM.csv')
    df_props_forTest.to_csv(df_path)
    
    
    #%%
    #Plot freq. sel alone, for main figure
    meanLineWidth =0.25
    meanLineWidth_small =0.25
    fig = plt.figure(figsize=(ops['mm']*80, ops['mm']*80),constrained_layout=True)
    ax0 = fig.add_subplot(1,1,1)
    plt.plot([- meanLineWidth, meanLineWidth], [green_aud_prop_sel_median_freq,green_aud_prop_sel_median_freq],linewidth = 2,c = 'k',zorder =2, alpha=1)     
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(green_aud_prop_sel_freq)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_freq), s = 10, facecolors = 'white' , edgecolors ='k', linewidths =0.5,zorder = 1,alpha =0.3)
   
    data = green_aud_sel_byArea_freq
    # ref =  green_aud_prop_sel_median_azi
    data_medians_byArea = np.array([np.nanmedian(data[j]) for j in range(len(data))])       
    # pVals = doWilcoxon_againstRef(data, ref, multiComp = 'hs')
    
    formula = 'proportion_sel_freq ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
    
    ylim_sel = [-0.05, 1.05]       
   
    plt.vlines(1,0, ylim_sel[-1], linewidth = 0.5, color = 'gray',zorder =0)
    for i in range(len(areas)):
        plt.plot([i-meanLineWidth_small+2,i+meanLineWidth_small+2], [data_medians_byArea[i],data_medians_byArea[i]] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][i]],zorder = 2, alpha =1)
        xVals_scatter = np.random.normal(loc =i+2,scale =0.05,size = len(data[i])) 
        plt.scatter(xVals_scatter, data[i], s = 10, facecolors = 'white' , edgecolors =  ops['myColorsDict']['HVA_colors'][ops['areas'][i]], linewidths =0.5,zorder = 1, alpha =0.3) 
        if p_LMM < 0.05:
            p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(data, multiComp = 'fdr')
            cnt = 0
            for c in range(len(compIdx)):
                if p_mannWhitney[c] < 0.05:
                    pos = compIdx[c].split('_')
                    plt.hlines(ylim_sel[-1] - cnt, int(pos[0] + 2), int(pos[1] +2), colors = 'k')                    
                    cnt +=0.01
       
                
    plt.ylim(ylim_sel)
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig,ax0,'Percentage of boutons (%)','',str(p_LMM), mySize=15)  
    plt.xticks([0,2,3,4,5,6,7,8,9,10,11], np.append('All',areas), rotation =90)
    plt.xlim([-1, 12])
    ax0.tick_params(axis='y', pad=1)   
    ax0.tick_params(axis='x', pad=1) 
    # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\propfrequencytuned_byArea.svg'))

    #%% Proportions for all sessions together
    ylim_resp = [-0.05,1.05]
        
    meanLineWidth = 0.5
    meanLineWidth_small = 0.3
    color_gray_dashedline = 'gray'
   
    fig = plt.figure(figsize=(50*ops['mm'], 80*ops['mm']), constrained_layout=True)

    # green aud, responsive
    ax0 = fig.add_subplot(1,1,1)
    xVals_scatter = np.random.normal(loc =0,scale =0.08,size = len(green_aud_prop_resp)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_resp), s = 8, facecolors = 'white' , edgecolors = 'k', linewidths =0.5,zorder = 1)
    plt.plot([- 0.4, 0.4], [green_aud_prop_resp_median,green_aud_prop_resp_median],linewidth = 3,c = 'k',zorder =2)     
    plt.ylim(ylim_resp)
   
    myPlotSettings_splitAxis(fig,ax0,'Percentage of boutons (%)','','', mySize=15)  
    plt.xticks([0], ['Responsive'])
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1],['0','50', '100'] )
    
    fig = plt.figure(figsize=(80*ops['mm'], 80*ops['mm']), constrained_layout=True)
    
    # green aud, responsive
    ax0 = fig.add_subplot(1,1,1)
    # green aud, selective
    xVals_scatter = np.random.normal(loc =0,scale =0.08,size = len(green_aud_prop_sel_freq)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_freq), s = 8, facecolors = 'white' , edgecolors = 'k', linewidths =0.5)
    plt.plot([- meanLineWidth_small, meanLineWidth_small], [green_aud_prop_sel_median_freq,green_aud_prop_sel_median_freq],linewidth = 3,c = 'k')     

    xVals_scatter = np.random.normal(loc =1,scale =0.08,size = len(green_aud_prop_sel_vol)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_vol), s = 8, facecolors = 'white' , edgecolors = 'k', linewidths =0.5)
    plt.plot([- meanLineWidth_small +1,meanLineWidth_small+1], [green_aud_prop_sel_median_vol,green_aud_prop_sel_median_vol],linewidth = 3,c = 'k')     

    plt.ylim(ylim_sel)
    plt.yticks([0,0.5,1], ['0', '50', '100'])
    myPlotSettings_splitAxis(fig,ax0,'Percentage of boutons (%)','','',mySize=15)  
    plt.xticks([0,1], ['Frequency', 'Sound intensity'], rotation = 0)
    
    # fig.savefig(os.path.join(pathStart, 'home', 'shared', 'Alex_analysis_camp', 'Thesis_figures','Frequencies','Prop_sel_freq.svg'))
    
    
#%% -------------------------------------------------------------------------------------------------------

def plotFreqDistribution_byArea(df, data, ops, fig):
    from matplotlib import gridspec
    
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.25,left=0.1, right=0.95, bottom=0.1, top=0.92)
    
    x_interp= np.linspace(0, 10.1, 1000)
    freq_byArea = []
    cnt = 0
    for ar in range(len(ops['areas'])):
        
        if np.mod(cnt,2) ==0:
            k = 0
        else:
            k=1
        ax = fig.add_subplot(gs[int(np.floor(cnt/2)), k])
        
        idx_thisArea = np.nonzero(np.array(df['area']) == ops['areas'][ar])[0]

        data_thisArea = np.array(data[idx_thisArea])
        data_all = data.copy()
        
        freq_byArea.append(np.squeeze(data_thisArea))

        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')               #density of 0.7 is nice
        kde.fit(data_thisArea.reshape(-1,1))
        logprob = kde.score_samples(x_interp.reshape(-1,1))
        plt.plot(x_interp, np.exp(logprob), alpha=1, linewidth = 2, color = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]]) 
        # plt.fill_between(x_interp, np.exp(logprob), alpha=0.3, color =  ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])
        
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')               #density of 0.7 is nice
        kde.fit(data_all.reshape(-1,1))
        logprob = kde.score_samples(x_interp.reshape(-1,1))
        plt.fill_between(x_interp, np.exp(logprob), alpha=1, color = '#C8C6C6')

        if k ==0:
            plt.yticks([0,0.3], ['0','30'])
        else:
            plt.yticks([0,0.3], ['',''])

        if ar ==8:
            myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Best tone frequency (kHz)', '', mySize=15)
            plt.xticks(np.arange(0,11,2),ops['freq_names'],rotation =0)
        elif ar == 9:
            myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=15)
            plt.xticks(np.arange(0,11,2),ops['freq_names'],rotation =0)
        else:
            ax.spines["bottom"].set_visible(False)
            plt.xticks([],[])
            myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=15)
  
        if ops['areas'][ar] == 'POR':
            plt.text(5.4, 0.26, 'POR',  fontsize=15, horizontalalignment ='center')
        else:
            plt.text(5.4, 0.26, ops['areas'][ar], fontsize=15, horizontalalignment ='center')
       
        plt.xlim([0,10])
        plt.ylim([0,0.3])
        ax.tick_params(axis='y', pad=1)   
        ax.tick_params(axis='x', pad=1)   
        ax.tick_params(axis='both', length=2)  # Change tick length for both axes

        cnt += 1
    
    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    x_interp= np.linspace(0, 10.1, 1000)

    kde = KernelDensity(bandwidth=0.7, kernel='gaussian')               
    kde.fit(data.reshape(-1,1))
    logprob = kde.score_samples(x_interp.reshape(-1,1))
    plt.plot(x_interp, np.exp(logprob), alpha=1, linewidth =1, color = 'k') 
    plt.fill_between(x_interp, np.exp(logprob), alpha=0.4, color = 'grey')
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', 'Best tone frequency (kHz)', '', mySize=15)
    plt.xticks(np.arange(0,11,2),ops['freq_names'] )
    plt.xlim([0,10])
    plt.ylim([0, 0.2])
    plt.yticks([0,0.1,0.2], ['0','10', '20'])
    
    
def plotHierarchicalBootstrap_FV(name, ops):
    
    path = os.path.join(ops['dataPath'], 'frequencies_dataset',name)
    files = os.listdir(path)
    nBatch = len(np.nonzero(np.array([name in files[i] for i in range(len(files))]) > 0.5)[0])
    
    median_dist_mat,p_difference_quantiles,sigLevels_quantiles,groups = getBootstrapResult(path,name, nBatch,ops,doMultiCorr=1)
    
    colors = sns.color_palette('binary', n_colors =100)
    myColors = [colors[10], colors[40], colors[60], colors[80]]

    xLabels = ops['areas'].copy()
    
    fig = plt.figure(figsize=(ops['mm']*70, ops['mm']*70),constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    # plt.title('Ipsi, 5 an')
      
    plt.imshow(median_dist_mat, cmap = 'Oranges',vmin =0, vmax =2)
    cbar = plt.colorbar(ticks = [0, 0.5, 1, 1.5, 2],fraction = 0.05, pad = 0.05)
    cbar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=7)
    
    plt.imshow(sigLevels_quantiles, cmap = LinearSegmentedColormap.from_list('myMap', myColors, N=4), vmax = 3) 
    cbar = plt.colorbar(ticks = [0.4, 1.15, 1.9, 2.62], fraction = 0.05, pad = 0.07)
    cbar.ax.set_yticklabels(['N.S.', 'p < 0.05', 'p < 0.01', 'p < 0.001'], fontsize=7)
 
    plt.xticks(np.arange(0,len(ops['areas'])), xLabels, rotation =90,fontsize=7)
    plt.yticks(np.arange(0,len(ops['areas'])), ops['areas'],fontsize=7)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\hierarchicalBootstrap_freq_medianComparison_boutons.svg'))
    
    
def plotFrequency_bySession(df, data,ops, eng,fig):
    
    sessionRef = makeSessionReference(df)
   
    sessionIdx_unique = np.array(sessionRef['seshIdx'])
    meanFreqs = []
    for j in range(len(sessionIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df['sessionIdx']) == sessionIdx_unique[j])[0]
        if len(idx_thisSession) > 10:
            meanFreqs.append(np.nanmedian(data[idx_thisSession]))
        else:
            meanFreqs.append(np.nan)

    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(meanFreqs)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_freqs_forTest = pd.DataFrame({'freqMedian': np.array(meanFreqs)[these],
                                    'area': np.array(sessionRef['seshAreas'])[these], 
                                    'animal':  np.array(sessionRef['seshAnimal'])[these], 
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})
    
    df_path = os.path.join(ops['outputPath'], 'freq_forLMM.csv')
    df_freqs_forTest.to_csv(df_path)
    
    formula = 'freqMedian ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
    
    #New shuffles
    nShuffles = 1000
    N = 200
    freq_sh = []
    for n in range(nShuffles):
        rand = np.random.choice(np.arange(0,len(df)), N, replace =True)
        
        freq_sh.append(np.nanmedian(data[rand]))
        
    freq_sh = np.mean(freq_sh)
        
    meanFreqs = np.array(meanFreqs) 
    areaIdx = np.array(sessionRef['seshAreas'])
    meanFreqs_byArea = []
    for ar in range(len(ops['areas'])):
        these = np.nonzero(areaIdx == ops['areas'][ar])
        vals_bySession_this = meanFreqs[these]
        vals_bySession_this_clean = vals_bySession_this[np.nonzero(np.isnan(vals_bySession_this) < 0.5)[0]]

        meanFreqs_byArea.append(vals_bySession_this_clean)
  
    data = meanFreqs_byArea
    data_means_byArea = np.array([np.nanmedian(data[j]) for j in range(len(data))])       
   
    ax = fig.add_subplot(1,1,1)
    p_wilcox_byArea =[]
    for i in range(len(ops['areas'])):
       
        plt.plot([i-0.25,i+0.25], [data_means_byArea[i],data_means_byArea[i]] , linewidth = 2, c =ops['myColorsDict']['HVA_colors'][ops['areas'][i]],zorder = 2)
        xVals_scatter = np.random.normal(loc =i,scale =0.05,size = len(data[i])) 
        plt.scatter(xVals_scatter, data[i], s = 20, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][i]], linewidths =0.5,zorder = 1, alpha=0.3) 
       
        U,p_wilcox = stats.wilcoxon(data[i] - freq_sh) 
        p_wilcox_byArea.append(p_wilcox)
        #print(str(p_wilcox))
    
    pVals_adj = statsmodels.stats.multitest.multipletests(np.array(p_wilcox_byArea), method='fdr_bh')[1]            

    for ar in range(len(ops['areas'])):
        if pVals_adj[ar] < 0.05:
            plt.text(ar,9, '*', fontsize=10)
            
        
    plt.yticks([0,2,4,6,8,10], ['2','4','8', '16', '32','64']) 
    plt.ylim([0,10])
    myPlotSettings_splitAxis(fig,ax,'Median best frequency (kHz)','',str(p_LMM), mySize=15) 

    if p_LMM < 0.05:
        pVals_adj_mannU, compIdx = doMannWhitneyU_forBoxplots(meanFreqs_byArea, multiComp = 'hs')
        cnt=0     
        for j in range(len(pVals_adj_mannU)):
            if pVals_adj_mannU[j] < 0.05:
                plt.hlines(6-cnt, int(compIdx[j][0]), int(compIdx[j][2]), color = 'k')
                cnt = cnt + 0.2
                
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation=90)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)
    
def plotFreqDistribution_byStream(df, data, ops, eng):
    
    t = []
    for i in range(len(df)):
        if df['area'].iloc[i] in ops['dorsal']:
            t.append('Dorsal')
        elif df['area'].iloc[i] in ops['ventral']:
            t.append('Ventral')
        elif df['area'].iloc[i] == 'V1':
            t.append('V1')
        else:
            t.append('')
                          
    df['streamIdx'] = t
       
    #Plot it by Session
    df['spline_peak'] = data
    sessionRef = makeSessionReference(df, varName = ['spline_peak'])
            
    freq_byStream = []
    for ar in range(len(ops['groups'])):
        idx_thisArea = np.nonzero(np.array(sessionRef['seshStream']) == ops['groups'][ar])[0]
        
        freq_this = np.array(sessionRef['myVar'])[idx_thisArea]
        idx =np.nonzero(np.isnan(freq_this) < 0.05)[0]
        freq_this = freq_this[idx]
        freq_byStream.append(freq_this)
            
    notV1 = np.nonzero(np.array(sessionRef['seshAreas']) != 'V1')[0]
    notNan = np.nonzero(np.isnan(np.array(sessionRef['myVar'])) <0.5)[0]
    thisIdx = np.intersect1d(notV1,notNan)
   
    df_forTest = pd.DataFrame({'freq_bySession': np.array(sessionRef['myVar'])[thisIdx],                                    
                            'area': np.array(sessionRef['seshAreas'])[thisIdx],
                            'stream': np.array(sessionRef['seshStream'])[thisIdx],
                            'elev': np.array(sessionRef['seshElev'])[thisIdx],
                            'animal':  np.array(sessionRef['seshAnimal'])[thisIdx],
                            'Inj_DV': np.array(sessionRef['pos_DV'])[thisIdx],
                            'Inj_AP': np.array(sessionRef['pos_AP'])[thisIdx],
                            'prop_ventral': np.array(sessionRef['prop_ventral'])[thisIdx]
                            })
    
    df_path = os.path.join(ops['outputPath'], 'df_forTest.csv')

    df_forTest.to_csv(df_path)
    
    formula = 'freq_bySession ~ stream + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
           
    #%%
    fig = plt.figure(figsize=(70*ops['mm'], 100*ops['mm']), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    for ar in range(1,len(ops['groups'])):
        xVals_scatter = np.random.normal(loc =ar-1,scale =0.05,size = len(freq_byStream[ar])) 
        plt.plot([ar-1.3,ar+0.3 -1], [np.nanmedian(freq_byStream[ar]),np.nanmedian(freq_byStream[ar])], linewidth = 3, c = ops['colors_groups'][ar],zorder = 2)
        plt.scatter(xVals_scatter, np.array(freq_byStream[ar]), s = 20, facecolors = 'white' , edgecolors =  ops['colors_groups'][ar], linewidths =1,zorder = 1,alpha=0.3)
           
  
    myPlotSettings_splitAxis(fig, ax, 'Best frequency (kHz)', '', 'p: ' + str(p_LMM),mySize=15)  
    plt.xticks([0,1], ['Ventral','Dorsal' ], rotation = 45, horizontalalignment='right')
    plt.ylim([2,8])
    plt.yticks([2,4,6,8], ['4', '8', '16', '32'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

   # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestFrequency_twoStreams.svg'))

def plotTuningWidth_byArea(df, data, ops,eng):
    # data = width
    sessionRef = makeSessionReference(df)
   
    sessionIdx_unique = np.array(sessionRef['seshIdx'])
    meanFreqs = []
    for j in range(len(sessionIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df['sessionIdx']) == sessionIdx_unique[j])[0]
        if len(idx_thisSession) > 10:
            meanFreqs.append(np.nanmedian(data[idx_thisSession]))
        else:
            meanFreqs.append(np.nan)

    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(meanFreqs)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_freqs_forTest = pd.DataFrame({'freqMedian': np.array(meanFreqs)[these],
                                    'area': np.array(sessionRef['seshAreas'])[these], 
                                    'animal':  np.array(sessionRef['seshAreas'])[these],
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})#
    
    df_path = os.path.join(ops['outputPath'], 'freq_forLMM.csv')
    df_freqs_forTest.to_csv(df_path)
    
    formula = 'freqMedian ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
           
    meanFreqs = np.array(meanFreqs) 
    areaIdx = np.array(sessionRef['seshAreas'])
    meanFreqs_byArea = []
    for ar in range(len(ops['areas'])):
        these = np.nonzero(areaIdx == ops['areas'][ar])
        vals_bySession_this = meanFreqs[these]/2
        vals_bySession_this_clean = vals_bySession_this[np.nonzero(np.isnan(vals_bySession_this) < 0.5)[0]]

        meanFreqs_byArea.append(vals_bySession_this_clean)
  
    data = meanFreqs_byArea
    data_means_byArea = np.array([np.nanmedian(data[j]) for j in range(len(data))])       

    #%%
    fig = plt.figure(figsize=(100*ops['mm'], 100*ops['mm']), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    p_wilcox_byArea =[]
    for i in range(len(ops['areas'])):
       
        plt.plot([i-0.25,i+0.25], [data_means_byArea[i],data_means_byArea[i]] , linewidth = 2, c =ops['myColorsDict']['HVA_colors'][ops['areas'][i]],zorder = 2)
        xVals_scatter = np.random.normal(loc =i,scale =0.05,size = len(data[i])) 
        plt.scatter(xVals_scatter, data[i], s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][i]], linewidths =0.5,zorder = 1, alpha=0.3) 
       
    myPlotSettings_splitAxis(fig,ax,'Tuning width (octaves)','','p: ' + str(p_LMM), mySize=15) 
    
    if p_LMM < 0.05:
        pVals_adj_mannU, compIdx = doMannWhitneyU_forBoxplots(meanFreqs_byArea, multiComp = 'fdr')
        cnt=0     
        for j in range(len(pVals_adj_mannU)):
            if pVals_adj_mannU[j] < 0.05:
                plt.hlines(2-cnt, int(compIdx[j][0]), int(compIdx[j][2]), color = 'k', linewidth=0.5)
                cnt = cnt + 0.2
                
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation=90)
    plt.yticks([0.5,1,1.5, 2], ['0.5', '1', '1.5', '2'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   


def plotProportionComplexFreq(df,includeIdx, doubleIdx, ops, eng):
    
    #shuffle
    nShuffles = 1000
    seshIdx_unique = np.unique(df['sessionIdx'])
    prop_double = np.empty(len(seshIdx_unique));prop_double[:] = np.nan
    for s in range(len(seshIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df['sessionIdx']) == seshIdx_unique[s])[0]
        
        include_thisSession = np.intersect1d(includeIdx, idx_thisSession)
        double_thisSesh = np.intersect1d(include_thisSession, doubleIdx)
        
        if len(include_thisSession) < 20:
            continue
        
        prop_double[s] = len(double_thisSesh)/len(include_thisSession)

    sessionRef = makeSessionReference(df)   
    
    prop_double_byArea = []
    for ar in range(len(ops['areas'])):
        idx_thisArea = np.nonzero(np.array(sessionRef['seshAreas']) == ops['areas'][ar])[0]
        
        prop_this = np.array(prop_double[idx_thisArea])
        idx =np.nonzero(np.isnan(prop_this) < 0.05)[0]
        prop_this = prop_this[idx]
        prop_double_byArea.append(prop_this)
        
    
    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(prop_double)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_props_forTest = pd.DataFrame({'prop_double_bySession': np.array(prop_double)[these],
                                    'area': np.array(sessionRef['seshAreas'])[these], 
                                    'animal':  np.array(sessionRef['seshAnimal'])[these],
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})#
    df_path = os.path.join(ops['outputPath'], 'freq_forLMM.csv')
    df_props_forTest.to_csv(df_path)
    
    formula = 'prop_double_bySession ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)

    #plot it
    fig = plt.figure(figsize = (ops['mm']*100,ops['mm']*100), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
   
    pVals_adj_mannU, compIdx = doMannWhitneyU_forBoxplots(prop_double_byArea, multiComp = 'fdr')
  
    for ar in range(len(ops['areas'])):
        xVals_scatter = np.random.normal(loc =ar,scale =0.05,size = len(prop_double_byArea[ar])) 
        plt.plot([ar-0.25,ar+0.25], [np.nanmedian(prop_double_byArea[ar]),np.nanmedian(prop_double_byArea[ar])], linewidth = 2, c = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]],zorder = 2)
        plt.scatter(xVals_scatter, np.array(prop_double_byArea[ar]), s = 7, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]], linewidths =0.5,zorder = 1,alpha =0.3)
              
    myPlotSettings_splitAxis(fig, ax, 'Percentage of complex-tuned boutons (%)', '', 'p: ' + str(np.round(p_LMM,5)), mySize=15)  
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'],  rotation = 90)
    plt.ylim([0, 0.5])
    plt.yticks([0,0.25, 0.5], ['0', '25', '50'])
    if p_LMM< 0.05:
        cnt = 0
        for j in range(len(pVals_adj_mannU)):
            if pVals_adj_mannU[j] < 0.05:
                plt.hlines(0.9, int(compIdx[j][0]), int(compIdx[j][2]), color = 'k')
                cnt += 0.05
                
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

 
def plotSparsityIdx_byArea(df, data, ops, eng):
    sessionRef = makeSessionReference(df)
   
    sessionIdx_unique = np.array(sessionRef['seshIdx'])
    meanFreqs = []
    for j in range(len(sessionIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df['sessionIdx']) == sessionIdx_unique[j])[0]
        if len(idx_thisSession) > 10:
            meanFreqs.append(np.nanmedian(data[idx_thisSession]))
        else:
            meanFreqs.append(np.nan)

    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(meanFreqs)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_freqs_forTest = pd.DataFrame({'freqMedian': np.array(meanFreqs)[these],
                                    'area': np.array(sessionRef['seshAreas'])[these], 
                                    'animal':  np.array(sessionRef['seshAreas'])[these],
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})
    
    df_path = os.path.join(ops['outputPath'], 'freq_forLMM.csv')
    df_freqs_forTest.to_csv(df_path)
    
    formula = 'freqMedian ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)

    meanFreqs = np.array(meanFreqs) 
    areaIdx = np.array(sessionRef['seshAreas'])
    meanFreqs_byArea = []
    for ar in range(len(ops['areas'])):
        these = np.nonzero(areaIdx == ops['areas'][ar])
        vals_bySession_this = meanFreqs[these]
        vals_bySession_this_clean = vals_bySession_this[np.nonzero(np.isnan(vals_bySession_this) < 0.5)[0]]

        meanFreqs_byArea.append(vals_bySession_this_clean)
  
    data0 = meanFreqs_byArea
    data_means_byArea = np.array([np.nanmedian(data0[j]) for j in range(len(data0))])       
   
    fig = plt.figure(figsize=(100*ops['mm'], 100*ops['mm']), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    p_wilcox_byArea =[]
    for i in range(len(ops['areas'])):
       
        plt.plot([i-0.25,i+0.25], [data_means_byArea[i],data_means_byArea[i]] , linewidth = 2, c =ops['myColorsDict']['HVA_colors'][ops['areas'][i]],zorder = 2)
        xVals_scatter = np.random.normal(loc =i,scale =0.05,size = len(data0[i])) 
        plt.scatter(xVals_scatter, data0[i], s = 7, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][ops['areas'][i]], linewidths =0.5,zorder = 1, alpha=0.3) 
        
    plt.ylim([0.5,0.9])
    plt.yticks([0.5, 0.7, 0.9], ['0.5', '0.7', '0.9'])
    # plt.xlim(-1, 5.5)               
    myPlotSettings_splitAxis(fig,ax,'Sparsity Index','','p: ' + str(np.round(p_LMM,2)), mySize=15) 
    
    if p_LMM < 0.05:
        pVals_adj_mannU, compIdx = doMannWhitneyU_forBoxplots(meanFreqs_byArea, multiComp = 'fdr')
        cnt=0     
        for j in range(len(pVals_adj_mannU)):
            if pVals_adj_mannU[j] < 0.05:
                plt.hlines(6-cnt, int(compIdx[j][0]), int(compIdx[j][2]), color = 'k')
                cnt = cnt + 0.2
                
    plt.xticks(np.arange(0,len(ops['areas'])), ops['areas'], rotation=90)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    
def plotSignalCorrelation_byArea_FV(df, ops, computeMatrix =1):               

    areas = ['P', 'POR', 'LI', 'LM', 'AL', 'RL', 'A', 'AM', 'PM','V1'] 

    name = 'signal_corr_frequencies_resp_motorSub'
    #pairwise signal correlations are a very large NxN matrix. to avoid loading the whole matrix at once, it is loaded in by area, in chuncks
    if computeMatrix: #takes a while
        nBatch =40
        corrMatrix_byArea0 = np.zeros((len(areas), len(areas)))
        for ar0 in range(len(areas)):
            idx_thisArea = np.nonzero(np.array(df['area']) == areas[ar0])[0]


            corr_byArea = [[] for _ in range(len(areas))]
            sumPrev = 0
            for i in tqdm(range(nBatch+1)):
                # corrs = np.load(os.path.join(outputPath, 'signal_corr_coliseum_2dim' + str(i) + '.npy'))
                corrs = np.load(os.path.join(ops['dataPath'],'frequencies_dataset', 'signal_correlations', name, name + '_' + str(i) + '.npy'))

                nRois = corrs.shape[0]
                idx_batch = np.arange(0,nRois) + sumPrev

                idx_thisArea_batch = np.intersect1d(idx_thisArea, idx_batch)
                corrs_thisArea = corrs[idx_thisArea_batch-sumPrev,:]

                for ar1 in range(len(areas)):
                    idx_area1 = np.nonzero(np.array(df['area']) == areas[ar1])[0]

                    these_corr = corrs_thisArea[:,idx_area1].reshape(-1,1)
                    notNanIdx = np.nonzero(~np.isnan(these_corr)==1)[0]
                    these_corr = these_corr[notNanIdx]

                    corr_byArea[ar1].append(these_corr)

                sumPrev = nRois + sumPrev
            for j in range(len(areas)):
                for k in range(len(corr_byArea[j])-1):
                    if k ==0:
                        this = corr_byArea[j][k]
                    else:
                        this = np.concatenate((this,corr_byArea[j][k]),0)
                corrMatrix_byArea0[ar0, j] = np.nanmean(this)
    else:
        corrMatrix_byArea0 = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset','signal_correlations', 'signalCorr_matrix_freqs_resp_motorSub.npy'))

        
    fig = plt.figure(figsize= (ops['mm']*100,ops['mm']*100), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
   
    plt.imshow(corrMatrix_byArea0, cmap = 'Reds', vmin=0, vmax =0.04)
   
    plt.xticks(np.arange(0,len(areas)), areas, rotation = 90, fontsize=15)
    plt.yticks(np.arange(0,len(areas)), areas,fontsize=15)
    cbar = plt.colorbar(fraction = 0.05,ticks=[0, 0.02,0.04])
    cbar.ax.set_yticklabels([0, 0.02,0.04],fontsize=10)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.75)   
        
    plt.title('Signal correlation')
    
    #%% ###########################################################################################################
    boot_df = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset','signal_correlations','signal_corr_frequencies_resp_motorSub', 
                                   'corrs_frequencies_resp_byStream_allData.npy'), allow_pickle=True).item()

    diff_dd_dv = boot_df['mean_dd'] - boot_df['mean_dv']
    diff_vv_dv = boot_df['mean_vv'] - boot_df['mean_dv']

    p_dd_dv = np.float64(getBootstrapDiffP(diff_dd_dv))*2    
    p_vv_dv = np.float64(getBootstrapDiffP(diff_vv_dv))*2      
    
    color_dorsal = ops['myColorsDict']['HVA_colors']['AM']
    color_ventral = ops['myColorsDict']['HVA_colors']['POR']
    color_mixed = '#A160A4'
    
    fig = plt.figure(figsize=(ops['mm']*26, ops['mm']*39), constrained_layout=True)
    ax= fig.add_subplot(1,1,1)
    plt.bar(0,np.median(boot_df['mean_vv']), color= color_ventral, alpha = 0.7, edgecolor=color_ventral)
    plt.vlines(0, np.percentile(boot_df['mean_vv'],2.5), np.percentile(boot_df['mean_vv'],97.5), color='k', linewidth=0.5)
    plt.bar(1,np.median(boot_df['mean_dd']), color= color_dorsal, alpha = 0.7, edgecolor=color_dorsal)
    plt.vlines(1, np.percentile(boot_df['mean_dd'],2.5), np.percentile(boot_df['mean_dd'],97.5), color='k', linewidth=0.5)
    plt.bar(2,np.median(boot_df['mean_dv']), color= color_mixed, alpha = 0.7, edgecolor=color_mixed)
    plt.vlines(2, np.percentile(boot_df['mean_dv'],2.5), np.percentile(boot_df['mean_dv'],97.5), color='k', linewidth=0.5)


    plt.hlines(0,-0.5,2.5,color = 'dimgray', linewidth=0.5, linestyle='dashed')
    myPlotSettings_splitAxis(fig, ax, 'Signal correlation', '', 'p_ven-ven_vs_ven-dor: ' + str(p_vv_dv) + '\np_dor-dor_vs_ven-dor: ' + str(p_dd_dv), mySize=6)
    plt.xticks([0,1,2], ['Ventral-Ventral','Dorsal-Dorsal', 'Ventral-Dorsal'], rotation = 45, horizontalalignment='right')
    plt.ylim([-0.0005, 0.02])
    plt.yticks([ 0, 0.01, 0.02],['0', '0.01', '0.02'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    # plt.text(0,np.percentile(boot_df['mean_vv'],97.5), p_vv_dv, color = 'k')
    # plt.text(0.5,np.percentile(boot_df['mean_dd'],97.5), p_dd_dv, color = 'k')

    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\signalCorr_byStream_frequencies_resp_barplots_hierarchical_allData.svg'))


def plotFrequency_againstSource(df, data, ops, eng):
 
    sessionRef = makeSessionReference(df)
   
    sessionIdx_unique = np.array(sessionRef['seshIdx'])
    meanFreqs = []
    for j in range(len(sessionIdx_unique)):
        idx_thisSession = np.nonzero(np.array(df['sessionIdx']) == sessionIdx_unique[j])[0]
        if len(idx_thisSession) > 10:
            meanFreqs.append(np.nanmedian(data[idx_thisSession]))
        else:
            meanFreqs.append(np.nan)
            
    notOut = np.nonzero(np.array(sessionRef['seshAreas']) != 'OUT')[0]
    notNan = np.nonzero(np.isnan(np.array(meanFreqs)) <0.5)[0]
    these = np.intersect1d(notOut,notNan)
    df_freqs_forTest = pd.DataFrame({'freqMedian': np.array(meanFreqs)[these],
                                    'area': np.array(sessionRef['seshAreas'])[these], 
                                    'animal':  np.array(sessionRef['seshAnimal'])[these], 
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[these],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[these],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[these]})
            
    df_freqs_forTest['Inj_DV'] = df_freqs_forTest['Inj_DV'] - min(df_freqs_forTest['Inj_DV'])  
    df_freqs_forTest['Inj_AP'] = abs(df_freqs_forTest['Inj_AP'] - max(df_freqs_forTest['Inj_AP']))  

 #%%
    df_path= os.path.join(ops['outputPath'],'df_freqs_forLMM.csv')
    df_freqs_forTest.to_csv(df_path)
    formula = 'freqMedian ~ 1 + Inj_DV + (1|animal)'
    # formula = 'meanElevs_green ~ 1 + fitElevs_red + (1|animal)'

    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    
    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_DV'][0][0]
    slope_p = res['Inj_DV'][0][1]
    xVals = np.arange(0,max(df_freqs_forTest['Inj_DV']),1)
    yVals = intercept + slope*xVals
     
    #
    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*43,ops['mm']*40), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_freqs_forTest['Inj_DV']), np.array(df_freqs_forTest['freqMedian']), c= 'k', s =1)
    x_axis = 'Inj_DV'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(df_freqs_forTest[x_axis]), max(df_freqs_forTest[x_axis]), len(fitLine))
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'gray',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 1, linestyle ='dashed') 
    myPlotSettings_splitAxis(fig, ax, 'Best frequency (kHz)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(3,8,'p: ' + str(np.round(slope_p,4)))
    plt.xticks([0,50,100,150], ['0', '500', '1000', '1500'])
    plt.yticks([2,4,6,8],[4,8,16,32])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   
        
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestFreq_bySession_againstDV_pos.svg'))

    df_path= os.path.join(ops['outputPath'],'df_freqs_forLMM.csv')
    df_freqs_forTest.to_csv(df_path)
    formula = 'freqMedian ~ 1 + Inj_AP + (1|animal)'
    savePath = os.path.join(ops['outputPath'], 'LMM_green.mat')
    
    #run LMM and load results
    res, fitLines, fitCI = eng.linearMixedModel_fromPython(df_path, formula,savePath, nargout=3) 

    mat_file = scipy.io.loadmat(savePath)   
    res = getDict_fromMatlabStruct(mat_file, 'res')
    
    intercept = res['Intercept'][0][0] # from matlab LMM 
    slope = res['Inj_AP'][0][0]
    slope_p = res['Inj_AP'][0][1]
    xVals = np.arange(0,max(df_freqs_forTest['Inj_AP']),1)
    yVals = intercept + slope*xVals
     
    #
    #this is the nice one
    fig = plt.figure(figsize =(ops['mm']*43,ops['mm']*40), constrained_layout = True)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(np.array(df_freqs_forTest['Inj_AP']), np.array(df_freqs_forTest['freqMedian']), c= 'k', s =1)
    x_axis = 'Inj_AP'
    fitLine = np.array(fitLines[x_axis])
    fitLine_down = np.array(fitCI[x_axis])[:,0]
    fitLine_up = np.array(fitCI[x_axis])[:,1]
    xVals = np.linspace(min(df_freqs_forTest[x_axis]), max(df_freqs_forTest[x_axis]), len(fitLine))
    plt.fill_between(xVals, fitLine_up, fitLine_down, facecolor = 'gray',alpha = 0.3)
    plt.plot(xVals, fitLine, c = 'k', linewidth = 1, linestyle ='dashed') 
    myPlotSettings_splitAxis(fig, ax, 'Best frequency (kHz)', 'Injection centre position (\u03BCm)','', mySize=6)
    plt.text(3,8,'p: ' + str(np.round(slope_p,4)))
    plt.xticks([0,50,100], ['0', '500', '1000'])
    plt.yticks([2,4,6,8],[4,8,16,32])
    ax.tick_params(axis='y', pad=1)  
    ax.tick_params(axis='x', pad=1)   
    
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\bestFreq_bySession_againstAP_pos.svg'))
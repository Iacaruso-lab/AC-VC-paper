# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:56:21 2025

@author: egeaa
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import stats
import statsmodels.stats.multitest
from sklearn.neighbors import KernelDensity
import os
#import pims
from tqdm import tqdm
import sys
import pandas as pd
import seaborn as sns
import imageio
import scipy
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
    # green_aud_resp_idx, _ = G.applySignificanceTests(df, modality = 'green_aud',extra = '', sig_policy = 'responsive_maxWilcoxon', nSDs = nSDs, alpha = 0.05, capped = 1, GLM=1)
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
    # green_aud_selective_freq, _ = G.applySignificanceTests(df, modality = 'green_aud',extra = '', dimension = 'long', sig_policy = 'selective_maxWilcoxon', nSDs = nSDs, alpha = 0.05, capped = 1, GLM=1)
    # green_aud_selective_vol, _ = G.applySignificanceTests(df, modality = 'green_aud',extra = '', dimension = 'short', sig_policy = 'selective_maxWilcoxon', nSDs = nSDs, alpha = 0.05, capped = 1, GLM=1)
    # # green_aud_selective_int, _ = G.applySignificanceTests(df, modality = 'green_aud',extra = '', dimension = 'int', sig_policy = 'selective_maxWilcoxon', nSDs = nSDs, alpha = 0.05, capped =1, GLM=1)
    # np.save(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\data_axonsPaper\\frequencies_dataset\\selective_freq_idx_axons.npy'), green_aud_selective_freq)
    # np.save(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\data_axonsPaper\\frequencies_dataset\\selective_vol_idx_axons.npy'), green_aud_selective_vol)
    
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
    # upper = [np.percentile(shuffled_prop_sel_azi[j,:], 97.5) for j in range(len(areas))]
    # lower = [np.percentile(shuffled_prop_sel_azi[j,:], 2.5) for j in range(len(areas))]
    # t,p_kruskal = stats.kruskal(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9])
    # plt.hlines(prop_sel_azi_sh, 1,len(areas) +2, linestyle = 'dashed', linewidth =1, color ='k')
         
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
        # t, p_signRank = stats.wilcoxon(data[i]-prop_sel_azi_sh)
        # if p_signRank < 0.05:
        #     plt.text(i+2,ylim_resp[-1] -0.1, '*', fontsize=10)
        #     print(str(p_signRank))
        
                
    plt.ylim(ylim_sel)
    # plt.yticks(yTickValues_resp)
    # ax0.xaxis.set_ticklabels([])
    # ax0.xaxis.set_ticks([])
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
    # plt.yticks(yTickValues_resp)
    # ax0.xaxis.set_ticklabels([])
    # ax0.xaxis.set_ticks([])
    myPlotSettings_splitAxis(fig,ax0,'Percentage of boutons (%)','','', mySize=15)  
    # plt.gca().set_xticklabels([])
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

    # xVals_scatter = np.random.normal(loc =2,scale =0.08,size = len(green_aud_prop_sel_int)) 
    # plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_int), s = 8, facecolors = 'white' , edgecolors = 'k', linewidths =0.5)
    # plt.plot([- meanLineWidth_small+2, meanLineWidth_small+2], [green_aud_prop_sel_median_int,green_aud_prop_sel_median_int],linewidth = 3,c = 'k')     

    plt.ylim(ylim_sel)
    plt.yticks([0,0.5,1], ['0', '50', '100'])
    # plt.yticks(yTickValues_resp)
    # ax0.xaxis.set_ticklabels([])
    # ax0.xaxis.set_ticks([])
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
        # ax = fig.add_subplot(gs[ar, 0])

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

       
        # plt.plot(x_interp, np.exp(logprob), alpha=1, linewidth = 0.25, color = 'k') 
        # plt.scatter(np.median(data_all), 0.29, marker ='v', s= 30, color = 'k')
        # plt.scatter(np.median(data_thisArea), 0.29, marker ='v', s= 30, color = ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])

        if k ==0:
            plt.yticks([0,0.3], ['0','30'])
        else:
            plt.yticks([0,0.3], ['',''])

        # U, p = stats.mannwhitneyu(data_thisArea,data_all)
        # plt.vlines(np.nanmedian(data_thisArea), 0, 0.25, color='r')
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

        # if p < 0.05: 
        #     plt.text(5,0.25,'*', fontsize=10)
        if ops['areas'][ar] == 'POR':
            plt.text(5.4, 0.26, 'POR',  fontsize=15, horizontalalignment ='center')
            # plt.text(5.4, 0.26, ops['areas'][ar], horizontalalignment ='center', weight='bold')
        else:
            plt.text(5.4, 0.26, ops['areas'][ar], fontsize=15, horizontalalignment ='center')
        # plt.text(0,0.23,  'n: ' + str(len(data_thisArea)), fontsize=5)
        # plt.text(6,0.23,  ops['areas'][ar], fontsize=5, weight ='bold')
        
        plt.xlim([0,10])
        plt.ylim([0,0.3])
        ax.tick_params(axis='y', pad=1)   
        ax.tick_params(axis='x', pad=1)   
        ax.tick_params(axis='both', length=2)  # Change tick length for both axes

        # plt.legend()
        cnt += 1
    
    
    # fig, ax = plt.subplots(figsize = (12,5))
    # vp = ax.violinplot(freq_byArea, positions = np.arange(0,len(ops['areas'])), vert =True, bw_method = 0.3,widths=0.7, showmedians = False, showextrema = False)

    # for ar in range(len(ops['areas'])):
    #     body = vp['bodies'][ar]
    #     body.set_facecolor(ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])
    #     body.set_edgecolor(ops['myColorsDict']['HVA_colors'][ops['areas'][ar]])
    #     body.set_linewidth(1)
    #     body.set_alpha(0.8)
    # plt.xticks(np.arange(0,len(areas)), areas)
    
    
    # plt.yticks(np.arange(0,13,2), ['-108', '-72', '-36', '0', '36', '72', '108'])
    # myPlotSettings(fig,ax, 'Sound source azimuth','Area','') 
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
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

def quantifySignificance_coliseum_v2(df,eng, ops):
    
    areas = ops['areas']
    #####################################################
    #get responsive ones 
    # green_aud_resp_idx, _ = self.applySignificanceTests(df, modality = modality,extra = extra, 
    #                                                       sig_policy = 'responsive_maxWilcoxon', nSDs = 1, alpha = 0.05, capped = 0, GLM=1)
    green_aud_resp_idx= np.load(os.path.join(ops['dataPath'], 'locations_dataset','responsive_idx_coliseum_boutons.npy'))

    df_green_aud_resp = df.iloc[green_aud_resp_idx]

    green_aud_prop_resp = makeProportions_bySession_v2(df_green_aud_resp, df) #includes responsive to both
    green_aud_prop_resp_median = np.nanmedian(green_aud_prop_resp)

    #####################################################
    #divide responsive sessions by area
    areas_green_aud = asignAreaToSession(df, policy='mostRois')
    green_aud_resp_byArea = divideSessionsByArea(green_aud_prop_resp, areas, areas_green_aud)
    #############################################
    #get selective
    # green_aud_selective_azi, _ = self.applySignificanceTests(df_green_aud, modality = modality,extra = extra, dimension = 'long', sig_policy = 'selective_maxWilcoxon', nSDs = 1, alpha = 0.05, capped = 0, GLM=1)
    # green_aud_selective_elev, _ = self.applySignificanceTests(df_green_aud, modality = modality,extra = extra, dimension = 'short',sig_policy = 'selective_maxWilcoxon', nSDs = 1, alpha = 0.05, capped = 0, GLM=1)
    # green_aud_selective_int, _= self.applySignificanceTests(df_green_aud, modality = modality,extra = extra, dimension = 'int', sig_policy = 'selective_maxWilcoxon', nSDs = 1, alpha = 0.05, capped = 0, GLM=1)

    # green_aud_selective_all = np.unique(np.concatenate((green_aud_selective_azi, green_aud_selective_elev, green_aud_selective_int),0))
    #load index of selective boutons
    green_aud_selective_azi= np.load(os.path.join(ops['dataPath'],'locations_dataset', 'selective_green_aud_azimuth_maxWilcoxon_a1.npy'))
    green_aud_selective_elev =np.load(os.path.join(ops['dataPath'], 'locations_dataset','selective_green_aud_elevation_maxWilcoxon_a1.npy'))
    
    df_sel_azi = df.iloc[green_aud_selective_azi]
    df_sel_elev = df.iloc[green_aud_selective_elev]
    # df_sel_int = df_green_aud.iloc[green_aud_selective_int]
    # df_sel_all = df_green_aud.iloc[green_aud_selective_all]

    #####################################
    #make the proportions selective
    green_aud_prop_sel_azi = makeProportions_bySession_v2(df_sel_azi, df_green_aud_resp,thresh=10)
    green_aud_prop_sel_median_azi = np.nanmedian(green_aud_prop_sel_azi)

    green_aud_prop_sel_elev = makeProportions_bySession_v2(df_sel_elev, df_green_aud_resp)
    green_aud_prop_sel_median_elev = np.nanmedian(green_aud_prop_sel_elev)

#     green_aud_prop_sel_int = makeProportions_bySession_v2(df_sel_int, df_green_aud_resp)
#     green_aud_prop_sel_median_int = np.nanmedian(green_aud_prop_sel_int)

#     green_aud_prop_sel_all = makeProportions_bySession_v2(df_sel_all, df_green_aud_resp)
#     green_aud_prop_sel_median_all = np.nanmedian(green_aud_prop_sel_all)

    #########################################
    #assign area to session
    areas_green_aud = asignAreaToSession(df_green_aud_resp, policy='mostRois')
    # noneResp = np.nonzero(np.array(green_aud_prop_resp) == 0)[0]
    # areas_green_aud['areas'] = np.delete(areas_green_aud['areas'],noneResp)
    # areas_green_aud['sessionIdx'] = np.delete(areas_green_aud['sessionIdx'],noneResp)

    green_aud_sel_byArea_azi = divideSessionsByArea(green_aud_prop_sel_azi, areas, areas_green_aud)
    green_aud_sel_byArea_elev = divideSessionsByArea(green_aud_prop_sel_elev, areas, areas_green_aud)
    # green_aud_sel_byArea_int = divideSessionsByArea(green_aud_prop_sel_int, areas, areas_green_aud)
    # green_aud_sel_byArea_all = divideSessionsByArea(green_aud_prop_sel_all, areas, areas_green_aud)

    #################################################
    #do shuffles
   
    # Save for LMM
    sessionRef = makeSessionReference(df)
    notOut = np.nonzero(np.array(areas_green_aud['areas']) != 'OUT')[0]
    df_props_forTest = pd.DataFrame({'proportion_resp': np.array(green_aud_prop_resp)[notOut],
                                     'proportion_sel_azi': np.array(green_aud_prop_sel_azi)[notOut],
                                     'proportion_sel_elev': np.array(green_aud_prop_sel_elev)[notOut],
                                    # 'proportion_sel_int': np.array(green_aud_prop_sel_int)[notOut],
                                    'area': np.array(areas_green_aud['areas'])[notOut], 
                                    'animal':  np.array(areas_green_aud['animals'])[notOut],
                                    'Inj_DV': np.array(sessionRef['pos_DV'])[notOut],
                                    'Inj_AP': np.array(sessionRef['pos_AP'])[notOut],
                                    'prop_ventral': np.array(sessionRef['prop_ventral'])[notOut]})#

    df_path = os.path.join(ops['outputPath'], 'prop_freq_forLMM.csv')
    df_props_forTest.to_csv(df_path)

    meanLineWidth = 0.5
    meanLineWidth_small = 0.3


    #%%
    ylim_sel = [-0.05, 1.05]
    #Plot azi and elev
    fig = plt.figure(figsize=(ops['mm']*40, ops['mm']*52), constrained_layout=True)
    ax = fig.add_subplot(2,1,1)
    plt.plot([- meanLineWidth, meanLineWidth], [green_aud_prop_sel_median_azi,green_aud_prop_sel_median_azi],linewidth = 2,c = 'k',zorder =2)     
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(green_aud_prop_sel_azi)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_azi), s = 10, facecolors = 'white' , edgecolors ='k', linewidths =0.5,zorder = 1, alpha=0.3)

    data = green_aud_sel_byArea_azi
    data0 = []
    for i in range(len(data)):
        this = np.nonzero(np.isnan(data[i]) < 0.5)[0]
        data0.append(data[i][this])
    data = data0   
    # ref =  green_aud_prop_sel_median_azi
    data_medians_byArea = np.array([np.nanmedian(data[j]) for j in range(len(data))])       
    # pVals = doWilcoxon_againstRef(data, ref, multiComp = 'hs')

    formula = 'proportion_sel_azi ~ area +  Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
    print(p_LMM)
    # upper = [np.percentile(shuffled_prop_sel_azi[j,:], 97.5) for j in range(len(areas))]
    # lower = [np.percentile(shuffled_prop_sel_azi[j,:], 2.5) for j in range(len(areas))]
    # t,p_kruskal = stats.kruskal(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9])
    # plt.hlines(prop_sel_azi_sh, 1,len(areas) +2, linestyle = 'dashed', linewidth =1, color ='k')
    #p_mannWhitney0, compIdx = doMannWhitneyU_forBoxplots(data, multiComp = 'fdr')

    plt.vlines(1,0, 1, linewidth = 0.5, color = 'gray',zorder =0)
    for i in range(len(areas)):
        plt.plot([i-meanLineWidth_small+2,i+meanLineWidth_small+2], [data_medians_byArea[i],data_medians_byArea[i]] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][areas[i]],zorder = 2)
        xVals_scatter = np.random.normal(loc =i+2,scale =0.05,size = len(data[i])) 
        plt.scatter(xVals_scatter, data[i], s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][areas[i]], linewidths =0.5,zorder = 1, alpha=0.3) 

        # p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(data, multiComp = 'hs')
    if p_LMM < 0.05:
        cnt = 0
        p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(data, multiComp = 'fdr')

        for c in range(len(compIdx)):
            if p_mannWhitney[c] < 0.05:
                pos = compIdx[c].split('_')
                plt.hlines(ylim_sel[-1] - cnt, int(pos[0])+2, int(pos[1])+2, colors = 'k', linewidth=0.5)                    
                cnt +=0.01
       # t, p_signRank = stats.wilcoxon(data[i]-prop_sel_azi_sh)
       # print(str(p_signRank))

        # if p_signRank < 0.00511:
        #     plt.text(i+2,ylim_sel[-1] -0.1, '*', fontsize=10)

    plt.ylim(ylim_sel)
    plt.xlim([-1,12])
    # plt.yticks(yTickValues_resp)
    # ax0.xaxis.set_ticklabels([])
    # ax0.xaxis.set_ticks([])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig,ax,'Percentage of azimuth- \n selective boutons (%)','','', mySize=6)  
    # myPlotSettings_splitAxis(fig,ax,'Percentage of azimuth- \n selective boutons (%)','','p= ' + str(p_LMM), mySize=6)  
    
    plt.xticks([0,2,3,4,5,6,7,8,9,10,11], np.append('All',areas), rotation =90)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    # green aud, selective, ELEVATION ONLY
    ax = fig.add_subplot(2,1,2)
    # plt.hlines(0,-1, 1+len(areas)+1, linewidth = 1, color = self.myColorsDict['color_gray_zeroline'],zorder =0)
    plt.plot([- meanLineWidth, meanLineWidth], [green_aud_prop_sel_median_elev,green_aud_prop_sel_median_elev],linewidth = 2,c = 'k',zorder =2)     
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(green_aud_prop_sel_elev)) 
    plt.scatter(xVals_scatter, np.array(green_aud_prop_sel_elev), s = 10, facecolors = 'white' , edgecolors ='k', linewidths =0.5,zorder = 1, alpha =0.3)

    data = green_aud_sel_byArea_elev
    data0 = []
    for i in range(len(data)):
        this = np.nonzero(np.isnan(data[i]) < 0.5)[0]
        data0.append(data[i][this])
    data = data0   

    # ref =  green_aud_prop_sel_median_elev
    data_medians_byArea = np.array([np.nanmedian(data[j]) for j in range(len(data))])       
    # pVals = doWilcoxon_againstRef(data, ref, multiComp = 'hs')
    formula = 'proportion_sel_elev ~ area + Inj_DV + Inj_AP + (1|animal)'                 
    p_LMM, all_pVals = eng.linearMixedModel_fromPython_anova_multiVar(df_path, formula, nargout=2)
    print(p_LMM)

    # plt.hlines(prop_sel_elev_sh, 1,len(areas) +2, linestyle = 'dashed', linewidth =1, color ='k')  

    # upper = [np.percentile(shuffled_prop_sel_elev[j,:], 97.5) for j in range(len(areas))]
    # lower = [np.percentile(shuffled_prop_sel_elev[j,:], 2.5) for j in range(len(areas))]
    # t,p_kruskal = stats.kruskal(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9])

    plt.vlines(1,0, 1, linewidth = 0.5, color = 'gray',zorder =0)
    for i in range(len(areas)):
        plt.plot([i-meanLineWidth_small+2,i+meanLineWidth_small+2], [data_medians_byArea[i],data_medians_byArea[i]] , linewidth = 2, c = ops['myColorsDict']['HVA_colors'][areas[i]],zorder = 2)
        xVals_scatter = np.random.normal(loc =i+2,scale =0.05,size = len(data[i])) 
        plt.scatter(xVals_scatter, data[i], s = 10, facecolors = 'white' , edgecolors = ops['myColorsDict']['HVA_colors'][areas[i]], linewidths =0.5,zorder = 1, alpha=0.3) 

        # plt.fill_between([i-meanLineWidth_small+2,i+meanLineWidth_small+2],[lower[i], lower[i]], [upper[i],upper[i]], color= 'gray', alpha = 0.2)
        # # plt.hlines(pVals_ks[ar], ar - 0.3,ar + 0.3, color = 'r', label='real')            
        # plt.hlines(lower[i],i-meanLineWidth_small+2,i+meanLineWidth_small+2, linewidth = 0.5, color = self.myColorsDict['color_gray_dashedline'],zorder =0)      
        # plt.hlines(upper[i],i-meanLineWidth_small+2,i+meanLineWidth_small+2, linewidth = 0.5, color = self.myColorsDict['color_gray_dashedline'],zorder =0) 

    if p_LMM < 0.05:
        p_mannWhitney, compIdx = doMannWhitneyU_forBoxplots(data, multiComp = 'fdr')
        cnt = 0
        for c in range(len(compIdx)):
            if p_mannWhitney[c] < 0.05:
                pos = compIdx[c].split('_')
                plt.hlines(ylim_sel[-1] - cnt, int(pos[0])+2, int(pos[1])+2, colors = 'k', linewidth=0.5)                    
                cnt +=0.01
      #  t, p_signRank = stats.wilcoxon(data[i]-prop_sel_elev_sh)
        # if p_signRank <  0.00511:
        #     plt.text(i+2,ylim_resp[-1] -0.1, '*', fontsize=10)
        #     print(str(p_signRank))

    plt.ylim(ylim_sel)
    # plt.yticks(yTickValues_resp)
    # ax0.xaxis.set_ticklabels([])
    # ax0.xaxis.set_ticks([])
    plt.xlim([-1,12])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
   # myPlotSettings_splitAxis(fig,ax,'Percentage of elevation- \n selective boutons (%)','','p= ' + str(p_LMM), mySize=15)
    myPlotSettings_splitAxis(fig,ax,'Percentage of elevation- \n selective boutons (%)','','', mySize=6)  

    plt.xticks([0,2,3,4,5,6,7,8,9,10,11], np.append('All',areas), rotation =90)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   

    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\significance_withSessions_byArea_coliseum.svg'))

    # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\TuningProportions_coliseum_aziElev.svg'))



def plotSignalCorrelation_byArea_CS(df, ops, mode = 'all' , computeMatrix =0): 
    
    # areas = ops['areas']
    areas = ['P', 'POR', 'LI', 'LM', 'AL', 'RL', 'A', 'AM', 'PM','V1'] 

    if computeMatrix: #takes a while
        if mode=='all':
            name = 'signal_corr_coliseum_resp_motorSub'
        elif mode=='azi':
            name = 'signal_corr_coliseum_resp_motorSub_azi'
        elif mode=='elev':
            name = 'signal_corr_coliseum_resp_motorSub_elev'
        elif mode=='axons':
            name = 'signal_corr_coliseum_resp_axons_motorSub'

        nBatch =40
        corrMatrix_byArea0 = np.zeros((len(areas), len(areas)))
        for ar0 in range(len(areas)):
            idx_thisArea = np.nonzero(np.array(df['area']) == areas[ar0])[0]

            corr_byArea = [[] for _ in range(len(areas))]
            sumPrev = 0
            for i in tqdm(range(nBatch+1)):
                corrs = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'signal_correlations',name,  name + '_' + str(i) + '.npy'))
                #corrs = np.load(os.path.join(outputPath, 'signal_corr_frequencies_new_2dim' + str(i) + '.npy'))

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
                
        #np.save(os.path.join(ops['dataPath'], 'locations_dataset', 'signalCorr_matrix_resp_axons_coliseum.npy'),corrMatrix_byArea0)
    else:
        if mode=='all':
            name = 'signalCorr_matrix_resp_coliseum_boutons.npy'
            limits = [-0.002, 0.002]
        elif mode=='azi':
            name = 'signalCorr_matrix_resp_coliseum_azi_boutons.npy'
            limits = [-0.003, 0.003]
        elif mode=='elev':
            name = 'signalCorr_matrix_resp_coliseum_elev_boutons.npy'
            limits = [-0.004, 0.004]
        elif mode=='axons':
            name = 'signalCorr_matrix_resp_coliseum_axons.npy'
            limits = [-0.001, 0.001]

            
        corrMatrix_byArea0 = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'signal_correlations', name))
        
    #if small enough to load full matrix
    # nBatch =40
    # for i in tqdm(range(nBatch+1)):
    #     corrs = np.load(os.path.join(outputPath, 'signal_corr_coliseum_resp_motorSub_' + str(i) + '.npy'))
    #     if i ==0:
    #         allCorr = corrs
    #     else:
    #         allCorr = np.concatenate((allCorr, corrs),0)       


    fig = plt.figure(figsize= (ops['mm']*100,ops['mm']*100), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
   
    plt.imshow(corrMatrix_byArea0, cmap = 'RdBu_r', vmin = limits[0], vmax = limits[1])
    
    plt.xticks(np.arange(0,len(areas)), areas, rotation = 90, fontsize=12)
    plt.yticks(np.arange(0,len(areas)), areas,fontsize=12)
    cbar = plt.colorbar(fraction = 0.05,ticks=[limits[0], 0,limits[1]])
    cbar.ax.set_yticklabels([str(limits[0]), '0',str(limits[1])], fontsize=12)
   
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.75)
    plt.title('Signal corr, boutons, both dimensions', fontsize=8)
    
    #%%############################## by stream
    if mode=='all':
        name0 = 'corrs_coliseum_resp_byStream_allData'
        name1 = 'signal_corr_coliseum_resp_motorSub'
    elif mode=='azi':
        name0 = 'corrs_coliseum_resp_azi_byStream_allData'
        name1 = 'signal_corr_coliseum_resp_azi_motorSub'
    elif mode=='elev':
        name0 = 'corrs_coliseum_resp_elev_byStream_allData'
        name1 = 'signal_corr_coliseum_resp_elev_motorSub'
    elif mode=='axons':
        name0 = 'corrs_coliseum_resp_axons_byStream_allData'
        name1 = 'signal_corr_coliseum_resp_axons_motorSub'

    boot_df = np.load(os.path.join(ops['dataPath'], 'locations_dataset','signal_correlations', name1, name0 + '.npy'), allow_pickle=True).item()
   
    color_dorsal = ops['myColorsDict']['HVA_colors']['AM']
    color_ventral = ops['myColorsDict']['HVA_colors']['POR']
    color_mixed = '#A160A4'
    
    diff_dd_dv = boot_df['mean_dd'] - boot_df['mean_dv']
    diff_vv_dv = boot_df['mean_vv'] - boot_df['mean_dv']

    p_dd_dv = np.float64(getBootstrapDiffP(diff_dd_dv))*2    
    p_vv_dv = np.float64(getBootstrapDiffP(diff_vv_dv))*2      

    fig = plt.figure(figsize=(ops['mm']*28, ops['mm']*39), constrained_layout=True)
    ax= fig.add_subplot(1,1,1)
    plt.bar(0,np.median(boot_df['mean_vv']), color= color_ventral, alpha = 0.7, edgecolor=color_ventral)
    plt.vlines(0, np.percentile(boot_df['mean_vv'],2.5), np.percentile(boot_df['mean_vv'],97.5), color='k', linewidth=0.5)
    plt.bar(1,np.median(boot_df['mean_dd']), color= color_dorsal, alpha = 0.7, edgecolor=color_dorsal)
    plt.vlines(1, np.percentile(boot_df['mean_dd'],2.5), np.percentile(boot_df['mean_dd'],97.5), color='k', linewidth=0.5)
    plt.bar(2,np.median(boot_df['mean_dv']), color= color_mixed, alpha = 0.7, edgecolor=color_mixed)
    plt.vlines(2, np.percentile(boot_df['mean_dv'],2.5), np.percentile(boot_df['mean_dv'],97.5), color='k', linewidth=0.5)

    plt.hlines(0,-0.5,2.5,color = 'dimgray', linewidth=0.5, linestyle='dashed')
    myPlotSettings_splitAxis(fig, ax, 'Signal correlation', '', '', mySize=6)
    plt.xticks([0,1,2], ['Ventral-Ventral','Dorsal-Dorsal', 'Ventral-Dorsal'], rotation = 45, horizontalalignment='right')
    # plt.ylim([-0.0005, 0.003])
    # plt.yticks([ 0, 0.001, 0.002,0.003],['0', '0.001', '0.002','0.003'])
    if mode == 'all':
        plt.ylim([-0.0005, 0.003])
        plt.yticks([ 0, 0.001, 0.002, 0.003],['0', '0.001', '0.002', '0.003'])
    elif mode == 'azi':
        plt.ylim([-0.002, 0.006])
        plt.yticks([-0.002, 0, 0.002, 0.004, 0.006],['-0.002', '0', '0.002', '0.004', '0.006'])
    elif mode == 'elev':
        plt.ylim([-0.002, 0.006])
        plt.yticks([-0.002, 0, 0.002, 0.004, 0.006],['-0.002', '0', '0.002', '0.004', '0.006'])
    elif mode == 'axons':
        plt.ylim([-0.0005, 0.002])
        plt.yticks([0, 0.001, 0.002],['0', '0.001', '0.002'])

    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    plt.text(0,np.percentile(boot_df['mean_vv'],97.5), p_vv_dv, color = 'k')
    plt.text(0.5,np.percentile(boot_df['mean_dd'],97.5), p_dd_dv, color = 'k')

    # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\signalCorr_byStream_resp_axons_barplots_hierarchical_allData.svg'))


    
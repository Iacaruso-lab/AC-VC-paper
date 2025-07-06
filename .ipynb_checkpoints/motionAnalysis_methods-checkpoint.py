import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from scipy.io import loadmat
from scipy import stats
import statsmodels.stats.multitest
from sklearn.neighbors import KernelDensity
import os
from tqdm import tqdm
import pandas as pd
import imageio
from sklearn.metrics import confusion_matrix
import mat73

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

from analysis_utils import *

def plotAvgFaceMotion_example(dataPath, ops):
    
    motion_vis = np.squeeze(np.load(os.path.join(dataPath, 'A112_S28_22_02_07_coliseum_avgMotion_visual_zscored.npy')))
    motion_aud = np.squeeze(np.load(os.path.join(dataPath, 'A112_S28_22_02_07_coliseum_avgMotion_auditory_zscored.npy')))
    periods = np.load(os.path.join(dataPath, 'A112_S28_22_02_07_coliseum_locations_definedPeriods.npy'), allow_pickle =True)
    respFrames, baseFrames = periods
    
    nFrames, nAzi, nElev,nTrials = motion_aud.shape
    motion_aud = np.reshape(motion_aud,(nFrames,nAzi*nElev*nTrials))
    
    nFrames, nAzi, nElev,nTrials = motion_vis.shape
    motion_vis = np.reshape(motion_vis,(nFrames,nAzi*nElev*nTrials))

    motion_aud_mean = np.nanmedian(motion_aud,1)
    motion_aud_std = stats.sem(motion_aud,1)
    motion_vis_mean = np.nanmedian(motion_vis,1)
    motion_vis_std = stats.sem(motion_vis,1)

    stimDuration = 0.5
    stimStart = respFrames[0]-1
    myColor = 'red'
    timePerFrame = 1/6.69 #to put the x axis of STAs in seconds
    timeVals = [-1,0,1,2,3]
    timelabels = ['-1','0','1','2','3']

    valinFrames = np.divide(timeVals,timePerFrame)
    adjVals = valinFrames + (stimStart)
    stimEnd = (stimDuration/timePerFrame)+ (stimStart)
    thisRange =np.linspace(stimStart,stimEnd)
    xVals =np.arange(0,nFrames)

    fig = plt.figure(figsize=(ops['mm']*80,ops['mm']*80), constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    plt.fill_between(thisRange,-0.4, 0.4, alpha = 0.7, color = 'lightgray')       

    
    plt.plot(xVals, motion_aud_mean,color = myColor,linewidth =0.8)
    plt.plot(xVals, motion_vis_mean, color='b',linewidth =0.8)

    plt.fill_between(xVals,motion_aud_mean+motion_aud_std,motion_aud_mean-motion_aud_std, facecolor = myColor,alpha = 0.2, edgecolor = myColor,linewidth =0.5)   
    plt.fill_between(xVals,motion_vis_mean+motion_vis_std,motion_vis_mean-motion_vis_std, facecolor = 'b',alpha = 0.2, edgecolor = 'b',linewidth =0.5)
    
    myPlotSettings_splitAxis(fig, ax, 'Avg. face motion (z-scored)', 'Time (s)', '', mySize=15)
    plt.xticks(adjVals,timelabels)
    plt.ylim([-0.42, 0.4])
    plt.xlim([2,29.5])
        
    
    
#     #%% Extra, plot the average motion frame for an example mouse
    motionIm = np.load(os.path.join(dataPath, 'A151_S18_avgeFaceMotion_frame_example.npy'))
    
    plt.figure()
    plt.imshow(motionIm, cmap ='gray_r',vmin =0, vmax=50)
    # plt.colorbar()
    plt.axis('off')
    
    facePCs = np.load(os.path.join(dataPath,'A151_S18_faceMotionSVD_masks.npy'))
    
    fig = plt.figure(figsize=(10,6))
    for pc in range(0,3):
        ax = fig.add_subplot(1,3,pc+1)
        plt.imshow(facePCs[pc,:,:], cmap = 'RdBu_r', vmin = -6, vmax = +6)
        plt.axis('off')
        plt.title('PC ' + str(pc+1))

        
def plotAvgFaceMotion(ops):
    #location
    motion_byAzi_byAnimal = np.load(os.path.join(ops['dataPath'], 'movement_analysis', 'motion_byLocation_azimuth_byAnimal.npy'))
    motion_byElev_byAnimal = np.load(os.path.join(ops['dataPath'], 'movement_analysis', 'motion_byLocation_elevation_byAnimal.npy'))

    motion_byAzi_byAnimal= np.array(motion_byAzi_byAnimal)
    t, p_friedman = stats.friedmanchisquare(motion_byAzi_byAnimal[:,0],motion_byAzi_byAnimal[:,1],motion_byAzi_byAnimal[:,2],
                                            motion_byAzi_byAnimal[:,3],motion_byAzi_byAnimal[:,4],motion_byAzi_byAnimal[:,5],motion_byAzi_byAnimal[:,6])

    animals = np.arange(0,len(motion_byAzi_byAnimal))
    
    fig = plt.figure(figsize=(ops['mm']*160,ops['mm']*80),constrained_layout=True)
    ax = fig.add_subplot(1,2,1)
    for a in range(0,len(animals)):
        plt.plot(motion_byAzi_byAnimal[a,:], linewidth =0.25, c = 'k', alpha =0.5)

    plt.plot(np.nanmean(motion_byAzi_byAnimal,0), c = 'k', linewidth =1.5)
    myPlotSettings_splitAxis(fig, ax, 'Face motion energy (z)', 'Sound azimuth (\u00B0)','p:' + str(np.round(p_friedman,3)), mySize=15)
    plt.xticks([0,3,6], ['-108','0','108'])
    plt.xlim([0,6])
    plt.ylim([-0.5, 0.5])
    plt.yticks([-0.5,-0.25, 0, 0.25, 0.5],['-0.5','-0.25', '0', '0.25', '0.5'])


    motion_byElev_byAnimal= np.array(motion_byElev_byAnimal)
    t, p_friedman = stats.friedmanchisquare(motion_byElev_byAnimal[:,0],motion_byElev_byAnimal[:,1],motion_byElev_byAnimal[:,2])


    ax = fig.add_subplot(1,2,2)
    for a in range(0,len(animals)):
        plt.plot([2,1,0], motion_byElev_byAnimal[a,:], linewidth =0.25, c = 'k', alpha =0.5)

    plt.plot([2,1,0],np.nanmean(motion_byElev_byAnimal,0), c = 'k', linewidth =1.5)
    myPlotSettings_splitAxis(fig, ax, 'Face motion energy (z)', 'Sound elevation (\u00B0)','p:' + str(np.round(p_friedman,3)), mySize=15)
    plt.xticks([0,1,2],['-36', '0', '36'] )
    plt.xlim([0,2])
    plt.ylim([-0.2, 0.2])
    plt.yticks([-0.2, 0, 0.2],['-0.2', '0', '0.2'])
    
    #frequency
    motion_byFreq_byAnimal = np.load(os.path.join(ops['dataPath'], 'movement_analysis', 'motion_byFreq_byAnimal.npy'))
    motion_byVol_byAnimal = np.load(os.path.join(ops['dataPath'], 'movement_analysis', 'motion_byVol_byAnimal.npy'))

    motion_byFreq_byAnimal= np.array(motion_byFreq_byAnimal)
    t, p_friedman = stats.friedmanchisquare(motion_byFreq_byAnimal[:,0],motion_byFreq_byAnimal[:,1],motion_byFreq_byAnimal[:,2],motion_byFreq_byAnimal[:,3],
                                            motion_byFreq_byAnimal[:,4],motion_byFreq_byAnimal[:,5],motion_byFreq_byAnimal[:,6],motion_byFreq_byAnimal[:,7],
                                            motion_byFreq_byAnimal[:,8],motion_byFreq_byAnimal[:,9],motion_byFreq_byAnimal[:,10])


    fig = plt.figure(figsize=(ops['mm']*160, ops['mm']*80),constrained_layout=True)
    ax = fig.add_subplot(1,2,1)
    for a in range(0,len(animals)):
        plt.plot(motion_byFreq_byAnimal[a,:], linewidth =0.25, c = 'k', alpha =0.5)

    plt.plot(np.nanmean(motion_byFreq_byAnimal,0), c = 'k', linewidth =1.5)

    myPlotSettings_splitAxis(fig, ax, 'Face motion energy (z)', 'Sound frequency (kHz)','p:' + str(np.round(p_friedman,3)),mySize=15)
    plt.xticks(np.arange(0,11,2), ['2','4','8','16','32','64'])
    plt.xlim([0,10])
    plt.ylim([-0.5, 0.5])
    plt.yticks([-0.5,-0.25, 0, 0.25, 0.5],['-0.5','-0.25', '0', '0.25', '0.5'])


    motion_byVol_byAnimal= np.array(motion_byVol_byAnimal)
    t, p_friedman = stats.friedmanchisquare(motion_byVol_byAnimal[:,0],motion_byVol_byAnimal[:,1],motion_byVol_byAnimal[:,2])
    ax = fig.add_subplot(1,2,2)
    for a in range(0,len(animals)):
        plt.plot(motion_byVol_byAnimal[a,:], linewidth =0.25, c = 'k', alpha =0.5)

    plt.plot(np.nanmean(motion_byVol_byAnimal,0), c = 'k', linewidth =1.5)

    myPlotSettings_splitAxis(fig, ax, 'Face motion energy (z)', 'Sound intensity (dB SPL)','p:' + str(np.round(p_friedman,3)),mySize=15)
    plt.xticks([0,1,2], ['40','50','60'])
    plt.xlim([0,2]) 
    plt.ylim([-0.5, 0.5])
    plt.yticks([-0.5,-0.25, 0, 0.25, 0.5],['-0.5','-0.25', '0', '0.25', '0.5'])

    
def getDecoderOutputs(dataName, bigDir):

    analysisPath = os.path.join(bigDir,dataName)
    ops = np.load(os.path.join(analysisPath, 'options.npy'), allow_pickle = True).item()
    
    equalize_rounds = ops['nRounds_equalizeTrials']
    nBatches = ops['nBatches']
    nFold = ops['k_fold']
    nShuffles = ops['nShuffles']
    
    
    all_accuracy, all_accuracy_sh = [],[]
    for n in tqdm(range(equalize_rounds)):
        name = os.path.join(analysisPath, 'result_' + str(n) + '.npy')
        if os.path.exists(name):
            res = np.load(name, allow_pickle =True)
        else:
            continue

        results_bySession = []

        N = len(res)/nFold
        sessions = np.arange(0,N)
        sessionIndices = np.repeat(sessions, nFold)
        acc0 = np.zeros(len(sessions),)
        acc0_sh = np.zeros(len(sessions),)
        for s in range(len(sessions)):

            idx = np.nonzero(sessionIndices == sessions[s])[0]

            results = [res[i] for i in idx]
            n_testTrials = len(results[0][0][0])
            nShuffles = len(results[0][1])
            all_predicted = np.empty((len(results), n_testTrials)); all_predicted[:] = np.nan
            all_test = np.empty((len(results), n_testTrials)); all_test[:] = np.nan

            all_predicted_shuffled = np.empty((len(results), n_testTrials, nShuffles)); all_predicted_shuffled[:] = np.nan
            all_test_shuffled = np.empty((len(results), n_testTrials, nShuffles)); all_predicted_shuffled[:] = np.nan
            all_SNR = np.empty((len(results),))

            for i in range(len(results)):
                all_predicted[i,0:len(results[i][0][0])] = results[i][0][0]
                all_test[i,0:len(results[i][0][1])] = results[i][0][1]
                all_SNR[i] = np.mean(results[i][2])

                for j in range(nShuffles):
                    all_predicted_shuffled[i,0:len(results[i][1][j][0]),j] = results[i][1][j][0]
                    all_test_shuffled[i,0:len(results[i][1][j][1]),j] = results[i][1][j][1]

            #data
            predicted = np.squeeze(all_predicted.reshape(1,-1))
            test = np.squeeze(all_test.reshape(1,-1))

            nanIdx0 = np.nonzero(np.isnan(predicted) < 0.5)[0]
            nanIdx1 = np.nonzero(np.isnan(test) < 0.5)[0]
            nanIdx = np.intersect1d(nanIdx0, nanIdx1)

            chance = 1/len(np.unique(test[nanIdx]))
            cm = confusion_matrix(predicted[nanIdx], test[nanIdx],normalize='true')

            acc = np.diag(cm)
            acc0[s] = np.nanmean(acc)

            #shuffle   
            acc_sh= []
            for n in range(nShuffles):
                predicted = np.squeeze(all_predicted_shuffled[:,:,n].reshape(1,-1))
                test = np.squeeze(all_test_shuffled[:,:,n].reshape(1,-1))

                nanIdx0 = np.nonzero(np.isnan(predicted) < 0.5)[0]
                nanIdx1 = np.nonzero(np.isnan(test) < 0.5)[0]
                nanIdx = np.intersect1d(nanIdx0, nanIdx1)

                cm = confusion_matrix(predicted[nanIdx], test[nanIdx],normalize='true')

                acc = np.diag(cm)
                acc_sh.append(np.nanmean(acc))

            acc0_sh[s] = np.nanmean(acc_sh)             

        all_accuracy.append(acc0)
        all_accuracy_sh.append(acc0_sh)

    all_acc = np.array(all_accuracy)        
    all_acc_sh = np.array(all_accuracy_sh)        
   
    return all_acc, all_acc_sh, chance

def plotFaceMotionDecoders(ops):
    #locations_azimuth
    dataName = '350Trials_balanced_average_noPCA_faceSVD_bySession_coliseum_azi'        
    bigDir = os.path.join(ops['dataPath'], 'movement_analysis','decoder_location_azimuth')

    acc0,acc_sh0, chance = getDecoderOutputs(dataName, bigDir)

    acc = np.nanmean(acc0,0)
    acc_sh = np.nanmean(acc_sh0,0)

    analysisPath = os.path.join(bigDir,dataName)
    animalPaths = np.load(os.path.join(analysisPath, 'dataPaths_b0_' + dataName + '_0.npy'))
    animals0 = []
    for i in range(len(animalPaths)):
        an = animalPaths[i].split('/')[-3][1::]
        animals0.append(int(an))

    animals0 = np.array(animals0)
    animalList = np.unique(animals0)
    acc_byAnimal, acc_byAnimal_sh = [],[]
    for a in range(len(animalList)):
        these = np.nonzero(animals0 == animalList[a])[0]
        acc_byAnimal.append(np.mean(acc[these]))            
        acc_byAnimal_sh.append(np.mean(acc_sh[these]))            

    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout= True)
    ax = fig.add_subplot(1,1,1)

    plt.hlines(chance, -0.5, 1.5, linewidth =1, linestyle ='dashed', color = 'gray')
    for i in range(len(acc_byAnimal)):
        plt.plot([0,1], [acc_byAnimal[i], acc_byAnimal_sh[i]], linewidth =0.25, color = 'gray')

    plt.plot([-0.3,+0.3], [np.nanmean(acc_byAnimal),np.nanmean(acc_byAnimal)] , linewidth = 3, c = 'k', zorder =1)
    xVals_scatter = np.repeat(0, len(acc_byAnimal)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal), s = 10, facecolors = 'white' , edgecolors = 'k', linewidths =0.5, alpha =0.8, zorder =2)

    plt.plot([1-0.3,1+0.3], [np.nanmean(acc_byAnimal_sh),np.nanmean(acc_byAnimal_sh)] , linewidth = 3, c = 'gray',zorder =1)
    xVals_scatter = np.repeat(1, len(acc_byAnimal_sh)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal_sh), s = 10, facecolors = 'white' , edgecolors = 'gray', linewidths =0.5, alpha =0.8, zorder=2)

    t, p = stats.wilcoxon(np.array(acc_byAnimal), np.array(acc_byAnimal_sh))
    plt.hlines(0.3, 0,1, color = 'k', linewidth = 0.5)
    plt.text(0.4, 0.18, 'p= ' + str(np.round(p,3)))
    myPlotSettings_splitAxis(fig, ax, 'Decoder accuracy', '', 'Sound azimuth', mySize = 15)
    plt.xticks([0,1],['Data', 'Shuffle'])
    plt.ylim([0.1, 0.2])
    plt.yticks([0.1, 0.15, 0.2])

    #locations_elevation
    dataName = '150Trials_balanced_average_noPCA_faceSVD_bySession_coliseum_elev'        
    bigDir = os.path.join(ops['dataPath'],  'movement_analysis','decoder_location_elevation')

    acc0,acc_sh0, chance = getDecoderOutputs(dataName, bigDir)

    acc = np.nanmean(acc0,0)
    acc_sh = np.nanmean(acc_sh0,0)

    analysisPath = os.path.join(bigDir,dataName)
    animalPaths = np.load(os.path.join(analysisPath, 'dataPaths_b0_' + dataName + '_0.npy'))
    animals0 = []
    for i in range(len(animalPaths)):
        an = animalPaths[i].split('/')[-3][1::]
        animals0.append(int(an))

    animals0 = np.array(animals0)
    animalList = np.unique(animals0)
    acc_byAnimal, acc_byAnimal_sh = [],[]
    for a in range(len(animalList)):
        these = np.nonzero(animals0 == animalList[a])[0]
        acc_byAnimal.append(np.mean(acc[these]))            
        acc_byAnimal_sh.append(np.mean(acc_sh[these]))            

    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout= True)
    ax = fig.add_subplot(1,1,1)

    plt.hlines(chance, -0.5, 1.5, linewidth =1, linestyle ='dashed', color = 'gray')
    for i in range(len(acc_byAnimal)):
        plt.plot([0,1], [acc_byAnimal[i], acc_byAnimal_sh[i]], linewidth =0.25, color = 'gray')

    plt.plot([-0.3,+0.3], [np.nanmean(acc_byAnimal),np.nanmean(acc_byAnimal)] , linewidth = 3, c = 'k', zorder =1)
    xVals_scatter = np.repeat(0, len(acc_byAnimal)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal), s = 10, facecolors = 'white' , edgecolors = 'k', linewidths =0.5, alpha =0.8, zorder =2)

    plt.plot([1-0.3,1+0.3], [np.nanmean(acc_byAnimal_sh),np.nanmean(acc_byAnimal_sh)] , linewidth = 3, c = 'gray',zorder =1)
    xVals_scatter = np.repeat(1, len(acc_byAnimal_sh)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal_sh), s = 10, facecolors = 'white' , edgecolors = 'gray', linewidths =0.5, alpha =0.8, zorder=2)

    t, p = stats.wilcoxon(np.array(acc_byAnimal), np.array(acc_byAnimal_sh))
    plt.hlines(0.3, 0,1, color = 'k', linewidth = 0.5)
    plt.text(0.4, 0.38, 'p= ' + str(np.round(p,3)))
    myPlotSettings_splitAxis(fig, ax, 'Decoder accuracy', '', 'Sound elevation', mySize = 15)
    plt.xticks([0,1],['Data', 'Shuffle'])
    plt.ylim([0.3, 0.4])
    plt.yticks([0.3, 0.35, 0.4])

    #frequency octaves
    dataName = '300Trials_balanced_average_noPCA_faceSVD_bySession_freq_octaves'        
    bigDir = os.path.join(ops['dataPath'], 'movement_analysis','decoder_freq_octaves')

    acc0,acc_sh0, chance = getDecoderOutputs(dataName, bigDir)

    acc = np.nanmean(acc0,0)
    acc_sh = np.nanmean(acc_sh0,0)

    analysisPath = os.path.join(bigDir,dataName)
    animalPaths = np.load(os.path.join(analysisPath, 'dataPaths_b0_' + dataName + '_0.npy'))
    animals0 = []
    for i in range(len(animalPaths)):
        an = animalPaths[i].split('/')[-3][1::]
        animals0.append(int(an))

    animals0 = np.array(animals0)
    animalList = np.unique(animals0)
    acc_byAnimal, acc_byAnimal_sh = [],[]
    for a in range(len(animalList)):
        these = np.nonzero(animals0 == animalList[a])[0]
        acc_byAnimal.append(np.mean(acc[these]))            
        acc_byAnimal_sh.append(np.mean(acc_sh[these]))            

    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout= True)
    ax = fig.add_subplot(1,1,1)

    plt.hlines(chance, -0.5, 1.5, linewidth =1, linestyle ='dashed', color = 'gray')
    for i in range(len(acc_byAnimal)):
        plt.plot([0,1], [acc_byAnimal[i], acc_byAnimal_sh[i]], linewidth =0.25, color = 'gray')

    plt.plot([-0.3,+0.3], [np.nanmean(acc_byAnimal),np.nanmean(acc_byAnimal)] , linewidth = 3, c = 'k', zorder =1)
    xVals_scatter = np.repeat(0, len(acc_byAnimal)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal), s = 10, facecolors = 'white' , edgecolors = 'k', linewidths =0.5, alpha =0.8, zorder =2)

    plt.plot([1-0.3,1+0.3], [np.nanmean(acc_byAnimal_sh),np.nanmean(acc_byAnimal_sh)] , linewidth = 3, c = 'gray',zorder =1)
    xVals_scatter = np.repeat(1, len(acc_byAnimal_sh)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal_sh), s = 10, facecolors = 'white' , edgecolors = 'gray', linewidths =0.5, alpha =0.8, zorder=2)

    t, p = stats.wilcoxon(np.array(acc_byAnimal), np.array(acc_byAnimal_sh))
    #plt.hlines(0.3, 0,1, color = 'k', linewidth = 0.5)
    plt.text(0.4, 0.28, 'p= ' + str(np.round(p,3)))
    myPlotSettings_splitAxis(fig, ax, 'Decoder accuracy', '', 'Tone frequency', mySize = 15)
    plt.xticks([0,1],['Data', 'Shuffle'])
    plt.ylim([0.1, 0.3])
    plt.yticks([0.1, 0.2, 0.3])

    #frequency intensity
    dataName = '150Trials_balanced_average_noPCA_faceSVD_bySession_freq_vol'        
    bigDir = os.path.join(ops['dataPath'], 'movement_analysis','decoder_freq_volume')

    acc0,acc_sh0, chance = getDecoderOutputs(dataName, bigDir)

    acc = np.nanmean(acc0,0)
    acc_sh = np.nanmean(acc_sh0,0)

    analysisPath = os.path.join(bigDir,dataName)
    animalPaths = np.load(os.path.join(analysisPath, 'dataPaths_b0_' + dataName + '_0.npy'))
    animals0 = []
    for i in range(len(animalPaths)):
        an = animalPaths[i].split('/')[-3][1::]
        animals0.append(int(an))

    animals0 = np.array(animals0)
    animalList = np.unique(animals0)
    acc_byAnimal, acc_byAnimal_sh = [],[]
    for a in range(len(animalList)):
        these = np.nonzero(animals0 == animalList[a])[0]
        acc_byAnimal.append(np.mean(acc[these]))            
        acc_byAnimal_sh.append(np.mean(acc_sh[these]))            

    fig = plt.figure(figsize=(ops['mm']*100, ops['mm']*100), constrained_layout= True)
    ax = fig.add_subplot(1,1,1)

    plt.hlines(chance, -0.5, 1.5, linewidth =1, linestyle ='dashed', color = 'gray')
    for i in range(len(acc_byAnimal)):
        plt.plot([0,1], [acc_byAnimal[i], acc_byAnimal_sh[i]], linewidth =0.25, color = 'gray')

    plt.plot([-0.3,+0.3], [np.nanmean(acc_byAnimal),np.nanmean(acc_byAnimal)] , linewidth = 3, c = 'k', zorder =1)
    xVals_scatter = np.repeat(0, len(acc_byAnimal)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal), s = 10, facecolors = 'white' , edgecolors = 'k', linewidths =0.5, alpha =0.8, zorder =2)

    plt.plot([1-0.3,1+0.3], [np.nanmean(acc_byAnimal_sh),np.nanmean(acc_byAnimal_sh)] , linewidth = 3, c = 'gray',zorder =1)
    xVals_scatter = np.repeat(1, len(acc_byAnimal_sh)) 
    plt.scatter(xVals_scatter, np.array(acc_byAnimal_sh), s = 10, facecolors = 'white' , edgecolors = 'gray', linewidths =0.5, alpha =0.8, zorder=2)

    t, p = stats.wilcoxon(np.array(acc_byAnimal), np.array(acc_byAnimal_sh))
    #plt.hlines(0.3, 0,1, color = 'k', linewidth = 0.5)
    plt.text(0.4, 0.38, 'p= ' + str(np.round(p,5)))
    myPlotSettings_splitAxis(fig, ax, 'Decoder accuracy', '', 'Tone intensity', mySize = 15)
    plt.xticks([0,1],['Data', 'Shuffle'])
    plt.ylim([0.3, 0.4])
    plt.yticks([0.3, 0.35, 0.4])
    
def plotCumulativeDist(df, paths,dataset, ops):

    sessionIdx_unique = np.array(df['sessionIdx'].unique())

    color_stim = ops['color_stim']
    color_motor = ops['color_motor']
    color_full = ops['color_full']

    fig = plt.figure(figsize=(31*ops['mm'],28*ops['mm']), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    sumPrev =0
    goodOnes_bySession, varExp_high_bySession, noHighMotor_bySession = [],[],[]
    varExp_motor_bySession, varExp_stim_bySession, varExp_full_bySession = [],[],[]
    propSigMotor_bySession, propSigStim_bySession, propSigFull_bySession = [],[],[]

    for r in tqdm(range(len(sessionIdx_unique))):
        path  = paths[sessionIdx_unique[r]]
       # session = flat_sessions[sessionIdx_unique[r]]
        if os.path.exists(os.path.join(path, 'weights_green_SVD_splitStim_gaussian_v2.mat')):
            mat_file0 = scipy.io.loadmat(os.path.join(path, 'weights_green_SVD_splitStim_gaussian_v2.mat'))    
            weights = getDict_fromMatlabStruct(mat_file0, 'W')

            nRois = len(weights['p_full_varExp'])

            these = df[df['sessionIdx'] == sessionIdx_unique[r]]
            if not len(these) == nRois:
                print(str(r))

            #classify rois into group based on what significant models they have
            alpha = 0.05
            if 'locations' in dataset:
                sigStim_idx = np.nonzero(weights['p_aud_varExp'] < alpha)[0]
            else:
                sigStim_idx = np.nonzero(weights['p_stim_varExp'] < alpha)[0]

            sigMotor_idx = np.nonzero(weights['p_motor_varExp'] < alpha)[0]
            sigFull_idx = np.nonzero(weights['p_full_varExp'] < alpha)[0]
            sig_any_idx = np.unique(np.concatenate((sigStim_idx,sigMotor_idx,sigFull_idx)))
            sig_none_idx =np.setdiff1d(np.arange(0, nRois), sig_any_idx)

            stimOnly_idx = np.setdiff1d(sigStim_idx, sigMotor_idx)
            motorOnly_idx = np.setdiff1d(sigMotor_idx, sigStim_idx)
            # both_idx = np.setdiff1d(sigMotor_idx, sigStim_idx)
            bothOrFull_idx = np.setdiff1d(sig_any_idx, np.unique(np.concatenate((stimOnly_idx, motorOnly_idx))))

            if 'locations' in dataset:
                varExp_stim = np.mean(weights['varExp_aud'],1) #average across folds
            else:
                varExp_stim = np.mean(weights['varExp_stim'],1) #average across folds

            varExp_motor = np.mean(weights['varExp_motor'],1) #average across folds
            varExp_full = np.mean(weights['varExp_full'],1) #average across folds

            varExp_motor_bySession.append(np.nanmedian(varExp_motor[sigMotor_idx]))
            varExp_stim_bySession.append(np.nanmedian(varExp_stim[sigStim_idx]))
            varExp_full_bySession.append(np.nanmedian(varExp_full[sigFull_idx]))

            propSigStim_bySession.append(len(sigStim_idx)/nRois)
            propSigMotor_bySession.append(len(sigMotor_idx)/nRois)
            propSigFull_bySession.append(len(sigFull_idx)/nRois)


            N = len(sigStim_idx)
            y = np.arange(N) / float(N)
            stim_sorted = np.sort(np.squeeze(varExp_stim)[sigStim_idx])
            plt.semilogx(stim_sorted,y, c= color_stim, linewidth = 0.1, alpha = 0.2,label = 'stim only')
            N = len(sigMotor_idx)
            y = np.arange(N) / float(N)
            motor_sorted = np.sort(np.squeeze(varExp_motor)[sigMotor_idx])
            plt.semilogx(motor_sorted,y, c= color_motor, linewidth = 0.1, alpha = 0.2,label = 'motor only')
            N = len(sigFull_idx)
            y = np.arange(N) / float(N)
            full_sorted = np.sort(np.squeeze(varExp_full)[sigFull_idx])
            plt.semilogx(full_sorted,y, c= color_full, linewidth = 0.1, alpha = 0.2,label = 'full')

            sessions0 = np.repeat(r, nRois)
            if r ==0:
                varExp_stim_all = varExp_stim
                varExp_motor_all =varExp_motor
                varExp_full_all = varExp_full
                sigStim_idx0 = sigStim_idx
                sigMotor_idx0 = sigMotor_idx
                sigFull_idx0 = sigFull_idx
                sessions1 = sessions0
            else:   
                varExp_stim_all = np.concatenate((varExp_stim_all, varExp_stim),0)
                varExp_motor_all = np.concatenate((varExp_motor_all,varExp_motor),0)
                varExp_full_all = np.concatenate((varExp_full_all, varExp_full),0)  
                sigStim_idx0 = np.concatenate((sigStim_idx0, sigStim_idx +sumPrev),0)
                sigMotor_idx0 = np.concatenate((sigMotor_idx0, sigMotor_idx +sumPrev),0)
                sigFull_idx0 = np.concatenate((sigFull_idx0, sigFull_idx +sumPrev),0)
                sessions1 = np.concatenate((sessions1, sessions0),0)

            sumPrev = sumPrev + nRois


    N = len(sigStim_idx0)
    y = np.arange(N) / float(N)
    stim_sorted = np.sort(np.squeeze(varExp_stim_all)[sigStim_idx0])
    plt.semilogx(stim_sorted,y, c= color_stim, linewidth = 1, alpha = 1)
    N = len(sigMotor_idx0)
    y = np.arange(N) / float(N)
    motor_sorted = np.sort(np.squeeze(varExp_motor_all)[sigMotor_idx0])
    plt.semilogx(motor_sorted,y, c= color_motor, linewidth = 1, alpha = 1)
    N = len(sigFull_idx0)
    y = np.arange(N) / float(N)
    full_sorted = np.sort(np.squeeze(varExp_full_all)[sigFull_idx0])
    plt.semilogx(full_sorted,y, c= color_full, linewidth = 1, alpha = 1)
    plt.xlim([0.001, 1]) 
    plt.yticks([0,0.25, 0.5, 0.75, 1],['0','0.25', '0.5', '0.75', '1'])
    plt.ylim([-0.03,1.03])
    plt.xticks([0.001, 0.01, 0.1,1 ], ['0.001', '0.01', '0.1','1'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    myPlotSettings_splitAxis(fig, ax, 'Cumulative probability', 'Variance Explained', '', mySize=6)

    result0 = {'varExp_motor_all' : varExp_motor_all,
                'varExp_stim_all' : varExp_stim_all,
                'varExp_full_all' : varExp_full_all,
                'varExp_motor_bySession' :  varExp_motor_bySession,
                'varExp_stim_bySession' :  varExp_stim_bySession,
                'varExp_full_bySession' :  varExp_full_bySession,
                'sessionIdx' : sessions1}
                           
    result1 = {'propSigStim_bySession' : propSigStim_bySession,
               'propSigMotor_bySession' : propSigMotor_bySession,
               'propSigFull_bySession' : propSigFull_bySession,
               'sigStim_idx' : sigStim_idx0,
               'sigMotor_idx' : sigMotor_idx0,
               'sigFull_idx': sigFull_idx0}
    
    if 'locations' in dataset:        
        np.save(os.path.join(ops['dataPath'], 'locations_dataset','varExp_GLM_boutons_locations.npy'), result0)
        np.save(os.path.join(ops['dataPath'], 'locations_dataset','statistics_GLM_boutons_locations.npy'),result1)
        #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\VarianceExplained_cumulative_locations.svg'))
    else:        
        np.save(os.path.join(ops['dataPath'], 'frequencies_dataset','varExp_GLM_boutons_frequencies.npy'), result0)
        np.save(os.path.join(ops['dataPath'], 'frequencies_dataset','statistics_GLM_boutons_frequencies.npy'),result1)
       # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\VarianceExplained_cumulative_frequenciesz.svg'))


    return result0, result1

def plotVarExp_example(path, ops):
    
    import matplotlib as mpl
    new_rc_params = {'text.usetex': False,
    "svg.fonttype": 'none'
    }
    mpl.rcParams.update(new_rc_params)
    
    mat_file0 = scipy.io.loadmat(os.path.join(path,'weights_green_SVD_splitStim_gaussian_v2.mat'))    
    weights = getDict_fromMatlabStruct(mat_file0, 'W')
    
    alpha = 0.05
    
    sigStim_idx = np.nonzero(weights['p_stim_varExp'] < alpha)[0]
    sigMotor_idx = np.nonzero(weights['p_motor_varExp'] < alpha)[0]
    sigFull_idx = np.nonzero(weights['p_full_varExp'] < alpha)[0]
    sig_any_idx = np.unique(np.concatenate((sigStim_idx,sigMotor_idx,sigFull_idx)))

   # varExp_stim = np.mean(weights['varExp_stim'][sig_any_idx,:],1) #average across folds
   # varExp_motor = np.mean(weights['varExp_motor'][sig_any_idx,:],1) #average across folds
    varExp_stim = np.mean(weights['varExp_stim'],1) #average across folds
    varExp_motor = np.mean(weights['varExp_motor'],1) #average across folds

    fig = plt.figure(figsize=(ops['mm']*33,ops['mm']*30), constrained_layout=True)
    #fig = plt.figure(figsize=(ops['mm']*80,ops['mm']*80), constrained_layout=True)

    ax = fig.add_subplot(1,1,1)
    plt.scatter(varExp_motor, varExp_stim, s= 3, linewidth =0, c= 'k', alpha = 0.3)
    plt.scatter(varExp_motor[1596], varExp_stim[1596], s= 10, linewidth =0.7, c= 'k', alpha = 1)

    plt.hlines(0, -0.05, 0.30, linestyle='dashed', color = 'gray', linewidth =0.5)
    plt.vlines(0, -0.05, 0.3, linestyle='dashed', color = 'gray', linewidth =0.5)
    plt.xlim([-0.02, 0.3])
    plt.ylim([-0.02, 0.3])
    myPlotSettings_splitAxis(fig, ax, '', '','', mySize=6)
    plt.xticks([0, 0.1, 0.2, 0.3], ['0', '0.1', '0.2', '0.3'])
    plt.yticks([0,0.1, 0.2, 0.3], ['0', '0.1', '0.2', '0.3'])

    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 


def plotVarExp_bySession(varExp, ops, dataset):
    fig = plt.figure(figsize=(82*ops['mm'],26*ops['mm']), constrained_layout = True)#
    axLim = 0.08
    ax = fig.add_subplot(1,3,1)
    plt.scatter(np.array(varExp['varExp_motor_bySession']), np.array(varExp['varExp_stim_bySession']), facecolors='none', s=8, edgecolors ='k', linewidths=0.2)
    plt.plot([0,axLim],[0,axLim], linestyle = 'dashed', color = 'gray', linewidth =0.5 )
    plt.xlim([0,axLim])
    plt.ylim([0,axLim])
    plt.xticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    plt.yticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    myPlotSettings_splitAxis(fig, ax, 'Var.Exp. Stim Model', 'Var.Exp. Motor Model', '', mySize=6)

    ax = fig.add_subplot(1,3,2)
    plt.scatter(np.array(varExp['varExp_motor_bySession']), np.array(varExp['varExp_full_bySession']), facecolors='none', s=8, edgecolors ='k', linewidths=0.2)
    plt.plot([0,axLim],[0,axLim], linestyle = 'dashed', color = 'gray', linewidth =0.5 )
    plt.xlim([0,axLim])
    plt.ylim([0,axLim])
    plt.xticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    plt.yticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    myPlotSettings_splitAxis(fig, ax, 'Var.Exp. Full Model', 'Var.Exp. Motor Model', '', mySize=6)


    ax = fig.add_subplot(1,3,3)
    plt.scatter(np.array(varExp['varExp_stim_bySession']), np.array(varExp['varExp_full_bySession']), facecolors='none', s=8, edgecolors ='k', linewidths=0.2)
    plt.plot([0,axLim],[0,axLim], linestyle = 'dashed', color = 'gray', linewidth =0.5 )
    plt.xlim([0,axLim])
    plt.ylim([0,axLim])
    plt.xticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    plt.yticks([0, 0.04, 0.08], ['0', '0.04', '0.08'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    myPlotSettings_splitAxis(fig, ax, 'Var.Exp. Full Model', 'Var.Exp. Stim Model', '', mySize=6)
    
   # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\varExp_bySession_scatters_' + dataset + '.svg'))

def plotPropSig_GLM(sig_GLM,ops, dataset):
    
    color_stim = ops['color_stim']
    color_motor = ops['color_motor']
    color_full = ops['color_full']
    
    fig = plt.figure(figsize=(31*ops['mm'], 23*ops['mm']), constrained_layout =True)
    ax = fig.add_subplot(1,1,1)
    plt.plot([0-0.25,0+0.25], [np.median(np.array(sig_GLM['propSigStim_bySession'])),np.median(np.array(sig_GLM['propSigStim_bySession']))] , linewidth = 2, c = color_stim,zorder = 1)
    xVals_scatter = np.random.normal(loc =0,scale =0.05,size = len(sig_GLM['propSigStim_bySession'])) 
    plt.scatter(xVals_scatter, np.array(np.array(sig_GLM['propSigStim_bySession'])), s = 8, facecolors = 'white' , edgecolors = color_stim, linewidths =0.5,zorder = 2, alpha=0.3)

    plt.plot([1-0.3,1+0.3], [np.median(np.array(sig_GLM['propSigMotor_bySession'])),np.median(np.array(sig_GLM['propSigMotor_bySession']))] , linewidth = 2, c = color_motor,zorder = 1)
    xVals_scatter = np.random.normal(loc =1,scale =0.05,size = len(sig_GLM['propSigMotor_bySession'])) 
    plt.scatter(xVals_scatter, np.array(np.array(sig_GLM['propSigMotor_bySession'])), s = 8, facecolors = 'white' , edgecolors = color_motor, linewidths =0.5,zorder = 2, alpha=0.3)

    plt.plot([2-0.3,2+0.3], [np.median(np.array(sig_GLM['propSigFull_bySession'])),np.median(np.array(sig_GLM['propSigFull_bySession']))] , linewidth = 2, c = color_full,zorder = 1)
    xVals_scatter = np.random.normal(loc =2,scale =0.05,size = len(sig_GLM['propSigFull_bySession'])) 
    plt.scatter(xVals_scatter, np.array(np.array(sig_GLM['propSigFull_bySession'])), s = 8, facecolors = 'white' , edgecolors = color_full, linewidths =0.5,zorder = 2, alpha=0.3)
    myPlotSettings_splitAxis(fig, ax, 'Pecentage of boutons \nwith sig. model (%)', '', '', mySize=6)
    plt.ylim([-0.05,1])
    plt.yticks([0,0.25,0.5,0.75, 1], ['0', '25','50', '75','100'])
    plt.xticks([0,1,2], ['Stim', 'Motor', 'Full'], rotation = 0)
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\PropSigModel_' + dataset + '.svg'))

    this = np.concatenate((np.expand_dims(np.array(sig_GLM['propSigStim_bySession']),0),np.expand_dims(np.array(sig_GLM['propSigMotor_bySession']),0),
                           np.expand_dims(np.array(sig_GLM['propSigFull_bySession']),0)),0)

    p, compIdx = doWilcoxon_forBoxplots(this, multiComp = 'fdr')
    for i in range(len(compIdx)):
        print(str(p[i]))
        
def plotGLM_traces_example(path, ops):
    
    #load reconstructed traces, weights, and stim and motor subtracted traces
    from scipy.ndimage import gaussian_filter1d

    traces_rec = mat73.loadmat(os.path.join(path, 'reconstructed_traces_green_SVD_splitStim_gaussian_v1.mat'))
    traces_rec = traces_rec['traces_rec']

    behavTraces = np.load(os.path.join(path, '2_23_10_18_behaviourTraces.npy'), allow_pickle =True).item()
    nanIdx = np.load(os.path.join(path, 'nanIdx_GLM_gaussian.npy'))

    loc = behavTraces['encoder']
    idx = np.arange(0, loc.shape[1])
    notNan = np.setdiff1d(idx, nanIdx)

    loc0 = loc[0,notNan]

    niceIdx = 1596 #nice example ROI
    color_motor = '#FF9D00'
    color_stim = '#1368F0'
    color_full = '#C90700'

    fs = 6
    nFrames = traces_rec['traces_real'].shape[0]
    frames = np.arange(0,nFrames)
    time = frames/fs

    fig = plt.figure(figsize=(100*ops['mm'], 100*ops['mm']), constrained_layout=False)

    ax =fig.add_subplot(1,1,1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_real'][:,niceIdx],1), color ='k')
    plt.plot(time,gaussian_filter1d(traces_rec['traces_full'][:,niceIdx],1), color =color_full)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_stim'][:,niceIdx],1), color =color_stim)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_motor'][:,niceIdx],1), color =color_motor)
    plt.xlim([2150, 2300]) #this bit is nice, has example of stim and motor events close to each other
    plt.ylim([-5,40])
    plt.axis('off')
    #Now plot example of motor and stim subtraction
    # adjust locomotion first
    loc0_smooth = gaussian_filter1d(loc0,1)
    val0 = np.median(loc0)
    loc_fixed = abs(loc0 - val0)

    #% PLot motorSub
    fig = plt.figure(figsize=(100*ops['mm'], 100*ops['mm']), constrained_layout=False)
    color_stimSub =  '#BF65C9' 
    color_motorSub = '#008D36'
    ax =fig.add_subplot(1,1,1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_real'][:,niceIdx],1), color ='k', linewidth=1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_real'][:,niceIdx] - traces_rec['traces_motor'][:,niceIdx] ,1) , color =color_motorSub, linewidth=1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_real'][:,niceIdx] - traces_rec['traces_stim'][:,niceIdx] ,1), color =color_stimSub, linewidth=1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_motor'][:,niceIdx] ,1), color =color_motor, linewidth=1)
    plt.plot(time,gaussian_filter1d(traces_rec['traces_stim'][:,niceIdx],1), color =color_stim,linewidth=1)
    plt.plot(time,loc_fixed, color ='gray',linewidth=1)
    plt.xlim([2150, 2300])
    plt.ylim([-10,40])
    plt.axis('off')

def plotMotorSub_example(path, ops):
    color_stimSub =  '#BF65C9' 
    color_motorSub = '#008D36'
    mat_file = scipy.io.loadmat(os.path.join(path, 'motorTuning_stats_green_splitStim_gaussian_v1.mat'))     
    tuning_stats = getDict_fromMatlabStruct(mat_file, 'tuning_stats') 

    mat_file = scipy.io.loadmat(os.path.join(path, 'motorTuning_stats_motorSub_green_splitStim_gaussian_v1.mat'))     
    tuning_stats_motorSub = getDict_fromMatlabStruct(mat_file, 'tuning_stats') 

    tuning = np.load(os.path.join(path,'motionTuning_green_GLM_gaussian.npy'))
    tuning_motorSub = np.load(os.path.join(path, 'motionTuning_green_GLM_gaussian_motorSub.npy'))
    tuning_stimSub = np.load(os.path.join(path, 'motionTuning_green_GLM_gaussian_stimSub.npy'))

    plt.close('all')
    nVar = 2
    idx = [4,580,1635,1554,1868, 1596]

    for i in range(len(idx)):
        fig = plt.figure(figsize=(ops['mm']*62, ops['mm']*29),constrained_layout=True)

        for n in range(nVar):
            ax = fig.add_subplot(1,nVar,n+1)

            tuning0 = tuning[idx[i],:,n,0]
            thisMin = np.min(tuning0)
            thisMax = np.max(tuning0)

            tuning0 = tuning[idx[i],:,n,0] #-thisMin

            tuning0_std = tuning[idx[i],:,n,1]
            tuning_motorSub0 = tuning_motorSub[idx[i],:,n,0] + abs(thisMin - min(tuning_motorSub[idx[i],:,n,0]))
            tuning_motorSub0_std = tuning_motorSub[idx[i],:,n,1]
            tuning_stimSub0 = tuning_stimSub[idx[i],:,n,0]  + abs(thisMin - min(tuning_stimSub[idx[i],:,n,0]))
            tuning_stimSub0_std = tuning_stimSub[idx[i],:,n,1]


            tuning0 = (tuning0 - thisMin)/(thisMax - thisMin)
            tuning0_std = (tuning0_std)/(thisMax - thisMin)       
            tuning_motorSub0 = (tuning_motorSub0 - thisMin)/(thisMax - thisMin)
            tuning_motorSub0_std = (tuning_motorSub0_std)/(thisMax - thisMin)
            tuning_stimSub0 = (tuning_stimSub0 - thisMin)/(thisMax - thisMin)
            tuning_stimSub0_std = (tuning_stimSub0_std)/(thisMax - thisMin)

            xVals = np.arange(0, tuning.shape[1])
            plt.scatter(xVals, tuning0,s=5, c= 'k', label = 'Real')
            plt.errorbar(xVals, tuning0,tuning0_std,  color = 'k',linewidth=0.5)

            plt.scatter(xVals, tuning_motorSub0,s=5, c= color_motorSub , label = 'F - Fmotor')
            plt.errorbar(xVals, tuning_motorSub0,tuning_motorSub0_std,  color = color_motorSub, linewidth=0.5 )

            plt.scatter(xVals, tuning_stimSub0,s=5, c= color_stimSub , label = 'F - Fstim')
            plt.errorbar(xVals, tuning_stimSub0,tuning_stimSub0_std,  color = color_stimSub,linewidth=0.5 )

            myPlotSettings_splitAxis(fig, ax, 'Norm. binned. fluorescence', '', '',mySize=6)    
            plt.xticks([0,5,10,15])
            ax.tick_params(axis='y', pad=1)   
            ax.tick_params(axis='x', pad=1) 
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6','0.8', '1'])     
            
        #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\motorSub_example_tuningCurve_A151_S2_roi' + str(idx[i]) + '.svg'))
 

def getMotorSub_vars(df, paths, dataset):
    
    sessionIdx_unique = np.array(df['sessionIdx'].unique())

    nVar = 31 #locomotion + 5 first PCs
    prop_highVar = np.empty((len(sessionIdx_unique), nVar)); prop_highVar[:] = np.nan
    prop_highVar_motorSub = np.empty((len(sessionIdx_unique), nVar)); prop_highVar_motorSub[:] = np.nan
    prop_highVar_stimSub = np.empty((len(sessionIdx_unique), nVar)); prop_highVar_stimSub[:] = np.nan
    if 'locations' in dataset:
        version = 'v2'
    else:
        version ='v1'
    
    alpha =0.05      
    for r in tqdm(range(len(sessionIdx_unique))):
        path = paths[sessionIdx_unique[r]]
        
        if not os.path.exists(os.path.join(path, 'motorTuning_stats_green_splitStim_gaussian_' + version + '.mat')):
            continue
        mat_file = scipy.io.loadmat(os.path.join(path, 'motorTuning_stats_green_splitStim_gaussian_' + version + '.mat'))     
        tuning_stats = getDict_fromMatlabStruct(mat_file, 'tuning_stats')  

        mat_file = scipy.io.loadmat(os.path.join(path, 'motorTuning_stats_motorSub_green_splitStim_gaussian_' + version + '.mat'))     
        tuning_stats_motorSub = getDict_fromMatlabStruct(mat_file, 'tuning_stats')  
        
        mat_file = scipy.io.loadmat(os.path.join(path, 'motorTuning_stats_stimSub_green_splitStim_gaussian_' + version + '.mat'))     
        tuning_stats_stimSub = getDict_fromMatlabStruct(mat_file, 'tuning_stats')  

        nRois = tuning_stats['higherVar'].shape[0]
        for n in range(nVar):
            #levene test of unequal variance
            highVar0 = np.nonzero(tuning_stats['higherVar'][:,n])[0]
            sigVar = np.nonzero(tuning_stats['p_levene'][:,n] < alpha)[0]
            highVar = np.intersect1d(highVar0, sigVar)

            prop_highVar[r,n] = len(highVar)/nRois
            #
            highVar0 = np.nonzero(tuning_stats_motorSub['higherVar'][:,n])[0]
            sigVar = np.nonzero(tuning_stats_motorSub['p_levene'][:,n] < alpha)[0]
            highVar = np.intersect1d(highVar0, sigVar)

            prop_highVar_motorSub[r,n] = len(highVar)/nRois

            #
            highVar0 = np.nonzero(tuning_stats_stimSub['higherVar'][:,n])[0]
            sigVar = np.nonzero(tuning_stats_stimSub['p_levene'][:,n] < alpha)[0]
            highVar = np.intersect1d(highVar0, sigVar)

            prop_highVar_stimSub[r,n] = len(highVar)/nRois
    
    return prop_highVar, prop_highVar_stimSub, prop_highVar_motorSub

def plotMotorSub_quantification(prop_highVar, prop_highVar_stimSub, prop_highVar_motorSub, ops, dataset):
    median_byVar = np.nanmean(prop_highVar,0) 
    sem_byVar = np.nanstd(prop_highVar,0) 
    mean_byVar_motorSub = np.nanmean(prop_highVar_motorSub, 0)    
    sem_byVar_motorSub = np.nanstd(prop_highVar_motorSub,0) 
    mean_byVar_stimSub = np.nanmean(prop_highVar_stimSub, 0)    
    sem_byVar_stimSub = np.nanstd(prop_highVar_stimSub,0)  
    color_stimSub =  '#BF65C9' 
    color_motorSub = '#008D36'
    
    fig = plt.figure(figsize=(ops['mm']*37,ops['mm']*37), constrained_layout =True)
    ax= fig.add_subplot(1,2,1)

    plt.scatter(0,np.nanmedian(prop_highVar[:,0]),s=8, c= 'k', label = 'Real')
    plt.vlines(0, [np.nanpercentile(prop_highVar[:,0],25)], [np.nanpercentile(prop_highVar[:,0],75)],  color = 'k',linewidth=0.5)

    plt.scatter(1,np.nanmedian(prop_highVar_motorSub[:,0]),s=8, c= color_motorSub, label = 'Motor')
    plt.vlines(1, [np.nanpercentile(prop_highVar_motorSub[:,0],25)], [np.nanpercentile(prop_highVar_motorSub[:,0],75)],  color = color_motorSub,linewidth=0.5)

    plt.scatter(2,np.nanmedian(prop_highVar_stimSub[:,0]),s=8, c= color_stimSub, label = 'Stim')
    plt.vlines(2, [np.nanpercentile(prop_highVar_stimSub[:,0],25)], [np.nanpercentile(prop_highVar_stimSub[:,0],75)],  color = color_stimSub,linewidth=0.5)

    this = np.array([prop_highVar[:,0], prop_highVar_motorSub[:,0], prop_highVar_stimSub[:,0]])
    p_loc, compIdx = doWilcoxon_forBoxplots(this, multiComp = 'fdr_bh')

    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%)', '', '',mySize=6)  
    plt.yticks([0,0.1,0.2], ['0','10','20'])
    plt.xticks([0,1,2], ['Measured', 'Motor sub.', 'Stim. sub.'], rotation=45, horizontalalignment='right')
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 

    ax= fig.add_subplot(1,2,2)

    plt.scatter(0, np.nanmedian(np.nanmean(prop_highVar[:,1::],1)),s=8, c= 'k')
    plt.vlines(0, np.nanpercentile(np.nanmean(prop_highVar[:,1::],1),25),np.nanpercentile(np.nanmean(prop_highVar[:,1::],1),75),  color = 'k',linewidth=0.5)

    plt.scatter(1, np.nanmedian(np.nanmean(prop_highVar_motorSub[:,1::],1)),s=8, c= color_motorSub)
    plt.vlines(1, np.nanpercentile(np.nanmean(prop_highVar_motorSub[:,1::],1),25),np.nanpercentile(np.nanmean(prop_highVar_motorSub[:,1::],1),75),  color =color_motorSub,linewidth=0.5)

    plt.scatter(2, np.nanmedian(np.nanmean(prop_highVar_stimSub[:,1::],1)),s=8, c= color_stimSub)
    plt.vlines(2, np.nanpercentile(np.nanmean(prop_highVar_stimSub[:,1::],1),25),np.nanpercentile(np.nanmean(prop_highVar_stimSub[:,1::],1),75),  color =color_stimSub,linewidth=0.5)

    myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=6)  
    plt.yticks([], [])
    ax.spines["left"].set_visible(False)
    plt.xticks([0,1,2], ['Measured', 'Motor sub.', 'Stim. sub'], rotation=45, horizontalalignment='right')
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    
    this = np.array([np.nanmean(prop_highVar[:,1::],1), np.nanmean(prop_highVar_motorSub[:,1::],1), np.nanmean(prop_highVar_stimSub[:,1::],1)])
    p_face, compIdx = doWilcoxon_forBoxplots(this, multiComp = 'fdr_bh')
    
    fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\motorSub_quantification_' + dataset + '.svg'))

    return p_loc, p_face
    
def plotResp_motorSub_freq(df, ops):

    resp_axons_measured = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'resp_green_measured.npy'))
    resp_axons_motorSub = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'resp_green_motorSub.npy'))

    color_stimSub = ops['color_stimSub']
    color_motorSub = ops['color_motorSub']

    df_resp = df.iloc[resp_axons_measured]
    df_resp_motorSub = df.iloc[resp_axons_motorSub]

    prop_resp = makeProportions_bySession_v2(df_resp, df) #includes responsive to both
    prop_resp_motorSub = makeProportions_bySession_v2(df_resp_motorSub, df) #includes responsive to both
    prop_resp_median = np.nanmedian(prop_resp)
    prop_resp_median_motorSub = np.nanmedian(prop_resp_motorSub)

    sel_axons_measured = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'sel_freq_green_measured.npy'))
    sel_axons_motorSub = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'sel_freq_green_motorSub.npy'))

    df_sel = df.iloc[sel_axons_measured]
    df_sel_motorSub = df.iloc[sel_axons_motorSub]

    prop_sel_freq = makeProportions_bySession_v2(df_sel, df_resp)
    prop_sel_freq_median = np.nanmedian(prop_sel_freq)

    prop_sel_freq_motorSub = makeProportions_bySession_v2(df_sel_motorSub, df_resp_motorSub)
    prop_sel_freq_median_motorSub = np.nanmedian(prop_sel_freq_motorSub)

    fig = plt.figure(figsize=(ops['mm']*48, ops['mm']*21), constrained_layout =True)
    ax = fig.add_subplot(1,2,1)
    t, p = scipy.stats.wilcoxon(prop_resp, prop_resp_motorSub)

    plt.plot([-0.3,+0.3], [prop_resp_median,prop_resp_median], linewidth = 2, c = 'k', label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(0, len(prop_resp)), np.array(prop_resp), s = 3, facecolors = 'white' , edgecolors ='k', linewidths =0.25)

    plt.plot([1-0.3,1+0.3], [prop_resp_median_motorSub,prop_resp_median_motorSub], linewidth = 2, c = color_motorSub, label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(1, len(prop_resp)), np.array(prop_resp_motorSub), s = 3, facecolors = 'white' , edgecolors =color_motorSub, linewidths =0.25)
    for i in range(len(prop_resp)):
        plt.plot([0,1],[prop_resp[i], prop_resp_motorSub[i]], linewidth = 0.1, color = 'lightgray')
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig, ax, 'Percentage of \n boutons (%)', '', '', mySize=6)
    if p < 0.05:
        plt.hlines(0.90, 0,1,color = 'k', linewidth =1)
        # plt.text(0.4, 0.95, 'p : ' )
    print(str(p))
    plt.xticks([0,1], ['', ''])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 


    ax = fig.add_subplot(1,2,2)
    t, p = scipy.stats.wilcoxon(prop_sel_freq, prop_sel_freq_motorSub)

    plt.plot([-0.3,+0.3], [prop_sel_freq_median,prop_sel_freq_median], linewidth = 2, c = 'k', label = 'Real')
    plt.scatter(np.repeat(0, len(prop_sel_freq)), np.array(prop_sel_freq), s = 3, facecolors = 'white' , edgecolors ='k', linewidths =0.25)

    plt.plot([1-0.3,1+0.3], [prop_sel_freq_median_motorSub,prop_sel_freq_median_motorSub], linewidth = 2, c = color_motorSub, label = 'Real')
    plt.scatter(np.repeat(1, len(prop_sel_freq)), np.array(prop_sel_freq_motorSub), s = 3, facecolors = 'white' , edgecolors =color_motorSub, linewidths =0.25)
    for i in range(len(prop_sel_freq)):
        plt.plot([0,1],[prop_sel_freq[i], prop_sel_freq_motorSub[i]], linewidth = 0.1, color = 'lightgray')
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=6)
    if p < 0.05:
        plt.hlines(0.90, 0,1,color = 'k', linewidth =1)
    print(str(p))
    plt.xticks([0,1], ['', ''])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 

    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\propResp_motorSub_freqs.svg'))

    
def plotResp_motorSub_locations(df, ops):

    resp_green = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'resp_green_measured.npy'))
    resp_green_motorSub = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'resp_green_motorSub.npy'))
    
    color_motorSub = ops['color_motorSub']
    color_stimSub = ops['color_stimSub']

    #divide by session
    df_resp = df.iloc[resp_green]
    df_resp_motorSub = df.iloc[resp_green_motorSub]

    prop_resp = makeProportions_bySession_v2(df_resp, df) #includes responsive to both
    prop_resp_motorSub = makeProportions_bySession_v2(df_resp_motorSub, df) #includes responsive to both
    prop_resp_median = np.nanmedian(prop_resp)
    prop_resp_median_motorSub = np.nanmedian(prop_resp_motorSub)

    sel_green = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'sel_azimuth_green_measured.npy'))
    sel_green_motorSub = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'sel_azimuth_green_motorSub.npy'))

    df_sel = df.iloc[sel_green]
    df_sel_motorSub = df.iloc[sel_green_motorSub]

    prop_sel = makeProportions_bySession_v2(df_sel, df_resp)
    prop_sel_median = np.nanmedian(prop_sel)

    prop_sel_motorSub = makeProportions_bySession_v2(df_sel_motorSub, df_resp_motorSub)
    prop_sel_median_motorSub = np.nanmedian(prop_sel_motorSub)

    sel_green_elev = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'sel_elevation_green_measured.npy'))
    sel_green_motorSub_elev = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'sel_elevation_green_motorSub.npy'))

    df_sel_elev = df.iloc[sel_green_elev]
    df_sel_motorSub_elev = df.iloc[sel_green_motorSub_elev]

    prop_sel_elev = makeProportions_bySession_v2(df_sel_elev, df_resp)
    prop_sel_median_elev = np.nanmedian(prop_sel_elev)

    prop_sel_motorSub_elev = makeProportions_bySession_v2(df_sel_motorSub_elev, df_resp_motorSub)
    prop_sel_median_motorSub_elev = np.nanmedian(prop_sel_motorSub_elev)

    ####################################################
    fig = plt.figure(figsize=(ops['mm']*70, ops['mm']*21), constrained_layout =True)
    ax = fig.add_subplot(1,3,1)
    t, p = scipy.stats.wilcoxon(prop_resp, prop_resp_motorSub)

    plt.plot([-0.3,+0.3], [prop_resp_median,prop_resp_median], linewidth = 2, c = 'k', label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(0, len(prop_resp)), np.array(prop_resp), s = 3, facecolors = 'white' , edgecolors ='k', linewidths =0.25)

    plt.plot([1-0.3,1+0.3], [prop_resp_median_motorSub,prop_resp_median_motorSub], linewidth = 2, c = color_motorSub, label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(1, len(prop_resp)), np.array(prop_resp_motorSub), s = 3, facecolors = 'white' , edgecolors =color_motorSub, linewidths =0.25)
    for i in range(len(prop_resp)):
        plt.plot([0,1],[prop_resp[i], prop_resp_motorSub[i]], linewidth = 0.1, color = 'lightgray')
    plt.xticks([0,1], ['', ''], rotation =45)
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig, ax, 'Percentage of boutons (%) ', '', '', mySize=6)
    if p < 0.05:
        plt.hlines(0.90, 0,1,color = 'k', linewidth =0.5)
        # plt.text(0.4, 0.95, 'p : ' + str(np.round(p, 4)))
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)  
    print(p)


    ax = fig.add_subplot(1,3,2)
    t, p = scipy.stats.wilcoxon(prop_sel, prop_sel_motorSub)

    plt.plot([-0.3,+0.3], [prop_sel_median,prop_sel_median], linewidth = 2, c = 'k', label = 'Real')
    plt.scatter(np.repeat(0, len(prop_sel)), np.array(prop_sel), s = 3, facecolors = 'white' , edgecolors ='k', linewidths =0.25)

    plt.plot([1-0.3,1+0.3], [prop_sel_median_motorSub,prop_sel_median_motorSub], linewidth = 2, c = color_motorSub, label = 'Real')
    plt.scatter(np.repeat(1, len(prop_sel)), np.array(prop_sel_motorSub), s = 3, facecolors = 'white' , edgecolors =color_motorSub, linewidths =0.25)
    for i in range(len(prop_sel)):
        plt.plot([0,1],[prop_sel[i], prop_sel_motorSub[i]], linewidth = 0.1, color = 'lightgray')
    plt.xticks([0,1], ['', ''], rotation =45)
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=6)
    if p < 0.05:
        plt.hlines(0.90, 0,1,color = 'k', linewidth =0.5)
        # plt.text(0.4, 0.95, 'p : ' + str(np.round(p, 4)))
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)   
    print(p)

    ax = fig.add_subplot(1,3,3)
    t, p = scipy.stats.wilcoxon(prop_sel_elev, prop_sel_motorSub_elev)

    plt.plot([-0.3,+0.3], [prop_sel_median_elev,prop_sel_median_elev], linewidth = 2, c = 'k', label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(0, len(prop_sel_elev)), np.array(prop_sel_elev), s = 3, facecolors = 'white' , edgecolors ='k', linewidths =0.25)

    plt.plot([1-0.3,1+0.3], [prop_sel_median_motorSub_elev,prop_sel_median_motorSub_elev], linewidth = 2, c = color_motorSub, label = 'Real')
    # xVals_scatter = np.random.normal(loc =0,scale =0.15,size = len(prop_resp)) 
    plt.scatter(np.repeat(1, len(prop_sel_elev)), np.array(prop_sel_motorSub_elev), s = 3, facecolors = 'white' , edgecolors =color_motorSub, linewidths =0.25)
    for i in range(len(prop_sel_elev)):
        plt.plot([0,1],[prop_sel_elev[i], prop_sel_motorSub_elev[i]], linewidth = 0.1, color = 'lightgray')
    plt.xticks([0,1], ['', ''], rotation =45)
    plt.ylim([0,1])
    plt.yticks([0,0.5, 1], ['0','50', '100'])
    myPlotSettings_splitAxis(fig, ax, '', '', '', mySize=6)
    if p < 0.05:
        plt.hlines(0.90, 0,1,color = 'k', linewidth =0.5)

    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)  
    print(p)


   # fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\propResp_motorSub_locations.svg'))

def plotFrequencyChange(df, ops):
    
    df_fit = np.load(os.path.join(ops['dataPath'],'frequencies_dataset','df_fit_freqs_sel_measured.npy'), allow_pickle=True).item()
    df_fit_motorSub = np.load(os.path.join(ops['dataPath'],'frequencies_dataset','df_fit_freqs_sel_motorSub.npy'), allow_pickle=True).item()
    resp_green_freq = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'resp_green_measured.npy'))
    sel_green_freq = np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'sel_freq_green_measured.npy'))

    maps_freq =  np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'maps_freq_green_dataset_all_measured.npy'))
    maps_freq_motorSub =  np.load(os.path.join(ops['dataPath'], 'frequencies_dataset', 'maps_freq_green_dataset_all_GLM_motorSub.npy'))

    color_motorSub = ops['color_motorSub']
    #########################################
    fig = plt.figure(figsize=(ops['mm']*72,ops['mm']*24), constrained_layout=True)

    maps0 = maps_freq[resp_green_freq][:,1:12,:]
    maps0_motorSub = maps_freq_motorSub[resp_green_freq][:,1:12,:]

    nRois = maps0.shape[0]
    signal_corrs= np.zeros(nRois,)

    for roi in tqdm(range(nRois)):
        data = np.squeeze(maps0[roi,::].reshape(-1,1))
        data_motorSub = np.squeeze(maps0_motorSub[roi,::].reshape(-1,1))

        r, p = stats.pearsonr(data,data_motorSub)
        signal_corrs[roi] = r

    ax = fig.add_subplot(1,3,1)
    bins_corr = np.arange(-1,1.05, 0.05)
    hist_norm, bins =np.histogram(signal_corrs,bins_corr)
    hist_norm = hist_norm/np.sum(hist_norm)
    plt.hist(bins[:-1], bins, weights = hist_norm, color ='#69635E')
    myPlotSettings_splitAxis(fig, ax, 'Percentage of \nboutons (%)', 'Signal corr.', '',mySize=6)
    plt.xlim([0.5,1])
    plt.ylim([0,1])
    plt.xticks([0.5,0.75, 1],['0.5','0.75', '1'])
    plt.yticks([0,0.5, 1],['0','50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)    

    
    ###########################
    x_interp= np.linspace(0, 10.1, 1000)       

    ax = fig.add_subplot(1,3,2)
    kde = KernelDensity(bandwidth=0.7, kernel='gaussian')   
    data = df_fit['spline_peak'][df_fit['singlePeak']]            
    kde.fit(data.reshape(-1,1))
    logprob = kde.score_samples(x_interp.reshape(-1,1))
    plt.plot(x_interp, np.exp(logprob), alpha=1, linewidth = 0.5, color = 'k') 
    plt.fill_between(x_interp, np.exp(logprob), alpha=0.1, color = 'k')
    plt.scatter(np.median(data), 0.18, marker ='v', s= 20, color ='k')

    data = df_fit_motorSub['spline_peak'][df_fit['singlePeak']]           
    kde.fit(data.reshape(-1,1))
    logprob = kde.score_samples(x_interp.reshape(-1,1))
    plt.plot(x_interp, np.exp(logprob), alpha=1, linewidth = 0.5, color = color_motorSub) 
    plt.fill_between(x_interp, np.exp(logprob), alpha=0.1, color = color_motorSub)
    plt.scatter(np.median(data), 0.18, marker ='v', s= 20, color =color_motorSub)

    myPlotSettings_splitAxis(fig, ax, '', 'Frequency (kHz)', '', mySize=6)
    plt.xticks([0,2,4,6,8,10], ['2', '4', '8', '16', '32', '64'])
    plt.yticks([0,0.1, 0.2],['0','10', '20'])
    plt.ylim([0, 0.2])
    plt.xlim([0, 10])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)    

    D,p = stats.kstest(np.squeeze(df_fit['spline_peak'][df_fit['singlePeak']]),np.squeeze(df_fit_motorSub['spline_peak'][df_fit['singlePeak']]))
    D,p = stats.wilcoxon(np.squeeze(df_fit['spline_peak'][df_fit['singlePeak']]),np.squeeze(df_fit_motorSub['spline_peak'][df_fit['singlePeak']]))
    print(str(p))
    
    #####################################
    ax = fig.add_subplot(1,3,3)
    bins_delta = np.arange(0,1.05, 0.05)
    delta = abs(df_fit['spline_peak'][df_fit['singlePeak']]-df_fit_motorSub['spline_peak'][df_fit['singlePeak']])/2
    hist_norm, bins =np.histogram(delta,bins_delta)
    hist_norm = hist_norm/np.sum(hist_norm)
    plt.hist(bins[:-1], bins, weights = hist_norm, color ='#69635E')
    myPlotSettings_splitAxis(fig, ax, '', '\u0394 Frequency (octaves)', '',mySize=6)
    plt.xlim([0,0.5])
    plt.ylim([0,1])
    plt.xticks([0,0.25,0.5],['0','0.25','0.5'])
    plt.yticks([0,0.5, 1],['0','50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)    

    
    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\deltaFreq_motorSub_dist.svg'))


def plotLocationChange(df, ops):
    
    df_fit = np.load(os.path.join(ops['dataPath'],'locations_dataset','df_fit_1d_green_aud_GLM.npy'), allow_pickle=True).item()
    df_fit_motorSub = np.load(os.path.join(ops['dataPath'],'locations_dataset','df_fit_1d_green_aud_GLM_motorSub.npy'), allow_pickle=True).item()
    resp_green = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'resp_green_measured.npy'))
    sel_green = np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'sel_azimuth_green_measured.npy'))

    maps_green_aud_GLM =  np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'maps_green_audGLM.npy'))
    maps_green_aud_motorSub =  np.load(os.path.join(ops['dataPath'], 'locations_dataset', 'maps_green_audGLM_motorSub.npy'))

    color_motorSub = ops['color_motorSub']

    #########################################
   
    fig = plt.figure(figsize=(ops['mm']*72,ops['mm']*24), constrained_layout=True)
    ax = fig.add_subplot(1,3,1)
    
    maps0 = maps_green_aud_GLM[resp_green]
    maps0_motorSub  = maps_green_aud_motorSub[resp_green]

    nRois = maps0.shape[0]
    signal_corrs= np.zeros(nRois,)

    for roi in tqdm(range(nRois)):
        data = np.squeeze(maps0[roi,::].reshape(-1,1))
        data_motorSub = np.squeeze(maps0_motorSub[roi,::].reshape(-1,1))

        notNan0 = np.nonzero(~np.isnan(data))[0]
        notNan1 = np.nonzero(~np.isnan(data_motorSub))[0]
        notNan = np.intersect1d(notNan0,notNan1)

        r, p = stats.pearsonr(data[notNan],data_motorSub[notNan])
        signal_corrs[roi] = r

    bins_corr = np.arange(-1,1.05, 0.05)
    hist_norm, bins =np.histogram(signal_corrs,bins_corr)
    hist_norm = hist_norm/np.sum(hist_norm)
    plt.hist(bins[:-1], bins, weights = hist_norm, color ='#69635E')
    myPlotSettings_splitAxis(fig, ax, 'Percentage of \nboutons (%)', 'Signal corr.', '',mySize=6)
    plt.xlim([0.5,1])
    plt.ylim([0,1])
    plt.xticks([0.5,0.75, 1],['0.5','0.75', '1'])
    plt.yticks([0,0.5, 1],['0','50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
    
    ########################################################
    gaussFit = np.nonzero(np.array(df_fit['r2_gauss']) > 0.6)[0]
        
    peak = df_fit['gaussian_peak'][gaussFit]
    peak_motorSub = df_fit_motorSub['gaussian_peak'][gaussFit]

    D, p = stats.kstest(peak,peak_motorSub) 
    print(p)
    ax = fig.add_subplot(1,3,2)

    bins_peak = np.arange(0,13, 1.2)
    hist_norm, bins =np.histogram(peak,bins_peak)
    hist_norm = hist_norm/np.sum(hist_norm)   
    plt.hist(bins[:-1],bins,weights = hist_norm, color = 'k',histtype='stepfilled', alpha = 0.2)           
    plt.hist(bins[:-1],bins,weights = hist_norm, color = 'k',histtype='step',linewidth = 0.5, alpha = 1)  

    hist_norm, bins =np.histogram(peak_motorSub,bins_peak)
    hist_norm = hist_norm/np.sum(hist_norm)   
    plt.hist(bins[:-1],bins,weights = hist_norm, color = color_motorSub,histtype='stepfilled', alpha = 0.2)           
    plt.hist(bins[:-1],bins,weights = hist_norm, color = color_motorSub,histtype='step',linewidth = 0.5, alpha = 1)  
    myPlotSettings_splitAxis(fig, ax, '', '', '',mySize=6) 
    plt.xticks([0,6,12], ['-108', '0', '108'])
    plt.ylim([0,0.2])
    plt.yticks([0,0.1, 0.2], ['0','10','20'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1)  
   
    #################################################
    ax = fig.add_subplot(1,3,3)
    bins_delta = np.arange(0,1.666, 0.1)
    delta = abs(peak - peak_motorSub)
    hist_norm, bins =np.histogram(delta,bins_delta)
    hist_norm = hist_norm/np.sum(hist_norm)
    plt.hist(bins[:-1], bins, weights = hist_norm, color ='#69635E')
    myPlotSettings_splitAxis(fig, ax, '', '\u0394 Best azimuth (\u00B0)', '',mySize=6)
    plt.xlim([0,1.7])
    plt.ylim([0,1])
    plt.xticks([0,0.83333, 1.66666],['0','15', '30'])
    plt.yticks([0,0.5, 1],['0','50', '100'])
    ax.tick_params(axis='y', pad=1)   
    ax.tick_params(axis='x', pad=1) 
   
    #fig.savefig(os.path.join('Z:\\home\\shared\\Alex_analysis_camp\\paperFigures\\Plots\\deltaAzimuth_motorSub_dist.svg'))

    
def plotVarexpRatio(varExp, ops):
    #plot variance explained ratio, not used atm
    varExp_stim = varExp['varExp_stim_all']
    varExp_motor = varExp['varExp_motor_all']
    varExp_stim[varExp_stim < 0] =0
    varExp_motor[varExp_motor < 0] =0

    varExp_ratio = (varExp_stim  - varExp_motor)/(varExp_stim  + varExp_motor)
    full_idx = sig_GLM['sigFull_idx']
    plt.figure()
    plt.hist(varExp_ratio[full_idx], 20)


    resp_boutons_idx= np.load(os.path.join(ops['dataPath'], 'locations_dataset','responsive_idx_coliseum_axons.npy'))
    full_resp_idx = np.intersect1d(sig_GLM['sigFull_idx'], resp_boutons_idx)
    plt.figure()
    plt.hist(varExp_ratio[full_resp_idx], 20)

    U, p = stats.mannwhitneyu(varExp_ratio, varExp_ratio_resp)

    fig = plt.figure(figsize=(self.mm*37,self.mm*34), constrained_layout=True)
    bins_weights  = np.arange(-1,1.01,0.1)
    ax = fig.add_subplot(1,1,1)
    hist_w, bins = np.histogram(varExp_ratio_full,bins_weights)
    hist_w_norm = hist_w/np.sum(hist_w)
    plt.hist(bins[:-1],bins,weights = hist_w_norm, color = 'k', histtype ='bar', linewidth = 2)   
    hist_w, bins = np.histogram(varExp_ratio_resp,bins_weights)
    hist_w_norm = hist_w/np.sum(hist_w)
    plt.hist(bins[:-1],bins,weights = hist_w_norm, color = 'b', histtype ='step', linewidth = 1)
    plt.vlines(0.0,0, 0.1, linewidth =0.5, linestyle='--', color ='gray')
    plt.scatter(np.median(varExp_ratio_full), 0.15, marker ='v', s= 10, color = 'k')
    plt.scatter(np.median(varExp_ratio_resp), 0.15, marker ='v', s= 10, color = 'b')

    myPlotSettings_splitAxis(fig, ax, '','','')
    plt.xlim([-1.05,1.05])
    plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5','1'])
    plt.ylim([0,0.15])
    plt.yticks([0,0.05, 0.1, 0.15], ['0', '0.05', '0.1','0.15'])

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import stats
import statsmodels.stats.multitest
from scipy.stats import percentileofscore

#from sklearn.neighbors import KernelDensity
import os
from tqdm import tqdm
#Simport sys
import pandas as pd
#import seaborn as sns
import imageio
import itertools
#import random

def makeDataPaths(dataPath, animalPaths):
    paths =[]
    for p in range(len(animalPaths)):
        path = os.path.join(dataPath, animalPaths[p])
        paths.append(path)
    return paths

def getBootstrapResult(path,name, nBatch,ops, doMultiCorr=1):

    boot_df = {}
    for i in range(nBatch):
        res = np.load(os.path.join(path, name + '_' + str(i) + '.npy'),allow_pickle=True).item()
        if i ==0:
            if 'groups' in res:
                groups = res['groups']
            else:
                groups = ops['areas']
        res.pop('groups',None)
        items = res.keys()
        for j in items:
            if i ==0:
                boot_df[j] = res[j]
            else:
                boot_df[j] = np.concatenate((boot_df[j],res[j]),1)
                  
    pairs = list(itertools.combinations(range(0, len(groups)), 2))
#   
    p_difference_quantiles_list = np.zeros(len(pairs),)
    for i in range(len(pairs)):
        diff_this = boot_df['median_diff'][i,:]
        notNan = np.nonzero(np.isnan(diff_this) < 0.5)[0]
        diff_notNan = diff_this[notNan]
        target_value = 0
        percentile = percentileofscore(diff_notNan, target_value, kind='rank')
        quantile = percentile / 100
        
        p = 2*np.amin([quantile, 1-quantile])
        p_difference_quantiles_list[i] = p
        
    if doMultiCorr:
          p_difference_quantiles_list = statsmodels.stats.multitest.multipletests(p_difference_quantiles_list, method='fdr_bh')[1]  

    median_dist_mat = np.empty((len(groups), len(groups))); median_dist_mat[:] = np.nan
    p_difference_quantiles =  np.empty((len(groups), len(groups))); p_difference_quantiles[:] = np.nan
    sigLevels_quantiles =  np.empty((len(groups), len(groups))); sigLevels_quantiles[:] = np.nan
    for i in range(len(pairs)):
        pos0 = pairs[i][0]
        pos1 = pairs[i][1]
        median_dist_mat[pos1,pos0] = np.nanmedian(abs(boot_df['median_diff'][i,:]))
        
        p = p_difference_quantiles_list[i]
        p_difference_quantiles[pos0,pos1] = p
        
        if p < 0.001:
            sigLevels_quantiles[pos0,pos1] = 3
        elif p >= 0.001  and p < 0.01:
            sigLevels_quantiles[pos0,pos1] = 2
        elif p >= 0.01  and p < 0.05:
            sigLevels_quantiles[pos0,pos1] = 1
        elif p > 0.05:
            sigLevels_quantiles[pos0,pos1] = 0
                       
    return median_dist_mat,p_difference_quantiles,sigLevels_quantiles,groups


def makeProportions_bySpatialBin_v3(df,binned_map, idx, thresh = 0, mask = 'none', V1_mask=[]):
    chance = len(idx)/len(df)
    
    x = np.array(df['x'])
    y = np.array(df['y'])
    
    binIndices = binned_map[y.astype(int),x.astype(int)]            
    binIndices_unique = np.unique(binIndices)
    
    binned_mean_map = np.empty((binned_map.shape[0],binned_map.shape[1])); binned_mean_map[:] = np.nan

    meanVal = []
    for b in tqdm(binIndices_unique):
        binPos = np.nonzero(binned_map == b)
        binRange_y = np.arange(min(binPos[0]),max(binPos[0])+1)
        binRange_x = np.arange(min(binPos[1]),max(binPos[1])+1)
        
        binCentre_y = int((binRange_y[0] + binRange_y[-1])/2)
        binCentre_x = int((binRange_x[0] + binRange_x[-1])/2)
        if mask == 'HVAs':
            values_onMask = V1_mask[binCentre_y,binCentre_x] 
            if all(values_onMask == [227, 6, 19, 255]):
                # print('Excluded V1 bin')
                continue
        elif mask == 'V1':
            values_onMask = V1_mask[binCentre_y,binCentre_x] 
            if not all(values_onMask == [227, 6, 19, 255]):
                # print('Excluded V1 bin')
                continue
        
        idx_thisBin = []
        for j in range(len(df)):
            if df['x'].iloc[j] in binRange_x and df['y'].iloc[j] in binRange_y:
                idx_thisBin.append(j)
                
        if len(idx_thisBin) > thresh:    
            idx0_thisBin = np.intersect1d(idx, idx_thisBin)
            prop_thisBin = (len(idx0_thisBin)/len(idx_thisBin))#/chance
        else:
            prop_thisBin = np.nan
        binned_mean_map[min(binRange_y):max(binRange_y), min(binRange_x):max(binRange_x)] = prop_thisBin
  
    return binned_mean_map


def makeMeanValue_bySpatialBin_v2(df,binned_map, thresh = 0, varName = [], mask = 'V1', V1_mask=[]):
    
    x = np.array(df['x'])
    y = np.array(df['y'])
    
    binIndices = binned_map[y.astype(int),x.astype(int)]            
    binIndices_unique = np.unique(binIndices)
    
    if len(varName) > 0:
        name = varName
    else:
        name = df.keys()[0]
    
    binned_mean_map = np.empty((binned_map.shape[0],binned_map.shape[1])); binned_mean_map[:] = np.nan

    meanVal = []
    for b in tqdm(binIndices_unique):
        binPos = np.nonzero(binned_map == b)
        binRange_y = np.arange(min(binPos[0]),max(binPos[0])+1)
        binRange_x = np.arange(min(binPos[1]),max(binPos[1])+1)
        
        binCentre_y = int((binRange_y[0] + binRange_y[-1])/2)
        binCentre_x = int((binRange_x[0] + binRange_x[-1])/2)
        if mask == 'HVAs':
            values_onMask = V1_mask[binCentre_y,binCentre_x] 
            if all(values_onMask == [227, 6, 19, 255]):
                # print('Excluded V1 bin')
                continue
        elif mask == 'V1':
            values_onMask = V1_mask[binCentre_y,binCentre_x] 
            if not all(values_onMask == [227, 6, 19, 255]):
                # print('Excluded V1 bin')
                continue

        vals_thisBin = []
        for j in range(len(df)):
            if df['x'].iloc[j] in binRange_x and df['y'].iloc[j] in binRange_y:
                vals_thisBin.append(df[varName].iloc[j])
                
        if len(vals_thisBin) > thresh:          
            mean_thisBin = np.nanmean(np.array(vals_thisBin))
        else:
            mean_thisBin = np.nan
        binned_mean_map[min(binRange_y):max(binRange_y), min(binRange_x):max(binRange_x)] = mean_thisBin
  
    return binned_mean_map

def get_critical_ranksum(n1, n2, alpha, tail = 'two-sided'):
    from scipy.stats import norm

    # Mean of the U distribution
    mean_u = n1 * n2 / 2

    # Standard deviation of the U distribution
    sd_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    # Critical z-value for the desired alpha
    if tail == 'two-sided':
        z_critical = norm.ppf(1 - alpha / 2)
    elif tail == 'one-sided':
        z_critical = norm.ppf(1 - alpha)
    else:
        raise ValueError("tail must be 'two-sided' or 'one-sided'")

    # Critical value of U
    u_critical_lower = mean_u - z_critical * sd_u
    u_critical_upper = mean_u + z_critical * sd_u

    return  u_critical_lower, u_critical_upper

def doLinearRegression(x,y):
    from sklearn.linear_model import LinearRegression

    #remove nans
    nanIdx1 = np.nonzero(np.isnan(x))[0] 
    nanIdx2 = np.nonzero(np.isnan(y))[0] 
    nanIdx = np.unique(np.concatenate((nanIdx1,nanIdx2),0))
    x = np.delete(x,nanIdx).reshape((-1,1))
    y = np.delete(y,nanIdx)
     
    reg = LinearRegression().fit(x, y) #fit
     
    r2 = reg.score(x,y)  #get fit parameters
    intercept = reg.intercept_
    slope = reg.coef_
    
    x_vals = np.arange(min(x),max(x), 0.01) #make line from slope and intercept for plotting
    y_vals = intercept + slope*x_vals
    
    corr, pVal_corr = stats.pearsonr(np.squeeze(x),np.squeeze(y))
    
    result_dict = {'r2': r2,
                   'intercept': intercept,
                   'slope': slope,
                   'x_vals': x_vals,
                   'y_vals': y_vals,
                   'corr': corr,
                   'pVal_corr': pVal_corr}
    
    return result_dict


def doLinearRegression_withCI(x,y):
    import statsmodels.api as sm
    
    nanIdx1 = np.nonzero(np.isnan(x))[0] 
    nanIdx2 = np.nonzero(np.isnan(y))[0] 
    nanIdx = np.unique(np.concatenate((nanIdx1,nanIdx2),0))
    x = np.delete(x,nanIdx).reshape((-1,1))
    y = np.delete(y,nanIdx)

    X = sm.add_constant(x)  # Adds intercept term
    model = sm.OLS(y, X)
    results = model.fit()
    
    x_vals = np.arange(min(x),max(x), 0.1) #make line from slope and intercept for plotting
    X_pred = sm.add_constant(x_vals)
    predictions = results.get_prediction(X_pred)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

    corr, pVal_corr = stats.pearsonr(np.squeeze(x),np.squeeze(y))
    intercept = results.params[0]
    slope =  results.params[1]
    
    result_dict = {'r2': results.rsquared,
                   'intercept': intercept,
                   'slope': slope,
                   'x_vals': x_vals,
                   'y_vals':np.array(pred_summary['mean']),
                   'corr': corr,
                   'pVal_corr': pVal_corr, 
                   'ci_lower':np.array(pred_summary["mean_ci_lower"]),
                   'ci_upper':np.array(pred_summary["mean_ci_upper"])}
    
    return result_dict


def smooth_spatialBins(binned_values_map, spatialBin =300, nSmoothBins=1):
    
    nY,nX = binned_values_map.shape
    
    um_per_pixel = 9 #this is from measuring the WF FOV size
     
    spatialBin_px = int(spatialBin/um_per_pixel)
    smooth_range = spatialBin_px*nSmoothBins
    
    binned_vals_map_smooth = binned_values_map.copy()
    for x in range(nX):
        xRange = np.arange(x-smooth_range-1, x+smooth_range +1)
        if any(xRange < 0):
            xRange =  np.arange(0, x+smooth_range+1)
        if any(xRange >= nX):
            xRange =  np.arange(x-smooth_range-1, nX-1)
            
        for y in range(nY):
            if not np.isnan(binned_values_map[y,x]):
                yRange = np.arange(y-smooth_range-1, y+smooth_range+1)
                if any(yRange < 0):
                    yRange =  np.arange(0, y+smooth_range+1)
                if any(yRange >= nY):
                    yRange =  np.arange(y-smooth_range-1, nY-1)
                    
                vals0 = binned_values_map[yRange,:]
                vals = vals0[:,xRange]
                uniqueVals = np.unique(vals)
                
                smoothVal = np.nanmean(uniqueVals)
                
                binned_vals_map_smooth[y,x] = smoothVal
                
    return binned_vals_map_smooth

def makeSpatialBinnedMap(ref,spatialBin = 100):
    
    um_per_pixel = 9 #this is from measuring the WF FOV size
     
    spatialBin_px = int(spatialBin/um_per_pixel)
            
    nY,nX = ref.shape[0:2]

    nBins_x = np.ceil(nX/spatialBin_px)
    nBins_y = np.ceil(nY/spatialBin_px)
                    
    X,Y = np.meshgrid(np.arange(0,nBins_x), np.arange(0,nBins_y))
        
    Z = X*100 + Y
        
    matrix = np.repeat(Z, spatialBin_px, axis=1).repeat(spatialBin_px, axis=0)            
    matrix = matrix[0:nY,0:nX]
            
    return matrix

def getBinValues(binned_map, binned_map_values, map_colors, colors_LUT):
    bins_unique = np.unique(binned_map)

    bins, values, positions = [],[],[]
    for b in range(len(bins_unique)):
        idx = np.nonzero(binned_map == bins_unique[b])
        y,x = np.mean(idx[0]), np.mean(idx[1])
        positions.append([x,y])
        
        bins.append(binned_map[int(y),int(x)])
        values.append(binned_map_values[int(y),int(x)])
        
    bins_df =pd.DataFrame({'binIdx': np.array(bins), 
                           'values': np.array(values)})
    
    binAreas = []
    for roi in range(len(positions)):
        color = map_colors[int(positions[roi][1]), int(positions[roi][0])]
        
        f = [color == colors_LUT['colors'][i] for i in range(len(colors_LUT['colors']))]
        f = [all(f[i]) for i in range(len(f))]
        foundIt = any(np.array(f)) 
        
        if foundIt:
            idx = np.nonzero(np.array(f) > 0.5)[0]
            area = colors_LUT['areas'][int(idx)]
        else:
            if np.sum(color) == 765:
                area = 'OUT'
            else: #if color not an area color and not white, its on the border. Search for closest color that matches an area color and assign that area
                increments = np.arange(1,5)
                cnt = 0
                for incr in increments: #search around the centre pixel with increasing radius
                    #incr =1
                    yVals = np.arange(positions[roi][1] - incr,positions[roi][1] + incr +1)
                    xVals = np.arange(positions[roi][0] - incr,positions[roi][0] + incr +1)
                    
                    x = np.repeat(xVals,len(yVals))
                    y = np.tile(yVals,len(xVals))
                    
                    pixels = np.array([x,y])                        
                    for p in range(len(pixels)):
                        thisCol = map_colors[int(pixels[1,p]), int(pixels[0,p])]
                        f = [thisCol == colors_LUT['colors'][i] for i in range(len(colors_LUT['colors']))]
                        f = [all(f[i]) for i in range(len(f))]
                        foundIt = any(np.array(f)) 
                        if foundIt:
                            idx = np.nonzero(np.array(f) > 0.5)[0]
                            area = colors_LUT['areas'][int(idx)]
                            cnt = 1
                            break
                    
                    if cnt:
                        break
        
        binAreas.append(area)
        
         
    bins_df =pd.DataFrame({'binIdx': np.array(bins), 
                           'values': np.array(values),
                           'binArea': np.array(binAreas)})
    
    return bins_df


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def makeSessionReference(df,varName=[]):
    sessionIdx = df['sessionIdx'].unique()  
    #
    dorsal = ['AM', 'PM', 'A', 'RL'] 
    ventral = ['P', 'POR', 'LI', 'LM', 'AL']
    
    abs_index, rel_index = [],[]
    seshX, seshY, seshAzi, seshElev, seshAreas, seshAnimal, seshSource, seshIdx, seshVar, seshStream, seshBatch, seshMapGood = [],[],[],[],[],[],[],[],[],[],[],[]
    pos_DV, pos_AP, prop_ventral = [],[],[]
    for sesh in range(len(sessionIdx)):
        session = sessionIdx[sesh]
              
        idx_thisSession_rel = np.nonzero(np.array(df['sessionIdx']) == session)[0]
        df_thisSession = df.iloc[idx_thisSession_rel]
        idx_thisSession_abs = np.array(df_thisSession.index)   
                  
        rel_index.append(idx_thisSession_rel)
        abs_index.append(idx_thisSession_abs)
        
        #area
        theseAreas = np.array(df_thisSession['area'])
        areas1, counts = np.unique(theseAreas, return_counts=True)
                   
        if 'OUT' in areas1:
            t = np.nonzero(areas1 == 'OUT')[0]
            areas1 = np.delete(areas1, t)
            counts = np.delete(counts, t)
        
        if len(areas1) > 0:  
            this = areas1[np.argmax(counts)]
        else:
            this = 'OUT'
            
        seshAreas.append(this)
            
        if this in dorsal:
            seshStream.append('Dorsal')
        elif this in ventral:
            seshStream.append('Ventral')
        elif this =='V1':
            seshStream.append('V1')
        else:
            seshStream.append('OUT')
                       
            
        #retinotopy
        if df_thisSession['isMapGood'].iloc[0] == 0:
            seshAzi.append(np.nanmedian(np.array(df_thisSession['azi'])))
            seshElev.append(np.nanmedian(np.array(df_thisSession['elev'])))
        else:
            seshAzi.append(np.nanmedian(np.array(df_thisSession['azi_orig'])))
            seshElev.append(np.nanmedian(np.array(df_thisSession['elev_orig'])))
            
        #anat.loc.
        seshX.append(np.nanmean(np.array(df_thisSession['x'])))
        seshY.append(np.nanmean(np.array(df_thisSession['y'])))
        
        #animal
        animal = np.array(df_thisSession['animal'])[0]
        seshAnimal.append(animal)
        
        #batch
        if animal < 149:
            batch = 1
        else:
            batch = 2
        seshBatch.append(batch)
        
        if len(varName) >0:
            seshVar.append(np.nanmean(np.array(df_thisSession[varName])))
        
        #source
        #
        if 'pos_DV' in df_thisSession.keys():
            pos_DV.append(df_thisSession['pos_DV'].iloc[0])
        if 'pos_DV' in df_thisSession.keys():
            pos_AP.append(df_thisSession['pos_AP'].iloc[0])
        if 'prop_ventral' in df_thisSession.keys():
            prop_ventral.append(df_thisSession['prop_ventral'].iloc[0])
          
        #sessionIdx 
        sessionIdx0 = np.array(df_thisSession['sessionIdx'])[0]
        seshIdx.append(sessionIdx0)
        
        #map good
        seshMap = np.array(df_thisSession['isMapGood'])[0]
        seshMapGood.append(seshMap)

    sessionRef = {'abs_index' : abs_index,
              'rel_index' : rel_index,
              'seshAzi' : seshAzi,
              'seshElev' : seshElev,
              'seshX' : seshX,
              'seshY' : seshY,
              'seshAreas': seshAreas, 
              'seshAnimal': seshAnimal, 
               'myVar': seshVar, 
               'seshStream': seshStream,
               'seshBatch':np.array(seshBatch),
              # 'seshSource': seshSource,#
              'seshIdx': seshIdx, 
              'seshMapGood': seshMapGood, 
              'prop_ventral': prop_ventral,
              'pos_DV': pos_DV,
              'pos_AP': pos_AP}
            
    return sessionRef


def makeProportions_bySession_v2(df, ref_df, thresh =10):
     
    sessions = ref_df['sessionIdx'].unique()
    prop = []
    for sesh in sessions:
        ref_thisSession = ref_df[ref_df['sessionIdx'] == sesh]
        df_thisSession = df[df['sessionIdx'] == sesh]
        if len(df_thisSession) < thresh:
            thisProp = np.nan
        else:
            thisProp = len(df_thisSession)/len(ref_thisSession)
        prop.append(thisProp)
        
    return prop


def asignAreaToSession(df, policy='mostRois'):
    sessions = df['sessionIdx'].unique()
    
    areas, animals = [],[]
    for sesh in sessions:
        df_thisSession = df[df['sessionIdx'] == sesh]
        animals.append(int(df_thisSession['animal'].unique()))
        area_thisSession, counts = np.unique(np.array(df_thisSession['area']), return_counts=True)
        
        if 'OUT' in area_thisSession:
            t = np.nonzero(area_thisSession == 'OUT')[0]
            area_thisSession = np.delete(area_thisSession, t)
            counts = np.delete(counts, t)
        
        if len(area_thisSession) > 0:
            if policy == 'mostRois':
                area = area_thisSession[np.argmax(counts)]
                areas.append(area)
        else:
            areas.append('OUT')

    areas = np.array(areas)
    
    areaBySession = {'areas': areas,
                     'sessionIdx': sessions,
                     'animals': animals}
    
    return areaBySession

def divideSessionsByArea(prop_bySession, areas, areaBySession):
   
    prop_byArea = []
    for area in areas:
        t = np.nonzero(areaBySession['areas'] == area)[0]
        theseProps = np.array([prop_bySession[t[i]] for i in range(len(t))])
        prop_byArea.append(theseProps)
        
    return prop_byArea  


def classifyFreqSplines(freq_curve, threshold):
    nRois = freq_curve.shape[0]
    
    doublePeak, singlePeak = [],[]
    for roi in tqdm(range(nRois)):
        freq = freq_curve[roi,:]
        
        if all(freq < 0):
            freq = abs(freq)
        #actual spline fitting:
        maxVal = max(freq)
        thresh = maxVal*threshold #7
        
        aboveThresh = np.nonzero(freq > thresh)[0]
        jump = np.diff(aboveThresh)
        if len(aboveThresh) ==1:
            singlePeak.append(roi)
        else:
            if np.max(jump) > 1: #double peaked
                doublePeak.append(roi)
            else:
                singlePeak.append(roi)

    singlePeak = np.array(singlePeak) 
    doublePeak = np.array(doublePeak)    
    
    return singlePeak,doublePeak    


def doMannWhitneyU_forBoxplots(data, multiComp = 'fdr'):
    #data will be a list
    
    nBox = len(data)
    
    pVals, compIdx = [],[]
    for b in range(nBox):
        for b1 in range(nBox):
            if b < b1:           
                if len(data[b]) > 0 and len(data[b1]) > 0:
                    t, p = stats.mannwhitneyu(data[b],data[b1])
                    pVals.append(p)
                else:
                    pVals.append(np.nan)
                compIdx.append(str(b) + '_' + str(b1))

    n_comp = len(compIdx)
    pVals = np.array(pVals)
    
    if multiComp == 'bonferroni':
        pVals_adj = pVals*n_comp
    elif multiComp == 'fdr':       
        pVals_adj = statsmodels.stats.multitest.multipletests(pVals, method='fdr_bh')[1]   
        # pVals_adj = statsmodels.stats.multitest.multipletests(pVals, method='hs')[1]                        
    else:
        pVals_adj = pVals

    return pVals_adj, compIdx


def getElevation_greenAud(df, maps, peak, onlyPeakSide = 1):

    leftBorder = 4.4
    rightBorder = 7.6

    left_tuned = np.nonzero(peak < leftBorder)[0]
    right_tuned = np.nonzero(peak > rightBorder)[0]
    centre_tuned0 = np.setdiff1d(np.arange(0,len(peak)), left_tuned)
    centre_tuned1 = np.setdiff1d(np.arange(0,len(peak)), right_tuned)
    centre_tuned = np.intersect1d(centre_tuned0, centre_tuned1)
    
    elevPeak = np.zeros(len(df),)
    
    if onlyPeakSide:

        for i in range(len(df)):
            thisMap = maps[i,::]
            
            if df['animal'].iloc[i] < 149:
                if i in left_tuned:
                    this = np.nanmean(thisMap[0:3],0)
                elif i in centre_tuned:
                    this =thisMap[3,:]
                elif i in right_tuned:
                    this = np.nanmean(thisMap[4:7],0)
            else:
                if i in left_tuned:
                    this = np.nanmean(thisMap[0:5],0)
                elif i in centre_tuned:
                    this = np.nanmean(thisMap[5:8],0)
                elif i in right_tuned:
                    this = np.nanmean(thisMap[8::],0)    
            
            elevPeak[i] = np.argmax(this)
    else:
        
        for i in range(len(df)):
            thisMap = maps[i,::]
            this = np.nanmean(thisMap,0)
            elevPeak[i] = np.argmax(this)
        
    
    elevPeak[elevPeak ==0] = 4   
    elevPeak[elevPeak ==2] = 0      
    elevPeak[elevPeak ==1] = 2  
    
    return elevPeak


def getSparsityIdx(maps):
    
    def sparseIdx_inv(responses):
        si0 =  (np.nansum(responses/len(responses)))**2/np.nansum(((responses**2)/len(responses)))  
        si = (1-si0)/(1-(1/len(responses)))
        return si
    freqs = maps
  
    freqs = np.reshape(freqs, (freqs.shape[0], freqs.shape[1]*freqs.shape[2]))
  
    nRois = freqs.shape[0]
            
    all_si = []
    for roi in range(nRois):
        these = freqs[roi,:]
        these[these < 0] = 0
        si = sparseIdx_inv(these)
        all_si.append(si)
    all_si = np.array(all_si)
    
    return all_si

def getBootstrapDiffP(diff):        
    sort_diff = np.sort(diff)
    firstAbove = np.nonzero(sort_diff < 0)[0]
    if len(firstAbove) == 0:
        pVal = '< ' + str(1/len(diff))
    else:
        pVal = str(len(firstAbove)/len(diff))
        
    return pVal

def interpolateAzimuth_coliseum(df):
    idx_0 = np.nonzero(np.array(df['animal']).astype(int) < 149)[0]
    idx_1 = np.nonzero(np.array(df['animal']).astype(int) >= 149)[0]
    
    df_0 = df.iloc[idx_0]
    df_1 = df.iloc[idx_1]
    
    peakAzi_0 = np.array(df_0['aziPeak'])
    
    max0 = int(np.round(max(df_0['aziPeak'])))
    min0 = 0
    small_range = np.linspace(min0, max0, int(max0*100))
    max1 = int(np.round(max(df_1['aziPeak'])))
    min1 =0
    large_range = np.linspace(min1, max1, int(max0*100))
    
    data_interp = np.array([np.interp(peakAzi_0[k],small_range,large_range) for k in range(len(peakAzi_0))])
    df_0['aziPeak'] = data_interp
    
    if 'elevPeak' in df.keys():
        peakElev_0 = np.array(df_0['elevPeak']) 
        
        max0 = int(np.round(max(df_0['elevPeak'])))
        min0 = 0
        small_range = np.linspace(min0, max0, int(max0*100))
        max1 = int(np.round(max(df_1['elevPeak'])))
        min1 =0
        large_range = np.linspace(min1, max1, int(max0*100))
    
        data_interp = np.array([np.interp(peakElev_0[k],small_range,large_range) for k in range(len(peakElev_0))])
        df_0['elevPeak'] = data_interp

    df_interp = pd.concat([df_0, df_1])

    return df_interp

def getDict_fromMatlabStruct(mat_file, struct_name):
    
    # Access the MATLAB structure by its name
    mat_struct = mat_file[struct_name]

    # Convert the MATLAB structure to a Python dictionary
    py_dict = {name: np.array(mat_struct[name][0]) for name in mat_struct.dtype.names}
    
    py_dict2 = {name: np.array(py_dict[name][0]) for name in py_dict.keys()}

    return py_dict2


def doWilcoxon_forBoxplots(data, multiComp = 'hs'):
    # Data should be a numpy array with the following shape: CatagoricalVar x observationsPerCategory    
    nBox, nObservations = data.shape
    
    pVals, compIdx = [],[]
    for b in range(nBox):
        for b1 in range(nBox):
            if b < b1:         
                goodIdx_1= np.nonzero(~np.isnan(data[b,:]))[0]
                goodIdx_2= np.nonzero(~np.isnan(data[b1,:]))[0]
                goodIdx = np.intersect1d(goodIdx_1, goodIdx_2)
                
                t, p = stats.wilcoxon(data[b,goodIdx],data[b1,goodIdx],zero_method = 'pratt')
                pVals.append(p)
                compIdx.append(str(b) + '_' + str(b1))
            
    n_comp = len(compIdx)
    pVals = np.array(pVals)
    
    if multiComp == 'bonferroni':
        pVals_adj = pVals*n_comp
    elif multiComp == 'hs':       
        pVals_adj = statsmodels.stats.multitest.multipletests(pVals, method='hs')[1]  
    elif multiComp == 'fdr':       
        pVals_adj = statsmodels.stats.multitest.multipletests(pVals, method='fdr_bh')[1]   
    else:
        pVals_adj = pVals

    return pVals_adj, compIdx

def myPlotSettings_splitAxis(fig,ax,ytitle,xtitle,title,axisColor = 'k', mySize=7, myAxisSize =5):
    from matplotlib import font_manager

    font_dirs = ['C:\\Users\\egeaa\\Desktop\\myFonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    myFont = 'Arial'
    # mySize = 7 #18 for posters
    ax.spines['left'].set_color(axisColor)
    ax.spines['bottom'].set_color(axisColor)
    ax.xaxis.label.set_color(axisColor)
    ax.yaxis.label.set_color(axisColor)
    ax.tick_params(axis='x', colors=axisColor)
    ax.tick_params(axis='y', colors=axisColor)

    plt.rcParams["font.family"] = myFont
    # plt.rcParams["font.family"] = myFont

    plt.rcParams["font.size"] = mySize
    # ax.set_ylabel(ytitle)
    # ax.set_xlabel(xtitle)
    # ax.set_title(title,weight = 'bold')
    ax.set_ylabel(ytitle, fontname=myFont, fontsize=mySize, labelpad = 1)
    ax.set_xlabel(xtitle, fontname=myFont, fontsize=mySize)
    ax.set_title(title, fontname=myFont, fontsize=mySize, weight = 'bold')
    for tick in ax.get_xticklabels():
        tick.set_fontname(myFont)
        tick.set_fontsize(myAxisSize)        
    for tick in ax.get_yticklabels():
        tick.set_fontname(myFont)
        tick.set_fontsize(myAxisSize)    
    right = ax.spines["right"]
    right.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False) 
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.25)
    for line in ["left","bottom"]:
        ax.spines[line].set_linewidth(0.25)
        ax.spines[line].set_position(("outward",3))
        # ax.spines['bottom'].set_position(('data', 7)) 
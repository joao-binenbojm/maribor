from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from time import time

def eigen_cutoff(explained_var, thrs=0.99):
    ''' Determines eigenvalue at which the cumulative sum of explained variance reaches a designated threshold.'''
    cum_var = np.cumsum(explained_var)
    loc = np.argwhere(cum_var >= thrs)[0,0] # first index where explained variance goes above threshold
    return loc + 1 # eigenvalue at which explained variance goes above threshold

def sortMUAPs(muaps: np.ndarray, J: int):
    ''' Given MUAPs data array, and extracts indices of MUAPs in ascending order of amplitude variance.'''
    print('SORTING MUAPs...')
    totvars = [] # variance of MUAP sizes across electrodes
    for jdx in range(len(muaps[0, :])): # for each motor unit
        totvar = []
        for cdx in range(len(muaps[0,0][0,:])):
            for rdx in range(muaps[0,0][0,0].shape[0]):
                totvar.append(np.var(muaps[0, jdx][0, cdx][rdx,:])) # variability of given MU/Channel time impulse response
        totvars.append(totvar)
    avg_totvars = np.mean(totvars, axis=1) # variability of MU averaged across channels
    MUranks = np.flip(np.argsort(avg_totvars)) # indices for sorting muaps based on MUAP size
    return MUranks

def getH(data: np.ndarray, K: int, J: int) -> np.ndarray:
    ''' Returns mixing matrix based on simulated MUAPS.'''
    Jmax = len(data['MUAPs']) # total number of motor units
    ncols = len(data['MUAPs'][0, 0][0]) # number of columns in simulated grid
    nrows, L = data['MUAPs'][0, 0][0, 0].shape # number of rows in simulated grid and length of MUAP
    M = ncols*nrows # total number of channels in the grid
    H = np.zeros((K*M, J*(L + K - 1))) # initialize mixing matrix
    MUranks = sortMUAPs(data['MUAPs'], J=J) # get indices of MUs sorted by average variability

    for jdx in range(J): # for each motor unit
        jrank = MUranks[jdx] # which motor unit to extract
        for mdx in range(M): # for each channel
            for kdx in range(K): # for each delayed repetition
                muap = data['MUAPs'][0, jrank][0, mdx // nrows][mdx % nrows, :] # get the current muap
                interval = (L + K - 1)*jdx # interval between each AP shape
                H[mdx*K + kdx, interval+kdx:interval+kdx+L] = muap.ravel() # adding motor unit shapes to mixing matrix
    return H

if __name__ == '__main__':
    DIR = '../SynthSigs'
    name = 'SynthMUAP_BB_lib1_F5-5-5_Len20_ramp1_SNR10_20-Nov-2023_16_50'
    muap_data = loadmat(os.path.join(DIR, name))
    t0 = time()

    # Run desired simulation with different parameters
    data = [] # where to store each run

    for run in tqdm(range(1, 11)):
        print('Run #{}...'.format(run)) 
        for K in tqdm([40, 10, 1]):
            print('For K = {}...'.format(K))
            for J in np.arange(1, 152, 20):

                    # Obtain singular values
                    H = getH(muap_data, K=K, J=J)
                    _, sigmas,_ = np.linalg.svd(H)
                    eigs = sigmas ** 2 # square singular vals to get eigenvalues
                    eigs = eigs / np.sum(eigs)

                    # Compute eigen cutoffs
                    eig1 =  eigen_cutoff(eigs, thrs=0.95) # accounting for 95% explained variance
                    eig2 = eigen_cutoff(eigs, thrs=0.99) # accounting for 99% explained variance

                    data.append([run, K, J, eig1, eig2])
    
    columns = ['Run', 'K', 'J', 'eig1', 'eig2']
    df = pd.DataFrame(data=data, columns=columns) # store simulation results in a dataframe
    df.to_csv('sims.csv')

    # RUNTIME CALCULATION
    tf = time()
    print('Experiment took {} s'.format(tf - t0))
    print('Experiment took {} min'.format((tf - t0)/60))
    print('Experiment took {} h'.format((tf - t0)/3600))



    # H = getH(os.path.join(DIR, name), J=5)
    # nrows = H.shape[0]
    # max_amp = H.max()
    # plt.figure()
    # for idx in range(nrows):
    #     plt.plot(H[idx, :] + nrows - 3*max_amp*idx)
    # plt.show()

    # # Plotting covariance of H
    # cov = np.cov(H) # across rows
    # plt.figure()
    # sns.heatmap(cov)
    # plt.title('Covariance of simulated matrix')
    # plt.show()

    # # Plotting eigenspectrum of quadratic form of H
    # _, sigmas,_ = np.linalg.svd(H)
    # eigs = sigmas ** 2 # square singular vals to get eigenvalues
    # eigs = eigs / np.sum(eigs)
    # plt.figure()
    # plt.stem(eigs)
    # plt.title('Eigenspectrum of Simulated H')
    # plt.show()
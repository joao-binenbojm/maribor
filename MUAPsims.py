from scipy.io import loadmat
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

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

def getH(path: str, K: int, J: int) -> np.ndarray:
    ''' Returns mixing matrix based on simulated MUAPS.'''
    data = loadmat(path) # load data array
    Jmax = len(data['MUAPs']) # total number of motor units
    ncols = len(data['MUAPs'][0, 0][0]) # number of columns in simulated grid
    nrows, L = data['MUAPs'][0, 0][0, 0].shape # number of rows in simulated grid and length of MUAP
    M = ncols*nrows # total number of channels in the grid
    H = np.zeros((K*M, J*(L + K - 1))) # initialize mixing matrix
    MUranks = sortMUAPs(data, J=J) # get indices of MUs sorted by average variability

    for jdx in range(J): # for each motor unit
        jrank = MUranks[jdx] # which motor unit to extract
        for mdx in range(M): # for each channel
            for kdx in range(K): # for each delayed repetition
                muap = data['MUAPs'][0, jrank][0, mdx // nrows][mdx % nrows, :] # get the current muap
                interval = (L + K - 1)*jdx # interval between each AP shape
                H[mdx*K + kdx, interval+kdx:interval+kdx+L] = muap.ravel() # adding motor unit shapes to mixing matrix
    return H

def IA(sig_out: np.ndarray, K:int =1) -> np.ndarray:
    ''' Computes the index of activity based on the EMG and chosen extension factor.'''

    # Unfold the simulation EMG shape
    EMG = np.array([sig_out[idx, jdx][0,:] for idx in range(10) for jdx in range(9)])
    T = EMG.shape[1] # number of samples of EMG signal

    # Computing the mahalanobis distance of each vector
    EMG = np.concatenate((np.zeros((EMG.shape[0], K-1)), EMG), axis=1) # zero-padding given K-1 delayed repetitions
    EMG_extend = np.zeros((EMG.shape[0]*K, EMG.shape[1])) # where to store extended observations

    # Extend observations and compute covariance matrix
    print('Extracting covariance matrix...')
    for t in range(T): # for each sample
        delreps = EMG[:, t : t+K] # get all the samples needed for extended observations
        y = np.flip(delreps, axis=1).ravel() # flip observations to match representation in original paper
        EMG_extend[:, t]  = y # store in extended observations matrix
    cov_y = np.cov(EMG_extend) # extended observations covariance matrix
    cov_y_inv = np.linalg.inv(cov_y) # inverse of extended observations
    EMG_extend = EMG_extend - EMG_extend.mean(axis=1).reshape(-1, 1) # subtract mean from each row

    # Extract the index of activity at all given timepoints
    print('Extracting IA...')
    gamma = np.zeros(EMG.shape[1]) # as many values as samples available of EMG
    for t in range(T):
        gamma[t] = EMG_extend[:, t].T @ cov_y_inv @ EMG_extend[:, t] 
    
    return gamma

def generate_MUAPs(M, J, L):
    ''' Generate sine waves with guaranteed different frequencies.'''
    MUAPs = np.zeros((M, J, L))
    intervals = np.linspace(start=int(L/3), stop=L, num=M*J).reshape((M, J))
    for mdx in range(M):
        for jdx in range(J):
            f = 1/intervals[mdx, jdx] # frequency of given sine waveform
            MUAP = np.sin(2*np.pi*f*np.arange(intervals[mdx, jdx]))
            start = (L - len(MUAP))//2
            MUAPs[mdx, jdx, start:(start + len(MUAP))] = MUAP # add MUAP shape
            # plt.figure()
            # plt.plot(MUAPs[mdx, jdx, :])
            # plt.show()
    return MUAPs

def generate_random_MUAPs(M, J, L):
    ''' Generate random gaussian waveforms to use as independent 'MUAPs'.'''
    MUAPs = np.zeros((M, J, L))
    for mdx in range(M):
        for jdx in range(J):
            MUAPs[mdx, jdx, :] = np.random.normal(size=L) # add MUAP shape
    return MUAPs

def get_mixing_matrix(MUAPs, K=1):
    '''Based on genered MUAPs, get H.'''
    M, J, L = MUAPs.shape
    H = np.zeros((K*M, J*(L + K - 1)))
    for mdx in range(M): # for each channel
        for kdx in range(K): # for each delayed repetition
            for jdx in range(J): # for each motor unit
                interval = (L + K - 1)*jdx # interval between each AP shape
                H[mdx*K + kdx, interval+kdx:interval+kdx+L] = MUAPs[mdx, jdx, :].ravel() # adding motor unit shapes to mixing matrix
    
    return H

def generate_spikes(N, J, fr, fs):
    '''Generates spikes randomly by creating spike trains with the chosen properties.'''
    spike_trains = np.zeros((N, J))

    Tsamp = int(fs/fr) # number of samples in interval between spikes
    interval = int(fs/(fr*J)) # number of samples between firing rates of consecutive motor units

    init_train = []
    while len(init_train) < N: # generate spike train
        if N - len(init_train) > Tsamp: 
            init_train.extend([0]*(Tsamp - 1) + [1])
        else:
            init_train.extend([0]*(N - len(init_train)))

    # Compute individual spike trains
    spike_trains[:, 0] = init_train # initial spike train
    for jdx in range(1, J):
        spike_trains[:, jdx] = init_train[-interval*jdx:] + init_train[:-interval*jdx] # circularly roll signal

    return spike_trains


if __name__ == '__main__':
    DIR = '../SynthSigs'
    name = 'SynthMUAP_BB_lib1_F5-5-5_Len20_ramp1_SNR10_20-Nov-2023_16_50.mat'
    data = loadmat(os.path.join(DIR, name))
    print(data.keys())
    fs = data['fsamp'].ravel()
    firings = data['sFirings']
    sig_out = data['sig_out']
    frs = [(fs/(np.diff(firings[0, idx].ravel()).mean())) for idx in range(firings.shape[1])]

    # plt.figure()
    # plt.eventplot(firings[0,1].ravel())
    # plt.show()

    # plt.figure()
    # plt.plot(frs)
    # plt.ylim([0, 15])
    # plt.show()

    # plt.figure()
    # plt.plot(sig_out[0,0].ravel())
    # plt.show()

    # Validating the global index of activity
    T, fs = 20, 2048
    N = int(T * fs) # number of samples
    J = 20 # 20 MUs
    spike = generate_spikes(N=N, J=J, fr=12, fs=fs)
    spike = (spike - spike.mean(axis=0)) / spike.std(axis=0) # make unit variance so cov is identity 
    

    # gamma = IA(sig_out, K=15)
    # plt.figure()
    # plt.plot(gamma)
    # plt.show()

    # force = data['Force']
    # print(type(force))
    # print(force.shape)
    # muaps = data['MUAPs']
    # print(type(muaps))
    # print(muaps.shape)
    # fsamp = data['fsamp'][0,0]

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

    # Validating global index of activity

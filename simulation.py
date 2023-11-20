import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from tqdm import tqdm

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
    
    # plt.figure()
    # plt.plot(spike_trains)
    # plt.show()

    return spike_trains

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
            # plt.figure()
            # plt.plot(MUAPs[mdx, jdx, :])
            # plt.show()
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
    
    # fig, axs = plt.subplots(K*M, 1)
    # plt.figure()
    # for idx in range(K*M):
    #     # axs[idx].plot(H[idx, :])
    #     # axs[idx].set_title(idx)
    #     plt.plot(H[idx, :] + K*M - 2*(idx))
    # plt.show()
    return H

def eigen_cutoff(explained_var, thrs=0.99):
    ''' Determines eigenvalue at which the cumulative sum of explained variance reaches a designated threshold.'''
    cum_var = np.cumsum(explained_var)
    loc = np.argwhere(cum_var >= thrs)[0] # first index where explained variance goes above threshold
    return loc + 1 # eigenvalue at which explained variance goes above threshold

# Fitting exponential curve
def exponent(x, a, b):
    return a*np.exp(-b * x)

if __name__ == '__main__':
    df = pd.read_csv('sims.csv')
    df['M'] = df['M'].astype(int)
    df['J'] = df['J'].astype(int)
    plt.figure()
    sns.barplot(df[df['K'] == 1], x='J', y='eig1', hue='M')
    plt.show()
    # fs = 2000 # Hz in time
    # t = 100 # 1000s of data
    # N = int(fs*t) # length of virtual recording in samples
    # L = 40 # 40 samples MUAP, roughly corresponding to 20ms

    # # Run desired simulation with different parameters
    # data = [] # where to store each run

    # for run in tqdm(range(1, 11)):
    #     print('Run #{}...'.format(run)) 
    #     for K in tqdm([1, 10, 40, 100]):
    #         print('For K = {}...'.format(K))
    #         for M in np.arange(10, 61, 10):
    #             for J in np.arange(10, 61, 10):

    #                 # Obtain singular values
    #                 MUAPs = generate_random_MUAPs(M, J, L) 
    #                 H = get_mixing_matrix(MUAPs, K=K)
    #                 _, sigmas,_ = np.linalg.svd(H)
    #                 eigs = sigmas ** 2 # square singular vals to get eigenvalues
    #                 eig = eigs / np.sum(eigs)

    #                 # Obtain exponential fit and threshold crossing eigenvalues
    #                 x = np.arange(1, len(eigs) + 1)
    #                 y = eigs
    #                 popt, pcov = curve_fit(exponent, x, y)

    #                 eig1 =  eigen_cutoff(eigs, thrs=0.95) # accounting for 95% explained variance
    #                 eig2 = eigen_cutoff(eigs, thrs=0.99) # accounting for 99% explained variance

    #                 data.append([run, K, M, J, popt[0], popt[1], eig1, eig2])
    
    # columns = ['Run', 'K', 'M', 'J', 'a', 'b', 'eig1', 'eig2']
    # df = pd.DataFrame(data=data, columns=columns) # store simulation results in a dataframe
    # df.to_csv('sims.csv')

    # plt.figure()
    # plt.stem(x, y)
    # plt.plot(x, y_pred)
    # plt.title('Explained Variance of PCA(H)')
    # plt.xlabel('Component Number')
    # plt.ylabel('Explained Variance')
    # plt.legend(['Original', 'Exponential Fit'])
    # plt.show()




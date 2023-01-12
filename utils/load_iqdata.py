import numpy as np
from scipy.io import loadmat

def loadIQ(fpath, nlmfilt=True):
    matfile = loadmat(fpath)
    if nlmfilt == True:
        ens = np.abs(matfile['nlmfilterAT'])
    else:
        ens = np.abs(matfile['blinkbubble'])
    ens = ens / np.max(ens)
    return ens

def loadIQ_dirfilt(fpath, nlmfilt=True):
    matfile = loadmat(fpath)
    if nlmfilt == True:
        ens_pos = np.abs(matfile['nlmfilterAT_pos'])
        ens_neg = np.abs(matfile['nlmfilterAT_neg'])
    else:
        ens_pos = np.abs(matfile['blinkbubble_pos'])
        ens_neg = np.abs(matfile['blinkbubble_neg'])
    ens_pos = ens_pos/np.max(ens_pos)
    ens_neg = ens_neg/np.max(ens_neg)
    return ens_pos, ens_neg
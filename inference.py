import torch
import torch.nn.functional as F
from models.lstm_unet import UNet_ConvLSTM
import argparse
import numpy as np
from scipy.io import savemat
from utils import load_iqdata, smv_process_iq
import threading
import time
import socket
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import cmasher as cmr

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('-m', '--model', help='Input model file name', required=True)
parser.add_argument('-mf', '--modelfolder', help='Input model folder', required=True)
parser.add_argument('-fn', '--filename', help='Input IQ file name', required=True)
parser.add_argument('-ff', '--filefolder', help='Input IQ file folder folder', required=True)
parser.add_argument('-sp', '--savepath', help='Where to save output file', required = True)
parser.add_argument('-w', '--windowsize', help='Number of frames in each window', type = int,
                    default = 16)
parser.add_argument('-s', '--stepsize', help='Step size between two consecutive windows', type = int,
                    default = 16)
parser.add_argument('-rw', '--rollingwindow', help='Rolling window average size', type = int)
parser.add_argument('-bd', '--bidirectional', help='Whether to plot bidirectional map', type = int,
                    default = 0)
parser.add_argument('-df', '--directionalfilter', help='Whether input data is after directional filtering', type = int,
                    default = 1)


args = parser.parse_args()
model_path = os.path.join(args.modelfolder, args.model)
nt = args.windowsize
step_t = args.stepsize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

model = UNet_ConvLSTM(n_channels=1, n_classes=2, use_LSTM=True, parallel_encoder=False, lstm_layers=1)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
model = model.to(device)

state_dict = torch.load(model_path)['state_dict']
model.load_state_dict(state_dict, strict=False)

if args.directionalfilter == 1:
    # loading IQ data
    ens_pos, ens_neg = load_iqdata.loadIQ_dirfilt(args.filefolder+args.filename)
    # smv processing
    amplitude_arr, vmap, dmap = smv_process_iq.smv_dirfilt(ens_pos, ens_neg, model, args.windowsize, args.stepsize, device, single_frame = False)
else:
    # loading IQ data
    ens = load_iqdata.loadIQ(args.filefolder+args.filename)
    # smv processing
    amplitude_arr, vmap, angles_map, dmap = smv_process_iq.smv_estdir(ens, model, args.windowsize, args.stepsize, device, single_frame = False)

    
savemat(args.savepath, {'vmap_mean':vmap, 'amplitude_arr':np.asarray(amplitude_arr), 'dmap':dmap, 'args':args})
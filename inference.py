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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
torch.cuda.empty_cache()
model = UNet_ConvLSTM(n_channels=1, n_classes=2, use_LSTM=True, parallel_encoder=False, lstm_layers=1)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
model = model.to(device)

model_name = '100_brain_0304_fast_track'
state_dict = torch.load(f'velocimetry_models/{model_name}.pth.tar')['state_dict']
model.load_state_dict(state_dict, strict=False)

fname = 'Experiments/MouseBrain/AD_1123/dirfilt_IQData_20211123T144420_1.mat'
matfile = loadmat(fname)
ens_pos = np.abs(matfile['blinkbubble_pos'])
ens_neg = np.abs(matfile['blinkbubble_neg'])

ens_pos = ens_pos[20:,1:,:]
ens_neg = ens_neg[20:,1:,:]
ens_pos = ens_pos/np.max(ens_pos)
ens_neg = ens_neg/np.max(ens_neg)

# H, W needs to be integer multiples of 16
h, w = ens_pos.shape[:2]
H_i = ((h*5)//16)*16
W_i = ((w*10)//16)*16
H_start = 0
W_start = 0
H_sub = H_start+H_i#H_i#1392
W_sub = W_start+W_i#W_i#1792
nt = 16
step_t = 4
savefile = False
dir_filt = False
local_filt = False
batch_scale = False
noiseprof_name = 'noise_mayo_kidney.mat'
for file in range(1):#len(datafiles)):
    pos = ((file%2)==1)
    acc_vmap = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    acc_angle = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    counter = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    vel_curve = []
    roi1_curve = []
    roi2_curve = []
    ims = []
    PI = torch.tensor(np.pi).to(device).float()
    ZERO = torch.tensor(0).to(device).float()
    ONE = torch.tensor(1).to(device).float()
    thresh_out = torch.tensor(0.0).to(device).float()
    ftime = 0
    ens_tensor_p = torch.from_numpy(ens_pos).float().to(device).permute((2, 0, 1))
    ens_tensor_n = torch.from_numpy(ens_neg).float().to(device).permute((2, 0, 1))
    amplitude_arr_p = []
    amplitude_arr_n = []
    angle_arr = []
    if batch_scale == True:
        ens_tensor_p /= torch.max(ens_tensor_p)
        ens_tensor_n /= torch.max(ens_tensor_n)
    # ens_tensor = (ens_tensor-torch.min(ens_tensor))/(torch.max(ens_tensor)-torch.min(ens_tensor))
#     stime = time.time()
    
#     total_len = step_t+1
    total_len = ens_tensor_p.shape[0]
    start_f = 0
    
    amp_p = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    amp_n = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    with torch.no_grad():
        h = None
        for i in np.arange(start_f,total_len-step_t, step_t):
            for p in range(2):
                if p == 0:
                    seq = ens_tensor_p[i:i+nt,:, :]#20:120]#[i:i+nt, 100:250,:]#[i:i+nt, 90:140,:100]
                if p == 1:
                    seq = ens_tensor_n[i:i+nt,:, :]
    #             seq = seq.permute((0, 2, 1))
    #             if torch.max(seq)>0:
    #                 seq = seq/torch.max(seq)
        #         seq = (seq-np.min(seq))/np.ptp(seq)
                seq = torch.unsqueeze(seq, 0)
                seq_interp = F.interpolate(seq, (H_i, W_i), mode='bicubic', align_corners=True).unsqueeze(2)
                seq_interp = seq_interp[:,:,:,H_start:H_sub, W_start:W_sub]
                seq_interp[seq_interp<0] = 0

                stime_1 = time.time()
                y, h = model(seq_interp)



                etime_1 = time.time()
                amplitudes = y[0][0]

                amplitudes = torch.where(amplitudes<thresh_out, ZERO,amplitudes)


                vel_curve.append(torch.mean(amplitudes))

                ones = torch.where(torch.abs(amplitudes)>0, ONE, ZERO)
                counter += ones
                acc_vmap += amplitudes
                if p == 0:
#                     amp_p = amplitudes
                    amplitude_arr_p.append(np.asarray(amplitudes.detach().cpu()))
                if p == 1:
                    amplitude_arr_n.append(np.asarray(amplitudes.detach().cpu()))


                ftime += etime_1 - stime_1

    etime = time.time()
    vmap_mean = torch.zeros(acc_vmap.shape).to(device)
    vmap_mean[counter>0] = acc_vmap[counter>0]/counter[counter>0]
    vmap_mean[vmap_mean<0] = 0

    vmap_mean = np.asarray(vmap_mean.detach().cpu())
    counter = np.asarray(counter.detach().cpu())
    amplitude_arr_p = np.asarray(amplitude_arr_p)
    amplitude_arr_n = np.asarray(amplitude_arr_n)
    amplitude_arr = np.copy(amplitude_arr_p)
    amplitude_arr[amplitude_arr_p<amplitude_arr_n] = amplitude_arr_n[amplitude_arr_p<amplitude_arr_n]
    mean_pos = np.mean(amplitude_arr_p, axis=0)
    mean_neg = np.mean(amplitude_arr_n, axis=0)
    acc_vmap = np.copy(mean_pos)#np.mean(amplitude_arr, axis=0)
    acc_vmap[mean_pos<mean_neg] = mean_neg[mean_pos<mean_neg]
    dmap = np.ones(mean_pos.shape)
    dmap[mean_pos<mean_neg] = -1
    if savefile == True:
        savemat(f'{fname[12:-4]}_{model_name}.mat', 
                    {'acc_vmap':acc_vmap,'vmap_mean':vmap_mean,
                     'counter':counter,'vel_curve':vel_curve, 'amplitude_arr':np.asarray(amplitude_arr),
                    'dmap':dmap})
#     break
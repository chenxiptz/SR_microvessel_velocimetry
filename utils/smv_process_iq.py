import time
from matplotlib import gridspec
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
def smv_dirfilt(ens_pos, ens_neg, model, nt, step_t, device, single_frame = False, display = True):
    # H, W needs to be integer multiples of 16
    h, w, _ = ens_pos.shape
    H_i = int(np.round((h*5)/16)*16)
    W_i = int(np.round((w*10)/16)*16)
    H_start = 0
    W_start = 0
    H_sub = H_start+H_i
    W_sub = W_start+W_i
    acc_vmap = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    acc_angle = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    counter = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)

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
    stime = time.time()
    if single_frame == False:
        total_len = ens_tensor_p.shape[0]
    else:
        total_len = step_t + 1
    start_f = 0
    
    amp_p = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    amp_n = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    if display == True:
        fig = plt.figure()
    with torch.no_grad():
        h = None
        for i in np.arange(start_f,total_len-step_t, step_t):
            for p in range(2):
                if p == 0:
                    seq = ens_tensor_p[i:i+nt,:, :]
                if p == 1:
                    seq = ens_tensor_n[i:i+nt,:, :]
                seq = torch.unsqueeze(seq, 0)
                seq_interp = F.interpolate(seq, (H_i, W_i), mode='bicubic', align_corners=True).unsqueeze(2)
                seq_interp = seq_interp[:,:,:,H_start:H_sub, W_start:W_sub]
                seq_interp[seq_interp<0.0] = 0

                stime_1 = time.time()
                y, h = model(seq_interp)

                etime_1 = time.time()
                amplitudes = y[0][0]
                amplitudes = torch.where(amplitudes<thresh_out, ZERO,amplitudes)


                ones = torch.where(torch.abs(amplitudes)>0, ONE, ZERO)
                counter += ones
                acc_vmap += amplitudes
                if p == 0:
                    amplitude_arr_p.append(np.asarray(amplitudes.detach().cpu()))
                if p == 1:
                    amplitude_arr_n.append(np.asarray(amplitudes.detach().cpu()))
                    if display == True:
                        plt.imshow(np.abs(amplitude_arr_p[-1]+amplitude_arr_n[-1]),cmap='gray')
                        plt.title(f'Frames: {i} to {i + step_t - 1} ')
                        plt.draw()
                        plt.pause(0.001)
                ftime += etime_1 - stime_1
    plt.show()
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
    return amplitude_arr, vmap_mean, dmap

def smv_estdir(ens, model, nt, step_t, device, single_frame = False, display = True):
    # H, W needs to be integer multiples of 16
    h, w, _ = ens.shape
    H_i = int(np.round((h*5)/16)*16)
    W_i = int(np.round((w*10)/16)*16)
    H_start = 0
    W_start = 0
    H_sub = H_start+H_i
    W_sub = W_start+W_i
    acc_vmap = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    acc_angle = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    counter = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)

    PI = torch.tensor(np.pi).to(device).float()
    ZERO = torch.tensor(0).to(device).float()
    ONE = torch.tensor(1).to(device).float()
    thresh_out = torch.tensor(0.0).to(device).float()
    ftime = 0
    ens_tensor = torch.from_numpy(ens).float().to(device).permute((2, 0, 1))
    amplitude_arr = []
    angle_arr = []
    stime = time.time()
    if single_frame == False:
        total_len = ens_tensor.shape[0]
    else:
        total_len = step_t + 1
    start_f = 0
    
    amp = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    # if display == True:
    #     fig = plt.figure()
    with torch.no_grad():
        h = None
        for i in np.arange(start_f,total_len-step_t, step_t):
        
            seq = ens_tensor[i:i+nt,:, :]
            seq = torch.unsqueeze(seq, 0)
            seq_interp = F.interpolate(seq, (H_i, W_i), mode='bicubic', align_corners=True).unsqueeze(2)
            seq_interp = seq_interp[:,:,:,H_start:H_sub, W_start:W_sub]
            seq_interp[seq_interp<0.0] = 0

            stime_1 = time.time()
            y, h = model(seq_interp)

            etime_1 = time.time()
            amplitudes = y[0][0]
            amplitudes = torch.where(amplitudes<thresh_out, ZERO,amplitudes)
            angles = y[0][1]*(2*PI)-PI


            ones = torch.where(torch.abs(amplitudes)>0, ONE, ZERO)
            counter += ones
            acc_vmap += amplitudes
            acc_angle += angles
            amplitude_arr.append(np.asarray(amplitudes.detach().cpu()))
            
            angle_arr.append(np.asarray(angles.detach().cpu()))

            ftime += etime_1 - stime_1
    # plt.show()
    etime = time.time()
    vmap_mean = torch.zeros(acc_vmap.shape).to(device)
    vmap_mean[counter>0] = acc_vmap[counter>0]/counter[counter>0]
    vmap_mean[vmap_mean<0] = 0
    vmap_mean = np.asarray(vmap_mean.detach().cpu())
    counter = np.asarray(counter.detach().cpu())
    amplitude_arr = np.asarray(amplitude_arr)
    acc_vmap = np.mean(amplitude_arr, axis=0)
    angle_mean = np.mean(angle_arr, axis=0)
    dmap = torch.ones(acc_angle.shape).to(device)
    dmap[angle_mean<0] = -1
    return amplitude_arr, vmap_mean, angle_mean, np.asarray(dmap.detach().cpu())



def smv_singleframe(ens, model, nt, step_t, device, display = True):
    # H, W needs to be integer multiples of 16
    h, w, _ = ens.shape
    H_i = int(np.round((h*5)/16)*16)
    W_i = int(np.round((w*10)/16)*16)
    H_start = 0
    W_start = 0
    H_sub = H_start+H_i
    W_sub = W_start+W_i

    PI = torch.tensor(np.pi).to(device).float()
    ZERO = torch.tensor(0).to(device).float()
    ONE = torch.tensor(1).to(device).float()
    thresh_out = torch.tensor(0.0).to(device).float()
    ens_tensor = torch.from_numpy(ens).float().to(device).permute((2, 0, 1))
    start_f = 0
    
    amp = torch.zeros((H_sub-H_start, W_sub-W_start)).to(device)
    # if display == True:
    #     fig = plt.figure()
    with torch.no_grad():
        
        seq = ens_tensor[:nt,:, :]
        seq = torch.unsqueeze(seq, 0)
        seq_interp = F.interpolate(seq, (H_i, W_i), mode='bicubic', align_corners=True).unsqueeze(2)
        seq_interp = seq_interp[:,:,:,H_start:H_sub, W_start:W_sub]
        seq_interp[seq_interp<0.0] = 0

        stime_1 = time.time()
        y, h = model(seq_interp)

        etime_1 = time.time()
        amplitudes = y[0][0]
        amplitudes = torch.where(amplitudes<thresh_out, ZERO,amplitudes)
        angles = y[0][1]
        # angles[angles>0.5] = 1
        # angles[angles<-0.5] = -1
        dmap = torch.ones(angles.shape).to(device)
        dmap[angles<0] = -1
    # plt.show()
    pd = torch.squeeze(seq_interp)
    pd = torch.mean(pd, 0)
    return amplitudes, angles, dmap, pd
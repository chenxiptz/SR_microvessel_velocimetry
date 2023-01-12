import torch
from loss.pytorch_ssim import SSIM
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

parser.add_argument('-l', '--lstm_layers', help='Number of frames in each window', type = int,
                    default = 1)
parser.add_argument('-w', '--windowsize', help='Number of frames in each window', type = int,
                    default = 16)
parser.add_argument('-s', '--stepsize', help='Step size between two consecutive windows', type = int,
                    default = 16)
parser.add_argument('-rw', '--rollingwindow', help='Rolling window average size', type = int)
parser.add_argument('-bd', '--bidirectional', help='Whether to plot bidirectional map', type = int,
                    default = 0)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

args = parser.parse_args()

rw = False
rw_size = 0

if args.rollingwindow is not None:
    rw = True
    rw_size = args.rollingwindow

# Initialize model
model = UNet_ConvLSTM(n_channels=1, n_classes=2, use_LSTM=True, parallel_encoder=False, lstm_layers=args.lstm_layers)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
model = model.to(device)

# Load a pre-trained model
state_dict = torch.load(f'{args.modelfolder}{args.model}.pth.tar')['state_dict']
model.load_state_dict(state_dict, strict=False)

# Initialize tcpip connection
# Create a TCP/IP socket

print('Initializing TCP connection...')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect the socket to the port where the server is listening
server_address = ('localhost', 8192) # VSX IP: 128.174.226.58'
sock.connect(server_address)

# read data size from matlab
Na = int.from_bytes(sock.recv(4), 'big', signed=True)
Nl = int.from_bytes(sock.recv(4), 'big', signed=True)
Nt = args.windowsize
print(f'Received IQ data size is: ({Na}, {Nl})')
Nai = int(np.round((Na*5)/16)*16)
Nli = int(np.round((Nl*10)/16)*16)
sock.send(bytes(str(Nt), 'utf8'))

if rw == True:
    rw_arr = torch.zeros((Nai, Nli, rw_size)).to(device)

# smv processing
stream_status = 1
byte_size = Na*Nl*Nt
ens = np.zeros((Na, Nl, Nt))
dmap_avg = torch.zeros((Nai, Nli)).to(device)
angle_avg = torch.zeros((Nai, Nli)).to(device)
fig = plt.figure()
counter = torch.tensor(0).to(device)
counter_op = torch.tensor(0).to(device)

cmap = cmr.iceburn

plotdir = args.bidirectional

ax1 = plt.subplot(1,1,1)

#create two image plots
if plotdir == True:
    im1 = ax1.imshow(np.zeros((Nai, Nli)), cmap=cmap, vmax=5, vmin = -5)
else:
    im1 = ax1.imshow(np.zeros((Nai, Nli)),cmap='gray', vmax=5)


plt.ion()
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
print('Begin SMV processing...')
while True:
    stime0 = time.time()
    stream_status = int.from_bytes(sock.recv(4), 'big', signed=True)
    if stream_status == -1:
        break
    char_len = int.from_bytes(sock.recv(4), 'big', signed=True)
    fname = sock.recv(char_len)
    print(fname.decode())
    ens = loadmat(fname.decode())['blinkbubble']
    s_time = time.time()
    vmap, angle_mean, dmap = smv_process_iq.smv_singleframe(ens, model, args.windowsize, args.stepsize, device)
    angle_avg = (angle_avg * counter_op + angle_mean)/(counter_op + 1)
    dmap_avg = torch.sign(angle_mean)
    if rw == True:
        rw_arr[:, :, counter_op%rw_size] = vmap
    e_time = time.time()
    # sock.send(b'fin')
    print(f'Finished processing batch in {e_time - s_time} seconds')
    # plt.clf()
    if plotdir == 1:
        if rw == True and counter_op >= rw_size:
            disp_img = torch.abs(torch.mean(rw_arr, dim = 2)) * dmap_avg * 4.928
            # im1.set_data(disp_img)
            # plt.imshow(np.asarray(disp_img.detach().cpu()),cmap=cmap, vmax=5, vmin = -5)
        else:
            disp_img = torch.abs(torch.sum(rw_arr, dim = 2))/(counter_op+1)*4.928*dmap_avg   
            
            # im1.set_data(disp_img)
    else:
        if rw == True and counter_op >= rw_size:
            disp_img = torch.abs(torch.mean(rw_arr, dim = 2)) * 4.928
            # plt.imshow(np.asarray(disp_img.detach().cpu()),cmap='gray', vmax=5)
        else:
            disp_img = torch.abs(torch.sum(rw_arr, dim = 2))/(counter_op+1)*4.928
            # plt.imshow(np.asarray(disp_img.detach().cpu()),cmap='gray', vmax=5)
    
    im1.set_data(np.asarray(disp_img.detach().cpu()))
    plt.title(f'Frames: {counter} to {counter + Nt - 1} ')
    # plt.draw()
    plt.pause(0.0001)
    
    e_time2 = time.time()
    if os.path.exists(fname):
        os.remove(fname)
    counter = counter + Nt
    counter_op += 1
    print(e_time2 - e_time, e_time2 - stime0)
# savemat(f'{args.datafolder}_{args.data}_{args.model}_smv.mat', 
#                     {'vmap_mean':vmap,
#                      'amplitude_arr':np.asarray(amplitude_arr),
#                     'dmap':dmap})
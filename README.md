# Localization free super-resolution microbubble velocimetry
## About
Python implementation of super-resolution microbubble velocimetry using a long short-term memory neural network \
U-Net model adapted from https://github.com/milesial/Pytorch-UNet \
Pytorch implementation of convolutional LSTM from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

## Pre-trained models and ultrasound data
Two pre-trained models (CAM and mouse brain) and example ultrasound data available at https://doi.org/10.7910/DVN/SECUFD

## Use
### Loading ultrasound data
We have provided some data loading functions in [load_iqdata.py](utils\load_iqdata.py) to load the ultrasound data available at https://doi.org/10.7910/DVN/SECUFD.

If pre-directional filtered data was loaded, SMV processing will estimate the flow direction using the IQ data. Otherwise it will provide an up-down flow direction map based on the magnitude of the filtered data.

### SMV processing
[inference.py](inference.py) example inference code to perform SMV processing on a set of IQ data.

Usage: `inference.py [-h] -m MODEL -mf MODELFOLDER -fn FILENAME -ff FILEFOLDER
                    -sp SAVEPATH [-w WINDOWSIZE] [-s STEPSIZE]
                    [-rw ROLLINGWINDOW] [-bd BIDIRECTIONAL]
                    [-df DIRECTIONALFILTER]`

Use the `-h` or `--help` option to see description of each argument.

Example command (16 frames block, no overlap, with directional filtering):
`python3 inference.py -m 'SMV_pretrained_weights_mousebrain.pth.tar' -mf '/home/ultrasound/Desktop/' -fn 'dirfilt_IQData_20220502T105116_3.mat' -ff '/home/ultrasound/Desktop/Deep-SMV/Experiments/MouseBrain/mouse_ecg_502/' -sp 'example_out.mat' -w 16 -bd 1 -df 1`

If you want to use your own data and data loading function, note that the provided SMV processing code expects the input ultrasound image stack to be of shape (height, width, n_frames).

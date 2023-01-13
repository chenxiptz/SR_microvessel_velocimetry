# Localization free super-resolution microbubble velocimetry
## About
Python implementation of super-resolution microbubble velocimetry using a long short-term memory neural network \
U-Net model adapted from https://github.com/milesial/Pytorch-UNet \
Pytorch implementation of convolutional LSTM from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

## Pre-trained models and ultrasound data
Two pre-trained models (CAM and mouse brain) and example ultrasound data available at https://doi.org/10.7910/DVN/SECUFD

## Use
### Loading ultrasound data
TODO

### SMV processing
[inference.py](inference.py) example inference code to perform SMV processing on a set of IQ data after directional filtering.\
Example use:\
`python3 inference.py -m 'SMV_pretrained_weights_mousebrain.pth.tar' -mf '/home/ultrasound/Desktop/' -fn 'dirfilt_IQData_20220502T105116_3.mat' -ff '/home/ultrasound/Desktop/Deep-SMV/Experiments/MouseBrain/mouse_ecg_502/' -sp 'example_out.mat' -w 16 -bd 1`

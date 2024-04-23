#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of validation code of this paper:
# Remote Sensing Image Fusion Method Based on Retinex Model and Hybrid Attention Mechanism
# author: Yongxu Ye
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import h5py
from model import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import importlib
import scipy.io

importlib.reload(sys)

NET = 'MYNET'
condition = 'new_1'

checkpoint = f'../../disk3/xmm/{NET}/checkpoint/wv3_{condition}/{NET}_'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
devicesList = [0]
dtype = torch.cuda.FloatTensor
ms_channels = 8
pan_channel = 1

if __name__ == "__main__":
    ##### read dataset #####
    test_bs = 1
    tmpPath = checkpoint + f"WV3_9714_{condition}_best.pth"
    ReducedData = 'test_wv3_multiExm1'

    SaveReducedDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_{ReducedData}.mat"
    test_Reduced_data_name = f'../../disk3/xmm/dataset/testing/{ReducedData}.h5'
    test_Reduced_data = h5py.File(test_Reduced_data_name, 'r')
    test_Reduced_dataset = my_dataset(test_Reduced_data)
    del test_Reduced_data
    test_Reduced_dataloader = torch.utils.data.DataLoader(test_Reduced_dataset, batch_size=test_bs, shuffle=False)

    FullData = 'test_wv3_OrigScale_multiExm1'
    SaveFullDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_{FullData}.mat"
    test_Full_data_name = f'../../disk3/xmm/dataset/testing/{FullData}.h5'
    test_Full_data = h5py.File(test_Full_data_name, 'r')
    test_Full_dataset = my_full_dataset(test_Full_data)
    del test_Full_data
    test_Full_dataloader = torch.utils.data.DataLoader(test_Full_dataset, batch_size=test_bs, shuffle=False)

    #fc = 32 ## AFEFPNN
    #CNN =  LapPanNet(nc,fc) ## AFEFPNN
    CNN = MyNet(ms_channels)  ## others
    CNN = nn.DataParallel(CNN, device_ids=devicesList).cuda()

    CNN.load_state_dict(torch.load(tmpPath))
    CNN.eval()
    reduced_count = 0
    for index, data in enumerate(test_Reduced_dataloader):
        gtVar = Variable(data[0]).type(dtype)
        panVar = Variable(data[1]).type(dtype)
        lmsVar = Variable(data[2]).type(dtype)
        msVar = Variable(data[3]).type(dtype)
        with torch.no_grad():
            output, shade, ref = CNN(msVar, panVar)  ##PNN/APNN
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        shade_np = shade.cpu().data.numpy()
        ref_np = ref.cpu().data.numpy()
        lms_np = data[2].numpy()
        ms_np = data[3].numpy()
        pan_np = data[1].numpy()
        gt_np = data[0].numpy()
        if reduced_count == 0:
            Output_np = netOutput_np
            shade1 = shade_np
            ref1 = ref_np
            ms = ms_np
            lms = lms_np
            pan = pan_np
            gt = gt_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            shade1 = np.concatenate((shade_np, shade1), axis=0)
            ref1 = np.concatenate((ref_np, ref1), axis=0)
            ms = np.concatenate((ms_np, ms), axis=0)
            lms = np.concatenate((lms_np, lms), axis=0)
            pan = np.concatenate((pan_np, pan), axis=0)
            gt = np.concatenate((gt_np, gt), axis=0)
        reduced_count = reduced_count + 1
    print(Output_np.shape)
    scipy.io.savemat(SaveReducedDataPath, {'Res256': Output_np, 'PAN': pan, 'MS': ms, 'LMS': lms, 'GT': gt,
                                           'shade': shade1, 'ref': ref1})
    #scipy.io.savemat(SaveDataPath,{'QB256':Output_np, 'GT256': gt})

    full_count = 0
    for index, data in enumerate(test_Full_dataloader):
        panVar = Variable(data[0]).type(dtype)
        lmsVar = Variable(data[1]).type(dtype)
        msVar = Variable(data[2]).type(dtype)
        with torch.no_grad():
            output, shade, ref = CNN(msVar, panVar)  ##  PNN/APNN
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        shade_np = shade.cpu().data.numpy()
        ref_np = ref.cpu().data.numpy()
        lms_np = data[1].numpy()
        ms_np = data[2].numpy()
        pan_np = data[0].numpy()
        if full_count == 0:
            Output_np = netOutput_np
            shade1 = shade_np
            ref1 = ref_np
            ms = ms_np
            lms = lms_np
            pan = pan_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            shade1 = np.concatenate((shade_np, shade1), axis=0)
            ref1 = np.concatenate((ref_np, ref1), axis=0)
            ms = np.concatenate((ms_np, ms), axis=0)
            lms = np.concatenate((lms_np, lms), axis=0)
            pan = np.concatenate((pan_np, pan), axis=0)
        full_count = full_count + 1
    scipy.io.savemat(SaveFullDataPath, {'Res256': Output_np, 'PAN': pan, 'MS': ms, 'LMS': lms,
                                        'shade': shade1, 'ref': ref1})
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torchvision import transforms
import spectral as spy
import torch.nn as nn

#
# class my_dataset(data.Dataset):
#     def __init__(self, mat_data):
#         gt_set = mat_data['gt'][...]
#         pan_set = mat_data['pan'][...]
#         ms_set = mat_data['ms'][...]
#         lms_set = mat_data['lms'][...]
#
#         self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
#         self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
#         self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
#         self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.
#
#     def __getitem__(self, index):
#         gt = self.gt_set[index, :, :, :]
#         pan = self.pan_set[index, :, :, :]
#         ms = self.ms_set[index, :, :, :]
#         lms = self.lms_set[index, :, :, :]
#         return gt, pan, lms, ms
#
#     def __len__(self):
#         return self.gt_set.shape[0]
#

class my_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047. * 255
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047. * 255
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047. * 255
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047. * 255

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]


class my_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]


if __name__ == "__main__":
    validation_data_name = '../../disk3/xmm/dataset/training/train_wv3.h5'  # your data path
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = my_dataset(validation_data)
    del validation_data
    data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
    plt.figure(1)
    gauss_kernel_size = 5
    sigma = 10
    transform1 = transforms.GaussianBlur(gauss_kernel_size, sigma)
    for index, item in enumerate(data_loader):
        b, c, h, w = item[0].shape
        pan = item[1]
        pan_repeat = np.repeat(pan, c, 1)
        ms = item[0]
        lms = item[2]
        seed = random.randint(0, w - 20)
        pan_pixel = []
        ms_pixel = []
        pan_diff_x_p = []
        ms_diff_x_p = []
        pan_patch_pixel = []
        ms_patch_pixel = []

        pan_r = F.pad(pan, (0, 1, 0, 0))[:, :, :, 1:]
        pan_l = F.pad(pan, (1, 0, 0, 0))[:, :, :, :w]
        pan_t = F.pad(pan, (0, 0, 1, 0))[:, :, :h, :]
        pan_b = F.pad(pan, (0, 0, 0, 1))[:, :, 1:, :]
        pan_diff_x = pan_r - pan_l
        pan_diff_y = pan_t - pan_b

        pan_repeat_r = F.pad(pan_repeat, (0, 1, 0, 0))[:, :, :, 1:]
        pan_repeat_l = F.pad(pan_repeat, (1, 0, 0, 0))[:, :, :, :w]
        pan_repeat_t = F.pad(pan_repeat, (0, 0, 1, 0))[:, :, :h, :]
        pan_repeat_b = F.pad(pan_repeat, (0, 0, 0, 1))[:, :, 1:, :]
        pan_repeat_diff_x = pan_repeat_r - pan_repeat_l
        pan_repeat_diff_y = pan_repeat_t - pan_repeat_b

        ms_r = F.pad(ms, (0, 1, 0, 0))[:, :, :, 1:]
        ms_l = F.pad(ms, (1, 0, 0, 0))[:, :, :, :w]
        ms_t = F.pad(ms, (0, 0, 1, 0))[:, :, :h, :]
        ms_b = F.pad(ms, (0, 0, 0, 1))[:, :, 1:, :]
        ms_diff_x = ms_r - ms_l
        ms_diff_y = ms_t - ms_b

        # ms_f = F.pad(ms, (0, 0, 0, 0, 1, 0))[:, :c, :, :]
        ms_ba = F.pad(ms, (0, 0, 0, 0, 0, 1))[:, 1:, :, :]
        ms_diff_spc = ms - ms_ba
        # ms_diff_spc = transform1(ms_diff_spc)

        # lms_f = F.pad(lms, (0, 0, 0, 0, 1, 0))[:, :c, :, :]
        lms_ba = F.pad(lms, (0, 0, 0, 0, 0, 1))[:, 1:, :, :]
        lms_diff_spc = lms - lms_ba

        pan_diff_x_pixel = []
        pan_diff_y_pixel = []
        ms_diff_x_pixel = []
        ms_diff_y_pixel = []
        for i in range(h):
            for j in range(w):
                pan_pixel.append(item[1][0][0][i][j])
                pan_diff_x_p.append(pan_diff_x[0][0][i][j])
                if i >= seed and i < seed + 20 and j >= seed and j < seed + 20:
                    pan_patch_pixel.append(item[1][0][0][i][j])
                    pan_diff_x_pixel.append(pan_diff_x[0][0][i][j])
                    pan_diff_y_pixel.append(pan_diff_y[0][0][i][j])
                
        for j in range(h):
            for k in range(w):
                ms_band = []
                ms_diff_x_b = []
                ms_patch_band = []
                ms_diff_x_band = []
                ms_diff_y_band = []
                for i in range(c):
                    ms_band.append(item[0][0][i][j][k])
                    ms_diff_x_b.append(ms_diff_x[0][i][j][k])
                ms_pixel.append(ms_band)
                ms_diff_x_p.append(ms_diff_x_b)
                if j >= seed and j < seed + 20 and k >= seed and k < seed + 20:
                    for i in range(c):
                        ms_patch_band.append(item[0][0][i][j][k])
                        ms_diff_x_band.append(ms_diff_x[0][i][j][k])
                        ms_diff_y_band.append(ms_diff_y[0][i][j][k])
                    ms_patch_pixel.append(ms_patch_band)
                    ms_diff_x_pixel.append(ms_diff_x_band)
                    ms_diff_y_pixel.append(ms_diff_y_band)
        ms_pixel = np.array(ms_pixel)
        ms_diff_x_p = np.array(ms_diff_x_p)
        theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(ms_pixel.T, ms_pixel)), ms_pixel.T), pan_pixel)
        theta_dif_x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ms_diff_x_p.T, ms_diff_x_p)), ms_diff_x_p.T), pan_diff_x_p)
        ms_sin_pixel = np.matmul(ms_patch_pixel, theta)
        ms_diff_x = np.matmul(ms_diff_x_pixel, theta_dif_x)
        ms_diff_y = np.matmul(ms_diff_y_pixel, theta)

        # ms_dif_spc_pixel = []
        # lms_dif_spc_pixel = []
        for i in range(c):
            ms_b = []
            lms_b = []

            for j in range(h):
                for k in range(w):
                    if j >= seed and j < seed + 20 and k >= seed and k < seed + 20:
                        ms_b.append(ms_diff_spc[0][i][j][k])
                        lms_b.append(lms_diff_spc[0][i][j][k])

            # ms_dif_spc_pixel.append(ms_b)
            # lms_dif_spc_pixel.append(lms_b)
            # plt.title(f'{index + 1}th band-{i + 1} spectral gradient, kernel_size={gauss_kernel_size} $\sigma$={sigma}')

            plt.title(f'{index + 1}th band-{i + 1} spectral gradient')
            plt.plot(np.arange(0, 400, 5), ms_b[::5], 'b-', label='hrms_diff_spe')
            plt.plot(np.arange(0, 400, 5), lms_b[::5], 'r-', label='lrms_diff_spe')
            plt.legend()
            plt.savefig(f'./plt/{index + 1}_dif_spe_{i + 1}_noblur.png')
            plt.show()

        plt.title(f'{index + 1}th intensity')
        plt.plot(np.arange(0, 400, 5), pan_patch_pixel[::5], 'b-', label='pan')
        plt.plot(np.arange(0, 400, 5), ms_sin_pixel[::5], 'r-', label='hrms')
        plt.legend()
        plt.savefig(f'./plt/{index + 1}_i.png')
        plt.show()

        plt.title(f'{index + 1}th spatial gradient along x')
        plt.plot(np.arange(0, 400, 5), pan_diff_x_pixel[::5], 'b-', label='pan_diff_x')
        plt.plot(np.arange(0, 400, 5), ms_diff_x[::5], 'r-', label='hrms_diff_x')
        plt.legend()
        plt.savefig(f'./plt/{index + 1}_dif.png')
        plt.show()

        # item[0]=item[0].permute(0,2,3,1)
        # item[1] = item[1].permute(0, 2, 3, 1)
        # item[2] = item[2].permute(0, 2, 3, 1)
        # item[3] = item[3].permute(0, 2, 3, 1)
        # print(item[0].size())
        # print(item[1].size())
        # print(item[2].size())
        # print(item[3].size())
        # view1=spy.imshow(data=item[0][0,:,:,:].numpy(),bands=(0,1,2),title='gt')
        # view2=spy.imshow(data=item[1][0,:,:,:].numpy(),title='pan')
        # view3=spy.imshow(data=item[2][0,:,:,:].numpy(),bands=(0,1,2),title='lms')
        # view4=spy.imshow(data=item[3][0,:,:,:].numpy(),bands=(0,1,2),title='ms')

        # plt.subplot(2, 2, 1)
        # plt.imshow(item[0][0,:,:,:])
        # plt.title('ground truth')
        # plt.subplot(2, 2, 2)
        # plt.imshow(item[1][0,:,:,:])
        # plt.title('pan image')
        # plt.subplot(2, 2, 3)
        # plt.imshow(item[2][0,:,:,:])
        # plt.title('lms image')
        # plt.subplot(2, 2, 4)
        # plt.imshow(item[3][0,:,:,:])
        # plt.title('ms image')
        # plt.show()
        if index == 2:break

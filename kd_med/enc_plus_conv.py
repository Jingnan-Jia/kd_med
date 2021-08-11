# -*- coding: utf-8 -*-
# @Time    : 8/11/21 11:43 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import math

import torch
from torch import nn as nn


class EncPlusConvBase(nn.Module):
    """
    the out_chn of enc_s must be equal to the in_chn of conv.
    """
    def __init__(self, enc_s, conv):
        super().__init__()
        self.enc_s = enc_s
        self.conv = conv

    def forward(self, x):
        out = self.enc_s(x)
        out = self.conv(out)
        return out


class EncPlusConv:
    def __init__(self, input_tmp, enc_t: nn.Module, enc_s: nn.Module, dims: int = 3, no_cuda: bool = True):
        self.enc_s = enc_s

        # while not self.CORRECT_CONV:
        if dims == 3:
            # input_tmp = torch.ones((2, 1, 128, 128, 128))
            if not no_cuda:
                device = torch.device("cuda")
                enc_t.to(device)
                self.enc_s.to(device)
                input_tmp = input_tmp.to(device)
            out_t = enc_t(input_tmp)
            out_s = self.enc_s(input_tmp)
            batch_size, chn_t, dim1_t, dim2_t, dim3_t = out_t.shape
            batch_size, chn_s, dim1_s, dim2_s, dim3_s = out_s.shape
        else:
            # input_tmp = torch.ones((2, 1, 128, 128))
            if not no_cuda:
                device = torch.device("cuda")
                enc_t.to(device)
                self.enc_s.to(device)
                input_tmp = input_tmp.to(device)
            out_t = enc_t(input_tmp)
            out_s = self.enc_s(input_tmp)
            batch_size, chn_t, dim1_t, dim2_t = out_t.shape
            batch_size, chn_s, dim1_s, dim2_s = out_s.shape
        self.chn_t = chn_t
        self.chn_s = chn_s
        self.dim1_t = dim1_t
        self.dim2_t = dim2_t
        self.dim3_t = dim3_t if dims == 3 else None

        self.dim1_s = dim1_s
        self.dim2_s = dim2_s
        self.dim3_s = dim3_s if dims == 3 else None

        self.dims = dims

    def set_conv_config(self, enc_s):
        # print(f'in pad: {self.PAD}')

        # o = (n - f + 2 * p) / s + 1
        # f = n - ((o - 1) * s - 2 * p)
        # p = ((o - 1) * s - n + f)/2
        if self.dim1_t > self.dim1_s:
            # print(f'teacher model depth is less than student model, dim1_t: {self.dim1_t}, dim1_s: {self.dim1_s}')
            if self.dim1_t >= (1.5 * self.dim1_s):  # need upsampling or transposedconv
                s = math.ceil(self.dim1_t / self.dim1_s)
                # down sample using stride at first, pad more if over down-sampling
                if self.dims == 3:
                    conv = nn.Sequential(nn.ConvTranspose3d(self.chn_s, self.chn_t, 3, stride=s),
                                         nn.AdaptiveAvgPool3d((self.dim1_t, self.dim2_t, self.dim3_t)),
                                         nn.Conv3d(self.chn_t, self.chn_t, kernel_size=3, padding=1))
                else:
                    conv = nn.Sequential(nn.ConvTranspose2d(self.chn_s, self.chn_t, 3, stride=s),
                                         nn.AdaptiveAvgPool2d((self.dim1_t, self.dim2_t)),
                                         nn.Conv2d(self.chn_t, self.chn_t, kernel_size=3, padding=1))
            else:
                if self.dims == 3:
                    conv = nn.Sequential(nn.AdaptiveAvgPool3d((self.dim1_t, self.dim2_t, self.dim3_t)),
                                         nn.Conv3d(self.chn_s, self.chn_t, kernel_size=3, padding=1))
                else:
                    conv = nn.Sequential(nn.AdaptiveAvgPool2d((self.dim1_t, self.dim2_t)),
                                         nn.Conv2d(self.chn_s, self.chn_t, kernel_size=3, padding=1))

        else:  # teacher model is deeper
            s = math.ceil(self.dim1_s / self.dim1_t)  # down sample using stride at first, pad more if over down-sampling
            conv_sz = s + 1  # conv size should be bigger than stride
            if self.dims == 3:
                conv = nn.Sequential(nn.Conv3d(self.chn_s, self.chn_t, kernel_size=conv_sz, stride=s),
                                     nn.AdaptiveAvgPool3d((self.dim1_t, self.dim2_t, self.dim3_t)),
                                     nn.Conv3d(self.chn_t, self.chn_t, kernel_size=3, padding=1))
            else:
                conv = nn.Sequential(nn.Conv2d(self.chn_s, self.chn_t, kernel_size=conv_sz, stride=s),
                                     nn.AdaptiveAvgPool2d((self.dim1_t, self.dim2_t)),
                                     nn.Conv2d(self.chn_t, self.chn_t, kernel_size=3, padding=1))

        enc_plus_conv = EncPlusConvBase(enc_s, conv)
        return enc_plus_conv
        # else:
        #     return enc_s

    def get(self):
        """
        Chn_in = 1 always here.
        :param enc_t:
        :param enc_s:
        :param dims:
        :return:
        """

        enc_plus_conv = self.set_conv_config(self.enc_s)

        return enc_plus_conv
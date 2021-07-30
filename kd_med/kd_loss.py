# -*- coding: utf-8 -*-
# @Time    : 7/26/21 6:59 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import math

import torch
import torch.nn as nn
from kd_med.pre_trained_enc import pre_trained_enc


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


# class GetEncSConv:
#     """
#     Using class method to make sure the network only be created once and reusable many times.
#     """
#     enc_t = None
#     enc_s = None
#     enc_plus_conv = None
#     chn_t = None
#     chn_s = None
#     dims = 3
#     no_cuda = None
#
#
#     @classmethod


class EncPlusConv:
    def __init__(self):
        self.CORRECT_CONV = False
        self.chn_t = None
        self.chn_s = None
        self.PAD = 1
        self.dim1_t = None
        self.dim2_t = None
        self.dim3_t = None

        self.dim1_s = None
        self.dim2_s = None
        self.dim3_s = None

    def set_conv_config(self, enc_s):
        # print(f'in pad: {self.PAD}')

        # o = (n - f + 2 * p) / s + 1
        # f = n - ((o - 1) * s - 2 * p)
        # p = ((o - 1) * s - n + f)/2
        if self.dim1_t > self.dim1_s:
            print(f'teacher model depth is less than student model, dim1_t: {self.dim1_t}, dim1_s: {self.dim1_s}')
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
    def get_enc_plus_conv(self, input_tmp, enc_t: nn.Module, enc_s: nn.Module, dims: int = 3, no_cuda: bool = True):
        """
        Chn_in = 1 always here.
        :param enc_t:
        :param enc_s:
        :param dims:
        :return:
        """

        self.enc_plus_conv = enc_s
        # while not self.CORRECT_CONV:
        if dims == 3:
            # input_tmp = torch.ones((2, 1, 128, 128, 128))
            if not no_cuda:
                device = torch.device("cuda")
                enc_t.to(device)
                self.enc_plus_conv.to(device)
                input_tmp = input_tmp.to(device)
            out_t = enc_t(input_tmp)
            out_s = self.enc_plus_conv(input_tmp)
            batch_size, chn_t, dim1_t, dim2_t, dim3_t = out_t.shape
            batch_size, chn_s, dim1_s, dim2_s, dim3_s = out_s.shape
        else:
            # input_tmp = torch.ones((2, 1, 128, 128))
            if not no_cuda:
                device = torch.device("cuda")
                enc_t.to(device)
                self.enc_plus_conv.to(device)
                input_tmp = input_tmp.to(device)
            out_t = enc_t(input_tmp)
            out_s = self.enc_plus_conv(input_tmp)
            batch_size, chn_t, dim1_t, dim2_t = out_t.shape
            batch_size, chn_s, dim1_s, dim2_s = out_s.shape
        if self.chn_t is None:
            self.chn_t = chn_t
            self.chn_s = chn_s
            self.dim1_t = dim1_t
            self.dim2_t = dim2_t
            self.dim3_t = dim3_t if dims==3 else None

            self.dim1_s = dim1_s
            self.dim2_s = dim2_s
            self.dim3_s = dim3_s if dims==3 else None

            self.dims = dims

        # if (dim1_t==dim1_s) and (dim2_t==dim2_s) and (dim3_t==dim3_s) :
        #     self.CORRECT_CONV = True
        # else:
        print(f'config conv because dim1_t: {dim1_t}, dim1_s: {dim1_s}')
            # print(f'out pad: {self.PAD}')

        self.enc_plus_conv = self.set_conv_config(enc_s)


        return self.enc_plus_conv


def kd_loss(batch_x: torch.Tensor,
            enc_t: nn.Module,
            enc_s_conv: nn.Module,
            cuda: bool = False):
    """
    The enc_s will share the same memory with enc_s_conv, and the enc_s_conv will be optimized by the loss of kd.
    todo: I have to put enc_s_conv to outsize otherwise the last conv will not be updated at all !!!

    """

    # enc_t = PreTrainedEnc.get(net_t_name)
    # # enc_s share the same memory with the enc_s in enc_s_conv, only create it once and reuse it
    # enc_s_conv = GetEncSConv().get(enc_t, enc_s, dims)
    enc_t.eval()
    if cuda:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # batch_x.to(torch.device("cuda"))
                out_t = enc_t(batch_x)
    else:
        with torch.no_grad():
            out_t = enc_t(batch_x)

    out_s = enc_s_conv(batch_x)
    loss = nn.MSELoss()
    kd_loss = loss(out_t, out_s)
    return kd_loss


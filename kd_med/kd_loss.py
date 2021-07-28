# -*- coding: utf-8 -*-
# @Time    : 7/26/21 6:59 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import math

import torch
import torch.nn as nn
from kd_med.pre_trained_enc import PreTrainedEnc


class EncPlusConv(nn.Module):
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


class GetEncSConv:
    """
    Using class method to make sure the network only be created once and reusable many times.
    """
    enc_t = None
    enc_s = None
    enc_plus_conv = None
    chn_t = None
    chn_s = None
    dims = 3


    @classmethod
    def set_conv_config(cls, dim1_t, dim1_s):
        # o = (n - f + 2 * p) / s + 1
        # f = n - ((o - 1) * s - 2 * p)
        # p = ((o - 1) * s - n + f)/2
        if dim1_t == dim1_s:  # no need of conv at all
            cls.enc_plus_conv = cls.enc_s
            return cls.enc_plus_conv
        elif dim1_t > dim1_s:
            raise Exception('teacher model depth is less than student model')
        else:  # teacher model is deeper
            s = math.ceil(dim1_s / dim1_t) # down sample using stride at first, pad more if over down-sampling
            conv_sz = s + 1  # conv size should be bigger than stride
            exprected_in_size = s * (dim1_t - 1) + conv_sz
            p = math.ceil((exprected_in_size - dim1_s) / 2)
            if cls.dims == 3:
                conv = nn.Conv3d(cls.chn_s, cls.chn_t, kernel_size=(conv_sz, conv_sz, conv_sz),
                                 stride=(s, s, s), padding=p)
            else:
                conv = nn.Conv2d(cls.chn_s, cls.chn_t, kernel_size=(conv_sz, conv_sz),
                                 stride=(s, s), padding=p)
            cls.enc_plus_conv = EncPlusConv(cls.enc_s, conv)

    @classmethod
    def get(cls, enc_t: nn.Module, enc_s: nn.Module, dims: int = 3):
        """
        Chn_in = 1 always here.
        :param enc_t:
        :param enc_s:
        :param dims:
        :return:
        """
        # print(f"cls.enc_t: {cls.enc_t}, cls.enc_s: {cls.enc_s}")
        if cls.enc_t is not enc_t:  # when coming encoder is not the same as the stored one

            cls.dims = dims
            cls.enc_t = enc_t
            cls.enc_s = enc_s
            if dims == 3:
                input_tmp = torch.ones((2, 1, 128, 128, 128))
                out_t = cls.enc_t(input_tmp)
                out_s = cls.enc_s(input_tmp)
                batch_size, cls.chn_t, dim1_t, dim2_t, dim3_t = out_t.shape
                batch_size, cls.chn_s, dim1_s, dim2_s, dim3_s = out_s.shape
            else:
                input_tmp = torch.ones((2, 1, 128, 128))
                out_t = cls.enc_t(input_tmp)
                out_s = cls.enc_s(input_tmp)
                batch_size, cls.chn_t, dim1_t, dim2_t = out_t.shape
                batch_size, cls.chn_s, dim1_s, dim2_s = out_s.shape

            cls.set_conv_config(dim1_t, dim1_s)

        return cls.enc_plus_conv


def kd_loss(batch_x: torch.Tensor,
            enc_s: nn.Module,
            net_t_name: str ='resnet3d_34'):

    if len(batch_x.shape) == 5:
        dims: int = 3
    elif len(batch_x.shape) == 4:
        dims = 2
    else:
        raise Exception(f'shape of batch_x: {batch_x.shape} is not correct')

    enc_t = PreTrainedEnc.get(net_t_name)
    # enc_s share the same memory with the enc_s in enc_s_conv, only create it once and reuse it
    enc_s_conv = GetEncSConv().get(enc_t, enc_s, dims)
    with torch.no_grad():
        out_t = enc_t(batch_x)
    out_s = enc_s_conv(batch_x)
    loss = nn.MSELoss()
    kd_loss = loss(out_t, out_s)
    return kd_loss


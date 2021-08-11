# -*- coding: utf-8 -*-
# @Time    : 7/26/21 6:59 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import torch
import torch.nn as nn


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

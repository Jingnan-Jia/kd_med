# -*- coding: utf-8 -*-
# @Time    : 6/27/21 1:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
import tempfile
import os
import torch
import torch.nn as nn

from parameterized import parameterized
from kd_med.enc_plus_conv import EncPlusConvBase

import numpy as np

TEST_CASE_3d = [nn.Conv3d(1, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
               nn.Conv3d(48, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
               torch.ones((2, 1, 64, 64, 32))]

TEST_CASE_2d = [nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
               nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
               torch.ones((2, 3, 128, 96))]


class TestEncPlusConv(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3d, TEST_CASE_2d])
    def test_EncPlusConv(self, enc, conv, input_data):

        enc_plus_conv = EncPlusConvBase(enc, conv)
        out_enc = enc(input_data)
        out_conv = conv(out_enc)
        out_enc_plus_conv = enc_plus_conv(input_data)
        out_1 = out_conv.clone().detach().cpu().numpy()
        out_2 = out_enc_plus_conv.clone().detach().cpu().numpy()

        self.assertIsNone(np.testing.assert_allclose(out_1, out_2, rtol=1e-5, atol=0))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
# @Time    : 7/27/21 7:57 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import unittest
import tempfile
import os
import torch
import torch.nn as nn

from parameterized import parameterized
from kd_med.kd_loss import GetEncSConv

import numpy as np

TEST_CASE_3d = [nn.Sequential(nn.Conv3d(1, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
                              nn.Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3))),
               nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
               3]

TEST_CASE_2d = [nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
                              nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3))),
               nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
               2]


class TestGetEncSConv(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3d, TEST_CASE_2d])
    def test_GetEncSConv(self, enc_t, enc_s, dims):
        enc_s_and_conv1 = GetEncSConv().get(enc_t, enc_s, dims)
        enc_s_and_conv2 = GetEncSConv().get(enc_t, enc_s, dims)
        enc_s_and_conv3 = GetEncSConv().get(enc_t, enc_s, dims)

        self.assertIs(enc_s_and_conv1, enc_s_and_conv2)
        self.assertIs(enc_s_and_conv1, enc_s_and_conv3)



if __name__ == "__main__":
    unittest.main()
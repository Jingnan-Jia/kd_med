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
from kd_med.kd_loss import get_enc_plus_conv

import numpy as np

TEST_CASE_3d_1 = [nn.Sequential(nn.Conv3d(1, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
                                nn.Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3))),
                  nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
                  torch.ones((2, 1, 64, 64, 32)),
                  3]

TEST_CASE_3d_2 = [nn.Sequential(nn.Conv3d(1, 48, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
                                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                                nn.Conv3d(48, 18, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3))),
                  nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(3, 3, 3)),
                  torch.ones((2, 1, 32, 16, 8)),
                  3]

TEST_CASE_2d = [nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
                              nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3))),
                nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
                torch.ones((2, 1, 64, 32)),
                2]


class Testget_enc_plus_conv(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3d_1, TEST_CASE_3d_2, TEST_CASE_2d])
    def test_get_enc_plus_conv(self, enc_t, enc_s, input_img, dims):
        enc_s_and_conv1 = get_enc_plus_conv(enc_t, enc_s, dims)
        # enc_s_and_conv2 = GetEncSConv().get(enc_t, enc_s, dims)
        # enc_s_and_conv3 = GetEncSConv().get(enc_t, enc_s, dims)
        # # print(enc_s_and_conv1)
        #
        # self.assertIs(enc_s_and_conv1, enc_s_and_conv2)
        # self.assertIs(enc_s_and_conv1, enc_s_and_conv3)

        self.assertEqual(enc_s_and_conv1(input_img).shape, enc_t(input_img).shape)


if __name__ == "__main__":
    unittest.main()

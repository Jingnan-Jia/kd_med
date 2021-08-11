# -*- coding: utf-8 -*-
# @Time    : 7/28/21 3:23 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import unittest
import torch
from parameterized import parameterized

from kd_med import resnet3denc
from kd_med.pre_trained_enc import pre_trained_enc
from kd_med.unet3denc import UNet3DEnc

TEST_CASE_1 = ["resnet3d_10", resnet3denc.resnet10enc(shortcut_type='B', num_seg_classes=99)]
TEST_CASE_2 = ["resnet3d_18", resnet3denc.resnet18enc(shortcut_type='A', num_seg_classes=99)]
TEST_CASE_3 = ["resnet3d_34", resnet3denc.resnet34enc(shortcut_type='A', num_seg_classes=99)]
TEST_CASE_4 = ["resnet3d_50", resnet3denc.resnet50enc(shortcut_type='B', num_seg_classes=99)]
TEST_CASE_5 = ["resnet3d_101", resnet3denc.resnet101enc(shortcut_type='B', num_seg_classes=99)]
TEST_CASE_6 = ["resnet3d_152", resnet3denc.resnet152enc(shortcut_type='B', num_seg_classes=99)]
TEST_CASE_7 = ["resnet3d_200", resnet3denc.resnet200enc(shortcut_type='B', num_seg_classes=99)]
TEST_CASE_8 = ["unet3d", UNet3DEnc()]


class Testpre_trained_enc(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4,
                           TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_pre_trained_enc(self, net_name, expected_net):
        net = pre_trained_enc(net_name)
        layer_names = [name for name, param in net.named_parameters()]
        expected_names = [name for name, param in expected_net.named_parameters()]
        self.assertEqual(layer_names, expected_names)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4,
                           TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_pre_trained_enc_same_weights(self, net_name, expected_net):
        net1 = pre_trained_enc(net_name)
        net2 = pre_trained_enc(net_name)
        weights_1 = list(net1.parameters())
        weights_2 = list(net2.parameters())
        for w1, w2 in zip(weights_1, weights_2):
            self.assertTrue(torch.all(torch.eq(w1, w2)).item())


if __name__ == "__main__":
    unittest.main()

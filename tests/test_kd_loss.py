# -*- coding: utf-8 -*-
# @Time    : 7/28/21 8:34 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import unittest
import tempfile
import os
import torch
import torch.nn as nn
import copy
from parameterized import parameterized

from kd_med.kd_loss import kd_loss
from kd_med import EncPlusConv
from kd_med.pre_trained_enc import pre_trained_enc
import kd_med

class Cnn2_3dEnc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.level_node = level_node
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            # nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(base * 2 * 6 * 6 * 6, fc1_nodes),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(fc1_nodes, fc2_nodes),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(fc2_nodes, num_classes),
        # )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        if self.level_node == 0:
            x = input
        else:
            x, level = input[0], input[1]

        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class Vgg11_3dEnc(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8, level_node = 0):
        super().__init__()
        self.num_classes = num_classes
        self.base = base
        self.level_node = level_node

        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),)

        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.ft = nn.Flatten()
        self.dp1 = nn.Dropout()

        if self.level_node != 0:
            nb_fc0 = base * 16 * 6 * 6 * 6 + 1
        else:
            nb_fc0 = base * 16 * 6 * 6 * 6

        self.ln1 = nn.Linear(nb_fc0, fc1_nodes)
        self.rl1 = nn.ReLU(inplace=True)

        self.dp2 = nn.Dropout()
        self.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.rl2 = nn.ReLU(inplace=True)

        self.dp3 = nn.Dropout()
        self.ln3 = nn.Linear(fc2_nodes, self.num_classes)

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = self.features(input)

        return x

dims = 3
batch_x = torch.rand((2, 1, 96, 96, 64))
enc_s_ls = [Cnn2_3dEnc(), Vgg11_3dEnc()]  # todo: VGG11 #,,
enc_t_name_ls = [ 'resnet3d_10', 'resnet3d_18', 'resnet3d_18', 'resnet3d_34','resnet3d_50']



class Testkd_loss(unittest.TestCase):
    def test_kd_loss(self):
        batch_x = torch.rand((2, 1, 96, 96, 64))

        for enc_s in enc_s_ls:
            for enc_t_name in enc_t_name_ls:
                enc_t = pre_trained_enc(enc_t_name)
                enc_s_conv = EncPlusConv(batch_x, enc_t, enc_s, dims).get()

                enc_t_parameters = copy.deepcopy(list(enc_t.parameters()))
                enc_s_conv_parameters = copy.deepcopy(list(enc_s_conv.parameters()))

                loss = kd_loss(batch_x, enc_t, enc_s_conv)
                opt = torch.optim.Adam(enc_s_conv.parameters(), lr=0.001)
                opt.zero_grad()
                loss.backward()
                opt.step()

                for par, exp_par in zip(enc_t_parameters, enc_t.parameters()):
                    self.assertTrue(torch.all(torch.eq(par.data, exp_par.data)).item())
                    self.assertIsNone(par.grad)
                    self.assertIsNone(exp_par.grad)

                NotAllEqual = False
                for par, exp_par in zip(enc_s_conv_parameters, enc_s_conv.parameters()):
                    if not torch.all(torch.eq(par.data, exp_par.data)).item():
                        NotAllEqual = True
                    self.assertIsNone(par.grad)
                    # self.assertIsNotNone(exp_par.grad)
                self.assertTrue(NotAllEqual)

    def test_kd_loss_cuda(self):
        batch_x = torch.rand((2, 1, 96, 96, 64))

        if torch.cuda.is_available():
            device = torch.device("cuda")
            scaler = torch.cuda.amp.GradScaler()
            for enc_s in enc_s_ls:
                for enc_t_name in enc_t_name_ls:
                    # print(f'enc_t_name: {enc_t_name}')
                    enc_t = kd_med.pre_trained_enc(enc_t_name, no_cuda=False)
                    enc_t.to(device)
                    enc_t_parameters = copy.deepcopy(list(enc_t.parameters()))

                    enc_s_conv = EncPlusConv(batch_x, enc_t, enc_s, dims, no_cuda=False).get()
                    enc_s_conv.to(device)
                    enc_s_conv_parameters = copy.deepcopy(list(enc_s_conv.parameters()))

                    opt = torch.optim.Adam(enc_s_conv.parameters(), lr=0.001)

                    batch_x = batch_x.to(device)
                    loss = kd_med.kd_loss(batch_x, enc_t, enc_s_conv, cuda=False)

                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    for par, exp_par in zip(enc_t_parameters, enc_t.parameters()):
                        self.assertTrue(torch.all(torch.eq(par.data, exp_par.data)).item())
                        self.assertIsNone(par.grad)
                        self.assertIsNone(exp_par.grad)

                    NotAllEqual = False
                    for par, exp_par in zip(enc_s_conv_parameters, enc_s_conv.parameters()):
                        if not torch.all(torch.eq(par.data, exp_par.data)).item():
                            NotAllEqual = True
                        self.assertIsNone(par.grad)
                        # self.assertIsNotNone(exp_par.grad)
                    self.assertTrue(NotAllEqual)



if __name__ == "__main__":
    unittest.main()


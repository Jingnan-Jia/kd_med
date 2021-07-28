# -*- coding: utf-8 -*-
# @Time    : 7/26/21 10:49 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import torch.nn as nn
from kd_med.get_resnet3d import generate_model
from kd_med.unet3d import UNet3D
from kd_med.resnet3d import ResNet
import os
from collections import OrderedDict
from google_drive_downloader import GoogleDriveDownloader as gdd


class Option:
    def __init__(self):
        pass


def download_weights(weights_dir: str):
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    # download weights and unzip weights

    file_id_path_dt = {'1yZF0WHUpri9ajUeJE3wmxqFUW2Pimg7g': os.path.join(weights_dir, 'resnet_10_23dataset.pth'),
                       '1vYvIamI4zWUTD7TGgatxwkMAwqTCpd_B': os.path.join(weights_dir, 'resnet_18_23dataset.pth'),
                       '1estlxOizcJmNTa85zOeyzjQ7-0rzsdK8': os.path.join(weights_dir, 'resnet_34_23dataset.pth'),
                       '18FQtxbbvGWlBXk8bGE7MphV7K0LMMhD0': os.path.join(weights_dir, 'resnet_50_23dataset.pth'),
                       '1izVHemAsKPGu3RvtyzmBJE46C2Qo-OF3': os.path.join(weights_dir, 'resnet_101.pth'),
                       '1yqIIM_PNin8lNeWlxGH_Qs5sjLkPqjSb': os.path.join(weights_dir, 'resnet_152.pth'),
                       '1FWtQqyuOUjf4zk-WbACrj5g4AnuSb_8Z': os.path.join(weights_dir, 'resnet_200.pth'),
                       '192k-iKbvDpnPc1j5Da88Ogrt11JsikVp': os.path.join(weights_dir, 'Genesis_Chest_CT.pt')
                       }
    for id, path in file_id_path_dt.items():
        if not os.path.isfile(path):
            gdd.download_file_from_google_drive(file_id=id, dest_path=path)

    return None


class UNet3DEnc(UNet3D):
    def __init__(self, output_layer = 'out_up_256'):
        super().__init__()
        self.output_layer = output_layer
        self._layers = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers, [getattr(self, l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


# class ResNet3DEnc(ResNet):
#     def __init__(self, output_layer = 'conv_seg'):
#         super().__init__()
#         self.output_layer = output_layer
#         self._layers = []
#         for l in list(self._modules.keys()):
#             self._layers.append(l)
#             if l == output_layer:
#                 break
#         self.layers = OrderedDict(zip(self._layers, [getattr(self, l) for l in self._layers]))
#
#     def _forward_impl(self, x):
#         for l in self._layers:
#             x = self.layers[l](x)
#         return x
#
#     def forward(self, x):
#         return self._forward_impl(x)



def rename_layers(loaded_dict):
    state_dict = loaded_dict['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    return unParalled_state_dict


def pretrained_enc(net_name: str):
    """
    Availabel net_name:
    Med3D (https://arxiv.org/pdf/1904.00625.pdf):
        resnet3d_10, resnet3d_18, resnet3d_34, resnet3d_50, resnet3d_101, resnet3d_152, resnet3d_200
    Model Genesis (https://arxiv.org/pdf/1908.06912.pdf):
        unet3d
        resnet2d_18  # I do not have its pre-trained weights yet, because it is not released.

opt: require following values: model, model_depth, input_W, input_H, input_D,
    resnet_shortcut, no_cuda, n_seg_classes, phase, pretrain_path

    :param net_name:
    :return:
    """
    # weights_dir = 'pretrained_weights'
    weights_dir = os.path.join(os.path.expanduser('~/.cache'), 'kd_med', 'pretrained_weights')

    download_weights(weights_dir)

    opt = Option()
    if 'resnet3d' in net_name:
        opt.model = 'resnet'  # do not change it
        opt.model_depth = int(net_name.split('_')[-1])
        if opt.model_depth in [18, 34]:  # https://github.com/Tencent/MedicalNet
            opt.resnet_shortcut = 'A'
        else:
            opt.resnet_shortcut = 'B'
        opt.no_cuda = True  # I think it does not matter
        opt.n_seg_classes = 2  # I think it does not matter
        if opt.model_depth in [10, 18, 34, 50]:
            opt.pretrain_path = weights_dir + "/resnet_" + str(opt.model_depth) + "_23dataset.pth"  # reset if needed
        else:
            opt.pretrain_path = weights_dir + "/resnet_" + str(opt.model_depth) + ".pth"  # reset if needed


        enc = generate_model(opt)  # generate resnet encoder
        # enc = get_enc(enc, 'conv_seg')
        enc.load_state_dict(rename_layers(torch.load(opt.pretrain_path)))

    elif 'resnet2d' in net_name:
        raise NotImplementedError
    elif 'unet3d' == net_name:
        weights_fpath = weights_dir + '/Genesis_Chest_CT.pt'
        base_model = UNet3DEnc()  # Unet encoder

        # Load pre-trained weights
        checkpoint = torch.load(weights_fpath)
        # state_dict = checkpoint['state_dict']
        # unParalled_state_dict = {}
        # for key in state_dict.keys():
        #     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        base_model.load_state_dict(rename_layers(checkpoint))
        enc = base_model  # encoder
    else:
        raise Exception(f'wrong net name {net_name}')
    return enc


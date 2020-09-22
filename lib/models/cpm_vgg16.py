# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from copy import deepcopy
# from .model_utils import get_parameters
# from .basic_batch import find_tensor_peak_batch
# from .initialization import weights_init_cpm
import torch.nn.functional as F
import numbers, math
import numpy as np


def weights_init_cpm(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L - 1)

    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    # affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    # theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius * 2 + 1, radius * 2 + 1])
    grid = F.affine_grid(theta, grid_size, align_corners=True)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, align_corners=True).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius + 1).to(heatmap).view(1, 1, radius * 2 + 1)
    Y = torch.arange(-radius, radius + 1).to(heatmap).view(1, radius * 2 + 1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1), score


def get_parameters(model, bias):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.BatchNorm2d):
            if bias:
                yield m.bias
            else:
                yield m.weight


def remove_module_dict(state_dict, is_print=False):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    if is_print: print(new_state_dict.keys())
    return new_state_dict


BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class VGG16_base(nn.Module):
    def __init__(self, config, pts_num):
        self.inplanes = 128
        self.deconv_with_bias = False
        super(VGG16_base, self).__init__()

        self.config = deepcopy(config)
        self.downsample = 8
        self.pts_num = pts_num
        self.heads = {'hm': 1, 'hmhp_offset': 136}
        self.config_stages = 3

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.CPM_feature = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),  # CPM_1
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))  # CPM_2

        assert self.config_stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config_stages)
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.config_stages):
            stagex = nn.Sequential(
                nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

        # Decoder of Hm and offset
        self.layer3_forhm = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_forhm = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.deconv_layers_forhm = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(256, 128,
                          kernel_size=3, padding=1,  stride=2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_output,
                          kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)

        # up sampling of motion field


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [
            {'params': get_parameters(self.features, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
        ]
        for stage in self.stages:
            params_dict.append(
                {'params': get_parameters(stage, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay})
            params_dict.append({'params': get_parameters(stage, bias=True), 'lr': base_lr * 8, 'weight_decay': 0})
        return params_dict

    # return : cpm-stages, locations
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms, batch_locs, batch_scos = [], [], []

        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)

        x_forhm = self.layer3_forhm(xfeature)
        x_forhm = self.layer4_forhm(x_forhm)
        x_forhm = self.deconv_layers_forhm(x_forhm)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x_forhm)
        # return [ret]

        for i in range(self.config_stages):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]], 1))
            batch_cpms.append(cpm)

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], 20,
                                                                 self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

        return batch_cpms, batch_locs, batch_scos, ret

    def init_weights(self, num_layers=18, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers_forhm.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            url = model_urls_res18
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)


# use vgg16 conv1_1 to conv4_4 as feature extracation
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'
model_urls_res18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def cpm_vgg16(config, pts):
    print('Initialize cpm-vgg16 with configure : {}'.format(config))
    model = VGG16_base(config, pts)
    model.apply(weights_init_cpm)
    model.init_weights(18, pretrained=True)

    if config.pretrained:
        print('vgg16_base use pre-trained model')
        weights = model_zoo.load_url(model_urls)
        model.load_state_dict(weights, strict=False)
        weights_res = model_zoo.load_url(model_urls_res18)
        model.load_state_dict(weights_res, strict=False)
    return model

if __name__=='__main__':
    model = VGG16_base(None, 68)
    input = torch.rand((10, 3, 256, 256))
    out = model(input)
    from thop import profile
    flop, params = profile(model, inputs=(input, ))
    print('flop:{}, params:{}'.format(flop, params))
    pass

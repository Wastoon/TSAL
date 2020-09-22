# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .cpm_vgg16 import cpm_vgg16
import torch


def obtain_model_pca(configure, points, heatmap_featurespace, heatmap_mean):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16_pca(configure, points, heatmap_featurespace, heatmap_mean)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net


def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'hourglass':
    net = get_large_hourglass_net(configure, points)
    #inputs = torch.randn((1,3,256,256))
    #out = net(inputs)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net


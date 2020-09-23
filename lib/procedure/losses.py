# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import numbers, torch
import torch.nn.functional as F

def compute_stage_loss(criterion, targets, outputs, masks):
  assert isinstance(outputs, list), 'The ouputs type is wrong : {:}'.format(type(outputs))
  total_loss = 0
  each_stage_loss = []
  masks = masks.type(torch.bool)
  #masks = masks.ge(0.5)
  #masks = masks.repeat(1, 1, 32, 32)
  for output in outputs:
    stage_loss = 0
    output = torch.masked_select(output , masks)
    target = torch.masked_select(targets, masks)
    #output = output.mul(masks).flatten()
    #target = targets.mul(masks).flatten()
    stage_loss = criterion(output, target)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  return total_loss, each_stage_loss

def compute_stage_loss_unsupervised(criterion, targets, outputs, seq_length):
  assert isinstance(outputs, list), 'The ouputs type is wrong : {:}'.format(type(outputs))
  total_loss = 0
  each_stage_loss = []
  video_batch_num = outputs.shape[0] // seq_length
  calculate_loss_id = torch.tensor([j * seq_length + i for j in range(video_batch_num) for i in range(seq_length - 1)])

  for out, tar in zip(outputs, targets):
    stage_loss = 0
    out = out.index_select(0, calculate_loss_id)
    stage_loss = criterion(out, tar)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  return total_loss, stage_loss

def compute_stage_loss_unsupervised_Loc(criterion, targets, outputs, seq_length):
  assert isinstance(outputs, list), 'The ouputs type is wrong : {:}'.format(type(outputs))
  total_loss = 0

  video_batch_num = outputs.shape[0] // seq_length
  calculate_loss_id = torch.tensor([j * seq_length + i for j in range(video_batch_num) for i in range(seq_length - 1)])

  outputs = outputs.index_select(0, calculate_loss_id)
  total_loss = criterion(outputs, targets)
  return total_loss


def compute_stage_loss_MFDS(criterion, targets, outputs):
  assert isinstance(outputs, list), 'The ouputs type is wrong : {:}'.format(type(outputs))
  assert isinstance(targets, list), 'The ouputs type is wrong : {:}'.format(type(targets))
  total_loss = 0
  each_stage_loss = []

  for out, tar in zip(outputs, targets):
    stage_loss = 0
    stage_loss = criterion(out, tar)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  return total_loss, stage_loss




def compute_FCMIL_loss(criterion, targets, outputs):
  outputs = outputs.unsqueeze(2).unsqueeze(3)
  FCMIL_loss = criterion(outputs, targets)
  return  FCMIL_loss

def compute_stage_loss_pca(criterion, target, outputs):
  total_loss = 0
  each_stage_loss = []

  for output in outputs:
    stage_loss = 0
    stage_loss = criterion(output, target)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  return total_loss, each_stage_loss



def show_stage_loss(each_stage_loss):
  if each_stage_loss is None:            return 'None'
  elif isinstance(each_stage_loss, str): return each_stage_loss
  answer = ''
  for index, loss in enumerate(each_stage_loss):
    answer = answer + ' : L{:1d}={:7.4f}'.format(index+1, loss)
  return answer

def show_stage_loss_pca(each_stage_loss):
  if each_stage_loss is None:            return 'None'
  elif isinstance(each_stage_loss, str): return each_stage_loss
  answer = ''
  for index, loss in enumerate(each_stage_loss):
    answer = answer + ' PCAloss: L{:1d}={:7.4f}'.format(index+1, loss)
  return answer

def show_loss(loss):
  if loss is None:            return 'None'
  elif isinstance(loss, str): return loss
  answer = ''
  answer = answer + 'FC_MIL-loss :{:7.4f}'.format(loss)
  return answer


def sum_stage_loss(losses):
  total_loss = None
  each_stage_loss = []
  for loss in losses:
    if total_loss is None:
      total_loss = loss
    else:
      total_loss = total_loss + loss
    each_stage_loss.append(loss.data[0])
  return total_loss, each_stage_loss

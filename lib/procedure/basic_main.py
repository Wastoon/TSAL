# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, os
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from xvision import Eval_Meta
from log_utils import AverageMeter, time_for_file, convert_secs2time
from .losses import compute_stage_loss, show_stage_loss, compute_FCMIL_loss, show_loss, compute_stage_loss_pca, \
    show_stage_loss_pca, compute_stage_loss_unsupervised
from utils.debugger import Debugger
from lib.criterion import _sigmoid, FocalLoss, RegLoss, _tranpose_and_gather_feat
from models.MFDS_model import unFlowLoss
from easydict import EasyDict
import json
import math

# train function (forward, backward, update)
pause = False
USE_MFEM = True

def rampweight(iteration, total_iteration=150, data_len=3148):
    ramp_up_end = data_len*60
    ramp_down_start = data_len * 110
    iter_max = total_iteration

    beta = 1

    # if (iteration < 100):
    #     ramp_weight = 0
    if (iteration < ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2)) * beta
    elif (iteration > ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (iter_max - iteration) / (iter_max - ramp_down_start)), 2)) * beta
    else:
        ramp_weight = 1 * beta

    if (iteration == 0):
        ramp_weight = 0
    return ramp_weight


def basic_train(args, loader, net, MFEM, criterion, optimizer, optimizer_MFEM, epoch_str, logger, opt_config):
    args = deepcopy(args)
    batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    visible_points, losses = AverageMeter(), AverageMeter()
    eval_meta = Eval_Meta()
    cpu = torch.device('cpu')
    cur_epoch = int(epoch_str.split('-')[1])
    total_epoch = int(epoch_str.split('-')[2])
    total_iteration = total_epoch * len(loader)

    # switch to train mode
    MFEM.train()
    Uncupervised_criterion = torch.nn.MSELoss().cuda()
    end = time.time()
    #import pdb
    #pdb.set_trace()
    for i, (inputs, target, mask, points, image_index, nopoints, cropped_size, nose_center_hm, hp_offset_Lco, kps_mask, nose_ind) in enumerate(loader):
        # inputs : Batch, Channel, Height, Width
        inputs = inputs.cuda()
        seq_length = inputs.shape[1] // 3
        cur_iteration = cur_epoch*len(loader) +i
        tracking_source_reconsimg_list = []
        weight_decay_unsupervisied = rampweight(cur_iteration, total_iteration=total_iteration, data_len=len(loader))
        if USE_MFEM:
            loss_MFEM = 0
            MFDS_consistency_loss = 0
            for j in range(seq_length-1):
               res_dict = MFEM(inputs[:,j*3:j*3+6,:,:], with_bk=True)
               flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
               flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
               cfg = EasyDict(json.load(open('/home/mry/PycharmProjects/SALD/configs/MFDS.json')))
               cfg = cfg.loss
               loss_func = unFlowLoss(cfg)
               loss_interframe, l_ph, l_sm, flow_mean, FW_consistency_loss, BW_consistency_loss, unsupervisied_target_by_tracking = loss_func(flows, inputs[:,i*3:i*3+6,:,:])
               tracking_source_reconsimg_list.append(unsupervisied_target_by_tracking)
               loss_MFEM += loss_interframe
               MFDS_consistency_loss += FW_consistency_loss + BW_consistency_loss
            optimizer_MFEM.zero_grad()
            scaled_loss = 1024. * loss_MFEM
            weight = 1e-3
            scaled_loss = scaled_loss * weight
            MFDS_consistency_loss = weight_decay_unsupervisied*MFDS_consistency_loss
            MFEM_total_loss = MFDS_consistency_loss + scaled_loss
            MFEM_total_loss.backward()
            for param in [p for p in MFEM.parameters() if p.requires_grad]:
               param.grad.data.mul_(1. / 1024)
            optimizer_MFEM.step()

        # switch to train mode
        net.train()
        criterion.train()

        inputs = inputs.view(-1, 3, inputs.shape[2], inputs.shape[3])
        input_from_tracking = torch.cat(tracking_source_reconsimg_list, dim=1)  ##batch*(27)*H*W
        input_from_tracking = input_from_tracking.view(-1, 3, inputs.shape[2], inputs.shape[3]) ##(batch*9)*3*H*W
        target = target.view(-1, args.num_pts+1, target.shape[2], target.shape[3])
        mask = mask.view(-1,args.num_pts+1, mask.shape[2], mask.shape[3])
        points = points.view(-1,args.num_pts, points.shape[2])
        image_index = image_index.view(-1,1)
        nopoints = nopoints.view(-1, 1)
        cropped_size = cropped_size.view(-1, 4)
        #nose_center_hm = torch.from_numpy(nose_center_hm)
        nose_center_hm = nose_center_hm.view(-1, nose_center_hm.shape[2], nose_center_hm.shape[2])
        #hp_offset_Lco = torch.from_numpy(hp_offset_Lco)
        hp_offset_Lco = hp_offset_Lco.view(-1, 32, hp_offset_Lco.shape[2])
        #kps_mask = torch.from_numpy(kps_mask)
        kps_mask = kps_mask.view(-1, 32, hp_offset_Lco.shape[2])
        #nose_ind = torch.from_numpy(nose_ind)
        nose_ind = nose_ind.view(-1, 32)

        target = target.cuda(non_blocking=True)

        image_index = image_index.numpy().squeeze(1).tolist()
        batch_size, num_pts = inputs.size(0), args.num_pts
        visible_point_num = float(np.sum(mask.numpy()[:, :-1, :, :])) / batch_size
        visible_points.update(visible_point_num, batch_size)
        nopoints = nopoints.numpy().squeeze(1).tolist()
        annotated_num = batch_size - sum(nopoints)

        # measure data loading time
        mask = mask.cuda(non_blocking=True)
        nose_center_hm = nose_center_hm.cuda()
        kps_mask = kps_mask.cuda()
        nose_ind = nose_ind.cuda()
        hp_offset_Lco = hp_offset_Lco.cuda()
        data_time.update(time.time() - end)

        batch_heatmaps, batch_locs, batch_scos, CLOR_dict = net(inputs)

        with torch.no_grad:
            Unsupervisied_traget_batch_heatmaps, Unsupervisied_traget_batch_locs, Unsupervisied_traget_batch__scos, Unsupervisied_traget_CLOR_dict = net(input_from_tracking)

        #Loss for nose center of face
        nose_center_hm_loss, Loc_loss = 0, 0
        nose_hm_crit = FocalLoss()
        CLOR_dict['hm'] = _sigmoid(CLOR_dict['hm'])
        nose_center_hm_loss += nose_hm_crit(CLOR_dict['hm'], nose_center_hm)

        #Loss for Lco regress
        Lco_crit = RegLoss()
        Loc_loss += Lco_crit(CLOR_dict['hmhp_offset'],kps_mask,nose_ind,hp_offset_Lco)

        #Loss for SIC
        CLOR_dict_SIC= CLOR_dict['hmhp_offset'].clone()
        CLOR_dict_SIC = _tranpose_and_gather_feat(CLOR_dict_SIC, nose_ind)
        CLOR_dict_SIC = CLOR_dict_SIC*kps_mask
        CLOR_dict_SIC = CLOR_dict_SIC.view(CLOR_dict_SIC.shape[0], CLOR_dict_SIC.shape[1], 2, -1)
        x = CLOR_dict_SIC[:,:,0,:]
        y = CLOR_dict_SIC[:,:,1,:]
        norm = x-y
        norm = norm.mul(norm)
        modulus_per_frames = torch.sum(norm, dim=2)
        modulus_per_frames = torch.sum(modulus_per_frames, dim=1)
        ##SIC loss
        loss_SIC = 0
        for cur_frames_id in range(modulus_per_frames.shape[0]-1):
            cur_frame_loss = 0
            for subframe_id in range(cur_frames_id, modulus_per_frames.shape[0]):
                cur_frame_loss += (modulus_per_frames[cur_frames_id] - modulus_per_frames[subframe_id]).pow(2)
            loss_SIC += cur_frame_loss
        loss_SIC = loss_SIC / (args.num_pts * modulus_per_frames.shape[0])

        forward_time.update(time.time() - end)

        ##Unsupervised_img_loss
        Uncupervised_criterion.train()
        Unsupervisied_total_loss, Unsupervisied_stage_loss = compute_stage_loss_unsupervised(Uncupervised_criterion, Unsupervisied_traget_batch_heatmaps, batch_heatmaps, seq_length)
        Unsupervisied_total_loss = weight_decay_unsupervisied * Unsupervisied_total_loss

        # FCMIL_loss = compute_FCMIL_loss(criterion, mask, FC_MIL)
        if np.random.random() < 1:
            fake_mask = mask
        else:
            pass
        # fake_mask = FC_MIL.detach().clone().unsqueeze(2).unsqueeze(3)
        loss = 0
        loss, each_stage_loss_value = compute_stage_loss(criterion, target, batch_heatmaps, fake_mask)

        if opt_config.lossnorm:
            loss, each_stage_loss_value = loss / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                         each_stage_loss_value]
            Unsupervisied_total_loss, Unsupervisied_stage_loss = Unsupervisied_total_loss / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                     Unsupervisied_stage_loss]

        #Adding SIC , nosecenterHM, Loc regress
        loss = loss + 0.01*loss_SIC + nose_center_hm_loss + 0.1*Loc_loss + Unsupervisied_total_loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        eval_time.update(time.time() - end)

        np_batch_locs, np_batch_scos = batch_locs.detach().to(cpu).numpy(), batch_scos.detach().to(cpu).numpy()
        cropped_size = cropped_size.numpy()
        # evaluate the training data

        # measure elapsed time
        batch_time.update(time.time() - end)
        last_time = convert_secs2time(batch_time.avg * (len(loader) - i - 1), True)
        end = time.time()


        logger.log(' -->>[Train]: [{:}][{:03d}/{:03d}] '
                   'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                   'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                   'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                   'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
            epoch_str, i, len(loader), batch_time=batch_time,
            data_time=data_time, forward_time=forward_time, loss=losses)
                   + last_time + show_stage_loss(each_stage_loss_value) \
                   + ' In={:} Tar={:}'.format(list(inputs.size()), list(target.size())) \
                   + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                   + 'Loss_SIC:{loss:7.4f} ,Loss_noseHM {loss_noseHM:7.4f}, Loss_Loc {loss_loc:7.4f}'.format(loss=loss_SIC, loss_noseHM=nose_center_hm_loss, loss_loc=Loc_loss) \
                   + 'Loss_motion {loss_motion:7.4f}'.format(loss_motion=MFEM_total_loss) +'Loss_motion_MFDS {loss_motion_MFDS:7.4f}'.format(loss_motion_MFDS=MFDS_consistency_loss)  \
                   + 'Loss_unsupervised_tracking_target {loss_unsupervised_tracking_target:7.4f}'.format(loss_unsupervised_tracking_target=Unsupervisied_total_loss)
                   )

    #nme, _, _ = eval_meta.compute_mse(logger)
    return losses.avg#, nme


def basic_train_Rma_model(args, loader, net, MFEM, net_Rma, MFEM_Rma, criterion, criterion_Rma,  optimizer, optimizer_Rma, optimizer_MFEM, optimizer_MFEM_Rma, epoch_str, logger, opt_config):
    args = deepcopy(args)
    batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    visible_points, losses, losses_Rma = AverageMeter(), AverageMeter(), AverageMeter()
    eval_meta = Eval_Meta()
    cpu = torch.device('cpu')
    cur_epoch = int(epoch_str.split('-')[1])
    total_epoch = int(epoch_str.split('-')[2])
    total_iteration = total_epoch * len(loader)

    # switch to train mode
    MFEM.train()
    MFEM_Rma.train()
    Uncupervised_criterion = torch.nn.MSELoss().cuda()
    end = time.time()
    #import pdb
    #pdb.set_trace()
    for i, (inputs, target, mask, points, image_index, nopoints, cropped_size, nose_center_hm, hp_offset_Lco, kps_mask, nose_ind) in enumerate(loader):
        # inputs : Batch, Channel, Height, Width
        inputs = inputs.cuda()
        seq_length = inputs.shape[1] // 3
        cur_iteration = cur_epoch*len(loader) +i
        tracking_source_reconsimg_list = []
        tracking_source_reconsimg_Rma_list = []

        weight_decay_unsupervisied = rampweight(cur_iteration, total_iteration=total_iteration, data_len=len(loader))
        if USE_MFEM:
            loss_MFEM = 0
            MFDS_consistency_loss = 0
            for j in range(seq_length-1):
               res_dict = MFEM(inputs[:,j*3:j*3+6,:,:], with_bk=True)
               flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
               flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
               cfg = EasyDict(json.load(open('/home/mry/PycharmProjects/SALD/configs/MFDS.json')))
               cfg = cfg.loss
               loss_func = unFlowLoss(cfg)
               loss_interframe, l_ph, l_sm, flow_mean, FW_consistency_loss, BW_consistency_loss, unsupervisied_target_by_tracking = loss_func(flows, inputs[:,i*3:i*3+6,:,:])
               tracking_source_reconsimg_list.append(unsupervisied_target_by_tracking)
               loss_MFEM += loss_interframe
               MFDS_consistency_loss += FW_consistency_loss + BW_consistency_loss
            optimizer_MFEM.zero_grad()
            scaled_loss = 1024. * loss_MFEM
            weight = 1e-3
            scaled_loss = scaled_loss * weight
            MFDS_consistency_loss = weight_decay_unsupervisied*MFDS_consistency_loss
            MFEM_total_loss = MFDS_consistency_loss + scaled_loss
            MFEM_total_loss.backward()
            for param in [p for p in MFEM.parameters() if p.requires_grad]:
               param.grad.data.mul_(1. / 1024)
            optimizer_MFEM.step()

            loss_MFEM_Rma = 0
            MFDS_consistency_loss_Rma = 0
            for j in range(seq_length-1):
               res_dict = MFEM_Rma(inputs[:,j*3:j*3+6,:,:], with_bk=True)
               flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
               flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
               cfg = EasyDict(json.load(open('/home/mry/PycharmProjects/SALD/configs/MFDS.json')))
               cfg = cfg.loss
               loss_func = unFlowLoss(cfg)
               loss_interframe_Rma, l_ph_Rma, l_sm_Rma, flow_mean_Rma, FW_consistency_loss_Rma, BW_consistency_loss_Rma, unsupervisied_target_by_tracking_Rma = loss_func(flows, inputs[:,i*3:i*3+6,:,:])
               tracking_source_reconsimg_Rma_list.append(unsupervisied_target_by_tracking_Rma)
               loss_MFEM_Rma += loss_interframe_Rma
               MFDS_consistency_loss_Rma += FW_consistency_loss_Rma + BW_consistency_loss_Rma
            optimizer_MFEM_Rma.zero_grad()
            scaled_loss_Rma = 1024. * loss_MFEM_Rma
            weight = 1e-3
            scaled_loss_Rma = scaled_loss_Rma * weight
            MFDS_consistency_loss_Rma = weight_decay_unsupervisied*MFDS_consistency_loss_Rma
            MFEM_total_loss_Rma = MFDS_consistency_loss_Rma + scaled_loss_Rma
            MFEM_total_loss_Rma.backward()
            for param in [p for p in MFEM_Rma.parameters() if p.requires_grad]:
               param.grad.data.mul_(1. / 1024)
            optimizer_MFEM_Rma.step()


        # switch to train mode
        net.train()
        criterion.train()
        net_Rma.train()
        criterion_Rma.train()

        inputs = inputs.view(-1, 3, inputs.shape[2], inputs.shape[3])
        input_from_tracking_Rma = torch.cat(tracking_source_reconsimg_Rma_list, dim=1)  ##batch*(27)*H*W
        input_from_tracking = torch.cat(tracking_source_reconsimg_list, dim=1)  ##batch*(27)*H*W
        input_from_tracking_Rma = input_from_tracking_Rma.view(-1, 3, inputs.shape[2], inputs.shape[3]) ##(batch*9)*3*H*W
        input_from_tracking = input_from_tracking.view(-1, 3, inputs.shape[2], inputs.shape[3])  ##(batch*9)*3*H*W
        target = target.view(-1, args.num_pts+1, target.shape[2], target.shape[3])
        mask = mask.view(-1,args.num_pts+1, mask.shape[2], mask.shape[3])
        points = points.view(-1,args.num_pts, points.shape[2])
        image_index = image_index.view(-1,1)
        nopoints = nopoints.view(-1, 1)
        cropped_size = cropped_size.view(-1, 4)
        #nose_center_hm = torch.from_numpy(nose_center_hm)
        nose_center_hm = nose_center_hm.view(-1, nose_center_hm.shape[2], nose_center_hm.shape[2])
        #hp_offset_Lco = torch.from_numpy(hp_offset_Lco)
        hp_offset_Lco = hp_offset_Lco.view(-1, 32, hp_offset_Lco.shape[2])
        #kps_mask = torch.from_numpy(kps_mask)
        kps_mask = kps_mask.view(-1, 32, hp_offset_Lco.shape[2])
        #nose_ind = torch.from_numpy(nose_ind)
        nose_ind = nose_ind.view(-1, 32)

        target = target.cuda(non_blocking=True)

        image_index = image_index.numpy().squeeze(1).tolist()
        batch_size, num_pts = inputs.size(0), args.num_pts
        visible_point_num = float(np.sum(mask.numpy()[:, :-1, :, :])) / batch_size
        visible_points.update(visible_point_num, batch_size)
        nopoints = nopoints.numpy().squeeze(1).tolist()
        annotated_num = batch_size - sum(nopoints)

        # measure data loading time
        mask = mask.cuda(non_blocking=True)
        nose_center_hm = nose_center_hm.cuda()
        kps_mask = kps_mask.cuda()
        nose_ind = nose_ind.cuda()
        hp_offset_Lco = hp_offset_Lco.cuda()
        data_time.update(time.time() - end)

        batch_heatmaps, batch_locs, batch_scos, CLOR_dict = net(inputs)
        batch_heatmaps_Rma, batch_locs_Rma, batch_scos_Rma, CLOR_dict_Rma = net(inputs)


        with torch.no_grad:
            Unsupervisied_traget_batch_heatmaps, Unsupervisied_traget_batch_locs, Unsupervisied_traget_batch__scos, Unsupervisied_traget_CLOR_dict = net(input_from_tracking)
            Unsupervisied_traget_batch_heatmaps_Rma, Unsupervisied_traget_batch_locs_Rma, Unsupervisied_traget_batch__scos_Rma, Unsupervisied_traget_CLOR_dict_Rma = net_Rma(
                input_from_tracking_Rma)


        #Loss for nose center of face
        nose_center_hm_loss, Loc_loss = 0, 0
        nose_hm_crit = FocalLoss()
        CLOR_dict['hm'] = _sigmoid(CLOR_dict['hm'])
        nose_center_hm_loss += nose_hm_crit(CLOR_dict['hm'], nose_center_hm)

        nose_center_hm_loss_Rma, Loc_loss_Rma = 0, 0
        CLOR_dict_Rma['hm'] = _sigmoid(CLOR_dict_Rma['hm'])
        nose_center_hm_loss_Rma += nose_hm_crit(CLOR_dict_Rma['hm'], nose_center_hm)

        #Loss for Lco regress
        Lco_crit = RegLoss()
        Loc_loss += Lco_crit(CLOR_dict['hmhp_offset'],kps_mask,nose_ind,hp_offset_Lco)

        #Loss for Lco regress
        Loc_loss_Rma += Lco_crit(CLOR_dict_Rma['hmhp_offset'],kps_mask,nose_ind,hp_offset_Lco)

        #Loss for SIC
        CLOR_dict_SIC= CLOR_dict['hmhp_offset'].clone()
        CLOR_dict_SIC = _tranpose_and_gather_feat(CLOR_dict_SIC, nose_ind)
        CLOR_dict_SIC = CLOR_dict_SIC*kps_mask
        CLOR_dict_SIC = CLOR_dict_SIC.view(CLOR_dict_SIC.shape[0], CLOR_dict_SIC.shape[1], 2, -1)
        x = CLOR_dict_SIC[:,:,0,:]
        y = CLOR_dict_SIC[:,:,1,:]
        norm = x-y
        norm = norm.mul(norm)
        modulus_per_frames = torch.sum(norm, dim=2)
        modulus_per_frames = torch.sum(modulus_per_frames, dim=1)
        ##SIC loss
        loss_SIC = 0
        for cur_frames_id in range(modulus_per_frames.shape[0]-1):
            cur_frame_loss = 0
            for subframe_id in range(cur_frames_id, modulus_per_frames.shape[0]):
                cur_frame_loss += (modulus_per_frames[cur_frames_id] - modulus_per_frames[subframe_id]).pow(2)
            loss_SIC += cur_frame_loss
        loss_SIC = loss_SIC / (args.num_pts * modulus_per_frames.shape[0])

        CLOR_dict_SIC_Rma= CLOR_dict_Rma['hmhp_offset'].clone()
        CLOR_dict_SIC_Rma = _tranpose_and_gather_feat(CLOR_dict_SIC_Rma, nose_ind)
        CLOR_dict_SIC_Rma = CLOR_dict_SIC_Rma*kps_mask
        CLOR_dict_SIC_Rma = CLOR_dict_SIC_Rma.view(CLOR_dict_SIC_Rma.shape[0], CLOR_dict_SIC_Rma.shape[1], 2, -1)
        x = CLOR_dict_SIC_Rma[:,:,0,:]
        y = CLOR_dict_SIC_Rma[:,:,1,:]
        norm = x-y
        norm = norm.mul(norm)
        modulus_per_frames_Rma = torch.sum(norm, dim=2)
        modulus_per_frames_Rma = torch.sum(modulus_per_frames_Rma, dim=1)
        ##SIC loss
        loss_SIC_Rma = 0
        for cur_frames_id in range(modulus_per_frames_Rma.shape[0]-1):
            cur_frame_loss = 0
            for subframe_id in range(cur_frames_id, modulus_per_frames_Rma.shape[0]):
                cur_frame_loss += (modulus_per_frames_Rma[cur_frames_id] - modulus_per_frames_Rma[subframe_id]).pow(2)
            loss_SIC_Rma += cur_frame_loss
        loss_SIC_Rma = loss_SIC_Rma / (args.num_pts * modulus_per_frames_Rma.shape[0])

        forward_time.update(time.time() - end)

        ##Unsupervised_img_loss
        Uncupervised_criterion.train()
        Unsupervisied_total_loss, Unsupervisied_stage_loss = compute_stage_loss_unsupervised(Uncupervised_criterion,
                                                                                             Unsupervisied_traget_batch_heatmaps,
                                                                                             batch_heatmaps, seq_length)
        Unsupervisied_total_loss = weight_decay_unsupervisied * Unsupervisied_total_loss

        Unsupervisied_total_loss_Rma, Unsupervisied_stage_loss_Rma = compute_stage_loss_unsupervised(Uncupervised_criterion,
                                                                                             Unsupervisied_traget_batch_heatmaps_Rma,
                                                                                             batch_heatmaps_Rma, seq_length)
        Unsupervisied_total_loss_Rma = weight_decay_unsupervisied * Unsupervisied_total_loss_Rma



        # FCMIL_loss = compute_FCMIL_loss(criterion, mask, FC_MIL)
        if np.random.random() < 1:
            fake_mask = mask
        else:
            pass
        # fake_mask = FC_MIL.detach().clone().unsqueeze(2).unsqueeze(3)
        loss = 0
        loss, each_stage_loss_value = compute_stage_loss(criterion, target, batch_heatmaps, fake_mask)

        loss_Rma = 0
        loss_Rma, each_stage_loss_value_Rma = compute_stage_loss(criterion_Rma, target, batch_heatmaps_Rma, fake_mask)

        if opt_config.lossnorm:
            loss, each_stage_loss_value = loss / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                         each_stage_loss_value]
            Unsupervisied_total_loss, Unsupervisied_stage_loss = Unsupervisied_total_loss / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                     Unsupervisied_stage_loss]

            loss_Rma, each_stage_loss_value_Rma = loss_Rma / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                     each_stage_loss_value_Rma]
            Unsupervisied_total_loss_Rma, Unsupervisied_stage_loss_Rma = Unsupervisied_total_loss_Rma / annotated_num / 2, [
                x / annotated_num / 2 for x in
                Unsupervisied_stage_loss_Rma]

        #Adding SIC , nosecenterHM, Loc regress
        loss = loss + 0.01*loss_SIC + nose_center_hm_loss + 0.1*Loc_loss + Unsupervisied_total_loss

        loss_Rma = loss_Rma + 0.01*loss_SIC_Rma + nose_center_hm_loss_Rma + 0.1*Loc_loss_Rma + Unsupervisied_total_loss_Rma
        losses.update(loss.item(), batch_size)
        losses_Rma.update(loss_Rma.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_Rma.zero_grad()
        loss_Rma.backward()
        optimizer_Rma.step()

        eval_time.update(time.time() - end)

        np_batch_locs, np_batch_scos = batch_locs.detach().to(cpu).numpy(), batch_scos.detach().to(cpu).numpy()
        cropped_size = cropped_size.numpy()
        # evaluate the training data

        # measure elapsed time
        batch_time.update(time.time() - end)
        last_time = convert_secs2time(batch_time.avg * (len(loader) - i - 1), True)
        end = time.time()


        logger.log(' -->>[TeacherTrain]: [{:}][{:03d}/{:03d}] '
                   'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                   'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                   'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                   'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
            epoch_str, i, len(loader), batch_time=batch_time,
            data_time=data_time, forward_time=forward_time, loss=losses)
                   + last_time + show_stage_loss(each_stage_loss_value) \
                   + ' In={:} Tar={:}'.format(list(inputs.size()), list(target.size())) \
                   + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                   + 'Loss_SIC:{loss:7.4f} ,Loss_noseHM {loss_noseHM:7.4f}, Loss_Loc {loss_loc:7.4f}'.format(loss=loss_SIC, loss_noseHM=nose_center_hm_loss, loss_loc=Loc_loss) \
                   + 'Loss_motion {loss_motion:7.4f}'.format(loss_motion=MFEM_total_loss) +'Loss_motion_MFDS {loss_motion_MFDS:7.4f}'.format(loss_motion_MFDS=MFDS_consistency_loss)  \
                   + 'Loss_unsupervised_tracking_target {loss_unsupervised_tracking_target:7.4f}'.format(loss_unsupervised_tracking_target=Unsupervisied_total_loss)
                   )

        logger.log(' -->>[StudentTrain]: [{:}][{:03d}/{:03d}] '
                   'Time {batch_time.val:4.2f} ({batch_time.avg:4.2f}) '
                   'Data {data_time.val:4.2f} ({data_time.avg:4.2f}) '
                   'Forward {forward_time.val:4.2f} ({forward_time.avg:4.2f}) '
                   'Loss {loss.val:7.4f} ({loss.avg:7.4f})  '.format(
            epoch_str, i, len(loader), batch_time=batch_time,
            data_time=data_time, forward_time=forward_time, loss=losses_Rma)
                   + last_time + show_stage_loss(each_stage_loss_value_Rma) \
                   + ' In={:} Tar={:}'.format(list(inputs.size()), list(target.size())) \
                   + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg) \
                   + 'Loss_SIC:{loss:7.4f} ,Loss_noseHM {loss_noseHM:7.4f}, Loss_Loc {loss_loc:7.4f}'.format(loss=loss_SIC_Rma, loss_noseHM=nose_center_hm_loss_Rma, loss_loc=Loc_loss_Rma) \
                   + 'Loss_motion {loss_motion:7.4f}'.format(loss_motion=MFEM_total_loss_Rma) +'Loss_motion_MFDS {loss_motion_MFDS:7.4f}'.format(loss_motion_MFDS=MFDS_consistency_loss_Rma)  \
                   + 'Loss_unsupervised_tracking_target {loss_unsupervised_tracking_target:7.4f}'.format(loss_unsupervised_tracking_target=Unsupervisied_total_loss_Rma)
                   )

    #nme, _, _ = eval_meta.compute_mse(logger)
    return losses.avg, losses_Rma.avg#, nme
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
    show_stage_loss_pca
from utils.debugger import Debugger
from lib.criterion import _sigmoid, FocalLoss, RegLoss, _tranpose_and_gather_feat
from models.MFDS_model import unFlowLoss
from easydict import EasyDict
import json

# train function (forward, backward, update)
pause = False


def basic_train(args, loader, net, MFEM, criterion, optimizer, optimizer_MFEM, epoch_str, logger, opt_config):
    args = deepcopy(args)
    batch_time, data_time, forward_time, eval_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    visible_points, losses = AverageMeter(), AverageMeter()
    eval_meta = Eval_Meta()
    cpu = torch.device('cpu')

    # switch to train mode
    net.train()
    criterion.train()
    MFEM.train()

    end = time.time()
    #import pdb
    #pdb.set_trace()
    for i, (inputs, target, mask, points, image_index, nopoints, cropped_size, nose_center_hm, hp_offset_Lco, kps_mask, nose_ind) in enumerate(loader):
        # inputs : Batch, Channel, Height, Width
        inputs = inputs.cuda()
        seq_length = inputs.shape[1] // 3
        #loss_MFEM = 0
        #for j in range(seq_length-1):
        #   res_dict = MFEM(inputs[:,j*3:j*3+6,:,:], with_bk=True)
        #    flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        #    flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
        #    cfg = EasyDict(json.load(open('/home/mry/PycharmProjects/SALD/configs/MFDS.json')))
        #    cfg = cfg.loss
        #    loss_func = unFlowLoss(cfg)
        #    loss_interframe, l_ph, l_sm, flow_mean = loss_func(flows, inputs[:,i*3:i*3+6,:,:])
        #    loss_MFEM += loss_interframe
        #optimizer_MFEM.zero_grad()
        #scaled_loss = 1024. * loss_MFEM
        #scaled_loss.backward()
        #for param in [p for p in MFEM.parameters() if p.requires_grad]:
        #    param.grad.data.mul_(1. / 1024)

        #optimizer_MFEM.step()

        inputs = inputs.view(-1, 3, inputs.shape[2], inputs.shape[3])
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

        # batch_heatmaps is a list for stage-predictions, each element should be [Batch, C, H, W]
        # batch_heatmaps, batch_locs, batch_scos = net(inputs)

        batch_heatmaps, batch_locs, batch_scos, CLOR_dict = net(inputs)

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


        '''
        #2.9 adding vis of the heatmap of 3 stages
    
        for idx in range(batch_size):
          debugger = Debugger(
            dataset='300w', ipynb=False, theme='white')
          img = inputs[idx].detach().cpu().numpy().transpose(1, 2, 0)#[:3,:,:]
          mean = np.array([0.40789654, 0.44719302, 0.47026115],
                          dtype=np.float32).reshape(1, 1, 3)
          std = np.array([0.28863828, 0.27408164, 0.27809835],
                         dtype=np.float32).reshape(1, 1, 3)
          img = np.clip(((img *std + mean) * 255.), 0, 255).astype(np.uint8)
          pred = debugger.gen_colormap_hp(batch_heatmaps[0][idx][:-1, :, :].detach().cpu().numpy())
          debugger.add_blend_img(img, pred, 'stage1_pred_hmhp')
          #pred = debugger.gen_colormap_hp(batch_heatmaps[1][idx][:-1, :, :].detach().cpu().numpy())
          #debugger.add_blend_img(img, pred, 'stage2_pred_hmhp')
          #pred = debugger.gen_colormap_hp(batch_heatmaps[2][idx][:-1, :, :].detach().cpu().numpy())
          #debugger.add_blend_img(img, pred, 'stage3_pred_hmhp')
          gt = debugger.gen_colormap_hp(target[idx][:-1, :, :].detach().cpu().numpy())
          debugger.add_blend_img(img, gt, 'gt_hmhp')
    
          #if opt.debug == 4:
          #  debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
          #else:
          debugger.show_all_imgs(pause=pause)
          '''

        forward_time.update(time.time() - end)

        # FCMIL_loss = compute_FCMIL_loss(criterion, mask, FC_MIL)
        if np.random.random() < 1:
            fake_mask = mask
        else:
            pass
        # fake_mask = FC_MIL.detach().clone().unsqueeze(2).unsqueeze(3)
        loss = 0
        if args.usefocalloss:

            each_stage_loss_value = []
            for i in range(opt_config.stages):
                heatmap = batch_heatmaps[i]
                loss_stage_i = criterion(heatmap, target)
                each_stage_loss_value.append(loss_stage_i.item())
                loss = loss + loss_stage_i
        else:
            loss, each_stage_loss_value = compute_stage_loss(criterion, target, batch_heatmaps, fake_mask)
        # if pca_flag:
        #  loss_pca, each_stage_loss_value_pca = compute_stage_loss_pca(criterion, heatmap_pca_weight, pca_weight)
        # loss = loss + loss_pca*(1e-7)

        if opt_config.lossnorm:
            if args.usepca:
                loss, each_stage_loss_value, each_stage_loss_value_pca = loss / annotated_num / 2, [
                    x / annotated_num / 2 for x in each_stage_loss_value], [x / annotated_num / 2 * (1e-7) for x in
                                                                            each_stage_loss_value_pca]
            else:
                loss, each_stage_loss_value = loss / annotated_num / 2, [x / annotated_num / 2 for x in
                                                                         each_stage_loss_value]
        # measure accuracy and record loss


        #Adding SIC , nosecenterHM, Loc regress
        loss = loss + 0.01*loss_SIC + nose_center_hm_loss + 0.1*Loc_loss
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
                   + ' Vis-PTS : {:2d} ({:.1f})'.format(int(visible_points.val), visible_points.avg)
                   + 'Loss_SIC:{loss:7.4f} ,Loss_noseHM {loss_noseHM:7.4f}, Loss_Loc {loss_loc:7.4f}'.format(loss=loss_SIC, loss_noseHM=nose_center_hm_loss, loss_loc=Loc_loss)
                   )
                   #+ 'Loss_motion {loss_motion:7.4f}'.format(loss_motion=loss_MFEM))

    #nme, _, _ = eval_meta.compute_mse(logger)
    return losses.avg#, nme

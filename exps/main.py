# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import numbers, numpy as np

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_basic_args
from procedure import prepare_seed, save_checkpoint, basic_train as train, basic_eval_all as eval_all
from datasets import GeneralDataset as Dataset
from xvision import transforms
from log_utils import Logger, AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_model
from optimizer import obtain_optimizer
from lib.criterion import FocalLoss
from utils.load_model import load_model
from models.MFDS_model import MFDS
from easydict import EasyDict
import json

def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    prepare_seed(args.rand_seed)

    logstr = 'seed-{:}-time-{:}'.format(args.rand_seed, time_for_file())
    logger = Logger(args.save_path, logstr)
    logger.log('Main Function with logger : {:}'.format(logger))
    logger.log('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  version : {}".format(sys.version.replace('\n', ' ')))
    logger.log("Pillow  version : {}".format(PIL.__version__))
    logger.log("PyTorch version : {}".format(torch.__version__))
    logger.log("cuDNN   version : {}".format(torch.backends.cudnn.version()))

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    assert args.arg_flip == False, 'The flip is : {}, rotate is {}'.format(args.arg_flip, args.rotate_max)
    train_transform = [transforms.PreCrop(args.pre_crop_expand)]
    train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
    train_transform += [transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max)]
    # if args.arg_flip:
    #  train_transform += [transforms.AugHorizontalFlip()]
    if args.rotate_max:
        train_transform += [transforms.AugRotate(args.rotate_max)]
    train_transform += [transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill)]
    temp = train_transform
    train_transform += [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transform)

    eval_transform = transforms.Compose(
        [transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH((args.crop_width, args.crop_height)),
         transforms.ToTensor(), normalize])
    assert (args.scale_min + args.scale_max) / 2 == args.scale_eval, 'The scale is not ok : {},{} vs {}'.format(
        args.scale_min, args.scale_max, args.scale_eval)

    # Model Configure Load
    model_config = load_configure(args.model_config, logger)
    args.sigma = args.sigma * args.scale_eval
    logger.log('Real Sigma : {:}'.format(args.sigma))

    # Training Dataset

    train_data = Dataset(train_transform, args.sigma, model_config.downsample, args.heatmap_type,
                             args.data_indicator, phase='train')
    import pdb
    pdb.set_trace()
    train_data.load_list(args.train_lists, args.num_pts, True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    #a = train_loader.dataset[4]

    # Evaluation Dataloader
    eval_loaders = []
    if args.eval_vlists is not None:
        for eval_vlist in args.eval_vlists:
            eval_vdata = Dataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type,
                                     args.data_indicator, phase='eval')
            eval_vdata.load_list(eval_vlist, args.num_pts, True)
            eval_vloader = torch.utils.data.DataLoader(eval_vdata, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.workers, pin_memory=True)
            eval_loaders.append((eval_vloader, True))

    if args.eval_ilists is not None:
        for eval_ilist in args.eval_ilists:
            eval_idata = Dataset(eval_transform, args.sigma, model_config.downsample, args.heatmap_type,
                                     args.data_indicator, phase='eval')
            eval_idata.load_list(eval_ilist, args.num_pts, True)
            eval_iloader = torch.utils.data.DataLoader(eval_idata, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.workers, pin_memory=True)
            # a = eval_iloader.dataset[4]
            eval_loaders.append((eval_iloader, False))

    # Define network
    logger.log('configure : {:}'.format(model_config))

    net = obtain_model(model_config, args.num_pts + 1)
    # assert model_config.downsample == net.downsample, 'downsample is not correct : {} vs {}'.format(model_config.downsample, net.downsample)
    logger.log("=> network :\n {}".format(net))

    logger.log('Training-data : {:}'.format(train_data))

    # Define Montion fiels estimation module
    cfg_MFEM = EasyDict(json.load(open('/home/mry/PycharmProjects/SALD/configs/MFDS.json')))
    MFEM = MFDS(cfg_MFEM)


    for i, eval_loader in enumerate(eval_loaders):
        eval_loader, is_video = eval_loader
        logger.log('The [{:2d}/{:2d}]-th testing-data [{:}] = {:}'.format(i, len(eval_loaders),
                                                                          'video' if is_video else 'image',
                                                                          eval_loader.dataset))

    logger.log('arguments : {:}'.format(args))

    opt_config = load_configure(args.opt_config, logger)

    if hasattr(net, 'specify_parameter'):
        net_param_dict = net.specify_parameter(opt_config.LR, opt_config.Decay)
    else:
        net_param_dict = net.parameters()

    #parameter of MFEM
    MFEM_param_dict = MFEM.parameters()
    optimizer_MFEM, scheduler_MFEM, criterion_MFEM = obtain_optimizer(MFEM_param_dict, opt_config, logger)


    optimizer, scheduler, criterion = obtain_optimizer(net_param_dict, opt_config, logger)
    if args.usefocalloss:
        criterion = FocalLoss()
    logger.log('criterion : {:}'.format(criterion))
    # net, criterion = net.cuda(), criterion.cuda()
    # net = torch.nn.DataParallel(net)
    resume = False
    last_info = logger.last_info()
    if last_info.exists():
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))

        last_info = torch.load(last_info)
        start_epoch = last_info['epoch'] + 1
        checkpoint = torch.load(last_info['last_checkpoint'])
        assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info,
                                                                                                     checkpoint[
                                                                                                         'epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done".format(logger.last_info(), checkpoint['epoch']))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch = 0

    if args.eval_once:
        test_model_once_time = torch.load(args.load_test_model)

        state_dict_ = test_model_once_time['state_dict']
        state_dict = {}
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = net.state_dict()

        # check loaded parameters and created model parameters
        msg = 'If you see this, your model does not fully load the ' + \
              'pre-trained weight. Please make sure ' + \
              'you have correctly specified --arch xxx ' + \
              'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
        net.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(test_model_once_time['optimizer'])
        scheduler.load_state_dict(test_model_once_time['scheduler'])
        cur_epoch = test_model_once_time['epoch']

        logger.log("=> load-ok test modeol '{:}' (epoch {:}) done".format(args.load_test_model, cur_epoch))

    net, criterion = net.cuda(), criterion.cuda()
    net = torch.nn.DataParallel(net)
    MFEM = MFEM.cuda()
    MFEM = torch.nn.DataParallel(MFEM)

    if args.eval_once:
        logger.log("=> only evaluate the model once")
        eval_results = eval_all(args, eval_loaders, net, criterion, 'eval-once', logger, opt_config)
        logger.log('NME Results : {:}'.format(eval_results))
        logger.close()
        return

    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, opt_config.epochs):

        scheduler.step()
        need_time = convert_secs2time(epoch_time.avg * (opt_config.epochs - epoch), True)
        epoch_str = 'epoch-{:03d}-{:03d}'.format(epoch, opt_config.epochs)
        LRs = scheduler.get_lr()
        logger.log('\n==>>{:s} [{:s}], [{:s}], LR : [{:.5f} ~ {:.5f}], Config : {:}'.format(time_string(), epoch_str,
                                                                                            need_time, min(LRs),
                                                                                            max(LRs), opt_config))

        # train for one epoch
        train_loss, train_nme = train(args, train_loader, net, MFEM, criterion, optimizer, optimizer_MFEM, epoch_str, logger, opt_config)
        # log the results
        logger.log(
            '==>>{:s} Train [{:}] Average Loss = {:.6f}, NME = {:.2f}'.format(time_string(), epoch_str, train_loss,
                                                                              train_nme * 100))

        if epoch % 10 == 0:
            # remember best prec@1 and save checkpoint
            save_path = save_checkpoint({
                'epoch': epoch,
                'args': deepcopy(args),
                'arch': model_config.arch,
                'state_dict': net.state_dict(),
                'detector': net.state_dict(),
                'MFEM' : MFEM.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'MFEM_optimizer':optimizer_MFEM.state_dict()
            }, logger.path('model') / '{:}-{:}.pth'.format(model_config.arch, epoch_str), logger)

            last_info = save_checkpoint({
                'epoch': epoch,
                'last_checkpoint': save_path,
            }, logger.last_info(), logger)

        eval_results = eval_all(args, eval_loaders, net, criterion, epoch_str, logger, opt_config)
        logger.log('NME Results : {:}'.format(eval_results))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.close()


if __name__ == '__main__':
    args = obtain_basic_args()
    main(args)

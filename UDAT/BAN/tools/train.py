# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from siamban.utils.lr_scheduler import build_lr_scheduler
from siamban.utils.log_helper import init_log, print_speed, add_file_handler
from siamban.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from siamban.utils.model_load import load_pretrain, restore_from
from siamban.utils.average_meter import AverageMeter
from siamban.utils.misc import describe, commit
from siamban.models.model_builder import ModelBuilder
from siamban.datasets.dataset import BANDataset
from siamban.core.config import cfg
from siamban.models.trans_discriminator import TransformerDiscriminator
import torch.nn.functional as F

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='experiments/siamban_r50_l234_otb/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='0'
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(domain):
    logger.info("build train dataset")
    # train_dataset
    if cfg.BAN.BAN:
        train_dataset = BANDataset(domain)
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.ALIGN.ALIGN:
        trainable_params += [{'params': model.align.parameters(),
                              'lr': cfg.TRAIN.BASE_LR_d}]

    trainable_params += [{'params': model.head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)
def weightedMSE(D_out, label):
    # D_label = torch.FloatTensor(D_out.data.size()).fill_(1).cuda() * label.unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
    # D_label = torch.FloatTensor(D_out.data.size()).fill_(label).cuda()
    return torch.mean((D_out - label.cuda()).abs() ** 2)
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.TRAIN.BASE_LR_d, i_iter, args.TRAIN.EPOCH, 0.8)
    for k in optimizer.param_groups:
        k['lr'] = lr
    # optimizer.param_groups[0]['lr'] = lr
    # if len(optimizer.param_groups) > 1:
    #     optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def train(source_loader, target_loader, model, optimizer, lr_scheduler, tb_writer, Disc, optimizer_D):
    cur_lr = lr_scheduler.get_cur_lr()
    cur_lr_d = adjust_learning_rate_D(cfg, optimizer_D, cfg.TRAIN.START_EPOCH)

    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(target_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    source_label = 0
    target_label = 1
    target_data = enumerate(target_loader)
    source_data = enumerate(source_loader)

    for idx in range(cfg.TRAIN.EPOCH*num_per_epoch):#    for idx, data in enumerate(train_loader):
        data = target_data.__next__()[1]
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))
                torch.save( # save discriminator
                        {'epoch': epoch,
                         'state_dict': Disc.module.state_dict(),
                         'optimizer': optimizer_D.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/d_checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            cur_lr_d = adjust_learning_rate_D(cfg, optimizer_D, epoch)
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx # + start_epoch * num_per_epoch
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        # train G
        for param in Disc.parameters():
            param.requires_grad = False
        outputs, zf, xf = model(data)
        loss_pse = outputs['total_loss']

        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_t = [interp(_zf) for _zf in zf]
        xf_up_t = [interp(_xf) for _xf in xf]
        D_out_z = torch.stack([Disc(F.softmax(_zf_up_t, dim=1)) for _zf_up_t in zf_up_t]).sum(0) / 3.
        D_out_x = torch.stack([Disc(F.softmax(_xf_up_t, dim=1)) for _xf_up_t in xf_up_t]).sum(0) / 3.
        D_source_label = torch.FloatTensor(D_out_z.data.size()).fill_(source_label)
        loss_adv = 0.1 * (weightedMSE(D_out_z, D_source_label) +  weightedMSE(D_out_x, D_source_label)) #  / cfg.TRAIN.BATCH_SIZE

        if is_valid_number(loss_adv.data.item()):
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            loss_adv.backward()

        data = source_data.__next__()[1]
        outputs, zf, xf = model(data)
        interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        zf_up_s = [interp(_zf) for _zf in zf]
        xf_up_s = [interp(_xf) for _xf in xf]
        loss_gt = outputs['total_loss'].mean()

        if is_valid_number(loss_gt.data.item()):
            loss_gt.backward()

        loss_train_adv = 0
        # train D
        for param in Disc.parameters():
            param.requires_grad = True 
        zf_up_t = [_zf_up_t.detach() for _zf_up_t in zf_up_t]
        xf_up_t = [_xf_up_t.detach() for _xf_up_t in xf_up_t]
        D_out_1 = torch.stack([Disc(F.softmax(_zf_up_t, dim=1)) for _zf_up_t in zf_up_t]).sum(0) / 3.
        D_out_2 = torch.stack([Disc(F.softmax(_xf_up_t, dim=1)) for _xf_up_t in xf_up_t]).sum(0) / 3.
        D_target_label = torch.FloatTensor(D_out_z.data.size()).fill_(target_label)
        loss_d = 0.1*weightedMSE(D_out_1, D_target_label) + 0.1*weightedMSE(D_out_2, D_target_label)

        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 
        
        loss_train_adv += loss_d.item()

        zf_up_s = [_zf_up_s.detach() for _zf_up_s in zf_up_s]
        xf_up_s = [_xf_up_s.detach() for _xf_up_s in xf_up_s]
        D_out_1 = torch.stack([Disc(F.softmax(_zf_up_s, dim=1)) for _zf_up_s in zf_up_s]).sum(0) / 3.
        D_out_2 = torch.stack([Disc(F.softmax(_xf_up_s, dim=1)) for _xf_up_s in xf_up_s]).sum(0) / 3.
        D_source_label = torch.FloatTensor(D_out_z.data.size()).fill_(source_label)
        loss_d = 0.1*weightedMSE(D_out_1, D_source_label) + 0.1*weightedMSE(D_out_2, D_source_label)

        if is_valid_number(loss_d.data.item()):
            loss_d.backward() 


        loss_train_adv += loss_d.item()
        if is_valid_number(loss_gt.data.item()):
            reduce_gradients(model)
            reduce_gradients(Disc)
            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            clip_grad_norm_(Disc.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
            optimizer_D.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        batch_info['loss_fool'] = average_reduce(loss_adv)
        batch_info['loss_train_adv'] = average_reduce(loss_train_adv)

        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f} lr_d: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr, cur_lr_d)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()
    dist_model = nn.DataParallel(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    target_loader = build_data_loader('target')
    source_loader = build_data_loader('source')

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)
    model_Disc = TransformerDiscriminator(channels=256) # 特征的通道数
    model_Disc.train()
    model_Disc.cuda()
    dist_Disc = nn.DataParallel(model_Disc)
    optimizer_D = torch.optim.Adam(model_Disc.parameters(), lr=cfg.TRAIN.BASE_LR_d, betas=(0.9, 0.99)) # TODO 写到cfg里
    # lr_scheduler_d = build_lr_scheduler(optimizer_D, epochs=cfg.TRAIN.EPOCH)
    # lr_scheduler_d.step(cfg.TRAIN.START_EPOCH)
    optimizer_D.zero_grad()

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    if cfg.TRAIN.RESUME_D:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME_D))
        assert os.path.isfile(cfg.TRAIN.RESUME_D), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME_D)
        model_Disc, optimizer_D, cfg.TRAIN.START_EPOCH = \
            restore_from(model_Disc, optimizer_D, cfg.TRAIN.RESUME_D)
    
    dist_model = DistModule(model)
    dist_Disc = nn.DataParallel(model_Disc)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(source_loader, target_loader, dist_model, optimizer, lr_scheduler, tb_writer, dist_Disc, optimizer_D)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

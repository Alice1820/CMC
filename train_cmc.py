"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import argparse
import socket

import numpy as np

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from datasets.dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.i3d import MyI3DCMC
from models.tsm import MyTSMCMC, TSN, ConsensusModule
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from datasets.dataset import ImageFolderInstance
from datasets.ntu import NTU

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=1.0, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='tsm', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--base_model', type=str, default='resnet18')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=511)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # video
    parser.add_argument('--num_segments', type=int, default=8, help='')
    parser.add_argument('--num_class', type=int, default=60, help='')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--data_folder', type=str, default='/data0/xifan/NTU_RGBD_60/', help='path to data')
    parser.add_argument('--model_path', type=str, default='checkpoints', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='logs', help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # CMC phase
    parser.add_argument('--phase', type=str, default='cmc', choices=['cmc', 'cls'])

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'cmc5_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(split='train', args=None):
    """get the train loader"""
    # data_folder = os.path.join(args.data_folder, 'train')

    # if args.view == 'Lab':
    #     mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
    #     std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    #     color_transfer = RGB2Lab()
    # elif args.view == 'YCbCr':
    #     mean = [116.151, 121.080, 132.342]
    #     std = [109.500, 111.855, 111.964]
    #     color_transfer = RGB2YCbCr()
    # else:
    #     raise NotImplemented('view not implemented {}'.format(args.view))
    # normalize = transforms.Normalize(mean=mean, std=std)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
    #     transforms.RandomHorizontalFlip(),ssss
    #     color_transfer,
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_dataset = NTU(root_dir=args.data_folder, stage=split, vid_len=(args.num_segments, args.num_segments))
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        # train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    model_l  = TSN()
    model_ab  = TSN()

    # encoder_l = nn.Linear(2048, args.feat_dim) # [2048, 128]
    encoder_l = nn.Linear(512, args.feat_dim) # [2048, 128]
    # encoder_ab = nn.Linear(2048, args.feat_dim)
    encoder_ab = nn.Linear(512, args.feat_dim)

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model_l = model_l.cuda()
        model_ab = model_ab.cuda()
        encoder_l = encoder_l.cuda()
        encoder_ab = encoder_ab.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True
        model_l = nn.DataParallel(model_l)
        model_ab = nn.DataParallel(model_ab)
        encoder_l = nn.DataParallel(encoder_l)
        encoder_ab = nn.DataParallel(encoder_ab)
        # contrast = nn.DataParallel(contrast)

    return model_l, model_ab, encoder_l, encoder_ab, contrast, criterion_ab, criterion_l


def set_optimizer(args, model, encoder):
    # return optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()),
                                lr=args.learning_rate,
                                betas=[args.beta1, args.beta2])
    return optimizer


def train(epoch, train_loader, model_l, model_ab, encoder_l, encoder_ab, contrast, criterion_l, criterion_ab, optimizer_l, optimizer_ab, args):
    """
    one epoch training
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()
    cls_l_loss_meter = AverageMeter()
    cls_ab_loss_meter = AverageMeter()
    acc_l_meter = AverageMeter()
    acc_ab_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # l, ab = inputs['rgb'], inputs['rgb']
        l, ab = inputs['rgb'], inputs['dep']
        # print (l.size())
        # print (ab.size())
        label = inputs['label']
        bsz = l.size(0)
        l = l.float()
        ab = ab.float()
        if torch.cuda.is_available():
            index = index.cuda()
            l = l.cuda()
            ab = ab.cuda()
            label = label.cuda()

        # ===================forward feature=====================
        model_l.train()
        model_ab.train()
        encoder_l.train()
        encoder_ab.train()
        # model.eval() # verify after loaded model, cmc loss start with 4.xx
        contrast.train()
        # contrast.eval()
        feat_l, cls_l = model_l(l)
        feat_ab, cls_ab = model_ab(ab) # [bs, 8, 2048]
        # print (feat_l.size()) # [bs*8, 2048]
        # print (cls_l.size()) # [bs*8]
        # ===================consensus feature=====================
        consensus = ConsensusModule('avg')
        # ===================forward encoder=====================
        enc_l = encoder_l(feat_l)
        enc_ab = encoder_ab(feat_ab)
        enc_l = enc_l.view((-1, args.num_segments) + enc_l.size()[1:])
        enc_ab = enc_ab.view((-1, args.num_segments) + enc_ab.size()[1:])
        # print (enc_l.size())
        enc_l = consensus(enc_l).squeeze()
        enc_ab = consensus(enc_ab).squeeze()
        # print (enc_l.size())
        out_l, out_ab = contrast(enc_l, enc_ab, index)
        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss

        # check if loss is nan or inf
        if torch.isfinite(loss):
            optimizer_l.zero_grad()
            optimizer_ab.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e2, norm_type=1)
            # print (model.encoder.module.l_to_ab.classifier.weight.grad)
            # print (model.encoder.module.l_to_ab.classifier.bias.grad)
            # print (model.encoder.module.l_to_ab.new_fc.weight.grad)
            # print (model.encoder.module.l_to_ab.new_fc.bias.grad)
            # print (model.encoder.module.l_to_ab.base_model.layer3[0].conv1.weight.grad) # learning_rate?
            optimizer_l.step()
            optimizer_ab.step()
        else:
            print ("Loss is not finite.")
            continue
        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)
        # cls_l_loss_meter.update(cls_l_loss.item(), bsz)
        # cls_ab_loss_meter.update(cls_ab_loss.item(), bsz)
        # acc_l_meter.update(acc_l, bsz)
        # acc_ab_meter.update(acc_ab, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            # print(out_l.shape)
            sys.stdout.flush()

    # return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg, cls_l_loss_meter.avg, cls_ab_loss_meter.avg 
    return l_loss_meter.avg, ab_loss_meter.avg

def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(split='train', args=args)
    eval_loader, _ = get_train_loader(split='dev', args=args)
    test_loader, _ = get_train_loader(split='test', args=args)

    # set the model
    model_l, model_ab, encoder_l, encoder_ab, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # cls_l = nn.Linear(2048, args.num_classes)
    # cls_ab = nn.Linear(2048, args.num_classes)

    # if torch.cuda.is_available():
    #     cls_l = cls_l.cuda()
    #     cls_ab = cls_ab.cuda()
    #     criterion_cls = criterion_cls.cuda()

    # set the optimizer
    optimizer_l = set_optimizer(args, model_l, encoder_l)
    optimizer_ab = set_optimizer(args, model_ab, encoder_ab)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model_l.load_state_dict(checkpoint['model_l'])
            encoder_l.load_state_dict(checkpoint['encoder_l'])
            model_ab.load_state_dict(checkpoint['model_ab'])
            encoder_ab.load_state_dict(checkpoint['encoder_ab'])
            optimizer_l.load_state_dict(checkpoint['optimizer_l'])
            optimizer_ab.load_state_dict(checkpoint['optimizer_ab'])
            contrast.load_state_dict(checkpoint['contrast'])
            contrast.K = args.nce_k
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    
    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer_l)
        adjust_learning_rate(epoch, args, optimizer_ab)
        print("==> training...")

        time1 = time.time()
        cls_l_loss, cls_ab_loss = train(epoch, train_loader, model_l, model_ab, encoder_l, encoder_ab, contrast, 
                                        criterion_l, criterion_ab, optimizer_l, optimizer_ab, args)
        time2 = time.time()
        print('epoch {} train, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('l_loss', l_loss, epoch)
        # logger.log_value('l_prob', l_prob, epoch)
        # logger.log_value('ab_loss', ab_loss, epoch)
        # logger.log_value('ab_prob', ab_prob, epoch)
        logger.log_value('l_loss', cls_l_loss, epoch)
        logger.log_value('ab_loss', cls_ab_loss, epoch)

        # print("==> evaluating...")
        
        # time1 = time.time()
        # cls_l_loss, cls_ab_loss = eval(epoch, eval_loader, model_l, model_ab, encoder_l, encoder_ab, contrast, 
        #                                 criterion_l, criterion_ab, optimizer_l, optimizer_ab, args)
        # time2 = time.time()
        # print('epoch {} test, total time {:.2f}'.format(epoch, time2 - time1))

        # # print("==> testing...")
        
        # # time1 = time.time()
        # # cls_l_loss, cls_ab_loss = eval(epoch, test_loader, model, contrast, criterion_l, criterion_ab, criterion_cls,
        # #                                          optimizer, args)
        # # time2 = time.time()
        # # print('epoch {} test, total time {:.2f}'.format(epoch, time2 - time1))
        # # exit()
        # # tensorboard logger
        # # logger.log_value('l_loss', l_loss, epoch)
        # # logger.log_value('l_prob', l_prob, epoch)
        # # logger.log_value('ab_loss', ab_loss, epoch)
        # # logger.log_value('ab_prob', ab_prob, epoch)
        # logger.log_value('test_cls_l_loss', cls_l_loss, epoch)
        # logger.log_value('test_cls_ab_loss', cls_ab_loss, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model_l': model_l.state_dict(),
                'encoder_l': encoder_l.state_dict(),
                'model_ab': model_ab.state_dict(),
                'encoder_ab': encoder_ab.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_l': optimizer_l.state_dict(),
                'optimizer_ab': optimizer_ab.state_dict(),
                'epoch': epoch,
                # 'cls_l': cls_l.state_dict(),
                # 'cls_ab': cls_ab.state_dict(),
                # 'cls_l_optimizer': cls_l_optimizer.state_dict(),
                # 'cls_ab_optimizer': cls_ab_optimizer.state_dict(),
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, '{phase}_epoch_{epoch}.pth'.format(phase=args.phase, epoch=epoch))    
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

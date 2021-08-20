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
from models.tsm import MyTSMCMC
from models.discriminator import Discriminator
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
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
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
    parser.add_argument('--base_model', type=str, default='resnet50')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=2048)
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
    opt.model_name = 'dis_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
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
    if args.model == 'i3d':
        model = MyI3DCMC()
    elif args.model == 'tsm':
        model = MyTSMCMC(args=args)
    elif args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, discriminator, criterion, optimizer_model, optimizer, opt):
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
        # label = inputs['label']
        subject_id = inputs['subject_id']
        # print (subject_id)
        bsz = l.size(0)
        l = l.float()
        ab = ab.float()
        if torch.cuda.is_available():
            index = index.cuda()
            l = l.cuda()
            ab = ab.cuda()
            # label = label.cuda()
            subject_id = subject_id.cuda()

        # ===================forward feature=====================
        # model.eval()
        model.train()
        discriminator.train()
        logits_l, cmc_l, feat_l, logits_ab, cmc_ab, feat_ab = model(l, ab) # [bs, 128]
        # print (feat_l.size()) # [2, 2048]
        # pred_l = discriminator(feat_l.detach())
        pred_l = discriminator(feat_l)

        # ===================forward dis=====================
        # print (pred_l.size(), subject_id.size())
        loss = criterion(pred_l, subject_id)

        # ===================backward dis=====================
        optimizer.zero_grad()
        optimizer_model.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_model.step()

        # ===================meters=====================
        pred_cls = torch.squeeze(pred_l.max(1)[1])
        acc_l = (pred_cls == subject_id).float().mean()

        losses.update(loss.item(), bsz)
        acc_l_meter.update(acc_l, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_acc {laccs.val:.3f} ({laccs.avg:.3f})\t'.format(
                #   'ab_acc {abaccs.val:.3f} ({abaccs.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, laccs=acc_l_meter))
            # print(out_l.shape)
            sys.stdout.flush()

    # return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg, cls_l_loss_meter.avg, cls_ab_loss_meter.avg 
    return losses.avg, acc_l_meter.avg


def eval(epoch, train_loader, model, contrast, criterion_l, criterion_ab, criterion_cls, optimizer, opt):
    """
    one epoch evaluation
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    # l_loss_meter = AverageMeter()
    # ab_loss_meter = AverageMeter()
    # l_prob_meter = AverageMeter()
    # ab_prob_meter = AverageMeter()
    cls_l_loss_meter = AverageMeter()
    cls_ab_loss_meter = AverageMeter()
    acc_l_meter = AverageMeter()
    acc_ab_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # l, ab = inputs['rgb'], inputs['rgb']
        l, ab = inputs['rgb'], inputs['dep']
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
        model.eval()
        contrast.eval()
        # hidden_l, feat_l, hidden_ab, feat_ab = model(l, ab) # [bs, 128]
        with torch.no_grad():
            logits_l, feat_l, logits_ab, feat_ab = model(l, ab) # [bs, 60], [bs, 128]
        # print (feat_l.size())
        # print (feat_ab.size())
        # out_l, out_ab = contrast(feat_l, feat_ab, index)
        # l_loss = criterion_l(out_l)
        # ab_loss = criterion_ab(out_ab)
        # l_prob = out_l[:, 0].mean()
        # ab_prob = out_ab[:, 0].mean()

        # loss = l_loss + ab_loss

        # ===================backward feature=====================
        # optimizer.zero_grad()
        # if opt.amp:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        # optimizer.step()

        # ===================forward cls=====================
        # hidden_l, hidden_ab = hidden_l.detach(), hidden_ab.detach()
        # logits_l = F.softmax(cls_l(hidden_l), dim=1)
        # logits_ab = F.softmax(cls_ab(hidden_ab), dim=1)
        # logits_l = cls_l(hidden_l)
        # logits_ab = cls_ab(hidden_ab)
        cls_l_loss = criterion_cls(logits_l, label)
        cls_ab_loss = criterion_cls(logits_ab, label)
        cls_loss = cls_l_loss + cls_ab_loss

        # ===================backward cls=====================
        # cls_l_optimizer.zero_grad()   
        # cls_l_loss.backward()
        # cls_l_optimizer.step()

        # cls_ab_optimizer.zero_grad()
        # cls_ab_loss.backward()
        # cls_ab_optimizer.step()

        # ===================meters=====================
        _, pred_l = torch.max(logits_l, dim=1) # top1 accuracy
        _, pred_ab = torch.max(logits_ab, dim=1)
        acc_l = np.mean((pred_l==label).cpu().numpy())*100
        acc_ab = np.mean((pred_ab==label).cpu().numpy())*100

        # losses.update(loss.item(), bsz)
        # l_loss_meter.update(l_loss.item(), bsz)
        # l_prob_meter.update(l_prob.item(), bsz)
        # ab_loss_meter.update(ab_loss.item(), bsz)
        # ab_prob_meter.update(ab_prob.item(), bsz)
        cls_l_loss_meter.update(cls_l_loss.item(), bsz)
        cls_ab_loss_meter.update(cls_ab_loss.item(), bsz)
        acc_l_meter.update(acc_l, bsz)
        acc_ab_meter.update(acc_ab, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Eval: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #   'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                #   'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                #   'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})\t'
                  'cls_l_loss {clsllosses.val:.3f} ({clsllosses.avg:.3f})\t'
                  'cls_ab_loss {clsablosses.val:.3f} ({clsablosses.avg:.3f})\t'
                  'l_acc {laccs.val:.3f} ({laccs.avg:.3f})\t'
                  'ab_acc {abaccs.val:.3f} ({abaccs.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, clsllosses=cls_l_loss_meter, clsablosses=cls_ab_loss_meter,
                   laccs=acc_l_meter, abaccs=acc_ab_meter))
            # print(out_l.shape)
            sys.stdout.flush()

    # return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg, cls_l_loss_meter.avg, cls_ab_loss_meter.avg 
    return cls_l_loss_meter.avg, cls_ab_loss_meter.avg 


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(split='train', args=args)
    eval_loader, _ = get_train_loader(split='dev', args=args)
    test_loader, _ = get_train_loader(split='test', args=args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # cls_l = nn.Linear(2048, args.num_classes)
    # cls_ab = nn.Linear(2048, args.num_classes)
    discriminator = Discriminator()
    discriminator = nn.DataParallel(discriminator)
    discriminator = discriminator.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(discriminator.parameters(),
                            lr=1e-3,
                            betas=(0.5, 0.9))
    # if torch.cuda.is_available():
    #     cls_l = cls_l.cuda()
    #     cls_ab = cls_ab.cuda()
    #     criterion_cls = criterion_cls.cuda()
    optimizer_model = set_optimizer(args, model)
    # set the optimizer
    # cls_l_optimizer =  torch.optim.SGD(cls_l.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # cls_ab_optimizer =  torch.optim.SGD(cls_ab.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # set mixed precision
    # if args.amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
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

        # adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, acc_l = train(epoch, train_loader, model, discriminator, criterion, optimizer_model, optimizer, args)
        time2 = time.time()
        print('epoch {} train, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('l_loss', l_loss, epoch)
        # logger.log_value('l_prob', l_prob, epoch)
        # logger.log_value('ab_loss', ab_loss, epoch)
        # logger.log_value('ab_prob', ab_prob, epoch)
        # logger.log_value('train_cls_l_loss', cls_l_loss, epoch)
        # logger.log_value('train_cls_ab_loss', cls_ab_loss, epoch)

        # print("==> evaluating...")
        
        # time1 = time.time()
        # cls_l_loss, cls_ab_loss = eval(epoch, eval_loader, model, contrast, criterion_l, criterion_ab, criterion_cls,
        #                                          optimizer, args)
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
                'model': model.state_dict(),
                # 'contrast': contrast.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'cls_l': cls_l.state_dict(),
                # 'cls_ab': cls_ab.state_dict(),
                # 'cls_l_optimizer': cls_l_optimizer.state_dict(),
                # 'cls_ab_optimizer': cls_ab_optimizer.state_dict(),
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'dist_epoch_{epoch}.pth'.format(epoch=epoch))    
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

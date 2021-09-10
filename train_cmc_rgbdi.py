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
from models.resnet import MyResNetsCMC, Normalize
from models.i3d import MyI3DCMC, I3D
from models.tsm import MyTSMCMC, TSN, ConsensusModule
from NCE.NCEAverage import NCEAverageXYZ
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from datasets.dataset import ImageFolderInstance
from datasets.ntu import get_dataloaders
from datasets.ntuv3 import get_dataloaders_v3

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
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--batch_size_glb', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
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
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3',
                                                                         'tsm', 'i3d'])
    parser.add_argument('--base_model', type=str, default='resnet18')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=511)
    parser.add_argument('--nce_t', type=float, default=7.0)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # video
    parser.add_argument('--num_segments', type=int, default=8, help='')
    parser.add_argument('--num_class', type=int, default=120, help='')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--data_folder', type=str, default='/data0/xifan/NTU_RGBD_60/', help='path to data')
    parser.add_argument('--model_path', type=str, default='checkpoints', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='logs', help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='RGBD', choices=['Lab', 'YCbCr', 'RGBD'])

    # mixed precision setting   
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # CMC phase
    parser.add_argument('--task', type=str, default=None)

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08
            l2norm = Normalize(2)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.accum = int(opt.batch_size_glb / opt.batch_size)

    opt.method = 'softmax' if opt.softmax else 'nce'
    if opt.task is None:
        raise Exception('Task name is None.')
    opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_view_RGBDI'.format(opt.task, opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size_glb)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

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
    if args.model == 'tsm':
        model_x  = TSN()
        model_y  = TSN()
        model_z  = TSN()
        encoder_x = nn.Linear(512, args.feat_dim) # [2048, 128]
        encoder_y = nn.Linear(512, args.feat_dim)
        encoder_z = nn.Linear(512, args.feat_dim)
    elif args.model == 'i3d':
        model_x = I3D()
        model_y = I3D()
        model_z = I3D()
        encoder_x = nn.Linear(2048, args.feat_dim) # [2048, 128]
        encoder_y = nn.Linear(2048, args.feat_dim)
        encoder_z = nn.Linear(2048, args.feat_dim)
    else:
        raise Exception("model not implemented.")

    contrast = NCEAverageXYZ(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_x = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_y = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_z = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model_x = model_x.cuda()
        model_y = model_y.cuda()
        model_z = model_z.cuda()
        encoder_x = encoder_x.cuda()
        encoder_y = encoder_y.cuda()
        encoder_z = encoder_z.cuda()
        contrast = contrast.cuda()
        criterion_x = criterion_x.cuda()
        criterion_y = criterion_y.cuda()
        criterion_z = criterion_z.cuda()
        cudnn.benchmark = True
        model_x = nn.DataParallel(model_x)
        model_y = nn.DataParallel(model_y)
        model_z = nn.DataParallel(model_z)
        encoder_x = nn.DataParallel(encoder_x)
        encoder_y = nn.DataParallel(encoder_y)
        encoder_z = nn.DataParallel(encoder_z)
        # contrast = nn.DataParallel(contrast)

    return model_x, model_y, model_z, encoder_x, encoder_y, encoder_z, contrast, criterion_x, criterion_y, criterion_z


def set_optimizer(args, model, encoder):
    # return optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()),
                                lr=args.learning_rate,
                                betas=[args.beta1, args.beta2])
    return optimizer


def train(epoch, train_loader, 
        model_x, model_y, model_z, 
        encoder_x, encoder_y, encoder_z, 
        contrast, 
        criterion_x, criterion_y, criterion_z,
        optimizer_x, optimizer_y, optimizer_z,
        args):
    """
    one epoch training
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    x_loss_meter = AverageMeter()
    y_loss_meter = AverageMeter()
    z_loss_meter = AverageMeter()
    x_prob_meter = AverageMeter()
    y_prob_meter = AverageMeter()
    z_prob_meter = AverageMeter()

    end = time.time()
    optimizer_x.zero_grad()
    optimizer_y.zero_grad()
    optimizer_z.zero_grad()
    for idx, (inputs, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # x, y = inputs['rgb'], inputs['rgb']
        x, y, z = inputs['rgb'], inputs['dep'], inputs['ir']
        # print (x.size())
        # print (y.size())
        label = inputs['label']
        bsz = x.size(0)
        x = x.float()
        y = y.float()
        z = z.float()

        # print (torch.max(y[0]), 'max')
        # print (torch.min(y[0]), 'min')
        # print (torch.mean(y[0]), 'mean')
        if torch.cuda.is_available():
            index = index.cuda()
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            label = label.cuda()

        # ===================forward feature=====================
        model_x.train()
        model_y.train()
        model_z.train()
        encoder_x.train()
        encoder_y.train()
        encoder_z.train()
        # model.eval() # verify after loaded model, cmc loss start with 4.xx
        contrast.train()
        # contrast.eval()
        feat_x, cls_x = model_x(x)
        feat_y, cls_y = model_y(y) # [bs, 8, 2048]
        feat_z, cls_z = model_z(z) # [bs, 8, 2048]
        # print (feat_x.size()) # [bs*8, 2048]
        # print (cls_x.size()) # [bs*8]
        if args.model == 'tsm':
            # ===================consensus feature=====================
            consensus = ConsensusModule('avg')
            l2norm = Normalize(2)
            # ===================forward encoder=====================
            enc_x = l2norm(encoder_x(feat_x))
            enc_y = l2norm(encoder_y(feat_y))
            enc_z = l2norm(encoder_z(feat_z))
            enc_x = enc_x.view((-1, args.num_segments) + enc_x.size()[1:])
            enc_y = enc_y.view((-1, args.num_segments) + enc_y.size()[1:])
            enc_z = enc_z.view((-1, args.num_segments) + enc_z.size()[1:])
            # print (enc_x.size())
            enc_x = consensus(enc_x).squeeze()
            enc_y = consensus(enc_y).squeeze()
            enc_z = consensus(enc_z).squeeze()
            # print (enc_x.size())
        elif args.model == 'i3d':
            enc_x = encoder_x(feat_x)
            enc_y = encoder_y(feat_y)
            enc_z = encoder_z(feat_z)
        out_x, out_y, out_z = contrast(enc_x, enc_y, enc_z, index)
        x_loss = criterion_x(out_x)
        y_loss = criterion_y(out_y)
        z_loss = criterion_z(out_z)
        x_prob = out_x[:, 0].mean()
        y_prob = out_y[:, 0].mean()
        z_prob = out_z[:, 0].mean()

        loss = x_loss + y_loss + z_loss

        # check if loss is nan or inf
        loss.backward()
        if idx % args.accum == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e2, norm_type=1)
            # print (model.encoder.module.l_to_ab.classifier.weight.grad)
            # print (model.encoder.module.l_to_ab.classifier.bias.grad)
            # print (model.encoder.module.l_to_ab.new_fc.weight.grad)
            # print (model.encoder.module.l_to_ab.new_fc.bias.grad)
            # print (model.encoder.module.l_to_ab.base_model.layer3[0].conv1.weight.grad) # learning_rate?
            optimizer_x.step()
            optimizer_y.step()
            optimizer_z.step()
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()
            optimizer_z.zero_grad()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        x_loss_meter.update(x_loss.item(), bsz)
        x_prob_meter.update(x_prob.item(), bsz)
        y_loss_meter.update(y_loss.item(), bsz)
        y_prob_meter.update(y_prob.item(), bsz)
        z_loss_meter.update(z_loss.item(), bsz)
        z_prob_meter.update(z_prob.item(), bsz)
        # cls_x_loss_meter.update(cls_l_loss.item(), bsz)
        # cls_y_loss_meter.update(cls_ab_loss.item(), bsz)
        # acc_l_meter.update(acc_l, bsz)
        # acc_ab_meter.update(acc_ab, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % (args.print_freq * args.accum) == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'x_p {xprobs.val:.3f} ({xprobs.avg:.3f})\t'
                  'y_p {xprobs.val:.3f} ({yprobs.avg:.3f})\t'
                  'z_p {yprobs.val:.3f} ({zprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, xprobs=x_prob_meter,
                   yprobs=y_prob_meter, zprobs=z_prob_meter))
            # print(out_x.shape)
            sys.stdout.flush()

    # return x_loss_meter.avg, x_prob_meter.avg, y_loss_meter.avg, y_prob_meter.avg, cls_x_loss_meter.avg, cls_y_loss_meter.avg 
    return losses.avg, x_loss_meter.avg, y_loss_meter.avg, z_loss_meter.avg

def main():

    best_loss = 1e3
    # parse the args
    args = parse_option()

    # set the loader
    # train_loader, n_data = get_train_loader(split='train', args=args)
    # eval_loader, _ = get_train_loader(split='dev', args=args)
    # test_loader, _ = get_train_loader(split='test', args=args)

    # set the loader
    train_loader, n_data = get_dataloaders_v3(args=args, stage='train')
    eval_loader, _ = get_dataloaders_v3(args=args, stage='dev')
    test_loader, _ = get_dataloaders_v3(args=args, stage='test')

    # set the model
    model_x, model_y, model_z, encoder_x, encoder_y, encoder_z, contrast, criterion_x, criterion_y, criterion_z = set_model(args, n_data)

    # cls_x = nn.Linear(2048, args.num_classes)
    # cls_y = nn.Linear(2048, args.num_classes)

    # if torch.cuda.is_available():
    #     cls_x = cls_x.cuda()
    #     cls_y = cls_y.cuda()
    #     criterion_cls = criterion_cls.cuda()

    # set the optimizer
    optimizer_x = set_optimizer(args, model_x, encoder_x)
    optimizer_y = set_optimizer(args, model_y, encoder_y)
    optimizer_z = set_optimizer(args, model_z, encoder_z)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model_x.load_state_dict(checkpoint['model_x'])
            model_y.load_state_dict(checkpoint['model_y'])
            model_z.load_state_dict(checkpoint['model_z'])
            encoder_x.load_state_dict(checkpoint['encoder_x'])
            encoder_y.load_state_dict(checkpoint['encoder_y'])
            encoder_z.load_state_dict(checkpoint['encoder_z'])
            optimizer_x.load_state_dict(checkpoint['optimizer_x'])
            optimizer_y.load_state_dict(checkpoint['optimizer_y'])
            optimizer_z.load_state_dict(checkpoint['optimizer_z'])
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

        adjust_learning_rate(epoch - args.start_epoch, args, optimizer_x)
        adjust_learning_rate(epoch - args.start_epoch, args, optimizer_y)
        adjust_learning_rate(epoch - args.start_epoch, args, optimizer_z)
        print("==> training...")

        time1 = time.time()
        loss, x_loss, y_loss, z_loss = train(epoch, train_loader, 
                                        model_x, model_y, model_z, 
                                        encoder_x, encoder_y, encoder_z, 
                                        contrast, 
                                        criterion_x, criterion_y, criterion_z,
                                        optimizer_x, optimizer_y, optimizer_z,
                                        args)
        time2 = time.time()
        print('epoch {} train, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('x_loss', x_loss, epoch)
        # logger.log_value('x_prob', x_prob, epoch)
        # logger.log_value('y_loss', y_loss, epoch)
        # logger.log_value('y_prob', y_prob, epoch)
        logger.log_value('loss', loss, epoch)
        logger.log_value('x_loss', x_loss, epoch)
        logger.log_value('y_loss', y_loss, epoch)
        logger.log_value('z_loss', z_loss, epoch)

        # save the best model
        if loss < best_loss:
            best_loss = loss
            state = {
                'opt': args,
                'model_x': model_x.state_dict(),
                'model_y': model_y.state_dict(),
                'model_z': model_z.state_dict(),
                'encoder_x': encoder_x.state_dict(),
                'encoder_y': encoder_y.state_dict(),
                'encoder_z': encoder_z.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_y': optimizer_y.state_dict(),
                'optimizer_z': optimizer_z.state_dict(),
                'epoch': epoch,
            }
            save_name = '{}_{}_best.pth'.format(args.task, args.model)
            save_name = os.path.join(args.model_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model_x': model_x.state_dict(),
                'model_y': model_y.state_dict(),
                'model_z': model_z.state_dict(),
                'encoder_x': encoder_x.state_dict(),
                'encoder_y': encoder_y.state_dict(),
                'encoder_z': encoder_z.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_y': optimizer_y.state_dict(),
                'optimizer_z': optimizer_z.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, '{task}_epoch_{epoch}.pth'.format(task=args.task, epoch=epoch))    
            print('saving regular model!')
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

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
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC, Normalize
from models.i3d import MyI3DCMC, I3D
from models.tsm import MyTSMCMC, TSN, ConsensusModule
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from datasets.dataset import ImageFolderInstance
from datasets.ntu import NTU, get_dataloaders

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
    parser.add_argument('--batch_size', type=int, default=128   , help='batch_size')
    parser.add_argument('--batch_size_glb', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
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
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
    parser.add_argument('--test', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   

    # model definition
    parser.add_argument('--model', type=str, default='tsm', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3',
                                                                         'tsm', 'i3d'])
    parser.add_argument('--base_model', type=str, default='resnet18')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=511)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='coefficient of unsupervised loss')

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

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.accum = int(opt.batch_size_glb / opt.batch_size)

    opt.method = 'softmax' if opt.softmax else 'nce'
    if opt.task is None:
        raise Exception('Task name is None.')
    opt.model_name = '{}_{}_{}_{}_lam_bsz_{}_view_RGBD'.format(opt.task, opt.method, opt.nce_k, opt.model, opt.lambda_u, opt.batch_size_glb)

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
        model_x  = TSN(num_class=args.num_class)
        model_y  = TSN(num_class=args.num_class)
        encoder_x = nn.Linear(512, args.feat_dim) # [2048, 128]
        encoder_y = nn.Linear(512, args.feat_dim)
        # classifier_x = nn.Linear(512, 120)
        # classifier_y = nn.Linear(512, 120)
        # classifier = nn.Linear(1024, 120)
    elif args.model == 'i3d':
        model_x = I3D()
        model_y = I3D()
        encoder_x = nn.Linear(2048, args.feat_dim) # [2048, 128]
        encoder_y = nn.Linear(2048, args.feat_dim)
        # classifier_x = nn.Linear(2048, 120)
        # classifier_y = nn.Linear(2048, 120)
        # classifier = nn.Linear(4096, 120)
    else:
        raise Exception("model not implemented.")

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_x = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_y = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion = nn.CrossEntropyLoss()
    # ===================classifier=====================
    # classifier_x = classifier_x.cuda()
    # classifier_x = nn.DataParallel(classifier_x)
    # classifier_x.train()

    # classifier_y = classifier_y.cuda()
    # classifier_y = nn.DataParallel(classifier_y)
    # classifier_y.train()

    # classifier = classifier.cuda()
    # classifier = nn.DataParallel(classifier)
    # classifier.train()

    if torch.cuda.is_available():
        model_x = model_x.cuda()
        model_y = model_y.cuda()
        encoder_x = encoder_x.cuda()
        encoder_y = encoder_y.cuda()
        contrast = contrast.cuda()
        criterion_y = criterion_y.cuda()
        criterion_x = criterion_x.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model_x = nn.DataParallel(model_x)
        model_y = nn.DataParallel(model_y)
        encoder_x = nn.DataParallel(encoder_x)
        encoder_y = nn.DataParallel(encoder_y)
        # contrast = nn.DataParallel(contrast)

    # return model_x, model_y, encoder_x, encoder_y, contrast, criterion_y, criterion_x, classifier_x, classifier_y, classifier, criterion
    return model_x, model_y, encoder_x, encoder_y, contrast, criterion_y, criterion_x, criterion


def set_optimizer(args, model, encoder):
    # return optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()),
                                lr=args.learning_rate,
                                betas=[args.beta1, args.beta2])
    return optimizer


def train(epoch, labeled_loader, unlabeled_loader, model_x, model_y, encoder_x, encoder_y, contrast, criterion_x, criterion_y, criterion, optimizer_x, optimizer_y, args):
    """
    one epoch training
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_y = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()
    cls_l_loss_meter = AverageMeter()
    cls_ab_loss_meter = AverageMeter()
    acc_l_meter = AverageMeter()
    acc_ab_meter = AverageMeter()
    top1 = AverageMeter()
    top1_x = AverageMeter()
    top1_y = AverageMeter()
    top5 = AverageMeter()
    top5_x = AverageMeter()
    top5_y = AverageMeter()

    end = time.time()
    optimizer_x.zero_grad()
    optimizer_y.zero_grad()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    for idx in range(len(labeled_loader)):
        data_time.update(time.time() - end)
        try:
            inputs, index = labeled_iter.next()
        except:
            labeled_iter = iter(labeled_loader)
            inputs, index = labeled_iter.next()
        try:
            inputs_u, index_u = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_loader)
            inputs_u, index_u = unlabeled_iter.next()
        # l, ab = inputs['rgb'], inputs['rgb']
        l, ab = inputs['rgb'], inputs['dep']
        label = inputs['label']
        l_u, ab_u = inputs_u['rgb'], inputs_u['dep']
        # print (l.size())
        # print (ab.size())
        bsz = l.size(0)
        l = l.float()
        ab = ab.float()
        l_u = l.float()
        ab_u = ab.float()

        # print (torch.max(ab[0]), 'max')
        # print (torch.min(ab[0]), 'min')
        # print (torch.mean(ab[0]), 'mean')
        if torch.cuda.is_available():
            index = index.cuda()
            l = l.cuda()
            ab = ab.cuda()
            l_u = l_u.cuda()
            ab_u = ab_u.cuda()
            label = label.cuda()

        # ===================forward feature=====================
        model_x.train()
        model_y.train()
        encoder_x.train()
        encoder_y.train()
        # model.eval() # verify after loaded model, cmc loss start with 4.xx
        contrast.train()
        # contrast.eval()
        # ===================supervised=====================
        _, logit_l = model_x(l)
        _, logit_ab = model_y(ab) # [bs, 8, 2048]
        # ===================unsupervised=====================
        feat_l, _ = model_x(l_u)
        feat_ab, _ = model_y(ab_u) # [bs, 8, 2048]
        # print (logit_l.size()) # [bs, 120]
        if args.model == 'tsm':
            # ===================consensus feature=====================
            consensus = ConsensusModule('avg')
            l2norm = Normalize(2)
            # ===================forward encoder=====================
            enc_l = l2norm(encoder_x(feat_l))
            enc_ab = l2norm(encoder_y(feat_ab))
            enc_l = enc_l.view((-1, args.num_segments) + enc_l.size()[1:])
            enc_ab = enc_ab.view((-1, args.num_segments) + enc_ab.size()[1:])
            # logit_l = logit_l.view((-1, args.num_segments) + logit_l.size()[1:])
            # logit_ab = logit_ab.view((-1, args.num_segments) + logit_ab.size()[1:])
            # print (enc_l.size())
            enc_l = consensus(enc_l).squeeze()
            enc_ab = consensus(enc_ab).squeeze()
            # logit_l = consensus(logit_l).squeeze()
            # logit_ab = consensus(logit_ab).squeeze()
            # print (enc_l.size())
        elif args.model == 'i3d':
            enc_l =  encoder_x(feat_l)
            enc_ab = encoder_y(feat_ab) 
        out_l, out_ab = contrast(enc_l, enc_ab, index)
        l_loss = criterion_x(out_l)
        ab_loss = criterion_y(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        cls_l_loss = criterion(logit_l, label)
        cls_ab_loss = criterion(logit_ab, label)

        us_loss = l_loss + ab_loss
        ss_loss = cls_l_loss + cls_ab_loss
        loss = ss_loss + args.lambda_u * us_loss
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
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)
        cls_l_loss_meter.update(cls_l_loss.item(), bsz)
        cls_ab_loss_meter.update(cls_ab_loss.item(), bsz)
        # acc_l_meter.update(acc_l, bsz)
        # acc_ab_meter.update(acc_ab, bsz)
        # ===================accuracy=====================
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_x, acc5_x = accuracy(logit_l, label, topk=(1, 5))
        acc1_y, acc5_y = accuracy(logit_ab, label, topk=(1, 5))
        top1_x.update(acc1_x[0], bsz)
        top1_y.update(acc1_y[0], bsz)
        top5_x.update(acc5_x[0], bsz)
        top5_y.update(acc5_y[0], bsz)
        # losses.update(loss.item(), input_x.size(0))
        losses_x.update(cls_l_loss.item(), bsz)
        losses_y.update(cls_ab_loss.item(), bsz)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % (args.print_freq * args.accum) == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(unlabeled_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            # print(out_l.shape)
            sys.stdout.flush()
            print('ViewX: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(unlabeled_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses_x, top1=top1_x, top5=top5_x))
            sys.stdout.flush()
            print('ViewY: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(unlabeled_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses_y, top1=top1_y, top5=top5_y))
            sys.stdout.flush()
            print ('')

    # return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg, cls_l_loss_meter.avg, cls_ab_loss_meter.avg 
    return losses.avg, l_loss_meter.avg, ab_loss_meter.avg, top1_x.avg, top5_x.avg, losses_x.avg, top1_y.avg, top5_y.avg, losses_y.avg

def validate(val_loader, model_x, model_y, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_y = AverageMeter()
    top1 = AverageMeter()
    top1_x = AverageMeter()
    top1_y = AverageMeter()
    top5 = AverageMeter()
    top5_x = AverageMeter()
    top5_y = AverageMeter()

    # switch to evaluate mode
    model_x.eval()
    model_y.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, index) in enumerate(val_loader):

            input_x = inputs['rgb']
            input_y = inputs['dep']
            input_x = input_x.float()
            input_y = input_y.float()
            target = inputs['label']
            if torch.cuda.is_available():
                input_x = input_x.cuda()
                input_y = input_y.cuda()
                target = target.cuda()
            # ===================forward=====================
            _, logit_l = model_x(input_x) # [bs, 8, 512]
            _, logit_ab = model_y(input_y)   # [bs, 8, 512]
            # feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)

            # ===================consensus feature=====================
            # if opt.model == 'tsm':
            #     consensus = ConsensusModule('avg')
            #     logit_l = logit_l.view((-1, args.num_segments) + logit_l.size()[1:])
            #     logit_ab = logit_ab.view((-1, args.num_segments) + logit_ab.size()[1:])
            #     logit_l = consensus(logit_l).squeeze()
            #     logit_ab = consensus(logit_ab).squeeze()
            # print (output.size()) # [bs, 120]
            loss_x = criterion(logit_l, target)
            loss_y = criterion(logit_ab, target)

            acc1_x, acc5_x = accuracy(logit_l, target, topk=(1, 5))
            acc1_y, acc5_y = accuracy(logit_ab, target, topk=(1, 5))
            losses_x.update(loss_x.item(), input_x.size(0))
            losses_y.update(loss_y.item(), input_y.size(0))
            top1_x.update(acc1_x[0], input_x.size(0))
            top1_y.update(acc1_y[0], input_y.size(0))
            top5_x.update(acc5_x[0], input_x.size(0))
            top5_y.update(acc5_y[0], input_y.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: ViewX: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses_x,
                       top1=top1_x, top5=top5_x))
                print('Test ViewY: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses_y,
                       top1=top1_y, top5=top5_y))
                print ('')

        print(' *[ViewX] Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1_x, top5=top5_x))
        print(' *[ViewY] Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1_y, top5=top5_y))

    return top1_x.avg, top5_x.avg, losses_x.avg, top1_y.avg, top5_y.avg, losses_y.avg


def main():
    global best_acc1
    global best_acc1_x
    global best_acc1_y
    best_acc1 = 0
    best_acc1_x = 0
    best_acc1_y = 0
    best_loss = 1e3
    # parse the args
    args = parse_option()

    # set the loader
    # train_loader, n_data = get_train_loader(split='train', args=args)
    # eval_loader, _ = get_train_loader(split='dev', args=args)
    # test_loader, _ = get_train_loader(split='test', args=args)

    # set the loader
    unlabeled_loader, n_data = get_dataloaders(args=args, stage='train')
    labeled_loader, _ = get_dataloaders(args=args, stage='train25') # 5% labeled data
    eval_loader, _ = get_dataloaders(args=args, stage='dev')
    test_loader, _ = get_dataloaders(args=args, stage='test')
    
    # set the model
    model_x, model_y, encoder_x, encoder_y, contrast, criterion_y, criterion_x, criterion = set_model(args, n_data)
    
    if args.test:
        # load pre-trained model
        print('==> loading pre-trained model for testing')
        ckpt = torch.load(args.test)
        model_x.load_state_dict(ckpt['model_x']) # rgb
        model_y.load_state_dict(ckpt['model_y']) # depth
        print("==> loaded checkpoint for testing'{}' (epoch {})".format(args.test, ckpt['epoch']))
        print('==> done')
        top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = validate(test_loader, model_x, model_y, criterion, args)
        exit()
    # cls_l = nn.Linear(2048, args.num_classes)
    # cls_ab = nn.Linear(2048, args.num_classes)

    # if torch.cuda.is_available():
    #     cls_l = cls_l.cuda()
    #     cls_ab = cls_ab.cuda()
    #     criterion_cls = criterion_cls.cuda()

    # set the optimizer
    optimizer_x = set_optimizer(args, model_x, encoder_x)
    optimizer_y = set_optimizer(args, model_y, encoder_y)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model_x.load_state_dict(checkpoint['model_x'])
            encoder_x.load_state_dict(checkpoint['encoder_x'])
            model_y.load_state_dict(checkpoint['model_y'])
            encoder_y.load_state_dict(checkpoint['encoder_y'])
            optimizer_x.load_state_dict(checkpoint['optimizer_x'])
            optimizer_y.load_state_dict(checkpoint['optimizer_y'])
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

        # adjust_learning_rate(epoch - args.start_epoch, args, optimizer_x)
        # adjust_learning_rate(epoch - args.start_epoch, args, optimizer_y)
        print("==> training...")

        time1 = time.time()
        loss, l_loss, ab_loss, top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = \
                                        train(epoch, labeled_loader, unlabeled_loader, model_x, model_y, encoder_x, encoder_y, contrast, 
                                        criterion_x, criterion_y, criterion, optimizer_x, optimizer_y, args)
        # top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = \
                                                    # validate(eval_loader, model_x, model_y, criterion, args)
        time2 = time.time()
        print('epoch {} train, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('l_loss', l_loss, epoch)
        # logger.log_value('l_prob', l_prob, epoch)
        # logger.log_value('ab_loss', ab_loss, epoch)
        # logger.log_value('ab_prob', ab_prob, epoch)
        logger.log_value('loss', loss, epoch)
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('x/train_acc', top1_x, epoch)
        logger.log_value('x/train_acc5', top5_x, epoch)
        logger.log_value('x/train_loss', losses_x, epoch)
        logger.log_value('y/train_acc', top1_y, epoch)
        logger.log_value('y/train_acc5', top5_y, epoch)
        logger.log_value('y/train_loss', losses_y, epoch)
        # print("==> evaluating...")
        
        # time1 = time.time()
        # cls_l_loss, cls_ab_loss = eval(epoch, eval_loader, model_x, model_y, encoder_x, encoder_y, contrast, 
        #                                 criterion_x, criterion_y, optimizer_x, optimizer_y, args)
        # time2 = time.time()
        # print('epoch {} test, total time {:.2f}'.format(epoch, time2 - time1))

        # # print("==> testing...")
        
        # # time1 = time.time()
        # # cls_l_loss, cls_ab_loss = eval(epoch, test_loader, model, contrast, criterion_x, criterion_y, criterion_cls,
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

        print("==> evaluating...")
        top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = \
                                                    validate(eval_loader, model_x, model_y, criterion, args)

        logger.log_value('x/eval_acc', top1_x, epoch)
        logger.log_value('x/eval_acc5', top5_x, epoch)
        logger.log_value('x/eval_loss', losses_x, epoch)
        logger.log_value('y/eval_acc', top1_y, epoch)
        logger.log_value('y/eval_acc5', top5_y, epoch)
        logger.log_value('y/eval_loss', losses_y, epoch)

        # save the best model
        if top1_x > best_acc1_x:
            best_acc1_x = top1_x
            state = {
                'opt': args,
                'epoch': epoch,
                'model_x': model_x.state_dict(),
                'model_y': model_y.state_dict(),
                'best_acc1': best_acc1_x,
                'encoder_x': encoder_x.state_dict(),
                'encoder_y': encoder_y.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_y': optimizer_y.state_dict(),
            }
            save_name = '{}_best_rgb.pth'.format(args.model)
            save_name = os.path.join(args.model_folder, save_name)
            print('saving best rgb model!')
            torch.save(state, save_name)
        
        # save the best model
        if top1_y > best_acc1_y:
            best_acc1_y = top1_y
            state = {
                'opt': args,
                'epoch': epoch,
                'model_x': model_x.state_dict(),
                'model_y': model_y.state_dict(),
                'best_acc1': best_acc1_y,
                'encoder_x': encoder_x.state_dict(),
                'encoder_y': encoder_y.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_y': optimizer_y.state_dict(),
            }
            save_name = '{}_best_dep.pth'.format(args.model)
            save_name = os.path.join(args.model_folder, save_name)
            print('saving best dep model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'model_x': model_x.state_dict(),
                'model_y': model_y.state_dict(),
                'best_acc1': best_acc1,
                'encoder_x': encoder_x.state_dict(),
                'encoder_y': encoder_y.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer_x': optimizer_x.state_dict(),
                'optimizer_y': optimizer_y.state_dict(),
            }
            save_name = 'rgbd_ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.model_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

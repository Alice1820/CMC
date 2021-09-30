from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import argparse
import socket
from torch.utils.data import distributed
import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
# from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNet
from models.tsm import MyTSMCMC, TSN, ConsensusModule
from models.i3d import I3D

from datasets.ntu import NTU, get_dataloaders

# from spawn import spawn


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--batch_size_glb', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30, 60, 90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='i3d', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3',
                                                                         'tsm', 'i3d'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--test', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    # dataset
    parser.add_argument('--dataset', type=str, default='train25', choices=['train100', 'train5', 'train25', 'train50'])

    # video
    parser.add_argument('--num_segments', type=int, default=8, help='')
    parser.add_argument('--num_class', type=int, default=60, help='')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # path definition
    parser.add_argument('--data_folder', type=str, default='/data0/xifan/NTU_RGBD_60', help='path to data')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='path to save linear classifier')
    parser.add_argument('--tb_path', type=str, default='logs', help='path to tensorboard')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')
    parser.add_argument('--task', type=str, default=None)

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.save_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    opt.accum = int(opt.batch_size_glb / opt.batch_size)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.task is None:
        raise Exception('Task name is None.')
    # opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = '{}_data_{}_bsz_{}_lr_{}_decay_{}'.format(opt.task, opt.dataset, opt.batch_size_glb, opt.learning_rate,
                                                                  opt.weight_decay)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder) and not opt.test:
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder) and not opt.test:
        os.makedirs(opt.save_folder)

    if opt.dataset == 'imagenet100':
        opt.n_label = 100
    if opt.dataset == 'imagenet':
        opt.n_label = 1000

    return opt


def get_train_loader(split='train', args=None):
    """get the train loader"""
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

def set_model(args):
    if args.model == 'tsm':
        model_x = TSN()
        model_y = TSN()
        classifier_x = nn.Linear(512, 120)
        classifier_y = nn.Linear(512, 120)
        classifier = nn.Linear(1024, 120)
    elif args.model == 'i3d':
        model_x = I3D(low_dim=120)
        model_y = I3D(low_dim=120)
        classifier_x = nn.Linear(2048, 120)
        classifier_y = nn.Linear(2048, 120)
        classifier = nn.Linear(4096, 120)
    # ===================model x=====================
    model_x = model_x.cuda()
    model_x = nn.DataParallel(model_x)
    # ===================model y=====================
    model_y = model_y.cuda()
    model_y = nn.DataParallel(model_y)
    # ===================classifier=====================
    classifier_x = classifier_x.cuda()
    classifier_x = nn.DataParallel(classifier_x)
    # classifier_x.train()

    classifier_y = classifier_y.cuda()
    classifier_y = nn.DataParallel(classifier_y)
    # classifier_y.train()

    classifier = classifier.cuda()
    classifier = nn.DataParallel(classifier)
    # classifier.train()

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load(args.test)
    # model_x.load_state_dict(ckpt['model_x']) # rgb
    # model_y.load_state_dict(ckpt['model_y']) # depth
    # classifier.load_state_dict(ckpt['classifier'])
    # classifier_x.load_state_dict(ckpt['classifier_x'])
    # classifier_y.load_state_dict(ckpt['classifier_y'])

    # 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # print (ckpt['classifier_x'])
    for k in (ckpt['classifier_x']).keys():
        name = k[7:] # remove module.
        new_state_dict['classifier.'+name] = ckpt['classifier_x'][k]
    for k in (ckpt['model_x']).keys():
        name = k[7:] # remove module.
        if not 'classifier' in name:
            # print (k)
            new_state_dict[name] = ckpt['model_x'][k]
    model_x.module.load_state_dict(new_state_dict)
    new_state_dict = OrderedDict()
    for k in (ckpt['classifier_y']).keys():
        name = k[7:] # remove module.
        new_state_dict['classifier.'+name] = ckpt['classifier_y'][k]
    for k in (ckpt['model_y']).keys():
        name = k[7:] # remove module.
        if not 'classifier' in name:
            new_state_dict[name] = ckpt['model_y'][k]
    model_y.module.load_state_dict(new_state_dict)
    

    # save pre-trained model
    # torch.save(model_x.module.state_dict(), os.path.join(args.save_path, 'tsm_rgb_resnet18_ntu60.pth'))
    torch.save(model_x.module.state_dict(), os.path.join(args.save_path, 'i3d_rgb_resnet50_ntu60.pth'))
    # torch.save(model_y.module.state_dict(), os.path.join(args.save_path, 'tsm_dep_resnet18_ntu60.pth'))    
    torch.save(model_y.module.state_dict(), os.path.join(args.save_path, 'i3d_dep_resnet50_ntu60.pth'))    
    print("==> loaded checkpoint for testing'{}' (epoch {})".format(args.test, ckpt['epoch']))
    print('==> done')
    
    exit()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model_x, model_y, classifier_x, classifier_y, classifier, criterion

def main():
    global best_acc1
    global best_acc1_x
    global best_acc1_y
    best_acc1 = 0
    best_acc1_x = 0
    best_acc1_y = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    # train_loader, n_data = get_train_loader('train', args)
    # val_loader, _ = get_train_loader('dev', args)
        # set the loader
    # train100_loader, n_data = get_dataloaders(args=args, stage='train')
    # train5_loader, n_data = get_dataloaders(args=args, stage='train5')
    # train25_loader, n_data = get_dataloaders(args=args, stage='train25')
    # train50_loader, n_data = get_dataloaders(args=args, stage='train50')
    # train_loader = {'train100': train100_loader, 'train5': train5_loader, 'train25': train25_loader, 'train50': train50_loader}[args.dataset]
    # eval_loader, _ = get_dataloaders(args=args, stage='dev')
    # test_loader, _ = get_dataloaders(args=args, stage='test')
    # set the model
    model_x, model_y, classifier_x, classifier_y, classifier, criterion = set_model(args)

    # set optimizer
    optimizer = set_optimizer_cls(args, classifier)
    optimizer_x = set_optimizer_joint(args, model_x, classifier_x)
    optimizer_y = set_optimizer_joint(args, model_y, classifier_y)

    cudnn.benchmark = True

    # optionally resume linear classifier
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not args.test:
        # tensorboard
        logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

        # routine
        for epoch in range(args.start_epoch, args.epochs + 1):

            # adjust_learning_rate(epoch, args, optimizer)
            # adjust_learning_rate(epoch, args, optimizer_x)
            # adjust_learning_rate(epoch, args, optimizer_y)
            print("==> training...")

            time1 = time.time()
            top1, top5, losses, top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = \
                                                        train(epoch, train_loader, model_x, model_y, \
                                                        classifier_x, classifier_y, classifier, criterion, optimizer_x, optimizer_y, optimizer, args)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            logger.log_value('joint/train_acc', top1, epoch)
            logger.log_value('joint/train_acc5', top5, epoch)
            logger.log_value('joint/train_loss', losses, epoch)
            logger.log_value('x/train_acc', top1_x, epoch)
            logger.log_value('x/train_acc5', top5_x, epoch)
            logger.log_value('x/train_loss', losses_x, epoch)
            logger.log_value('y/train_acc', top1_y, epoch)
            logger.log_value('y/train_acc5', top5_y, epoch)
            logger.log_value('y/train_loss', losses_y, epoch)

            print("==> evaluating...")
            top1, top5, losses, top1_x, top5_x, losses_x, top1_y, top5_y, losses_y = \
                                                        validate(eval_loader, model_x, model_y, classifier_x, classifier_y, classifier, criterion, args)

            logger.log_value('joint/eval_acc', top1, epoch)
            logger.log_value('joint/eval_acc5', top5, epoch)
            logger.log_value('joint/eval_loss', losses, epoch)
            logger.log_value('x/eval_acc', top1_x, epoch)
            logger.log_value('x/eval_acc5', top5_x, epoch)
            logger.log_value('x/eval_loss', losses_x, epoch)
            logger.log_value('y/eval_acc', top1_y, epoch)
            logger.log_value('y/eval_acc5', top5_y, epoch)
            logger.log_value('y/eval_loss', losses_y, epoch)

            # save the best model
            if top1 > best_acc1:
                best_acc1 = top1
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'model_x': model_x.state_dict(),
                    'model_y': model_y.state_dict(),
                    'classifier': classifier.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'classifier_y': classifier_y.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_x': optimizer_x.state_dict(),
                    'optimizer_y': optimizer_y.state_dict(),
                }
                save_name = '{}_best_joint.pth'.format(args.model)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving best joint model!')
                torch.save(state, save_name)

            # save the best model
            if top1_x > best_acc1_x:
                best_acc1_x = top1_x
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'model_x': model_x.state_dict(),
                    'model_y': model_y.state_dict(),
                    'classifier': classifier.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'classifier_y': classifier_y.state_dict(),
                    'best_acc1': best_acc1_x,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_x': optimizer_x.state_dict(),
                    'optimizer_y': optimizer_y.state_dict(),
                }
                save_name = '{}_best_rgb.pth'.format(args.model)
                save_name = os.path.join(args.save_folder, save_name)
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
                    'classifier': classifier.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'classifier_y': classifier_y.state_dict(),
                    'best_acc1': best_acc1_y,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_x': optimizer_x.state_dict(),
                    'optimizer_y': optimizer_y.state_dict(),
                }
                save_name = '{}_best_dep.pth'.format(args.model)
                save_name = os.path.join(args.save_folder, save_name)
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
                    'classifier': classifier.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'classifier_y': classifier_y.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_x': optimizer_x.state_dict(),
                    'optimizer_y': optimizer_y.state_dict(),
                }
                save_name = 'rgbd_ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving regular model!')
                torch.save(state, save_name)

            # tensorboard logger
            pass
    else:
        print("==> testing...")
        validate(test_loader, model_x, model_y, classifier_x, classifier_y, classifier, criterion, args)


if __name__ == '__main__':
    best_acc1 = 0
    main()

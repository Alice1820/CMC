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
    parser.add_argument('--model', type=str, default='tsm', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3',
                                                                         'tsm', 'i3d'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--test', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    # dataset
    parser.add_argument('--dataset', type=str, default='train25', choices=['train100', 'train5', 'train25', 'train50'])

    # modality
    parser.add_argument('--modal', type=str, default='ir', choices=['rgb', 'dep', 'ir'])

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
    opt.model_name = '{}_moda_{}_data_{}_bsz_{}_lr_{}_decay_{}'.format(opt.task, opt.modal, opt.dataset, opt.batch_size_glb, opt.learning_rate,
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

def get_train_val_loader(args):
    train_folder = os.path.join(args.data_folder, 'train')
    val_folder = os.path.join(args.data_folder, 'val')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))

    normalize = transforms.Normalize(mean=mean, std=std)
    train_dataset = datasets.ImageFolder(
        train_folder,
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        val_folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    )
    print('number of train: {}'.format(len(train_dataset)))
    print('number of val: {}'.format(len(val_dataset)))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


def set_model(args):
    if args.model == 'tsm':
        model_x = TSN()
        classifier_x = nn.Linear(512, 120)
    elif args.model == 'i3d':
        model_x = I3D()
        classifier_x = nn.Linear(2048, 120)
    # ===================model x=====================
    model_x = model_x.cuda()
    model_x = nn.DataParallel(model_x)
    classifier_x = classifier_x.cuda()
    classifier_x = nn.DataParallel(classifier_x)
    classifier_x.train()
    if args.test:
        # load pre-trained model
        print('==> loading pre-trained model')
        ckpt = torch.load(args.test)
        model_x.load_state_dict(ckpt['model_x']) # rgb
        classifier_x.load_state_dict(ckpt['classifier_x'])
        print("==> loaded checkpoint for testing'{}' (epoch {})".format(args.test, ckpt['epoch']))
        print('==> done')

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model_x, model_y, classifier_x, classifier_y, classifier, criterion


def set_optimizer_joint(args, model, classifier):
    optimizer = optim.Adam(list(classifier.parameters()) + list(model.parameters()),
                          lr=args.learning_rate,
                          betas=[args.beta1, args.beta2])
    return optimizer

def set_optimizer_cls(args, classifier):
    optimizer = optim.Adam(classifier.parameters(),
                          lr=args.learning_rate,
                          betas=[args.beta1, args.beta2])
    return optimizer


def train(epoch, train_loader, model_x, model_y, classifier_x, classifier_y, classifier, criterion, optimizer_x, optimizer_y, optimizer, opt):
    """
    one epoch training
    """
    model_x.train()
    classifier_x.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_x = AverageMeter()
    top1_x = AverageMeter()
    top5_x = AverageMeter()

    optimizer_x.zero_grad()
    end = time.time()
    for idx, (inputs, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_x = inputs['ir']
        input_x = input_x.float()
        target = inputs['label']
        if opt.gpu is not None:
            input_x = input_x.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        # with torch.no_grad():
        feat_x, _ = model_x(input_x) # [bs, 8, 512]

        # ===================consensus feature=====================
        if opt.model == 'tsm':
            consensus = ConsensusModule('avg')
            enc_x = classifier_x(feat_x) # [bs, 8, 120]
            enc_x = enc_x.view((-1, opt.num_segments) + enc_x.size()[1:])
            output_x = consensus(enc_x).squeeze()
        elif opt.model == 'i3d':
            output_x = classifier_x(feat_x) # [bs, 120]
        # print (output.size()) # [bs, 120]
        loss_x = criterion(output_x, target)

        acc1_x, acc5_x = accuracy(output_x, target, topk=(1, 5))
        losses_x.update(loss_x.item(), input_x.size(0))
        top1_x.update(acc1_x[0], input_x.size(0))
        top5_x.update(acc5_x[0], input_x.size(0))

        # ===================backward=====================
        loss_x.backward()
        if idx % opt.accum == 0:
            optimizer_x.step()
            optimizer_x.zero_grad()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % (opt.print_freq * opt.accum) == 0:
            print('ViewX: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses_x, top1=top1_x, top5=top5_x))
            sys.stdout.flush()

    return top1_x.avg, top5_x.avg, losses_x.avg

def validate(val_loader, model_x, model_y, classifier_x, classifier_y, classifier, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses_x = AverageMeter()
    top1_x = AverageMeter()
    top5_x = AverageMeter()

    # switch to evaluate mode
    model_x.eval()
    classifier_x.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, index) in enumerate(val_loader):

            input_x = inputs['ir']
            input_x = input_x.float()
            target = inputs['label']
            if opt.gpu is not None:
                input_x = input_x.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)
            # ===================forward=====================
            feat_x, _ = model_x(input_x) # [bs, 8, 512]
            feat_y, _ = model_y(input_y) # [bs, 8, 512]

            # ===================consensus feature=====================
            if opt.model == 'tsm':
                consensus = ConsensusModule('avg')
                enc_x = classifier_x(feat_x) # [bs, 8, 120]
                enc_x = enc_x.view((-1, opt.num_segments) + enc_x.size()[1:])
                output_x = consensus(enc_x).squeeze()
            elif opt.model == 'i3d':
                output_x = classifier_x(feat_x) # [bs, 120]
            loss_x = criterion(output_x, target)

            acc1_x, acc5_x = accuracy(output_x, target, topk=(1, 5))
            losses_x.update(loss_x.item(), input_x.size(0))
            top1_x.update(acc1_x[0], input_x.size(0))
            top5_x.update(acc5_x[0], input_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('ViewX: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses_x,
                       top1=top1_x, top5=top5_x))

        print(' *[ViewX] Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1_x, top5=top5_x))

    return top1_x.avg, top5_x.avg, losses_x.avg

def main():
    global best_acc1_x
    best_acc1_x = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    # train_loader, n_data = get_train_loader('train', args)
    # val_loader, _ = get_train_loader('dev', args)
        # set the loader
    train100_loader, n_data = get_dataloaders(args=args, stage='train')
    train5_loader, n_data = get_dataloaders(args=args, stage='train5')
    train25_loader, n_data = get_dataloaders(args=args, stage='train25')
    train50_loader, n_data = get_dataloaders(args=args, stage='train50')
    train_loader = {'train100': train100_loader, 'train5': train5_loader, 'train25': train25_loader, 'train50': train50_loader}[args.dataset]
    eval_loader, _ = get_dataloaders(args=args, stage='dev')
    test_loader, _ = get_dataloaders(args=args, stage='test')
    # set the model
    model_x, classifier_x, criterion = set_model(args)

    # set optimizer
    optimizer_x = set_optimizer_joint(args, model_x, classifier_x)

    cudnn.benchmark = True

    # optionally resume linear classifier

    if not args.test:
        # tensorboard
        logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

        # routine
        for epoch in range(args.start_epoch, args.epochs + 1):

            adjust_learning_rate(epoch, args, optimizer_x)
            print("==> training...")

            time1 = time.time()
            top1_x, top5_x, losses_x = \
                                    train(epoch, train_loader, model_x, classifier_x,criterion, optimizer_x)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            logger.log_value('x/train_acc', top1_x, epoch)
            logger.log_value('x/train_acc5', top5_x, epoch)
            logger.log_value('x/train_loss', losses_x, epoch)

            print("==> evaluating...")
            top1_x, top5_x, losses_x = validate(eval_loader, model_x, classifier_x, criterion, args)

            logger.log_value('x/eval_acc', top1_x, epoch)
            logger.log_value('x/eval_acc5', top5_x, epoch)
            logger.log_value('x/eval_loss', losses_x, epoch)

            # save the best model
            if top1_x > best_acc1_x:
                best_acc1_x = top1_x
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'model_x': model_x.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'best_acc1': best_acc1_x,
                    'optimizer_x': optimizer_x.state_dict(),
                }
                save_name = '{}_best_rgb.pth'.format(args.model)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving best ir model!')
                torch.save(state, save_name)

            # save model
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'model_x': model_x.state_dict(),
                    'classifier_x': classifier_x.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer_x': optimizer_x.state_dict(),
                }
                save_name = 'ir_ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving regular model!')
                torch.save(state, save_name)

            # tensorboard logger
            pass
    else:
        print("==> testing...")
        validate(test_loader, model_x, classifier_x, classifier, criterion, args)


if __name__ == '__main__':
    best_acc1 = 0
    main()

'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
# this code is modified from https://github.com/naver/cgd

import torch, math, time, argparse, os, sys, random
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from tqdm import *
import wandb

from loss import *
# import random, dataset, utils, losses, net
# from net.resnet import *
# from net.googlenet import *
# from net.bn_inception import *
# from dataset import sampler

#----------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx

import dataset as D
import transforms as T
from model import Model
from loss import HPHNTripletLoss
from runner import Trainer, Evaluator
from util import SummaryWriter

# define argparse

parser = argparse.ArgumentParser(description='Embedding Expansion PyTorch codes')
parser.add_argument('--gpu_id', default = 0, type = int)
parser.add_argument('--num_workers', default = 10, type = int)
parser.add_argument('--epochs', default = 5000, type = int, dest = 'nb_epochs')
parser.add_argument('--datapath', default='./dataset', type=str)
parser.add_argument('--dataset', default='cub', help = 'Training dataset, e.g. cub, cars')
parser.add_argument('--logpath', default='./logs', type = str)
parser.add_argument('--batch_size', default = 128, type = int, dest = 'bs')
parser.add_argument('--backbone', default = 'googlenet', type=str, help = 'Model for training')
parser.add_argument('--loss', default = 'HPHNTriplet', type=str, help = 'Criterion for training')
parser.add_argument('--optimizer', default = 'adam')
parser.add_argument('--lr', default = 1e-4, type =float)
parser.add_argument('--weight_decay', default = 5e-4, type =float)
parser.add_argument('--lr_decay_gamma', default = 0.5, type =float)
parser.add_argument('--lr_decay_step', default = '10,20,40,80', type =str)
parser.add_argument('--embedding_dim', default = 512, type = int )
parser.add_argument('--alpha', default = 10, type = float, help = 'Scaling Parameter setting')
parser.add_argument('--margin', default = 1e-5, type = float, help = 'Margin parameter setting')
parser.add_argument('--recall_k', default='1,2,4,8', type=str, help='k values for recall')
parser.add_argument('--num_instances', default=32, type=int, help='how many instances per class')
parser.add_argument('--n_inner_pts', default=2, type=int, help='the number of inner points. when it is 0, no EE')
parser.add_argument('--ee_l2norm', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'], help='whether do l2 normalizing augmented embeddings')
parser.add_argument('--soft_margin', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'], help='parameter for hphn triplet loss')
parser.add_argument('--img_size', default=227, type=int, help='width and height of input image')
#---------------

parser.add_argument('--base_lr_mult', default=1.0, type=float,
                    help='scale for gradients calculated at backbone')
parser.add_argument('--eval_epoch_term', default=50, type=int,
                    help='check every eval_epoch_term')
parser.add_argument('--beta', default=1.2, type=float,
                    help='beta is beta')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--summary_step', default=10, type=int,
                    help='write summary every summary_step')

def main():
    args = parser.parse_args()

    LOG_DIR = args.logpath + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}'.format(args.dataset, args.model, args.loss, args.embedding_dim, args.alpha, 
                                                                                                args.margin, args.optimizer, args.lr, args.bs)

    # wandb
    wandb.init(project=args.dataset, entity="cvmetriclearning", notes=LOG_DIR)
    wandb.config.update(args)

    # define args more
    args.train_meta = './meta/CARS196/train.txt'
    args.test_meta = './meta/CARS196/test.txt'
    args.lr_decay_step = [int(epoch) for epoch in args.lr_decay_step.split(',')]
    args.recall_k = [int(k) for k in args.recall_k.split(',')]
    print(args)

    # Set random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus

    # Load model
    model = Model(args.embed_dim, args.ctx)
    model.hybridize()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model = nn.DataParallel(model)
    model.to(device)

    # DML Losses
    if args.loss == 'MS':
        criterion = MultiSimilarityLoss().cuda()
    elif args.loss == 'Contrastive':
        criterion = ContrastiveLoss().cuda()
    elif args.loss == 'Triplet':
        criterion = TripletLoss().cuda()
    elif args.loss == 'NPair':
        criterion = NPairLoss().cuda()
    elif args.loss == 'HPHNTriplet':
        criterion = HPHNTripletLoss(margin=args.margin, soft_margin=False, num_instances=args.num_instances, n_inner_pts=args.n_inner_pts, l2_norm=args.ee_l2norm).cuda()

    # Optimizer Setting
    if args.optimizer == 'sgd':
	opt = torch.optim.SGD(params=model.parameters(), lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
    elif args.optimizer == 'adam':
	opt = torch.optim.Adam(params=model.parameters(), lr=float(args.lr), weight_decay = args.weight_decay)
    elif args.optimizer == 'rmsprop':
	opt = torch.optim.RMSprop(params=model.parameters(), lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
    elif args.optimizer == 'adamw':
	opt = torch.optim.AdamW(params=model.parameters(), lr=float(args.lr), weight_decay = args.weight_decay)


    # Load image transform
    train_transform, test_transform = T.get_transform(image_size=args.img_size)

    # Load data loader
    train_loader, test_loader = D.get_data_loader(args.data_dir, args.train_meta, args.test_meta, train_transform, test_transform,
                                                  args.batch_size, args.num_instances, args.num_workers)


    # LR scheduler
    print("steps in epoch:", args.lr_decay_step)
    steps = list(map(lambda x: x*len(train_loader) , args.lr_decay_step))
    print("steps in iter:", steps)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=steps, gamma = args.lr_decay_gamma)

#-----------------------

    # Load logger and saver
    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard_log'))


    # Load trainer & evaluator
    trainer   = Trainer(model, loss, optimizer, train_loader, summary_writer, args.ctx,
                        summary_step=args.summary_step,
                        lr_schedule=lr_schedule)
    
    evaluator = Evaluator(model, test_loader, args.ctx)
    best_metrics = [0.0]  # all query

    global_step = args.start_epoch * len(train_loader)
    
    # Enter to training loop
    print("base lr mult:", args.base_lr_mult)
    for epoch in range(args.start_epoch, args.epochs):
        model.backbone.collect_params().setattr('lr_mult', args.base_lr_mult)
            
        trainer.train(epoch)
        global_step = (epoch + 1) * len(train_loader)
        if (epoch + 1) % args.eval_epoch_term == 0:
            old_best_metric = best_metrics[0]
            # evaluate_and_log(summary_writer, evaluator, ranks, step, epoch, best_metrics)
            best_metrics = evaluate_and_log(summary_writer, evaluator, args.recallk,
                                        global_step, epoch + 1,
                                        best_metrics=best_metrics)
            if best_metrics[0] != old_best_metric:
                save_path = os.path.join(args.save_dir, 'model_epoch_%05d.params' % (epoch + 1))
                model.save_parameters(save_path)
        sys.stdout.flush()

def add_best_values_summary(summary_writer, global_step, epoch, recallk, best_recall):
    if summary_writer is None:
        return
    summary_writer.add_scalar('metric/R%d/best' % (recallk), best_recall, global_step)
    summary_writer.add_scalar('metric_epoch/R%d/best' % (recallk), best_recall, epoch)

def add_summary(summary_writer, step, epoch, ranks, recall_at_ranks):
    for recallk, recall in zip(ranks, recall_at_ranks):
        if summary_writer is not None:
            summary_writer.add_scalar('metric/R%d' % (recallk), recall, step)
            summary_writer.add_scalar('metric_epoch/R%d' % (recallk), recall, epoch)
        print("R@{:3d}: {:.4f}".format(recallk, recall))

def evaluate_and_log(summary_writer, evaluator, ranks, step, epoch, best_metrics):
    metrics = []

    distmat, labels = evaluator.get_distmat()
    recall_at_ranks = evaluator.get_metric_at_ranks(distmat, labels, ranks)

    add_summary(summary_writer, step, epoch, ranks, recall_at_ranks)

    metrics.append(recall_at_ranks[0])

    for idx, best_recall1 in enumerate(best_metrics):
        recall1 = metrics[idx]
        if recall1 > best_recall1:
            best_recall1 = recall1
            best_metrics[idx] = best_recall1

            add_best_values_summary(summary_writer, step, epoch if epoch is not None else None,
                                    ranks[0], best_recall1)

    return best_metrics


if __name__ == '__main__':
    # https://github.com/dmlc/gluon-cv/issues/493
    sys.setrecursionlimit(2000)

    main()

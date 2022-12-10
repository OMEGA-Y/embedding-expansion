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
import random, utils 
# import random, dataset, utils, losses, net
# from net.resnet import *
# from net.googlenet import *
# from net.bn_inception import *
# from dataset import sampler

#----------------

import dataset as D
import transforms as T
from model import Model
from loss import HPHNTripletLoss

parser = argparse.ArgumentParser(description='Embedding Expansion PyTorch codes')
parser.add_argument('--gpu_id', default = 0, type = int)
parser.add_argument('--num_workers', default = 10, type = int)
parser.add_argument('--epochs', default = 5000, type = int, dest = 'nb_epochs')
parser.add_argument('--data_dir', default='./dataset', type=str)
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
    
    args.lr_decay_epochs = [int(epoch) for epoch in args.lr_decay_epochs.split(',')]
    args.recallk = [int(k) for k in args.recallk.split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
    args.ctx = [mx.cpu()]
    print(args)

    # Set random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus

    # Load model
    model = Model(args.embed_dim)
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

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_decay_step, gamma = args.lr_decay_gamma)

    # Dataset initialization
    train_transform, test_transform = T.get_transform(image_size=args.img_size)
    train_loader, test_loader = D.get_data_loader(args.data_dir, args.train_meta, args.test_meta, train_transform, test_transform,
                                                  args.batch_size, args.num_instances, args.num_workers)



    losses_list = []
    best_recall=[0]
    best_epoch = 0

    # Training 
    for epoch in range(0, args.epochs):
        model.train()

        losses_per_epoch = []
        pbar = tqdm(enumerate(train_loader))

        for batch_idx, batch in pbar:

            # To cuda
            batch = utils.to_cuda(batch)
            images, instance_labels, category_labels, _ = batch

            # Compute loss
            embeddings = model(images)
            loss = criterion(embeddings, instance_labels)

            opt.zero_grad()
            loss.backward()

            losses_per_epoch.append(loss.data.cpu().numpy())

            opt.step()

            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item()))

	losses_list.append(np.mean(losses_per_epoch))
	wandb.log({'loss': losses_list[-1]}, step=epoch)
	scheduler.step()

	if(epoch >= 0):
            with torch.no_grad():
		print("**Evaluating...**")
                Recalls = utils.evaluate_cos(model, test_loader)

            # Logging Evaluation Score
            for i in range(4):
                wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)

            # Best model save
            if best_recall[0] < Recalls[0]:
		best_recall = Recalls
		best_epoch = epoch
		if not os.path.exists('{}'.format(LOG_DIR)):
                    os.makedirs('{}'.format(LOG_DIR))
		torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
		with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                    f.write('Best Epoch: {}\n'.format(best_epoch))
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))


if __name__ == '__main__':

    main()

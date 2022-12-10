'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
import torch
import numpy as np

import time
import datetime
import csv
import os

from .embedding_aug import get_embedding_aug

def euclidean_dist(x, y, clip_min=1e-12, clip_max=1e12):
    
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    return torch.cdist(x,y,p=2).squeeze(0)


class HPHNTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(HPHNTripletLoss, self).__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm

    def forward(self, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]
        if self.n_inner_pts != 0:
            gen_start_time = time.time()
            embeddings, labels = get_embedding_aug(embeddings, labels, self.num_instance, self.n_inner_pts, self.l2_norm)
            gen_time = time.time() - gen_start_time
        dist_mat = euclidean_dist(embeddings, embeddings)
        dist_ap, dist_an = self.hard_example_mining(dist_mat, labels)

        if self.soft_margin:
            loss = torch.log(1 + torch.exp(dist_ap - dist_an))
        else:
            loss = torch.relu(dist_ap - dist_an + self.margin)
        total_time = time.time() - total_start_time

        return loss.mean()

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        assert len(dist_mat.shape) == 2
        assert dist_mat.shape[0] == dist_mat.shape[1]

        N = dist_mat.shape[0]

        is_pos = torch.eq(labels.repeat(N, 1), labels.repeat(N, 1).T)
        is_neg = torch.ne(labels.repeat(N, 1), labels.repeat(N, 1).T)

        dist_pos = dist_mat * is_pos
        if self.n_inner_pts != 0:
            dist_ap = torch.max(dist_pos[:self.batch_size, :self.batch_size], axis=1)[0]
        else:
            dist_ap = torch.max(dist_pos, axis=1)[0]

        dist_neg = dist_mat * is_neg + torch.max(dist_mat, axis=1, keepdims=True)[0] * is_pos
        dist_an = torch.min(dist_neg, axis=1)[0]

        if self.n_inner_pts != 0:
            num_group = N // self.batch_size
            dist_an = torch.min(torch.reshape(dist_an, (num_group, self.batch_size)), axis=0)[0] # include synthetic positives

        return dist_ap, dist_an

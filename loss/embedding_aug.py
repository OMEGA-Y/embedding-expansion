'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
import torch
import numpy as np

def get_embedding_aug(embeddings, labels, num_instance, n_inner_pts, l2=True):
    batch_size = embeddings.shape[0]
    
    assert num_instance % 2 == 0, 'num_instance should be even number for simple implementation'
    swap_axes_list = [i + 1 if i % 2 == 0 else i - 1 for i in range(batch_size)]
    swap_embeddings = embeddings[swap_axes_list]
    pos = embeddings
    anchor = swap_embeddings
    concat_embeddings = embeddings.clone()
    concat_labels = labels.clone()
    n_pts = n_inner_pts
    total_length = float(n_pts + 1)
    for n_idx in range(n_pts):
        left_length = float(n_idx + 1)
        right_length = total_length - left_length
        inner_pts = (anchor * left_length + pos * right_length) / total_length
        if l2:
            inner_pts = l2_norm(inner_pts)
        concat_embeddings = torch.cat([concat_embeddings, inner_pts], dim=0)
        concat_labels = torch.cat([concat_labels, labels], dim=0)

    return concat_embeddings, concat_labels

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-5)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

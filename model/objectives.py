import torch
import torch.nn as nn
import torch.nn.functional as F
# import build as get_triplets
from torch.autograd import Variable
from torch.autograd.function import Function
import numpy as np
from model import objectives



def class_contrastive_loss(embedding, class_labels):
    normalized_feat = F.normalize(embedding, dim=1)
    sim_matrix = torch.matmul(normalized_feat, normalized_feat.T)
    labels = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)
    pos_mask = labels.float()
    neg_mask = 1.0 - pos_mask

    pos_sim = torch.exp(sim_matrix) * pos_mask
    neg_sim = torch.exp(sim_matrix) * neg_mask
    loss = -torch.log(pos_sim.sum(1) / (pos_sim.sum(1) + neg_sim.sum(1) + 1e-8)).mean()
    return loss

def diversity_loss(z_list):  
    B, K, D = z_list.shape
    z = F.normalize(z_list, dim=2)
    sim_matrix = torch.matmul(z, z.transpose(1, 2))  
    identity = torch.eye(K, device=z.device).unsqueeze(0)
    diversity = ((sim_matrix - identity) ** 2).mean()
    return diversity

def compute_itc(x_pool, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = x_pool.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(x_pool.device)

    image_norm = x_pool / x_pool.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

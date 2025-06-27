import torch
import torch.nn as nn
import torch.nn.functional as F
# import build as get_triplets
from torch.autograd import Variable
from torch.autograd.function import Function
import numpy as np
from model import objectives

# class CPMLoss(nn.Module):
#     def __init__(self, margin=0.2):
#         super(CPMLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=0.2)
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 

def compute_cmpm(self, inputs, logit_scale):
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(logit_scale, 4, 0)
        
        lb_num = len(lb1.unique())    
        lbs = lb1.unique() 

        n = lbs.size(0)   

        ft1 = ft1.chunk(lb_num, 0)
        ft2 = ft2.chunk(lb_num, 0)
        ft3 = ft3.chunk(lb_num, 0)
        ft4 = ft4.chunk(lb_num, 0)
        center1 = []
        center2 = []
        center3 = []
        center4 = []
        for i in range(lb_num):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)

        dist_13 = pdist_torch(ft1, ft3)
        dist_23 = pdist_torch(ft2, ft3)
        dist_33 = pdist_torch(ft3, ft3)
        dist_11 = pdist_torch(ft1, ft1)

        dist_14 = pdist_torch(ft1, ft4)
        dist_24 = pdist_torch(ft2, ft4)
        dist_44 = pdist_torch(ft4, ft4)
        dist_22 = pdist_torch(ft2, ft2)

        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())
        
        dist_ap_123, dist_an_123, dist_ap_124, dist_an_124, dist_an_33, dist_an_44, dist_an_11, dist_an_22 = [], [], [], [], [], [], [], []
        for i in range(n):
            dist_ap_123.append(dist_23[i][mask[i]].max().unsqueeze(0))
            dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            dist_an_33.append(dist_33[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_11.append(dist_11[i][mask[i] == 0].min().unsqueeze(0))

            dist_ap_124.append(dist_14[i][mask[i]].max().unsqueeze(0))
            dist_an_124.append(dist_24[i][mask[i]].min().unsqueeze(0))
            dist_an_44.append(dist_44[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_22.append(dist_22[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap_123 = torch.cat(dist_ap_123)
        dist_an_123 = torch.cat(dist_an_123).detach()
        dist_an_33 = torch.cat(dist_an_33)
        dist_an_11 = torch.cat(dist_an_11)

        dist_ap_124 = torch.cat(dist_ap_124)
        dist_an_124 = torch.cat(dist_an_124).detach()
        dist_an_44 = torch.cat(dist_an_44)
        dist_an_22 = torch.cat(dist_an_22)

        loss_123 = self.ranking_loss(dist_an_123, dist_ap_123, torch.ones_like(dist_an_123)) + (self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_an_33)) + self.ranking_loss(dist_an_11, dist_ap_123, torch.ones_like(dist_an_33))) * 0.5
        loss_124 = self.ranking_loss(dist_an_124, dist_ap_124, torch.ones_like(dist_an_124)) + (self.ranking_loss(dist_an_44, dist_ap_124, torch.ones_like(dist_an_44)) + self.ranking_loss(dist_an_22, dist_ap_124, torch.ones_like(dist_an_44))) * 0.5
        return (loss_123 + loss_124)/2

def compute_sdmo(i_feats,t_feats,feats, pid, logit_scale, image_id=None, factor=0.1, epsilon=1e-8, temperature=0.3):
    """
    Similarity Distribution Matching //scanloss
    """
    # TODO dee+convd layer // +softmax 
    # i_fetures = F.softmax(i_fetures,dim=1)
    # t_fetures = F.softmax(t_fetures,dim=1)
    i_feats = torch.mul(i_feats,F.softmax(i_feats, dim=1))
    t_feats = torch.mul(t_feats, F.softmax(t_feats, dim=1))
    z1 = F.normalize(i_feats,dim=1)
    z2 = F.normalize(t_feats,dim=1)
    # z1 = F.normalize(F.softmax(i_feats,dim=1))
    # z2 = F.normalize(F.softmax(t_feats,dim=1))
    N, Z = z1.shape
    # N, Z = z1.size(0), z1.size(1)
 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)




def compute_sdm(ifeat, t_feats, pid, logit_scale, image_id=None, factor=0.1, epsilon=1e-8, temperature=0.3):

    ifeat = torch.mul(ifeat,F.softmax(ifeat, dim=1))
    t_feats = torch.mul(t_feats, F.softmax(t_feats, dim=1))
    z1 = F.normalize(ifeat,dim=1)
    z2 = F.normalize(t_feats,dim=1)
    N, Z = z1.shape
    # print("N2:", N)
    # print("z1 shape2:", z1.shape)

    device = z1.device 
    
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    M = similarity_matrix.size(0)  // 2  # 获取实际单批次大小
    l_pos = torch.diag(similarity_matrix, M)
    r_pos = torch.diag(similarity_matrix, -M)
    positives = torch.cat([l_pos, r_pos]).view(2 * M, 1)

    diag = torch.eye(M, dtype=torch.bool, device=device)  # 创建单批次对角线掩码
    diag = torch.block_diag(diag, diag)  # 创建双批次对角线掩码，对应 z1 和 z2 的组合
    negatives = similarity_matrix[~diag].view(2*M, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*M, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * M)

def compute_itc(x_pool, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = x_pool.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(x_pool.device)

    
    # normalized features
    image_norm = x_pool / x_pool.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def xpool(sims, logit_scale1):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        # print(sims.shape)
        # print(logit_scale1.shape)
        logits = sims * logit_scale1
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

def CTLoss(output, labels): 
    batch_size = output.size(0) 
    # print("labels",labels.shape)
    # print("batch_size",batch_size)
    # print("output",output.shape)

    targets_expand = labels.view(batch_size, 1).expand(batch_size, output.size(1)) 
    centers_batch = nn.Parameter(torch.randn(128, 128, device=output.device), requires_grad=True).gather(0, targets_expand) 
    centers_batch_expanded = centers_batch.unsqueeze(2).unsqueeze(2)

    centers_batch_bz = torch.stack([centers_batch_expanded]*batch_size)
    inputs_bz = torch.stack([output]*batch_size).transpose(0, 1)
    dist = torch.sum((centers_batch_bz - inputs_bz)**2, 2).squeeze()
    dist = dist.clamp(min=1e-12).sqrt()  
    mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
    dist_ap, dist_an = [], [] 
    for i in range(batch_size): 
        dist_ap.append(dist[i][mask[i]].max()) 
        dist_an.append(dist[i][mask[i]==0].min()) 
    # 在连接之前检查列表，如果有零维张量，则将其转换为具有一个维度的张量
    dist_ap = [item.view(1) if item.dim() == 0 else item for item in dist_ap]

    dist_ap = torch.cat(dist_ap)
    # 在连接之前检查列表，如果有零维张量，则将其转换为具有一个维度的张量
    dist_an = [item.view(1) if item.dim() == 0 else item for item in dist_an]
    dist_an = torch.cat(dist_an)
    y = torch.empty_like(dist_an).fill_(1)

    dist_an = torch.mul (dist_an, F.softmax(dist_an, dim=0))
    dist_ap = torch.mul (dist_ap, F.softmax(dist_ap, dim=0))

    loss = F.margin_ranking_loss(F.softmax(dist_an,dim=0), F.softmax(dist_ap,dim=0), y, margin=0.02)

    return loss
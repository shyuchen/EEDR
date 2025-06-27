# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .GaLR_utils import *
import copy
import ast
from .bert import Bert


#from .mca import SA,SGA

class Fusion_MIDF(nn.Module):
    # def __init__(self, high_dim, low_dim, flag):
    #     super(Fusion_MIDF, self).__init__()

    #     self.CNL = CNL(high_dim, low_dim, flag)
    #     self.PNL = PNL(high_dim, low_dim)
    # def forward(self, x, x0):
    #     z = self.CNL(x, x0)
    #     z = self.PNL(z, x0)
    #     return z
    def __init__(self, opt):
        super(Fusion_MIDF, self).__init__()
        self.opt = opt

        # local trans
        self.l2l_SA = SA(opt)

        # global trans
        self.g2g_SA = SA(opt)

        # local correction
        self.g2l_SGA = SGA(opt)

        # global supplement
        self.l2g_SGA = SGA(opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], 2),
            nn.Softmax()
        )

    def forward(self, global_feature, local_feature):

        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        visual_feature = weight_global*global_feature + weight_local*local_feature

        return visual_feature


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        # print("x",x.shape)  x torch.Size([100, 512, 1, 1])
        # print("x0",x0.shape)  x0 torch.Size([100, 512, 1, 1])
        z = self.CNL(x, x0)
        # print("z2",z.shape)  torch.Size([100, 512, 1, 1])
        z = self.PNL(z, x0)
        return z


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        # print("xh",x_h.shape) xh torch.Size([100, 512, 1, 1])
        # print("xl",x_l.shape)  xl torch.Size([100, 512, 1, 1])
        B = x_h.size(0)
        # print("b",B)  b 100
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        # print("gx",g_x.shape) gx torch.Size([100, 512, 1])

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        # print("z",z.shape)  z torch.Size([100, 512, 1, 1])
        
        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        # self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)

        # self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)

        # self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)

        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        # print("self g",self.g(x_l).shape)  torch.Size([100, 256, 1, 1])
        # print("self low",self.low_dim) 512
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        # y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        y = y.view(B, self.low_dim, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z 

class AdaptiveNoiseFilter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, img_feat, text_feat, aux_text_feat):
        sim_orig = F.cosine_similarity(img_feat, text_feat, dim=1)
        sim_aux = F.cosine_similarity(img_feat, aux_text_feat, dim=1)
        delta = torch.abs(sim_aux - sim_orig)

        combined = torch.cat([text_feat, aux_text_feat], dim=1)
        weight = torch.sigmoid(self.attn(combined)).squeeze(-1)
        refined_text_feat = weight.unsqueeze(1) * aux_text_feat + (1 - weight.unsqueeze(1)) * text_feat
        return refined_text_feat, delta

# ---------- DEGM ----------
class DEGM(nn.Module):
    def __init__(self, dim=512, K=6, steps=3):
        super().__init__()
        self.K = K
        self.steps = steps
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(K, dim))
        self.log_sigma = nn.Parameter(torch.zeros(K, dim))

        self.refine_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
            for _ in range(steps)
        ])

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, global_text_feat, word_feats):
        B = global_text_feat.shape[0]
        std = torch.exp(self.log_sigma)
        eps = torch.randn(B, self.K, self.dim, device=global_text_feat.device)
        Z = self.mu.unsqueeze(0) + std.unsqueeze(0) * eps

        for layer in self.refine_blocks:
            Z = layer(Z)

        Z_final = Z.mean(dim=1)
        text_embedding = self.mlp(Z_final + global_text_feat)
        return text_embedding

# ---------- Full Model ----------
class FullModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.visual_encoder = VisualEncoder(dim)
        self.anf = AdaptiveNoiseFilter(dim)
        self.degm = DEGM(dim)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, images, raw_text_feat, aux_text_feat, word_feats):
        img_feat = F.normalize(self.visual_encoder(images), dim=1)
        refined_text_feat, delta = self.anf(img_feat, raw_text_feat, aux_text_feat)
        text_feat = F.normalize(self.degm(refined_text_feat, word_feats), dim=1)
        img_feat = F.normalize(img_feat, dim=1)

        sim = torch.matmul(text_feat, img_feat.T) / self.temp
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss, img_feat, text_feat, delta

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()
        self.base_resnet = base_resnet(arch=arch)
        self.MFA3 = MFA_block(512, 512, 1)
        self.conv_transformation = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)



        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)
#diff
        # local feature /obj G 
        self.local_feature = GCN()
        self.drop_l_v = nn.Dropout(0.3)

        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # fusion
        self.fusion = Fusion_MIDF(opt = opt)
        # self.DEE_module = DEE_module(512)

        # weight
        self.gw = opt['global_local_weight']['global']
        self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, text_lens=None):

        # extract features
        # print("text",text.shape)
        
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
#        global_feature = solo_feature
        # extract local feature  /GCN MATHEMATICS
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion
        global_feature = torch.unsqueeze(global_feature,dim=2)
        global_feature = torch.unsqueeze(global_feature,dim=3)
        local_feature = torch.unsqueeze(local_feature,dim=2)
        local_feature = torch.unsqueeze(local_feature,dim=3)
        # print("1",global_feature.shape)
        # print("2",local_feature.shape)
        # global_feature = self.conv_transformation(global_feature)
        # local_feature = self.conv_transformation(local_feature)
        visual_feature = self.MFA3(global_feature, local_feature)
        # visual_feature = self.fusion(global_feature, local_feature)
        # print(visual_feature.shape)
        # visual_feature = torch.unsqueeze(visual_feature,dim=2)
        # visual_feature = torch.unsqueeze(visual_feature,dim=3)


        # print(visual_feature4.shape)  #[100, 512]

        # text features  /GCN MATHEMATICS
        text_feature = self.text_feature(text)


        sims = cosine_sim(visual_feature, text_feature)
        #sims = cosine_sim(self.lw*self.drop_l_v(local_feature) + self.gw*self.drop_g_v(global_feature), text_feature)
        return sims

def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model




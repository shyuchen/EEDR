# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import math
from layers import seq2vec
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, BertConfig
from . import init
# from einops import rearrange, repeat
# from .backbones.resnet import Bottleneck

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_k = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_q = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['fusion']['mca_HIDDEN_SIZE']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C['fusion']['mca_HIDDEN_SIZE'],
            mid_size=__C['fusion']['mca_FF_SIZE'],
            out_size=__C['fusion']['mca_HIDDEN_SIZE'],
            dropout_r=__C['fusion']['mca_DROPOUT_R'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout3 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm3 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

        self.pool_2x2 = nn.MaxPool2d(4)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        # print('img',img)
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # Lower Feature
        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        # Higher Feature
        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)
        # higher_feature = self.up_sample_4(higher_feature)

        # batch * 512
        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature,dim=-1))
        return lower_feature, higher_feature, solo_feature
    

    
class DEE_module(nn.Module):
    def __init__(self, channel=3, reduction=16,opt={}):
        super(DEE_module, self).__init__()
        

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)

        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)


    def forward(self, x):
        # print("1",x.shape)
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        x2 = self.FC2(F.relu(x2))
        out = x1
        # out = torch.cat((x, x1, x2), 0)
        # out = out[:100, :, :, :]

        out = self.dropout(out)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class VSA_Module(nn.Module):
    def __init__(self, opt = {}):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = opt['multiscale']['multiscale_input_channel']
        out_channels = opt['multiscale']['multiscale_output_channel']
        embed_dim = opt['embed']['embed_dim']

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=192, out_channels=channel_size, kernel_size=3, stride=4)
        self.HF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=1, stride=1)

        # visual attention
        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)

        # solo attention
        self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):

        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        # concat
        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)

        # residual
        concat_feature = higher_feature.mean(dim=1,keepdim=True).expand_as(concat_feature) + concat_feature

        # attention
        main_feature = self.conv1x1_1(concat_feature)
        attn_feature = torch.sigmoid(self.conv1x1_2(concat_feature).view(concat_feature.shape[0],1,-1)).view(concat_feature.shape[0], 1, main_feature.shape[2], main_feature.shape[3])
        atted_feature = (main_feature*attn_feature).squeeze(dim=1).view(attn_feature.shape[0], -1)

       # solo attention
        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature*solo_att

        return l2norm(solo_feature, -1)

class Skipthoughts_Embedding_Module(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text ):
        x_t_vec = self.seq2vec(input_text)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12
# ====================================================================
# About GCN

class GCN(nn.Module):
    def __init__(self , dim_in=20 , dim_out=20, dim_embed = 512):
        super(GCN,self).__init__()

        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)

        # self.bn1 = nn.BatchNorm1d(dim_in)
        # self.bn2 = nn.BatchNorm1d(dim_in // 2)
        # self.bn3 = nn.BatchNorm1d(dim_out)

        self.out = nn.Linear(dim_out * dim_in, dim_embed)

    def forward(self, A, X):
        batch, objects, rep = X.shape[0], X.shape[1], X.shape[2]

        # first layer
        tmp = (A.bmm(X)).view(-1, rep)
        X = F.relu(self.fc1(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # second layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc2(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # third layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc3(tmp))
        X = X.view(batch, -1)

        return l2norm(self.out(X), -1)
    

def _make_layer(block, inplanes, planes, blocks, stride=2):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

class HAT(nn.Module):
    def __init__(self, img_size, patch_size, in_dim, poi_dim, heads, mlp_dim, tfc_depth, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(HAT, self).__init__()

        ###################### parameters
        T_depth1, T_depth2, T_depth3 = tfc_depth
        inc1, inc2, inc3 = in_dim

        self.TFC_S1 = TFC(in_channel=inc1, out_channel=inc1, img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth1,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFC_S2 = TFC(in_channel=(inc1+inc2), out_channel=(inc1+inc2), img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth2,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFC_S3 = TFC(in_channel=(inc1+inc2+inc3), out_channel=poi_dim, img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth3,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)


    def forward(self, x1, x2, x3, mask=None):

        x1, x_mid_1 = self.TFC_S1(x1)
        x2, x_mid_2 = self.TFC_S2(torch.cat((x2 ,x1), dim=1))
        x3, x_mid_3 = self.TFC_S3(torch.cat((x3 ,x2), dim=1))

        return x_mid_1, x_mid_2, x_mid_3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TFC(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super(TFC, self).__init__()

        height, width = img_size

        self.p_size = p_size

        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.NeA = Bottleneck(out_channel, out_channel//4)

    def forward(self, x, mask=None):

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        
        x = self.patch_to_embedding(x)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_mid = x[:, 0]
        x_mid = self.to_latent(x_mid)
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x = self.NeA(x)

        return x, x_mid

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

        for i in range(depth):
            name = 'Non' + str(i)
            setattr(self, name, nn.Linear(2048, 2048))


    def forward(self, x, mask = None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x














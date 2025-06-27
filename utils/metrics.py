from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import time
import pandas as pd
from utils.iotools import read_json

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    # an_path='/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/data/RSICD_optimal-master/reid_raw.json'
    # annos = read_json(an_path)
    # output_t2l={'TEXT':[],'rank1':[],'rank2':[],'rank3':[],'rank4':[],'rank5':[]}
    # output_l2t={'IMG':[],'rank1':[],'rank2':[],'rank3':[],'rank4':[],'rank5':[]}
    # print(similarity.shape)
    # if similarity.shape[0]>2000: #t2i
    #     for i in range(5,30,5):
    #         output_t2l['TEXT'].append(annos[int(i/5)-1]['captions'][4])
    #         output_t2l['rank1'].append(annos[indices[i][0]]['file_path'])
    #         output_t2l['rank2'].append(annos[indices[i][1]]['file_path'])
    #         output_t2l['rank3'].append(annos[indices[i][2]]['file_path'])
    #         output_t2l['rank4'].append(annos[indices[i][3]]['file_path'])
    #         output_t2l['rank5'].append(annos[indices[i][4]]['file_path'])
            
    #     df = pd.DataFrame(output_t2l)
    #     df.to_csv('t2l.csv',columns=['TEXT', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'],index=False)
    # else: #i2t
    #     for i in range(5,30,5):
    #         output_l2t['IMG'].append(annos[i-1]['file_path'])
    #         output_l2t['rank1'].append(annos[int(indices[i-1][0]/5)]['captions'][indices[i-1][0]%5])
    #         output_l2t['rank2'].append(annos[int(indices[i-1][1]/5)]['captions'][indices[i-1][1]%5])
    #         output_l2t['rank3'].append(annos[int(indices[i-1][2]/5)]['captions'][indices[i-1][2]%5])
    #         output_l2t['rank4'].append(annos[int(indices[i-1][3]/5)]['captions'][indices[i-1][3]%5])
    #         output_l2t['rank5'].append(annos[int(indices[i-1][4]/5)]['captions'][indices[i-1][4]%5])
            
    #     df = pd.DataFrame(output_l2t)
    #     df.to_csv('l2t.csv',columns=['IMG', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'],index=False)
  
    pred_labels = g_pids[indices.cpu()]  # q * k
    
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
    
    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum

    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=True):
        # similarity time
        # start = time.time()

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()
        # similarity time
        # end = time.time()
        # print("calculate similarity time:", end - start)
        
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        print(t2i_cmc)
        print(t2i_mAP)
        print(t2i_mINP)
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]

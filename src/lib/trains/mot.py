from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, GroupSoftmaxLoss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        if opt.id_loss == "gs":
            self.gs_config=dict(
                group_num=opt.gs_group_num,
                label2binlabel=opt.gs_label2binlabel,
                pred_slice=opt.gs_pred_slice,
                id_newid=opt.gs_id_newid
            )
            
            if "/MOT20_train_correction/" in opt.gs_id_newid:
                newid_number = 2215
                print("yes  newid_number = 2215")
            elif "/MOT20_train_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 2215              
            elif "/mot20_add_our_method_correction_no_strength/" in opt.gs_id_newid:
                newid_number = 2215
            elif "/MOT20_train_DVA_GS_100_1000_correction/" in opt.gs_id_newid:
                newid_number = 2215
            elif "/MOT20_train_SDA_DVA_GS_100_1000_correction/" in opt.gs_id_newid:
                newid_number = 2215
            elif "/MOT20_train_LastQuarter_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 971
            elif "/MOT20_train_LastTwoQuarters_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 1418
            elif "/MOT20_train_LastThreeQuarters_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 1830
            elif "/MOT17_train_half_strengthened_before_correction/" in opt.gs_id_newid:
                newid_number = 359 
            elif "/MOT17_train_half_strengthened_after_correction/" in opt.gs_id_newid:
                newid_number = 359
            elif "/MOT17_train_half_after_gs_10_100_correction/" in opt.gs_id_newid:
                newid_number = 359
            elif "/MOT17_train_correction/" in opt.gs_id_newid:
                newid_number = 546
            elif "/MOT17_train_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 546
            elif "/MOT16_train_correction/" in opt.gs_id_newid:
                newid_number = 517
            elif "/MOT16_train_DCVDA_GS_correction/" in opt.gs_id_newid:
                newid_number = 517
            elif "/mot16_add_our_method_correction_no_strength/" in opt.gs_id_newid:
                newid_number = 517
            elif "/MOT15_train_correction/" in opt.gs_id_newid:
                newid_number = 501
            else:
                print("the input gs_id_newid is a error")

            self.nID = newid_number + opt.gs_group_num
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            self.GroupSoftmaxCELoss = GroupSoftmaxLoss(gs_config=self.gs_config)
        else:
            self.nID = opt.nID
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

        self.emb_dim = opt.reid_dim
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))




    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / id_output.size(0)
                else:
                    if self.opt.id_loss == 'gs':
                        id_loss += self.GroupSoftmaxCELoss.loss(IDLoss=self.IDLoss, cls_score=id_output, labels=id_target)
                    else:
                        id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats, id_target

class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

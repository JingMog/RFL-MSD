from __future__ import absolute_import
import logging
logger = logging.getLogger()
import numpy
from collections import namedtuple

from .my_encode import Encoder
from .my_decode import SequenceGenerator, PreDecoder
from .metric import EvalMetric, GenTranMetric
from . import xconfig

from picklable_itertools.extras import equizip
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn import Parameter
from torch.cuda.amp import autocast
import torch.jit as jit
import numpy as np
import math


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self._decoder_max_seq_len = xconfig.decoder_max_seq_len

        self._encoder = Encoder()
        self._pre_decoder = PreDecoder()
        self._decoder = SequenceGenerator()
        self._eval_metrics = GenTranMetric()

    def encoder_forward(self, source, source_mask, mem_index_data):
        encoded, encoded_mask = self._encoder(source, source_mask) # [b, 384, h, w]
        encoded_proj, enc_init_states, zero_init_states = self._pre_decoder(encoded, encoded_mask, mem_index_data)
        return encoded, encoded_mask, encoded_proj, enc_init_states, zero_init_states
        
    def decoder_forward(self, encoded, encoded_mask, encoded_proj, target, target_mask, target_branch, cond_data, mem_index_data, bond_index_data, mem_used_mask, mem_update_info, branch_update_info, bond_update_info, init_states):
        # encoded = DebugViewer(encoded)
        sm = self._decoder(encoded, encoded_mask, encoded_proj, target, target_mask, target_branch, cond_data, mem_index_data, bond_index_data, mem_used_mask, mem_update_info, branch_update_info, bond_update_info, init_states) # [loss, mem_loss] + states + [readout, mem_readout] + [target_branch, mem_cls_mask]
        # encoded, encoded_mask, encoded_proj, target_label, target_mask, init_states
        output = sm[-4:]
        sm = sm[0:3]
        return sm, output
        
    def update_metric(self, preds, label):
        for eval_metric in self._eval_metrics:
            eval_metric.update(label, preds)

    def forward(self, data_dict):
        data = data_dict["data"]
        data_mask = data_dict["data_mask"]
        target = data_dict["target"] #[B, L]
        target_mask = data_dict["target_mask"] #[B, L]
        cond_data = data_dict["cond_data"] #[B, L]
        mem_index_data = data_dict["mem_index_data"] #[B, M]
        mem_used_mask = data_dict["mem_used_mask"] #[B, L, M]
        mem_update_info = data_dict["mem_update_info"] #[B, L]
        target_branch = data_dict["branch_label"] # [B, L, L]
        branch_update_info = data_dict["branch_update_info"] # [B, L]
        bond_index_data = data_dict["bond_index_data"] # [B, l_bond]
        bond_update_info = data_dict["bond_update_info"] #[B, L]
        # self.eval() # only eval mode
        encoded, encoded_mask, encoded_proj, enc_init_states, zero_init_states = self.encoder_forward(data, data_mask, mem_index_data)
        init_states = zero_init_states + enc_init_states
        
        sm, output = self.decoder_forward(encoded, encoded_mask, encoded_proj, target, target_mask, target_branch, cond_data, mem_index_data, bond_index_data, mem_used_mask, mem_update_info, branch_update_info, bond_update_info, init_states)
        return sm + output

    def forward_backward(self, data_dict):

        # data, data_mask, target_ori, target_mask_ori
        data = data_dict["data"]
        data_mask = data_dict["data_mask"]
        target_ori = data_dict["target"] #[B, L]
        target_mask_ori = data_dict["target_mask"] #[B, L]
        cond_data_ori = data_dict["cond_data"] #[B, L]
        mem_index_data = data_dict["mem_index_data"] #[B, M]
        mem_used_mask = data_dict["mem_used_mask"] #[B, L, M]
        mem_update_info = data_dict["mem_update_info"] #[B, L]
        target_branch_ori = data_dict["branch_label"] # [B, l_branch, 2]
        branch_update_info = data_dict["branch_update_info"] # [B, L]
        bond_index_data = data_dict["bond_index_data"] # [B, l_bond]
        bond_update_info = data_dict["bond_update_info"] #[B, L]
        
        # encoder forward and initialize states
        encoded_ori, encoded_mask_ori, encoded_proj_ori, enc_init_states_ori, zero_init_states = self.encoder_forward(data, data_mask, mem_index_data)
        encoder_out = torch.cat([encoded_ori.view(-1), encoded_mask_ori.view(-1), encoded_proj_ori.view(-1)] + [state_ori.view(-1) for state_ori in enc_init_states_ori])
        
        # detach decoder with encoder
        encoded = encoded_ori.detach().clone().requires_grad_(True)
        encoded_mask = encoded_mask_ori.detach().clone().requires_grad_(True)
        encoded_proj = encoded_proj_ori.detach().clone().requires_grad_(True)
        init_states = zero_init_states + [state.detach().clone().requires_grad_(True) for state in enc_init_states_ori]
        
        # get the first truncated target
        _, whole_length = target_ori.shape
        end_pos = min(whole_length, 0 + self._decoder_max_seq_len)
        target = target_ori[:, :end_pos]
        target_mask = target_mask_ori[:, :end_pos]
        cond_data = cond_data_ori[:, :end_pos]
        mem_used_mask = mem_used_mask[:, :end_pos, :]
        mem_update_info = mem_update_info[:, :end_pos]
        branch_update_info = branch_update_info[:, :end_pos]
        bond_update_info = bond_update_info[:, :end_pos]
        
        target_branch = target_branch_ori
        bond_index_data = bond_index_data
        # forward and backward decoder once
        # 带有记忆模块的条件注意力机制
        decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, target, target_mask, target_branch, cond_data, mem_index_data, bond_index_data, mem_used_mask, mem_update_info, branch_update_info, bond_update_info, init_states)
        self.update_metric(decoder_outputs + preds, [target, target_branch, target_mask]) # 根据当前预测结果target, target_branch, target_mask更新评价指标
        loss, mem_loss, _ = decoder_outputs
        # decoder_outputs[0].backward() #
        total_loss = loss + mem_loss
        total_loss.backward()

        # calculate encode grad
        # import pdb; pdb.set_trace()
        encode_grad = torch.cat([encoded.grad.view(-1), encoded_mask.grad.view(-1), encoded_proj.grad.view(-1)] + 
                                 [state.grad.view(-1) for state in init_states[-len(enc_init_states_ori):]])
        
        # clear decoder input grad
        encoded.grad.detach_();encoded.grad.zero_()
        encoded_mask.grad.detach_();encoded_mask.grad.zero_()
        encoded_proj.grad.detach_();encoded_proj.grad.zero_()
        for state in init_states[-len(enc_init_states_ori):]:
            state.grad.detach_();state.grad.zero_()

        # truncated bptt start TBPTT算法, 梯度裁剪
        bptt_counter = 1
        while end_pos < whole_length:
            bptt_counter += 1

            start_pos = end_pos
            end_pos = min(whole_length, start_pos + self._decoder_max_seq_len)

            target = target_ori[:,start_pos:end_pos]
            target_mask = target_mask_ori[:,start_pos:end_pos]
            decoder_states = decoder_outputs[1:]
            decoder_states = [torch.from_numpy(state.cpu().data.numpy()).cuda() for state in decoder_states]
            decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, 
                                                          target, target_mask, decoder_states)  
            self.update_metric(preds, [target, target_mask])
            decoder_outputs[0].backward()
            encode_grad += torch.cat([encoded.grad.view(-1), encoded_mask.grad.view(-1), encoded_proj.grad.view(-1)] + [state.grad.view(-1) for state in init_states[-len(enc_init_states_ori):]])

            # clear decoder input grad
            encoded.grad.detach_();encoded.grad.zero_()
            encoded_mask.grad.detach_();encoded_mask.grad.zero_()
            encoded_proj.grad.detach_();encoded_proj.grad.zero_()

        # encode backward
        encoder_out.backward(encode_grad)

        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

        check_grad = True
        for param in self.parameters():

            if math.isnan(param.data.sum().cpu().item()):
                param.data.zero_()

            if param.grad is None:
                continue
                
            if math.isnan(param.grad.data.sum().cpu().item()):
                logger.info('gradient error')
                check_grad = False
                break
            param.grad.data.div_(bptt_counter)

        return check_grad

    def forward_backward_amp(self, data, data_mask, target_ori, target_mask_ori, scaler, optimizer):
        # encoder forward and initialize states
        with autocast():
            encoded_ori, encoded_mask_ori, encoded_proj_ori, enc_init_states_ori, zero_init_states = self.encoder_forward(data, data_mask)
        encoder_out = torch.cat([encoded_ori.view(-1), encoded_mask_ori.view(-1), 
                                 encoded_proj_ori.view(-1)] + 
                                 [state_ori.view(-1) for state_ori in enc_init_states_ori])

        # detach decoder with encoder
        encoded = encoded_ori.float().detach().clone().requires_grad_(True)
        encoded_mask = encoded_mask_ori.float().detach().clone().requires_grad_(True)
        encoded_proj = encoded_proj_ori.float().detach().clone().requires_grad_(True)
        init_states = zero_init_states + [state.float().detach().clone().requires_grad_(True) for state in enc_init_states_ori]

        # get the first truncated target
        _, whole_length = target_ori.shape
        end_pos = min(whole_length, 0 + self._decoder_max_seq_len)
        target = target_ori[:, :end_pos]
        target_mask = target_mask_ori[:, :end_pos]

        # forward and backward decoder once

        decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, target, target_mask, init_states)
        # decoder_outputs[0].backward()
        # with autocast():
        #     decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, target, target_mask, init_states)
        scaler.scale(decoder_outputs[0]).backward()

        self.update_metric(preds, [target, target_mask])

        # calculate encode grad
        # import pdb; pdb.set_trace()
        encode_grad = torch.cat([encoded.grad.view(-1), encoded_mask.grad.view(-1), encoded_proj.grad.view(-1)] + 
                                 [state.grad.view(-1) for state in init_states[-len(enc_init_states_ori):]])

        
        # clear decoder input grad
        encoded.grad.detach_();encoded.grad.zero_()
        encoded_mask.grad.detach_();encoded_mask.grad.zero_()
        encoded_proj.grad.detach_();encoded_proj.grad.zero_()
        for state in init_states[-len(enc_init_states_ori):]:
            state.grad.detach_();state.grad.zero_()

        # truncated bptt start
        bptt_counter = 1
        while end_pos < whole_length:
            bptt_counter += 1

            start_pos = end_pos
            end_pos = min(whole_length, start_pos + self._decoder_max_seq_len)

            target = target_ori[:,start_pos:end_pos]
            target_mask = target_mask_ori[:,start_pos:end_pos]
            decoder_states = decoder_outputs[1:]
            decoder_states = [state.float().detach().clone() for state in decoder_states]

            decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, 
                                                        target, target_mask, decoder_states)
            # decoder_outputs[0].backward()  
            # with autocast():
            #     decoder_outputs, preds = self.decoder_forward(encoded, encoded_mask, encoded_proj, 
            #                                                 target, target_mask, decoder_states)
            scaler.scale(decoder_outputs[0]).backward()  
            self.update_metric(preds, [target, target_mask])
            encode_grad += torch.cat([encoded.grad.view(-1), encoded_mask.grad.view(-1), encoded_proj.grad.view(-1)]+
                                       [state.grad.view(-1) for state in init_states[-len(enc_init_states_ori):]])

            # clear decoder input grad
            encoded.grad.detach_();encoded.grad.zero_()
            encoded_mask.grad.detach_();encoded_mask.grad.zero_()
            encoded_proj.grad.detach_();encoded_proj.grad.zero_()

        # encode backward
        # scaler.scale(encoder_out).backward(encode_grad)
        encoder_out.backward(encode_grad)
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

        check_grad = True
        for param in self.parameters():

            if math.isnan(param.data.sum().cpu().item()):
                param.data.zero_()

            self._check_fix_nan(param)
            self._check_clip_inf_gradents(param, scaler)

            if math.isnan(param.grad.data.sum().cpu().item()):
                logger.info('gradient error')
                check_grad = False
                break
            param.grad.data.div_(bptt_counter)

        return check_grad

    def _check_clip_inf_gradents(self, param, scaler, clip_value = 10):
        # check and fix inf grads
        clip_value = clip_value * scaler._scale[0].detach().item()
        is_inf_mask = param.grad.data.isinf().float()
        param.grad.data = param.grad.data.clip(-clip_value, clip_value)
        if is_inf_mask.sum() > 0:            
            mean_value = (param.grad.data * (1 - is_inf_mask)).mean()
            param.grad.data = param.grad.data * (1 - is_inf_mask) + (is_inf_mask * param.grad.data) * mean_value / clip_value

    def _check_fix_nan(self, param):
        is_nan_mask = param.grad.data.isnan()
        if is_nan_mask.sum() > 0:
            param.grad.data[is_nan_mask] = 0.
        
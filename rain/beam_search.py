# -*- coding: utf-8 -*- 
from __future__ import absolute_import
import logging
import os
logger = logging.getLogger()

import numpy
import torch
import torch.nn as nn
from picklable_itertools.extras import equizip

from . import xconfig
from .my_encode import Encoder
from .my_decode import SequenceGenerator, PreDecoder
import pdb
import random
import warnings
import torch.nn.functional as F
from .RFL_rain import cs_main, chemstem2chemfig


class BeamSearcher(nn.Module):
    def __init__(self, vocab_size, sos=0, eos=1, beam=3, frame_per_char=8):
        super(BeamSearcher, self).__init__()
        self._vocab_size     = vocab_size
        self._sos            = sos
        self._eos            = eos
        self._beam           = beam
        self._frame_per_char = frame_per_char
        self._encoder        = Encoder()
        self._pre_decoder    = PreDecoder()
        self._decoder        = SequenceGenerator()
        self._ea_idx = xconfig.vocab.getID("<ea>")
        self._chem_end_idx = xconfig.vocab.getID("}")
        self._end_flag = xconfig.vocab.getID("\\\\")
        self._eol_flag = xconfig.vocab.getID("\\eol")
        self._conn_flag = xconfig.vocab.getID("\\connbranch")
        
    def topk(self, input, index, cur_beam_size):
        #input [B, beam*V]
        batch_size = input.shape[0]
        linespace_data = torch.linspace(0, (cur_beam_size*(batch_size-1)), batch_size, device=input.device).view(batch_size,1).long() # {0, beam, beam*2, beam*3, ..., beam*(B-1)}
        if index == 0:
            input = input.view(input.shape[0], self._beam, -1) #[batch, beam, V]
            input = input[:, 0, :] #[batch, V] # 第一个beam
            if self._beam > input.shape[1]: #beam  > V
                tmp = torch.zeros(input.shape[0], self._beam, device=input.device)
                tmp[:, :self._beam] = input
                tmp[:, self._beam:] = input[:,0].view(-1,1) # copy first results
                input = tmp
            _, outputs = torch.topk(input, self._beam, sorted=True) # 取input在-1维度的top_beam
            out_inds = linespace_data.repeat(1, self._beam) #[batch, beam] 
        else:
            _, out_inds = torch.topk(input, self._beam, sorted=True) #out_inds [B, beam]
            outputs  = out_inds % self._vocab_size #[B, beam, ]
            out_inds = out_inds // self._vocab_size + linespace_data #[B, beam]

        out_inds  = out_inds.view(-1)#[Bb]
        outputs   = outputs.view(1,-1)#[1, Bb]
        return outputs, out_inds

    def res2list(self, outputs, masks, costs):
        outputs = [list(output[:int(mask.sum())]) for output, mask in equizip(outputs.T, masks.T)]
        costs = list(costs.sum(axis=0) / masks.sum(axis=0) * -1)
        beam_batch_size = len(costs)
        batch_size = beam_batch_size // self._beam
        indices = numpy.zeros((beam_batch_size,), dtype='int64')
        costs_batch = []
        outputs_batch = []
        for i in range(batch_size):
            start = i * self._beam
            end = (i + 1) * self._beam
            indices[start:end] = numpy.argsort(costs[start:end]) + start
            costs_ = [costs[i] for i in indices[start:end]]
            outputs_ = [outputs[i] for i in indices[start:end]]
            costs_batch.append(costs_)
            outputs_batch.append(outputs_)
        return outputs_batch, costs_batch


    def res2chemfiglist(self, outputs, masks, costs, in_conds, branch_output, batch_size=1, is_show=False):
        '''将解码结果转换为cs_list, 再根据cs_list转换为chemfig'''
        length = outputs.shape[0]
        beam_batch_size = outputs.shape[1]
        beam = outputs.shape[1] // batch_size
        
        outputs = outputs[1:] #[L-1, batch*beam] # 去掉开始的<s>
        masks = masks[:-1] #[L-1, batch*beam] #
        costs = (costs[1:]-costs[:-1])*masks#[L-1, batch*beam]
        branch_output = branch_output[1:] # [L-1, batch*beam, l_branch] # (time_t, bond_type)
        in_conds = in_conds[1:] # [L-1, batch*beam]
        outputs = [output[:int(mask.sum())].tolist() for output, mask in equizip(outputs.T, masks.T)]
        costs = (costs.sum(axis=0) / masks.sum(axis=0) * -1).tolist()
        indices = numpy.zeros((beam_batch_size,), dtype='int64')
        
        costs_batch = []
        outputs_batch = []
        cs_string_batch = []
        for i in range(batch_size):
            start = i * self._beam
            end = (i + 1) * self._beam
            indices[start:end] = numpy.argsort(costs[start:end]) + start
            cur_batch_outputs = []
            cur_cs_string_outputs = []
            for bid in indices[start: end]:
                words = [xconfig.vocab.getWord(int(x)) for x in outputs[bid]]
                word_str = " ".join(words)
                cur_branch_info = []
                for item in branch_output[:, bid]:
                    if -1 not in item:
                        cur_branch_info.append(item.cpu().tolist())
                
                cur_cond_data = in_conds[:, bid].cpu().tolist()
                cur_cond_data = [int(x) for x in cur_cond_data]
                cur_cond_data = [-1] + cur_cond_data[:len(cur_cond_data)-1] # 所有cond_dat往后推移1位

                if is_show:
                    print("word str: ", word_str)
                    print("branch info: ", cur_branch_info)
                
                cur_branch_output = [None] * len(words)
                
                # 检验connbranch与branch_info是否对应
                if words.count('\connbranch') != len(cur_branch_info):
                    raise ValueError("branch_info 预测有误")
                
                for tmp_id, (branch_id, bond_id) in enumerate(cur_branch_info):
                    branch_id = int(branch_id)
                    bond_id = int(bond_id)
                    if cur_branch_output[branch_id] is None:
                        cur_branch_output[branch_id] = [bond_id]
                    else:
                        cur_branch_output[branch_id].append(bond_id)
                # CS -> chemfig
                try:
                    generate_chemfig = chemstem2chemfig(words, [None]*len(words), cur_branch_output, cur_cond_data)
                    generate_chemfig = generate_chemfig.replace('(', 'branch(').replace(')', 'branch)')
                    if is_show:
                        print("成功转化!")
                except Exception as e:
                    if is_show:
                        print(e)
                    # try:
                    #     generate_chemfig = chemstem2chemfig(words, [None]*len(words), cur_branch_output, cur_cond_data)
                    # except:
                    #     print()
                    generate_chemfig = " ".join(words)
                    generate_chemfig = generate_chemfig.replace('(', 'branch(').replace(')', 'branch)')
                generate_chemfig = generate_chemfig.split(' ')
                cur_batch_outputs.append(generate_chemfig)
                cur_cs_string_outputs.append(words)
            costs_ = [costs[i] for i in indices[start:end]]
            costs_batch.append(costs_)
            outputs_batch.append(cur_batch_outputs)
            cs_string_batch.append(cur_cs_string_outputs)

        return outputs_batch, costs_batch, cs_string_batch

    def encoder_forward(self, source, source_mask):
        encoded, encoded_mask = self._encoder(source, source_mask)
        encoded_proj, enc_init_states, zero_init_states = self._pre_decoder(encoded, encoded_mask)
        # repeat
        batch_size, _, pool_h, pool_w = encoded.shape
        beam_batch_size = batch_size * self._beam
        encoded = encoded.unsqueeze(1).repeat(1,self._beam,1,1,1).view(beam_batch_size,-1,pool_h,pool_w) #[batch*beam, c, h, w]
        encoded_proj = encoded_proj.unsqueeze(1).repeat(1,self._beam,1,1,1).view(beam_batch_size,-1,pool_h,pool_w)
        encoded_mask = encoded_mask.unsqueeze(1).repeat(1,self._beam,1,1,1).view(beam_batch_size,-1,pool_h,pool_w)
        init_states = []
        for state in zero_init_states + enc_init_states:
            shape = state.shape
            new_shape = [beam_batch_size] + list(shape[1:])
            repeats = [1,self._beam] + [1]* (len(shape)-1)
            state = state.unsqueeze(1).repeat(*repeats).view(*new_shape)
            init_states.append(state)

        return encoded, encoded_mask, encoded_proj, init_states

    @torch.no_grad()
    def expand_state_dim(self, state, num=3):
        old_shape = list(state.shape)
        new_state = state.unsqueeze(1)
        new_shape = list(new_state.shape)
        repeat_num = [1 for _ in new_shape]
        repeat_num[1] = num
        repeat_num = tuple(repeat_num)
        new_state = new_state.repeat(repeat_num)
        old_shape[0] = old_shape[0] * num
        return new_state.view(tuple(old_shape))
    
    @torch.no_grad()
    def is_chemfig_end(self, outputs, masks):
        # outputs [L, Bb]
        length, batch = outputs.shape
        chemfig_end_mask = torch.zeros((batch,), device=outputs.device)
        for i in range(batch):
            if masks[-1, i] == 0:
                continue
            else:
                indexs = outputs[:, i].cpu().tolist()
                words = [xconfig.vocab.getWord(x) for x in indexs]
                cur_ind = len(words) -1
                if words[cur_ind] == "}":
                    curLevel = 1
                    cur_ind -= 1
                    while cur_ind >= 0:
                        if words[cur_ind] == "{":
                            curLevel -= 1
                        elif words[cur_ind] == "}":
                            curLevel += 1
                        if curLevel == 0:
                            if cur_ind > 1 and words[cur_ind-1] == "\\chemfig":
                                chemfig_end_mask[i] = 1
                                break
                        cur_ind -= 1
                    pass
                else:
                    continue
        return chemfig_end_mask

    @torch.no_grad()
    def find_cur_chemfig_mem(self, cur_output):
        '''find cur chemfig mem'''
        # print(cur_output)
        cur_mem_index = [-1]
        for index in range(len(cur_output)-1, 0, -1): # 逆序遍历
            unit = cur_output[index]
            unit = xconfig.vocab.getWord(unit.cpu().item())
            if unit == '\chemfig':
                break
            if '@' in unit:
                cur_mem_index.append(index-1) # 去掉<s>
        return cur_mem_index


    @torch.no_grad()
    def find_cand(self, outputs, masks, mem_index, mem_used_mask):
        '''find super token'''
        length, batch = outputs.shape
        #mem_index [B, M]
        cur_cand = torch.zeros((batch, 1), device=outputs.device)
        cur_cand_mask = torch.zeros((batch, 1), device=outputs.device)
        for i in range(batch):
            if masks[-1, i] == 0:
                continue
            
            indexs = outputs[1:, i].cpu().tolist()
            words = [xconfig.vocab.getWord(x) for x in indexs]
            cur_ind = len(words) - 1
            if words[cur_ind] != "<ea>":
                continue
            cur_ind -= 1
            cur_mem_token = {}
            cand_num = 0
            for ind in range(cur_ind, 0, -1): # 逆序遍历
                if words[ind] == '\chemfig':
                    break
                elif '@' in words[ind] or '\Superatom' in words[ind]:
                    if '@' in words[ind] and words[ind] in cur_mem_token.keys(): # superbond如果在骨干中出现,在环中再次出现则skip
                        # print("重复superbond, 清除")
                        cur_mem_token.pop(words[ind])
                        continue
                    mids = (mem_index[i] == ind).nonzero().squeeze(1).long().cpu().tolist() # 检验和memindex是否匹配
                    if len(mids) != 1:
                        break
                    # 判断当前memory是否用过
                    mid = mids[0] # mem index
                    cur_mem_token[words[ind]] = mid
                    if not mem_used_mask[i][mid]: # 用过的memory就不用pad
                        continue
            
            cand_num = 0
            for cur_len, (token, mid) in enumerate(cur_mem_token.items()):
                if mem_used_mask[i][mid]:
                    cur_len = cur_cand_mask[i].sum().long().cpu().item()
                    cur_capaticy = cur_cand.shape[1]
                    delta = cur_len + 1 - cur_capaticy
                    cur_cand = F.pad(cur_cand, (0,delta))
                    cur_cand_mask = F.pad(cur_cand_mask, (0,delta))

                    # if cand_num >= 1:
                    #     print(cand_num)
                    cur_cand[i, cand_num] = mid
                    cur_cand_mask[i, cand_num] = 1
                    cand_num += 1
                    
        return cur_cand, cur_cand_mask

    
    @torch.no_grad()
    def search_gpu(self, data, data_mask, args, names_list=None, is_show=False):
        is_show = is_show
        # state: [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]cur_memory_weight, [6]cur_state, [7]context
        encoded, encoded_mask, encoded_proj, pre_states = self.encoder_forward(data, data_mask) # encoded #[batch*beam, c, h, w]
        batch_size, _, src_height, src_width = data.shape
        beam_batch_size = self._beam * batch_size
        cur_memory = pre_states[3] #[batch*beam, l_M, query_dim] # initial M=1
        cur_memory_mask = pre_states[4] #[batch*beam, l_M]
        curM = cur_memory.shape[1] # l_m
        query_dim = cur_memory.shape[2] # query_dim

        _, _, ph, pw = encoded.shape
        max_char_num = int(src_height * src_width // (self._frame_per_char / 4) // self._frame_per_char)
        max_char_num = max_char_num * 100
        
        decode_states = {}
        decode_states["memory_used_mask"] = torch.zeros_like(cur_memory_mask, device=data.device) #[batch*beam, M]
        decode_states["mem_index"] = -torch.ones((beam_batch_size, curM), device=data.device) #[batch*beam, M]
        decode_states["cond_mem"] = torch.zeros((beam_batch_size, query_dim), device=data.device) #[batch*beam, query_dim]
        decode_states["cond_weight"] = torch.zeros((beam_batch_size, 2, ph, pw), device=data.device) #[batch*beam, 2, ph, pw],
        decode_states["mem_cls_cost"] = torch.zeros((beam_batch_size, max_char_num), device=data.device)#[batch*beam, l_tgt], 内存分类cost
        decode_states["mem_cls_res"] = -torch.ones((beam_batch_size, max_char_num), device=data.device)#[batch*beam, l_tgt], 内存分类结果
        decode_states["bond_index"] = -torch.ones((beam_batch_size, 1), device=data.device) # [B, l_bond]
        decode_states["selected_bonds"] = torch.zeros((beam_batch_size, 1, query_dim), device=data.device)
        decode_states["bonds_mask"] = torch.ones((beam_batch_size, 1), device=data.device)
        decode_states["branch_index"] = -torch.ones((beam_batch_size, 1), device=data.device) # [B, l_branch]
        decode_states["selected_branchs"] = torch.zeros((beam_batch_size, 1, query_dim), device=data.device) # [B, l_branch, query_dim]
        decode_states["branchs_mask"] = torch.ones((beam_batch_size, 1), device=data.device)

        batch_beam_mask = torch.ones((beam_batch_size), device=data.device) #[batch*beam]
        
        # initialize outputs
        # self._linspace_data = torch.linspace(0, (self._beam*(batch_size-1)), batch_size, device=data.device).view(batch_size,1).long() #把所有的beambatch分割成batch个向量, {0, beam, beam*2, beam*3, ..., beam*(B-1)},
        all_outputs = torch.ones((1, beam_batch_size), device=data.device) * self._sos #[L, batch*beam]
        all_masks   = torch.ones_like(all_outputs, device=data.device)
        all_costs   = torch.zeros_like(all_outputs, device=data.device)
        all_branch_outputs = -torch.ones((1, beam_batch_size, 2), device=data.device) #[l_branch, batch*beam, 2] #(branch_id, bond_id)
        all_cond_inputs = -torch.ones((1, beam_batch_size), device=data.device) #[L, batch*beam] #(time_t)
        all_att     = torch.zeros((1, beam_batch_size, ph, pw), device=data.device)
        all_att_cum = torch.zeros((1, beam_batch_size, ph, pw), device=data.device)

        bond_idx_tensor = self._decoder.bond_idx_tensor.to(data.device)
        conn_tensor = torch.tensor([self._conn_flag], device=data.device)
        conn_pre_token = torch.cat((bond_idx_tensor, conn_tensor), dim=0)

        # start search
        for i in range(max_char_num):
            if all_masks[-1].sum() == 0: # beam_search结束位置
                break
            log_probs, states, decode_states = self._decoder.eval_step_pred(encoded, encoded_proj, encoded_mask, pre_states, decode_states) # Attention, Readout

            # log_probs [Bb, V]
            log_probs = log_probs * batch_beam_mask.unsqueeze(1) - 1e6 * (1 - batch_beam_mask).unsqueeze(1)

            # 规则0
            # connbranch之前必须接bond, 如果不为bond, 概率赋0
            pre_connbranch_mask = (all_outputs[-1].unsqueeze(1) == conn_pre_token).sum(1)
            # 如果不为bond, connbranch, 则将branchconn概率置为0
            non_pre_connbranch_mask = torch.where(pre_connbranch_mask == 0, torch.tensor(1, device=all_outputs.device), torch.tensor(0, device=all_outputs.device)) # 将1变为0
            non_pre_connbranch_mask = non_pre_connbranch_mask.nonzero()
            log_probs[non_pre_connbranch_mask, self._conn_flag] = -1e8

            # 规则1 (暂时先不使用规则辅助,纯粹自回归解码)cur_branch_len
            # 如果y_{t-1}是<ea>,并且memory为None并且cond_mem为zero,输出'}'
            place_mask = (all_outputs[-1] == self._ea_idx) #[Bb]
            memory_empty_mask = (decode_states["memory_used_mask"].sum(1) == 0) # [Bb, l_m]
            cond_empty_mask = (decode_states["cond_mem"].sum(1) == 0) # [Bb, query_dim]
            chemfig_end_indexs = (place_mask * memory_empty_mask * cond_empty_mask).nonzero() #[Bb]
            log_probs[chemfig_end_indexs, self._chem_end_idx] = 0 # [Bb, vocab_size]

            next_costs = all_costs[-1,:,None] + log_probs * all_masks[-1,:,None] #[Bb, V]
            finished = torch.nonzero(all_masks[-1]==0)
            next_costs[finished, :self._eos] = -numpy.inf
            next_costs[finished, self._eos+1:] = -numpy.inf
            avg_costs = next_costs / all_masks.sum(axis=0)[:,None] # [Bb, V]
            avg_costs = avg_costs.view(batch_size, -1) # [B, beam*bV]
            cur_beam_size = log_probs.shape[0] // batch_size # 3
            # 选择topk
            outputs, out_inds = self.topk(avg_costs, i, cur_beam_size) #outputs [1, Bb] # out_inds [Bb]

            # select beam
            all_outputs = torch.cat([all_outputs[:, out_inds], outputs.float()])
            end_flag_mask = (all_outputs[-2:, ] == self._end_flag).sum(0, keepdim=True) < 2
            all_masks   = torch.cat([all_masks[:, out_inds], (outputs!=self._eos).float()*end_flag_mask])
            all_costs   = torch.cat([all_costs[:, out_inds], next_costs[out_inds,outputs]])
            all_cond_inputs = all_cond_inputs[:, out_inds] #[L-1, batch*beam]
            all_branch_outputs = all_branch_outputs[:, out_inds, :] # [l_branch-1, Bb, l_bond-1]
            states = [s[out_inds] for s in states] # beambatch dim must == 0
            decode_states = dict([(key, s[out_inds]) for key, s in decode_states.items()])

            y_t_1 = all_outputs[-1] #[Bb]
            
            if is_show:
                word_str = [xconfig.vocab.getWord(int(x)) for x in all_outputs[:, 0]] # beam 0
                word_str1 = [xconfig.vocab.getWord(int(x)) for x in all_outputs[:, 1]]
                word_str2 = [xconfig.vocab.getWord(int(x)) for x in all_outputs[:, 2]]
                print(i, " ", " ".join(word_str))
                print(i, " ", " ".join(word_str1))
                print(i, " ", " ".join(word_str2))
                print()
            # [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]cur_memory_weight, [6]cur_state, [7]context, [8]selected_bond, [9]bond_mask
            new_states, new_decode_states, _ = self._decoder.eval_step_update_memory(all_outputs, states, decode_states, time_t=i, is_show=is_show) # 更新memory, 并通过mem_cls_head预测branch_info
            mem_cls_res = new_decode_states["mem_cls_res"] #[Bb, 2]
            mem_cls_res = mem_cls_res.unsqueeze(0) # [1, Bb, 2]
            all_branch_outputs = torch.cat((all_branch_outputs, mem_cls_res), dim=0) # 将mem_cls_res添加到all_branch_outputs中 [L+1, Bb, 2]
            
            # 规则2
            # 如果当前chemfig结束,清空memory
            chemfig_end_mask = self.is_chemfig_end(all_outputs, all_masks) #[Bb]
            chemfig_end_indexs = chemfig_end_mask.nonzero()
            new_decode_states["memory_used_mask"][chemfig_end_indexs, :] = 0 # 清空memory

            '''
            # 暂时先按照顺序解码,不进行随机路径选择
            # RCGD->解码过程中,尝试解码memory中的所有候选分支角度并通过beamsearch计算候选得分,自动选择得分最高的路径来继续解码
            # CS-v1->首先按照顺序解码->尝试解码memory中所有Super,并通过beamsearch选择得分最高的Super
            '''
            # select branch, update cond input
            y_t_1 = all_outputs[-1] # [Bb]
            ea_mask = (y_t_1 == self._ea_idx) * all_masks[-1]
            if ea_mask.sum() > 0:
                pass
                _ = 1
            cur_cand, cur_cand_mask = self.find_cand(all_outputs, all_masks, new_decode_states["mem_index"], new_decode_states["memory_used_mask"]) # [Bb, max_cand_num], TODO:修改获取候选状态的函数
            cand_num = cur_cand_mask.sum(1) * ea_mask #[Bb] 候选数量
            max_cand_num = (cand_num * ea_mask).max() # 最大候选数量
            cur_batch_beam = y_t_1.shape[0]
            if ea_mask.sum() > 0: # 出现<ea>
                expand_beam = max_cand_num.cpu().long().item() if max_cand_num > 0 else 1
                if expand_beam > 1:
                    _ = expand_beam
                
                new_states = [self.expand_state_dim(s, expand_beam) for s in new_states]
                new_decode_states = dict([(key, self.expand_state_dim(s, expand_beam)) for key, s in new_decode_states.items()])
                cur_length = all_outputs.shape[0] # all_outputs: [L, Bb]
                all_outputs = all_outputs.unsqueeze(2).repeat(1,1,expand_beam).view(cur_length, -1) # [L, Bb*expand_beam]
                all_masks = all_masks.unsqueeze(2).repeat(1,1,expand_beam).view(cur_length, -1) # [L, Bb*expand_beam]
                all_costs = all_costs.unsqueeze(2).repeat(1,1,expand_beam).view(cur_length, -1) # [L, Bb*expand_beam]
                
                all_cond_inputs = all_cond_inputs.unsqueeze(2).repeat(1,1,expand_beam).view(cur_length -1, cur_batch_beam*expand_beam) #[L-1, Bb*expand_beam]
                cur_branch_len = all_branch_outputs.shape[2]
                all_branch_outputs = all_branch_outputs.unsqueeze(2).repeat(1,1,expand_beam, 1).view(cur_length, cur_batch_beam*expand_beam, cur_branch_len) #[L, Bb*ex, l_branch]
                
                # deal with cond mem
                old_cond_mem = new_decode_states["cond_mem"].view(cur_batch_beam, expand_beam, -1) #[Bb, expand_dim, mem_dim]
                old_cond_weight = new_decode_states["cond_weight"].view(cur_batch_beam, expand_beam, 2, ph, pw) # [Bb, ex, 2, ph, pw]
                
                next_cond_mem = torch.zeros_like(old_cond_mem)
                next_cond_weight = torch.zeros_like(old_cond_weight)
                next_cond_input = -torch.ones((cur_batch_beam, expand_beam), device=data.device) #[Bb, ex]
                batch_beam_mask =  torch.zeros((cur_batch_beam, expand_beam), device=data.device) #[cur_batch_beam, ex]

                tmp_new_memory = new_states[3] #[Bb*ex, M, mem_dim]
                _, _M, _D = tmp_new_memory.shape
                tmp_new_memory = tmp_new_memory.view(cur_batch_beam, expand_beam, _M, _D)#[Bb, ex, M, mem_dim]

                tmp_new_meory_weight = new_states[5] #[Bb*ex, M, 2, ph, pw]
                tmp_new_meory_weight = tmp_new_meory_weight.view(cur_batch_beam, expand_beam, _M, 2, ph, pw) #[Bb, ex, M, 2, ph, pw]
                
                new_memory_used_mask = new_decode_states["memory_used_mask"].clone().view(cur_batch_beam, expand_beam, _M)#[Bb, ex, M]
                new_mem_index = new_decode_states["mem_index"].view(cur_batch_beam, expand_beam, _M) #[Bb, ex, _M]

                last_cand_dict = {} # 候选字典
                blen = cur_cand.shape[0]
                for bid in range(blen):
                    _tmp_len =  cur_cand_mask[bid].sum().long().cpu()
                    cur_cands_list = cur_cand[bid, :_tmp_len].long().cpu().tolist()
                    if len(cur_cands_list) > 0:
                        last_cand_dict[bid] = cur_cands_list

                for bid in range(cur_batch_beam):
                    if ea_mask[bid] == 0:
                        batch_beam_mask[bid, 0] = 1
                        next_cond_mem[bid, 0, :] = 0
                        next_cond_weight[bid, 0, :] = 0
                    else:
                        if bid in last_cand_dict and len(last_cand_dict[bid]) > 0:
                            # 这里根据候选id给next_cond赋值
                            for cand_id, mid in enumerate(last_cand_dict[bid]):
                                batch_beam_mask[bid, cand_id] = 1
                                next_cond_mem[bid, cand_id, :] = tmp_new_memory[bid, cand_id, mid]
                                next_cond_weight[bid, cand_id, :] = tmp_new_meory_weight[bid, cand_id, mid]
                                next_cond_input[bid, cand_id] = new_mem_index[bid, cand_id, mid]
                                new_memory_used_mask[bid, cand_id, mid] = 0
                        else:
                            # 判断remain_mids是否在当前chemfig{}之前重复过
                            remain_mids = (new_memory_used_mask[bid, 0] == 1).nonzero().squeeze(1).cpu().tolist()   
                            if len(remain_mids) == 0: # we need to end this chemfig #TODO
                                batch_beam_mask[bid, 0] = 1
                                next_cond_mem[bid, 0, :] = 0
                                next_cond_weight[bid, 0, :] = 0
                                new_memory_used_mask[bid, 0, :] = 0
                            else: # random select up to expand_dim conds for next time
                                random.shuffle(remain_mids) # 随机从候选cond中挑选
                                remain_mids = remain_mids[:expand_beam]
                                # else:
                                    # remain_mids = remain_mids[-1:]
                                for cand_id, mid in enumerate(remain_mids):
                                    batch_beam_mask[bid, cand_id] = 1
                                    next_cond_mem[bid, cand_id, :] = tmp_new_memory[bid, cand_id, mid]
                                    next_cond_weight[bid, cand_id, :] = tmp_new_meory_weight[bid, cand_id, mid]
                                    next_cond_input[bid, cand_id] = new_mem_index[bid, cand_id, mid]
                                    new_memory_used_mask[bid, cand_id, mid] = 0
                new_decode_states["cond_mem"] = next_cond_mem.view(cur_batch_beam*expand_beam, -1)
                new_decode_states["cond_weight"] = next_cond_weight.view(cur_batch_beam*expand_beam, 2, ph, pw)
                new_decode_states["memory_used_mask"] = new_memory_used_mask.view(cur_batch_beam*expand_beam, -1)
                all_cond_inputs = torch.cat([all_cond_inputs, next_cond_input.view(1, cur_batch_beam*expand_beam)]) #[L, Bb*ex]
                batch_beam_mask = batch_beam_mask.view(-1)
                
            else: # 在非<ea>处保持cond不变
                batch_beam_mask = torch.ones((cur_batch_beam, ), device=data.device)
                new_decode_states["cond_mem"][:] = 0
                new_decode_states["cond_weight"][:] = 0 # 不变
                next_cond_input = -torch.ones((cur_batch_beam), device=data.device).unsqueeze(0) #[Bb, 2]
                all_cond_inputs = torch.cat([all_cond_inputs, next_cond_input]) #[L, Bb]
            
            # update rnn state for next
            pre_states, decode_states = self._decoder.eval_step_update(new_states, new_decode_states)

            # save attn weight
            if args.viz_path:
                att_yt = pre_states[0].squeeze(1).unsqueeze(0)
                att_yt_cum = pre_states[1].squeeze(1).unsqueeze(0)
                all_att = torch.cat([all_att[:, out_inds], att_yt])
                all_att_cum = torch.cat([all_att_cum[:, out_inds], att_yt_cum])  

        outputs_batch, costs_batch, cs_string_batch = self.res2chemfiglist(all_outputs, all_masks, all_costs, all_cond_inputs, all_branch_outputs, is_show=is_show)
        
        return outputs_batch, costs_batch, cs_string_batch

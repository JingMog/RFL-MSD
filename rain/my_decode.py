# -*- coding: utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from  torch.nn import Parameter
import os
import sys
import numpy as np
from . import xconfig
import pdb


class ImageAttention(nn.Module):
    def __init__(self, state_dim, encoder_dim, att_dim, chatt_dim, kernel_size, padding, cond_dim=0):
        super(ImageAttention, self).__init__()
        self.state_dim = state_dim
        self.encoder_dim = encoder_dim
        self.cond_dim = cond_dim
        self.att_dim = att_dim
        self.chatt_dim = chatt_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.energy = nn.Conv2d(self.att_dim, 1, 1, 1, padding=0, bias=False)
        self.weight_trans = nn.Conv2d(1, 1 * self.att_dim, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.cum_weight_trans = nn.Conv2d(1, 1 * self.att_dim, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)

        self.cond_weight_trans = nn.Conv2d(2, 2 * self.att_dim, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False, groups=2)
        
        self.state_trans = nn.Linear(self.state_dim, self.att_dim*1, bias=False)
        self.context_trans = nn.Linear(self.encoder_dim, self.att_dim*1, bias=False)
        self.cond_trans = nn.Linear(self.cond_dim, self.att_dim*1, bias=False)

        
    def calc_glimpses(self, concat_info, encode, encode_mask, cum_weight):
        energies = self.energy(concat_info) # va*concat_info得到eneragy向量
        energies = energies + (encode_mask - 1) * 1e8 # 被mask的energy变为负无穷,之后计算softmax时变为0
        n, c, h, w = energies.shape 
        energies = energies.reshape((n, c*h*w))

        att_weights = F.softmax(energies, dim=1) # softmax得到新的注意力权重
        n, c, h, w = encode.shape
        new_weight = att_weights.reshape((n, 1, h, w))
        new_cum_weight = cum_weight + new_weight # 更新cum_weight
        att_weights = att_weights.unsqueeze(dim=2)
        n, c, h, w = encode.shape
        encoder_data = encode.reshape((n, c, h*w))
        encoder_data = encoder_data.permute(0, 2, 1)
        ctx = encoder_data * att_weights # 根据注意力权重计算新的上下文向量context
        ctx = ctx.sum(dim=1, keepdim=False)
        return ctx, new_weight, new_cum_weight

    def forward(self, encode, encode_pro, encode_mask, state, pre_weight, cum_weight, context, cond_mem, cond_weight):
        transformed_preattention_weights = self.weight_trans(pre_weight) # [B, 128, h, w]
        transformed_cumulation_weighted = self.cum_weight_trans(cum_weight) # [B, 128, h, w]
        transformed_cond_weight = self.cond_weight_trans(cond_weight) #[B, 2*128, ph, pw]
        _b, _, ph, pw = transformed_cond_weight.shape
        transformed_cond_weight = transformed_cond_weight.view(_b, 2, self.att_dim, ph, pw).sum(1, keepdim=False) # [B, 128, H, W]

        transformed_states = self.state_trans(state) # [B, 128]
        transformed_context = self.context_trans(context) # 
        # pdb.set_trace()
        transformed_cond = self.cond_trans(cond_mem)
        state_trans = transformed_states + transformed_context + transformed_cond
        state_trans = state_trans.reshape(*state_trans.shape, 1, 1)
        temp2 = transformed_preattention_weights + transformed_cumulation_weighted + transformed_cond_weight
        new_state_trans, _ = torch.broadcast_tensors(state_trans, temp2)

        concat_info_context = new_state_trans + temp2
        
        concat_info_context = concat_info_context + encode_pro
        concat_info = torch.tanh(concat_info_context)
        ctx, new_weight, new_cum_weight = self.calc_glimpses(concat_info, encode, encode_mask, cum_weight) # 计算新的注意力权重
        return ctx, new_weight, new_cum_weight



class GRUTransition(nn.Module):
    def __init__(self, state_dim, encoder_dim, embed_dim):
        super(GRUTransition, self).__init__()
        self.state_dim = state_dim
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim

        self.output_to_state = nn.Linear(self.embed_dim, self.state_dim, bias=True)
        self.output_to_gate = nn.Linear(self.embed_dim, self.state_dim*2, bias=True)
        self.context_to_state = nn.Linear(self.encoder_dim, self.state_dim, bias=False)
        self.context_to_gate = nn.Linear(self.encoder_dim, self.state_dim*2, bias=False)
        self.state_to_gate_h2h = nn.Linear(self.state_dim, self.state_dim*2, bias=False)
        self.h2h = nn.Linear(self.state_dim, self.state_dim, bias=False)

    def forward(self, output, context, state):
        output_state = self.output_to_state(output)
        output_gate = self.output_to_gate(output)
        context_state = self.context_to_state(context)
        context_gate = self.context_to_gate(context)
        input_state = output_state + context_state
        input_gate = output_gate + context_gate
        state_gate = self.state_to_gate_h2h(state)
        gate = input_gate + state_gate
        slice_gate = gate.chunk(2, dim=1)
        update_gate = torch.sigmoid(slice_gate[0])
        reset_gate = torch.sigmoid(slice_gate[1])
        state_hat = self.h2h(state*reset_gate)
        state_hat = torch.tanh(input_state + state_hat)
        new_state = update_gate * state_hat + (1 - update_gate) * state
        return new_state

class MyGRUTransition(nn.Module):
    def __init__(self, state_dim, input_dim):
        super(MyGRUTransition, self).__init__()
        self.state_dim = state_dim
        # self.encoder_dim = encoder_dim
        # self.embed_dim = embed_dim
        self.input_dim = input_dim

        self.input_to_state = nn.Linear(self.input_dim, self.state_dim, bias=True)
        self.input_to_gate = nn.Linear(self.input_dim, self.state_dim*2, bias=True)

        self.state_to_gate_h2h = nn.Linear(self.state_dim, self.state_dim*2, bias=False)
        self.h2h = nn.Linear(self.state_dim, self.state_dim, bias=False)

    def forward(self, data, state):
        input_state = self.input_to_state(data)
        input_gate = self.input_to_gate(data)
        state_gate = self.state_to_gate_h2h(state)
        gate = input_gate + state_gate
        slice_gate = gate.chunk(2, dim=1)
        update_gate = torch.sigmoid(slice_gate[0])
        reset_gate = torch.sigmoid(slice_gate[1])
        state_hat = self.h2h(state*reset_gate)
        state_hat = torch.tanh(input_state + state_hat)
        new_state = update_gate * state_hat + (1 - update_gate) * state
        return new_state

class MlpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super(MlpHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        # create module
        self.dp_module = nn.Dropout(p=self.dropout)
        self.linear_mlp_l1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_mlp_l2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    # [*, input_dim]
    def forward(self, x):
        h = self.linear_mlp_l1(x)
        h = F.relu(h)
        if self.dropout > 0:
            h = self.dp_module(h)
        out = self.linear_mlp_l2(h)
        return out

class MemoryClsHead(nn.Module):
    def __init__(self, input_dim, mem_dim, match_dim, num_cls=2):
        super(MemoryClsHead, self).__init__()
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.match_dim = match_dim
        self.num_cls = num_cls
        self.input_proj = nn.Linear(self.input_dim, self.match_dim, bias=True)
        self.mem_proj = nn.Linear(self.input_dim, self.match_dim, bias=False)
        self.energy_proj = nn.Linear(self.match_dim, self.num_cls, bias=True)
        #input [B, input_dim]
    

    def forward(self, selected_branchs, selected_bonds, branch_update_info, bond_update_info):
        batch_size, l_branch, query_dim = selected_branchs.shape
        selected_branchs_pro = self.input_proj.forward(selected_branchs) #[B, l_tgt, match_dim]
        selected_bonds_pro = self.mem_proj.forward(selected_bonds) # [B, l_tgt, match_dim]
        selected_branchs_pro = selected_branchs_pro.unsqueeze(2) # [B, l_branch, 1, match_dim]
        selected_bonds_pro = selected_bonds_pro.unsqueeze(1) # [B, 1, l_bond, match_dim]
        hidden = torch.tanh(selected_bonds_pro + selected_branchs_pro) #[B, l_branch, l_bond, match_dim]
        mem_energy = self.energy_proj.forward(hidden) #[B, l_branch, l_bond, num_cls]
        
        return mem_energy

class Readout(nn.Module):
    def __init__(self, vocab_size, embed_dim, state_dim, encoder_dim, merge_dim, dropout=0, embed_drop = 0, angle_embed_dim=100):
        super(Readout, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.angle_embed_dim = angle_embed_dim
        self.state_dim = state_dim
        self.encoder_dim = encoder_dim
        self.merge_dim = merge_dim
        self.dropout = dropout
        self.embed_drop = embed_drop

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        # self.angle_embed = nn.Embedding(num_embeddings=24, embedding_dim=self.angle_embed_dim)
        self.embed_dp_module = nn.Dropout(p=self.embed_drop)

        self.token_head = MlpHead(input_dim=self.embed_dim + self.state_dim + self.encoder_dim, hidden_dim = self.merge_dim, output_dim = self.vocab_size, dropout=self.dropout)
        
    # output [B, emb] cand_output [B, angle_emb] 
    def step(self, output, state, context):
        input_merge = torch.cat([output, state, context], dim = 1)
        readout = self.token_head.forward(input_merge)
        # cand_readout = self.cand_head.forward(input_merge)
        return readout

    def get_embed(self, label_id):
        embed_output = self.embed(label_id.long())
        if self.embed_drop > 0:
            embed_output = self.embed_dp_module(embed_output)
        return embed_output

    # [**, 24]
    # def get_angle_embed(self, cand_mhot):
    #     origin_shape = list(cand_mhot.shape)
    #     cand_mhot_trans = cand_mhot.view(-1, cand_mhot.shape[-1]) #[N, 24]
    #     out_embed = torch.matmul(cand_mhot_trans, self.angle_embed.weight) #[N, dim]
    #     origin_shape[-1] = self.angle_embed_dim
    #     out_embed = out_embed.view(origin_shape)
    #     return out_embed

    def forward(self, output, state, context, label_id):
        #readout, cand_readout = self.step(output, cand_output, state, context)
        readout  = self.step(output, state, context)
        embed_output = self.get_embed(label_id)
        # cand_output = self.get_angle_embed(cand_mhot) 
        return readout, embed_output # output is embedding


class PreDecoder(nn.Module):
    def __init__(self):
        super(PreDecoder, self).__init__()
        self.encoder_dim = xconfig.encoder_dim
        self.att_dim = xconfig.decoder_att_dim
        self.state_dim = xconfig.decoder_state_dim
        self.embed_dim = xconfig.decoder_embed_dim
        self.angle_embed_dim = xconfig.decoder_angle_embed_dim

        self.attention_conv = nn.Conv2d(self.encoder_dim, self.att_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.transition_gru_state_init = nn.Linear(self.encoder_dim, self.state_dim, bias=True)    

    def enc_init_states(self, encode, encode_mask):
        encode = encode * encode_mask
        fir_crx_info = encode.sum(dim=[2, 3], keepdim=False)
        fir_encoder_mask_ex = encode_mask.sum(dim=[2, 3], keepdim=False)
        fir_crx_info = fir_crx_info / fir_encoder_mask_ex # [16,384]
        initial_state = self.transition_gru_state_init(fir_crx_info) # [16, 256]
        return [initial_state, fir_crx_info]

    def zero_init_states(self, encode, mem_index_data=None):
        batch_size, _, pool_h, pool_w = encode.shape
        attention_weight_init = torch.zeros(batch_size, 1, pool_h, pool_w, device=encode.device) # coverage attention
        attention_cum_weight_init = torch.zeros(batch_size, 1, pool_h, pool_w, device=encode.device) # coverage attention
        readout_embed_init = torch.zeros(batch_size, self.embed_dim, device=encode.device)
        # angle_embed_init = torch.zeros(batch_size, self.angle_embed_dim, device=encode.device) #[B, angle_emb]
        memory_length = 1 if mem_index_data is None else mem_index_data.shape[1] # mem_index_data:[B, len]
        memory_init = torch.zeros(batch_size, memory_length, self.embed_dim + self.state_dim + self.encoder_dim, device=encode.device) # memory contains (embed, state, context)
        memory_init_mask = torch.zeros(batch_size, memory_length, device=encode.device) #[B, M]
        memory_weight =  torch.zeros(batch_size, memory_length, 2, pool_h, pool_w, device=encode.device)  #[B, L, 2, ph, pw]
        memory_init_mask[:,0] = 1
        return [attention_weight_init, attention_cum_weight_init, readout_embed_init, memory_init, memory_init_mask, memory_weight]
    def forward(self, encode, encode_mask, mem_index_data=None):

        encoded_proj = self.attention_conv(encode) # encode_dim->atten_dim
        enc_init_states = self.enc_init_states(encode, encode_mask) # [initial_state, fir_crx_info]
        zero_init_states = self.zero_init_states(encode, mem_index_data)
        return encoded_proj, enc_init_states, zero_init_states


class SequenceGenerator(nn.Module):
    def __init__(self):
        super(SequenceGenerator, self).__init__()
        self.encoder_dim = xconfig.encoder_dim
        self.att_dim = xconfig.decoder_att_dim
        self.state_dim = xconfig.decoder_state_dim
        self.embed_dim = xconfig.decoder_embed_dim
        self.vocab_size = xconfig.vocab_size
        self.chatt_dim = xconfig.decoder_chatt_dim
        self.kernel_size = xconfig.decoder_cover_kernel
        self.padding = xconfig.decoder_cover_padding
        self.query_dim = self.embed_dim + self.state_dim + self.encoder_dim
        self.mem_dim = self.embed_dim + self.state_dim + self.encoder_dim
        self.mem_match_dim = xconfig.decoder_mem_match_dim
        
        self.embed_drop = xconfig.decoder_embed_drop
        self.merge_dim = xconfig.decoder_merge_dim
        self.attention = ImageAttention(self.state_dim, self.encoder_dim, self.att_dim, self.chatt_dim, self.kernel_size, self.padding, cond_dim=self.mem_dim)
        self.transition = MyGRUTransition(self.state_dim, self.embed_dim + self.encoder_dim + self.mem_dim)
        self.readout = Readout(self.vocab_size, self.embed_dim, self.state_dim, self.encoder_dim, self.merge_dim, dropout = xconfig.decoder_dropout, embed_drop = self.embed_drop)
        self.mem_cls = MemoryClsHead(input_dim=self.query_dim, mem_dim=self.mem_dim, match_dim=self.mem_match_dim, num_cls=2) # 内存分类
        
        self.ea_idx = xconfig.vocab.getID("<ea>")
        # self.place_idx = xconfig.vocab.getID("\\place")
        # self.place_idx_tensor = self.getPlaceIds ()
        self.conntoken_idx = xconfig.vocab.getID("\\connbranch")
        self.super_idx_tensor = self.getSuperIds() # supertoken在vocab中的id
        self.bond_idx_tensor = self.getBondIds() # 获取bond在vocab中的id,如果为bond,则需要预测branch_info
        

    def getSuperIds(self):
        super_idxs = []
        for idx, word in xconfig.vocab.id2word.items():
            if '@' in word or '\Superatom' in word:
                super_idxs.append(idx)
        super_idx_tensor = torch.Tensor(super_idxs)
        return super_idx_tensor

    def getBondIds(self):
        bond_idxs = []
        for idx, word in xconfig.vocab.id2word.items():
            if '[:' in word and word.endswith(']'):
                bond_idxs.append(idx)
            elif word.startswith('?[') and word.endswith(']') and ',' in word:
                bond_idxs.append(idx)
        bond_idx_tensor = torch.Tensor(bond_idxs)
        return bond_idx_tensor

    # 每个时间步的操作
    def step(self, encode, encode_pro, encode_mask, states, label=None, cond_mem=None, next_cond=None, time_t=0, mem_index_data=None, cur_mem_used_mask = None, cur_mem_update_info = None, cond_weight=None):
        
        weight, cum_weight, output, memory, memory_mask, memory_weight, state, context = states # [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]memory_weight, [6]cur_state, [7]context
        cur_context, cur_weight, cur_cum_weight = self.attention(encode, encode_pro, encode_mask, state, weight, cum_weight, context, cond_mem, cond_weight) # 通过Attention更新当前时间步的上下文向量,注意力权重,和sum_weight
        cur_energy, cur_output = self.readout(output, state, cur_context, label)#label是 y_t, teacher forcing; cur_output是 embed(y_t); output是embed(y_{t-1}), 根据新的context计算新的energy
        
        # update memory
        temp_states = [cur_weight, cur_cum_weight, cur_output, memory, memory_mask, memory_weight, state, cur_context]
        cur_memory = memory
        cur_memory_mask = memory_mask
        cur_memory_weight = memory_weight
        cur_memory,  cur_memory_mask, cur_memory_weight = self.update_memory_for_train(time_t, mem_index_data, temp_states, cur_mem_update_info) # 更新memory部分,这一部分可能需要根据化繁为简方案修改

        # select next cond
        # next cond [B]
        batch_size, _, ph, pw = encode.shape
        
        next_cond_memory = torch.gather(cur_memory.detach(), 1, next_cond.view(batch_size, 1, 1).repeat(1,1,self.mem_dim).long()) #[B, 1, mem_dim] # next_cond_memory[i][j][k] = cur_memory[i][next_cond[i, j, k]][k]
        next_cond_mem = next_cond_memory.squeeze(1) #[B, mem_dim]

        # prepare for next step
        # data = torch.cat([cur_output, cur_cand_output, cur_context, next_cond_mem], dim=1)
        data = torch.cat([cur_output, cur_context, next_cond_mem], dim=1)
        
        cur_state = self.transition(data, state) # 通过GRU更新hidden_state
        
        cur_states =  [cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, cur_state, cur_context] # 更新状态,在时间步t,更新了αt,y_t,memory,hidden_state,context
        return cur_energy, cur_states
    
    def eval_step_pred(self, encode, encode_pro, encode_mask, states, decode_states:dict):
        weight, cum_weight, output, memory, memory_mask, memory_weight, state, context = states # state: [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]cur_memory_weight, [6]cur_state, [7]context
        encode_batch, _c, _h, _w = encode.shape
        _pro_c = encode_pro.shape[1] # 128
        decode_batch = state.shape[0]
        if decode_batch > encode_batch: 
            ex_dim = decode_batch//encode_batch
            assert ex_dim * encode_batch == decode_batch
            encode = encode.unsqueeze(1).repeat(1, ex_dim, 1, 1, 1).view(decode_batch, _c, _h, _w) #[B*b*ex, c, h, w]
            encode_pro = encode_pro.unsqueeze(1).repeat(1, ex_dim, 1, 1, 1).view(decode_batch, _pro_c, _h, _w)
            encode_mask = encode_mask.unsqueeze(1).repeat(1, ex_dim, 1, 1, 1).view(decode_batch, 1, _h, _w)
        
        cond_mem = decode_states["cond_mem"]
        cond_weight = decode_states["cond_weight"]
        cur_context, cur_weight, cur_cum_weight = self.attention(encode, encode_pro, encode_mask, state, weight, cum_weight, context, cond_mem, cond_weight) # 通过Attention更新当前step的上下文向量,注意力权重,和sum_weight
        
        cur_energy = self.readout.step(output, state, cur_context) #fixed at 2022.04.27
        cur_states =  [cur_weight, cur_cum_weight, output, memory, memory_mask, memory_weight, state, cur_context] #[2-6] not update
        log_prob = F.log_softmax(cur_energy, dim=1)
        return log_prob, cur_states, decode_states
    
    def eval_step_update_memory(self, all_outputs, states, decode_states:dict, time_t=None, is_show=False):
        '''推理阶段, 更新memory'''
        cur_weight, cur_cum_weight, cur_output, memory, memory_mask, memory_weight, state, cur_context = states
        label = all_outputs[-1]
        batch_beam = label.shape[0]
        
        cur_output = self.readout.get_embed(label) # [Bb, embed_dim]

        # ----------------------update memory-------------------
        memory_used_mask = decode_states["memory_used_mask"] #[batch*beam, l_m]
        mem_index = decode_states["mem_index"] #[batch*beam, l_m]
        super_idx_tensor = self.super_idx_tensor.to(cur_weight.device).unsqueeze(0) #[1, n_super]
        super_idx_cmp = (label.unsqueeze(1)==super_idx_tensor).sum(1) #[B, n_super] -> [B,]
        cur_len = memory_mask.sum(dim=1) # [B]
        delta = cur_len + super_idx_cmp - memory_mask.shape[1] #[B]
        pad = delta.max().long().cpu().item() # 添加memory到第0维
        if pad > 0:
            super_ids = super_idx_cmp.cpu().tolist()
            # print("pad memory len with ", pad)
        # pad (上,下)扩充最后一维  (上,下,左,右)扩充最后两维  (上,下,左,右,前,后)扩充最后三维,依此类推
        cur_memory = F.pad(memory, (0,0,0,pad)) #memory:[Bb, l_m, query_dim],pad l_m
        cur_memory_weight = F.pad(memory_weight, (0,0,0,0,0,0,0,pad))# memory_weight:[Bb, l_m, 2, ph, pw], pad l_m
        cur_memory_mask = F.pad(memory_mask, (0,pad)) # [Bb, l_m]
        cur_memory_used_mask = F.pad(memory_used_mask, (0,pad)) # [Bb, l_m]
        cur_mem_index = F.pad(mem_index, (0,pad), value=-1) # [Bb, l_m]
        
        cands_index = super_idx_cmp.nonzero().long().squeeze(1) #[N]
        delta_memory = torch.cat([cur_output[cands_index], state[cands_index], cur_context[cands_index] ], dim=1) #[N, mem_dim]
        delta_weight = torch.cat([cur_weight[cands_index], cur_cum_weight[cands_index]], dim=1) # [N, 2, ph, pw]
        binfo = torch.arange(0, batch_beam, device=cur_context.device).unsqueeze(1) #[B, 1] #(0,1,2,..,B-1)
        cmp_res = (binfo == cands_index) #[B, N]
        offset_idx = (cmp_res.cumsum(dim=1) * cmp_res).sum(0) #[N]
        base_idx = cur_len[cands_index] #[N]
        offset = (offset_idx + base_idx -1).long()
        tgt_index = torch.cat([cands_index.unsqueeze(1), offset.unsqueeze(1)], dim=1) #[N, 2]
        tgt_indexs = tgt_index.cpu().tolist()
        # pdb.set_trace()
        for nid in range(len(tgt_indexs)):
            bid, mid = tgt_indexs[nid]
            cur_memory[bid, mid, :] = delta_memory[nid]
            cur_memory_weight[bid, mid, :] = delta_weight[nid]
            cur_memory_mask[bid, mid] = 1
            cur_memory_used_mask[bid, mid] = 1
            cur_mem_index[bid, mid] = time_t
            # cur_mem_index[bid, mid, 1] = cands_index[nid, 1]
        
        # ----------------------update bond-------------------
        bond_index = decode_states["bond_index"] # [Bb, l_bond]
        bonds_mask = decode_states["bonds_mask"]
        selected_bonds = decode_states["selected_bonds"]
        bond_idx_tensor = self.bond_idx_tensor.to(cur_weight.device).unsqueeze(0) # [1, n_bond] # bond 包括-[:0] 和 ?[a,{-}]
        bond_idx_cmp = (label.unsqueeze(1) == bond_idx_tensor).sum(1)
        cur_bond_len = bonds_mask.sum(dim=1) # [B]
        bond_delta = cur_bond_len + bond_idx_cmp - bonds_mask.shape[1] # [B]
        bond_pad = bond_delta.max().long().cpu().item()
        if bond_pad > 0:
            # print("pad bonds ", bond_idx_cmp)
            pass
        cur_bonds = F.pad(selected_bonds, (0,0,0,bond_pad))
        cur_bonds_mask = F.pad(bonds_mask, (0,bond_pad))
        cur_bond_index = F.pad(bond_index, (0,bond_pad), value=-1)
        bond_cands_index = bond_idx_cmp.nonzero().long().squeeze(1)
        delta_bond = torch.cat([cur_output[bond_cands_index], state[bond_cands_index], cur_context[bond_cands_index]], dim=1) #[N, mem_dim] feature
        bond_binfo = torch.arange(0, batch_beam, device=cur_context.device).unsqueeze(1) #[B, 1] #(0,1,2,..,B-1)
        bond_cmp_res = (bond_binfo == bond_cands_index) #[B, N]
        bond_offset_idx = (bond_cmp_res.cumsum(dim=1) * bond_cmp_res).sum(0)
        bond_base_idx = cur_bond_len[bond_cands_index]
        bond_offset = (bond_offset_idx + bond_base_idx -1).long()
        bond_tgt_index = torch.cat([bond_cands_index.unsqueeze(1), bond_offset.unsqueeze(1)], dim=1)
        bond_tgt_indexs = bond_tgt_index.cpu().tolist()
        for nid in range(len(bond_tgt_indexs)):
            bid, mid = bond_tgt_indexs[nid]
            cur_bonds[bid, mid, :] = delta_bond[nid]
            cur_bonds_mask[bid, mid] = 1
            cur_bond_index[bid, mid] = time_t
        
        # ----------------------update branch-------------------
        branch_index = decode_states["branch_index"] # [Bb, l_bond]
        branchs_mask = decode_states["branchs_mask"]
        branch_idx_tensor = torch.tensor(self.conntoken_idx, device=cur_weight.device).unsqueeze(0) # [1, n_branch]
        branch_idx_cmp = (label.unsqueeze(1) == branch_idx_tensor).sum(1)
        cur_branch_len = branchs_mask.sum(dim=1) # [B]
        branch_delta = cur_branch_len + branch_idx_cmp - branchs_mask.shape[1] # [B]
        branch_pad = branch_delta.max().long().cpu().item()
        
        cur_branchs_mask = F.pad(branchs_mask, (0, branch_pad))
        cur_branch_index = F.pad(branch_index, (0, branch_pad), value=-1)
        branch_cands_index = branch_idx_cmp.nonzero().long().squeeze(1)
        branch_binfo = torch.arange(0, batch_beam, device=cur_context.device).unsqueeze(1) # [B, 1] # (0,1,2,...,B-1)
        branch_cmp_res = (branch_binfo == branch_cands_index) #[B, N]
        branch_offset_idx = (branch_cmp_res.cumsum(dim=1) * branch_cmp_res).sum(0)
        branch_base_idx = cur_branch_len[branch_cands_index]
        branch_offset = (branch_offset_idx + branch_base_idx -1).long()
        branch_tgt_index = torch.cat([branch_cands_index.unsqueeze(1), branch_offset.unsqueeze(1)], dim=1)
        branch_tgt_indexs = branch_tgt_index.cpu().tolist()
        for bid, mid in branch_tgt_indexs:
            cur_branchs_mask[bid, mid] = 1
            # final_branch_index = [x for x in cur_bond_index[bid] if x != -1][-1]
            cur_branch_index[bid, mid] = max(cur_bond_index[bid])
        

        # bond更新了cur_bonds, cur_bonds_mask, cur_bond_index
        # memory更新了cur_memory, cur_memory_mask, cur_memory_weight, cur_memory_used_mask, cur_mem_index
        new_states = [cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, state, cur_context]
        decode_states["memory_used_mask"] = cur_memory_used_mask
        decode_states["mem_index"] = cur_mem_index

        decode_states["bond_index"] = cur_bond_index
        decode_states["selected_bonds"] = cur_bonds
        decode_states["bonds_mask"] = cur_bonds_mask

        decode_states["branch_index"] = cur_branch_index
        decode_states["branchs_mask"] = cur_branchs_mask

        # ------------------------branch cls------------------------
        connbranch_idx_tensor = torch.tensor(self.conntoken_idx, device=label.device).repeat(batch_beam)
        branch_mask = (label == connbranch_idx_tensor)
        pre_branch_mask = (all_outputs[-2] == connbranch_idx_tensor)
        multi_branch = pre_branch_mask * branch_mask
        if branch_mask.sum() > 0:
            # [xconfig.vocab.getWord(int(x)) for x in all_outputs[:, 0]]
            # 出现connbranch token
            # selected_bond有0填充
            last_bonds = decode_states["bonds_mask"].sum(dim=1)-1 # 获取最后非0的index
            selected_branch = decode_states["selected_bonds"]
            selected_branch = selected_branch[torch.arange(batch_beam), last_bonds.long()].unsqueeze(1) # 获取最后一个bond隐状态 [Bb, 1, query_dim]
            branch_mask = branch_mask.unsqueeze(1) # [Bb, 1]
            selected_branch = selected_branch * branch_mask.unsqueeze(2) # [Bb, 1, query_dim]
            
            selected_bonds = decode_states["selected_bonds"][:, 1:] # 去掉第一个-1 # [Bb, l_bond, query_dim]
            bonds_mask = decode_states["bonds_mask"][:, 1:]
            mem_cls_mask = branch_mask.unsqueeze(2) * bonds_mask.unsqueeze(1)  # [B, 1, l_bond]
            mem_cls_logit = self.mem_cls.forward(selected_branch, selected_bonds, None, None) #[B, 1, l_bond, 2]
            
            mem_cls_logit = mem_cls_logit * mem_cls_mask.unsqueeze(-1)
            mem_cls_logit = mem_cls_logit.squeeze(1) # [B, l_bond, 2]
            mem_cls_cost = F.log_softmax(mem_cls_logit, dim=2) # [B, l_bond, 2]
            mem_cls_prob = F.softmax(mem_cls_logit, dim=2) # [B, l_bond, 2]
            mem_cls_res = mem_cls_prob.argmax(dim=2) # [B, l_bond]
            mem_cls_res = (mem_cls_res * mem_cls_mask.squeeze(1)).long() #[B, l_bond]
            mem_cls_cost = torch.gather(mem_cls_cost, 2, mem_cls_res.unsqueeze(-1)).squeeze(2) # [B, l_bond]  mem_cls_cost[i,j,k]=mem_cls_cost[i,j,mem_cls_res[i,j,k]]
            
            # 二分类结果转化为绝对的index
            # 判断是否出现多个connbranch或者分类结果出现多个1
            mem_cls_res_index = -torch.ones((batch_beam, 2), device=mem_cls_res.device) # [branch_id, conn_bond_id]
            for bid, bi_res in enumerate(mem_cls_res):
                if branch_mask[bid]:
                    cur_last_bond_index = int(last_bonds[bid])
                    cur_mem_prob = mem_cls_prob[bid, :cur_last_bond_index, 1]

                    if multi_branch[bid]: # 如果有多个conn_branch, 第i个connbranch选择第i大的
                        cur_output = all_outputs[:, bid].tolist()
                        cur_connbranch_id = 0
                        for token in cur_output[::-1]:
                            if int(token) == self.conntoken_idx:
                                cur_connbranch_id += 1
                            else:
                                break
                        topk_value, topk_indices = torch.topk(cur_mem_prob, k=cur_connbranch_id)
                        bond_index = topk_indices[-1] # 每次都选择最小的,保证不会重复
                    else: # 只有一个connbranch, 直接选最大的
                        bond_index = cur_mem_prob.argmax(dim=0) # 从多个1中选出概率最大的1， # 这里会出现空tensor
                    
                    bond_index = decode_states["bond_index"][bid, 1:][bond_index] # 最前面有-1
                    # branch_index = decode_states["branch_index"][bid, -1]
                    branch_index = max(decode_states["branch_index"][bid])
                    if not(bond_index != -1 and branch_index != -1):
                        print("branch_info 预测有误")
                    mem_cls_res_index[bid, 0] = branch_index # branch_index, begin with 0
                    mem_cls_res_index[bid, 1] = bond_index # bond_index
                else:
                    mem_cls_res_index[bid, 0] = -1
                    mem_cls_res_index[bid, 1] = -1
            
            mem_cls_res = mem_cls_res_index
            if is_show:
                print("预测branch_info结果: ", mem_cls_res.cpu().tolist())
        else:
            mem_cls_cost = -torch.ones((batch_beam, 2), device=cur_context.device)
            mem_cls_res = -torch.ones((batch_beam, 2), device=cur_context.device)
        decode_states["mem_cls_cost"] = mem_cls_cost
        decode_states["mem_cls_res"] = mem_cls_res

        return new_states, decode_states, tgt_indexs
    
    #cur_output [batch*beam, emb_dim]
    #cur_cand_output [batch*beam, angle_emb_dim]
    def eval_step_update(self, states, decode_states:dict):
        cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, state, cur_context = states
        # update rnn state
        cond_mem = decode_states["cond_mem"]
        data = torch.cat([cur_output, cur_context, cond_mem], dim=1)
        cur_state = self.transition(data, state) # h_t_1 -> ht
        # outputs
        cur_states =  [cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, cur_state, cur_context] #only update cur_state
        return cur_states, decode_states

    def update_memory_for_train(self, time_t, mem_index_data, states, cur_mem_update_info):
        # states: [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]memory_weight, [6]cur_state, [7]context
        # head_ouputs: y_t[B, 1] angle_t[B, 1, 24] hook_t[B, 1, L_hook] cond_t [B, 1]
        # cur_memory [B, L_M, dim] dim contains [(embed, angle_embed, state, context)]
        # mem_index_data [B, M]
        cur_weight = states[0] #[B,1,ph,pw]
        cur_cum_weight = states[1] #[B,1,ph,pw]
        cur_output = states[2]
        cur_memory = states[3] #[B, L_M, embed_dim + angle_embed_dim + state_dim + encoder_dim] 
        cur_memory_mask = states[4] #[B, L_M]
        cur_memory_weight = states[5] #[B, L, 2, ph, pw]
        cur_state = states[6]
        cur_ctx = states[7] # cur_context
        
        batch_size = cur_mem_update_info.shape[0]
        mem_dim = cur_memory.shape[-1]
        _, _, _, ph, pw = cur_memory_weight.shape
        update_mask = (cur_mem_update_info != -1).detach() #[B]
        if update_mask.sum() <= 0:
            return cur_memory, cur_memory_mask, cur_memory_weight
        new_feature = torch.cat([cur_output, cur_state, cur_ctx], dim=1)#[B, embed_dim + state_dim + encoder_dim]
        new_weight_feature = torch.cat([cur_weight, cur_cum_weight], dim=1) #[B, 2, ph, pw]
        new_weight_feature = new_weight_feature.unsqueeze(1) * update_mask.view(batch_size, 1, 1, 1, 1)  # [B, 1, 2, ph, pw]
        # new_feature = self.mem_trans.forward(new_feature).unsqueeze(1) #[B, 1, mem_dim]
        new_feature = new_feature.unsqueeze(1) #[B, 1, embed_dim + state_dim + encoder_dim]
        new_feature = new_feature * update_mask.view(batch_size, 1, 1) #[B, 1, embed_dim + state_dim + encoder_dim]
        index = (cur_mem_update_info * update_mask).long().detach() #[B]
        new_memory = torch.scatter_add(cur_memory, 1, index.view(batch_size, 1, 1).repeat(1, 1, mem_dim) , new_feature) # scatter_add和torch.gather用法含义基本相同
        new_memory_mask = torch.scatter_add(cur_memory_mask, 1, index.view(batch_size, 1), update_mask.float().view(batch_size, 1))
        new_memory_weight = torch.scatter_add(cur_memory_weight, 1, index.view(batch_size, 1, 1, 1, 1).repeat(1, 1, 2, ph, pw) , new_weight_feature)
        return new_memory, new_memory_mask, new_memory_weight


    def cost(self, readout, label, label_mask = None):
        readout = readout.permute(1, 0, 2)
        n, c, h = readout.shape
        readout = readout.reshape((n*c, h))
        
        if label_mask is not None:
            n, l = label_mask.shape
            out_mask = label_mask.reshape((n*l,))
            out_mask = out_mask.unsqueeze(dim=1)
            readout = readout * out_mask
        n, l = label.shape
        label = label.reshape((n*l,))
        output = F.softmax(readout, dim=1)
        loss = F.cross_entropy(input=readout, target=label.long(), weight=None, size_average=None, ignore_index=-1, reduce=None, reduction='mean')    
        return loss, output

    def cost_memory(self, mem_cls_logits, branch_target, mem_cls_mask):
        b, l_branch, l_bond, num_cls = mem_cls_logits.shape
        branch_target = branch_target * mem_cls_mask - (1 - mem_cls_mask)
        branch_target = branch_target.long()
        output = F.softmax(mem_cls_logits, dim=3)
        loss = F.cross_entropy(input = mem_cls_logits.view(-1,num_cls), target = branch_target.view(-1), size_average=None, ignore_index=-1, reduce=None, reduction='mean')
        return loss, output


    # label, label_mask, cond_data [B, L]
    # target_hook [B, L, L_hook, 2]
    # mem_index_data [B, M, 2]
    # init_states { cur_weight, cur_cum_weight [B, 1, ph, pw], cur_output [B, embed_dim], cand_output [B, angle_embed_dim], cur_state [B, state_dim], context [B, enc_dim] }
    def forward(self, encode, encode_mask, encode_pro, label, label_mask, target_branch, cond_data, mem_index_data, bond_index_data, mem_used_mask, mem_update_info, branch_update_info, bond_update_info, init_states):
        is_show = True
        batch_size, tgt_len = label.shape # [B, L]
        memory_len = mem_index_data.shape[1] # [B, m_len]
        branch_len = target_branch.shape[1] # [B, l_branch, l_bond]
        bond_len = bond_index_data.shape[1] # [B, l_bond]
        states = init_states #[0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]memory_weight, [6]cur_state, [7]context, cum_weight相当于coverage,记录了过去的注意力
        label = label * label_mask
        label_slice = label.chunk(chunks=tgt_len, dim=1) # anotation vector
        target_branch_slice = target_branch.chunk(chunks=branch_len, dim=1) # [B, 1, 2]
        branch_update_info_slice = branch_update_info.chunk(chunks=tgt_len, dim=1) # [B, 1]
        bond_update_info_slice = bond_update_info.chunk(chunks=tgt_len, dim=1) # [B, 1]

        cond_data = cond_data * label_mask # [B, L]
        cond_data_slice = cond_data.chunk(chunks=tgt_len, dim=1)
        mem_used_mask = mem_used_mask * label_mask.unsqueeze(-1) # [B, tgt_len, l_m]
        mem_used_mask[:, :, 0] = 1
        mem_used_mask_slice = mem_used_mask.chunk(chunks=tgt_len, dim=1)
        mem_update_info_slice = mem_update_info.chunk(chunks=tgt_len, dim=1) #list of [B,1]
        outputs_energy = [] # e_t,i
        
        max_branch_len = branch_len
        selected_branchs = torch.zeros((batch_size, max_branch_len, self.query_dim), device=encode.device) #[B, tgt_L, query_dim]
        selected_branchs_mask = torch.zeros((batch_size, max_branch_len), device=encode.device) #[B, L]
        max_bond_len = bond_len
        selected_bonds = torch.zeros((batch_size, max_bond_len, self.query_dim), device=encode.device) # [B, tgt_l, query_dim]
        selected_bonds_mask = torch.zeros((batch_size, max_bond_len), device=encode.device) # [B,L]

        default_next_cond = torch.zeros((batch_size,), device=encode.device) #[B],
        
        # if is_show:
            # print([xconfig.vocab.getWord(i) for i in label[0].cpu().tolist()])
        for i in range(tgt_len):
            _b, _l = label_slice[i].shape
            cur_memory = states[3] #[B, M, mem_dim]
            cur_memory_weight = states[5] #[B, L, 2, ph, pw]
            old_state = states[6] #[B, decoder_state_dim]

            _, _, _, ph, pw = cur_memory_weight.shape # cur_memory:alpha_t,context,hiddden_state
            cur_cond = cond_data_slice[i].squeeze(-1) #[B]
            if cur_cond.sum() > 0:
                _ = 1
            cond_memory = torch.gather(cur_memory.detach(), 1, cur_cond.view(_b,1,1).repeat(1,1,self.mem_dim).long()) #[B, 1, mem_dim] # cond_memory[i][j][k] = cur_memory[i][cur_cond[i, j, k]][k],   detach用来从计算图中脱离出来   cond_memory用来指导解码
            cond_mem = cond_memory.squeeze(1) #[B, mem_dim]
            cond_weight =  torch.gather(cur_memory_weight.detach(), 1, cur_cond.view(batch_size, 1, 1, 1, 1).repeat(1,1,2, ph, pw).long()) # [B, 1, 2, ph, pw]    # cond_weight[i][j][k][v][w] = cur_memory_weight[i][cur_cond[i,j,k,v,w]][k][v][w]
            cond_weight = cond_weight.squeeze(1) #[B, 2, ph, pw]
            next_cond =  cond_data_slice[i+1].squeeze(-1) if i < tgt_len -1 else default_next_cond.detach()
            # 逐步解码,计算新的GRU hidden state, 注意力权重, yt
            cur_energy, states = self.step(encode, encode_pro, encode_mask, states, label_slice[i].squeeze(1), cond_mem, next_cond, i, mem_index_data, mem_used_mask_slice[i].squeeze(1), mem_update_info_slice[i].squeeze(1), cond_weight)

            cur_bond_update_info = bond_update_info_slice[i].squeeze(1)
            bond_update_info_mask = (cur_bond_update_info != -1).detach()
            # 保存所有bond的隐藏状态
            if bond_update_info_mask.sum() > 0:
                new_feature = torch.cat([states[2], old_state, states[7]], dim=1) # [B, query_dim]
                new_feature = (new_feature * bond_update_info_mask.unsqueeze(1)).unsqueeze(1) # [B, 1, 1, emb_dim+state_dim+encoder_dim]
                index = (cur_bond_update_info * bond_update_info_mask).long().detach() #[B]
                selected_bonds.scatter_add_(1, index.view(batch_size, 1, 1).repeat(1, 1, self.query_dim), new_feature) # [B,L, query_dim]
                selected_bonds_mask.scatter_add_(1, index.view(batch_size, 1), bond_update_info_mask.float().view(batch_size, 1)) # [B, L]
            
            # 保存branch_info对应的隐藏状态
            cur_branch_update_info = branch_update_info_slice[i].squeeze(1)
            branch_update_mask = (cur_branch_update_info != -1).detach() # [B, len(branch)]
            if branch_update_mask.sum() > 0: # 如果有branch_info需要解码,就保存state到seleceted_queries
                # state: [0]cur_weight, [1]cur_cum_weight, [2]cur_output, [3]cur_memory, [4]cur_memory_mask, [5]memory_weight, [6]cur_state, [7]context
                selected_output = states[2] #[B, emd_dim], yt
                selected_state = old_state # old hidden state, ht
                selected_contex = states[7] # [encoder_dim], ct
                new_feature = torch.cat([selected_output, selected_state, selected_contex], dim=1) # [B, emb_dim + state_dim + encoder_dim]
                new_feature = (new_feature * branch_update_mask.unsqueeze(1)).unsqueeze(1) # [B, 1, 1, emb_dim+state_dim+encoder_dim], 需要保存new_feature
                index = (cur_branch_update_info * branch_update_mask).long().detach() #[B]
                # 将new_feature中index处的tensor加到selected_queries中
                selected_branchs.scatter_add_(1, index.view(batch_size, 1, 1).repeat(1, 1, self.query_dim), new_feature) # [B, L, emb_dim + state_dim + encoder_dim]
                # selected_queries[i, index[i,j,k], k] += new_feature[i,j,k]
                selected_branchs_mask.scatter_add_(1, index.view(batch_size, 1), branch_update_mask.float().view(batch_size, 1)) # [B, L]
                # selected_queries_mask[i, index[i,j], k] += branch_update_mask[i,j,k]

            outputs_energy.append(cur_energy) #cur_energy [B, V]
        

        mem_cls_logits = self.mem_cls.forward(selected_branchs, selected_bonds, branch_update_info, bond_update_info) #[B, l_branch, l_bond, 2]
        mem_cls_mask = selected_branchs_mask.unsqueeze(2) * selected_bonds_mask.unsqueeze(1) #[B, l_branch, l_bond]
        mem_cls_logits = mem_cls_logits * mem_cls_mask.unsqueeze(-1) # [B, l_branch, l_bond, 2] # mask

        outputs = torch.stack(outputs_energy, dim=0) #[T, B, V]

        # calc loss, mem_class_loss
        loss, readout = self.cost(outputs, label, label_mask)
        # cand_loss, cand_readout = self.cost_cand(cands, target_cand_angle, label_mask)
        mem_loss, mem_readout = self.cost_memory(mem_cls_logits, target_branch, mem_cls_mask) # branch_info预测的损失
        # mem_readout [b, l_tgt, l_tgt, 2]
        sm = [loss, mem_loss] + states + [readout, mem_readout] + [target_branch, mem_cls_mask]
        # pdb.set_trace()
        return sm

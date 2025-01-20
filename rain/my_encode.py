# -*- coding: utf-8 -*- 
# Model Architecture Definition for Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from  torch.nn import Parameter
import os
import sys

from . import xconfig
import logging
import math
logger = logging.getLogger()

class ConvBnRelu(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), stride = (1, 1), dilation = (1, 1)):
        super().__init__()

        my_padding = ( int((kernel_size[0] - 1)/2) + dilation[0] - 1, int((kernel_size[1] - 1)/2) + dilation[1] - 1 )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=my_padding, dilation = dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

    

class SELayerMask(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerMask, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask = None):
        y = (x * mask).sum([2, 3]) / (mask.sum([2, 3]) + 1e-6)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)

class ResBasicBlockSE(nn.Module):

    def __init__(self, in_channels, out_channels, basic_groups = 4, kernel_size = (3, 3), stride = (1, 1), dilation = (1, 1)):
        super().__init__()

        my_padding = ( int((kernel_size[0] - 1)/2) + dilation[0] - 1, int((kernel_size[1] - 1)/2) + dilation[1] - 1 )
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=my_padding, dilation = dilation, stride = stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=my_padding, dilation = dilation, stride = 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        if stride[0] > 1 or stride[1] > 1:
            self.res_conv = True
        else:
            self.res_conv = False

        if self.res_conv:
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), dilation = (1, 1), stride = stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )            

        self.se_layer = SELayerMask(out_channels)

    def forward(self, x, mask = None):
        
        x_feat = self.conv1(x) * mask
        x_feat = self.conv2(x_feat) * mask

        x_feat = self.se_layer(x_feat, mask)

        if self.res_conv:
            return self.relu(x_feat + self.res_conv(x))
        else:
            return self.relu(x_feat + x)



class PosSelfAtten(nn.Module):
    def __init__(self, psa_dim_in, psa_dim_att, psa_dim_out, name = 'encoder_pos_selfAtten'):
        super(PosSelfAtten, self).__init__()
        self.name = name
        self._psa_dim_out = psa_dim_out
        self._psa_dim_att = psa_dim_att
        self._psa_dim_in = psa_dim_in

        self.reduc_conv = ConvBnRelu(psa_dim_in, psa_dim_out, kernel_size = (1, 1))
        self.atten_conv = ConvBnRelu(psa_dim_in, psa_dim_att, kernel_size = (1, 1))

    def forward(self, x, mask = None):

        x_feat = self.reduc_conv(x)
        

        x_atten = self.atten_conv(x) # N C H W
        n, c, h, w = x_atten.shape
        x_atten_ = x_atten.view(n, c, -1).contiguous()
        x_atten_trans = x_atten_.transpose(1, 2).contiguous()
        
        energy = torch.matmul(x_atten_trans, x_atten_)  / (1e-8 + math.sqrt(c))

        if mask is not None:
            energy = energy + (mask.view(n, 1, -1).contiguous() - 1) * 1e8

        weight = energy.softmax(2)
        x_feat_trans = x_feat.view(n, self._psa_dim_out, -1).contiguous().transpose(1, 2).contiguous()
        out_feat = torch.matmul(weight, x_feat_trans)
        out_feat = out_feat.transpose(1, 2).contiguous().view(n, self._psa_dim_out, h, w).contiguous()

        if mask is not None:
            out_feat = out_feat * mask
        return out_feat 

class ChanSelfAtten(nn.Module):
    def __init__(self, csa_dim_in, csa_dim_att, csa_dim_out, name = 'encoder_chan_selfAtten'):
        super(ChanSelfAtten, self).__init__()
        self.name = name
        self._csa_dim_out = csa_dim_out
        self._csa_dim_att = csa_dim_att
        self._csa_dim_in = csa_dim_in

        self.reduc_conv = ConvBnRelu(csa_dim_in, csa_dim_out, kernel_size = (1, 1))
        self.atten_conv = ConvBnRelu(csa_dim_in, csa_dim_out, kernel_size = (1, 1))

        
    def forward(self, x, mask = None):

        x_feat = self.reduc_conv(x)
        

        x_atten = self.atten_conv(x) # N C H W

        n, c, h, w = x_atten.shape
        x_atten_ = x_atten.view(n, c, -1).contiguous()
        x_atten_trans = x_atten_.transpose(1, 2).contiguous()
        
        energy = torch.matmul(x_atten_, x_atten_trans) / (1e-8 + math.sqrt(h * w)) # N C C
        weight = energy.softmax(2)
        x_feat_trans = x_feat.view(n, self._csa_dim_out, -1).contiguous()

        out_feat = torch.matmul(weight, x_feat_trans) 
        out_feat = out_feat.view(n, self._csa_dim_out, h, w).contiguous()
        

        return out_feat 

class EncodeSelfAtten(nn.Module):
    def __init__(self, dim_in, dim_att, dim_out, name = 'encoder_selfAtten'):
        super(EncodeSelfAtten, self).__init__()
        self.name = name
        self._dim_out = dim_in
        self._dim_att = dim_att
        self._dim_in = dim_out

        self.pos_selfAtten = PosSelfAtten(dim_in, dim_att, dim_out)
        self.chan_selfAtten = ChanSelfAtten(dim_in, dim_att, dim_out)

        self.residual_conv = ConvBnRelu(dim_in, dim_out, kernel_size = (3, 3))
    def forward(self, x, mask = None):

        out_feat = self.pos_selfAtten(x, mask) + self.chan_selfAtten(x, mask) + self.residual_conv(x)
        
        if mask is not None:
            out_feat = out_feat * mask

        return out_feat 

class Backbone(nn.Module):
    def __init__(self, in_channels, num_level=4, num_block=[4, 4, 4, 4], num_filters_arr=[32, 64, 64, 128], num_stride_arr = [(2, 2), (2, 2), (2, 2), (2, 2)], num_groups = [4, 4, 4, 4], encoder_use_res = [1, 1, 1, 1],
                 residual=True, dropout=0,  name='cnet'):
        super(Backbone, self).__init__()
        self.num_level = num_level
        self.num_block = num_block
        self.name = name
        num_channels = 1
        stem_channels = num_filters_arr[0]
        self.stem_conv = ConvBnRelu(in_channels, stem_channels)
        num_filters_arr = num_filters_arr[1:]
        for level in range(num_level):
            dr = 0
            if level == 4:
                dr = dropout
            residual_conv = False
            if num_filters_arr[level] != num_channels:
                residual_conv = True
            
            if level < 2 or residual is False:
                if level == 0:
                    self.make_conv_block(level, num_block[level], stem_channels, num_filters_arr[level], num_stride_arr[level], dr, encoder_use_res[level], residual_conv, num_groups[level])
                else:
                    self.make_conv_block(level, num_block[level], num_filters_arr[level-1], num_filters_arr[level], num_stride_arr[level], dr, encoder_use_res[level], residual_conv, num_groups[level])
            num_channels = num_filters_arr[level]


    def make_conv_block(self, level, num_block, in_channels, out_channels, stride, dropout, residual, residual_conv, groups = 2):
        for block in range(num_block):

            if block == 0:
                self.add_module('{}_maskpool{}'.format(self.name, level), nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0, dilation=1, ceil_mode=True))
            if residual > 0:
                if block == 0:
                    self.add_module('{}_conv_l{}_b{}'.format(self.name, level, block), ResBasicBlockSE(in_channels, out_channels, basic_groups = groups, stride=stride))
                else:
                    self.add_module('{}_conv_l{}_b{}'.format(self.name, level, block), ResBasicBlockSE(out_channels, out_channels, basic_groups = groups))



    def forward(self, x, source_mask):
        x = self.stem_conv(x) * source_mask
        for level in range(self.num_level):
            for block in range(self.num_block[level]):
                if block == 0:
                    source_mask = self._modules['{}_maskpool{}'.format(self.name, level)](source_mask)

                x = self._modules['{}_conv_l{}_b{}'.format(self.name, level, block)](x, source_mask)
        return x, source_mask



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = Backbone(in_channels = xconfig.source_dim,
                             num_level       = len(xconfig.encoder_units),
                             num_block       = xconfig.encoder_units,
                             num_filters_arr = xconfig.encoder_filter_list,
                             num_stride_arr  = xconfig.encoder_stride_list,
                             num_groups      = xconfig.encoder_basic_group,
                             encoder_use_res = xconfig.encoder_use_res,
                             dropout         = xconfig.encode_dropout,
                             residual        = False,
                             name            = 'encoder')

        self.feat_drop = nn.Dropout(xconfig.encode_feat_dropout)
        self.selfAtten = EncodeSelfAtten(xconfig.encoder_filter_list[-1], xconfig.encoder_position_att, xconfig.encoder_position_dim)

    def forward(self, source, source_mask):
        encode_out, encode_mask  = self.backbone(source, source_mask)
        encode_out = self.feat_drop(encode_out)
        encode_out = self.selfAtten(encode_out, encode_mask)

        return encode_out, encode_mask
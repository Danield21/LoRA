#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
import random
import os
import numpy as np
def seed_torch(seed=4):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class IncreLayer():
    '''basic parameters for IncreLayer
       self.r: rank for the increment: W_{t}-W_{0} 
       self.incre_alpha: coefficient times with the increment
       self.incre_dropout: perform dropout on the increment 
                         - float data given by the user 
       self.merged: indicate whether the updated increment is merged with the pretrained weight (initilized weight) 
                         - default as the false
       self.merge_weights: decide wether perform merging action 
                        - bool data given by the user 
    '''
    def __init__(
        self, 
        r: int, 
        incre_alpha: int, 
        incre_dropout: float,
        merge_weights: bool,
        init_state: tuple,
        use_scaling: bool
    ):
        self.r = r
        self.incre_alpha = incre_alpha
        # Optional dropout
        if incre_dropout > 0.:
            self.incre_dropout = nn.Dropout(p=incre_dropout)
        else:
            self.incre_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.use_scaling=use_scaling # use_scaling denotes whether to scale the increment 
        self.init_seed=init_state[-1]
        self.init_method=init_state[0]

class Embedding(nn.Embedding, IncreLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        incre_alpha: int = 1,
        merge_weights: bool = True,
        init_state: tuple = ('zero',4),
        use_scaling: bool=True,
        **kwargs
    ):
        # configure original nn.Embedding
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        # configure the IncreLayer:
        # assign r, incre_alpha, incre_dropout=0, merge_weights=True (performing merge action)
        IncreLayer.__init__(self, r=r, incre_alpha=incre_alpha, incre_dropout=0,
                           merge_weights=merge_weights, init_state=init_state, use_scaling=use_scaling)
        # Actual trainable parameters
        if r > 0:
            #self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            #self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.incre_w= nn.Parameter(self.weight.new_zeros(embedding_dim,num_embeddings))
            if self.use_scaling:
                self.scaling = self.incre_alpha / self.r
            else: 
                self.scaling=1
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
    

    def reset_parameters(self):
        '''
        current we only have the zero_init for incre_w
        '''
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'incre_w'):
            # initialize incre_w
            seed_torch(self.init_seed)
            if self.init_method=='zero':
                nn.init.zeros_(self.incre_w)
            elif self.init_method=='normal':
                nn.init.normal_(self.incre_w)
            elif self.init_method=='torch.randn': #same as rosa setting 
                self.incre_w=nn.Parameter(torch.randn(self.incre_w.shape), requires_grad=True)
            elif self.init_method=='kaiming_uniform':
                nn.init.kaiming_uniform_(self.incre_w, a=math.sqrt(5))
            elif self.init_method=='random_BA':
                seed_torch(self.init_seed)
                lora_A = nn.Parameter(torch.randn(self.r, self.num_embeddings))
                seed_torch(self.init_seed+1)
                lora_B = nn.Parameter(torch.randn(self.embedding_dim, self.r))
                B_A=lora_B@lora_A 
                self.incre_w=nn.Parameter(B_A.data,requires_grad=True)



    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.incre_w).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.incre_w).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x) # self.weight.data @ one_hot(x) 
            after_incre_w = F.embedding(
                x, self.incre_w.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            ) # self.incre_w.T @ one_hot(x)
            result += after_incre_w * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, IncreLayer):
    # Incre implemented in a dense layer
    '''
    Note: In a dense layer, the weight.data.shape will be [out_features, in_features]
    Example:
    >>> linear = nn.Linear(3, 5)
    >>> linear.weight.data.shape
    >>> torch.Size([5, 3])
    '''
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        incre_alpha: int = 1, 
        incre_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        init_state: tuple = ('zero',4),
        use_scaling: bool=True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IncreLayer.__init__(self, r=r, incre_alpha=incre_alpha, incre_dropout=incre_dropout,
                           merge_weights=merge_weights,init_state=init_state,use_scaling=use_scaling)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            #self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            #self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.incre_w = nn.Parameter(self.weight.new_zeros((out_features, in_features)))
            if self.use_scaling:
                self.scaling = self.incre_alpha / self.r
            else:
                self.scaling=1
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out: # default as false, so do not operate the following line in default
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'incre_w'):
            # initialize incre_w
            seed_torch(self.init_seed)
            if self.init_method=='zero':
                nn.init.zeros_(self.incre_w)
            elif self.init_method=='normal':
                nn.init.normal_(self.incre_w)
            elif self.init_method=='torch.randn': #same as rosa setting 
                self.incre_w=nn.Parameter(torch.randn(self.incre_w.shape), requires_grad=True)
            elif self.init_method=='kaiming_uniform':
                nn.init.kaiming_uniform_(self.incre_w, a=math.sqrt(5))
            elif self.init_method=='random_BA':
                seed_torch(self.init_seed)
                lora_A = nn.Parameter(torch.randn(self.r, self.in_features))
                seed_torch(self.init_seed+1)
                lora_B = nn.Parameter(torch.randn(self.out_features, self.r))
                B_A=lora_B@lora_A 
                self.incre_w=nn.Parameter(B_A.data,requires_grad=True)

    def train(self, mode: bool = True):
        def T(w): # operate when fan_in_fan_out is true
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.incre_w) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.incre_w) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.incre_dropout(x) @ self.incre_w.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, IncreLayer):
    # Incre implemented in a dense layer
    '''
    # ===== Before =====
    >>> qkv_proj = nn.Linear(d_model, 3*d_model)
    # ===== After =====
    # Break it up (remember to modify the pretrained checkpoint accordingly)
    >>> q_proj = lora.Linear(d_model, d_model, r=8)
    >>> k_proj = nn.Linear(d_model, d_model)
    >>> v_proj = lora.Linear(d_model, d_model, r=8)
    # Alternatively, use lora.MergedLinear (recommended)
    >>> qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])
    '''
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        incre_alpha: int = 1, 
        incre_dropout: float = 0.,
        enable_incre: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IncreLayer.__init__(self, r=r, incre_alpha=incre_alpha, incre_dropout=incre_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_incre) == 0, \
            'The length of enable_incre must divide out_features'
        self.enable_incre = enable_incre
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_incre): # any(enable_incre) will be True, if at least one True exists in enable_incre 
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_incre), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_incre) * sum(enable_incre), r))
            ) # weights for Conv1D with groups=sum(enable_incre)
            if self.use_scaling:
                self.scaling = self.incre_alpha / self.r
            else:
                self.scaling=1
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_incre), -1) 
            self.lora_ind[enable_incre, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_incre)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_incre):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_incre):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.incre_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, IncreLayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, incre_alpha=1, incre_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        IncreLayer.__init__(self, r=r, incre_alpha=incre_alpha, incre_dropout=incre_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.incre_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

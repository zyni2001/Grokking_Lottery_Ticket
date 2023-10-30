import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb
import torch.nn.init as init
import math

def add_mask(model, init_mask_dict=None):

    if init_mask_dict is None:
        phi_mask_train = nn.Parameter(torch.ones_like(model.phi))
        phi_mask_fixed = nn.Parameter(torch.ones_like(model.phi), requires_grad=False)
        
        psi_mask_train = nn.Parameter(torch.ones_like(model.psi))
        psi_mask_fixed = nn.Parameter(torch.ones_like(model.psi), requires_grad=False)
        
        w_mask_train = nn.Parameter(torch.ones_like(model.w))
        w_mask_fixed = nn.Parameter(torch.ones_like(model.w), requires_grad=False)
        
    else:
        phi_mask_train = nn.Parameter(init_mask_dict['phi_mask_train'])
        phi_mask_fixed = nn.Parameter(init_mask_dict['phi_mask_fixed'], requires_grad=False)
        
        psi_mask_train = nn.Parameter(init_mask_dict['psi_mask_train'])
        psi_mask_fixed = nn.Parameter(init_mask_dict['psi_mask_fixed'], requires_grad=False)
        
        w_mask_train = nn.Parameter(init_mask_dict['w_mask_train'])
        w_mask_fixed = nn.Parameter(init_mask_dict['w_mask_fixed'], requires_grad=False)

    AddTrainableMask.apply(model, 'phi', phi_mask_train, phi_mask_fixed)
    AddTrainableMask.apply(model, 'psi', psi_mask_train, psi_mask_fixed)
    AddTrainableMask.apply(model, 'w', w_mask_train, w_mask_fixed)

class AddTrainableMask(ABC):

    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):

        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


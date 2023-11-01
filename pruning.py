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

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def print_sparsity(model):
    
    weight1_total = model.phi_mask_fixed.numel()
    weight2_total = model.psi_mask_fixed.numel()
    weight3_total = model.w_mask_fixed.numel()
    weight_total = weight1_total + weight2_total + weight3_total

    weight1_nonzero = torch.nonzero(model.phi_mask_fixed).size(0)
    weight2_nonzero = torch.nonzero(model.psi_mask_fixed).size(0)
    weight3_nonzero = torch.nonzero(model.w_mask_fixed).size(0)
    weight_nonzero = weight1_nonzero + weight2_nonzero + weight3_nonzero

    wei_sparsity = (1 - weight_nonzero / weight_total) * 100
    print("-" * 100)
    print("Sparsity of model weights: {:.2f}%".format(wei_sparsity))
    print("-" * 100)
    
    return wei_sparsity

def soft_mask_init(model, init_type, seed):
    setup_seed(seed)
    if init_type == 'all_one':
        add_trainable_mask_noise(model, c=1e-5)
    else:
        raise NotImplementedError

def add_trainable_mask_noise(model, c):

    model.phi_mask_train.requires_grad = False
    model.psi_mask_train.requires_grad = False
    model.w_mask_train.requires_grad = False
    
    rand1 = (2 * torch.rand(model.phi_mask_train.shape) - 1) * c
    rand1 = rand1.to(model.phi_mask_train.device)
    rand1 = rand1 * model.phi_mask_train
    model.phi_mask_train.add_(rand1)

    rand2 = (2 * torch.rand(model.psi_mask_train.shape) - 1) * c
    rand2 = rand2.to(model.psi_mask_train.device)
    rand2 = rand2 * model.psi_mask_train
    model.psi_mask_train.add_(rand2)

    rand3 = (2 * torch.rand(model.w_mask_train.shape) - 1) * c
    rand3 = rand3.to(model.w_mask_train.device)
    rand3 = rand3 * model.w_mask_train
    model.w_mask_train.add_(rand3)

    model.phi_mask_train.requires_grad = True
    model.psi_mask_train.requires_grad = True
    model.w_mask_train.requires_grad = True

def get_final_mask_epoch(model, wei_percent):
    # get the mask distribution of the final epoch
    wei_mask = get_mask_distribution(model, if_numpy=False)

    wei_total = wei_mask.shape[0]
    # sort by the absolute value of the mask
    wei_y, _ = torch.sort(wei_mask.abs())

    # get the threshold
    wei_thre = wei_y[int(wei_total * wei_percent)]

    mask_dict = {}
    mask_dict['phi_mask'] = get_each_mask(model.phi_mask_train, wei_thre)
    mask_dict['psi_mask'] = get_each_mask(model.psi_mask_train, wei_thre)
    mask_dict['w_mask'] = get_each_mask(model.w_mask_train, wei_thre)

    return mask_dict

def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_mask_distribution(model, if_numpy=True):

    weight_mask_tensor0 = model.phi_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor0) > 0
    weight_mask_tensor0 = weight_mask_tensor0[nonzero]

    weight_mask_tensor1 = model.psi_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor1) > 0
    weight_mask_tensor1 = weight_mask_tensor1[nonzero]

    weight_mask_tensor2 = model.w_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor2) > 0
    weight_mask_tensor2 = weight_mask_tensor2[nonzero]

    weight_mask_tensor = torch.cat([weight_mask_tensor0, weight_mask_tensor1, weight_mask_tensor2])

    if if_numpy:
        return weight_mask_tensor.detach().cpu().numpy()
    else:
        return weight_mask_tensor.detach().cpu()
    

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


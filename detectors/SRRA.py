import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='SRRA')
class SRRADetector(nn.Module):
    def __init__(self, config=None):
        super(SRRADetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):

        clip_model = CLIPModel.from_pretrained("/")

        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024 - 1)

        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
       label = data_dict['label']
       pred = pred_dict['cls']
       loss = self.loss_func(pred, label)

       if self.training:
           # Regularization term
           lambda_reg = 1.0
           reg_term = 0.0
           num_reg = 0
           for module in self.backbone.modules():
               if isinstance(module, SVDResidualLinear):
                   reg_term += module.REC()
                   reg_term += module.ROC()
                   num_reg += 1

           loss += lambda_reg * reg_term / num_reg

       loss_dict = {'overall': loss}
       return loss_dict


    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict



class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r


        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))


        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main

        return F.linear(x, weight, self.bias)

    def ROC(self):
        if self.S_residual is not None:

            UUT = self.U_residual.t() @ self.U_residual
            VVT = self.V_residual @ self.V_residual.t()

            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)

            loss = 0.1 * torch.norm(UUT - UUT_identity, p='fro') + 0.1 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0

        return loss

    def REC(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = 0.1 * torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2) + 0.1 * self.S_residual.sum()
        else:
            loss = 0.0

        return loss




def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)


        energy = S.pow(2).sum()
        Srank = energy / S.max().pow(2) / 10
        Srank = Srank.int()
        r = 1024 - Srank


        r = min(r, len(S))
        U_r = U[:, :r]  # Shape: (out_features, r)
        S_r = S[:r]  # Shape: (r,)
        Vh_r = Vh[:r, :]  # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]  # Shape: (out_features, n - r)
        S_residual = S[r:]  # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)

        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module
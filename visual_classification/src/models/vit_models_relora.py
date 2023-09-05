#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models
import loralib as lora

from .mlp import MLP
from ..utils import logging
import sys
from .vit.vit_top_down import *
from .vit.vit_bottom_up import *

from .relora import ReLoRaModel


logger = logging.get_logger("visual_prompt")



class Topdown_ViTClass(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(Topdown_ViTClass, self).__init__()
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            print("prompt config loaded! ")
            prompt_cfg = cfg.MODEL.PROMPT
            print(prompt_cfg)
        else:
            prompt_cfg = None
        self.cfg = cfg
        self.build_backbone(cfg, prompt_cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [self.cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )
        self.side = None
    
    def build_backbone(self, cfg, prompt_cfg, load_pretrain, vis):
        if cfg.MODEL.SIZE == 'base':
            self.enc, self.feat_dim = vit_topdown_base_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
        elif cfg.MODEL.SIZE == 'large':
            self.enc, self.feat_dim = vit_topdown_large_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)

        if cfg.MODEL.TRANSFER_TYPE == 'toast':
            for k, p in self.enc.named_parameters():
                if "decoders" not in k and "prompt" not in k and "head" not in k:
                    p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'toast-lite':
            for decoder in self.enc.decoders:
                old_weight = decoder.linear.weight.data
                old_weight2 = decoder.linear2.weight.data
                decoder.linear = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4, bias=False)
                decoder.linear2 = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4, bias=False)
                decoder.linear.weight.data = old_weight
                decoder.linear2.weight.data = old_weight2
            lora.mark_only_lora_as_trainable(self.enc)
            self.enc.prompt.requires_grad = True
            self.enc.top_down_transform.requires_grad = True
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        # print(self.enc(x))
        x, visualizations = self.enc(x) 
        # x = self.enc(x) 
        x = self.head(x)
        return x, visualizations


class Bottomup_ViTClass_relora(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(Bottomup_ViTClass_relora, self).__init__()
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            print("prompt config loaded! ")
            prompt_cfg = cfg.MODEL.PROMPT
            print(prompt_cfg)
        else:
            prompt_cfg = None
        self.cfg = cfg
        self.build_backbone(cfg, prompt_cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [self.cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )
        self.side = None
        #init new lora layers in dict
        self.lora_layers = []
        # self.lora_layer1= lora.MergedLinear(self.enc.embed_dim, 3*self.enc.embed_dim, r=cfg.MODEL.LORA_RANK, enable_lora=cfg.MODEL.LORA_LAYER)
        # self.lora_layer2 = lora.MergedLinear(self.enc.embed_dim, 3*self.enc.embed_dim, r=cfg.MODEL.LORA_RANK, enable_lora=cfg.MODEL.LORA_LAYER)
        # self.lora_layer3 =lora.MergedLinear(self.enc.embed_dim, 3*self.enc.embed_dim, r=cfg.MODEL.LORA_RANK, enable_lora=cfg.MODEL.LORA_LAYER)


    def build_backbone(self, cfg, prompt_cfg, load_pretrain, vis):
        if cfg.MODEL.SIZE == 'base':
            self.enc, self.feat_dim = vit_base_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
        elif cfg.MODEL.SIZE == 'large':
            self.enc, self.feat_dim = vit_large_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)

        if cfg.MODEL.TRANSFER_TYPE == 'linear':
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'prompt':
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "head" not in k:
                    p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'lora':
            target_modules_list = ["attn", "mlp"]  
            if isinstance(target_modules_list, str):
                target_modules_list = [target_modules_list]  
            self.enc = ReLoRaModel(
                self.enc,
                r=4,
                lora_alpha=1,
                lora_dropout=0.,
                target_modules=["attn", "mlp"],
                trainable_scaling=False,
                keep_original_weights=True,
                lora_only=False,
            )
            train_ln=True
            for name, param in self.enc.named_parameters():
                # LLaMa: model.norm, model.layers.input_layernorm, model.layers.post_attention_layernorm
                if train_ln and "norm" in name:
                    param.requires_grad = True        
                elif "lm_head" in name:
                    param.requires_grad = True
                elif "embed_tokens" in name:
                    param.requires_grad = True
                elif "bias" in name:
                    param.requires_grad = True
                elif "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            params_after = sum(p.numel() for p in self.enc.parameters())
            trainable_after = sum(p.numel() for p in self.enc.parameters() if p.requires_grad)

            for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(f" Parameter {name} requires gradient.")
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    # def _create_lora_layer(self, cfg):
    #     for block in self.enc.blocks:
    #         if cfg.MODEL.LORA_MLP:
    #             old_fc1_weight= block.mlp.fc1.weight.data
    #             old_fc1_bias = block.mlp.fc1.bias.data
    #             block.mlp.fc1 = lora.Linear(self.enc.embed_dim, self.enc.embed_dim * self.enc.mlp_ratio, r=cfg.MODEL.LORA_RANK)
    #             block.mlp.fc1.weight.data = old_fc1_weight#用之前的weight初始化
    #             block.mlp.fc1.bias.data = old_fc1_bias

    #             old_fc2_weight= block.mlp.fc2.weight.data
    #             old_fc2_bias = block.mlp.fc2.bias.data
    #             block.mlp.fc2 = lora.Linear(self.enc.embed_dim * self.enc.mlp_ratio,self.enc.embed_dim,  r=cfg.MODEL.LORA_RANK)
    #             block.mlp.fc2.weight.data = old_fc2_weight#用之前的weight初始化
    #             block.mlp.fc2.bias.data = old_fc2_bias
    #         if cfg.MODEL.LORA_O:
    #             old_o_weight=block.attn.proj.weight.data
    #             old_0_bias = block.attn.proj.bias.data
    #             block.attn.proj = lora.Linear(self.enc.embed_dim, self.enc.embed_dim , r=cfg.MODEL.LORA_RANK)
    #             block.attn.proj.weight.data=old_o_weight
    #             block.attn.proj.bias.data=old_0_bias

            
    #         #qkv apply lora
    #         if True in cfg.MODEL.LORA_LAYER:
    #             old_weight = block.attn.qkv.weight.data
    #             old_bias = block.attn.qkv.bias.data
    #             block.attn.qkv = lora.MergedLinear(self.enc.embed_dim, 3*self.enc.embed_dim, r=cfg.MODEL.LORA_RANK, enable_lora=cfg.MODEL.LORA_LAYER)
    #             # block.attn.qkv = lora.Linear(self.enc.embed_dim, self.enc.embed_dim*3, r=4)
    #             block.attn.qkv.weight.data = old_weight#用之前的weight初始化
    #             block.attn.qkv.bias.data = old_bias
    #     lora.mark_only_lora_as_trainable(self.enc)
    #     # 将新的LoRA层返回，这里我们实际上只返回了一个引用
    #     return self.enc.blocks  # 注意：这只是返回了一个模型部分的引用



    def forward(self, x, return_feature=False):
        x, visualizations = self.enc(x)  
        x = self.head(x)
        return x, visualizations


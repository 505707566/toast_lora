class Bottomup_ViTClass(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(Bottomup_ViTClass, self).__init__()
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
            self.enc, self.feat_dim = vit_base_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
        elif cfg.MODEL.SIZE == 'large':
            self.enc, self.feat_dim = vit_large_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)

        if cfg.MODEL.TRANSFER_TYPE == 'lora':
            for block in self.enc.blocks:
                old_weight = block.attn.qkv.weight.data
                old_bias = block.attn.qkv.bias.data
                
                # split the weight and bias for q, k and v
                old_weight_q, old_weight_k, old_weight_v = old_weight.chunk(3, dim=0)
                old_bias_q, old_bias_k, old_bias_v = old_bias.chunk(3, dim=0)
                
                # create separate Linear (or LoRA) layers for q, k and v
                if q in cfg.
                block.attn.q = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4)
                block.attn.k = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4)
                block.attn.v = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4)
                
                # initialize them with the original weights and biases
                block.attn.q.weight.data = old_weight_q
                block.attn.k.weight.data = old_weight_k
                block.attn.v.weight.data = old_weight_v
                block.attn.q.bias.data = old_bias_q
                block.attn.k.bias.data = old_bias_k
                block.attn.v.bias.data = old_bias_v
                # reinitialize qkv with the concatenated weights and biases
                block.attn.qkv.weight.data = torch.cat([block.attn.q.weight.data, block.attn.k.weight.data, block.attn.v.weight.data], dim=0)
                block.attn.qkv.bias.data = torch.cat([block.attn.q.bias.data, block.attn.k.bias.data, block.attn.v.bias.data], dim=0)
            lora.mark_only_lora_as_trainable(self.enc)
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        x, visualizations = self.enc(x)  
        x = self.head(x)
        return x, visualizations

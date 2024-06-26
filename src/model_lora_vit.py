### Instantiates a ViT model with/without low-rank adaptation and transfer-learned init ###
import torch
import torch.nn as nn

# we replace the qkv in Attention.qkv with a custom module
class Lora_qkv(nn.Module):
    def __init__(self, qkv, qa, qb, va, vb):
        super().__init__()
        self.qkv = qkv
        self.qa = qa
        self.qb = qb
        self.va = va
        self.vb = vb
        self.dim = self.qkv.in_features

    def forward(self, x):
        B, N, C = x.shape
        old_qkv = self.qkv(x) # B, N, 3C
        new_q = self.qb(self.qa(x)) # B, N, C
        new_v = self.vb(self.va(x)) # B, N, C
        
        old_qkv[:, :, : self.dim] += new_q # first C channels belong to q
        old_qkv[:, :, -self.dim:] += new_v # last C channels belong to v
        return old_qkv

# implementing the qv lora mechanism
# design similar to https://github.com/BeileiCui/SurgicalDINO/blob/main/surgicaldino.py
class Lora_vit(nn.Module):
    def __init__(self, base_vit, lora_rank=4, full_ft=False):
        super().__init__()
        if not full_ft:
            # constrain the model to only train lora weights
            for param in base_vit.parameters():
                param.requires_grad = False
        self.base = base_vit
        self.embed_dim = self.base.embed_dim
        
        self.r = lora_rank
        self.in_ftrs = self.base.blocks[0].attn.qkv.in_features
        out_ftrs_qkv = self.base.blocks[0].attn.qkv.out_features
        assert out_ftrs_qkv % 3 == 0
        self.out_ftrs = out_ftrs_qkv // 3
        
        if self.r > 0:
            self.initialize_lora_layers()
            print(f"Initialized LoRA with rank={self.r}")
        else:
            print(f"Initialized without LoRA")

    def initialize_lora_layers(self):
        # instantiate lora weights for each of the blocks
        qa, qb = [], []
        va, vb = [], []
        for i, block in enumerate(self.base.blocks):
            qa.append(nn.Linear(self.in_ftrs, self.r, bias=False))
            qb.append(nn.Linear(self.r, self.out_ftrs, bias=False))
            va.append(nn.Linear(self.in_ftrs, self.r, bias=False))
            vb.append(nn.Linear(self.r, self.out_ftrs, bias=False))
            # replacing the qkv from the original vit with the lora qkv
            block.attn.qkv = Lora_qkv(block.attn.qkv, qa[i], qb[i], va[i], vb[i]) 

    def forward(self, x): # class token
        return self.base.forward(x)

    def get_intermediate_layers(self, x, n=1):
        return self.base.get_intermediate_layers(x, n)
        

# Instantiate a new set of ViT weights with lora features
# define which parameters are differentiable
# weight loading is delegated to main_dino.. TODO check to see if it is compatible with the new arch
def get_vit(arch, patch_size, lora_rank=0): #, ckpt_override_path=None):
    torch.hub.set_dir("../pretrained_weights")
    
    # load arch
    if arch == 'vit_small':
        if patch_size == 8:
            dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        elif patch_size == 16:
            dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        else:
            assert ValueError
    elif arch == 'vit_base':
        if patch_size == 8:
            dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        elif patch_size == 16:
            dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        else:
            assert ValueError
    else:
        assert ValueError
    
    # define which params are differentiable
    if lora_rank == 0:
        full_ft = True
    else:
        full_ft = False
    model = Lora_vit(dino_backbone, lora_rank, full_ft)
    return model
    
    
def load_lora_vit_from_dino_ckpt(model, ckpt_path):
    # we are loading the teacher model from the DINO checkpoint
    teacher = torch.load(ckpt_path)['teacher']
    
    # edit the dictionary to remove the projector and rename backbone entries
    for k in list(teacher.keys()):
        if 'backbone' in k:
            teacher[k.replace('backbone.', '')] = teacher.pop(k)
        else:
            teacher.pop(k)

    # model is the lora_vit model with a consistent lora rank as the checkpoint
    model.load_state_dict(teacher, strict=False)
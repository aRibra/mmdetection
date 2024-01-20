from mmdet.registry import MODELS
from mmengine.model import BaseModule
import torch.nn as nn
from torch import load


@MODELS.register_module()
class ChannelMapperPruned(BaseModule):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.checkpoint = checkpoint
        print("ChannelMapperPruned()/ checkpoint", self.checkpoint)
        gd_backbone_and_neck = load(self.checkpoint)
        self.pruned_neck = gd_backbone_and_neck.neck
    
    def forward(self, x):
        return self.pruned_neck(x)
    
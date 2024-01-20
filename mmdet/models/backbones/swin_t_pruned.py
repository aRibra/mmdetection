from mmdet.registry import MODELS
from mmengine.model import BaseModule
# import torch.nn as nn
from torch import load


@MODELS.register_module()
class SwinTransformerPruned(BaseModule):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.checkpoint = checkpoint
        print("SwinTransformerPruned25()/ checkpoint", self.checkpoint)
        gd_backbone_and_neck = load(self.checkpoint)
        self.pruned_backbone = gd_backbone_and_neck.backbone

    def forward(self, x):
        return self.pruned_backbone(x)

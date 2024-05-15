import os
import torch.nn as nn
from model.head import build_head
from model.backbone import build_backbone

class Resnet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.batchsize = args.batch_size

        self.backbone, feat_dim = build_backbone(args.backbone, 
                                                 args.pretrained,
                                                 os.path.join(args.root, args.project, 'results'))
        
        self.head = build_head(512*1*1, args.num_classes)
    
    def forward(self, x):
        feature = self.backbone(x)
        feature = feature.view(feature.size(0), -1)

        cls_prediction = self.head(feature)

        return cls_prediction
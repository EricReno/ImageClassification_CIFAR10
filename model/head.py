import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k = 1, p = 0, s = 1, d = 1, act_type = 'lrelu', norm_type = 'BN') -> None:
        super(Conv, self).__init__()
        
        convs = []
        convs.append(nn.Conv2d(c1, c2, 1, stride=s, padding=p, dilation=d, groups=1, bias=False))
        convs.append(nn.LeakyReLU(c2))
        convs.append(nn.BatchNorm2d(c2))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

class Head(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # cls head
        cls_feats = []
        cls_feats.append(nn.Linear(self.in_dim, 4096))
        cls_feats.append(nn.ReLU(inplace=True))
        cls_feats.append(nn.Dropout())

        cls_feats.append(nn.Linear(4096, 4096))
        cls_feats.append(nn.ReLU(inplace=True))
        cls_feats.append(nn.Dropout())

        cls_feats.append(nn.Linear(4096, 1000))
        cls_feats.append(nn.Linear(1000, self.out_dim))

        self.cls_feats = nn.Sequential(*cls_feats)

    def forward(self, x):
        cls_feats = self.cls_feats(x)

        return cls_feats
    
def build_head(feat_dim, out_dim):
    head = Head(feat_dim, out_dim)

    return head
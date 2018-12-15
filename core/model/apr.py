import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50, resnet101, resnet152


class APR(nn.Module):
    """ This is a baseline of region-based classification """
    def __init__(
        self, 
        num_classes=100,
        last_conv_stride=2
    ):
        super(APR, self).__init__()
        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

        self.att_group = []
        self.att_group.append([0])
        self.att_group.append([1,2,3,4])
        self.att_group.append([5,6,7])
        for i in range(8,54):
            self.att_group.append([i])
        # for triplet, no fc
        self.fc_group = nn.ModuleList()
        fc = nn.Linear(2048, num_classes)
        init.normal(fc.weight, std=0.001)
        self.fc_group.append(fc)
        for group in self.att_group:
            cnt = len(group)
            if cnt == 1:
                cnt = 2
            fc = nn.Linear(2048, cnt)
            init.normal(fc.weight, std=0.001)
            self.fc_group.append(fc)
        print len(self.fc_group)
        
    def forward(self, x):
        """
        Returns:
            local_feat_list: each one with shape [N, c]
            logits_list: each one with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        stripe_h = feat.size(2)
        stripe_w = feat.size(3)
        # shape [N, C, 1, 1]
        feat = F.avg_pool2d(feat, (stripe_h, stripe_w))
        feat = feat.view(feat.size(0), -1)

        logits = []
        for i in range(len(self.fc_group)):
            logits.append( self.fc_group[i](feat) )

        if not self.training:
            return feat

        return logits

class APRExtractFeature(object):
    """ A feature extraction function
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_mode = self.model.training

        # set the mode to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print 'imgs should be type: Variable'
            raise ValueError
        feat =  self.model(imgs)
        # no-local-normalize
        feat = feat.data.cpu().numpy()
         
        self.model.train(old_train_eval_mode)
        return feat

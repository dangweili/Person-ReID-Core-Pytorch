import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50


class Res50Model(nn.Module):
    """ This is a baseline of region-based classification """
    def __init__(
        self, 
        last_conv_stride=2,
        num_classes=100
    ):
        super(Res50Model, self).__init__()
        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

    def forward(self, x):
        """
        Returns:
            local_feat_list: each one with shape [N, c]
            logits_list: each one with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        ms = feat.shape
        # shape [N, C, 1, 1]
        feat = F.max_pool2d(feat, ms[2:]).view(ms[0], ms[1])
        return feat

class Res50ExtractFeature(object):
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
        # feat, logits =  self.model(imgs)
        feat =  self.model(imgs)
        # may normalize for speed
        # with-local-normalize
        # feat = [lf.div(lf.norm(2, 1, True).expand_as(lf)).data.cpu().numpy() for lf in local_feat_list]
        # no-local-normalize
        feat = feat.data.cpu().numpy()
         
        self.model.train(old_train_eval_mode)
        return feat

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50


class PCBModel(nn.Module):
    """ This is a baseline of region-based classification """
    def __init__(
        self, 
        last_conv_stride=2,
        num_stripes=6,
        local_conv_out_channels=256,
        num_classes=100
    ):
        super(PCBModel, self).__init__()
        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

        self.num_stripes = num_stripes
        
        self.local_conv_list = nn.ModuleList()
        self.local_bn_list = nn.ModuleList()
        self.local_relu_list = nn.ModuleList()
        self.local_fc_list = nn.ModuleList()
        self.local_dropout_list = nn.ModuleList()

        for _ in range(num_stripes):
            local_conv = nn.Conv2d(2048, local_conv_out_channels, 1)
            local_bn = nn.BatchNorm2d(local_conv_out_channels)
            local_relu = nn.ReLU(inplace=True)
            local_fc = nn.Linear(local_conv_out_channels, num_classes)
            local_dropout = nn.Dropout(p=0.5, inplace=True)
            init.normal(local_fc.weight, std=0.001)
            init.constant(local_fc.bias, 0)

            self.local_conv_list.append(local_conv)
            self.local_bn_list.append(local_bn)
            self.local_relu_list.append(local_relu)
            self.local_fc_list.append(local_fc)
            self.local_dropout_list.append(local_dropout)

    def forward(self, x):
        """
        Returns:
            local_feat_list: each one with shape [N, c]
            logits_list: each one with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        stripe_w = feat.size(3)
        # shape [N, C, num_stripes, 1]
        feat = F.avg_pool2d(feat, (stripe_h, stripe_w))

        local_feat_list = []
        local_logits_list = []
        for i in range(self.num_stripes):
            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](feat[:, :, i:i+1, :])
            local_feat = self.local_bn_list[i](local_feat)
            local_feat = self.local_relu_list[i](local_feat)
            # local_feat = self.local_dropout_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            # multi-class classification
            local_logits_list.append(self.local_fc_list[i](local_feat))

        return local_feat_list, local_logits_list

class PCBExtractFeature(object):
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
        local_feat_list, local_logits_list =  self.model(imgs)
        # may normalize for speed
        # with-local-normalize
        # feat = [lf.div(lf.norm(2, 1, True).expand_as(lf)).data.cpu().numpy() for lf in local_feat_list]
        # no-local-normalize
        feat = [lf.data.cpu().numpy() for lf in local_feat_list]
        feat = np.concatenate(feat, axis=1)
         
        self.model.train(old_train_eval_mode)
        return feat

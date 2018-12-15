from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable


__all__ = ['MuDeep', 'MuDeepExtractFeature']
# 186/158/173 lines should be changed if follow the same protocol in original paper

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s, p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvLayers(nn.Module):
    """Preprocessing layers."""
    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = ConvBlock(3, 48, k=3, s=1, p=1)
        self.conv2 = ConvBlock(48, 96, k=3, s=1, p=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x

class FirstConvLayers(nn.Module):
    """Preprocessing layers."""
    def __init__(self):
        super(FirstConvLayers, self).__init__()
        self.conv1 = ConvBlock(3, 48, k=3, s=1, p=0)
        self.conv2 = ConvBlock(48, 96, k=3, s=1, p=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x

class MultiScaleA(nn.Module):
    """Multi-scale stream layer A (Sec.3.1)"""
    def __init__(self):
        super(MultiScaleA, self).__init__()
        self.stream1 = nn.Sequential(
            ConvBlock(96, 96, k=1, s=1, p=0),
            ConvBlock(96, 24, k=3, s=1, p=1),
        )
        self.stream2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(96, 24, k=1, s=1, p=0),
        )
        self.stream3 = ConvBlock(96, 24, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(
            ConvBlock(96, 16, k=1, s=1, p=0),
            ConvBlock(16, 24, k=3, s=1, p=1),
            ConvBlock(24, 24, k=3, s=1, p=1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class Reduction(nn.Module):
    """Reduction layer (Sec.3.1)"""
    def __init__(self):
        super(Reduction, self).__init__()
        self.stream1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stream2 = ConvBlock(96, 96, k=3, s=2, p=1)
        self.stream3 = nn.Sequential(
            ConvBlock(96, 48, k=1, s=1, p=0),
            ConvBlock(48, 56, k=3, s=1, p=1),
            ConvBlock(56, 64, k=3, s=2, p=1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class MultiScaleB(nn.Module):
    """Multi-scale stream layer B (Sec.3.1)"""
    def __init__(self):
        super(MultiScaleB, self).__init__()
        self.stream1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, k=1, s=1, p=0),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(256, 64, k=1, s=1, p=0),
            ConvBlock(64, 128, k=(1, 3), s=1, p=(0, 1)),
            ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)),
        )
        self.stream3 = ConvBlock(256, 256, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(
            ConvBlock(256, 64, k=1, s=1, p=0),
            ConvBlock(64, 64, k=(1, 3), s=1, p=(0, 1)),
            ConvBlock(64, 128, k=(3, 1), s=1, p=(1, 0)),
            ConvBlock(128, 128, k=(1, 3), s=1, p=(0, 1)),
            ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        return s1, s2, s3, s4


class Fusion(nn.Module):
    """Saliency-based learning fusion layer (Sec.3.2)"""
    def __init__(self):
        super(Fusion, self).__init__()
        self.a1 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a2 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a3 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a4 = nn.Parameter(torch.rand(1, 256, 1, 1))
        
        # We add an average pooling layer to reduce the spatial dimension
        # of feature maps, which differs from the original paper.
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x1, x2, x3, x4):
        s1 = self.a1.expand_as(x1) * x1
        s2 = self.a2.expand_as(x2) * x2
        s3 = self.a3.expand_as(x3) * x3
        s4 = self.a4.expand_as(x4) * x4
        y = self.avgpool(s1 + s2 + s3 + s4)
        # y = s1 + s2 + s3 + s4
        return y


class MuDeep(nn.Module):
    """Multiscale deep neural network.

    Reference:
    Qian et al. Multi-scale Deep Learning Architectures for Person Re-identification. ICCV 2017.
    """
    def __init__(self, num_classes, **kwargs):
        super(MuDeep, self).__init__()
        # self.loss = loss

        # self.block1 = FirstConvLayers()
        self.block1 = ConvLayers()
        self.block2 = MultiScaleA()
        self.block3 = Reduction()
        self.block4 = MultiScaleB()
        self.block5 = Fusion()
        
        # Due to this fully connected layer, input image has to be fixed
        # in shape, i.e. (3, 256, 128), such that the last convolutional feature
        # maps are of shape (256, 16, 8). If input shape is changed,
        # the input dimension of this layer has to be changed accordingly.
        self.fc = nn.Sequential(
            # nn.Linear(256*39*14, 4096),
            nn.Linear(256*16*8, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4096, num_classes)
        self.feat_dim = 4096
    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(*x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.dropout(x, p=0.3, training=self.training)
        y = self.classifier(x)

        if not self.training:
            # l2 normalization before concatenation
            # x = x / x.norm(p=2, dim=1, keepdim=True)
            return x

        return y

        #if self.loss == {'xent'}:
        #    return y
        #elif self.loss == {'xent', 'htri'}:
        #    return y, x
        #else:
        #    raise KeyError("Unsupported loss: {}".format(self.loss))

class MuDeepExtractFeature(object):
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
        local_feat = self.model(imgs)
        feat = local_feat.data.cpu().numpy()
         
        self.model.train(old_train_eval_mode)
        return feat

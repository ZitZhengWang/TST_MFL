import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


import models
import utils
from .BGF import BGF
from .models import register


@register('2LayerFC-classifier')
class TwoLayerFCClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, mid_dim=256):
        super(TwoLayerFCClassifier, self).__init__()

        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, n_classes)
        self.LeakyReLU = nn.LeakyReLU()
    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))
        return x


@register('GlobalNetForPretrain')
class GlobalNetForPretrain(nn.Module):
    """
    GlobalNetForPretrain几乎和之前的Classifier完全一样
    encoder：ResNet12
    classifier：2LayerFC-classifier
    """
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)    # 直接输出的是特征向量
        classifier_args["in_dim"] = self.encoder.out_dim    # 获取encoder的输出维度，作为分类器的输入维度
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


@register("LocalNetForPretrain")
class LocalNetForPretrain(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args,
                 input_dim, head_num, feedforward_dim, batch_first):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)

        # 新想法是在局部网络中增加自注意力机制，来融合随机采样的样本间的关系
        self.transformerEncoderlayer = TransformerEncoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforward_dim,
                                                               batch_first=batch_first)

        classifier_args["in_dim"] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        """
        输入：x [B,N,C,H,W]
        """
        BN_shape = x.shape[:2]
        img_shape = x.shape[-3:]
        x = x.view(-1, *img_shape)
        x = self.encoder(x)     # 输出 x [128*36, 512]
        x = x.view(*BN_shape, -1)    # [B, N, 512]

        # Transformer层计算
        x = self.transformerEncoderlayer(x)    # # [B, N, 512]

        x = self.classifier(x)      # 输出 x [128*36, 64]
        return x


@register('FusionNetForPretrain')
class FusionNetForPretrain(nn.Module):
    def __init__(self, globalNet, globalNet_args,
                 localNet, localNet_args):
        super().__init__()

        self.GlobalNet = models.make(globalNet, **globalNet_args)
        self.LocalNet = models.make(localNet, **localNet_args)

    def forward(self, x_global, x_local):
        # 传统训练时： global_imgs [128,3,84,84]    local_imgs [128,36,3,26,26]
        x_global = self.GlobalNet(x_global)
        x_local = self.LocalNet(x_local)
        return x_global, x_local


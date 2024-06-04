import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

import models
import utils
from .BGF import BGF
from .models import register
import utils.few_shot as fs

@register('GlobalNetV4')
class GlobalNetV4(nn.Module):
    """
    全局的实现参考原 Meta-Baseline
    encoder：ResNet12
    classifier：
    """
    def __init__(self, encoder, encoder_args,
                 classifier='cos', classifier_args={},
                 temp=1., temp_learnable=True):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)
        self.classifier = classifier

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.classifier == 'cos':
            x_proto = x_shot.mean(dim=-2)    # 取均值，会默认减小一个维度，[4,5,5,512]-> [4,5,512]
            x_proto = F.normalize(x_proto, dim=-1)    # 归一化
            x_query = F.normalize(x_query, dim=-1)    # [4,75,512]
            metric = 'dot'
        elif self.classifier == 'sqr':
            x_proto = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_proto, metric=metric, temp=self.temp)

        # 返回的是归一化后的query
        # 返回三维的张量logits    [B, way*query, way]

        return logits, x_shot, x_query



@register("LocalNetV4")
class LocalNetV4(nn.Module):
    """
    局部的实现参考 KLSA
    """
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args,
                 input_dim, head_num, feedforward_dim, batch_first,
                 use_BGF=True, threshold=0.2, weight=False,
                 temp=1., temp_learnable=True):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)

        # 新想法是在局部网络中增加自注意力机制，来融合随机采样的样本间的关系
        self.transformerEncoderlayer = TransformerEncoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforward_dim,
                                                               batch_first=batch_first)
        self.use_BGF = use_BGF

        if self.use_BGF:
            self.BGF = BGF(threshold=threshold, weight=weight)

        if classifier == "localPatchClassifier" or classifier == "localPatchClassifierV2":
            self.classifier = models.make(classifier, **classifier_args)
        else:
            self.classifier = classifier

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, g_query_fea=None):
        # l_shot: torch.Size([3, 5, 5, 36, 3, 21, 21])
        # l_query: torch.Size([3, 75, 36, 3, 21, 21])
        shot_shape = x_shot.shape[:-3]    # [1, 5, 5, 36]
        query_shape = x_query.shape[:-3]    # [1, 75, 36]
        img_shape = x_shot.shape[-3:]    # [3,21,21]

        x_shot = x_shot.view(-1, *img_shape)    # [720,3,26,26]
        x_query = x_query.view(-1, *img_shape)    # [10800,3,26,26]
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))    # [11520, 512]

        _, fea_dim = x_tot.shape

        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(shot_shape[0]*shot_shape[1]*shot_shape[2], shot_shape[3], fea_dim)
        x_query = x_query.view(-1, shot_shape[3], fea_dim)

        x_shot = self.transformerEncoderlayer(x_shot)
        x_query = self.transformerEncoderlayer(x_query)

        x_shot = x_shot.view(*shot_shape, fea_dim)    # l_shot: torch.Size([3, 5, 5, 36, 512])
        x_query = x_query.view(*query_shape, fea_dim)    # l_query: torch.Size([3, 75, 36, 512])

        if self.use_BGF:
            # 背景筛选模块, 背景筛选模块不会改变x_query的尺寸, 并且返回的筛选过后的、归一化过的 l_query
            x_query = self.BGF(g_query_fea, x_query)
        else:
            # 如果不使用BGF，就简单的做一个归一化操作
            x_query = F.normalize(x_query, dim=-1)  # 加权以后重新归一化

        # 首先对support归一化
        x_shot = F.normalize(x_shot, dim=-1)    # [3, 5, 5, 36, 512]

        if self.classifier == 'cos':
            x_shot = x_shot.mean(dim=-2)

            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        elif self.classifier == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        else:
            logits = self.classifier(x_query, x_shot) * self.temp

        return logits


@register("FusionMetricModule")   # calculate_task_variability
class FusionMetricModule(nn.Module):
    def __init__(self, n_way=5, n_metric=2):
        super(FusionMetricModule, self).__init__()

        self.NormLayer = nn.BatchNorm1d(n_way*n_metric, affine=True)
        self.FCLayer = nn.Conv1d(1, 1, kernel_size=n_metric, stride=1, dilation=5, bias=False)

    def forward(self, g_logits, l_logits):
        """
        输入全局和局部的logits
        输出为融合后的logits
        """
        # [B*75, 5]
        logits = torch.cat((g_logits, l_logits), dim=1)

        logits = self.NormLayer(logits).unsqueeze(1)
        logits = self.FCLayer(logits).squeeze(1)

        return logits    # [B,2]

@register('FusionNetV4')
class FusionNetV4(nn.Module):
    def __init__(self, globalNet, globalNet_args,
                 localNet, localNet_args, n_way=5, n_metric=2):
        super().__init__()

        self.GlobalNet = models.make(globalNet, **globalNet_args, temp_learnable=False)
        self.LocalNet = models.make(localNet, **localNet_args, temp_learnable=False)

        self.FusionModule = FusionMetricModule(n_way=n_way, n_metric=n_metric)

    def forward(self, x_global, x_local, config):
        # 小样本时 global_imgs [320,3,84,84]    local_imgs [320,36,3,26,26]
        # 要对support 和 query进行分割

        # 处理全局块
        # g_shot [4,5,1,3,84,84]    g_query [4,75,3,84,84]
        g_shot, g_query = fs.split_shot_query(x_global.cuda(), config["n_way"], config["n_shot"], config["n_query"],
                                              ep_per_batch=config["ep_per_batch"])
        # 处理局部快
        # l_shot [4,5,1,36,3,26,26]    l_query [4,75,36,3,26,26]
        l_shot, l_query = fs.split_shot_query(x_local.cuda(), config["n_way"], config["n_shot"], config["n_query"],
                                              ep_per_batch=config["ep_per_batch"])

        # 取消 temp参数以后，全局和局部输出的结果都是经过归一化后算出来的相似度；并不是分布
        # 返回3维的logits，方便加权;[1,75,5]  [1,75,5]
        g_logits, g_shot, g_query = self.GlobalNet(g_shot, g_query)
        l_logits = self.LocalNet(l_shot, l_query, g_query)    # 局部的是 tok相加的值  应该是 180个相似度

        # g_logits = self.softmax(g_logits)
        # l_logits = self.softmax(l_logits)

        # 融合两个的logits
        B, _, n_way = g_logits.shape
        g_logits = g_logits.view(-1, n_way)
        l_logits = l_logits.view(-1, n_way)

        logits = self.FusionModule(g_logits, l_logits)

        # logits = logits.view(-1, n_way)

        return logits


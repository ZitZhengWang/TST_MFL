import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

import models
import utils
from .BGF import BGF
from .models import register
import utils.few_shot as fs

@register('FusionNetFusionAblation')
class FusionNetFusionAblation(nn.Module):
    def __init__(self, globalNet, globalNet_args,
                 localNet, localNet_args, fusion_mode="Conv1d", n_way=5, n_metric=2,
                 Alpha=0.5):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.GlobalNet = models.make(globalNet, **globalNet_args, temp_learnable=False)
        self.LocalNet = models.make(localNet, **localNet_args, temp_learnable=False)

        if self.fusion_mode == "Conv1d":
            self.FusionModule = FusionMetricModule(n_way=n_way, n_metric=n_metric)

        elif self.fusion_mode == "FC":
            self.FusionModule = nn.Sequential(
                nn.Linear(n_way*n_metric, n_way),
                nn.Tanh()
            )

        elif self.fusion_mode == "Manual":
            self.Alpha = Alpha
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
        g_logits, g_shot, _ = self.GlobalNet(g_shot, g_query)
        l_logits = self.LocalNet(l_shot, l_query)    # 局部的是 tok相加的值  应该是 180个相似度


        # 融合两个的logits
        B, _, n_way = g_logits.shape
        g_logits = g_logits.view(-1, n_way)
        l_logits = l_logits.view(-1, n_way)

        if self.fusion_mode == "Conv1d":
            logits = self.FusionModule(g_logits, l_logits)

        elif self.fusion_mode == "FC":
            logits = torch.cat((g_logits, l_logits), dim=-1)
            logits = self.FusionModule(logits)

        elif self.fusion_mode == "Manual":
            logits = self.Alpha * g_logits + (1 - self.Alpha) * l_logits

        # logits = logits.view(-1, n_way)

        return logits


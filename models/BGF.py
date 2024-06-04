import torch
import torch.nn as nn
from .models import register
import torch.nn.functional as F


@register('BGF')
class BGF(nn.Module):
    """
        kLSM, Key local screening module
    """
    def __init__(self, threshold, weight=False):
        super(BGF, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, g_query_fea, l_query):
        # l_query: torch.Size([3, 75, 36, 512])    query_globle_fea: [3, 75, 512]

        batch, n_wq, n_patches, dim = l_query.shape

        # 首先将 g_query_fea 和 l_query做归一化处理，并将g_query_fea在最后一个维度扩展
        g_query_fea = F.normalize(g_query_fea, dim=-1).unsqueeze(-1)    # [3, 75, 512, 1]
        l_query = F.normalize(l_query, dim=-1)    # [3, 75, 36, 512]

        # 将g_query_fea 和 l_query 的前两个维度合并，以便进行 bmm操作，bmm要求三维张亮
        g_query_fea = g_query_fea.view(batch*n_wq, dim, -1)
        l_query = l_query.view(batch*n_wq, n_patches, dim)

        # 计算每张图像的局部块与 全局块 做cos相似度计算
        innerproduct_matrix = torch.bmm(l_query, g_query_fea)  # [3*75, 36, 512] * [3*75, 512, 1] = [3*75, 36, 1]

        # 根据相似度 和 阈值来得到 mask
        mask = (innerproduct_matrix > self.threshold) + 0.0  # 3维张量

        # 注意：
        # 加权是对mask过后的进一步加权
        # 加权是对同一个query图像的不同query块进行加权，来做一个重要程度的区分;
        # 相似度不就是权重吗，先用阈值过滤一部分，在对过滤后的相似度归一化
        # 如果设置了加权选项，则进行加权，则
        if self.weight:
            weight = innerproduct_matrix * mask    # [225,36,1]
            weight = F.normalize(weight, p=1, dim=-2)    # 1范数归一化的权重
            l_query = l_query * weight    # [225, 36, 512] * [225,36,1]
            l_query = F.normalize(l_query, dim=-1)    # 加权以后重新归一化
        else:
        # 由 mask 将小于阈值的 query给遮盖掉；直接做乘法即可
            l_query = l_query * mask  # [3*75, 36, 512] * [3*75, 36, 1] =

        l_query = l_query.view(batch, n_wq, n_patches, dim)    # [3, 75, 36, 512]
        # 返回筛选过后的、归一化过的 l_query [n_way*n_query, n_patches, 64]
        return l_query

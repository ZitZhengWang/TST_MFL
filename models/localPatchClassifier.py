import torch
import torch.nn as nn
from .models import register
import torch.nn.functional as F

@register('localPatchClassifier')
class localPatchClassifier(nn.Module):
    """
    SSMM, Similarity semantic measure module
    """
    # l_shot: torch.Size([3, 5, 5, 36, 512])    l_query: torch.Size([3, 75, 36, 512])
    # localPatchClassifier分类器需要把一批的 局部块shot和query
    def __init__(self, neighbor_k=3):
        super(localPatchClassifier, self).__init__()
        self.neighbor_k = neighbor_k

    # def calculate_cos_similarity(self, query_fea, support_fea):
    #     # [3, 5, 180, 512]    [3, 75, 36, 512]
    #     # 后两个维度相乘表示 【36 512】*【512 180】=【36 180】 表示每个query块儿 同类中所有support块的相似度
    #     similarity_batch = []
    #     for i in range(len(support_fea)):
    #         # 遍历每一个batch
    #         similarity_matrix = []
    #         for query in query_fea[i]:
    #             # query_fea[i]表示每个batch中包含的所有 n_wq个query
    #             # support_fea[i]表示第 i 个batch中的所有 support类



    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, query_fea, support_fea):
        # [3, 5, 180, 512]    [3, 75, 36, 512]
        # 【batch, way, shot*patch_num, dim】    【batch, n_wq, patch_num, dim】
        # 后两个维度相乘表示 【36 512】*【512 180】=【36 180】 表示每个query块儿 同类中所有support块的相似度

        batch, n_wq, patch_num, dim = query_fea.shape
        _, n_way, _, dim = support_fea.shape

        Similarity_list = []

        for k in range(len(support_fea)):    # 遍历每一个batch
            for i in range(n_wq):  # 遍历每一个query

                query = query_fea[k][i]   # 【n_patches, dim】   第k个batch的 第 i个 query
                # query_norm = torch.norm(query, 2, 1, True)
                # query = query / query_norm

                if torch.cuda.is_available():
                    inner_sim = torch.zeros(1, n_way).cuda()  # 创建一个空的向量，存放最终计算出的相似度

                for j in range(n_way):  # 遍历support中的每个类
                    support_class = support_fea[k][j]      # [n_support*n_patches, dim] 第k个batch的 第 j 个类

                    support_class = torch.transpose(support_class, 0, 1)     # [dim, n_support*n_patches]

                    # cosine similarity between a query sample and a support category  [36, 180]
                    innerproduct_matrix = query @ support_class  # 【n_patches, dim】* [dim, n_support*n_patches]

                    # 选出与每个 query_patch最相似的k个 support_patch    topk_value [36,5]
                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, dim=1)
                    inner_sim[0, j] = torch.mean(topk_value)    # topk_value、topk_index 【36, 3】 将全部的值加起来；

                Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list  # [75, 5]


    def forward(self, query_fea, support_fea):
        # query_fea: torch.Size([3, 75, 36, 512]) 归一化后的
        # support_fea    [3, 5, 5, 36, 512]  归一化后的

        batch, way, shot, patch_num, dim = support_fea.shape
        _, n_wq, _, _= query_fea.shape

        support_fea = support_fea.view(batch, way, shot*patch_num, dim)    # [3, 5, 180, 512]

        # query_fea = query_fea.view(batch, n_wq, patch_num, dim)    # [3, 75, 36, 512]

        # 用bmm实现 批计算cos相似度，
        # 那么均输入4为的张量;
        # 结果表示每个query与 每个类的相似度
        Similarity_list = self.cal_cosinesimilarity(query_fea, support_fea)


        return Similarity_list


@register('localPatchClassifierV2')
class localPatchClassifierV2(nn.Module):
    # l_shot: torch.Size([3, 5, 5, 36, 512])    l_query: torch.Size([3, 75, 36, 512])
    # localPatchClassifier分类器需要把一批的 局部块shot和query
    def __init__(self, neighbor_k=3):
        super(localPatchClassifierV2, self).__init__()
        self.neighbor_k = neighbor_k

    # def calculate_cos_similarity(self, query_fea, support_fea):
    #     # [3, 5, 180, 512]    [3, 75, 36, 512]
    #     # 后两个维度相乘表示 【36 512】*【512 180】=【36 180】 表示每个query块儿 同类中所有support块的相似度
    #     similarity_batch = []
    #     for i in range(len(support_fea)):
    #         # 遍历每一个batch
    #         similarity_matrix = []
    #         for query in query_fea[i]:
    #             # query_fea[i]表示每个batch中包含的所有 n_wq个query
    #             # support_fea[i]表示第 i 个batch中的所有 support类



    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, query_fea, support_fea):
        # [3, 5, 180, 512]    [3, 75, 36, 512]
        # 【batch, way, shot*patch_num, dim】    【batch, n_wq, patch_num, dim】
        # 后两个维度相乘表示 【36 512】*【512 180】=【36 180】 表示每个query块儿 同类中所有support块的相似度

        batch, n_wq, patch_num, dim = query_fea.shape
        _, n_way, _, dim = support_fea.shape

        Similarity_list = []


        for k in range(len(support_fea)):    # 遍历每一个batch
            Similarity_list_bag = []

            for i in range(n_wq):  # 遍历每一个query

                query = query_fea[k][i]   # 【n_patches, dim】   第k个batch的 第 i个 query
                # query_norm = torch.norm(query, 2, 1, True)
                # query = query / query_norm

                if torch.cuda.is_available():
                    inner_sim = torch.zeros(1, n_way).cuda()  # 创建一个空的向量，存放最终计算出的相似度

                for j in range(n_way):  # 遍历support中的每个类
                    support_class = support_fea[k][j]      # [n_support*n_patches, dim] 第k个batch的 第 j 个类

                    support_class = torch.transpose(support_class, 0, 1)     # [dim, n_support*n_patches]

                    # cosine similarity between a query sample and a support category  [36, 180]
                    innerproduct_matrix = query @ support_class  # 【n_patches, dim】* [dim, n_support*n_patches]

                    # 选出与每个 query_patch最相似的k个 support_patch    topk_value [36,5]
                    topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, dim=1)
                    inner_sim[0, j] = torch.mean(topk_value)    # topk_value、topk_index 【36, 3】 将全部的值加起来；

                Similarity_list_bag.append(inner_sim)

            Similarity_list_bag = torch.cat(Similarity_list_bag, dim=0)
            Similarity_list.append(Similarity_list_bag)   # [75,5]

        Similarity_list = torch.stack(Similarity_list, dim=0)  # [B, 75, 5]

        return Similarity_list  # [75, 5]


    def forward(self, query_fea, support_fea):
        # query_fea: torch.Size([3, 75, 36, 512]) 归一化后的
        # support_fea    [3, 5, 5, 36, 512]  归一化后的

        batch, way, shot, patch_num, dim = support_fea.shape
        _, n_wq, _, _= query_fea.shape

        support_fea = support_fea.view(batch, way, shot*patch_num, dim)    # [3, 5, 180, 512]

        # query_fea = query_fea.view(batch, n_wq, patch_num, dim)    # [3, 75, 36, 512]

        # 用bmm实现 批计算cos相似度，
        # 那么均输入4为的张量;
        # 结果表示每个query与 每个类的相似度
        Similarity_list = self.cal_cosinesimilarity(query_fea, support_fea)


        return Similarity_list


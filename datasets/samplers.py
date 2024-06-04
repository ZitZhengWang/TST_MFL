import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls    # 类别数
        self.n_per = n_per    #
        self.ep_per_batch = ep_per_batch    #

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))    # np.argwhere(label == c) 返回标签label中，值等于c的元素的索引
            # 把所有的验证图像按照相同的标签组织在一起；
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)    # 从所有的验证类别中，随机选择 n_cls个类别
                for c in classes:    # 遍历每一个验证类，
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)    # 在该类别中随机选择 n_per 个图像
                    episode.append(torch.from_numpy(l))    # 将一个类别中的图像添加到一个任务中
                episode = torch.stack(episode)    # 把一个list变成一个tensor的类型【5，16】，里面保存的是一个任务中所有图像的索引号
                batch.append(episode)    # 把一个任务添加到batch中
            batch = torch.stack(batch)    # bs * n_cls * n_per    【batch，n_way，num_s+q】
            yield batch.view(-1)


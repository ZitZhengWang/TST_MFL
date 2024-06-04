import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from . import few_shot


_log_path = None

def set_log_path(path):
    """
    函数的作用是将path设置成一个全局的变量，供其他文件使用（特别是日志文件）
    Args:
        path:

    Returns:

    """
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    # 提取路径最后的文件夹名称
    basename = os.path.basename(path.rstrip('/'))

    # 如果给定的路径已经存在，则确定是否删除旧路径；否则直接创建一个新的空文件夹
    # 优先级 and > or
    # 如果 remove标志为真且basename以 _开始   或者   用户输入不为 n， 则删除已有的文件，并且重新创新一个同名的空文件夹
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)    # shutil.rmtree(path)递归的删除指定路径中的所有文件
            os.makedirs(path)    #
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)

def time_str_6f(t):
    if t >= 3600:
        return '{:.6f}h'.format(t / 3600)
    if t >= 60:
        return '{:.6f}m'.format(t / 60)
    return '{:.6f}s'.format(t)

def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))    # [4,75,512] * [4, 512, 5] = [4, 75, 5]
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))    # 从数据集中选择出第i个样本标签对，并取出【0】的图像数据进行转换
    # 利用tensorboard对图像进行可视化
    writer.add_images('visualize_' + name, torch.stack(demo))    # 输入标题名称 和 一个batch的图像 torch.Size([16, 3, 80, 80])
    writer.flush()    # 刷新缓冲区


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


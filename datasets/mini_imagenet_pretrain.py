import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.datasets import register


@register('miniImageNet_pretrain')
class miniImageNet_pretrain(Dataset):
    """
    该类别专门用于为基于局部快的DN4准备数据集
    """
    def __init__(self, root_path="/home/zit/21class/ZhengWang/Datasets/MiniImagenet",
                 split='train',
                 image_size=80,
                 patch_size=None,
                 patch_num=None,
                 **kwargs):
        # 根据指定的 split 确定 加载的 数据部分的文件名称 split_file
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)

        # 通过指定的root_path 和 split_file 找到数据文件，并打开该文件；
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')

        # 提取需要的图像数据 和 对应的标签
        data = pack['data']
        label = pack['labels']

        # image_size = 80    # 设定图像的大小
        data = [Image.fromarray(x) for x in data]    # 将 nparray的图像数据用 PIL 打开；

        min_label = min(label)    # 获取最小的标签值 为 0
        label = [x - min_label for x in label]    # 将每个标签值 减去 最小的标签值
        # 将需要的数据初始化给类
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1    # 将最大标签值+1得到 总的类别数
        self.patch_size = patch_size
        self.patch_num = patch_num

        # ============================= 预处理操作 ========================================
        # 实例化 做归一化的transforms类
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)

        # 全局的图像使用默认的transform即可；
        # 设置默认的预处理的操作，包括：调整尺寸、中心裁剪、变换为张量 以及 归一化
        self.default_transform = transforms.Compose([
            # 如果只输入一个数字，则会把图像中的短边resize成对应尺寸，并且按照原图像的比例缩放长边
            # 我希望的是尽量保持图像原有的比例，不去拉升图像，所以只输入一个数更合理；
            transforms.Resize(int(image_size*1.15)),    # 先Resiez的大一点，然后再中心裁剪成想要的尺寸
            transforms.CenterCrop([image_size, image_size]),
            transforms.ToTensor(),
            normalize,  
        ])

        # 根据设定的 augment参数，设置预处理操作
        jitter_params = dict(brightness=0.4, contrast=0.4, saturation=0.4)

        augment = kwargs.get('augment')
        if augment == 'Crop_Resize':
            # 先随机裁剪，然后调整尺寸为指定大小；（图像比例可能变形）
            # 包括：随机调整尺寸并裁剪、随机左右翻转、变成张量、归一化
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(**jitter_params),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'Resize_Crop':
            # 先裁剪出整张图像，然后裁剪出整个图像块；（图片比例不会变形）
            # 包括：调整尺寸、随机裁剪、随机左右翻转、变成张量、归一化
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.ColorJitter(**jitter_params),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        # 局部块的处理思路：1、先增强所有图像，然后再采样
        local_augment = kwargs.get('local_augment')
        if local_augment == "local_Resize":
            self.localPatchCrop = transforms.RandomResizedCrop(patch_size)
        elif local_augment== "local_Crop":
            self.localPatchCrop = transforms.RandomCrop(patch_size)
        else:
            self.localPatchCrop = None

        def convert_raw(x):
            # 将字典中的均值和标准差转换为和 图像数据 一样的 Tensor类型，并做变换 x * std + mean（将归一化后的图像转换为原始图像）
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # self.data[i] 是一张PIL打开后的图像
        img = self.data[i]

        # global_image = self.default_transform(self.data[i])    # 从原始图像经过一个默认的数据增强得到全局图像；

        img = self.transform(img)    # 把一张原始图像经过数据增强 or 默认的变换 得到预处理后的图像

        global_image = img

        # 将一张图像随机裁剪成 patch_num 个图像
        if self.localPatchCrop != None:
            local_images = []
            local_labels = []
            for number in range(self.patch_num):
                local_images.append(self.localPatchCrop(img))
                local_labels.append(self.label[i])
            local_images = torch.stack(local_images)
        else:
            local_images = img

        local_labels = torch.tensor(local_labels)

        return global_image, local_images, self.label[i], local_labels


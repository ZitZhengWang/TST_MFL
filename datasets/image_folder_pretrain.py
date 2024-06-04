import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .datasets import register
# CUB /home/zit/21class/ZhengWang/Datasets/Caltech-UCSD Birds-200 2011/CUB_200_2011/images
#     /home/zit/21class/ZhengWang/Datasets/Caltech-UCSD Birds-200 2011/CUB_200_2011/images
@register('image_folder_pretrain')
class ImageFolderPretrain(Dataset):
    """
    从按类划分的数据集中构建Dataset类
    """
    def __init__(self, root_path,
                 image_size=224,
                 patch_size=None,
                 patch_num=None,
                 box_size=256,
                 **kwargs):
        if box_size is None:
            box_size = image_size

        self.patch_num = patch_num
        self.filepaths = []    # 用于保存所有图像数据的列表
        self.label = []    # 注意：每个split的标签都是从 0开始生成的；
        classes = sorted(os.listdir(root_path))    # 将类文件夹保存为列表

        if kwargs.get('split'):
            path = kwargs.get('split_file')
            if path is None:
                path = os.path.join(
                        os.path.dirname(root_path.rstrip('/')),"splits", f'{kwargs["split"]}.json')

            # 读取json文件中的内容
            split = json.load(open(path, 'r'))    # 加载json形式的split文件,打开的split.json也是一个字典
            classes = sorted(split["label_names"])    # 从split中提取出 kwargs中指定的值部分 label_names

        # 遍历各类，其中 c 表示具体的类
        for i, c in enumerate(classes):
            # 遍历以类名为文件夹名的路径，将每一张图像添加到 filepaths 列表中，并把遍历类的顺序作为其标签；
            for filename in sorted(os.listdir(os.path.join(root_path, c))):
                self.filepaths.append(os.path.join(root_path, c, filename))    #
                self.label.append(i)
        self.n_classes = max(self.label) + 1

        # ============================== 预处理操作 ==========================================

        # 实例化 做归一化的transforms类
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)

        # 全局的图像使用默认的transform即可；
        # 设置默认的预处理的操作，包括：调整尺寸、中心裁剪、变换为张量 以及 归一化
        self.default_transform = transforms.Compose([
            # 如果只输入一个数字，则会把图像中的短边resize成对应尺寸，并且按照原图像的比例缩放长边
            # 我希望的是尽量保持图像原有的比例，不去拉升图像，所以只输入一个数更合理；
            transforms.Resize(int(image_size * 1.15)),  # 先Resiez的大一点，然后再中心裁剪成想要的尺寸
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
        elif local_augment == "local_Crop":
            self.localPatchCrop = transforms.RandomCrop(patch_size)
        else:
            self.localPatchCrop = None

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.filepaths)    # 图像的总数量

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')

        # global_image = self.default_transform(img)    # 从原始图像经过一个默认的数据增强得到全局图像；

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
            # local_labels = torch.stack(local_labels)
        else:
            local_images = img

        local_labels = torch.tensor(local_labels)

        return global_image, local_images, self.label[i], local_labels


import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from utils.criterion import KLDivLoss, DistillLoss, DiscrepancyLoss

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #总是报显存不足的问题，是因为碎片没完全释放
if hasattr(torch.cuda, 'empty_cache'):
   torch.cuda.empty_cache()


def main(config):
    # 如果没有指定保存的文件名，则根据数据集以及主干网络、分类器等信息生成一个文件名；
    svname = args.name
    if svname is None:
        svname = 'pretraining_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr

    # 为文件名添加额外的标签、注释
    if args.tag is not None:
        svname += '_' + args.tag

    # 由指定路径 和 文件名生成保存的路径；
    save_path = os.path.join('./save/pretrained', svname)

    # 检查保存路径是否已经存在，如果存在是否删除旧路径
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)    # 设置日志文件的保存路径
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))    # 设置tensorboard的保存路径

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))    # 保存实验中 ymal 文件的配置情况（记录实验参数）

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)

    # 日志记录 数据集中一个图像经过处理后的样本的形状、数据的个数 以及 总类别数
    # train_dataset[0]表示从Dataset的getitem中获取索引为0结果，结果是一个元组（self.transform(self.data[i]), self.label[i]）包含处理后的图像及其标签；
    utils.log('train dataset: {}  {}  {}  {}  (x{}), {}'.format(
        train_dataset[0][0].shape, train_dataset[0][1].shape,
        train_dataset[0][2], train_dataset[0][3].shape,
        len(train_dataset), train_dataset.n_classes))

    # 如果 config中配置了 可视化数据集的选项，则利用tensorboard进行可视化
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val 传统验证数据集准备
    if config.get('val_dataset'):
        eval_val = True
        # 验证集的数据使用默认的预处理操作，没有数据增强
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=config["num_workers"], pin_memory=True)

        utils.log('val dataset: {}  {}  {}  {}  (x{}), {}'.format(
            val_dataset[0][0].shape, val_dataset[0][1].shape,
            val_dataset[0][2], val_dataset[0][3].shape,
            len(val_dataset), val_dataset.n_classes))

        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # few-shot eval 小样本任务验证数据集准备
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')   # 每隔ef_epoch 个 epoch做一次 few-shot验证
        if ef_epoch is None:
            ef_epoch = 10
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'], **config['fs_dataset_args'])

        utils.log('fs dataset: {}  {}  {}  {}  (x{}), {}'.format(
            fs_dataset[0][0].shape, fs_dataset[0][1].shape,
            fs_dataset[0][2], fs_dataset[0][3].shape,
            len(fs_dataset), fs_dataset.n_classes))

        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_way = config["n_way"]
        n_query = config["n_query"]
        n_shots = config["n_shots"]   # 1-shot 和 5-shot 都要测试
        fs_loaders = []

        # 构建出 1-shot 和 5-shot 的dataloader
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                fs_dataset.label,    # 20个测试类别的所有图像，共计12000张，的标签；
                200,    # 应该表示采样的任务数量
                n_way,    # n_way
                n_shot + n_query,    # n_shot + n_query
                ep_per_batch=config["ep_per_batch"])    # 每个batch包含多少个任务
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=config["num_workers"], pin_memory=True)

            fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])    # 加载模型，需要指定模型的名字和相应的参数

    if eval_fs:
        # fs_model 模型使用 和 model一样的特征提取器，只是分类器是一个余弦分类器
        fs_model = models.make(config['fs_model'], **config['fs_model_args'])

    # 多GPU的并行设置
    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    # 计算模型的参数量，并且记录在日志中
    utils.log('num params pretrain model: {}'.format(utils.compute_n_params(model)))
    utils.log('num params FusionNet model: {}'.format(utils.compute_n_params(fs_model)))
    utils.log("epoch, train |loss|total_acc|global_acc|local_acc|, val |loss|total_acc|global_acc|local_acc|, fs|1:acc|5:acc|, time used_time/total_time")

    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(),
                                                   config['optimizer'],
                                                   **config['optimizer_args'])

    # =================================== 损失函数准备 ==================================
    # 实例化KL计算的类
    klDivLoss = KLDivLoss(T=3)

    # 实例化 蒸馏损失
    distillLoss = DistillLoss(klDivLoss)
    discrepancyLoss = DiscrepancyLoss(num_classes=train_dataset.n_classes)

    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.use_amp:
        scalar = GradScaler()  # 1.这是你需要添加的

    for epoch in range(1, max_epoch + 1 + 1):    # [1,2,3,...,101]
        # 如果是最后一个额外的 epoch
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break

            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(
                    train_dataset, config['batch_size'], shuffle=True,
                    num_workers=config["num_workers"], pin_memory=True)

        timer_epoch.s()

        # 设置几个关键字，分别表示 训练损失、训练准确率、验证损失、验证准确率  #
        aves_keys = ['tl', 'ta', 't_ga', 't_la', 'vl', 'va', 'v_ga', 'v_la']


        # 添加 1-shot 和5-shot 的准确率关键字
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]

        trlog = dict()
        for k in aves_keys:
            trlog[k] = []

        # 根据刚才设置的关键字生成对应的Averager
        aves = {k: utils.Averager() for k in aves_keys}

        # ====================================== 模型训练 train =============================================
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)    # tensorboard记录学习了的变化情况

        # 所有train_loader中的所有数据训练一遍
        for global_imgs, local_imgs, global_labels, local_labels in tqdm(train_loader, desc='train', leave=False):
            # global_imgs [128,3,84,84]    local_imgs [128,36,3,26,26]
            # global_labels [128, ]    local_labels [128, 36]
            global_imgs, local_imgs, global_labels, local_labels = global_imgs.cuda(), local_imgs.cuda(), global_labels.cuda(), local_labels.cuda()

            optimizer.zero_grad()
            if args.use_amp:
                with torch.cuda.amp.autocast():  # 2.这是你需要添加的
                    # 输入 global_imgs 和 local_imgs -> [128, 100] [128, 36, 100]
                    g_logits, l_logits = model(global_imgs, local_imgs)    #

                    # 计算 l_avg_logits
                    l_avg_logits = l_logits.mean(dim=1)    # [128, 36, 100] -> [128, 100]

                    # 计算蒸馏损失
                    if config["Lambda"] is not None:
                        loss_distill = distillLoss(g_logits, l_avg_logits)
                    else:
                        config["Lambda"] = 0
                        loss_distill = 0

                    # 计算差异度损失
                    if config["Gamma"] is not None:
                        loss_discre = discrepancyLoss(l_logits, l_avg_logits)
                    else:
                        config["Gamma"] = 0
                        loss_discre = 0

                    # 分别计算交叉熵损失
                    local_labels = local_labels.view(-1,)

                    n_batch, n_patch, dim = l_logits.size()
                    l_logits = l_logits.view(-1, dim)
                    loss_l_ce = F.cross_entropy(l_logits, local_labels)    # [128*36, 100] || [128*36, ]
                    loss_g_ce = F.cross_entropy(g_logits, global_labels)    # [128, 100] || [128, ]
                    loss_CE = config["Beta"] * loss_g_ce + (1 - config["Beta"]) * loss_l_ce

                    total_loss = loss_CE + config["Lambda"] * loss_distill + config["Gamma"] * loss_discre

                scalar.scale(total_loss).backward()  # 3.这是你需要添加的,进行损失缩放工作
                scalar.step(optimizer)  # 4.这是你需要添加的
                scalar.update()  # 5.这是你需要添加的

            else:
                # 输入 global_imgs 和 local_imgs -> [128, 100] [128, 36, 100]
                g_logits, l_logits = model(global_imgs, local_imgs)  #

                # 计算 l_avg_logits
                l_avg_logits = l_logits.mean(dim=1)  # [128, 36, 100] -> [128, 100]

                # 计算蒸馏损失
                if config["Lambda"] is not None:
                    loss_distill = distillLoss(g_logits, l_avg_logits)
                else:
                    config["Lambda"] = 0
                    loss_distill = 0

                # 计算差异度损失
                if config["Gamma"] is not None:
                    loss_discre = discrepancyLoss(l_logits, l_avg_logits)
                else:
                    config["Gamma"] = 0
                    loss_discre = 0

                # 分别计算交叉熵损失
                local_labels = local_labels.view(-1, )

                n_batch, n_patch, dim = l_logits.size()
                l_logits = l_logits.view(-1, dim)
                loss_l_ce = F.cross_entropy(l_logits, local_labels)  # [128*36, 100] || [128*36, ]
                loss_g_ce = F.cross_entropy(g_logits, global_labels)  # [128, 100] || [128, ]
                loss_CE = config["Beta"] * loss_g_ce + (1 - config["Beta"]) * loss_l_ce

                total_loss = loss_CE + config["Lambda"] * loss_distill + config["Gamma"] * loss_discre

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # 计算分类准确率
            g_acc = utils.compute_acc(g_logits, global_labels)
            l_acc = utils.compute_acc(l_logits, local_labels)

            total_logits = torch.cat((g_logits, l_logits), dim=0)
            total_labels = torch.cat((global_labels, local_labels), dim=0)
            total_acc = utils.compute_acc(total_logits, total_labels)

            aves['tl'].add(total_loss.item())
            aves['ta'].add(total_acc)
            aves['t_ga'].add(g_acc)
            aves['t_la'].add(l_acc)

        # ====================================== 模型验证 eval ==============================================
        if eval_val:
            model.eval()
            # val_loader中的所有数据训练一遍
            for global_imgs, local_imgs, global_labels, local_labels in tqdm(val_loader, desc='val', leave=False):
                global_imgs, local_imgs, global_labels, local_labels = global_imgs.cuda(), local_imgs.cuda(), global_labels.cuda(), local_labels.cuda()

                with torch.no_grad():
                    # 输入 global_imgs 和 local_imgs -> [128, 100] [128, 36, 100]
                    g_logits, l_logits = model(global_imgs, local_imgs)    #

                    # 计算 l_avg_logits
                    l_avg_logits = l_logits.mean(dim=1)    # [128, 36, 100] -> [128, 100]

                    # 计算蒸馏损失
                    if config["Lambda"] is not None:
                        loss_distill = distillLoss(g_logits, l_avg_logits)
                    else:
                        config["Lambda"] = 0
                        loss_distill = 0

                    # 计算差异度损失
                    if config["Gamma"] is not None:
                        loss_discre = discrepancyLoss(l_logits, l_avg_logits)
                    else:
                        config["Gamma"] = 0
                        loss_discre = 0

                    # 分别计算交叉熵损失
                    local_labels = local_labels.view(-1,)

                    n_batch, n_patch, dim = l_logits.size()
                    l_logits = l_logits.view(-1, dim)
                    loss_l_ce = F.cross_entropy(l_logits, local_labels)    # [128*36, 100] || [128*36, ]
                    loss_g_ce = F.cross_entropy(g_logits, global_labels)    # [128, 100] || [128, ]
                    loss_CE = config["Beta"] * loss_g_ce + (1 - config["Beta"]) * loss_l_ce

                    total_loss = loss_CE + config["Lambda"] * loss_distill + config["Gamma"] * loss_discre

                    # 计算分类准确率
                    g_acc = utils.compute_acc(g_logits, global_labels)
                    l_acc = utils.compute_acc(l_logits, local_labels)

                    total_logits = torch.cat((g_logits, l_logits), dim=0)
                    total_labels = torch.cat((global_labels, local_labels), dim=0)
                    total_acc = utils.compute_acc(total_logits, total_labels)

                aves['vl'].add(total_loss.item())
                aves['va'].add(total_acc)
                aves['v_ga'].add(g_acc)
                aves['v_la'].add(l_acc)

        # ========================================= 小样本验证 ===========================================
        # 以前代码的问题，fs_model的参数没有更新
        if eval_fs and (epoch == 1 or epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots):
                config["n_shot"] = n_shot
                # 设置固定的种子，做few-shot验证
                np.random.seed(0)    # data [320, 3, 80, 80]
                for global_imgs, local_imgs, global_labels, local_labels in tqdm(fs_loaders[i],
                                                                                 desc='fs-' + str(n_shot),
                                                                                 leave=False):
                    # global_imgs [320,3,84,84]    local_imgs [320,36,3,26,26]
                    with torch.no_grad():
                        # 把model的参数拿过来，直接把数据送入fs_model
                        fs_model.GlobalNet.encoder = model.GlobalNet.encoder
                        fs_model.LocalNet.encoder = model.LocalNet.encoder
                        fs_model.LocalNet.transformerEncoderlayer = model.LocalNet.transformerEncoderlayer
                        logits = fs_model(global_imgs, local_imgs, config)

                        label = fs.make_nk_label(config["n_way"],
                                                 config["n_query"],
                                                 ep_per_batch=config["ep_per_batch"]).cuda()  # [300, ]

                        acc = utils.compute_acc(logits, label)   # 计算一个batch 4个任务中query 的准确率

                    aves['fsa-' + str(n_shot)].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()    # 返回数值结果，存在字典中
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())    # 每个epoch的用时
        t_used = utils.time_str(timer_used.t())    # 使用的时间
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)    # 估计总用时

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'

        # 显示的内容依次为：epoch数，训练损失，训练准确率
        log_str = 'epoch {}, train |{:.4f}||{:.4f}|{:.4f}|{:.4f}|'.format(
                epoch_str, aves['tl'], aves['ta'], aves['t_ga'], aves['t_la'])

        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train_total': aves['ta']}, epoch)
        writer.add_scalars('acc', {'train_global': aves['t_ga']}, epoch)
        writer.add_scalars('acc', {'train_local': aves['t_la']}, epoch)

        if eval_val:
            # 显示内容分别为验证损失，验证准确率
            log_str += ', val |{:.4f}||{:.4f}|{:.4f}|{:.4f}|'.format(
                aves['vl'], aves['va'], aves['v_ga'], aves['v_la'])

            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val_total': aves['va']}, epoch)
            writer.add_scalars('acc', {'val_global': aves['v_ga']}, epoch)
            writer.add_scalars('acc', {'val_local': aves['v_la']}, epoch)

        if eval_fs and (epoch == 1 or epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += '|{}: {:.4f}|'.format(n_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            # 追加显示 一个epoch训练用时，一轮用时，估计总用时
            log_str += ', {}||{}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        # 保存训练过程 优化器的状态
        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }    # 其他保存的事项

        save_obj = {
            'file': __file__,    # 保存运行的文件名
            'config': config,    # 保存参数设置

            'model': config['model'],    # 保存模型名称
            'model_args': config['model_args'],    # 保存模型的参数设置
            'model_sd': model_.state_dict(),    # 保存模型的参数状态

            'training': training,   # 以及优化器的参数状态
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    # 通过argparse 和 yaml解析实验中的参数；
    # argparse主要包括需要经常变动的参数；
    # yaml中主要包括数据加载、模型相关的参数，不需要经常变动；
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/pretrainOnCUB.yaml", help="配置文件的路径")
    parser.add_argument('--name', default="FusionNet")
    parser.add_argument('--tag', default="TEST_pretrain")
    parser.add_argument('--lr', type=float, default=0.001, help="")

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_amp', action="store_false", help="使用混合精度训练")

    parser.add_argument('--Beta', type=float, default=0.5, help="全局和局部的比例（全局的系数）")
    parser.add_argument('--Lambda', type=float, default=None, help="蒸馏损失系数")
    parser.add_argument('--Gamma', type=float, default=None, help="差异度损失系数")

    args = parser.parse_args()

    # 将yaml中的参数加载为一个字典的形式，存放在config中；
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    config["optimizer_args"]['lr'] = args.lr

    config["Beta"] = args.Beta
    config["Lambda"] = args.Lambda
    config["Gamma"] = args.Gamma

    # 配置多GPU的情况
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    # 设置GPU
    utils.set_gpu(args.gpu)

    main(config)


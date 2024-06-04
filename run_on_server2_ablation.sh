# 做关于网络框架的消融实验，
# 不重新预训练网络，就在预训练好的模型上加载部分参数实验
# 其他事项：
#     仅元训练
#     是和完整的模型对比，三种损失
#     CUB数据集上
#
# 仅全局主干网络 5shot 1shot
python metaTraining_ablation.py --config "configs/metaTrainingOnCUBV4_net_ablation.yaml" --name "AblationOnlyGlobal_CUB_5shot_max" --tag "1e-6" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 5 --lr 1e-6 --max_epoch 50 --NetMode "Global"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/AblationOnlyGlobal_CUB_5shot_max_1e-6/max-va.pth"

python metaTraining_ablation.py --config "configs/metaTrainingOnCUBV4_net_ablation.yaml" --name "AblationOnlyGlobal_CUB_1shot_max" --tag "1e-6" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 1 --lr 1e-6 --max_epoch 50 --NetMode "Global"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/AblationOnlyGlobal_CUB_1shot_max_1e-6/max-va.pth"


# 仅局部主干网络
python metaTraining_ablation.py --config "configs/metaTrainingOnCUBV4_net_ablation.yaml" --name "AblationOnlyLocal_CUB_5shot_max" --tag "1e-6" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 5 --lr 1e-6 --max_epoch 50 --NetMode "Local"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/AblationOnlyLocal_CUB_5shot_max_1e-6/max-va.pth"

python metaTraining_ablation.py --config "configs/metaTrainingOnCUBV4_net_ablation.yaml" --name "AblationOnlyLocal_CUB_1shot_max" --tag "1e-6" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 1 --lr 1e-6 --max_epoch 50 --NetMode "Local"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/AblationOnlyLocal_CUB_1shot_max_1e-6/max-va.pth"

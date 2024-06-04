 # 消融实验，来验证融合方式的有效性——关于各种融合方式的消融
# 主试验是度量融合模块，结果已经有了——就取CUB-上的Loss++的结果就好；
# 那么预训练模型就是  Loss4

# 方式一：手动加权融合——V1版本吗？
# 5-shot 和 1-shot
python metaTraining_fusionAblation.py --config "configs/metaTrainingOnCUBV4_fusion_ablation.yaml" --name "FusionAblation_CUB_5shot_Manual" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth"  --n_shot 5 --lr 1e-5 --max_epoch 50 --FusionMode "Manual"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/FusionAblation_CUB_5shot_Manual_1e-5/max-va.pth"

python metaTraining_fusionAblation.py --config "configs/metaTrainingOnCUBV4_fusion_ablation.yaml" --name "FusionAblation_CUB_1shot_Manual" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth"  --n_shot 1 --lr 1e-5 --max_epoch 50 --FusionMode "Manual"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/FusionAblation_CUB_1shot_Manual_1e-5/max-va.pth"

# 方式二：全连接层融合
# 5-shot 和 1-shot
python metaTraining_fusionAblation.py --config "configs/metaTrainingOnCUBV4_fusion_ablation.yaml" --name "FusionAblation_CUB_5shot_FC" --tag "1e-4" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth"  --n_shot 5 --lr 1e-4 --max_epoch 50 --FusionMode "FC"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/FusionAblation_CUB_5shot_FC_1e-4/max-va.pth"

python metaTraining_fusionAblation.py --config "configs/metaTrainingOnCUBV4_fusion_ablation.yaml" --name "FusionAblation_CUB_1shot_FC" --tag "1e-4" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth"  --n_shot 1 --lr 1e-4 --max_epoch 50 --FusionMode "FC"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/FusionAblation_CUB_1shot_FC_1e-4/max-va.pth"

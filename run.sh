
# ======================================================= CUB ==========================================================
# CUB数据集预训练
python pretrain.py --config "configs/pretrainOnCUB.yaml" --name "pretrainedFusionNetLossCELdstLdsc" --tag "4" --lr 0.0001 --Lambda 0.2 --Gamma 0.2

# CUB数据集元训练
# CUB的5shot的结果
python metaTraining.py --config "configs/metaTrainingOnCUBV4.yaml" --name "V4+_CUB_5shot_max" --tag "1e-6" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 5 --lr 1e-6 --max_epoch 50
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/V4+_CUB_5shot_max_1e-6/max-va.pth"

# CUB 1-shot
python metaTraining.py --config "configs/metaTrainingOnCUBV4.yaml" --name "FusionNetV4_CUB_1shot" --tag "4" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_4/max-va.pth" --n_shot 1 --lr 0.0001
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/FusionNetV4_CUB_1shot_4/max-va.pth"

# ======================================================= Dogs ==========================================================
python pretrain.py --config "configs/pretrainOnDogs.yaml" --name "pretrainedFusionNetLossCELdstLdsc" --tag "Dogs_4" --lr 0.0001 --Lambda 0.2 --Gamma 0.2
# Dogs数据集的 5-shot
python metaTraining.py --config "configs/metaTrainingOnDogsV4.yaml" --name "V4+_Dogs_5shot_max" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_Dogs_4/max-va.pth" --n_shot 5 --lr 1e-5 --max_epoch 50
python testing.py --config "configs/testingOnDogs.yaml" --n_shot 5 --loadpath "./save/metaTrained/V4+_Dogs_5shot_max_1e-5/max-va.pth"


# Dogs数据集的 1-shot
python metaTraining.py --config "configs/metaTrainingOnDogsV4.yaml" --name "FusionNetV4_Dogs_1shot" --tag "4" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_Dogs_4/max-va.pth" --n_shot 1 --lr 0.0001
python testing.py --config "configs/testingOnDogs.yaml" --n_shot 1 --loadpath "./save/metaTrained/FusionNetV4_Dogs_1shot_4/max-va.pth"


# ======================================================= Cars ==========================================================
python pretrain.py --config "configs/pretrainOnCars.yaml" --name "pretrainedFusionNetLossCELdstLdsc" --tag "Cars_2" --lr 0.0001 --Lambda 0.2 --Gamma 0.2

# Cars 5-shot
python metaTraining.py --config "configs/metaTrainingOnCarsV4.yaml" --name "V4+_Cars_5shot_max" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_Cars_2/max-va.pth" --n_shot 5 --lr 1e-5 --max_epoch 50
python testing.py --config "configs/testingOnCars.yaml" --n_shot 5 --loadpath "./save/metaTrained/V4+_Cars_5shot_max_1e-5/max-va.pth"


# Cars 1-shot
python metaTraining.py --config "configs/metaTrainingOnCarsV4.yaml" --name "FusionNetV4_Cars_1shot" --tag "4" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_Cars_2/max-va.pth" --n_shot 1 --lr 0.0001
python testing.py --config "configs/testingOnCars.yaml" --n_shot 1 --loadpath "./save/metaTrained/FusionNetV4_Cars_1shot_4/max-va.pth"


# ======================================================= Mini ==========================================================
python pretrain.py --config "configs/pretrainOnMini.yaml" --name "pretrainedFusionNetLossCELdstLdsc" --tag "mini_2" --lr 0.0001 --Lambda 0.2 --Gamma 0.2

# miniImageNet 5-shot
python metaTraining.py --config "configs/metaTrainingOnMiniV4.yaml" --name "V4+_mini_5shot_max" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_mini_2/max-va.pth" --n_shot 5 --lr 1e-5 --max_epoch 50
python testing.py --config "configs/testingOnMini.yaml" --n_shot 5 --loadpath "./save/metaTrained/V4+_mini_5shot_max_1e-5/max-va.pth"

# mini 数据集 的 1-shot
python metaTraining.py --config "configs/metaTrainingOnMiniV4.yaml" --name "V4+_mini_1shot_max" --tag "1e-5" --load_encoder "save/pretrained/pretrainedFusionNetLossCELdstLdsc_mini_2/max-va.pth" --n_shot 1 --lr 1e-5 --max_epoch 50
python testing.py --config "configs/testingOnMini.yaml" --n_shot 1 --loadpath "./save/metaTrained/V4+_mini_1shot_max_1e-5/max-va.pth"


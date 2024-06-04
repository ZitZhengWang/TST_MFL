# 正式实验，消融实验的结果
# 做不使用预训练，直接元训练模型的结果；
# 先在CUB上实验，消融实验并不需要在所有数据集上实验，最多两个就够了；
python metaTraining.py --config "configs/metaTrainingOnCUBV4.yaml" --name "V4_CUB_noPretrain_5shot" --tag "CR_1e-5" --n_shot 5 --lr 1e-5 --max_epoch 300 --train_augment "Crop_Resize"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 5 --loadpath "./save/metaTrained/V4_CUB_noPretrain_5shot_CR_1e-5/max-va.pth" --test-epochs 5

python metaTraining.py --config "configs/metaTrainingOnCUBV4.yaml" --name "V4_CUB_noPretrain_1shot" --tag "CR_1e-5" --n_shot 1 --lr 1e-5 --max_epoch 300 --train_augment "Crop_Resize"
python testing.py --config "configs/testingOnCUB.yaml" --n_shot 1 --loadpath "./save/metaTrained/V4_CUB_noPretrain_1shot_CR_1e-5/max-va.pth" --test-epochs 5
## Introduction

This document describes our submission to the [STSR 2025 Task 1](https://www.codabench.org/competitions/6468/). 
The model is our proposed UMamba2 pretrained via self-supervised learning, trained with consistency regularization and second stage pseudolabel supervision.
The final models are trained for 1500 epochs with a batch size of 2.
For more details, please see our [paper] (coming soon).

## Dataset Preparation
After downloading the datasets, move the dataset to your `nnUNet_raw` folder, then move all labeled images into `Images` and `Masks`, and unlabeled images into `Unlabeled`.
Adapt and run the [dataset conversion script](../../../nnunetv2/dataset_conversion/Dataset319_STSR25_All.py). 
The script mainly renames and create the necessary files for nnUNet, and define the splits in `splits_final.json`.

## Extract fingerprint:
`nnUNetv2_extract_fingerprint -d 319 -np 48`

## Run planning:
`nnUNetv2_plan_experiment -d 319 -pl nnUNetPlannerResEncL_torchres`

## Edit the plans files
Add the following configuration to the generated plans file
```json
        "3d_fullres_torchres_mambabot2_ps160x256x256_bs1": {"inherits_from":"3d_fullres","data_identifier": "nnUNetPlans_3d_fullres_torchres_ctnorm","normalization_schemes":["CTNormalization"],"batch_dice": false,"batch_size": 1,"patch_size": [160,256,256],"architecture":{"network_class_name":"nnunetv2.nets.UMambaBot2","arch_kwargs":{"n_stages":7,"features_per_stage":[32,64,128,256,320,320,320],"conv_op":"torch.nn.modules.conv.Conv3d","kernel_sizes":[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],"strides":[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]],"n_blocks_per_stage":[1,3,4,6,6,6,6],"n_conv_per_stage_decoder":[1,1,1,1,1,1],"conv_bias":true,"norm_op":"torch.nn.modules.instancenorm.InstanceNorm3d","norm_op_kwargs":{"eps":0.00001,"affine":true},"dropout_op":null,"dropout_op_kwargs":null,"nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"inplace":true}},"_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}}
```

The changes are mainly changing the architecture from the ResEncUNet to our UMamba2, increasing the patch size, use 'CTNormalization' for data normalization,
increasing the stages to 7, enabling it to make effective use of the larger input.

## Preprocessing
First, run the usual nnUNet preprocessing with:
```bash
nnUNetv2_preprocess -d 319 -c 3d_fullres_torchres_mambabot2_ps160x256x256_bs1 -plans_name nnUNetResEncUNetLPlans_torchres -np 48
```
Then, adapt `preprocess_unlabeled.py` and run preprocessing for the unlabeled data. 
This creates dummy segmentation ground truth (not used for training) for the unlabeled data so that nnUNet can load them properly for semi-supervised learning.
For the training script to utilize the unlabeled data properly, ensure that `splits_final.json` (should be automatically created in the dataset conversion script)
contains the proper splits with keys `train`, `unlabeled` and `val`.

Your final nnUNet_preprocessed directory should look like this:
```
└── Dataset319_STSR2025
    ├── gt_segmentations
    │   └── STS25_Train_Labeled_xxxx.nii.gz
    ├── nnUNetPlans_3d_fullres_torchres_ctnorm
    │   ├── STS25_Train_Labeled_xxxx{.b2nd,.pkl,_seg.b2nd}
    │   │   ...
    │   └── STS25_Train_Unlabeled_xxxx{.b2nd,.pkl,_seg.b2nd
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── nnUNetResEncUNetLPlans_torchres.json
    └── splits_final.json
```

## Training
Models are initialized using our pretrained weights via self-supervised learning. 
You can download them from [here](https://drive.google.com/drive/folders/1xhUkHCpo_50sNWvGH9CrN8Ws0hSjoa_k?usp=sharing) or run your own pretraining with the codes [here](../Pretrain_DAE). 
The codes are adapted from [DAE](https://github.com/Project-MONAI/research-contributions/tree/main/DAE/Pretrain_full_contrast).
Modify `data/data_pretrain.py` to use the STSR dataset and run the following command for pretraining:
```bash
python main_runner.py --batch_size=2 --sw_batch_size=1 --epoch=200 --mask_patch_size=16 --loss_type=all_img --base_lr=5e-5 --min_lr=5e-6 --warmpup_epoch=4 --warmup_lr=5e-7 --cache_dataset --cache_rate=0.2 --model_type=nnunet --save_freq=5 --log_dir="./output/umamba2_st25/logdir" --output="./output/umamba2_st25" --out_channels=1 --choice "all" --mm_con 0.03 --temperature 0.5 --img_size 256 --roi_x 160 --roi_y 256 --roi_z 256 --mask_ratio 0.3  --wandb --nnunet_plan <path to plan> --nnunet_conf <configuration name>
```

Then, we train the models on all training cases using gradient accumulation to simulate batch size of 2. For experimenting, create train and validation split manually.
```bash
nnUNetv2_train 319 3d_fullres_torchres_mambabot2_ps160x256x256_bs1 0 -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_STSR_Task1_accum2 -pretrained_weights <path to your pretarined weights>
```

We recommend to increase the number of processes used for data augmentation. Otherwise, you can run into CPU bottlenecks.
Use `export nnUNet_n_proc_DA={number_of_cpus}`.

Then, run second stage training by enabling pseudolabel using trainer `nnUNetTrainer_STSR_Task1_accum2_cont`.
Set the `-pretrained_weights` argument of `nnUNetv2_train` command to the final checkpoint of the stage 1 training and 
the modified `load_pretrained_weights.py` will load the weights correctly.

## Inference
Run inference with the inference script [here](optimized_inference.py).

Use [`documentation/competitions/optimize_checkpoint.py`](../optimize_checkpoint.py) to reduce the 
size of the checkpoint file.

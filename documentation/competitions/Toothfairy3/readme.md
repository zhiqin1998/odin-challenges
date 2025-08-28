## Introduction

This document describes our submission to both of the tasks of the [Toothfairy3 Challenge](https://toothfairy3.grand-challenge.org/). 
For Task 1, the model is our proposed UMamba2 while for Task2, the model is the UMamba2 + cross attention blocks with a SAM-style point encoder.
The final models are trained for 1500 epochs with a batch size of 2.
For more details, please see our [paper] (coming soon).

## Dataset Preparation
Download the raw dataset from the [Ditto](https://ditto.ing.unimore.it/toothfairy3/) webpage.

Adapt and run the [dataset conversion script](../../../nnunetv2/dataset_conversion/Dataset119_ToothFairy3_All.py). 
This script removes the unused label ids and combine all pulp classes into a single class.
The final mapping used for Task 1 is `mapping_DS119_singlepulp` and for Task 2, it is `mapping_DS219`.
Your nnUNet_raw directory should be similar to:
```
nnUNet_raw
  ├── Dataset{task1_dataset_id}_ToothFairy3
  │   ├── imagesTr
  │   ├── labeslTr
  │   └── dataset.json
  └── Dataset{task1_dataset_id}_ToothFairy3_IAN
      ├── imagesTr
      ├── labeslTr
      ├── ToothFairy3_clicks
      └── dataset.json
```

## Extract Dataset Fingerprint:
`nnUNetv2_extract_fingerprint -d {dataset_id} -np 48`

## Generate Experiment Plans:
`nnUNetv2_plan_experiment -d {dataset_id} -pl nnUNetPlannerResEncL_torchres`

## Edit the Plans
Add the following configuration to the generated plans file, this will use the corresponding UMamba2 models with 7 stages,
use 'CTNormalization' for data normalization, update the batch size and patch size.
- For Task 1:
    ```json
            "3d_fullres_torchres_mambabot2_ps128x256x256_bs1": {"inherits_from":"3d_fullres","data_identifier": "nnUNetPlans_3d_fullres_torchres_ctnorm","normalization_schemes": ["CTNormalization"],"batch_size":1,"patch_size":[128,256,256],"architecture":{"network_class_name":"nnunetv2.nets.UMambaBot2","arch_kwargs":{"n_stages":7,"features_per_stage":[32,64,128,256,320,320,320],"conv_op":"torch.nn.modules.conv.Conv3d","kernel_sizes":[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],"strides":[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]],"n_blocks_per_stage":[1,3,4,6,6,6,6],"n_conv_per_stage_decoder":[1,1,1,1,1,1],"conv_bias":true,"norm_op":"torch.nn.modules.instancenorm.InstanceNorm3d","norm_op_kwargs":{"eps":0.00001,"affine":true},"dropout_op":null,"dropout_op_kwargs":null,"nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"inplace":true}},"_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}},
            "3d_fullres_torchres_mambabot2_ps160x288x288_bs1": {"inherits_from":"3d_fullres_torchres_mambabot2_ps128x256x256_bs1","patch_size":[160,288,288],"architecture":{"network_class_name":"nnunetv2.nets.UMambaBot2","arch_kwargs":{"n_stages":7,"features_per_stage":[32,64,128,256,320,320,320],"conv_op":"torch.nn.modules.conv.Conv3d","kernel_sizes":[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],"strides":[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,1]],"n_blocks_per_stage":[1,3,4,6,6,6,6],"n_conv_per_stage_decoder":[1,1,1,1,1,1],"conv_bias":true,"norm_op":"torch.nn.modules.instancenorm.InstanceNorm3d","norm_op_kwargs":{"eps":0.00001,"affine":true},"dropout_op":null,"dropout_op_kwargs":null,"nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"inplace":true}},"_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}}
    ```
- For Task 2:
    ```json
            "3d_fullres_torchres_mambabot2click_crossattn2_ps128x256x256_bs1": {"inherits_from":"3d_fullres","data_identifier": "nnUNetPlans_3d_fullres_torchres_ctnorm","normalization_schemes": ["CTNormalization"],"batch_size":1,"patch_size":[128,256,256],"architecture":{"network_class_name":"nnunetv2.nets.UMambaBot2_CrossAttn_Click2","arch_kwargs":{"n_stages":7,"features_per_stage":[32,64,128,256,320,320,320],"conv_op":"torch.nn.modules.conv.Conv3d","kernel_sizes":[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],"strides":[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]],"n_blocks_per_stage":[1,3,4,6,6,6,6],"n_conv_per_stage_decoder":[1,1,1,1,1,1],"conv_bias":true,"norm_op":"torch.nn.modules.instancenorm.InstanceNorm3d","norm_op_kwargs":{"eps":0.00001,"affine":true},"dropout_op":null,"dropout_op_kwargs":null,"nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"inplace":true},"patch_size":[128,256,256]},"_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}},
            "3d_fullres_torchres_mambabot2click_crossattn2_ps160x288x288_bs1": {"inherits_from":"3d_fullres_torchres_mambabot2click_crossattn2_ps128x256x256_bs1","patch_size":[160,288,288],"architecture":{"network_class_name":"nnunetv2.nets.UMambaBot2_CrossAttn_Click2","arch_kwargs":{"n_stages":7,"features_per_stage":[32,64,128,256,320,320,320],"conv_op":"torch.nn.modules.conv.Conv3d","kernel_sizes":[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],"strides":[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,1]],"n_blocks_per_stage":[1,3,4,6,6,6,6],"n_conv_per_stage_decoder":[1,1,1,1,1,1],"conv_bias":true,"norm_op":"torch.nn.modules.instancenorm.InstanceNorm3d","norm_op_kwargs":{"eps":0.00001,"affine":true},"dropout_op":null,"dropout_op_kwargs":null,"nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"inplace":true},"patch_size":[160,288,288]},"_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}}
    ```

## Preprocessing
- Task 1:
`nnUNetv2_preprocess -d {task1_dataset_id} -c 3d_fullres_torchres_mambabot2_ps128x256x256_bs1 -plans_name nnUNetResEncUNetLPlans_torchres -np 48`
- Task 2:
`nnUNetv2_preprocess -d {task2_dataset_id} -c 3d_fullres_torchres_mambabot2click_crossattn2_ps128x256x256_bs1 -plans_name nnUNetResEncUNetLPlans_torchres -np 48`

## Training
Models are initialized using our pretrained weights via self-supervised learning. 
You can download them from [here](https://drive.google.com/drive/folders/1xhUkHCpo_50sNWvGH9CrN8Ws0hSjoa_k?usp=sharing) or run your own pretraining with the codes [here](../Pretrain_DAE). 
The codes are adapted from [DAE](https://github.com/Project-MONAI/research-contributions/tree/main/DAE/Pretrain_full_contrast).
Download the [ToothFairy3](https://ditto.ing.unimore.it/toothfairy3/) and [SD-Tooth](https://zenodo.org/records/10597292) datasets,
update the paths in `data/data_pretrain.py` and run the following command for pretraining:
```bash
python main_runner.py --batch_size=2 --sw_batch_size=1 --epoch=200 --mask_patch_size=16 --loss_type=all_img --base_lr=5e-5 --min_lr=5e-6 --warmpup_epoch=4 --warmup_lr=5e-7 --cache_dataset --cache_rate=0.2 --model_type=nnunet --save_freq=5 --log_dir="./output/umamba2/logdir" --output="./output/umamba2" --out_channels=1 --choice "all" --mm_con 0.03 --temperature 0.5 --img_size 288 --roi_x 160 --roi_y 288 --roi_z 288 --mask_ratio 0.3  --wandb --nnunet_plan <path to plan> --nnunet_conf <configuration name>
```

Then, we train the models on all training cases using gradient accumulation to simulate batch size of 2. For experimenting, create train and validation split manually.
- Task 1:
    ```bash
    nnUNetv2_train {task1_dataset_id} 3d_fullres_torchres_mambabot2_ps160x288x288_bs1 all -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_TF3_Task1_1500ep_accum2 -pretrained_weights <path to your pretarined weights>
    ```
- Task 2:
    ```bash
    nnUNetv2_train {task2_dataset_id} 3d_fullres_torchres_mambabot2click_crossattn2_ps160x288x288_bs1 all -p nnUNetResEncUNetLPlans_torchres -tr nnUNetTrainer_TF3_Task2_Click_1500ep_accum2 -pretrained_weights <path to your pretarined weights>
    ```

We recommend to increase the number of processes used for data augmentation. Otherwise, you can run into CPU bottlenecks.
Use `export nnUNet_n_proc_DA={number_of_cpus}`.

Optionally, you can further train the model with smaller learning rate using trainer `nnUNetTrainer_TF3_Task1_accum2_cont` or `nnUNetTrainer_TF3_Task2_Click_accum2_cont`.
Set the `-pretrained_weights` argument of `nnUNetv2_train` command to the final checkpoint and 
the modified `load_pretrained_weights.py` will load the weights correctly.

## Inference
Run inference with the inference script for [Task 1](task1_inference.py) and [Task 2](task2_inference_wclicks.py).

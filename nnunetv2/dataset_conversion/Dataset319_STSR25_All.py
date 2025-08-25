from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm



if __name__ == "__main__":
    root = "/home/zhiqin/Projects/nnUNet/data/nnUNet_raw/Dataset319_STSR2025"
    # print({v: k for k, v in mapping_DS119_nocanal().items()})
    os.makedirs(join(root, 'imagesTr'))
    os.makedirs(join(root, 'labelsTr'))
    for img_path in os.listdir(join(root, 'Images')):
        new_path = img_path.replace('.nii.gz', '_0000.nii.gz')
        os.symlink(join('..', 'Images', img_path), join(root, 'imagesTr', new_path))

    for img_path in os.listdir(join(root, 'Masks')):
        new_path = img_path.replace('_Masks.nii.gz', '.nii.gz')
        os.symlink(join('..', 'Masks', img_path), join(root, 'labelsTr', new_path))

    with open(join(root, 'dataset.json'), 'w') as f:
        json.dump(
            {
              "name": "STSR 2025",
              "description": "Segmentation of CBCT volumes",
              "reference": "",
              "license": "",
              "release": "",
              "tensorImageSize": "4D",
              "labels": {
                "background": 0,
                "Tooth": 1,
                "Pulp": 2,
                "Upper 1": 3,
                "Upper Molar Root 1": 4,
                "Lower Molar Root 1": 5,
                "Lower Molar Root 2": 6,
                "Molar Root 1": 7,
                "Lower Molar Root 3": 8,
                "Molar Root 2": 9,
                "Lower 1": 10,
                "Upper 2": 11,
                "Wisdom tooth": 12
              },
              "numTraining": 30,
              "numTest": 0,
              "file_ending": ".nii.gz",
              "channel_names": {
                "0": "CBCT"
              }
            }, f, indent=4
        )
    with open(join(root, 'splits_final.json'), 'w') as f:
        split = [{"train": sorted([x.replace('.nii.gz', '') for x in os.listdir(join(root, 'labelsTr')) if 'Train' in x]),
                  "unlabeled": sorted([x.replace('_0000.nii.gz', '') for x in os.listdir(join(root, 'unlabeledTr'))]),
                  "val": sorted([x.replace('.nii.gz', '') for x in os.listdir(join(root, 'labelsTr')) if 'Train' not in x])
                  }]
        json.dump(split, f)

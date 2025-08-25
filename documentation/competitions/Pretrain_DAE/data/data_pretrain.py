import json
import os
import pdb

import numpy as np
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Dataset
from torch.utils.data._utils.collate import default_collate

from monai.data import (
    CacheDataset,
    DataLoader,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
)
from monai.transforms import (
    clip,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    Flipd,
    Lambda, Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Transposed,
    Transform, Identityd
)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    # pdb.set_trace()
    return tr, val


class MaskGenerator:
    def __init__(self, input_size=(96, 96, 96), mask_patch_size=16, model_patch_size=(2, 2, 2), mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size[0]
        self.mask_ratio = mask_ratio
        assert all(x % self.mask_patch_size == 0 for x in self.input_size)
        assert self.mask_patch_size % self.model_patch_size == 0
        self.rand_size = [x // self.mask_patch_size for x in self.input_size]
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size[0] * self.rand_size[1] * self.rand_size[2]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape(self.rand_size)
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        return mask


class CustomDataset(Dataset):
    def __init__(self, args, datalists, transforms):
        super().__init__()
        assert len(datalists) == len(transforms)
        self.datalists = datalists
        self.transforms = transforms
        self.lens = np.cumsum([len(datalist) for datalist in datalists])
        self.datasets = []
        for datalist, transform in zip(self.datalists, self.transforms):
            self.datasets.append(CacheDataset(data=datalist, transform=transform,
                                              cache_rate=args.cache_rate, num_workers=args.num_workers))

        model_patch_size = args.patch_size
        self.mask_generator = MaskGenerator(
            input_size=(args.roi_x, args.roi_y, args.roi_z),
            mask_patch_size=args.mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=args.mask_ratio,
        )

    def __len__(self):
        return self.lens[-1]

    def __getitem__(self, idx):
        for i, l in enumerate(self.lens):
            if idx < l:
                break
        img = self.datasets[i][idx - (self.lens[i-1] if i > 0 else 0)]
        if 'label' in img:
            del img['label']
        mask = self.mask_generator()
        return img, mask

class MyTransform(Transform):
    def __init__(self, args):
        self.transform_tf3 = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Lambdad(keys=["image"], func=lambda img: clip(img, -997., 3558.)),
                NormalizeIntensityd(keys=["image"], subtrahend=809.5, divisor=1014.369),
            ]
        )
        self.transform_sdt_int = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=2),
                NormalizeIntensityd(keys=["image"], subtrahend=0.3434, divisor=0.2385377),
            ]
        )
        self.transform_sdt_roi = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=(1, 2)),
                Lambdad(keys=["image"], func=lambda img: clip(img, -997., 3558.)),
                NormalizeIntensityd(keys=["image"], subtrahend=809.5, divisor=1014.369),
            ]
        )
        self.transform_sdt_int_ul = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=2),
                Lambdad(keys=["image"], func=lambda img: clip(img, -997., 3558.)),
                NormalizeIntensityd(keys=["image"], subtrahend=809.5, divisor=1014.369),
            ]
        )
        self.transform_sdt_roi_ul = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=2),
                Lambdad(keys=["image"], func=lambda img: clip(img, -997., 3558.)),
                NormalizeIntensityd(keys=["image"], subtrahend=809.5, divisor=1014.369),
            ]
        )


    def __call__(self, img):
        if 'label' in img:
            del img['label']
        if img["class"] == "tf3":
            img = self.transform_tf3(img)
        elif img["class"] == "sdt_int":
            img = self.transform_sdt_int(img)
        elif img["class"] == "sdt_roi":
            img = self.transform_sdt_roi(img)
        elif img["class"] == "sdt_int_ul":
            img = self.transform_sdt_int_ul(img)
        elif img["class"] == "sdt_roi_ul":
            img = self.transform_sdt_roi_ul(img)
        else:
            raise ValueError
        # img = self.transform_img(img)
        return img


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(args):
    base_dir = '~/datasets' # update your base directory here

    tf3_splits = './dataset_tf3.json' # 0.3mm spacing

    sdt_int_splits = './dataset_sdt_integrity_labeled.json' # 0.25-0.3mm
    sdt_int_ul_splits = './dataset_sdt_integrity_unlabeled.json'

    sdt_roi_splits = './dataset_sdt_roi_labeled.json'  # 0.25-0.3mm
    sdt_roi_ul_splits = './dataset_sdt_roi_unlabeled.json'

    tf3_datalist = load_decathlon_datalist(tf3_splits, False, "training", base_dir=base_dir)
    print("Dataset ToothFairy3: number of data: {}".format(len(tf3_datalist)))
    sdt_int_datalist = load_decathlon_datalist(sdt_int_splits, False, "training", base_dir=base_dir)
    sdt_int_ul_datalist = load_decathlon_datalist(sdt_int_ul_splits, False, "training", base_dir=base_dir)
    print("Dataset SD-Tooth Integrity: number of data: {}".format(len(sdt_int_datalist) + len(sdt_int_ul_datalist)))
    sdt_roi_datalist = load_decathlon_datalist(sdt_roi_splits, False, "training", base_dir=base_dir)
    sdt_roi_ul_datalist = load_decathlon_datalist(sdt_roi_ul_splits, False, "training", base_dir=base_dir)
    print("Dataset SD-Tooth ROI: number of data: {}".format(len(sdt_roi_datalist) + len(sdt_roi_ul_datalist)))


    tf3_vallist = load_decathlon_datalist(tf3_splits, False, "validation", base_dir=base_dir)
    sdt_int_vallist = load_decathlon_datalist(sdt_int_splits, False, "validation", base_dir=base_dir)
    sdt_int_ul_vallist = load_decathlon_datalist(sdt_int_ul_splits, False, "validation", base_dir=base_dir)
    sdt_roi_vallist = load_decathlon_datalist(sdt_roi_splits, False, "validation", base_dir=base_dir)
    sdt_roi_ul_vallist = load_decathlon_datalist(sdt_roi_ul_splits, False, "validation", base_dir=base_dir)

    for dl, cls in zip([tf3_datalist, sdt_int_datalist, sdt_int_ul_datalist, sdt_roi_datalist, sdt_roi_ul_datalist, ],
                       ['tf3', 'sdt_int', 'sdt_int_ul', 'sdt_roi', 'sdt_roi_ul',]):
        for i in range(len(dl)):
            dl[i]['class'] = cls

    for dl, cls in zip([tf3_vallist, sdt_int_vallist, sdt_int_ul_vallist, sdt_roi_vallist, sdt_roi_ul_vallist,],
                       ['tf3', 'sdt_int', 'sdt_int_ul', 'sdt_roi', 'sdt_roi_ul', 'liu', 'mdd']):
        for i in range(len(dl)):
            dl[i]['class'] = cls

    datalist = tf3_datalist + sdt_int_datalist + sdt_int_ul_datalist + sdt_roi_datalist + sdt_roi_ul_datalist
    vallist = tf3_vallist + sdt_int_vallist + sdt_int_ul_vallist + sdt_roi_vallist + sdt_roi_ul_vallist

    if args.all_data:
        datalist = datalist + vallist

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(vallist)))

    my_transform = MyTransform(args)
    model_patch_size = args.patch_size
    mask_generator = MaskGenerator(
        input_size=(args.roi_x, args.roi_y, args.roi_z),
        mask_patch_size=args.mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=args.mask_ratio,
    )
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            my_transform,
            Transposed(keys=["image"], indices=(0, 3, 2, 1)) if args.model_type == "nnunet" else Identityd(keys=["image"]),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropd(roi_size=[args.roi_x, args.roi_y, args.roi_z], keys=["image"], random_size=False,
                             random_center=True),
            ToTensord(keys=["image"]),
            Lambda(func=lambda data: (data, mask_generator())),
         ]
    )

    # dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=1.0, num_workers=8, cache_num=4759)
    # dataset_val = CacheDataset(data=val_files, transform=transform, cache_rate=1.0, num_workers=8, cache_num=260)
    dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=args.cache_rate, num_workers=args.num_workers)
    dataset_val = CacheDataset(data=vallist, transform=transform, cache_rate=args.cache_rate, num_workers=args.num_workers)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        # sampler=sampler_train,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        # prefetch_factor=4,
    )

    dataloader_val = DataLoader(
        dataset_val, batch_size=1,
        # sampler=sampler_val,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        # prefetch_factor=4,
    )

    return dataloader_train, dataloader_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("DAE pre-training script", add_help=False)
    parser.add_argument("--patch_size", default=(2, 2, 2), type=tuple, help="window size")
    parser.add_argument("--mask_patch_size", default=16, type=int, help="window size")
    parser.add_argument("--img_size", default=96, type=int, help="image size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--mask_ratio", default=0.6, type=float, help="drop path rate")
    parser.add_argument("--cache_rate", default=0.2, type=float, help="drop path rate")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    # parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    # parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--choice", default="mae", type=str, help="choice")
    parser.add_argument("--inf", default="notsim", type=str, help="choice")

    parser.add_argument("--variance", default=0.1, type=float, help="")
    parser.add_argument("--interpolate", default=4, type=float, help="")
    parser.add_argument("--temperature", default=0.07, type=float, help="drop path rate")
    parser.add_argument("--mm_con", default=0.02, type=float, help="drop path rate")

    parser.add_argument("--wandb", action="store_true", help="use wandb")

    args = parser.parse_args()

    base_dir = '~/data/datasets'

    sdt_int_splits = '../dataset_sdt_integrity_labeled.json'  # 0.25-0.3mm
    sdt_int_ul_splits = '../dataset_sdt_integrity_unlabeled.json'

    sdt_roi_splits = '../dataset_sdt_roi_labeled.json'  # 0.25-0.3mm
    sdt_roi_ul_splits = '../dataset_sdt_roi_unlabeled.json'

    sdt_int_datalist = load_decathlon_datalist(sdt_int_splits, False, "training", base_dir=base_dir)
    sdt_int_ul_datalist = load_decathlon_datalist(sdt_int_ul_splits, False, "training", base_dir=base_dir)
    print("Dataset SD-Tooth Integrity: number of data: {}".format(len(sdt_int_datalist) + len(sdt_int_ul_datalist)))
    sdt_roi_datalist = load_decathlon_datalist(sdt_roi_splits, False, "training", base_dir=base_dir)
    sdt_roi_ul_datalist = load_decathlon_datalist(sdt_roi_ul_splits, False, "training", base_dir=base_dir)
    print("Dataset SD-Tooth ROI: number of data: {}".format(len(sdt_roi_datalist) + len(sdt_roi_ul_datalist)))

    for dl, cls in zip([sdt_int_datalist, sdt_int_ul_datalist, sdt_roi_datalist, sdt_roi_ul_datalist,],
                       ['sdt_int', 'sdt_int_ul', 'sdt_roi', 'sdt_roi_ul',]):
        for i in range(len(dl)):
            dl[i]['class'] = cls

    datalist = sdt_int_datalist + sdt_int_ul_datalist + sdt_roi_datalist + sdt_roi_ul_datalist

    print("Dataset all training: number of data: {}".format(len(datalist)))

    my_transform = MyTransform(args)
    model_patch_size = args.patch_size
    mask_generator = MaskGenerator(
        input_size=(args.roi_x, args.roi_y, args.roi_z),
        mask_patch_size=args.mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=args.mask_ratio,
    )
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            my_transform,
            # SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # RandSpatialCropd(roi_size=[args.roi_x, args.roi_y, args.roi_z], keys=["image"], random_size=False,
            #                  random_center=True),
            ToTensord(keys=["image"]),
            Lambda(func=lambda data: (data, mask_generator())),
         ]
    )
    # dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=1.0, num_workers=8, cache_num=4759)
    # dataset_val = CacheDataset(data=val_files, transform=transform, cache_rate=1.0, num_workers=8, cache_num=260)
    dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=0.0,
                                 num_workers=0)

    dataloader_train = DataLoader(
        dataset_train,
        1,
        num_workers=0,
        # pin_memory=True,
        drop_last=False,
        shuffle=True,
        collate_fn=collate_fn,
    )
    for idx, (img, mask) in enumerate(dataloader_train):
        print(img['class'], img['image'].mean(dim=(1, 2, 3, 4)), img['image'].std(dim=(1, 2, 3, 4)))
        print(img['image'].shape, mask.shape)

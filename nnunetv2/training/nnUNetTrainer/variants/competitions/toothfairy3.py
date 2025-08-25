from typing import Union, Tuple, List, Dict

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoaderWithClick
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_CE_smooth_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainer_onlyMirror01
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context


class LRMirrorTransform(MirrorTransform):
    def __init__(self, allowed_axes: Tuple[int, ...], mapping: List[Tuple[int, int]], n_class: int):
        super().__init__(allowed_axes)
        assert 2 in allowed_axes, 'only support 3d now'
        self.lr_mapping = torch.arange(n_class, dtype=torch.int16) # nnunet default dtype
        for l, r in mapping:
            self.lr_mapping[l] = r
            self.lr_mapping[r] = l
        # print(self.lr_mapping)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return segmentation
        axes = [i + 1 for i in params['axes']]
        segmentation = torch.flip(segmentation, axes)
        if 2 in params['axes']:
            # print('flipping lr label')
            segmentation = self.lr_mapping[segmentation.to(dtype=torch.int)]
        return segmentation

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        # probably dont need to do this
        raise NotImplementedError


class nnUNetTrainer_LRMirror(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D during inference
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            self.inference_allowed_mirroring_axes = (0, )
        else:
            self.inference_allowed_mirroring_axes = (0, 1)
        # self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        Use LRMirrorTransform instead of MirrorTransform
        """
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            mapping = [(3, 4), (5, 6), (43, 44)] # ian, sinus, incisive canal
            for l, r in zip(range(19, 27), range(11, 19)): # upper
                mapping.append((l, r))
            for l, r in zip(range(27, 35), range(35, 43)): # lower
                mapping.append((l, r))
            print('defining lr mapping as', mapping)
            transforms.append(
                LRMirrorTransform(
                    allowed_axes=mirror_axes,
                    mapping=mapping,
                    n_class=46 + 1
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainer_TF3_Task1(nnUNetTrainer_LRMirror):
    # smoothloss, weighted, no smooth dice, LRMirror
    def _build_loss(self):
        class_weight = [1] + [1] * 42 + \
                       [10., 10., 10.] + [1]  # 3 canals + pulp
        assert len(class_weight) == 47
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                          'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp},
                                         {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label,
                                         dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainer_TF3_Task1_accum2(nnUNetTrainer_TF3_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50 # save some time

class nnUNetTrainer_TF3_Task1_accum2_cont(nnUNetTrainer_TF3_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 0.001
        self.num_epochs = 500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50 # save some time

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1., 0.01, total_iters=self.num_epochs - 1)
        return optimizer, lr_scheduler

class nnUNetTrainer_TF3_Task1_1500ep(nnUNetTrainer_TF3_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500

class nnUNetTrainer_TF3_Task1_1500ep_accum2(nnUNetTrainer_TF3_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50

class nnUNetTrainer_TF3_Task2_LRMirror(nnUNetTrainer_LRMirror):#, nnUNetTrainerDiceCELoss_noSmooth):
    # LRMirror
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        Use LRMirrorTransform instead of MirrorTransform
        """
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            mapping = [(1, 2)]  # ian
            print('defining lr mapping as', mapping)
            transforms.append(
                LRMirrorTransform(
                    allowed_axes=mirror_axes,
                    mapping=mapping,
                    n_class=2 + 1
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainer_TF3_Task2_Click(nnUNetTrainer_TF3_Task2_LRMirror):#, nnUNetTrainerDiceCELoss_noSmooth):
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # # set validation step to max len
        # self.num_val_iterations_per_epoch = int(np.ceil(len(dataset_val.identifiers) / max(self.batch_size // 2, 1)))
        # self.print_to_log_file(f"reconfiguring num_val_iterations_per_epoch to {self.num_val_iterations_per_epoch}")

        dl_tr = nnUNetDataLoaderWithClick(dataset_tr, self.batch_size,
                                         initial_patch_size,
                                         self.configuration_manager.patch_size,
                                         self.label_manager,
                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                         sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                         probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderWithClick(dataset_val, max(self.batch_size // 2, 1),
                                          self.configuration_manager.patch_size,
                                          self.configuration_manager.patch_size,
                                          self.label_manager,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                          probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=False,  # self.device.type == 'cuda',
                                                        wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=False,  # self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict, step_optimizer=True) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if step_optimizer:
            self.optimizer.zero_grad(set_to_none=True)

        points = (batch['point_coords'].to(self.device, non_blocking=True), batch['point_labels'].to(self.device, non_blocking=True))
        # print(points)
        #     print('resetting gradients')
        # else:
        #     print('accumulating gradients')
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, points)
            # del data
            l = self.loss(output, target)
            if self.accumulate_steps > 1:
                l = l / self.accumulate_steps
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            if step_optimizer:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grap_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
        else:
            l.backward()
            if step_optimizer:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grap_norm)
                self.optimizer.step()
        l = l.detach().cpu().numpy()
        if self.accumulate_steps > 1: # reverse grad accum
            l = l * self.accumulate_steps
        return {'loss': l}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        points = (batch['point_coords'].to(self.device, non_blocking=True), batch['point_labels'].to(self.device, non_blocking=True))
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, points)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            # output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.bfloat16) # save mem
            predicted_segmentation_onehot.scatter_(1, output.argmax(1)[:, None], 1)
            # del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().float().numpy()
        fp_hard = fp.detach().cpu().float().numpy()
        fn_hard = fn.detach().cpu().float().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

class nnUNetTrainer_TF3_Task2_Click_1500ep(nnUNetTrainer_TF3_Task2_Click):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500

class nnUNetTrainer_TF3_Task2_Click_1500ep_accum2(nnUNetTrainer_TF3_Task2_Click):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10 # 50

class nnUNetTrainer_TF3_Task2_Click_accum2_cont(nnUNetTrainer_TF3_Task2_Click):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 0.001
        self.num_epochs = 500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50 # save some time

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1., 0.01, total_iters=self.num_epochs - 1)
        return optimizer, lr_scheduler

# old stuff below
class nnUNetTrainer_onlyMirror01_weighted(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        class_weight = [1] + [1] * 42 + \
                       [10., 10., 10.] + [1]  # canals
        assert len(class_weight) == 47
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainer_onlyMirror01_smoothLoss(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainer_onlyMirror01_smoothLoss_weighted(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        class_weight = [1] + [1] * 42 + \
                       [10., 10., 10.] + [1]  # canals
        assert len(class_weight) == 47
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                         {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label,
                                         dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainer_onlyMirror01_weighted_nolr(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        class_weight = [1] + [1] * 27 + \
                       [10., 10.]  # canals
        assert len(class_weight) == 30
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainer_onlyMirror01_smoothLoss_nolr(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            def smooth_tensor(cls_id, smooth_ids, smooth=0.1):
                if not isinstance(smooth_ids, list):
                    smooth_ids = [smooth_ids]
                temp = torch.zeros(30, device='cuda')
                temp[cls_id] = 1 - smooth
                temp[smooth_ids] = smooth / len(smooth_ids)
                return temp
            smooth_mapping = {3: smooth_tensor(3, 28),
                              9: smooth_tensor(9, [10, 25, 27]), 10: smooth_tensor(10, [9, 11, 25, 27]),
                              17: smooth_tensor(17, [18, 26, 27]), 18: smooth_tensor(18, [17, 19, 26, 27]),
                              25: smooth_tensor(25, [9, 10, 27]),
                              26: smooth_tensor(26, [17, 18, 27]),
                              28: smooth_tensor(28, 3),
                              }
            for i in list(range(11, 16)) + list(range(19, 24)):
                smooth_mapping[i] = smooth_tensor(i, [i-1, i+1, 27])
            for i in [16, 24]:
                smooth_mapping[i] = smooth_tensor(i, [i-1, 27])

            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss,
                                         smooth_mapping=smooth_mapping)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainer_onlyMirror01_smoothLoss_weighted_nolr(nnUNetTrainer_onlyMirror01):
    def _build_loss(self):
        class_weight = [1] + [1] * 27 + \
                       [10., 10.]  # canals
        assert len(class_weight) == 30
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            def smooth_tensor(cls_id, smooth_ids, smooth=0.1):
                if not isinstance(smooth_ids, list):
                    smooth_ids = [smooth_ids]
                temp = torch.zeros(30, device='cuda')
                temp[cls_id] = 1 - smooth
                temp[smooth_ids] = smooth / len(smooth_ids)
                return temp
            smooth_mapping = {3: smooth_tensor(3, 28),
                              9: smooth_tensor(9, [10, 25, 27]), 10: smooth_tensor(10, [9, 11, 25, 27]),
                              17: smooth_tensor(17, [18, 26, 27]), 18: smooth_tensor(18, [17, 19, 26, 27]),
                              25: smooth_tensor(25, [9, 10, 27]),
                              26: smooth_tensor(26, [17, 18, 27]),
                              28: smooth_tensor(28, 3),
                              }
            for i in list(range(11, 16)) + list(range(19, 24)):
                smooth_mapping[i] = smooth_tensor(i, [i-1, i+1, 27])
            for i in [16, 24]:
                smooth_mapping[i] = smooth_tensor(i, [i-1, 27])

            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                         {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label,
                                         dice_class=MemoryEfficientSoftDiceLoss, smooth_mapping=smooth_mapping)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


if __name__ == "__main__":
    from time import time
    mapping = [(3, 4), (5, 6), (43, 44)]
    for l, r in zip(range(19, 27), range(11, 19)):
        mapping.append((l, r))
    for l, r in zip(range(27, 35), range(35, 43)):
        mapping.append((l, r))
    print('defining lr mapping as', mapping)
    lrm_t = LRMirrorTransform(
                allowed_axes=(0, 1, 2),
                mapping=mapping,
                n_class=46 + 1
            )

    times_torch = []
    for _ in range(100):
        data_dict = {'segmentation': torch.randint(47, (1, 1, 96, 96, 96), dtype=torch.int16)}
        st = time()
        out = lrm_t(**data_dict)
        assert out['segmentation'].dtype is torch.int16
        times_torch.append(time() - st)
    print('lrm_t', np.mean(times_torch))

    m_t = MirrorTransform((0, 1, 2))
    times_bg = []
    for _ in range(100):
        data_dict = {'segmentation': torch.randint(47, (1, 1, 96, 96, 96), dtype=torch.int16)}
        st = time()
        out = m_t(**data_dict)
        times_bg.append(time() - st)
    print('m_t', np.mean(times_bg))

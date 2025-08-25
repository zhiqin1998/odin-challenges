import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Union, List

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.blank_rectangle import BlankRectangleTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform, OneOfTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, nnUNetDataLoaderSemiSupervised
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_smooth_loss, DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.semisupervised_loss import SoftmaxMSELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainer_onlyMirror01_500ep, nnUNetTrainerNoMirroring_500ep
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_500epochs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from torch.distributions.uniform import Uniform

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer_onlyMirror01_noDCSmooth_500ep(nnUNetTrainer_onlyMirror01_500ep, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainerNoMirroring_noDCSmooth_500ep(nnUNetTrainerNoMirroring_500ep, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainer_noDCSmooth_500ep(nnUNetTrainer_500epochs, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainer_noDCSmooth_DA5_500ep(nnUNetTrainer_500epochs, nnUNetTrainerDA5, nnUNetTrainerDiceCELoss_noSmooth):
    pass

class nnUNetTrainer_noDCSmooth_500ep_smooth_weighted(nnUNetTrainer_500epochs):
    def _build_loss(self):
        class_weight = [1., 1., 1., 10., 1., 1.,
                        1., 1., 1., 1., 10., 10., 1.]
        assert len(class_weight) == 13
        class_weight = torch.tensor(class_weight, device=self.device)
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            def smooth_tensor(cls_id, smooth_ids, smooth=0.1):
                if not isinstance(smooth_ids, list):
                    smooth_ids = [smooth_ids]
                temp = torch.zeros(13, device='cuda')
                temp[smooth_ids] = smooth / len(smooth_ids)
                temp[cls_id] = 1 - smooth
                return temp
            smooth_mapping = {1: smooth_tensor(1, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 0.15),
                              2: smooth_tensor(2, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                              3: smooth_tensor(3, [1, 2, 4, 7, 9, 11]),
                              4: smooth_tensor(4, [1, 2, 3, 7, 9, 11]),
                              5: smooth_tensor(5, [1, 2, 6, 7, 8, 9, 10]),
                              6: smooth_tensor(6, [1, 2, 5, 7, 8, 9, 10]),
                              7: smooth_tensor(7, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]),
                              8: smooth_tensor(8, [1, 2, 5, 6, 7, 9, 10]),
                              9: smooth_tensor(9, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]),
                              10: smooth_tensor(10, [1, 2, 5, 6, 7, 8, 9]),
                              11: smooth_tensor(11, [1, 2, 3, 4, 7, 9]),
                              12: smooth_tensor(12, 1)
                              }

            loss = DC_and_CE_smooth_loss({'batch_dice': self.configuration_manager.batch_dice,
                                          'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp},
                                         {"weight": class_weight}, weight_ce=1, weight_dice=1,
                                         ignore_label=self.label_manager.ignore_label,
                                         dice_class=MemoryEfficientSoftDiceLoss,
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

# @torch.compile
def feature_perturbation(x_ori: torch.Tensor, random_dropout_p: float,
                         feature_channel_keep: Tuple[float, float], noise_vector: torch.Tensor):
    # random channel dropout
    x = F.dropout3d(x_ori, p=random_dropout_p)
    # feature drop
    attention = torch.mean(x_ori, dim=1, keepdim=True)
    threshold = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)[0] * float(
        np.random.uniform(*feature_channel_keep))
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    del attention, threshold, x_ori
    # add noise
    noise_vector = noise_vector.to(x.device).unsqueeze(0)
    x = x.mul(noise_vector) + x
    return x

class nnUNetTrainer_semisupervised(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.semisupervised_loss = None
        self.dataloader_unlabeled = None  # dataloader is infinite
        # semisupervised settings
        self.unlabeled_min_ratio = 0.1
        self.unlabeled_max_ratio = 0.5
        self.perturb_layers = [-1]
        self.feature_channel_keep = (0.6, 0.9) # was (0.7, 0.9)
        self.feature_noise_dist = Uniform(-0.3, 0.3)
        self.random_dropout_p = 0.5
        self.semisupervised_w = 50. # mse magnitude is small, was 30
        self.rampup_iter = int(0.2 * self.num_epochs * self.num_iterations_per_epoch)
        # pseudo labels
        self.use_pseudolabel = False
        self.pseudolabel_loss = None
        self.pseudolabel_loss_w = 0.1
        self.pseudolabel_p = 0.25
        self.pseudolabel_thres = 0.75

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            self.print_to_log_file(f'in_chan: {self.num_input_channels}\tout chan: {self.label_manager.num_segmentation_heads}')
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # TODO dont know if it is safe to compile here
            if self._do_i_compile():# and 'UMamba' not in self.configuration_manager.network_arch_class_name:
                self.print_to_log_file('Using torch.compile...')
                self.network.encoder.compile()
                self.network.mamba_layer.compile()
                self.network.decoder.compile()
            #     self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                raise NotImplementedError

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def get_unlabeled_transforms(
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
        # only apply spatial transform here
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

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            raise NotImplementedError

        if regions is not None:
            raise NotImplementedError

        return ComposeTransforms(transforms)

    @staticmethod
    def get_strong_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            do_dummy_2d_data_aug: bool,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(OneOfTransform([
            RandomTransform(
                MedianFilterTransform(
                    (2, 8),
                    p_per_channel=1
                ), apply_probability=0.3
            ),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.3, 1.5),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=1, benchmark=True
                ), apply_probability=0.3
            )
        ]))

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.5, 1.5)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.3
        ))

        transforms.append(OneOfTransform(
            [
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.5, 1.5)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1
                    ), apply_probability=0.3
                ),
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.5, 1.5)),
                        preserve_range=False,
                        synchronize_channels=False,
                        p_per_channel=1
                    ), apply_probability=0.3
                ),
            ]
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.25, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=1
            ), apply_probability=0.4
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

        transforms.append(
            RandomTransform(
                BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                        rectangle_value=0.,
                                        num_rectangles=(1, 5),
                                        force_square=False,
                                        p_per_channel=1
                ), apply_probability=0.4
            )
        )

        transforms.append(
            RandomTransform(
                SharpeningTransform(
                    strength=(0.1, 1),
                    p_per_channel=1
                ) , apply_probability=0.3
            )
        )

        return ComposeTransforms(transforms)

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

        # unlabeled pipeline
        unlabeled_transforms = self.get_unlabeled_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        strong_transform = self.get_strong_transforms(patch_size, do_dummy_2d_data_aug)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        if not isfile(splits_file):
            raise NotImplementedError
        else:
            splits = load_json(splits_file)
        if self.fold == 'all':
            unlabeled_keys = splits[0]['unlabeled']
        elif self.fold < len(splits):
            unlabeled_keys = splits[self.fold]['unlabeled']
        else:
            raise NotImplementedError
        dataset_unlabeled = self.dataset_class(self.preprocessed_dataset_folder, unlabeled_keys,
                                               folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

        # # set validation step to max len
        # self.num_val_iterations_per_epoch = int(np.ceil(len(dataset_val.identifiers) / max(self.batch_size // 2, 1)))
        # self.print_to_log_file(f"reconfiguring num_val_iterations_per_epoch to {self.num_val_iterations_per_epoch}")

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, max(self.batch_size // 2, 1),
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        dl_unlabeled = nnUNetDataLoaderSemiSupervised(dataset_unlabeled, self.batch_size,
                                                    initial_patch_size,
                                                    self.configuration_manager.patch_size,
                                                    self.label_manager,
                                                    oversample_foreground_percent=0.,
                                                    sampling_probabilities=None, pad_sides=None, transforms=[unlabeled_transforms, strong_transform],
                                                    probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            mt_gen_unlabeled = SingleThreadedAugmenter(dl_unlabeled, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes // 2, # half
                                                        num_cached=max(6, allowed_num_processes // 4), seeds=None,
                                                        pin_memory=False,#self.device.type == 'cuda',
                                                        wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=False,#self.device.type == 'cuda',
                                                      wait_time=0.002)
            mt_gen_unlabeled = NonDetMultiThreadedAugmenter(data_loader=dl_unlabeled, transform=None,
                                                            num_processes=allowed_num_processes // 2,  # half
                                                            num_cached=max(6, allowed_num_processes // 4), seeds=None,
                                                            pin_memory=False,  # self.device.type == 'cuda',
                                                            wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        _ = next(mt_gen_unlabeled)
        self.dataloader_unlabeled = mt_gen_unlabeled
        return mt_gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        semisupervised_loss = SoftmaxMSELoss()
        if self._do_i_compile():
            semisupervised_loss = torch.compile(semisupervised_loss)
        # loss without bg class
        pseudolabel_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                          'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                          ignore_label=0, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            pseudolabel_loss.dc = torch.compile(pseudolabel_loss.dc)

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
            semisupervised_loss = DeepSupervisionWrapper(semisupervised_loss, weights)
            pseudolabel_loss = DeepSupervisionWrapper(pseudolabel_loss, weights)
        self.semisupervised_loss = semisupervised_loss
        self.pseudolabel_loss = pseudolabel_loss

    def calculate_semisupervised_weight(self, curr_iter):
        curr_total_iter = curr_iter + (self.current_epoch * self.num_iterations_per_epoch)
        current = np.clip(curr_total_iter, 0.0, self.rampup_iter)
        phase = 1.0 - current / self.rampup_iter
        current_rampup = float(np.exp(-5.0 * phase * phase))
        return self.semisupervised_w * current_rampup

    # TODO: assuming network is umamba, will work for unet if comment out mamba code
    def semisupervised_forward(self, x):
        skips = self.network.encoder(x)
        # last n/2 layer feature perturbation, for a 6 skips model (depth 7), this is last 3
        for i in self.perturb_layers:
            skips[i] = feature_perturbation(skips[i], self.random_dropout_p, self.feature_channel_keep,
                                            self.feature_noise_dist.sample(skips[i].shape[1:])
                                            )
        # resume model
        skips[-1] = self.network.mamba_layer(skips[-1])
        return self.network.decoder(skips)

    def semisupervised_step(self, batch: dict, curr_iter: int, step_optimizer: bool = True) -> dict:
        ori_data = batch['data']
        data = batch['aug_data']

        ori_data = ori_data.to(self.device, non_blocking=True)
        data = data.to(self.device, non_blocking=True)

        if step_optimizer:
            self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            if self.use_pseudolabel and np.random.uniform() < self.pseudolabel_p:
                output = self.network(ori_data)
                target = []
                with torch.no_grad():
                    t = F.softmax(output[0].clone().detach(), 1)  # b, c, h, w, d
                    p_label = t.argmax(1, keepdim=True)  # b, 1, h, w, d
                    p_label[torch.gather(t, 1, p_label) < self.pseudolabel_thres] = 0  # set to bg (ignored)
                    target.append(p_label)
                    for x in output[1:]:
                        spatial_size = x.shape[-3:]
                        target.append(F.interpolate(p_label.float(), spatial_size, mode='nearest').long())
                l = self.pseudolabel_loss(output, target) * self.pseudolabel_loss_w
            else:
                with torch.no_grad():
                    target = self.network(ori_data)
                del ori_data
                output = self.semisupervised_forward(data)
                del data
                w = self.calculate_semisupervised_weight(curr_iter)
                l =  w * self.semisupervised_loss(output, target)
            # print(w, l)

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

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            semisupervised_losses = []
            for batch_id in range(self.num_iterations_per_epoch):
                step_optimizer = ((batch_id + 1) % self.accumulate_steps == 0) if self.accumulate_steps > 1 else True
                if not step_optimizer and batch_id + 1 == self.num_iterations_per_epoch:  # last index always step
                    step_optimizer = True
                if np.random.uniform() < min(self.unlabeled_max_ratio, epoch / self.num_epochs + self.unlabeled_min_ratio): # linearly increase unlabeled samples
                    semisupervised_losses.append(self.semisupervised_step(next(self.dataloader_unlabeled), batch_id,
                                                                          step_optimizer=step_optimizer)['loss'])
                else:
                    train_outputs.append(self.train_step(next(self.dataloader_train), step_optimizer=step_optimizer))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            self.print_to_log_file(f'Semisupervised weight: {self.calculate_semisupervised_weight(0)}')
            self.print_to_log_file(f'Semisupervised loss: {np.mean(semisupervised_losses):.5f} (n={len(semisupervised_losses)})')

        self.on_train_end()

class nnUNetTrainer_noDCSmooth_500ep_semisupervised(nnUNetTrainer_semisupervised):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.rampup_iter = int(0.2 * self.num_epochs * self.num_iterations_per_epoch)

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainer_semisupervised_saug(nnUNetTrainer_semisupervised):
    @staticmethod
    def get_strong_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            do_dummy_2d_data_aug: bool,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(OneOfTransform([
            RandomTransform(
                MedianFilterTransform(
                    (2, 8),
                    p_per_channel=1
                ), apply_probability=0.5
            ),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.3, 1.5),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=1, benchmark=True
                ), apply_probability=0.5
            )
        ]))

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.5
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.5, 1.5)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.5
        ))

        transforms.append(OneOfTransform(
            [
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.5, 1.5)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1
                    ), apply_probability=0.5
                ),
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.5, 1.5)),
                        preserve_range=False,
                        synchronize_channels=False,
                        p_per_channel=1
                    ), apply_probability=0.5
                ),
            ]
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.25, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=1
            ), apply_probability=0.7
        ))

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.5
        ))

        transforms.append(
            RandomTransform(
                BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                        rectangle_value=0.,
                                        num_rectangles=(1, 5),
                                        force_square=False,
                                        p_per_channel=1
                                        ), apply_probability=0.7
            )
        )

        transforms.append(
            RandomTransform(
                SharpeningTransform(
                    strength=(0.1, 1),
                    p_per_channel=1
                ), apply_probability=0.5
            )
        )

        return ComposeTransforms(transforms)

class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug(nnUNetTrainer_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.rampup_iter = int(0.2 * self.num_epochs * self.num_iterations_per_epoch)

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug_n3(nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-3, -2, -1] # last 3 layer features

class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug_n5(nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features

class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug_n5_w100(nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features
        self.semisupervised_w = 100

class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug_n5_w500(nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features
        self.semisupervised_w = 500

class nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug_n5_w1000(nnUNetTrainer_noDCSmooth_500ep_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features
        self.semisupervised_w = 1000

# no dc smooth, saug, feature n5
class nnUNetTrainer_STSR_Task1(nnUNetTrainer_semisupervised_saug):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainer_STSR_Task1_accum2(nnUNetTrainer_STSR_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50
        self.rampup_iter = int(0.2 * self.num_epochs * self.num_iterations_per_epoch)

class nnUNetTrainer_STSR_Task1_500ep_accum2(nnUNetTrainer_STSR_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1] # last 5 layer features
        self.num_epochs = 500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.rampup_iter = int(0.2 * self.num_epochs * self.num_iterations_per_epoch)

class nnUNetTrainer_STSR_Task1_accum2_cont(nnUNetTrainer_STSR_Task1):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.perturb_layers = [-5, -4, -3, -2, -1]  # last 5 layer features
        self.initial_lr = 0.001
        self.num_epochs = 500
        self.accumulate_steps = 2
        self.num_iterations_per_epoch = self.num_iterations_per_epoch * 2
        self.num_val_iterations_per_epoch = 10  # 50
        self.unlabeled_min_ratio = 0.3
        self.rampup_iter = int(0.05 * self.num_epochs * self.num_iterations_per_epoch)
        self.use_pseudolabel = True # enable for final round training

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1., 0.01, total_iters=self.num_epochs - 1)
        return optimizer, lr_scheduler

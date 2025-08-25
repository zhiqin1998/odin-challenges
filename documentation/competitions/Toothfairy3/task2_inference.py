# import os
# n_threads = 4  # set to 4 to emulate grand challenge runtime env
# os.environ["OMP_NUM_THREADS"] = str(n_threads)
# os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
# os.environ["MKL_NUM_THREADS"] = str(n_threads)
# os.environ["BLIS_NUM_THREADS"] = str(n_threads)

import argparse
import gc
import itertools
import json
import time
import os
import traceback
from pathlib import Path
from typing import Union, Tuple
from copy import deepcopy

import nnunetv2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
# torch.cuda.set_per_process_memory_fraction(16/24) #16gb


class CustomPredictor(nnUNetPredictor):
    def __init__(self, *args, tta_batch_size=1, lr_mapping=None, n_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta_batch_size = tta_batch_size
        if lr_mapping is not None:
            self.lr_mapping = list(range(n_class))
            for l, r in lr_mapping:
                self.lr_mapping[l] = r
                self.lr_mapping[r] = l
        else:
            self.lr_mapping = lr_mapping

    def predict_single_cbct_npy_array(self, input_image: np.ndarray, image_properties: dict,):
        # torch.set_num_threads(7)
        image_properties = deepcopy(image_properties)
        with torch.no_grad():
            if self.verbose:
                print('preprocessing')

            data, _, image_properties = self.preprocessor.run_case_npy(input_image, None, image_properties,
                                                                       self.plans_manager,
                                                                       self.configuration_manager,
                                                                       self.dataset_json)

            data = torch.from_numpy(data)
            del input_image

            if self.verbose:
                print('predicting')
            predicted_logits = self.predict_preprocessed_image(data)
            if self.verbose:
                print('Prediction done')

            return self.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, image_properties)

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        assert len(use_folds) == 1
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=self.device, weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        assert len(self.list_of_parameters) == 1
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        if self.lr_mapping is not None and 2 not in self.allowed_mirroring_axes:
            print('adding axis2 to allowed_mirroring_axes')
            self.allowed_mirroring_axes = (*self.allowed_mirroring_axes, 2)
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        self.network = network
        self.network = self.network.to(device=self.device)
        self.network.load_state_dict(
            checkpoint['network_weights']
            # torch.load(self.list_of_parameters[0], map_location=self.device, weights_only=False)['network_weights']
        )
        self.network.eval()
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
            print('Using torch.compile')
            # self.network = torch.compile(self.network)
            self.network.compile()

        self.preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        data_device = self.device
        predicted_logits_device = gaussian_device = self.device
        interim_dtype = torch.float16
        compute_device = self.device

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image
        gc.collect()
        empty_cache(self.device)

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        data = data.to(device=data_device, non_blocking=True)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=interim_dtype,
                                       device=predicted_logits_device)
        # n_predictions = torch.zeros(data.shape[1:], dtype=interim_dtype, device=predicted_logits_device)

        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=interim_dtype)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        with torch.autocast(self.device.type, enabled=True):
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                pred = self._internal_maybe_mirror_and_predict(data[sl][None].to(device=compute_device, non_blocking=True))[0].to(
                    device=predicted_logits_device, dtype=interim_dtype)
                pred /= (pred.max() / 100)
                # pred /= 10 # doing this save ~0.02s per image lol
                predicted_logits[sl] += (pred * gaussian)
                # n_predictions[sl[1:]] += gaussian

            # torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # if torch.any(torch.isinf(predicted_logits)):
            #     raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
            #                        'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
            #                        'predicted_logits to fp32')
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        return predicted_logits

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        # prediction = self.network(x)

        if mirror_axes is None:
            return self.network(x)

        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

        mirror_axes = [m + 2 for m in mirror_axes]
        axes_combinations = [
            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        ]

        if self.tta_batch_size > 1:
            # assert (len(axes_combinations) + 1) % self.tta_batch_size == 0, '(len(axes_combinations) + 1) must be divisible by tta_batch_size'
            x_combinations = [torch.flip(x, axes) for axes in axes_combinations]
            x_combinations.insert(0, x)
            assert len(x_combinations) % self.tta_batch_size == 0, f'{len(x_combinations)} not divisible by {self.tta_batch_size}'
            prediction = None
            for i in range(0, len(x_combinations), self.tta_batch_size):
                batch_prediction = self.network(torch.cat(x_combinations[i:i + self.tta_batch_size], dim=0))
                for j in range(batch_prediction.shape[0]):
                    original_idx = i + j
                    if original_idx == 0:
                        prediction = batch_prediction[j:j + 1]
                    else:
                        axes_to_flip_back = axes_combinations[original_idx - 1]
                        prediction += (torch.flip(batch_prediction[j:j + 1], axes_to_flip_back)[:, self.lr_mapping] if self.lr_mapping is not None and 4 in axes_to_flip_back else torch.flip(batch_prediction[j:j + 1], axes_to_flip_back))
                        # temp = torch.flip(batch_prediction[j:j + 1], axes_to_flip_back)
                        # if self.lr_mapping is not None and 2 + 2 in axes_to_flip_back:
                        #     temp = temp[:, self.lr_mapping]
                        # prediction += temp
        else:
            prediction = self.network(x)
            for axes in axes_combinations:
                prediction += (torch.flip(self.network(torch.flip(x, axes)), axes)[:, self.lr_mapping] if self.lr_mapping is not None and 4 in axes else torch.flip(self.network(torch.flip(x, axes)), axes))
                # temp = torch.flip(self.network(torch.flip(x, axes)), axes)
                # if self.lr_mapping is not None and 2 + 2 in axes:
                #     temp = temp[:, self.lr_mapping]
                # prediction += temp

        prediction /= (len(axes_combinations) + 1)
        return prediction

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
        # old = torch.get_num_threads()
        # torch.set_num_threads(7)

        # resample to original shape
        spacing_transposed = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration_manager.spacing if \
            len(self.configuration_manager.spacing) == \
            len(props['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration_manager.spacing]
        predicted_logits = self.configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                                                  props[
                                                                                      'shape_after_cropping_and_before_resampling'],
                                                                                  current_spacing,
                                                                                  [props['spacing'][i] for i in
                                                                                   self.plans_manager.transpose_forward],
                                                                                   device=self.device)



        segmentation = None
        pp = None
        try:
            with torch.no_grad():
                pp = predicted_logits.to(device=self.device)
                segmentation = pp.argmax(0).cpu()
                del pp
        except RuntimeError:
            print('oom error, shape:', predicted_logits.shape)
            del segmentation, pp
            gc.collect()
            torch.cuda.empty_cache()
            segmentation = predicted_logits.argmax(0)
        del predicted_logits

        # segmentation may be torch.Tensor but we continue with numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(props['shape_before_cropping'],
                                                  dtype=np.uint8 if len(self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        # torch.set_num_threads(old)
        return segmentation_reverted_cropping


def predict_semseg(im, prop, base_predictor):
    semseg_pred = base_predictor.predict_single_cbct_npy_array(
        im, prop
    )
    gc.collect()
    torch.cuda.empty_cache()
    return semseg_pred


if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw

    # os.environ['nnUNet_compile'] = 'f'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_json', type=Path, default=os.path.join(nnUNet_preprocessed, 'Dataset119_ToothFairy3', 'splits_final.json'))
    parser.add_argument('-i', '--input_folder', type=Path, default=os.path.join(nnUNet_raw, 'Dataset119_ToothFairy3', 'imagesTr'))
    parser.add_argument('-o', '--output_folder', type=Path, default=os.path.join(nnUNet_results, 'Dataset119_ToothFairy3', 'eval_results'))
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--folds', type=str, nargs='+', default=[0,])
    args = parser.parse_args()

    (args.output_folder / 'predictions').mkdir(exist_ok=True, parents=True)

    folds = [i if i == 'all' else int(i) for i in args.folds]
    base_model = args.base_model

    mapping = [(1, 2)]  # ian, sinus, incisive canal
    print('defining lr mapping as', mapping)

    # initialize predictors
    base_predictor = CustomPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        allow_tqdm=True,
        tta_batch_size=1,
        lr_mapping=mapping,
        n_class=3,
        verbose=False
    )
    base_predictor.initialize_from_trained_model_folder(
        base_model,
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth'
    )

    rw = SimpleITKIO()

    with open(args.split_json, 'r') as f:
        input_files = [args.input_folder / (x + '_0000.nii.gz') for x in json.load(f)[0]['val']]
    # input_files = list(args.input_folder.glob('*.nii.gz')) + list(args.input_folder.glob('*.mha'))

    case_stats = {}

    try:
        for input_fname in input_files:
            output_fname = args.output_folder / 'predictions' / input_fname.name.replace('_0000.nii.gz', '.nii.gz')

            # we start with the instance seg because we can then start converting that while semseg is being predicted
            # load test image
            im, prop = rw.read_images([input_fname])
            print(input_fname, im.shape)

            with torch.no_grad():
                st = time.time()
                torch.cuda.reset_peak_memory_stats(device=None)
                semseg_pred = predict_semseg(im, prop, base_predictor)

            # now postprocess
            # semseg_pred = postprocess(semseg_pred, np.prod(prop['spacing']), True)

            time_elapsed = time.time() - st
            memory_used = torch.cuda.max_memory_allocated(device=None) / 1024 / 1024
            if memory_used / 1024 >= 16:
                print('memory usage exceeded', memory_used, 'shape', im.shape, im.size)
            case_stats[input_fname.name.replace('_0000.nii.gz', '.nii.gz')] = {'time_elapsed': float(time_elapsed), 'memory_used': float(memory_used)}

            # now save
            rw.write_seg(semseg_pred, output_fname, prop)

            del im, prop, semseg_pred
            gc.collect()
            torch.cuda.empty_cache()
    except RuntimeError:
        print(traceback.format_exc())
        print('stopping now at', input_fname)

    with open(args.output_folder / 'gpu_stats.json', 'w') as f:
        json.dump(case_stats, f)

    print(f'first case time: {case_stats[input_files[0].name.replace("_0000.nii.gz", ".nii.gz")]["time_elapsed"]:.5f}')
    print(f'average time: {np.mean([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'average gpu memory: {np.mean([x["memory_used"] for x in case_stats.values()]):.5f}')
    print(f'max time: {max([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'max gpu memory: {max([x["memory_used"] for x in case_stats.values()]):.5f}')

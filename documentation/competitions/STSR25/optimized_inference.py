import os

n_threads = 12 # set to 12 to emulate evaluation runtime env
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["BLIS_NUM_THREADS"] = str(n_threads)

import gc
import itertools
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Union, Tuple, List

import argparse

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image

import nnunetv2
import torch
import numpy as np
from queue import Queue
from threading import Thread
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from tqdm import tqdm

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule


class CustomPredictor(nnUNetPredictor):
    def __init__(self, *args, compile=True, tta_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile = compile
        self.tta_batch_size = tta_batch_size

    def predict_single_cbct_npy_array(self, input_image: np.ndarray, image_properties: dict):
        # torch.set_num_threads(7)
        image_properties = deepcopy(image_properties)
        with torch.no_grad():
            if self.verbose:
                print('preprocessing')

            data, _, image_properties = self.preprocessor.run_case_npy(input_image, None, image_properties,
                                                                       self.plans_manager,
                                                                       self.configuration_manager,
                                                                       self.dataset_json)
            print('predicting', data.shape)
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
                                             checkpoint_name: str = 'checkpoint_final.pth',):
        """
        This is used when making predictions with a trained model
        """
        # assert roi_model_path is not None
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        # dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        # plans = load_json(join(model_training_output_dir, 'plans.json'))
        # plans_manager = PlansManager(plans)

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
                dataset_json = checkpoint['init_args']['dataset_json']
                plans_manager = PlansManager(checkpoint['init_args']['plans'])

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        # trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        #                                             trainer_name, 'nnunetv2.training.nnUNetTrainer')
        # if trainer_class is None:
        #     raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
        #                        f'Please place it there (in any .py file)!')
        network = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = (1, 2)#inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        assert len(self.list_of_parameters) == 1
        self.network = self.network.to(self.device)
        self.network.load_state_dict(
            checkpoint['network_weights']
            # torch.load(self.list_of_parameters[0], map_location=self.device, weights_only=False)['network_weights']
        )
        self.network.eval()
        if self.compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            # self.network = torch.compile(self.network)
            self.network.compile()

        self.preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = 4,
                           num_processes_segmentation_export: int = 4,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        # ########################
        # # let's store the input arguments so that its clear what was used to generate the prediction
        # if output_folder is not None:
        #     my_init_kwargs = {}
        #     for k in inspect.signature(self.predict_from_files).parameters.keys():
        #         my_init_kwargs[k] = locals()[k]
        #     my_init_kwargs = deepcopy(
        #         my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
        #     recursive_fix_for_json_export(my_init_kwargs)
        #     maybe_mkdir_p(output_folder)
        #     save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
        #
        #     # we need these two if we want to do things with the predictions like for example apply postprocessing
        #     save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        #     save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        # #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    # replace with single model variant
    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        # n_threads = torch.get_num_threads()
        # torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        # prediction = None
        prediction = self.predict_sliding_window_return_logits(data).to('cpu')

        # for params in self.list_of_parameters:
        #
        #     # messing with state dict names...
        #     if not isinstance(self.network, OptimizedModule):
        #         self.network.load_state_dict(params)
        #     else:
        #         self.network._orig_mod.load_state_dict(params)
        #
        #     # why not leave prediction on device if perform_everything_on_device? Because this may cause the
        #     # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
        #     # this actually saves computation time
        #     if prediction is None:
        #         prediction = self.predict_sliding_window_return_logits(data).to('cpu')
        #     else:
        #         prediction += self.predict_sliding_window_return_logits(data).to('cpu')
        #
        # if len(self.list_of_parameters) > 1:
        #     prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        # torch.set_num_threads(n_threads)
        return prediction

    # bypass precise division
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            # n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    prediction /= (prediction.max() / 100)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    # n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            # torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    # replace with batched tta variant
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
                        prediction += torch.flip(batch_prediction[j:j + 1], axes_to_flip_back)

        else:
            prediction = self.network(x)
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)

        prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        data_device = self.device
        predicted_logits_device = gaussian_device = self.device if image.numel() < 8e7 else torch.device('cpu')
        interim_dtype = torch.float16
        compute_device = self.device

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image
        # gc.collect()
        # empty_cache(self.device)

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        data = data.to(device=data_device, non_blocking=True)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=interim_dtype,
                                       device=predicted_logits_device)
        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=interim_dtype)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        with torch.autocast(self.device.type, enabled=True):
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                pred = \
                self._internal_maybe_mirror_and_predict(data[sl][None].to(device=compute_device, non_blocking=True))[
                    0].to(
                    device=predicted_logits_device, dtype=interim_dtype)
                pred /= (pred.max() / 100)
                predicted_logits[sl] += (pred * gaussian)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        return predicted_logits

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
                                                  dtype=np.uint8 if len(
                                                      self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        # torch.set_num_threads(old)
        return segmentation_reverted_cropping


if __name__ == '__main__':
    os.environ['nnUNet_compile'] = 'f'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=Path, required=True)
    parser.add_argument('-o', '--output_folder', type=Path, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--folds', type=str, nargs='+', default=[0,])
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    folds = [i if i == 'all' else int(i) for i in args.folds]
    base_model = args.base_model

    # initialize predictors
    base_predictor = CustomPredictor(
        tile_step_size=0.8,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        allow_tqdm=True,
        tta_batch_size=1,
        compile=False
    )
    base_predictor.initialize_from_trained_model_folder(
        base_model,
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth',
    )

    rw = SimpleITKIO()

    st = time.time()
    input_files = [[join(args.input_folder, f)] for f in os.listdir(args.input_folder) if f.endswith('.nii.gz')]
    output_files = [join(args.output_folder, f.replace('.nii.gz', '_Masks')) for f in os.listdir(args.input_folder) if f.endswith('.nii.gz')]
    # base_predictor.predict_from_files(input_files, output_files,
    #                                  num_processes_preprocessing=1, num_processes_segmentation_export=1, # >1 ram can explode
    #                                  )
    case_stats = {}

    for input_fname, output_fname in zip(input_files, output_files):

        # we start with the instance seg because we can then start converting that while semseg is being predicted
        # load test image
        st = time.time()
        im, prop = rw.read_images(input_fname)
        print(input_fname[0], im.shape)

        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats(device=None)
            semseg_pred = base_predictor.predict_single_cbct_npy_array(im, prop)

        # now save
        rw.write_seg(semseg_pred, output_fname + '.nii.gz', prop)

        time_elapsed = time.time() - st
        memory_used = torch.cuda.max_memory_allocated(device=None) / 1024 / 1024
        if memory_used / 1024 >= 24:
            print('memory usage exceeded', memory_used, 'shape', im.shape, im.size)
        if time_elapsed > 30: # theortically 4090 is 2x 3090
            print('time exceeded', time_elapsed, 'shape', im.shape, im.size)
        case_stats[Path(input_fname[0]).name.replace('_0000.nii.gz', '.nii.gz')] = {'time_elapsed': float(time_elapsed),
                                                                           'memory_used': float(memory_used),
                                                                           'shape': semseg_pred.shape,
                                                                           }

        del im, prop, semseg_pred
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.output_folder / 'gpu_stats.json', 'w') as f:
        json.dump(case_stats, f)

    print(f'first case time: {case_stats[Path(input_files[0][0]).name.replace("_0000.nii.gz", ".nii.gz")]["time_elapsed"]:.5f}')
    print(f'average time: {np.mean([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'average gpu memory: {np.mean([x["memory_used"] for x in case_stats.values()]):.5f}')
    print(f'max time: {max([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'max gpu memory: {max([x["memory_used"] for x in case_stats.values()]):.5f}')

    # print(f'total time: {time.time() - st_:.5f}s')
    # print(f'per image time: {(time.time() - st_) / len(input_files):.5f}s')

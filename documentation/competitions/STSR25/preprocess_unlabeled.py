import multiprocessing
import shutil
from time import sleep
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name, convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, get_filenames_of_train_images_and_targets


def preprocess_dataset(dataset_id: int,
                       plans_identifier: str = 'nnUNetPlans',
                       configurations = ('2d', '3d_fullres', '3d_lowres'),
                       num_processes = (8, 4, 8),
                       verbose: bool = False) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f'Preprocessing labeled and unlabeled dataset {dataset_name}')
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...')
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.")
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        preprocessor_run(preprocessor, dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, 'gt_segmentations'))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(nnUNet_preprocessed, dataset_name, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)

def preprocessor_run(preprocessor, dataset_name_or_id, c, plans_identifier, num_processes):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

    assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                               "first." % plans_file
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(c)

    dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
    dataset_json = load_json(dataset_json_file)

    output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

    if isdir(output_directory):
        shutil.rmtree(output_directory)

    maybe_mkdir_p(output_directory)

    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

    raw_dataset_folder = join(nnUNet_raw, dataset_name)
    identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'unlabeledTr'), dataset_json['file_ending'])
    images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'unlabeledTr'), dataset_json['file_ending'], identifiers)
    unlabeled_dataset = {i: {'images': im, 'label': None} for i, im in zip(identifiers, images)}
    dataset = {**dataset, **unlabeled_dataset}
    print(dataset)

    # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
    # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

    # multiprocessing magic.
    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        remaining = list(range(len(dataset)))
        # p is pretty nifti. If we kill workers they just respawn but don't do any work.
        # So we need to store the original pool of workers.
        workers = [j for j in p._pool]
        for k in dataset.keys():
            r.append(p.starmap_async(preprocessor.run_case_save,
                                     ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                       plans_manager, configuration_manager,
                                       dataset_json),)))

        with tqdm(desc=None, total=len(dataset)) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                       'OK jokes aside.\n'
                                       'One of your background processes is missing. This could be because of '
                                       'an error (look for an error message) or because it was killed '
                                       'by your OS due to running out of RAM. If you don\'t see '
                                       'an error message, out of RAM is likely the problem. In that case '
                                       'reducing the number of workers might help')
                done = [i for i in remaining if r[i].ready()]
                # get done so that errors can be raised
                _ = [r[i].get() for i in done]
                for _ in done:
                    r[_].get()  # allows triggering errors
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)


if __name__ == '__main__':
    dataset_id = 319
    plan_name = 'nnUNetResEncUNetLPlans_torchres'
    configuration_names = ['3d_fullres_torchres_mambabot2_ps160x256x256_bs1']
    num_processes = [8]

    preprocess_dataset(dataset_id, plan_name, configuration_names, num_processes)

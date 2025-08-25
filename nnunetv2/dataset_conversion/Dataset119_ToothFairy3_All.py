from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def mapping_DS119() -> Dict[int, int]:
    """Remove all NA Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: i - 65 for i in range(111, 119)})  # [111-118]->[46-53]
    mapping.update({i: i - 67 for i in range(121, 129)})  # [121-128]->[54-61]
    mapping.update({i: i - 69 for i in range(131, 139)})  # [131-138]->[62-69]
    mapping.update({i: i - 71 for i in range(141, 149)})  # [141-148]->[70-77]
    return mapping


def mapping_DS119_nocanal(remove_gap=False) -> Dict[int, int]:
    """Remove all NA and canal Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 1 for i in range(103, 105)}) #[103-104]->1 incisive to lower jawbone
    mapping.update({105: 1}) # lingual to lower jawbone too?
    gap = 3 if remove_gap else 0
    mapping.update({i: i - 65 - gap for i in range(111, 119)})  # [111-118]->[46-53]
    mapping.update({i: i - 67 - gap for i in range(121, 129)})  # [121-128]->[54-61]
    mapping.update({i: i - 69 - gap for i in range(131, 139)})  # [131-138]->[62-69]
    mapping.update({i: i - 71 - gap for i in range(141, 149)})  # [141-148]->[70-77]
    return mapping


def mapping_DS119_nocanal_singlepulp(remove_gap=False) -> Dict[int, int]:
    """Remove all NA and canal Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 1 for i in range(103, 105)}) #[103-104]->1 incisive to lower jawbone
    mapping.update({105: 1}) # lingual to lower jawbone too?
    gap = 3 if remove_gap else 0
    mapping.update({i: 46 - gap for i in range(111, 119)})  # [111-118]->46
    mapping.update({i: 46 - gap for i in range(121, 129)})  # [121-128]->46
    mapping.update({i: 46 - gap for i in range(131, 139)})  # [131-138]->46
    mapping.update({i: 46 - gap for i in range(141, 149)})  # [141-148]->46
    return mapping

def mapping_DS119_singlepulp() -> Dict[int, int]:
    """Remove all NA classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 46 for i in range(111, 119)})  # [111-118]->46
    mapping.update({i: 46 for i in range(121, 129)})  # [121-128]->46
    mapping.update({i: 46 for i in range(131, 139)})  # [131-138]->46
    mapping.update({i: 46 for i in range(141, 149)})  # [141-148]->46
    return mapping

def mapping_DS119_eval() -> Dict[int, int]:
    """Remove all NA classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i for i in range(41, 49)})  # [41-48]->[35-42]
    mapping.update({i: i - 52 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 50 for i in range(111, 119)})  # [111-118]->46
    mapping.update({i: 50 for i in range(121, 129)})  # [121-128]->46
    mapping.update({i: 50 for i in range(131, 139)})  # [131-138]->46
    mapping.update({i: 50 for i in range(141, 149)})  # [141-148]->46
    return mapping

def mapping_DS119_nopulp() -> Dict[int, int]:
    """Remove all NA and pulp Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: i - 100 for i in range(111, 119)})  # [111-118]->[11-18]
    mapping.update({i: i - 102 for i in range(121, 129)})  # [121-128]->[19-26]
    mapping.update({i: i - 104 for i in range(131, 139)})  # [131-138]->[27-34]
    mapping.update({i: i - 106 for i in range(141, 149)})  # [141-148]->[35-42]
    return mapping

def mapping_DS119_nocanal_nopulp() -> Dict[int, int]:
    """Remove all NA and pulp Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 1 for i in range(103, 105)}) #[103-104]->1 incisive to lower jawbone
    mapping.update({105: 1}) # lingual to lower jawbone too?
    mapping.update({i: i - 100 for i in range(111, 119)})  # [111-118]->[11-18]
    mapping.update({i: i - 102 for i in range(121, 129)})  # [121-128]->[19-26]
    mapping.update({i: i - 104 for i in range(131, 139)})  # [131-138]->[27-34]
    mapping.update({i: i - 106 for i in range(141, 149)})  # [141-148]->[35-42]
    return mapping


def mapping_DS119_nocanal_nolr() -> Dict[int, int]:
    """Remove all NA and canal Classes remove lr distinction except central incisor and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 3)})  # [1-2]->[1-2]
    mapping.update({i: 3 for i in range(3, 5)})  # [3-4]->3
    mapping.update({i: 4 for i in range(5, 7)})  # [5-6]->4
    mapping.update({i: i - 2 for i in range(7, 11)})  # [7-10]->[5-8]
    mapping.update({i: i - 2 for i in range(11, 19)})  # [11-18]->[9-16]
    mapping.update({i: i - 12 for i in range(21, 29)})  # [21-28]->[9-16]
    mapping.update({i: i - 14 for i in range(31, 39)})  # [31-38]->[17-24]
    mapping.update({i: i - 24 for i in range(41, 49)})  # [41-48]->[17-24]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 1 for i in range(103, 105)}) #[103-104]->1 incisive to lower jawbone
    mapping.update({105: 1}) # lingual to lower jawbone too?
    mapping.update({i: 27 for i in range(111, 119)})  # [111-118]->27
    mapping.update({i: 27 for i in range(121, 129)})  # [121-128]->27
    mapping.update({i: 27 for i in range(131, 139)})  # [131-138]->27
    mapping.update({i: 27 for i in range(141, 149)})  # [141-148]->27
    mapping.update({21: 25, 31: 26})  # map left central to new class
    return mapping

def mapping_DS119_nolr() -> Dict[int, int]:
    """Remove all NA and remove lr distinction except central incisor and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 3)})  # [1-2]->[1-2]
    mapping.update({i: 3 for i in range(3, 5)})  # [3-4]->3
    mapping.update({i: 4 for i in range(5, 7)})  # [5-6]->4
    mapping.update({i: i - 2 for i in range(7, 11)})  # [7-10]->[5-8]
    mapping.update({i: i - 2 for i in range(11, 19)})  # [11-18]->[9-16]
    mapping.update({i: i - 12 for i in range(21, 29)})  # [21-28]->[9-16]
    mapping.update({i: i - 14 for i in range(31, 39)})  # [31-38]->[17-24]
    mapping.update({i: i - 24 for i in range(41, 49)})  # [41-48]->[17-24]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 28 for i in range(103, 105)}) #[103-104]->28
    mapping.update({105: 29}) # lingual to 29 too?
    mapping.update({i: 27 for i in range(111, 119)})  # [111-118]->27
    mapping.update({i: 27 for i in range(121, 129)})  # [121-128]->27
    mapping.update({i: 27 for i in range(131, 139)})  # [131-138]->27
    mapping.update({i: 27 for i in range(141, 149)})  # [141-148]->27
    mapping.update({21: 25, 31: 26})  # map left central to new class
    return mapping


def mapping_DS119_nocanal_nopulp_nolr() -> Dict[int, int]:
    """Remove all NA and pulp Classes remove lr distinction except central incisor and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 3)})  # [1-2]->[1-2]
    mapping.update({i: 3 for i in range(3, 5)})  # [3-4]->3
    mapping.update({i: 4 for i in range(5, 7)})  # [5-6]->4
    mapping.update({i: i - 2 for i in range(7, 11)})  # [7-10]->[5-8]
    mapping.update({i: i - 2 for i in range(11, 19)})  # [11-18]->[9-16]
    mapping.update({i: i - 12 for i in range(21, 29)})  # [21-28]->[9-16]
    mapping.update({i: i - 14 for i in range(31, 39)})  # [31-38]->[17-24]
    mapping.update({i: i - 24 for i in range(41, 49)})  # [41-48]->[17-24]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 1 for i in range(103, 105)}) #[103-104]->1 incisive to lower jawbone
    mapping.update({105: 1}) # lingual to lower jawbone too?
    mapping.update({i: i - 102 for i in range(111, 119)})  # [111-118]->[9-16]
    mapping.update({i: i - 112 for i in range(121, 129)})  # [121-128]->[9-16]
    mapping.update({i: i - 114 for i in range(131, 139)})  # [131-138]->[17-24]
    mapping.update({i: i - 124 for i in range(141, 149)})  # [141-148]->[17-24]
    mapping.update({21: 25, 31: 26, 121: 25, 131: 26})  # map left central to new class
    return mapping

def mapping_DS119_toothNpulponly() -> Dict[int, int]:
    """Remove all NA Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: 1 for i in range(11, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: 1 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: 1 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: 1 for i in range(41, 49)})  # [41-48]->[35-42]
    # mapping.update({i: i - 60 for i in range(103, 106)})  # [103-105]->[43-45]
    mapping.update({i: 2 for i in range(111, 119)})  # [111-118]->[46-53]
    mapping.update({i: 2 for i in range(121, 129)})  # [121-128]->[54-61]
    mapping.update({i: 2 for i in range(131, 139)})  # [131-138]->[62-69]
    mapping.update({i: 2 for i in range(141, 149)})  # [141-148]->[70-77]
    return mapping

def mapping_DS119_jawNcanalonly() -> Dict[int, int]:
    """Remove all NA Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: 1 for i in range(1, 3)})  # [1-2]->[1]
    # mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    # mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    # mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    mapping.update({i: i - 101 for i in range(103, 106)})  # [103-105]->[2-4]
    # mapping.update({i: i - 65 for i in range(111, 119)})  # [111-118]->[46-53]
    # mapping.update({i: i - 67 for i in range(121, 129)})  # [121-128]->[54-61]
    # mapping.update({i: i - 69 for i in range(131, 139)})  # [131-138]->[62-69]
    # mapping.update({i: i - 71 for i in range(141, 149)})  # [141-148]->[70-77]
    return mapping

def mapping_DS219() -> Dict[int, int]:
    """Dataset for inferior alveolar only"""
    mapping = {3: 1, 4: 2} # left and right IAC
    return mapping

def mapping_DS219_lowerJaw() -> Dict[int, int]:
    """Dataset for inferior alveolar only"""
    mapping = mapping_DS219()
    mapping[1] = 3 # lower jawbone
    return mapping

def mapping_DS219_lowerJawNIncisiveCanal() -> Dict[int, int]:
    """Dataset for inferior alveolar only"""
    mapping = mapping_DS219_lowerJaw()
    mapping.update({103: 4, 104: 5}) # incisive canals
    return mapping

def load_json(json_file: str) -> Any:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json(json_file: str, data: Any, indent: int = 4) -> None:
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)


def image_to_nifi(input_path: str, output_path: str) -> None:
    image_sitk = sitk.ReadImage(input_path)
    sitk.WriteImage(image_sitk, output_path)


def label_mapping(input_path: str, output_path: str, mapping: Dict[int, int] = None) -> None:

    label_sitk = sitk.ReadImage(input_path)
    if mapping is not None:
        label_np = sitk.GetArrayFromImage(label_sitk)

        label_np_new = np.zeros_like(label_np, dtype=np.uint8)
        for org_id, new_id in mapping.items():
            label_np_new[label_np == org_id] = new_id

        if 'ToothFairy3S_0048.nii.gz' in input_path:
            print('processing special case', input_path, mapping.get(38, 0), mapping.get(138, 0))
            assert 60 in label_np and 160 in label_np
            label_np_new[label_np == 60] = mapping.get(38, 0)
            label_np_new[label_np == 160] = mapping.get(138, 0)

        label_sitk_new = sitk.GetImageFromArray(label_np_new)
        label_sitk_new.CopyInformation(label_sitk)
        sitk.WriteImage(label_sitk_new, output_path)
    else:
        sitk.WriteImage(label_sitk, output_path)


def process_labels(
    files: str, lbl_dir_in: str, lbl_dir_out: str, mapping: Dict[int, int], n_processes: int = 12
) -> None:

    os.makedirs(lbl_dir_out, exist_ok=True)

    iterable = [
        {
            "input_path": join(lbl_dir_in, file),
            "output_path": join(lbl_dir_out, file),
            "mapping": mapping,
        }
        for file in files
    ]
    with multiprocessing.Pool(processes=n_processes) as pool:
        jobs = [pool.apply_async(label_mapping, kwds={**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc="Process Labels...")]


def process_ds(
    root: str, input_ds: str, output_ds: str, mapping: dict, save_json='mapped_dataset.json'
) -> None:
    os.makedirs(join(root, output_ds), exist_ok=True)
    # --- Handle Labels --- #
    lbl_files = os.listdir(join(root, input_ds))
    lbl_dir_in = join(root, input_ds)
    lbl_dir_out = join(root, output_ds)

    process_labels(lbl_files, lbl_dir_in, lbl_dir_out, mapping, n_processes=16)

    # --- Generate dataset.json --- #
    dataset = {}
    dataset_json = load_json(join(root, "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = "ToothFairy 3 Mapped"
    dataset_json["numTraining"] = len(lbl_files)
    if dataset != {}:
        dataset_json["dataset"] = dataset

    label_dict = dataset_json["labels"]
    label_dict_new = {"background": 0}
    for k, v in label_dict.items():
        if v in mapping.keys():
            label_dict_new[k] = mapping[v]
        elif k != 'background':
            print(k, 'not in mapping')
            label_dict_new[k] = 0
    dataset_json["labels"] = label_dict_new
    write_json(join(root, save_json), dataset_json)

    # # --- Generate splits_final.json --- #
    # img_names = [file.replace("_0000.mha", "") for file in img_files]
    #
    # random_seed = 42
    # random.seed(random_seed)
    # random.shuffle(img_names)
    #
    # split_index = int(len(img_names) * 0.7)  # 70:30 split
    # train_files = img_names[:split_index]
    # val_files = img_names[split_index:]
    # train_files.sort()
    # val_files.sort()
    #
    # split = [{"train": train_files, "val": val_files}]
    # write_json(join(root, output_ds, "splits_final.json"), split)


if __name__ == "__main__":
    root = "/home/zhiqin/datasets/CBCT/ToothFairy3"
    print(mapping_DS119())
    # print({v: k for k, v in mapping_DS119_nocanal().items()})
    process_ds(root, "labelsTrRaw", "labelsTr",
               mapping_DS119(), "mapped_dataset.json")

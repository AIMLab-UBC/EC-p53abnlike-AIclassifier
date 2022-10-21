import os
import sys
import enum
import json
import pickle

import yaml
from pathlib import Path
from utils.dataset import BagDataset, FeatDataset

def strip_extension(path: str) -> str:
    """
    Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

def create_patch_id(path,
                    patch_pattern: dict = None,
                    rootpath: str = None) -> str:
    """
    Function to create patch ID either by
    1) patch_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch

    patch_pattern : dict
        Dictionary describing the directory structure of the patch path. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    rootpath : str
        The root directory path containing patch to clip from patch path. Assumes patch contains rootpath.

    Returns
    -------
    patch_id : string
        Remove useless information before patch id for h5 file storage
    """
    if patch_pattern is not None:
        len_of_patch_id = -(len(patch_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either patch_pattern or rootpath should be set.")

def get_label_by_patch_id(patch_id: str,
                          patch_pattern: dict,
                          CategoryEnum: enum.Enum,
                          is_binary: bool = False) -> enum.Enum:
    """
    Get category label from patch id. The label can be either 'annotation' or 'subtype' based on is_binary flag.

    Parameters
    ----------
    patch_id : string
        Patch ID get label from

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths used to find the label word in the patch ID. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    CategoryEnum : enum.Enum
        Acts as the lookup table for category label

    is_binary : bool
        For binary classification, i.e., we will use BinaryEnum instead of SubtypeEnum

    Returns
    -------
    enum.Enum
        label from CategoryEnum
    """
    label = patch_id.split('/')[patch_pattern['annotation' if is_binary else 'subtype']]
    return CategoryEnum[label if is_binary else label.upper()]

def get_slide_by_patch_id(patch_id: str,
                          patch_pattern: dict) -> str:
    """
    Function to obtain slide id from patch id

    Parameters
    ----------
    patch_id : str

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths used to find the slide word in the patch ID. The words can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    Returns
    -------
    slide_id : str
        Slide id extracted from `patch_id`
    """
    slide_id = patch_id.split('/')[patch_pattern['slide']]
    return slide_id

def load_chunks(chunk_file_location, chunk_ids, patch_pattern):
    """
    Load patch paths from specified chunks in chunk file

    Parameters
    ----------
    chunks : list of int
        The IDs of chunks to retrieve patch paths from

    Returns
    -------
    list of list
        [Patch paths, slide_ID] from the chunks
    """
    patch_paths = []
    with open(chunk_file_location) as f:
        data = json.load(f)
        chunks = data['chunks']
        for chunk in data['chunks']:
            if chunk['id'] in chunk_ids:
                patch_paths.extend([[path, get_slide_by_patch_id(create_patch_id(path,
                                                    patch_pattern), patch_pattern)] for path in chunk['imgs']])
    if len(patch_paths) == 0:
        raise ValueError(
                f"chunks {tuple(chunk_ids)} not found in {chunk_file_location}")
    return patch_paths

def extract_label_from_patch(CategoryEnum, patch_pattern, patch_path):
    """
    Get the label value according to CategoryEnum from the patch path

    Parameters
    ----------
    patch_path : str

    Returns
    -------
    int
        The label id for the patch
    """
    patch_path = patch_path[0]
    patch_id = create_patch_id(patch_path, patch_pattern)
    label = get_label_by_patch_id(patch_id, patch_pattern,
            CategoryEnum, is_binary=False)
    return label.value

def extract_labels(CategoryEnum, patch_pattern, patch_paths):
    return [extract_label_from_patch(CategoryEnum, patch_pattern, path) for path in patch_paths]

def find_slide_idx(patch_paths):
    """
    Find which patches corresponds to specific slide id

    Parameters
    ----------
    patch_paths : list of str

    Returns
    -------
    dict : dict {slide_id: [list of idx ...]}
    """
    dict = {}
    for idx, path_slide in enumerate(patch_paths):
        # path = path_slide[0]
        slide_id = path_slide[1]
        if slide_id not in dict: dict[slide_id] = []
        dict[slide_id].append(idx)
    return dict

def create_data_set(cfg, chunk_id, state=None, slide_id=None, training_set=False):
    """
    Create dataset

    Parameters
    ----------
    cfg : dict
        config file

    chunk_id : int

    state: str

    training_set: bool
        whether activate augmentation (in traning mode) or not

    Returns
    -------
    patch_dataset : Dataset
    """
    patch_pattern = {k: i for i, k in enumerate(cfg["patch_pattern"].split('/'))}
    if state == 'train' or state == 'validation' or state == 'test':
        chunk_file = cfg["chunk_file_location"]
    elif state == 'external':
        chunk_file = cfg["external_chunk_file_location"]
    patch_paths = load_chunks(chunk_file, chunk_id, patch_pattern)
    CategoryEnum = enum.Enum('SubtypeEnum', cfg["subtypes"])
    slide_idx = find_slide_idx(patch_paths)
    labels = extract_labels(CategoryEnum, patch_pattern, patch_paths)
    if cfg['representation_calculation']:
        patch_dataset = FeatDataset(patch_paths, resize=cfg["resize"], method=cfg["method"], slide_id=slide_id, state=state)
    else:
        patch_dataset = BagDataset(patch_paths, labels, cfg['representation_dir'], state,
                                   cfg['CategoryEnum'], training_set=training_set,
                                   external_test_name=cfg['external_test_name'])
    return patch_dataset, labels

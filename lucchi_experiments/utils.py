#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Utils for training and testing.

Author: Ken C. L. Wong
"""

import os
import numpy as np
import SimpleITK as sitk
from configparser import ConfigParser, ExtendedInterpolation
import ast
from collections import OrderedDict
from io import StringIO

import torch
from torchinfo import summary

__author__ = 'Ken C. L. Wong'


def normalize_modalities(data, mask_val=None, clip_val=None):
    """Normalizes a multichannel input with each channel as a modality.
    Each modality is normalized separately.
    Note that the channel-first format is assumed.

    Args:
        data: A multichannel input with each channel as a modality (channel-first).
        mask_val: If not None, the intensities of `mask_val` are not used to compute mean and std (default: None).
        clip_val: A tuple of (min, max) values to be clipped (default: None).

    Returns:
        Normalized data.
    """
    data = [normalize_data(da, mask_val=mask_val, clip_val=clip_val) for da in data]
    data = np.stack(data)
    return data


def normalize_data(data, mask_val=None, clip_val=None):
    """Normalizes data of a single modality.

    Args:
        data: The data of a single modality.
        mask_val: If not None, the intensities of `mask_val` are not used to compute mean and std (default: None).
        clip_val: A tuple of (min, max) values to be clipped (default: None).

    Returns:
        Normalized data.
    """
    data = np.asarray(data, dtype=np.float32)

    if clip_val is not None:
        data = np.clip(data, *clip_val)

    if mask_val is not None:
        data = np.ma.array(data, mask=(data == mask_val))

    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    if mask_val is not None:
        data = data.filled(0)  # With mean 0 and std 1 after normalization, filling with 0 is a good choice.

    data = np.asarray(data, dtype=np.float32)  # As numpy may change the dtype

    return data


def to_categorical(y, num_classes=None):
    """Converts an int label tensor to one-hot.

    Args:
        y: Input label tensor with shape (B, 1, D, H, W).
        num_classes: Number of classes.

    Returns:
        A one-hot tensor of y with shape (B, num_classes, D, H, W).
    """
    assert y.shape[1] == 1, 'Can only handle single label per pixel.'
    y = y[:, 0]
    y = y.to(dtype=int)
    input_shape = y.shape
    y = y.ravel()
    if not num_classes:
        num_classes = y.max() + 1
    n = y.shape[0]
    categorical = torch.zeros((n, num_classes), dtype=torch.float32, device=y.device)
    categorical[torch.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = torch.reshape(categorical, output_shape)
    categorical = torch.moveaxis(categorical, -1, 1)  # Channel-first
    return categorical


def remap_labels(label, mapping):
    """Remaps labels.

    Args:
        label: The labels need to be remapped.
        mapping: A dict of mapping. Keys: old labels; values: new labels.

    Returns:
        A copy of remapped labels.
    """
    if isinstance(label, np.ndarray):
        label_cp = label.copy()
    elif isinstance(label, torch.Tensor):
        label_cp = label.detach().clone()
    else:
        raise ValueError('Input "label" must be a Numpy array or PyTorch tensor.')

    for k, v in mapping.items():
        label_cp[label == k] = v
    return label_cp


def save_model_summary(model, input_size, path=None):
    """Saves model summary to a text file.

    Args:
        model: A PyTorch model.
        input_size: Input size.
        path: A full output file path (default: None).
    """
    net_summary = summary(model, input_size=input_size, device='meta', verbose=0)
    if path is not None:
        with open(path, 'w') as f:
            print(net_summary, file=f)
    return net_summary


def get_config(config_file, source=None):
    """Get configurations from a file or a StringIO object.

    Args:
        config_file: A full file path or a StringIO object.
        source: A string specifying the file name to which the configurations
            are saved using the function save_config. It is overwritten by
            config_file if config_file is a file path (default: None).

    Returns:
        A dict of configurations.

    """
    # Read config file
    config = ConfigParser(interpolation=ExtendedInterpolation())
    if isinstance(config_file, StringIO):
        config.read_file(config_file, source)  # Read as a file obj
    else:
        config.read(config_file)  # Read as a file name
        source = config_file

    # Output is a dict of dict, format = {section: {key: val}}
    output = OrderedDict()
    for section in config.sections():
        output[section] = OrderedDict()
        for k, v in config.items(section):
            try:
                output[section][k] = ast.literal_eval(v)
            except ValueError as e:
                raise ValueError(str(e) + '\n%s: %s' % (k, v))

    output['config_file'] = os.path.basename(source) if source is not None else None
    output['config'] = StringIO()
    config.write(output['config'])

    return output


def save_config(config_args, output_dir):
    """Saves configurations.

    Args:
        config_args: The configurations.
        output_dir: The directory where the config file is saved.
            Note that the file basename is determined in config_args['config_file'].
    """
    with open(os.path.join(output_dir, config_args['config_file']), 'w') as f:
        f.write(config_args['config'].getvalue())


def load_np_data(file_path, allow_pickle=False):
    """Loads data from a single-array npy or npz file.

    Args:
        file_path: A full file path to a npy or npz file.
        allow_pickle : bool, optional
            Allow loading pickled object arrays stored in npy files. Reasons for
            disallowing pickles include security, as loading pickled data can
            execute arbitrary code. If pickles are disallowed, loading object
            arrays will fail (default: False).

    Returns:
        Loaded data.
    """
    if file_path is not None:
        data = np.load(file_path, allow_pickle=allow_pickle)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[data.files[0]]
        return data
    else:
        return None


def get_data_lists(data_lists_paths, data_dir=None):
    """Creates a multimodal data list for file reading.

    Args:
        data_lists_paths: A list of paths, each is a text file containing the list of filenames of a modality.
        data_dir: If not None, it is a str attached to the beginning of each filename.
            It is the directory that contains all input data when the filenames are relative paths.

    Returns:
        A list of filename lists, each filename list is for a modality.
    """
    if data_lists_paths is None:
        return None
    data_dir = data_dir or ''
    data_lists = []
    for dl_path in data_lists_paths:
        dl_path = os.path.expanduser(dl_path)
        with open(dl_path) as f:
            a_list = f.read().splitlines()
        a_list = [os.path.join(data_dir, fname) for fname in a_list]
        data_lists.append(a_list)
    return data_lists


def save_output(y, data_lists_test, idx_sample, output_dir, output_origin=None, suffix=''):
    """Saves a label map to a nii.gz file.
    Warning: this function is hard-coded for our BraTS'19 and KiTS'23 experiments.

    Args:
        y: A predicted or ground-truth label map.
        data_lists_test: A list of filename lists, each filename list is for a modality.
            We use data_lists_test[0][idx_sample] to get the patient ID.
        idx_sample: Index to the sample in data_lists_test.
        output_dir: The output directory.
        output_origin: If not None, it is the "image origin" (x, y, z) of the output (default: None).
            See ITK for the details of "image origin".
        suffix: The suffix attached to the output filename (default: '').
    """
    y = np.asarray(y, dtype=np.uint8)
    y = sitk.GetImageFromArray(y)
    if output_origin is not None:
        y.SetOrigin(output_origin)

    fname = data_lists_test[0][idx_sample]
    pid = fname.split('/')[-2]  # Patient IDs are the folder names for both BraTS and KiTS
    fname = os.path.join(output_dir, f'{pid}{suffix}.nii.gz')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    sitk.WriteImage(y, fname, True)


def read_img(filename):
    """Reads an image file to produce a Numpy array using SimpleITK.

    Args:
        filename: Image file name (full path).

    Returns:
        A Numpy array of the image.
    """
    img = sitk.ReadImage(filename)
    return sitk.GetArrayFromImage(img).astype(np.float32)

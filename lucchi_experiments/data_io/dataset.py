#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset

__author__ = 'Ken C. L. Wong'


class MultimodalImageDataset(Dataset):
    """For multimodal image data.

    Args:
        data_lists: A list of lists, each list contains the samples of a modality to be read.
        reader: For reading a sample as a Numpy array. If None (default), a dummy reader = lambda x: x is used.
        idx_x_modalities: Indexes to x (input) modalities (default: None).
        idx_y_modalities: Indexes to y (output) modalities (default: None).
        x_processing: A function that performs custom processing on x, e.g. data normalization (default: None).
        transform: A function that performs data transformation (default: None).
    """
    def __init__(self,
                 data_lists,
                 reader=None,
                 idx_x_modalities=None,
                 idx_y_modalities=None,
                 x_processing=None,
                 transform=None,
                 ):
        self.data_lists = data_lists
        self.reader = reader or (lambda x: x)
        self.idx_x_modalities = idx_x_modalities
        self.idx_y_modalities = idx_y_modalities
        self.x_processing = x_processing
        self.transform = transform

        if self.idx_x_modalities is None:
            assert self.idx_y_modalities is None
            self.idx_x_modalities = list(range(len(self.data_lists)))

    def __len__(self):
        return len(self.data_lists[0])

    def __getitem__(self, idx):
        x = np.stack([self.reader(self.data_lists[m][idx]) for m in self.idx_x_modalities])
        if self.x_processing is not None:
            x = self.x_processing(x)

        if self.idx_y_modalities is not None:
            y = np.stack([self.reader(self.data_lists[m][idx]) for m in self.idx_y_modalities])
            if self.transform is not None:
                x, y = self.transform(x, y)
            return x, y

        if self.transform is not None:
            x = self.transform(x)
        return x


class ImageTransform:
    """For transforming a 2D or 3D image with shape (C, H, W) or (C, D, H, W).
    For the arguments, the default value of None means no action is performed.

    Args:
        rotation_range: A scalar for 2D images and a list of length 3 (depth, height, width) for 3D images,
            in degrees (0 to 180) (default: None).
        shift_range: A list of length N for ND images, fraction of total size (default: None).
        zoom_range: Amount of zoom. A sequence of two (e.g., [0.7, 1.2]) (default: None).
        flip: A list of length N for ND images, boolean values indicating whether to randomly flip the
            corresponding axes or not (default: None).
        cval: Value used for points outside the boundaries after transformation (default: 0).
        augmentation_probability: Probability of performing augmentation for each sample (default: 1.0).
        seed: Random seed (default: None).
    """
    def __init__(self,
                 rotation_range=None,
                 shift_range=None,
                 zoom_range=None,
                 flip=None,
                 cval=0.,
                 augmentation_probability=1.0,
                 seed=None,
                 ):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.flip = flip
        self.cval = cval
        self.augmentation_probability = augmentation_probability
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, y=None):
        """Randomly augments a single image array.

        Args:
            x: A 3D or 4D array, a single image with channels, e.g., (C, D, H, W).
            y: The corresponding observation (default: None).

        Returns:
            A randomly transformed version of the input (same shape).
        """
        img_size_axis = np.arange(x.ndim)[1:]

        if self.rng.binomial(1, self.augmentation_probability):
            # Rotation
            theta = None
            if self.rotation_range is not None:
                if np.isscalar(self.rotation_range):
                    assert x.ndim == 3
                    if self.rotation_range:
                        theta = np.pi / 180 * self.rng.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0
                else:
                    assert len(self.rotation_range) == 3
                    theta = []
                    for rot in self.rotation_range:
                        theta.append(np.pi / 180 * self.rng.uniform(-rot, rot) if rot else 0)

            # Shift
            shift = None
            if self.shift_range is not None:
                assert len(self.shift_range) == x.ndim - 1
                shift = []
                for i, s in enumerate(self.shift_range):
                    shift.append(self.rng.uniform(-s, s) * x.shape[img_size_axis[i]] if s else 0)

            # Zoom
            zoom = None
            if self.zoom_range is not None:
                zoom = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])

            # Create transformation matrix

            transform_matrix = None

            # Rotation
            if theta is not None:
                if np.isscalar(theta) and theta != 0:
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]])
                    transform_matrix = rotation_matrix
                elif any(th != 0 for th in theta):
                    theta = theta[::-1]  # As sitk uses (x, y, z)
                    cd = np.cos(theta[0])
                    sd = np.sin(theta[0])
                    ch = np.cos(theta[1])
                    sh = np.sin(theta[1])
                    cw = np.cos(theta[2])
                    sw = np.sin(theta[2])
                    rotation_matrix = np.array(
                        [[ch * cw, -cd * sw + sd * sh * cw, sd * sw + cd * sh * cw, 0],
                         [ch * sw, cd * cw + sd * sh * sw, -sd * cw + cd * sh * sw, 0],
                         [-sh, sd * ch, cd * ch, 0],
                         [0, 0, 0, 1]]
                    )
                    transform_matrix = rotation_matrix

            # Shift
            if shift is not None and any(sh != 0 for sh in shift):
                shift = shift[::-1]  # As sitk uses (x, y, z)
                shift = np.asarray(shift)
                shift_matrix = np.eye(x.ndim)
                shift_matrix[:-1, -1] = shift
                transform_matrix = shift_matrix if transform_matrix is None else np.dot(shift_matrix, transform_matrix)

            # Zoom
            if zoom is not None and zoom != 1:
                zoom_matrix = np.eye(x.ndim)
                zoom_matrix[:-1, :-1] = np.eye(x.ndim - 1) * zoom
                transform_matrix = zoom_matrix if transform_matrix is None else np.dot(zoom_matrix, transform_matrix)

            if transform_matrix is not None:
                x = apply_transform(x, transform_matrix, self.cval)
                if y is not None:
                    y = apply_transform(y, transform_matrix, self.cval)

            if self.flip is not None:
                assert len(self.flip) == x.ndim - 1
                for i, fp in enumerate(self.flip):
                    if fp and self.rng.random() < 0.5:
                        x = flip_axis(x, img_size_axis[i])  # Warning: this function returns a view
                        if y is not None:
                            y = flip_axis(y, img_size_axis[i])  # Warning: this function returns a view

        if y is None:
            return x
        return x, y


def transform_matrix_offset_center(matrix, img_size):
    offset = np.array(img_size) / 2.0 + 0.5
    offset_matrix = np.eye(matrix.shape[0])
    offset_matrix[:-1, -1] = offset
    reset_matrix = np.eye(matrix.shape[0])
    reset_matrix[:-1, -1] = -offset
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, cval):
    """Applies the image transformation specified by a matrix.

    Args:
        x: A 3D or 4D numpy array, a single image with channel, e.g. (C, D, H, W).
        transform_matrix: A numpy array specifying the geometric transformation, 3D or 4D.
        cval: Value used for points outside the boundaries.

    Returns:
        The transformed version of the input.
    """
    img_size = x.shape[1:][::-1]  # As sitk uses (x, y, z)
    transform_matrix = transform_matrix_offset_center(transform_matrix, img_size)
    final_affine_matrix = transform_matrix[:-1, :-1]
    final_offset = transform_matrix[:-1, -1]

    transform = sitk.AffineTransform(final_affine_matrix.flatten(), final_offset)
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(cval)
    resample.SetTransform(transform)

    channel_images = []
    for x_channel in x:
        image = sitk.GetImageFromArray(x_channel)
        resample.SetSize(image.GetSize())
        resample.SetOutputSpacing(image.GetSpacing())
        resample.SetOutputOrigin(image.GetOrigin())
        image = resample.Execute(image)
        channel_images.append(sitk.GetArrayFromImage(image))

    x = np.stack(channel_images)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

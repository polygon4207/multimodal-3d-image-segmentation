import os
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image

from multimodal_3d_image_segmentation.experiments.utils import get_config, save_config


def read_tif(filename):
    """Reads a multipage tif file as a 3D Numpy array (D, H, W)."""
    with Image.open(filename) as img:
        slices = []
        for i in range(img.n_frames):
            img.seek(i)
            slices.append(np.array(img))
    return np.stack(slices)


def downsample(data, factor, is_label=False):
    """Reduces a 3D array by `factor` along each axis by averaging blocks of factor^3 voxels.
    For labels, a voxel is foreground if at least half of its block is foreground.
    """
    if factor == 1:
        return data
    d, h, w = data.shape
    data = data.reshape(d // factor, factor, h // factor, factor, w // factor, factor).mean(axis=(1, 3, 5))
    if is_label:
        return (data >= 0.5).astype(np.uint8)
    return data.astype(np.float32)


def tiling(
        input_dir: str,
        output_dir: str,
        tile_size=(160, 256, 256),
        scale_factor=1,
        modalities=None,
        ext=None,
):
    """Cuts the Lucchi training and testing volumes into non-overlapping tiles.
        Each tile is saved in its own folder, as a BraTS patient:
        `output_dir/<train or test>/<id>/<id>-<modality>.<ext>`

    Args:
        input_dir: The full path that contains the tif files.
        output_dir: The full path where the tiles are saved.
        tile_size: Tile size (d, h, w) at the original image size. Must be divisible by `scale_factor`.
        scale_factor: Downsampling factor of the tiles, 1 for the original image size.
        modalities: List of str of modalities (image, ground truth).
        ext: Image file extension, e.g. nii.gz
    """
    for volume, partition in [('training', 'train'), ('testing', 'test')]:
        image = read_tif(os.path.join(input_dir, f'{volume}.tif'))
        label = read_tif(os.path.join(input_dir, f'{volume}_groundtruth.tif'))
        label = (label > 0).astype(np.uint8)  # From {0, 255} to {0, 1}

        d, h, w = tile_size
        z = (image.shape[0] - d) // 2  # Only one tile along the depth (165 slices -> 160)
        for y in range(0, image.shape[1] - h + 1, h):
            for x in range(0, image.shape[2] - w + 1, w):
                tile_id = f'{partition}_y{y:03d}_x{x:04d}'
                tile_dir = os.path.join(output_dir, partition, tile_id)
                os.makedirs(tile_dir, exist_ok=True)
                for m, data, is_label in zip(modalities, (image, label), (False, True)):
                    tile = downsample(data[z:z + d, y:y + h, x:x + w], scale_factor, is_label)
                    tile = sitk.GetImageFromArray(tile)
                    sitk.WriteImage(tile, os.path.join(tile_dir, f'{tile_id}-{m}.{ext}'), True)


def main(config_file):
    config_args = get_config(config_file)

    output_dir = config_args['io']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_config(config_args, output_dir)

    tiling(config_args['io']['input_dir'], output_dir, **config_args['tiling'])

    print('Done!\n')


if __name__ == '__main__':
    main(sys.argv[1])  # python tiling.py config_tiling.ini

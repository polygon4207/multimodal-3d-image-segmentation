"""Diagnostic: are the predicted mitochondria probabilities too low to pass the 0.5 threshold?

Loads a trained model as run.py does, predicts the validation and testing samples while keeping the
probabilities, and prints:
1. the distribution of the mitochondria probability on the true mitochondria and on the background;
2. the Dice coefficient of the mitochondria for different thresholds (run.py uses argmax, i.e. 0.5).

Nothing is written: the pipeline, the configs, and the results are not modified.

Usage (from the repository root):
    python3 lucchi_experiments/debug/diagnose_probabilities.py <training config> [<data_dir>]

Examples:
    # Model trained at 128x128x80, tested at 128x128x80 (the data_dir of the config)
    python3 lucchi_experiments/debug/diagnose_probabilities.py lucchi_experiments/config_files/128x128x80/config_hartleymha.ini
    # Same model, tested at 256x256x160
    python3 lucchi_experiments/debug/diagnose_probabilities.py lucchi_experiments/config_files/128x128x80/config_hartleymha.ini ./data/tiles/256x256x160/
"""

import copy
import os
import sys
import numpy as np
import torch

from multimodal_3d_image_segmentation import nets
from multimodal_3d_image_segmentation.experiments.utils import get_config, get_data_lists, normalize_modalities, read_img

THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
NUM_BINS = 100  # Histogram of the probabilities, in steps of 0.01


def percentile(hist, q):
    """Approximate percentile (0-100) of the values summarized by a histogram on [0, 1]."""
    cum = np.cumsum(hist) / hist.sum()
    return (np.searchsorted(cum, q / 100) + 0.5) / NUM_BINS


def main(config_file, data_dir=None):
    config_args = get_config(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    input_lists = config_args['input_lists']
    data_dir = os.path.expanduser(data_dir or input_lists['data_dir'])
    input_args = config_args['input_args']
    idx_x = input_args['idx_x_modalities']
    idx_y = input_args['idx_y_modalities'][0]
    mask_val = input_args.get('mask_val', 0)  # Same default as run.py

    # Model, created and loaded as in run.py
    output_dir = os.path.expanduser(config_args['main']['output_dir'])
    model_args = copy.deepcopy(config_args['model'])
    model_args['in_channels'] = len(idx_x)
    model_args['ndim'] = 5
    model_args['device'] = device
    model = getattr(nets, model_args.pop('model_name'))(**model_args)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'model/model.pt'), weights_only=True,
                                     map_location=device))
    model.eval()

    print(f'Model:  {output_dir}')
    print(f'Data:   {data_dir}')
    print(f'Device: {device}')

    for split in ('valid', 'test'):
        data_lists = get_data_lists(input_lists[f'data_lists_{split}_paths'], data_dir)
        hist_mito = np.zeros(NUM_BINS)  # Probabilities on the true mitochondria
        hist_back = np.zeros(NUM_BINS)  # Probabilities on the background
        counts = {t: [0, 0, 0] for t in THRESHOLDS}  # Intersection, true, predicted (all samples pooled)

        for i in range(len(data_lists[0])):
            x = np.stack([read_img(data_lists[m][i]) for m in idx_x])
            x = normalize_modalities(x, mask_val=mask_val)
            y = read_img(data_lists[idx_y][i]) == 1
            with torch.no_grad():
                prob = model(torch.from_numpy(x[None]).to(device))[0, 1].cpu().numpy()  # Mitochondria probability

            hist_mito += np.histogram(prob[y], bins=NUM_BINS, range=(0, 1))[0]
            hist_back += np.histogram(prob[~y], bins=NUM_BINS, range=(0, 1))[0]
            for t in THRESHOLDS:
                pred = prob > t
                counts[t][0] += np.count_nonzero(pred & y)
                counts[t][1] += np.count_nonzero(y)
                counts[t][2] += np.count_nonzero(pred)

        print(f'\n===== {split} ({len(data_lists[0])} samples)')
        print('Mitochondria probability on the TRUE MITOCHONDRIA:')
        print('   ' + '  '.join(f'p{q}={percentile(hist_mito, q):.2f}' for q in (5, 25, 50, 75, 95))
              + f'   | > 0.5: {hist_mito[NUM_BINS // 2:].sum() / hist_mito.sum() * 100:.1f} %')
        print('Mitochondria probability on the BACKGROUND:')
        print('   ' + '  '.join(f'p{q}={percentile(hist_back, q):.2f}' for q in (50, 95, 99, 99.9))
              + f'   | > 0.5: {hist_back[NUM_BINS // 2:].sum() / hist_back.sum() * 100:.2f} %')
        print('Threshold   Dice (all samples pooled)   predicted / true mitochondria voxels')
        for t in THRESHOLDS:
            inter, true, pred = counts[t]
            dice = 2 * inter / (true + pred) if true + pred else float('nan')
            print(f'  {t:4.2f}{" (run.py)" if t == 0.5 else "         "}   {dice:.3f}                       {pred / true:.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])

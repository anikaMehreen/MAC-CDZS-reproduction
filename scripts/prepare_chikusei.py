"""
prepare_chikusei.py
====================
One-time preprocessing script that converts the raw Chikusei hyperspectral
dataset into the `Chikusei_imdb_128.pickle` file required by train_stage1.py.

Run this ONCE before any training. After the pickle exists, this script
is never needed again.

KNOWN ISSUES ENCOUNTERED (and how they were fixed here):
---------------------------------------------------------
1. HARDCODED ABSOLUTE PATHS (original bug):
   The original authors' script had their own machine paths hardcoded:
       '/home/zhangzhiyuan/data/Chikusei/...'
   These are replaced below with variables you can set before running.

2. hdf5storage REQUIRED (not in standard Colab):
   The Chikusei .mat file is saved in HDF5 format, not standard MATLAB
   format. scipy.io.loadmat cannot read it -- it silently loads garbage.
   You must install hdf5storage:
       pip install hdf5storage

3. MEMORY USAGE (the script uses ~8-10 GB RAM):
   Loading Chikusei (2517x2335x128 float32) + building 9x9 patches for
   all ~77,592 samples is heavy. On Colab free tier, run this in a fresh
   session with nothing else loaded. If it crashes with SIGKILL, use a
   High-RAM runtime (Runtime -> Change runtime type -> High-RAM).

4. PROTOCOL 4 PICKLE REQUIRED:
   The original used pickle.dump(..., protocol=4). Do NOT change this to
   protocol=5 -- train_stage1.py's pickle.load will fail if the protocol
   doesn't match the Python version that generated the file. protocol=4
   works on Python 3.4+ and is safe across Colab versions.

5. CHIKUSEI .mat FILE FORMAT:
   The data key inside the .mat file is 'chikusei' (lowercase).
   The label key path is: GT[0][0][0]
   Both are non-standard. load_data_HDF() handles this correctly already.

6. OUTPUT SHAPE:
   The resulting pickle contains:
       imdb['data']   : shape (77592, 9, 9, 128), dtype float32
       imdb['Labels'] : shape (77592,), dtype int64, values 0-18
       imdb['set']    : shape (77592,), dtype int64, all ones (all training)
   77592 = total labeled pixels in Chikusei across 19 classes.
   128 = number of spectral bands (after the authors' preprocessing).

HOW TO RUN ON GOOGLE COLAB:
----------------------------
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install hdf5storage -q
    !python scripts/prepare_chikusei.py
"""

import numpy as np
import pickle
import hdf5storage
from sklearn import preprocessing


# ============================================================
# UPDATE THESE THREE PATHS BEFORE RUNNING
# ============================================================
CHIKUSEI_DATA_FILE  = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei/HyperspecVNIR_Chikusei_20140729.mat'
CHIKUSEI_LABEL_FILE = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'
OUTPUT_PICKLE       = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei_imdb_128.pickle'
# ============================================================


def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    """Zero-pad a 3D array on the first two spatial dimensions."""
    new_matrix = np.lib.pad(
        old_matrix,
        ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)),
        'constant', constant_values=0
    )
    return new_matrix


def indexToAssignment(index_, Row, Col, pad_length):
    """Convert flat pixel indices to (row, col) assignments in the padded image."""
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    """Extract a (2*ex_len+1) x (2*ex_len+1) spatial patch centred at (pos_row, pos_col)."""
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def sampling(groundTruth):
    """
    Return a shuffled list of all labeled pixel indices.
    Shuffles within each class first, then shuffles the full list.
    """
    labels_loc = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
    np.random.shuffle(whole_indices)
    return whole_indices


def load_chikusei(image_file, label_file):
    """
    Load the Chikusei dataset from HDF5-format .mat files.

    NOTE: scipy.io.loadmat cannot read this file -- it is HDF5 format.
    hdf5storage.loadmat is required (pip install hdf5storage).

    Data key:  'chikusei'  -> shape (2517, 2335, 128)
    Label key: 'GT[0][0][0]' -> shape (2517, 2335)
    """
    print("Loading Chikusei data (this takes a few minutes)...")
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)

    data_all = image_data['chikusei']       # (2517, 2335, 128)
    label    = label_data['GT'][0][0][0]    # (2517, 2335)

    [nRow, nColumn, nBand] = data_all.shape
    print(f'Chikusei loaded: {nRow} x {nColumn} x {nBand}')

    gt = label.reshape(np.prod(label.shape[:2]), )  # flatten to (nRow*nColumn,)

    del image_data, label_data, label

    # Standardize: reshape to (N, bands), scale, reshape back
    data_flat   = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaled = preprocessing.scale(data_flat)
    data_scaled = data_scaled.reshape(2517, 2335, 128)

    print("Standardization done.")
    return data_scaled, gt


def build_imdb(data, gt):
    """
    Extract 9x9 spatial patches around every labeled pixel and pack into
    the imdb dictionary expected by train_stage1.py.

    patch_length = 4  ->  patch size = 2*4+1 = 9x9
    """
    patch_length = 4
    padded_data  = zeroPadding_3D(data, patch_length)

    np.random.seed(1334)  # fixed seed for reproducibility -- matches original authors'
    whole_indices = sampling(gt)
    print(f'Total labeled samples: {len(whole_indices)}')

    nSample = len(whole_indices)
    nBand   = data.shape[2]

    x = np.zeros((nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand))
    y = gt[whole_indices] - 1   # convert 1-indexed labels to 0-indexed

    whole_assign = indexToAssignment(
        whole_indices, data.shape[0], data.shape[1], patch_length
    )
    print("Extracting patches...")
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(
            padded_data, whole_assign[i][0], whole_assign[i][1], patch_length
        )
        if i % 5000 == 0:
            print(f'  {i} / {nSample}')

    del whole_assign, padded_data

    imdb = {
        'data':   np.zeros([nSample, 2*patch_length+1, 2*patch_length+1, nBand], dtype=np.float32),
        'Labels': np.zeros([nSample], dtype=np.int64),
        'set':    np.ones([nSample], dtype=np.int64),   # all 1 = all training
    }

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :] = x[iSample, :, :, :]
        imdb['Labels'][iSample]        = y[iSample]

    print(f"imdb built. data shape: {imdb['data'].shape}")
    return imdb


if __name__ == '__main__':
    data, gt = load_chikusei(CHIKUSEI_DATA_FILE, CHIKUSEI_LABEL_FILE)
    imdb     = build_imdb(data, gt)
    del data, gt

    print(f"Saving pickle to: {OUTPUT_PICKLE}")
    with open(OUTPUT_PICKLE, 'wb') as handle:
        # protocol=4 required -- do NOT change.
        # See docstring at the top of this file for why.
        pickle.dump(imdb, handle, protocol=4)

    print("Done. Pickle saved successfully.")
    print(f"  data shape:   {imdb['data'].shape}")
    print(f"  Labels shape: {imdb['Labels'].shape}")
    print(f"  Unique labels: {sorted(set(imdb['Labels'].tolist()))}")

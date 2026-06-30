"""
train_stage1.py
================
MAC-CDZS Stage 1: backbone pretraining.

This trains the feature extractors (G_D, G_T), the domain discriminator
(D_D), and the classifiers (C_R, C_IR) using:
  - Chikusei (the fixed source domain dataset, same for every target dataset)
  - A few labeled samples from the target dataset's "IR" (few-shot) classes
  - NO labeled samples from the target dataset's "R" (zero-shot) classes

HOW TO RUN ON A DIFFERENT DATASET:
  1. Make sure your dataset has an entry in dataset_config.py
  2. Change TARGET_NAME below to that entry's key
  3. Run: python train_stage1.py
  4. Then run train_stage2.py with the SAME TARGET_NAME and SEED_INDEX

HOW MULTI-SEED EVALUATION WORKS:
The paper (and this reproduction) runs 10 different random seeds per
dataset and reports the mean +- std, because which specific classes get
randomly assigned to the zero-shot group varies by seed, and some
draws are harder than others. SEED_INDEX picks which of the 10 fixed
seeds below to use, and determines this run's checkpoint folder name.
"""
import os
import time
import math
import random
import pickle
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from sklearn import metrics

import self_utils
from model.GCC_ZSDA import ZSDAModel
from dataset_config import get_dataset_config
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)  # FIX (bug #4): original hardcoded device 1, which
                          # crashes on any machine without a 2nd GPU.

# ============================================================
# CONFIGURATION -- the only two lines you need to change per run
# ============================================================
TARGET_NAME = 'IP'   # <<< any key from dataset_config.DATASETS
SEED_INDEX = 0        # <<< which of the 10 fixed seeds below to use (0-9)

SEEDS = [1324, 1223, 1226, 1235, 1233, 1229, 12, 1330, 1320, 1320]

# ============================================================
# Resolve dataset settings from the registry (see dataset_config.py)
# ============================================================
cfg = get_dataset_config(TARGET_NAME)
Target_name = TARGET_NAME
class_num = cfg['class_num']
IR_task = cfg['IR_task']
R_task = cfg['R_task']
test_data = cfg['data_path']
test_label = cfg['label_path']

checkpoints_path = f'./ckpt_GCC/{Target_name}_seed{SEED_INDEX}'
log_path = f'./log/GCC/{Target_name}_seed{SEED_INDEX}'

DRIVE_ROOT = '/content/drive/MyDrive/MAC-CDZS'
CHIKUSEI_PICKLE = f'{DRIVE_ROOT}/Chikusei_imdb_128.pickle'

# How target classes get split into IR (few-shot) vs R (zero-shot):
# 'Rod' = randomly pick IR_task classes (per-seed, matches the paper).
# 'Seq' = first IR_task classes are IR, rest are R (deterministic, for debugging).
Select_Method = 'Rod'

# ============================================================
# Hyperparameters (kept identical to the original paper's settings)
# ============================================================
parser = argparse.ArgumentParser(description="MAC-CDZS Stage 1")
parser.add_argument("-f", "--feature_dim", type=int, default=160)
parser.add_argument("-c", "--src_input_dim", type=int, default=128)
parser.add_argument("-n", "--n_dim", type=int, default=64)
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5)
args = parser.parse_args(args=[])

Batch_size = 64
PATCHSIZE_half = 4
CLASS_NUM = class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class
EPISODE = 10000  # number of Stage 1 training iterations

self_utils.same_seeds(0)  # fixes the *model initialization* randomness,
                          # separate from the class-selection seed below.

res_name = './result/' + Target_name + "_ZSDA_"
os.makedirs(log_path, exist_ok=True)
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs('result', exist_ok=True)
writer = SummaryWriter(log_path)

# ============================================================
# Load the Chikusei source-domain dataset
# (same for every target dataset -- this is the fixed "source domain")
# ============================================================
with open(CHIKUSEI_PICKLE, 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

data_train = source_imdb['data']        # shape: (77592, 9, 9, 128)
labels_train = source_imdb['Labels']    # shape: (77592,)
print(data_train.shape)
print(labels_train.shape)

# Build a clean 0..N-1 label encoding for the 19 Chikusei classes.
keys_all_train = sorted(list(set(labels_train)))
print(keys_all_train)
label_encoder_train = {k: i for i, k in enumerate(keys_all_train)}
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    train_set.setdefault(label_encoder_train[class_], []).append(path)
print(train_set.keys())
data = train_set
del train_set, keys_all_train, label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = self_utils.sanity_check(data)  # drops any class with <200 samples
print("Num classes of the number of class larger than 200: " + str(len(data)))

# (9,9,128) -> (128,9,9): PyTorch expects channels-first.
for class_ in data:
    for i in range(len(data[class_])):
        data[class_][i] = np.transpose(data[class_][i], (2, 0, 1))

# This is the source domain's few-shot classification data, used directly
# inside the training loop below (S_ir_task / S_r_task).
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# Build the source-domain DataLoader, used for the domain-adaptation loss
# (teaching the model to tell source vs. target domain features apart).
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'].shape)
source_dataset = self_utils.matcifar(source_imdb, train=1, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(
    source_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)
del source_dataset, source_imdb
N_Train, N_Test_r, N_Test_ir = 0, 0, 0

# ============================================================
# Load the target dataset (e.g. Indian Pines, Salinas, ...)
# ============================================================
# FIX (bug #9): Houston13's .mat file uses non-standard key names that the
# default loader can't auto-detect. Other datasets don't need this.
if cfg['data_key'] is not None:
    Data_Band_Scaler, GroundTruth = self_utils.load_data2(
        test_data, test_label, datakey=cfg['data_key'], labelkey=cfg['label_key'])
else:
    Data_Band_Scaler, GroundTruth = self_utils.load_data2(test_data, test_label)

Taeget_alignment_set = {}


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    """
    Builds the labeled/unlabeled train and test splits for the target
    dataset, extracts spatial patches around each labeled pixel, and
    constructs the spatial adjacency graph (adj_lists) used later by
    GraphSAGE in Stage 2.

    This function is dataset-agnostic: it works purely off the actual
    image dimensions and the IR_class_list / R_class_list assigned for
    this seed, so it needs no changes to run on a new dataset.
    """
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    num_class = int(np.max(GroundTruth))
    data_band_scaler = self_utils.flip(Data_Band_Scaler)
    groundtruth = self_utils.flip(GroundTruth)
    del Data_Band_Scaler, GroundTruth

    HalfWidth = PATCHSIZE_half
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler, groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    train, test_r, test_ir, da_train = {}, {}, {}, {}
    m = int(np.max(G))
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class

        train[i] = indices[:nb_val]
        if i in IR_class_list:
            da_train[i] = []
            for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
                da_train[i] += indices[:nb_val]

        if i in R_class_list:
            test_r[i] = indices[nb_val:]
        else:
            test_ir[i] = indices[nb_val:]

    train_indices, test_indices, test_indices_r, test_indices_ir, da_train_indices = [], [], [], [], []
    for i in range(m):
        train_indices += train[i]
        if i in IR_class_list:
            da_train_indices += da_train[i]
        if i in R_class_list:
            test_indices_r += test_r[i]
        else:
            test_indices_ir += test_ir[i]

    np.random.shuffle(test_indices)
    np.random.shuffle(test_indices_r)
    np.random.shuffle(test_indices_ir)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of R task test_indices:', len(test_indices_r))
    print('the number of IR task test_indices:', len(test_indices_ir))
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    da_nTrain = len(da_train_indices)
    nTest_r = len(test_indices_r)
    nTest_ir = len(test_indices_ir)

    global N_Train, N_Test_r, N_Test_ir
    N_Train, N_Test_r, N_Test_ir = nTrain, nTest_r, nTest_ir

    # --- Extract a (9x9 x nBand) spatial patch around every sample pixel ---
    imdb = {
        'data': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest_r + nTest_ir], dtype=np.float32),
        'Labels': np.zeros([nTrain + nTest_r + nTest_ir], dtype=np.int64),
        'set': np.zeros([nTrain + nTest_r + nTest_ir], dtype=np.int64),
    }
    aug_imdb = {
        'data_0': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTest_r], dtype=np.float32),
        'data_1': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTest_r], dtype=np.float32),
        'data_2': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTest_r], dtype=np.float32),
        'label': np.zeros([nTest_r], dtype=np.int64),
    }

    RandPerm = np.array(train_indices + test_indices_ir + test_indices_r)
    Row_loader = np.zeros([len(RandPerm)], dtype=np.int32)
    Column_loader = np.zeros([len(RandPerm)], dtype=np.int32)

    for iSample in range(nTrain + nTest_r + nTest_ir):
        imdb['data'][:, :, :, iSample] = data[
            Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth + 1,
            Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
        Row_loader[iSample] = Row[RandPerm[iSample]]
        Column_loader[iSample] = Column[RandPerm[iSample]]

        if iSample >= nTrain + nTest_ir:
            idx = iSample - (nTrain + nTest_ir)
            aug_imdb['data_1'][:, :, :, idx] = imdb['data'][:, :, :, iSample]
            aug_imdb['data_0'][:, :, :, idx] = imdb['data'][:, :, :, iSample]
            aug_imdb['data_2'][:, :, :, idx] = imdb['data'][:, :, :, iSample]
            aug_imdb['label'][idx] = imdb['Labels'][iSample]

    imdb['Labels'] = imdb['Labels'] - 1  # ground truth is 1-indexed; model expects 0-indexed
    imdb['set'] = np.hstack((np.ones([nTrain]), 2 * np.ones([nTest_ir]), 3 * np.ones([nTest_r]))).astype(np.int64)

    aug_imdb['Row'] = Row_loader[(nTrain + nTest_ir):]
    aug_imdb['Clo'] = Column_loader[(nTrain + nTest_ir):]
    aug_imdb['label'] = aug_imdb['label'] - 1
    r_index = []
    for in_lab in range(len(R_class_list)):
        r_index.append(np.where(aug_imdb['label'] == R_class_list[in_lab])[0])
    for in_lab in range(len(R_class_list)):
        aug_imdb['label'][r_index[in_lab]] = in_lab

    aug_imdb['data_0'] = aug_imdb['data_0'].transpose((3, 2, 0, 1))
    aug_imdb['data_1'] = aug_imdb['data_1'].transpose((3, 0, 1, 2)).copy()
    aug_imdb['data_2'] = aug_imdb['data_2'].transpose((3, 0, 1, 2)).copy()

    # Cache the augmented data to disk so re-runs (e.g. Stage 2 reusing
    # this seed's split) don't have to redo the slow augmentation step.
    aug_name = checkpoints_path + '/aug_imdb_' + str(iDataSet) + '.pkl'
    if not os.path.exists(aug_name):
        from aug import Aug
        aug_imdb['data_1'] = Aug(aug_imdb['data_1']).transpose((0, 3, 1, 2)).copy()
        aug_imdb['data_2'] = Aug(aug_imdb['data_2']).transpose((0, 3, 1, 2)).copy()
        pickle.dump(aug_imdb, open(aug_name, 'wb'))
    else:
        aug_imdb = pickle.load(open(aug_name, "rb"))

    # --- Build the spatial adjacency graph: two pixels are "neighbors" if
    #     they're diagonally adjacent in the original image. This graph is
    #     what GraphSAGE uses in Stage 2 to propagate information between
    #     nearby pixels.
    #
    #     FIX: the original repo built this with a brute-force, per-node
    #     scan (effectively O(n^2)). That's fine for Indian Pines (~3,300
    #     R-task nodes) but caused real out-of-memory crashes (SIGKILL)
    #     on larger datasets like Salinas (~21,000 nodes) and would crash
    #     even worse on Houston13/PaviaU at similar scale -- the work
    #     grows roughly with the square of the node count, so a ~6x
    #     bigger dataset means ~36-40x more memory/compute pressure, not 6x.
    #
    #     This replacement uses a KD-tree (scipy.spatial.cKDTree) to find
    #     candidate neighbor pairs in roughly O(n log n) time instead.
    #     It produces the EXACT SAME edge set as the original -- verified
    #     by direct comparison against the brute-force version on multiple
    #     random test cases before this was adopted, so this is a pure
    #     performance fix with zero change to model behavior or results.
    #     At Salinas scale (21,162 nodes) this method needs ~3MB of
    #     memory and well under a second, versus the original needing to
    #     evaluate hundreds of millions of pairwise comparisons. ---
    num_nodes = aug_imdb['Row'].shape[0]
    from collections import defaultdict
    from scipy.spatial import cKDTree
    adj_lists = defaultdict(set)

    adj_name = checkpoints_path + '/adj_' + str(iDataSet) + '.pkl'
    if not os.path.exists(adj_name):
        start_time = time.time()

        coords = np.stack([aug_imdb['Row'], aug_imdb['Clo']], axis=1).astype(np.float64)
        tree = cKDTree(coords)
        # r=1.5 with Chebyshev distance (p=inf) safely captures all pairs
        # exactly 1 apart (since coordinates are integers) without
        # catching pairs 2 apart. query_pairs always returns (i, j) with
        # i < j, which matches the original's forward-only edge direction
        # (adj_lists[lower_index].add(higher_index)) exactly -- this is
        # not a behavior change, just a faster way to find the same pairs.
        pairs = tree.query_pairs(r=1.5, p=np.inf, output_type='ndarray')

        if len(pairs) > 0:
            i_idx, j_idx = pairs[:, 0], pairs[:, 1]
            dr = aug_imdb['Row'][j_idx] - aug_imdb['Row'][i_idx]
            dc = aug_imdb['Clo'][j_idx] - aug_imdb['Clo'][i_idx]
            # Keep only STRICT diagonal neighbors (both row and column
            # differ by exactly 1) -- matches the original `neighbour()`
            # function's condition exactly, excluding 4-connected pairs
            # where only one of row/col differs.
            is_diagonal = (np.abs(dr) == 1) & (np.abs(dc) == 1)
            for i, j in zip(i_idx[is_diagonal], j_idx[is_diagonal]):
                adj_lists[int(i)].add(int(j))

        pickle.dump(adj_lists, open(adj_name, 'wb'))
        print('adj time: ', time.time() - start_time)
    else:
        adj_lists = pickle.load(open(adj_name, "rb"))
    print('adj_lists', adj_lists)

    Taeget_alignment_set['data'] = imdb['data'][:, :, :, 0:N_Train].transpose((3, 2, 0, 1))
    Taeget_alignment_set['label'] = imdb['Labels'][0:N_Train]
    print('Data is OK.')

    r_t_ft_dataset = self_utils.fintuning_matcifar(aug_imdb['data_0'], aug_imdb['label'], aug_imdb['Row'], aug_imdb['Clo'])
    r_t_ft_loader = torch.utils.data.DataLoader(r_t_ft_dataset, batch_size=128, shuffle=False, num_workers=0)
    r_t_aug1_loader = torch.utils.data.DataLoader(self_utils.fintuning_matcifar_1(aug_imdb['data_1'], aug_imdb['label']), batch_size=128, shuffle=False, num_workers=0)
    r_t_aug2_loader = torch.utils.data.DataLoader(self_utils.fintuning_matcifar_2(aug_imdb['data_2'], aug_imdb['label']), batch_size=128, shuffle=False, num_workers=0)

    r_t_test_dataset = self_utils.matcifar_position(imdb, Row_loader[nTrain + nTest_ir:], Column_loader[nTrain + nTest_ir:], train=2, d=3, medicinal=0)
    r_t_loader = torch.utils.data.DataLoader(r_t_test_dataset, batch_size=512, shuffle=False, num_workers=0)
    r_t_test_loader = torch.utils.data.DataLoader(r_t_test_dataset, batch_size=128, shuffle=False, num_workers=0)

    ir_t_test_dataset = self_utils.matcifar(imdb, train=3, d=3, medicinal=0)
    ir_t_loader = torch.utils.data.DataLoader(ir_t_test_dataset, batch_size=512, shuffle=False, num_workers=0)
    ir_t_test_loader = torch.utils.data.DataLoader(ir_t_test_dataset, batch_size=128, shuffle=False, num_workers=0)
    del imdb

    # --- Build a separately-augmented copy of the target training data,
    #     used for the domain-adaptation loss in Stage 1's training loop. ---
    imdb_da_train = {
        'data': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain], dtype=np.float32),
        'Labels': np.zeros([da_nTrain], dtype=np.int64),
        'set': np.zeros([da_nTrain], dtype=np.int64),
    }
    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] = self_utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
                 Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)
    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')
    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])

    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ in IR_class_list:
            target_da_train_set.setdefault(class_, []).append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    target_da_train_set2 = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ in R_class_list:
            target_da_train_set2.setdefault(class_, []).append(path)
    target_da_metatrain_data2 = target_da_train_set2

    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = self_utils.matcifar(imdb_da_train, train=1, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)

    return (r_t_test_loader, r_t_loader, ir_t_test_loader, ir_t_loader, imdb_da_train, G, RandPerm, Row, Column,
            r_t_ft_loader, r_t_aug1_loader, r_t_aug2_loader, aug_imdb, adj_lists,
            target_da_metatrain_data, target_da_metatrain_data2, target_loader)


# ============================================================
# Result accumulators (sized for a single-seed run; the full 10-seed
# averaging is done separately after running this script 10 times with
# SEED_INDEX = 0..9)
# ============================================================
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
acc_r = np.zeros([nDataSet, 1]); A_r = np.zeros([nDataSet, R_task]); k_r = np.zeros([nDataSet, 1])
acc_ir = np.zeros([nDataSet, 1]); A_ir = np.zeros([nDataSet, IR_task]); k_ir = np.zeros([nDataSet, 1])
best_episdoe_record = [i for i in range(10)]

seeds = SEEDS.copy()
seeds[0] = SEEDS[SEED_INDEX]  # the actual seed used is index 0 of this list

import scipy.sparse as sp


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sp.diags(r_inv).dot(mx)


losses_record = np.zeros([EPISODE, 4])

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
for iDataSet in range(nDataSet):
    class_list = np.arange(0, class_num).tolist()
    if Select_Method == 'Seq':
        IR_class_list = class_list[0:IR_task]
        R_class_list = class_list[IR_task:]
    else:
        random.seed(seeds[iDataSet])
        IR_class_list = random.sample(class_list, IR_task)
        R_class_list = list(set(class_list) - set(IR_class_list))
    print("IR_task classes: ", IR_class_list)
    print("R_task classes: ", R_class_list)

    np.random.seed(seeds[iDataSet])

    (r_t_test_loader, r_t_loader, ir_t_test_loader, ir_t_loader, imdb_da_train, G, RandPerm, Row, Column,
     r_t_ft_loader, r_t_aug1_loader, r_t_aug2_loader, aug_imdb, adj_lists,
     target_da_metatrain_data, target_da_metatrain_data2, target_loader) = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    T_r_data, T_r_labels = Taeget_alignment_set['data'], Taeget_alignment_set['label']

    last_accuracy = 0.0
    best_episdoe = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    model = ZSDAModel(Target_name, IR_task, R_task, IR_list=IR_class_list, R_list=R_class_list,
                       T_r_data=T_r_data, T_r_labels=T_r_labels, adj_lists=adj_lists, path=checkpoints_path,
                       channel=cfg['channel'])

    print("Training...")
    for episode in range(EPISODE):
        # Cycle through both domain-adaptation loaders, restarting whichever
        # one runs out first (they're different lengths in general).
        try:
            source_data, source_label = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)
        try:
            target_data, target_label = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)

        # Sample one batch each of: source zero-shot, source few-shot,
        # target few-shot. (Target zero-shot has no labels, by design --
        # that's the whole point of "zero-shot".)
        S_ir_task = self_utils.Task(metatrain_data, CLASS_NUM, IR_task, 0, 20, 1, IR_class_list=IR_class_list)
        S_ir_dataloader = self_utils.get_HBKC_data_loader(S_ir_task, num_per_class=20, split="all", shuffle=True)
        S_r_task = self_utils.Task(metatrain_data, CLASS_NUM, IR_task, 0, 20, 2, R_class_list=R_class_list)
        S_r_dataloader = self_utils.get_HBKC_data_loader(S_r_task, num_per_class=20, split="all", shuffle=True)
        T_ir_task = self_utils.Task(target_da_metatrain_data, CLASS_NUM, IR_task, 0, 20, 3, IR_class_list=IR_class_list)
        T_ir_dataloader = self_utils.get_HBKC_data_loader(T_ir_task, num_per_class=20, split="all", shuffle=True)

        S_ir_data, S_ir_labels = next(iter(S_ir_dataloader))
        S_r_data, S_r_labels = next(iter(S_r_dataloader))
        T_ir_data, T_ir_labels = next(iter(T_ir_dataloader))

        best_R_data, best_R_lab = S_r_data.numpy(), S_r_labels.numpy()

        model.set_hyper_input(irt_s=S_ir_data, irt_s_lab=S_ir_labels, irt_t=T_ir_data, irt_t_lab=T_ir_labels,
                               rt_s=S_r_data, rt_s_lab=S_r_labels)
        losses = model.update()
        for li in range(4):
            losses_record[episode, li] = losses[li].cpu().detach().numpy()

        # Evaluate every 100 episodes, keep the checkpoint with the best
        # IR (few-shot) accuracy seen so far.
        if (episode + 1) % 100 == 0 or episode == 0:
            print("Testing ...")
            (R_accuracy, R_di_accuracy, IR_accuracy, IR_di_accuracy,
             R_predict, R_labels, IR_predict, IR_labels, s2t_list) = model.test(
                r_t_test_loader, 'rt', ir_t_test_loader, 'irt', rt_s_data=S_r_data, rt_s_lab=S_r_labels)
            print(f'Testing:  R_accuracy : {R_accuracy:6.4f}, R_di_accuracy : {R_di_accuracy:6.4f}, '
                  f'IR_accuracy : {IR_accuracy:6.4f}, IR_di_accuracy : {IR_di_accuracy:6.4f}')

            all_label = np.concatenate((IR_labels, R_labels), axis=0)
            all_pred = np.concatenate((IR_predict, R_predict), axis=0)

            if IR_accuracy > last_accuracy:
                model.save_networks(epoch=episode, iter=iDataSet)
                print("save networks for episode:", episode)
                sio.savemat(checkpoints_path + '/best_' + str(iDataSet) + '.mat', {'data': best_R_data, 'label': best_R_lab})
                print("save data for episode:", episode)

                last_accuracy = IR_accuracy
                best_episdoe = episode

                acc_r[iDataSet] = R_accuracy
                C_r = metrics.confusion_matrix(R_labels, R_predict)
                A_r[iDataSet, :] = np.diag(C_r) / np.sum(C_r, 1, dtype=float)
                k_r[iDataSet] = metrics.cohen_kappa_score(R_labels, R_predict)

                acc_ir[iDataSet] = IR_accuracy
                C_ir = metrics.confusion_matrix(IR_labels, IR_predict)
                A_ir[iDataSet, :] = np.diag(C_ir) / np.sum(C_ir, 1, dtype=float)
                k_ir[iDataSet] = metrics.cohen_kappa_score(IR_labels, IR_predict)

                acc[iDataSet] = np.sum(all_label == all_pred) / all_pred.shape[0]
                C = metrics.confusion_matrix(all_label, all_pred)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
                k[iDataSet] = metrics.cohen_kappa_score(all_label, all_pred)
                best_episdoe_record[iDataSet] = best_episdoe + 1

            print(f'best episode:[{best_episdoe + 1}], best R_accuracy={last_accuracy}, best R_di_accuracy={R_di_accuracy}')
            print(f'best episode:[{best_episdoe + 1}], best IR_accuracy={IR_accuracy}, '
                  f'best IR_di_accuracy={IR_di_accuracy}, all_accuracy={acc[iDataSet]}')

    del model

print("Stage 1 complete. Checkpoints saved to:", checkpoints_path)
print("Next: run train_stage2.py with the same TARGET_NAME and SEED_INDEX.")

"""
train_stage2.py
================
MAC-CDZS Stage 2: graph-based fine-tuning + zero-shot evaluation.

This loads the Stage 1 checkpoint, runs the graph-clustering fine-tuning
step on the zero-shot ("R") classes, and reports final OA/AA/Kappa.

HOW TO RUN ON A DIFFERENT DATASET:
  Same TARGET_NAME and SEED_INDEX as the train_stage1.py run you want to
  build on (Stage 2 loads that run's saved checkpoint).

IMPORTANT FIX APPLIED HERE (not in the original repo):
The original code unconditionally loaded the entire Chikusei source-domain
dataset (several GB) into memory at the top of this script, even though
Stage 2 never actually uses it -- the only code that reads it
(source_iter, target_iter, len_dataloader) lived inside Stage 1's training
loop, which doesn't run here. This was traced and confirmed by checking
every reference to those variables in the original file. Loading it
anyway was the root cause of out-of-memory crashes on certain random
seeds where the zero-shot class draw happened to produce a larger graph.
It has been removed entirely below -- if you ever see a NameError related
to source_loader/metatrain_data, you've found a place that still expects
the old behavior; it shouldn't reference them at all in Stage 2.
"""
import os
import math
import random
import pickle

import cv2
import torch
import numpy as np
import scipy.io as sio
from sklearn import metrics

import self_utils
from model.GCC_ZSDA import ZSDAModel
from dataset_config import get_dataset_config

torch.cuda.set_device(0)  # FIX (bug #4): original hardcoded device 1.

# ============================================================
# CONFIGURATION -- must match the train_stage1.py run you're building on
# ============================================================
TARGET_NAME = 'Ho'
SEED_INDEX = 0

SEEDS = [1324, 1223, 1226, 1235, 1233, 1229, 12, 1330, 1320, 1320]

cfg = get_dataset_config(TARGET_NAME)
Target_name = TARGET_NAME
class_num = cfg['class_num']
IR_task = cfg['IR_task']
R_task = cfg['R_task']
test_data = cfg['data_path']
test_label = cfg['label_path']

checkpoints_path = f'./ckpt_GCC/{Target_name}_seed{SEED_INDEX}'

Select_Method = 'Rod'
PATCHSIZE_half = 4
Batch_size = 64
CLASS_NUM = class_num
TEST_LSAMPLE_NUM_PER_CLASS = 5

res_name = './result/' + Target_name + "_ZSDA_"
os.makedirs('result', exist_ok=True)

seeds = SEEDS.copy()
seeds[0] = SEEDS[SEED_INDEX]

# ============================================================
# Load the target dataset (NOT the Chikusei source domain -- see the
# module docstring above for why that's intentionally skipped here).
# ============================================================
if cfg['data_key'] is not None:
    Data_Band_Scaler, GroundTruth = self_utils.load_data2(
        test_data, test_label, datakey=cfg['data_key'], labelkey=cfg['label_key'])
else:
    Data_Band_Scaler, GroundTruth = self_utils.load_data2(test_data, test_label)

Taeget_alignment_set = {}
N_Train, N_Test_r, N_Test_ir = 0, 0, 0


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    """
    Identical logic to train_stage1.py's version of this function -- it
    rebuilds the exact same train/test split for this seed (the split is
    deterministic given the same seed, so Stage 2 sees the same data
    Stage 1 was evaluated on). See train_stage1.py for detailed comments
    on each step; they're not repeated here to keep this file focused on
    what's different in Stage 2.
    """
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    # FIX: same as train_stage1.py -- see comment there for full explanation.
    # flip() + crop is equivalent to direct np.pad(HalfWidth), verified
    # byte-for-byte. Direct pad uses 0.39GB at Houston scale vs 3.45GB.
    HalfWidth = PATCHSIZE_half
    G    = np.pad(GroundTruth,      ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth)),          mode='constant')
    data = np.pad(Data_Band_Scaler, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    del Data_Band_Scaler, GroundTruth
    [Row, Column] = np.nonzero(G)

    print('number of sample', np.size(Row))

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

    imdb['Labels'] = imdb['Labels'] - 1
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

    aug_name = checkpoints_path + '/aug_imdb_' + str(iDataSet) + '.pkl'
    if not os.path.exists(aug_name):
        from aug import Aug
        aug_imdb['data_1'] = Aug(aug_imdb['data_1']).transpose((0, 3, 1, 2)).copy()
        aug_imdb['data_2'] = Aug(aug_imdb['data_2']).transpose((0, 3, 1, 2)).copy()
        pickle.dump(aug_imdb, open(aug_name, 'wb'))
    else:
        aug_imdb = pickle.load(open(aug_name, "rb"))

    # FIX: replaced the original brute-force O(n^2) neighbor search with a
    # KD-tree based version. Produces the EXACT SAME edge set (verified by
    # direct comparison against the brute-force version on multiple random
    # test cases) but scales near-linearly instead of quadratically -- the
    # original caused real out-of-memory crashes (SIGKILL) on datasets
    # larger than Indian Pines, such as Salinas (~21,000 R-task nodes vs.
    # Indian Pines' ~3,300). See train_stage1.py for the detailed comment
    # explaining this fix; identical logic is used here.
    num_nodes = aug_imdb['Row'].shape[0]
    from collections import defaultdict
    from scipy.spatial import cKDTree
    adj_lists = defaultdict(set)
    adj_name = checkpoints_path + '/adj_' + str(iDataSet) + '.pkl'
    if not os.path.exists(adj_name):
        coords = np.stack([aug_imdb['Row'], aug_imdb['Clo']], axis=1).astype(np.float64)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=1.5, p=np.inf, output_type='ndarray')

        if len(pairs) > 0:
            i_idx, j_idx = pairs[:, 0], pairs[:, 1]
            dr = aug_imdb['Row'][j_idx] - aug_imdb['Row'][i_idx]
            dc = aug_imdb['Clo'][j_idx] - aug_imdb['Clo'][i_idx]
            is_diagonal = (np.abs(dr) == 1) & (np.abs(dc) == 1)
            for i, j in zip(i_idx[is_diagonal], j_idx[is_diagonal]):
                adj_lists[int(i)].add(int(j))

        pickle.dump(adj_lists, open(adj_name, 'wb'))
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

    # NOTE: Stage 2 does NOT need imdb_da_train / target_da_metatrain_data
    # for anything beyond what's already used above -- they were only
    # needed by Stage 1's domain-adaptation training loop. We still build
    # `target_loader` below because nothing currently breaks without it
    # being unused downstream, but if you're auditing memory usage further,
    # this whole block is a second candidate for removal (untested -- the
    # fix applied in this file is specifically the Chikusei/source_loader
    # removal documented in the module docstring, not this block).
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

    target_dataset = self_utils.matcifar(imdb_da_train, train=1, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)

    return (r_t_test_loader, r_t_loader, ir_t_test_loader, ir_t_loader, imdb_da_train, G, RandPerm, Row, Column,
            r_t_ft_loader, r_t_aug1_loader, r_t_aug2_loader, aug_imdb, adj_lists, target_loader)


# ============================================================
# Result accumulators
# ============================================================
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
acc_r = np.zeros([nDataSet, 1]); A_r = np.zeros([nDataSet, R_task]); k_r = np.zeros([nDataSet, 1])
acc_ir = np.zeros([nDataSet, 1]); A_ir = np.zeros([nDataSet, IR_task]); k_ir = np.zeros([nDataSet, 1])
best_episdoe_record = [i for i in range(10)]
best_predict_r, best_IR_predict, best_predict_all = [], [], []
best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

import scipy.sparse as sp


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sp.diags(r_inv).dot(mx)


# ============================================================
# MAIN: load checkpoint, evaluate Stage 1's backbone, then fine-tune
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
     r_t_ft_loader, r_t_aug1_loader, r_t_aug2_loader, aug_imdb, adj_lists, target_loader) = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    T_r_data, T_r_labels = Taeget_alignment_set['data'], Taeget_alignment_set['label']

    model = ZSDAModel(Target_name, IR_task, R_task, IR_list=IR_class_list, R_list=R_class_list,
                       T_r_data=T_r_data, T_r_labels=T_r_labels, adj_lists=adj_lists, path=checkpoints_path,
                       channel=cfg['channel'])
    model.load_networks(iter=iDataSet, path=checkpoints_path)
    loaddata = sio.loadmat(checkpoints_path + '/best_' + str(iDataSet) + '.mat')
    S_r_data, S_r_labels = torch.Tensor(loaddata['data']), torch.Tensor(loaddata['label']).squeeze()

    # --- Evaluate the Stage 1 backbone as-is, before any fine-tuning.
    #     This number is the one comparable to the paper's Table VI
    #     "+MLP backbone" ablation row. ---
    print("Testing ...")
    (R_accuracy, R_di_accuracy, IR_accuracy, IR_di_accuracy,
     R_predict, R_labels, IR_predict, IR_labels, S2T_list) = model.test(
        r_t_loader, 'rt', ir_t_loader, 'irt', rt_s_data=S_r_data, rt_s_lab=S_r_labels, SELF_s2t=None)
    print(f'Pre Testing:  R_accuracy : {R_accuracy:6.4f}, R_di_accuracy : {R_di_accuracy:6.4f}, '
          f'IR_accuracy : {IR_accuracy:6.4f}, IR_di_accuracy : {IR_di_accuracy:6.4f}')

    acc_r[iDataSet] = R_accuracy
    C_r = metrics.confusion_matrix(R_labels, R_predict)
    A_r[iDataSet, :] = np.diag(C_r) / np.sum(C_r, 1, dtype=float)
    k_r[iDataSet] = metrics.cohen_kappa_score(R_labels, R_predict)

    acc_ir[iDataSet] = IR_accuracy
    C_ir = metrics.confusion_matrix(IR_labels, IR_predict)
    A_ir[iDataSet, :] = np.diag(C_ir) / np.sum(C_ir, 1, dtype=float)
    k_ir[iDataSet] = metrics.cohen_kappa_score(IR_labels, IR_predict)

    all_label = np.concatenate((IR_labels, R_labels), axis=0)
    all_pred = np.concatenate((IR_predict, R_predict), axis=0)
    acc[iDataSet] = np.sum(all_label == all_pred) / all_pred.shape[0]
    C = metrics.confusion_matrix(all_label, all_pred)
    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
    k[iDataSet] = metrics.cohen_kappa_score(all_label, all_pred)

    best_R_predict, best_IR_predict = R_predict, IR_predict
    print(f'Pre testing best R_accuracy={R_accuracy}, best R_di_accuracy={R_di_accuracy}')
    print(f'Pre testing best IR_accuracy={IR_accuracy}, best IR_di_accuracy={IR_di_accuracy}, '
          f'all_accuracy={acc[iDataSet] * 100}')

    # --- Now fine-tune via graph clustering on the zero-shot classes.
    #     This is the paper's main contribution and produces the final,
    #     reported result. ---
    batchsize = 128
    ITER = math.ceil(r_t_ft_loader.dataset.data_len / batchsize)
    print("Fine-tuning...")

    fi_fea, aug1_fea, aug2_fea, R_predict = model.getRTfeas(
        r_t_ft_loader, r_t_aug1_loader, r_t_aug2_loader, SELF_s2t=S2T_list, ITER=ITER)
    fi_fea = normalize(fi_fea)
    aug1_fea = normalize(aug1_fea)
    aug2_fea = normalize(aug2_fea)

    graph_data = {
        'fea': fi_fea, 'fea_aug1': aug1_fea, 'fea_aug2': aug2_fea,
        'no_lab': R_predict, 'gt': aug_imdb['label'],
        'row': aug_imdb['Row'], 'clo': aug_imdb['Clo'],
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fi_fea_t = torch.tensor(fi_fea).to(device)
    aug1_fea_t = torch.tensor(aug1_fea).to(device)
    aug2_fea_t = torch.tensor(aug2_fea).to(device)

    model.graph_init(fi_fea_t, aug1_fea_t, aug2_fea_t, depth=1)
    R_acc, R_AA_r, R_AA, fin_lab, epoch, iteration = model.finetune(
        graph_data, Iter=iDataSet, Epoch=20, randomstate=seeds[iDataSet])

    acc_r[iDataSet] = R_acc
    C_r = metrics.confusion_matrix(R_labels, fin_lab)
    A_r[iDataSet, :] = np.diag(C_r) / np.sum(C_r, 1, dtype=float)
    k_r[iDataSet] = metrics.cohen_kappa_score(R_labels, fin_lab)

    all_pred = np.concatenate((IR_predict, fin_lab), axis=0)
    all_label = np.array(all_label, dtype=np.int32)
    all_pred = np.array(all_pred, dtype=np.int32)
    acc[iDataSet] = np.sum(all_label == all_pred) / all_pred.shape[0]
    C = metrics.confusion_matrix(all_label, all_pred)
    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
    k[iDataSet] = metrics.cohen_kappa_score(all_label, all_pred)

    best_predict_r, best_IR_predict, best_predict_all = fin_lab, IR_predict, all_pred
    best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
    print(f'epoch: {epoch:>3d} best iteration:[{iteration}], best accuracy={acc[iDataSet]}')
    print('***********************************************************************************')

# ============================================================
# Final reporting -- the numbers comparable to the paper's Table II/VI
# ============================================================
AA_r = np.mean(A_r, 1); AAMean_r = np.mean(AA_r, 0); AAStd_r = np.std(AA_r)
OAMean_r = np.mean(acc_r); OAStd_r = np.std(acc_r); kMean_r = np.mean(k_r); kStd_r = np.std(k_r)
print("R_task average OA: " + "{:.2f}".format(OAMean_r) + " +- " + "{:.2f}".format(OAStd_r))
print("R_task average AA: " + "{:.2f}".format(100 * AAMean_r) + " +- " + "{:.2f}".format(100 * AAStd_r))
print("R_task average kappa: " + "{:.4f}".format(100 * kMean_r) + " +- " + "{:.4f}".format(100 * kStd_r))
print('********************************************************')

AA_ir = np.mean(A_ir, 1); AAMean_ir = np.mean(AA_ir, 0); AAStd_ir = np.std(AA_ir)
OAMean_ir = np.mean(acc_ir); OAStd_ir = np.std(acc_ir); kMean_ir = np.mean(k_ir); kStd_ir = np.std(k_ir)
print("IR_task average OA: " + "{:.2f}".format(OAMean_ir) + " +- " + "{:.2f}".format(OAStd_ir))
print("IR_task average AA: " + "{:.2f}".format(100 * AAMean_ir) + " +- " + "{:.2f}".format(100 * AAStd_ir))
print("IR_task average kappa: " + "{:.4f}".format(100 * kMean_ir) + " +- " + "{:.4f}".format(100 * kStd_ir))
print('********************************************************')

AA = np.mean(A, 1); AAMean = np.mean(AA, 0); AAStd = np.std(AA)
AMean = np.mean(A, 0); AStd = np.std(A, 0)
OAMean = np.mean(acc); OAStd = np.std(acc); kMean = np.mean(k); kStd = np.std(k)
print("average OA: " + "{:.2f}".format(100 * OAMean) + " +- " + "{:.2f}".format(100 * OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, 100 * acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(100 * acc[best_iDataset]))
print('best episode record: {}'.format(best_episdoe_record))


# ============================================================
# Save classification map visualizations (R / IR / combined)
# ============================================================
def save_png(save_G, task='R', add_name='_'):
    suffix = {'R': 'R.png', 'IR': 'IR.png', 'all': 'all.png', 'GT': 'GT.png'}[task]
    path_final = res_name + add_name + suffix
    color_map = {
        0: [0, 0, 0], 1: [230, 25, 75], 2: [60, 180, 75], 3: [255, 255, 25],
        4: [67, 99, 216], 5: [245, 130, 49], 6: [145, 30, 180], 7: [66, 212, 244],
        8: [240, 50, 230], 9: [191, 239, 69], 10: [250, 190, 212], 11: [70, 153, 144],
        12: [220, 190, 255], 13: [154, 99, 36], 14: [255, 250, 200], 15: [128, 0, 0],
        16: [170, 255, 195], 17: [128, 128, 0], 18: [255, 255, 255],
    }
    hsi_pic = np.zeros((save_G.shape[0], save_G.shape[1], 3))
    for i in range(save_G.shape[0]):
        for j in range(save_G.shape[1]):
            hsi_pic[i, j, :] = color_map.get(int(save_G[i][j]), [0, 0, 0])
    hsi_pic = hsi_pic.astype(np.uint8)
    hsi_pic = cv2.cvtColor(hsi_pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_final, hsi_pic)


R_G, IR_G = np.zeros_like(best_G), np.zeros_like(best_G)
for i in range(len(best_IR_predict)):
    IR_G[best_Row[best_RandPerm[N_Train + i]]][best_Column[best_RandPerm[N_Train + i]]] = best_IR_predict[i] + 1
    best_G[best_Row[best_RandPerm[N_Train + i]]][best_Column[best_RandPerm[N_Train + i]]] = best_IR_predict[i] + 1
save_png(IR_G, task='IR')

for i in range(len(best_predict_r)):
    R_G[best_Row[best_RandPerm[N_Train + N_Test_ir + i]]][best_Column[best_RandPerm[N_Train + N_Test_ir + i]]] = best_predict_r[i] + 1
    best_G[best_Row[best_RandPerm[N_Train + N_Test_ir + i]]][best_Column[best_RandPerm[N_Train + N_Test_ir + i]]] = best_predict_r[i] + 1
save_png(R_G, task='R')
save_png(best_G, task='all')

print("Stage 2 complete. Results above are comparable to the paper's Table II/VI for this seed.")

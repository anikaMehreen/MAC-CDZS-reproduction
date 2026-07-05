import numpy as np
from sklearn.decomposition import PCA
import random
import pickle
import hdf5storage
from sklearn import preprocessing
import scipy.io as sio


def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(groundTruth):
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

def load_data_HDF(image_file, label_file):
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)
    data_all = image_data['chikusei']
    label = label_data['GT'][0][0][0]
    [nRow, nColumn, nBand] = data_all.shape
    print('chikusei', nRow, nColumn, nBand)
    gt = label.reshape(np.prod(label.shape[:2]), )
    del image_data, label_data, label
    data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data_all)
    data_scaler = data_scaler.reshape(2517, 2335, 128)
    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2):
    Data_Band_Scaler, gt = load_data_HDF(trainfn1, trainfn2)
    del trainfn1, trainfn2
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    patch_length = 4
    whole_data = Data_Band_Scaler
    padded_data = zeroPadding_3D(whole_data, patch_length)
    del Data_Band_Scaler

    np.random.seed(1334)
    whole_indices = sampling(gt)
    print('the whole indices', len(whole_indices))

    nSample = len(whole_indices)
    x = np.zeros((nSample, 2*patch_length+1, 2*patch_length+1, nBand))
    y = gt[whole_indices] - 1

    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1], patch_length)

    del whole_assign, whole_data, padded_data

    imdb = {}
    imdb['data']   = np.zeros([nSample, 2*patch_length+1, 2*patch_length+1, nBand], dtype=np.float32)
    imdb['Labels'] = np.zeros([nSample], dtype=np.int64)
    imdb['set']    = np.ones([nSample], dtype=np.int64)

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :] = x[iSample, :, :, :]
        imdb['Labels'][iSample] = y[iSample]
        if iSample % 1000 == 0:
            print('processed:', iSample, '/', nSample)

    print('Data is OK.')
    return imdb


if __name__ == '__main__':
    # ============================================================
    # UPDATE THESE THREE PATHS BEFORE RUNNING
    # ============================================================
    train_data_file  = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei/HyperspecVNIR_Chikusei_20140729.mat'
    train_label_file = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'
    output_pickle    = '/content/drive/MyDrive/MAC-CDZS_data/Chikusei_imdb_128.pickle'
    # ============================================================

    imdb = getDataAndLabels(train_data_file, train_label_file)

    with open(output_pickle, 'wb') as handle:
        pickle.dump(imdb, handle, protocol=4)

    print('Pickle saved to:', output_pickle)

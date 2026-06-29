import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import h5py


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



import torch.utils.data as data

class matcifar_position(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, Row, Clo, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.row = Row
        self.clo = Clo

        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)  # r_t

        self.x3 = np.argwhere(self.imdb['set'] == 2)  # ir_t

        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()

        self.x3 = self.x3.flatten()

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.r_t_test_data = self.imdb['data'][:, :, :, self.x2]
            self.r_t_test_labels = self.imdb['Labels'][self.x2]

            self.ir_t_test_data = self.imdb['data'][:, :, :, self.x3]
            self.ir_t_test_labels = self.imdb['Labels'][self.x3]

            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.r_t_test_data = self.r_t_test_data.transpose((3, 2, 0, 1))
                self.ir_t_test_data = self.ir_t_test_data.transpose((3, 2, 0, 1))

            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 1:
            img, target = self.train_data[index], self.train_labels[index]
            return img, target
        elif self.train == 2:
            img, target = self.r_t_test_data[index], self.r_t_test_labels[index]
            row_bat, clo_bat = self.row[index], self.clo[index]
            return img, target, row_bat, clo_bat
        else:
            img, target = self.ir_t_test_data[index], self.ir_t_test_labels[index]
            return img, target


    def __len__(self):
        if self.train == 1:
            return len(self.train_data)
        elif self.train == 2:
            return len(self.r_t_test_data)
        elif self.train == 3:
            return len(self.ir_t_test_data)

class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)  # r_t

        self.x3 = np.argwhere(self.imdb['set'] == 2)  # ir_t

        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()

        self.x3 = self.x3.flatten()

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.r_t_test_data = self.imdb['data'][:, :, :, self.x2]
            self.r_t_test_labels = self.imdb['Labels'][self.x2]

            self.ir_t_test_data = self.imdb['data'][:, :, :, self.x3]
            self.ir_t_test_labels = self.imdb['Labels'][self.x3]

            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.r_t_test_data = self.r_t_test_data.transpose((3, 2, 0, 1))
                self.ir_t_test_data = self.ir_t_test_data.transpose((3, 2, 0, 1))

            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 1:
            img, target = self.train_data[index], self.train_labels[index]
            return img, target
        elif self.train == 2:
            img, target = self.r_t_test_data[index], self.r_t_test_labels[index]

            return img, target
        else:
            img, target = self.ir_t_test_data[index], self.ir_t_test_labels[index]
            return img, target


    def __len__(self):
        if self.train == 1:
            return len(self.train_data)
        elif self.train == 2:
            return len(self.r_t_test_data)
        elif self.train == 3:
            return len(self.ir_t_test_data)

class fintuning_matcifar(data.Dataset):
    def __init__(self, data, label, row, clo):
        self.train_data = data
        self.train_labels = label
        self.row = row
        self.clo = clo
        self.data_len = self.train_labels.shape[0]
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        row_bat, clo_bat = self.row[index], self.clo[index]
        return img, target, row_bat, clo_bat
    def __len__(self):
       return len(self.train_data)

class fintuning_matcifar_1(data.Dataset):
    def __init__(self, data, label):
        self.train_data = data
        self.train_labels = label
        self.data_len = self.train_labels.shape[0]
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        return img, target
    def __len__(self):
       return len(self.train_data)
class fintuning_matcifar_2(data.Dataset):
    def __init__(self, data, label):
        self.train_data = data
        self.train_labels = label
        self.data_len = self.train_labels.shape[0]
    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        return img, target
    def __len__(self):
       return len(self.train_data)

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data
def load_data2(image_file, label_file, datakey=None, labelkey=None):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    if datakey is not None and labelkey is not None:
        data_key = datakey
        label_key = labelkey
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data(image_file, label_file, test_sup,  datakey=None, labelkey=None):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    sup_data = sio.loadmat(test_sup)


    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    if datakey is not None and labelkey is not None:
        data_key = datakey
        label_key = labelkey
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]
    sup_data = sup_data['suppixel_scale1']
    sup_data = np.expand_dims(sup_data, axis=-1)

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])
    Data_Band_Scaler = np.append(Data_Band_Scaler, sup_data, axis=-1)

    return Data_Band_Scaler, GroundTruth

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def flip_augmentation(data):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

class Task(object):
    def __init__(self, data, num_classes, IR_classnum, shot_num, query_num, thre, IR_class_list=None, R_class_list=None):
        '''
        Args:
            data:
            num_classes:
            shot_num:
            query_num:
            thre:  0 --> all num_classes; 1 --> part1 classes; 2--> rest classes
        '''
        self.data = data
        self.num_classes0 = num_classes
        self.IR_classnum = IR_classnum
        self.support_num = shot_num
        self.query_num = query_num
        self.num_classes = num_classes//2

        class_folders = sorted(list(data))
        if thre == 0:
            random.seed(123)
            class_list = random.sample(class_folders, self.num_classes0)
            labels = np.array(range(len(class_list)))
        elif thre == 1:
            self.sample_classes = self.IR_classnum
            class_list = IR_class_list
            labels = np.array(range(len(class_list)))
        elif thre == 2:
            class_list = R_class_list
            self.sample_classes = num_classes - IR_classnum
            labels = np.array(range(len(class_list)))
        elif thre == 3:
            random.seed(123)
            class_list = IR_class_list
            self.sample_classes = self.IR_classnum
            labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    if split == 'train':
        dataset = HBKC_dataset(task, split=split)
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle)
        loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    elif split == 'test':
        dataset = HBKC_dataset(task, split=split)
        sampler = ClassBalancedSampler(num_per_class, task.num_classes0, task.query_num, shuffle=shuffle)
        loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes0, sampler=sampler)
    elif split == 'all':
        dataset = HBKC_dataset(task, split='test')
        sampler = ClassBalancedSampler(num_per_class, task.sample_classes, task.query_num+task.support_num, shuffle=shuffle)
        loader = DataLoader(dataset, batch_size=num_per_class * task.sample_classes, sampler=sampler)

    return loader

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

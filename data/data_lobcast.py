import collections
import os.path
from collections import Counter

# import src.constants as cst
import numpy as np
import tqdm
import torch
from pprint import pprint

from torch.utils.data import DataLoader, random_split, Dataset

import lightning as L

from utils.padding import collate_fn, padding_mask

# NewLOB modules
from config import ExpConfigManager as ECM 

# cst = ECM("experiment/exp_settings/pred_FI_2010")

HORIZONS_MAPPINGS_FI = {
    1: -5,
    2: -4,
    3: -3,
    5: -2,
    10: -1
}

def rearrange_fi2010_feature(fi2010_data):
    """
    重新排列新的LOB数据的features顺序，以匹配原始顺序。
    
    参数:
    - new_data: tensor, 形状为 (batch_size, time_step, 40)，其中features的顺序为:
                [sell1,sellvol1,buy1,buyvol1,sell2,sellvol2,buy2,buyvol2,...,sell10,sellvol10,buy10,buyvol10]
    
    返回:
    - reordered_data: tensor, 形状为 (batch_size, time_step, 40)，重新排列后的features顺序为:
                      [buy10,buy9,...,buy1, sell1,sell2,...,sell10,buyvol10,buyvol9,...,buyvol1,sellvol1,sellvol2,...,sellvol10]
    """

    # 重新排列的索引列表
    reorder_indices = [
        38, 34, 30, 26, 22, 18, 14, 10, 6, 2,  # buy10, buy9, ..., buy1
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36,  # sell1, sell2, ..., sell10
        39, 35, 31, 27, 23, 19, 15, 11, 7, 3,  # buyvol10, buyvol9, ..., buyvol1
        1, 5, 9, 13, 17, 21, 25, 29, 33, 37   # sellvol1, sellvol2, ..., sellvol10
    ]

    # 执行特征重新排列
    reordered_data = fi2010_data[:, reorder_indices]

    return reordered_data


class FIDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        fi_data_dir,
        dataset_type='train',
        horizon=5,
        observation_length=100,
        train_val_split=0.8,
        n_trends=3,
        auction=False,
        normalization_method="Zscore",
        **kwargs
    ):
        assert horizon in [1, 2, 3, 5, 10]
        self.dataset_path = dataset_path
        self.fi_data_dir = fi_data_dir
        self.dataset_type = dataset_type
        self.train_val_split = train_val_split
        self.auction = auction
        self.normalization_type = normalization_method
        self.horizon = horizon
        self.observation_length = observation_length
        self.num_classes = n_trends

        # KEY call, generates the dataset
        self.data, self.samples_X, self.samples_y = None, None, None
        if os.path.exists(self.dataset_path):
            self.data = np.load(self.dataset_path)
            self.__prepare_X()
            self.__prepare_y()
        else: 
            self.__prepare_dataset()

        _, occs = self.__class_balancing(self.samples_y)
        # LOSS_WEIGHTS_DICT = {m: 1e6 for m in cst.Models}
        LOSS_WEIGHT = 1e6
        self.loss_weights = torch.Tensor(LOSS_WEIGHT / occs)

        self.samples_X = torch.from_numpy(self.samples_X).type(torch.FloatTensor)  # torch.Size([203800, 40])
        self.samples_y = torch.from_numpy(self.samples_y).type(torch.LongTensor)   # torch.Size([203800])
        self.x_shape = (self.observation_length, self.samples_X.shape[1])          # shape of a single sample

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.samples_X.shape[0] - self.observation_length

    def __getitem__(self, index):
        """ Generates samples of data. """
        sample = self.samples_X[index: index + self.observation_length], self.samples_y[index + self.observation_length - 1]
        return sample

    @staticmethod
    def __class_balancing(y):
        ys_occurrences = collections.Counter(y)
        occs = np.array([ys_occurrences[k] for k in sorted(ys_occurrences)])
        return ys_occurrences, occs

    def __parse_dataset(self):
        """ Reads the dataset from the FI files. """

        AUCTION = 'Auction' if self.auction else 'NoAuction'
        N = '1.' if self.normalization_type == 'Zscore' else '2.' if self.normalization_type == 'Minmax' else '3.'
        NORMALIZATION = 'Zscore' if self.normalization_type == 'Zscore' else 'MinMax' if self.normalization_type == 'Minmax' else 'DecPre'
        DATASET_TYPE = 'Training' if self.dataset_type == 'train' or self.dataset_type == 'valid' else 'Testing'
        DIR = self.fi_data_dir + \
                 "/{}".format(AUCTION) + \
                 "/{}{}_{}".format(N, AUCTION, NORMALIZATION) + \
                 "/{}_{}_{}".format(AUCTION, NORMALIZATION, DATASET_TYPE)

        NORMALIZATION = 'ZScore' if self.normalization_type == "Zscore" else 'MinMax' if self.normalization_type == "Minmax" else 'DecPre'
        DATASET_TYPE = 'Train' if self.dataset_type == 'train' or self.dataset_type == 'valid' else 'Test'

        F_EXTENSION = '.txt'

        # if it is training time, we open the 7-days training file
        # if it is testing time, we open the 3 test files
        if self.dataset_type == 'train' or self.dataset_type == 'valid':

            F_NAME = DIR + '/{}_Dst_{}_{}_CF_7'.format(DATASET_TYPE, AUCTION, NORMALIZATION) + F_EXTENSION

            if not os.path.exists(F_NAME):
                error =  "\n\nFile {} not found! Make sure to follow the following steps.".format(F_NAME)
                error += "\n\n (1) Download the dataset in data/datasets, by running:\n{}".format("donwload")
                error += "\n (2) Unzip the file."
                error += "\n (3) Run: mv data/datasets/published/ data/datasets/FI-2010"
                error += "\n (4) Unzip data/datasets/FI-2010/BenchmarkDatasets/BenchmarkDatasets.zip in data/datasets/FI-2010/BenchmarkDatasets"
                error += "\n"
                raise FileNotFoundError(error)

            out_df = np.loadtxt(F_NAME)

            n_samples_train = int(np.floor(out_df.shape[1] * self.train_val_split))
            if self.dataset_type == 'train':
                out_df = out_df[:, :n_samples_train]

            elif self.dataset_type == 'valid':
                out_df = out_df[:, n_samples_train:]

        else:
            F_NAMES = [DIR + '/{}_Dst_{}_{}_CF_{}'.format(DATASET_TYPE, AUCTION, NORMALIZATION, i) + F_EXTENSION for i in range(7, 10)]
            out_df = np.hstack([np.loadtxt(F_NAME) for F_NAME in F_NAMES])

        self.data = out_df
        np.save(self.dataset_path, self.data)
        

    def __prepare_X(self):
        """ we only consider the first 40 features, i.e. the 10 levels of the LOB"""
        LOB_TEN_LEVEL_FEATURES = 40
        self.samples_X = self.data[:LOB_TEN_LEVEL_FEATURES, :].transpose()
        self.samples_X = rearrange_fi2010_feature(self.samples_X)

    def __prepare_y(self):
        """ gets the labels """
        # the last five elements in self.data contain the labels
        # they are based on the possible horizon values [1, 2, 3, 5, 10]
        self.samples_y = self.data[HORIZONS_MAPPINGS_FI[self.horizon], :]
        self.samples_y -= 1

    def __prepare_dataset(self):
        """ Crucial call! """

        self.__parse_dataset()

        self.__prepare_X()
        self.__prepare_y()

        print("Dataset type:", self.dataset_type, " - normalization:", self.normalization_type)
        occs, occs_vec = self.__class_balancing(self.samples_y)

        perc = ["{}%".format(round(i, 2)) for i in (occs_vec / np.sum(occs_vec)) * 100]
        print("Balancing", occs, "=>", perc)
        print()
        
    
class FIDataModule(L.LightningDataModule):
    """ Splits the datasets in TRAIN, VALIDATION_MODEL, TEST. """

    def __init__(self, datasets, batch_size,  is_shuffle_train=True, num_workers=20, seq_len=100, **kwargs):
        super().__init__()

        self.train_set, self.val_set, self.test_set = random_split(datasets, [int(0.8*len(datasets)), int(0.1*len(datasets)), int(0.1*len(datasets))])

        self.batch_size = batch_size
        self.is_shuffle_train = is_shuffle_train
        self.num_workers = num_workers
        # self.x_shape = self.test_set.x_shape
        self.num_classes = 3
        self.pin_memory = True
        self.seq_len = seq_len

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.is_shuffle_train, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers, collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers, collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, drop_last=False,num_workers=self.num_workers, collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))
# def prepare_data_fi():
#     fi_train, fi_val, fi_test = None, None, None

#     if not cst.data.is_test_only:
#         fi_train = FIDataset(
#             cst.data.base_dir,
#             dataset_type='train',
#             horizon = cst.data.horizon,
#             observation_length=cst.data.observation_length,
#             train_val_split=cst.data.split,
#             n_trends=cst.data.n_trends
#         )
        
#         fi_val = FIDataset(
#             cst.data.base_dir,
#             dataset_type='valid',
#             horizon = cst.data.horizon,
#             observation_length=cst.data.observation_length,
#             train_val_split=cst.data.split,
#             n_trends=cst.data.n_trends
#         )
        
#     fi_test = FIDataset(
#             cst.data.base_dir,
#             dataset_type='test',
#             horizon = cst.data.horizon,
#             observation_length=cst.data.observation_length,
#             train_val_split=cst.data.split,
#             n_trends=cst.data.n_trends
#         )
        

#     fi_dm = DataModule(
#         fi_train, fi_val, fi_test,
#         cst.data.batch_size,
#         cst.data.is_shuffle_train
#     )
#     return fi_dm
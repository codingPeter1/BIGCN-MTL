import argparse
import os
import random
import json

import numpy
import torch
from loguru import logger

from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None,valid_dict=None, test_dict=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

        #ADD
        self.valid_dict = valid_dict
        self.test_dict = test_dict

        self.neg_count=1

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        all_items = self.behavior_dict['all'].get(str(idx + 1), None)
        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
                # signal = [0]*(self.neg_count+2)
            else:
                pos = random.sample(items, 1)[0]
                signal = [idx + 1, pos]
                for i in range(self.neg_count):
                    neg = random.randint(1, self.item_count)
                    while np.isin(neg, self.behavior_dict['all'][str(idx + 1)]):
                        neg = random.randint(1, self.item_count)
                    signal.append(neg)
            total.append(signal)
        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path+str(args.stage)
        self.data_path = args.data_path
        self.degree_path =  args.degree_path
        self.device = args.device
        self.stage = args.stage
        self.now_user_degree = {}
        self.now_item_degree = {}
        self.old_user_degree = {}
        self.old_item_degree = {}
        self.user_item_behavior_matrix = {}
        self.user_item_behavior_matrix_all = {}
        for behavior in self.behaviors:
            self.now_user_degree[behavior] = None
            self.now_item_degree[behavior] = None
            self.old_user_degree[behavior] = None
            self.old_item_degree[behavior] = None
            self.user_item_behavior_matrix[behavior] = None
            self.user_item_behavior_matrix_all[behavior] = None

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])
        self.behaviors_adj = self.build_behavior_UI_adjacent_matrix()

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __build_sparse_matrix(self, ui_matrix, shape1, shape2):
        R = ui_matrix
        adj_mat = sp.dok_matrix((self.user_count + 1 + self.item_count + 1, self.user_count + 1 + self.item_count + 1),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.user_count + 1, self.user_count + 1:] = R
        adj_mat[self.user_count + 1:, :self.user_count + 1] = R.T

        adj_mat = adj_mat.tocsr()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0], dtype=np.float32)
        adj_mat = self.__convert_sp_mat_to_sp_tensor(adj_mat)
        return adj_mat.coalesce().to(self.device)

    def build_behavior_UI_adjacent_matrix(self):
        self.behavior_adj = {}
        for behavior in self.behaviors:
            self.behavior_adj[behavior] = self.__build_sparse_matrix(self.user_item_behavior_matrix[behavior],
                                                                     self.user_count, self.item_count)
        return self.behavior_adj

    def get_degree(self, behavior):
        if self.now_user_degree[behavior] is None or self.now_item_degree[behavior] is None or self.old_user_degree[
            behavior] is None or self.old_item_degree[behavior] is None:
            now_item_degree = None
            now_user_degree = None
            all_user_degree=None
            all_item_degree=None
            try:
                now_item_degree = np.load(self.path +self.degree_path+ '/item_degree_only-' + str(self.stage) + '-' + behavior + '.npy')
                all_item_degree = np.load(self.path +self.degree_path+ '/item_degree_all-' + str(self.stage) + '-' + behavior + '.npy')
                now_user_degree = np.load(self.path +self.degree_path+ '/user_degree_only-' + str(self.stage) + '-' + behavior + '.npy')
                all_user_degree = np.load(self.path +self.degree_path+ '/user_degree_all-' + str(self.stage) + '-' + behavior + '.npy')
                logger.info('============================load degree success=================================')
            except:
                logger.info('============================load degree false=================================')
                ui_behavior_matrix = self.user_item_behavior_matrix[behavior].tolil()
                ui_behavior_matrix_all = self.__get_ui_behavior_matrix_all(behavior).tolil()

                now_item_degree: numpy.array
                now_item_degree = np.array(ui_behavior_matrix.sum(axis=0))
                all_item_degree = np.array(ui_behavior_matrix_all.sum(axis=0))
                is_equal = np.array_equal(now_item_degree, all_item_degree)
                logger.info(f'behavior {behavior} is equal {is_equal}')
                now_item_degree = now_item_degree.reshape(-1, 1)
                all_item_degree = all_item_degree.reshape(-1, 1)
                np.save(self.path +self.degree_path+ '/item_degree_only-' + str(self.stage) + '-' + behavior, now_item_degree)
                np.save(self.path +self.degree_path+ '/item_degree_all-' + str(self.stage) + '-' + behavior, all_item_degree)

                now_user_degree: numpy.array
                now_user_degree = np.array(ui_behavior_matrix.sum(axis=1))
                all_user_degree = np.array(ui_behavior_matrix_all.sum(axis=1))
                now_user_degree = now_user_degree.reshape(-1, 1)
                all_user_degree = all_user_degree.reshape(-1, 1)
                np.save(self.path +self.degree_path+ '/user_degree_only-' + str(self.stage) + '-'+ behavior, now_user_degree)
                np.save(self.path +self.degree_path+ '/user_degree_all-' + str(self.stage) + '-'+ behavior, all_user_degree)
                
            finally:
                self.now_user_degree[behavior] = torch.from_numpy(now_user_degree).type(torch.FloatTensor).to(
                    self.device)
                self.now_item_degree[behavior] = torch.from_numpy(now_item_degree).type(torch.FloatTensor).to(
                    self.device)
                self.old_user_degree[behavior] = torch.from_numpy(all_user_degree - now_user_degree).type(
                    torch.FloatTensor).to(self.device)
                self.old_item_degree[behavior] = torch.from_numpy(all_item_degree - now_item_degree).type(
                    torch.FloatTensor).to(self.device)

            if (now_user_degree.shape[0] != self.user_count + 1) or (all_user_degree.shape[0] != self.user_count + 1):
                raise ValueError(f'{now_user_degree.shape[1]}!={self.user_count + 1}')
            if (now_item_degree.shape[0] != self.item_count + 1) or (all_item_degree.shape[0] != self.item_count + 1):
                raise ValueError(f'{now_item_degree.shape[1]}!={self.item_count + 1}')

        return self.now_user_degree[behavior], self.now_item_degree[behavior], self.old_user_degree[behavior], self.old_item_degree[behavior]

    def __get_sparse_interact_dict(self):
        self.edge_index = {}
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
               
                self.user_item_behavior_matrix[behavior] = csr_matrix((np.ones(len(row)), (row, col)),
                                                                      shape=(self.user_count + 1, self.item_count + 1))

                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1)
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.edge_index[behavior] = edge_index
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])

    def __get_ui_behavior_matrix_all(self, behavior):
        row = []
        col = []
        for s in range(0, self.stage + 1):
            train_file = self.data_path + f'{s}/' + behavior + '.txt'
            logger.info(train_file)
            with open(train_file, encoding='utf-8') as f:
                data = f.readlines()
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
        self.user_item_behavior_matrix_all[behavior] = csr_matrix((np.ones(len(row)), (row, col)),
                                                                  shape=(self.user_count + 1, self.item_count + 1))
        return self.user_item_behavior_matrix_all[behavior]

    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors,self.validation_interacts, self.test_interacts)

    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['pv', 'fav', 'cart', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/UB/', help='')
    parser.add_argument('--degree_path', type=str, default='/degree', help='')
    parser.add_argument('--device', type=str, default='cpu', help='')
    args = parser.parse_args()
    args.stage=1
    dataset = DataSet(args)
    dataset.get_degree('pv')
    dataset.get_degree('fav')
    dataset.get_degree('cart')
    dataset.get_degree('buy')
  
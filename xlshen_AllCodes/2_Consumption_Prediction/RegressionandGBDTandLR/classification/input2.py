#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_from_file as rff
from numpy import *

# 简化后代码


class Input2:

    @staticmethod
    def read_train(train_file_path):

        data_lists = list()
        label_list = list()
        train_file = open(train_file_path)
        try:
            for line in train_file:
                line = line.strip('\n')
                if line.startswith('@'):
                    continue
                else:
                    cur_list = list()
                    tmp = line.split(',')
                    for i in range(len(tmp) - 1):
                        cur_list.append(float(tmp[i]))
                    data_lists.append(cur_list)
                    label = int(tmp[-1])
                    label_list.append(label)
        except Exception as e:
            print(e)
        finally:
            train_file.close()
        return array(data_lists), array(label_list)

    @staticmethod
    def read_test(test_file_path, groundtruth_path):

        data_lists = list()
        label_list = list()
        test_dic_data = list()
        test_file = open(test_file_path)
        try:
            for line in test_file:
                line = line.strip('\n')
                if line.startswith('@'):
                    continue
                else:
                    cur_list = list()
                    tmp = line.split(',')
                    session = int(tmp[0])
                    item = int(tmp[1])
                    dic = dict()
                    dic[session] = item
                    test_dic_data.append(dic)
                    for i in range(2, len(tmp) - 1):
                        cur_list.append(float(tmp[i]))
                    data_lists.append(cur_list)
                    label = int(tmp[-1])
                    label_list.append(label)
        except Exception as e:
            print(e)
        finally:
            test_file.close()
        # session_item_data = rff.get_data_lists(groundtruth_path)
        # session_idx_dic = dict()
        # extract_session(session_item_data, session_idx_dic)
        # extract_label(test_dic_data, session_item_data, session_idx_dic, label_list)
        return array(data_lists), array(label_list), test_dic_data

    # @staticmethod
    # # 用于读取回归实验测试集。差别在于标签的设置
    # def read_test(test_file_path, groundtruth_path):
    #
    #     data_lists = list()
    #     label_list = list()
    #     test_dic_data = list()
    #     test_file = open(test_file_path)
    #     try:
    #         for line in test_file:
    #             line = line.strip('\n')
    #             if line.startswith('@'):
    #                 continue
    #             else:
    #                 cur_list = list()
    #                 tmp = line.split(',')
    #                 session = int(tmp[0])
    #                 item = int(tmp[1])
    #                 dic = dict()
    #                 dic[session] = item
    #                 test_dic_data.append(dic)
    #                 for i in range(2, len(tmp) - 1):
    #                     cur_list.append(float(tmp[i]))
    #                 data_lists.append(cur_list)
    #                 label = int(tmp[-1])
    #                 if label == 1:
    #                     label = 10
    #                 else:
    #                     label = 5
    #                 label_list.append(label)
    #     except Exception as e:
    #         print(e)
    #     finally:
    #         test_file.close()
    #     # session_item_data = rff.get_data_lists(groundtruth_path)
    #     # session_idx_dic = dict()
    #     # extract_session(session_item_data, session_idx_dic)
    #     # extract_label(test_dic_data, session_item_data, session_idx_dic, label_list)
    #     return array(data_lists), array(label_list), test_dic_data


def extract_session(session_item_data, session_idx_dic):
    idx = 0
    for cur_list in session_item_data:
        session = cur_list[0]
        session_idx_dic[session] = idx
        idx += 1


# 提取测试数据的标签——回归分数
def extract_label(test_dic_data, session_item_data, session_idx_dic, label_list):
    for dic in test_dic_data:
        session = list(dic.keys())[0]
        item = dic[session]
        idx = session_idx_dic[session]
        groundtruth_buy_items = session_item_data[idx][1]
        if item in groundtruth_buy_items:
            label_list.append(1)
        else:
            label_list.append(0)

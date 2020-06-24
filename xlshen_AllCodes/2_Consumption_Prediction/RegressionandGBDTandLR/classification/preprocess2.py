#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os


class Preprocess2:

    # 整理后版本

    @staticmethod
    def extract_data(dataset_dir, yoochoose_data_dir, yoochoose_selected_dir, setting=0):

        # 提取分类实验数据集

        # 输入路径（注意：这里用于训练和测试的session_item.txt数据都来源于原始数据中的训练集点击和购买数据）
        RA_train_dir = dataset_dir + r'\train'
        RA_test_dir = dataset_dir + r'\test'
        RA_train_session_path = RA_train_dir + r'\session_item.txt'
        RA_test_session_path = RA_test_dir + r'\session_item.txt'

        # 输入路径
        # 截断数据setting设置为1。
        if setting == 0:
            clicks_path = yoochoose_data_dir + r'\yoochoose-clicks.dat'
            buys_path = yoochoose_data_dir + r'\yoochoose-buys.dat'
        else:
            clicks_path = yoochoose_data_dir + r'\yoochoose-clicks-selected.dat'
            buys_path = yoochoose_data_dir + r'\yoochoose-buys-selected.dat'

        # 输出路径
        clicks_selected_path = yoochoose_selected_dir + r'\yoochoose-clicks-selected.dat'
        buys_selected_path = yoochoose_selected_dir + r'\yoochoose-buys-selected.dat'
        # （注意：这里yoochoose-test-selected.dat文件的数据并非来源于yoochoose-test.dat文件，而是来源于yoochoose-clicks.dat
        # 和yoochoose-buys.dat文件，因为用于测试的session_item.txt数据也来源于原始数据中的训练集点击和购买数据）
        test_selected_path = yoochoose_selected_dir + r'\yoochoose-test-selected.dat'

        # 提取实验数据session
        train_session = set()
        extract_session(RA_train_session_path, train_session)
        test_session = set()
        extract_session(RA_test_session_path, test_session)

        # 根据实验数据session进行提取yoochoose-data-selected
        extract_and_print_data(clicks_path, train_session, clicks_selected_path)
        extract_and_print_data(buys_path, train_session, buys_selected_path)
        extract_and_print_data(clicks_path, test_session, test_selected_path)


def extract_session(file_path, session):
    file = open(file_path)
    try:
        for line in file:
            tmp = line.split(";")
            session_str = tmp[0]
            session.add(session_str)
    except Exception as e:
        print(e)
    finally:
        file.close()


def extract_and_print_data(in_file_path, session, out_file_path):
    in_file = open(in_file_path)
    out_file = open(out_file_path, 'w')
    try:
        for line in in_file:
            tmp = line.split(',')
            session_str = tmp[0]
            if session_str in session:
                out_file.write(line)
    except Exception as e:
        print(e)
    finally:
        in_file.close()
        out_file.close()


if __name__ == '__main__':

    # 提取Full/extracted中session_item.txt中的session对应的yoochoose-clicks.dat和yoochoose-buys.dat
    main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
    dataset_para = 'extracted'
    # 文件夹路径
    dataset_dir = main_dir + '\\' + dataset_para
    yoochoose_data_dir = main_dir
    # 输出文件夹路径
    yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
    # 文件路径
    RA_train_session_path = dataset_dir + r'\session_item.txt'
    # 输入文件路径
    clicks_path = yoochoose_data_dir + r'\yoochoose-clicks.dat'
    buys_path = yoochoose_data_dir + r'\yoochoose-buys.dat'
    # 输出文件路径
    clicks_selected_path = yoochoose_selected_dir + r'\yoochoose-clicks-selected.dat'
    buys_selected_path = yoochoose_selected_dir + r'\yoochoose-buys-selected.dat'
    # 假如输出文件夹不存在，则创建文件夹
    if not os.path.exists(yoochoose_selected_dir):
        os.makedirs(yoochoose_selected_dir)

    # 提取实验数据session
    train_session = set()
    extract_session(RA_train_session_path, train_session)

    # 根据实验数据session进行提取yoochoose-data-selected
    extract_and_print_data(clicks_path, train_session, clicks_selected_path)
    extract_and_print_data(buys_path, train_session, buys_selected_path)

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_from_file as rff
import os


def dataFormat(data_path, write_file_path):
    data = rff.get_data_lists(data_path)
    rating_lists = list()
    buy_score = 1
    unbuy_score = 0.5
    session_set = set() #无序不重复集合
    item_set = set()
    for cur_data in data:
        session = cur_data[0]
        session_set.add(session)
        buy_items = cur_data[1]
        unbuy_items = cur_data[2]
        for item in buy_items:
            item_set.add(item)
            rating_lists.append([session, item, buy_score])
        for item in unbuy_items:
            item_set.add(item)
            rating_lists.append([session, item, unbuy_score])
    print('完整 数据的全部session数目：', len(session_set))
    print('完整 数据的全部item数目：', len(item_set))
    print_rating_lists_to_file(rating_lists, write_file_path)


def print_rating_lists_to_file(rating_lists, write_file_path):
    write_file = open(write_file_path, 'w')
    try:
        for cur_rating in rating_lists:
            session = cur_rating[0]
            item = cur_rating[1]
            score = cur_rating[2]
            write_file.write(str(session)+' '+str(item)+' '+str(score)+'\n')
    except Exception as e:
        print(e)
    finally:
        write_file.close()


if __name__ == '__main__':

    main_dir=r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\sampling@alldata@partition'
    dataset_use_list = ['train', 'test']
    for dataset_use in dataset_use_list:
        #print(part, part_num, dataset_para ,dataset_use)

        #读入将要被预处理的数据
        cur_dir = main_dir + '\\' + dataset_use
        data_path = cur_dir + r"\session_item.txt"

        # 输出到一个文件夹中
        out_dir = r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\sampling@alldata@partition\BPRandMFalldata'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 输出到具体的文件中
        if dataset_use == 'train':
            print("------------此为训练集-------------")
            write_file_path = out_dir + r'\trainalldata.txt'
        else:
            print("------------此为测试集-------------")
            write_file_path = out_dir + r'\testalldata.txt'
        dataFormat(data_path, write_file_path)


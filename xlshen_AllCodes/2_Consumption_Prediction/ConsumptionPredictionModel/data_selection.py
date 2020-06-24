#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_from_file as rff
import print_to_file as p2f
import os

# 选出extracted数据集中购买了n个商品的session


# 选出数据集中只购买n个商品的session
def data_selection(in_file_path, out_file_dir, n):
    out_data_file_path = out_file_dir + r'\session_item.txt'
    out_items_file_path = out_file_dir + r'\items.txt'
    data = rff.get_data_lists(in_file_path)
    selected_data = list()
    for cur_data in data:
        buy_items = cur_data[1]
        if len(buy_items) == n:
            selected_data.append(cur_data)
    selected_items = extract_items(selected_data)
    p2f.print_data_lists_to_file(selected_data, out_data_file_path)
    p2f.print_list_to_file(selected_items, out_items_file_path)


# 提取session_item_data里的item
def extract_items(data):
    items_set = set()
    for cur_data in data:
        buy_items = cur_data[1]
        unbuy_items = cur_data[2]
        for item in buy_items:
            items_set.add(item)
        for item in unbuy_items:
            items_set.add(item)
    return list(items_set)


if __name__ == '__main__':
    # 选出extracted数据集中购买了n个商品的session
    main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
    in_file_path = main_dir + r'\extracted' + r'\session_item.txt'

    #那如果不要区分D1-6了，以下这部分代码就可以去掉

    for i in range(5,7): #要生成D1-6的话应该是range(1，7)才对
        # 选出数据集中购买了n个商品的session
        n = i
        out_data_para = 'D' + str(n)
        out_file_dir = main_dir + '\\' + out_data_para
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
        data_selection(in_file_path, out_file_dir, n)

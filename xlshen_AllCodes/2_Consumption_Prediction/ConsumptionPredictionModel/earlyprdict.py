#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_from_file as rff
import real_data
import statistic
import feature4
from matplotlib import pyplot

# early predict相关统计

# 统计各个（购买）session点击的商品中第一个出现的购买商品
def get_session_fisrtBuyItem(dic1, dic2, all_buy_sessions):
    session_fisrtBuyItem_dic = dict()
    for d in all_buy_sessions:
        click_items = dic1[d]
        buy_items = dic2[d]
        for click_item in click_items:
            if click_item in buy_items:
                session_fisrtBuyItem_dic[d] = click_item
                break
    return session_fisrtBuyItem_dic


if __name__ == '__main__':

    # 这里画图数据是那些同时包含购买商品和点击不购买商品的session
    main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'

    # 提取数据文件。
    print(1)
    sampling_para = 'extracted'
    dataset_dir = main_dir + '\\' + sampling_para
    data_path = dataset_dir + r"\session_item.txt"
    # items_path = main_dir + '\\' + sampling_para + r"\items.txt"
    # datapro = rff.get_2lists_dict(data_path)
    # all_items = rff.get_a_list(items_path)

    print(2)
    # 点击数据文件
    yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
    click_file_path = yoochoose_selected_dir + r"\yoochoose-clicks-selected.dat"
    buys_file_path = yoochoose_selected_dir + r"\yoochoose-buys-selected.dat"

    # 纵坐标数据——每个session第一次出现购买商品的位置
    # 纵坐标准备数据
    # dic1表示训练集点击数据每个session点击了哪些item（按点击顺序放置）
    dic1, all_sessions, all_items_set = real_data.get_session_itemList(click_file_path)
    # dic2表示训练集购买数据每个session购买了哪些item（按购买顺序放置）
    dic2, all_buy_sessions, all_buy_items_set = real_data.get_session_itemList(buys_file_path)
    # 每个session第一次出现购买商品的位置
    session_buyItemFirstClickedIdx_dic = statistic.get_session_buyIdx(dic1, dic2, all_buy_sessions)

    session_len_dic = feature4.get_session_len(click_file_path)
    #
    # 画session_len与session_buyItemFirstClickedIdx关系图
    y_len_list = list()
    y_buyItemFirstClickedIdx_list = list()
    for session in session_len_dic.keys():
        y_len = session_len_dic[session]
        y_buyItemFirstClickedIdx = session_buyItemFirstClickedIdx_dic[session]
        y_len_list.append(y_len)
        y_buyItemFirstClickedIdx_list.append(y_buyItemFirstClickedIdx)

    x = range(len(y_len_list))
    pyplot.plot(x, y_len_list, 'r', x, y_buyItemFirstClickedIdx_list, 'b')
    pyplot.show()

    # 纵坐标数据——各个（购买）session点击的商品中第一个出现的购买商品
    session_fisrtBuyItem_dic = get_session_fisrtBuyItem(dic1, dic2, all_buy_sessions)




#!/usr/bin/env python
# -*- coding:utf-8 -*-


import read_from_file as rff
from matplotlib import pyplot
import sys
sys.path.append(r'E:\ranking aggregation\code\classification')
import feature4
import feature5
import real_data
from itertools import combinations
import cleansing


# 相关统计
# #######画箱图（横坐标为购买商品数或购买商品在session浏览商品中第一次出现的位置）


# 获取商品所属类别——原始简单方法——对商品存在多个类别的情况，以后面出现的类别为准
def get_item_category(file_path):
    item_category_dic = dict()
    f = open(file_path, 'r')
    try:
        idx = 0
        for line in f:
            if idx % 1000000 == 0:
                print("processing current file, finish line:", idx)
            tmp = line.split(',')
            item_str = tmp[2]
            item = int(item_str)
            category = tmp[3]
            category = category.strip('\n')
            category = category.strip('\r')
            item_category_dic[item] = category
            idx += 1
    except Exception as e:
        print(e)
    finally:
        f.close()
    return item_category_dic


# 获取商品所属类别——对商品存在多个类别的情况，记录下这个商品所出现的所有类别情况
def get_item_category1(file_path, all_items):
    item_categoryList_dic = dict()
    item_categoryNum_dic = dict()
    item_flag_dic = dict()
    # 判断该item是否出现过。因为在原始数据集中存在item属于两个（可能有两个以上）类别的情况，要把这两种类别都记录下来
    for item in all_items:
        item_flag_dic[item] = 0
    f = open(file_path, 'r')
    try:
        idx = 0
        for line in f:
            if idx % 1000000 == 0:
                print("processing current file, finish line:", idx)
            tmp = line.split(',')
            item_str = tmp[2]
            item = int(item_str)
            category = tmp[3]
            category = category.strip('\n')
            category = category.strip('\r')
            # 判断该item是否出现过，若没有出现过：
            if item_flag_dic[item] == 0:
                item_categoryList_dic[item] = list()
                item_flag_dic[item] = 1
            # 该item出现过/没出现过，判断该item的类别是否已记录过
            if category in item_categoryList_dic[item]:
                continue
            else:
                item_categoryList_dic[item].append(category)
            idx += 1
    except Exception as e:
        print(e)
    finally:
        f.close()
    for item in item_categoryList_dic.keys():
        item_categoryNum_dic[item] = len(item_categoryList_dic[item])
    return item_categoryNum_dic


# 根据数据集中的商品类别信息计算相似度
def calc_similarity(data, item_category_dic):
    session_simiList_dic = dict()
    click1_count = 0
    for cur_data in data:
        session = cur_data[0]
        session_simiList_dic[session] = list()
        # 当前session的所有购买商品与点击不购买商品
        cur_items = cur_data[1] + cur_data[2]
        if len(cur_items) == 1:
            click1_count += 1
            continue
        # combinations列出从cur_items中取出两个元素的所有组合数
        item_pairs = list(combinations(cur_items, 2))
        for pair in item_pairs:
            item1 = pair[0]
            item2 = pair[1]
            category1 = item_category_dic[item1]
            category2 = item_category_dic[item2]
            simi = 0
            if category1 == category2:
                simi = 1
            session_simiList_dic[session].append(simi)
    print("只点击一个商品的session数目：", click1_count)
    return session_simiList_dic


def buyNum_sessionList_statistic(data):
    buyNum_sessionList_dic = dict()
    buyNum_list = list()
    idx = 1
    for cur_data in data:
        if idx % 100000 == 0:
            print("processing line:", idx)
        session = cur_data[0]
        buyNum = len(cur_data[1])
        if buyNum not in buyNum_list:
            buyNum_list.append(buyNum)
            buyNum_sessionList_dic[buyNum] = list()
        buyNum_sessionList_dic[buyNum].append(session)
        idx += 1
    return buyNum_sessionList_dic


# 对于购买了多个商品(2个或2个以上)的session，获取每个session每个相邻购买商品的位置差
# 输入：data, dic(dic表示点击数据每个session点击了哪些item，已按点击顺序放置)
def get_session_buyIntervalList(data, dic):
    session_buyIntervalList_dic = dict()
    for cur_data in data:
        session = cur_data[0]
        buy_items = cur_data[1]
        buy_num = len(buy_items)
        if buy_num > 1:
            click_items = dic[session]
            click_num = len(click_items)
            session_buyIntervalList_dic[session] = list()
            cur_idxs = list()
            # 获取当前session每个购买商品在当前session点击商品中的位置
            for item in buy_items:
                idx = click_items.index(item)
                cur_idxs.append(idx)
            cur_idxs.sort()
            # 计算购买商品的相邻位置差
            for i in range(buy_num-1):
                interval = cur_idxs[i+1] - cur_idxs[i]
                session_buyIntervalList_dic[session].append(interval)

    return session_buyIntervalList_dic


# 统计各个（购买）session点击的商品中第一次出现购买商品的位置
def get_session_buyIdx(dic1, dic2, all_buy_sessions):
    session_buyItemFirstClickedIdx_dic = dict()
    for d in all_buy_sessions:
        click_items = dic1[d]
        buy_items = dic2[d]
        idx = 1
        for click_item in click_items:
            if click_item in buy_items:
                session_buyItemFirstClickedIdx_dic[d] = idx
                break
            else:
                idx += 1
    return session_buyItemFirstClickedIdx_dic


# 统计购买数据中各个“第一个购买商品在session中的位置”下的sessionList
def buyIdx_sessionList_statistic(session_buyItemFirstClickedIdx_dic):
    buyIdx_sessionList_dic = dict()
    buyIdx_list = list()
    idx = 1
    for session in session_buyItemFirstClickedIdx_dic.keys():
        if idx % 100000 == 0:
            print("processing line:", idx)
        buyIdx = session_buyItemFirstClickedIdx_dic[session]
        if buyIdx not in buyIdx_list:
            buyIdx_list.append(buyIdx)
            buyIdx_sessionList_dic[buyIdx] = list()
        buyIdx_sessionList_dic[buyIdx].append(session)
        idx += 1
    return buyIdx_sessionList_dic


# 统计每个session的总购买商品数（每个商品如果购买多次算作1个）
def get_session_buyNum(data):
    session_buyNum_dic = dict()
    idx = 1
    for cur_data in data:
        if idx % 100000 == 0:
            print("processing line:", idx)
        session = cur_data[0]
        buy_items = cur_data[1]
        buyNum = len(buy_items)
        session_buyNum_dic[session] = buyNum
        idx += 1
    return session_buyNum_dic


# 画箱图——横坐标：x，如数据集中商品的购买数；纵坐标：y(list型)，如商品对相似度的list。
# 当flag=0时，不画出横坐标值为0和1的数据。
def drawBox(x_sessionList_dic, session_y_dic, flag=1):
    x_axis = list()
    y_axis = list()
    for x in x_sessionList_dic.keys():
        # 当不想画出x的值为0的数据时
        if flag == 0 and x == 0:
            continue
        if flag == 0 and x == 1:
            continue
        x_axis.append(x)
        # 这是横坐标值为x的session list，如购买商品数为buyNum的session list
        sessionList = x_sessionList_dic[x]
        # 这是值为x的所有session的y值的list，如购买商品数为buyNum的所有session的simiList
        cur_all_ls = list()
        for session in sessionList:
            ls = session_y_dic[session]
            for val in ls:
                if val <= 5:
                    cur_all_ls.append(val)
                    # cur_all_ls += ls
        y_axis.append(cur_all_ls)
    pyplot.boxplot(y_axis, labels=x_axis)
    pyplot.show()


# 画箱图——横坐标：x，如数据集中商品的购买数；纵坐标：y（数值型），如session len的值。
# 当flag=0时，不画出横坐标值为0的数据。
def drawBox1(x_sessionList_dic, session_y_dic, flag=1):
    x_axis = list()
    y_axis = list()
    for x in x_sessionList_dic.keys():
        # 当不想画出x的值为0的数据时
        if flag == 0 and x == 0:
            continue
        x_axis.append(x)
        # 这是横坐标值为x的session list，如购买商品数为buyNum的session list
        sessionList = x_sessionList_dic[x]
        # 这是值为x的所有session的y值的list，如购买商品数为buyNum的所有session len的list
        cur_all_val = list()
        for session in sessionList:
            val = session_y_dic[session]
            cur_all_val.append(val)
        y_axis.append(cur_all_val)
    pyplot.boxplot(y_axis, labels=x_axis)
    pyplot.show()


if __name__ == '__main__':
    # 这里箱图数据是整个原始数据集中的所有数据
    main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full1'

    # 提取数据文件。横坐标数据
    print(1)
    sampling_para = 'extracted1'
    data_path = main_dir + '\\' + sampling_para + r"\session_item_xxxxxxxxxxxxxxxxx.txt"
    # items_path = main_dir + '\\' + sampling_para + r"\items_xxxxxxxxxxxxxxxxx.txt"
    data = rff.get_data_lists(data_path)
    # # all_items = rff.get_a_list(items_path)
    # 横坐标数据——购买商品数
    buyNum_sessionList_dic = buyNum_sessionList_statistic(data)

    print(2)
    # 原始数据文件
    click_file_path = main_dir + r"\yoochoose-clicks_xxxxxxxxxxxxxxxxx.dat"
    buys_file_path = main_dir + r"\yoochoose-buys_xxxxxxxxxxxxxxxxx.dat"
    # 纵坐标数据。每个session长度（不同item，不是按照click算）
    # session_len_dic = feature4.get_session_len(click_file_path)
    # 纵坐标数据——session持续时间
    # session_lastTime_dic = feature5.get_session_lastTime(click_file_path)
    # 纵坐标数据——相似度——未完成
    # item_category_dic = get_item_category(click_file_path)
    # session_simiList_dic = calc_similarity(data, item_category_dic)
    # # 纵坐标数据——每个session第一次出现购买商品的位置
    # 纵坐标准备数据
    # # dic1表示训练集点击数据每个session点击了哪些item（按点击顺序放置）
    # dic1, all_sessions, all_items_set = real_data.get_session_itemList(click_file_path)
    # # dic2表示训练集购买数据每个session购买了哪些item（按购买顺序放置）
    # dic2, all_buy_sessions, all_buy_items_set = real_data.get_session_itemList(buys_file_path)
    # # 每个session第一次出现购买商品的位置
    # session_buyItemFirstClickedIdx_dic = get_session_buyIdx(dic1, dic2, all_buy_sessions)
    # 纵坐标数据——每个session第一次出现购买商品的位置
    # 纵坐标准备数据
    # dic1表示训练集点击数据每个session点击了哪些item（按点击顺序放置）
    dic1, all_sessions, all_items_set = real_data.get_session_itemList(click_file_path)
    # 购买了多个商品的session，每个session购买商品的相邻位置差
    session_buyIntervalList_dic = get_session_buyIntervalList(data, dic1)

    # # 画相似度的均值和方差
    # # cleansing.draw(buyNum_sessionList_dic, session_simiList_dic)
    # print(3)
    # 画箱形图
    # 横坐标是购买商品数
    # drawBox1(buyNum_sessionList_dic, session_len_dic)
    # 当flag=0时，不画出横坐标值为0的数据。
    # drawBox1(buyNum_sessionList_dic, session_buyItemFirstClickedIdx_dic, 0)
    # drawBox1(buyNum_sessionList_dic, session_lastTime_dic)
    # 当flag=0时，不画出横坐标值为0和1的数据
    drawBox(buyNum_sessionList_dic, session_buyIntervalList_dic, 0)

    # # ###############
    # #横坐标准备数据
    # # dic1表示训练集点击数据每个session点击了哪些item（按点击顺序放置）
    # dic1, all_sessions, all_items_set = real_data.get_session_itemList(click_file_path)
    # # dic2表示训练集购买数据每个session购买了哪些item（按购买顺序放置）
    # dic2, all_buy_sessions, all_buy_items_set = real_data.get_session_itemList(buys_file_path)
    # session_buyItemFirstClickedIdx_dic = get_session_buyIdx(dic1, dic2, all_buy_sessions)
    # # # 横坐标数据——购买商品在session中第一次出现的位置
    # buyIdx_sessionList_dic = buyIdx_sessionList_statistic(session_buyItemFirstClickedIdx_dic)
    # # # ###############
    # # # 纵坐标数据——session总购买数
    # session_buyNum_dic = get_session_buyNum(data)
    #
    # # 画箱形图
    # # 横坐标是购买商品数
    # # （这里横坐标只包含购买数据，因此第二个参数中没有购买的session不会进入画图）
    # drawBox1(buyIdx_sessionList_dic, session_buyNum_dic)

    # # test################test
    # # 统计数据集中商品类别情况
    # item_categoryNum_dic = get_item_category1(click_file_path, list(all_items_set))
    # categoryNum_itemList_dic = buyIdx_sessionList_statistic(item_categoryNum_dic)
    # categoryNum_itemNum_dic = dict()
    # itemNum_list = list()
    # for categoryNum in categoryNum_itemList_dic.keys():
    #     categoryNum_itemNum_dic[categoryNum] = len(categoryNum_itemList_dic[categoryNum])
    # print("stop here")

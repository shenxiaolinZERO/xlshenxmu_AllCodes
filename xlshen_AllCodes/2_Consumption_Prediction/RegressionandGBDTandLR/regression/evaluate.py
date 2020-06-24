#!/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
import random
import sys
#sys.path.append(r'C:\Users\ns\Documents\Python Scripts\projects\skyline_recommendation\rlso')
sys.path.append(r'I:\Papers\consumer\codeandpaper\RegressionandGBDTandLR\skyline_recommendation\rlso')
import recommendation1


class Evaluate:

    @staticmethod
    def go(y_predict, test_dic_data, session_item_data):#, session_idx_dic
        # session_score_dic_data, session_item_dic_data = extract_score_by_session(y_predict, test_dic_data)
        # 提取每个session各个item的回归分数（将两个dic的内容整合到一起，并进行随机扰动和排序）
        session_item_score_dic = extract_score_by_session2(y_predict, test_dic_data)
        #print("test**************************************")
        precision,MRR = recommendation1.evaluate(session_item_data, session_item_score_dic)
        # p1 = calc_precision_at_1(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # p2 = calc_precision_at_2(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # MRR = calc_MRR(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # print('p1: ' + ('%.4f' % p1))
        # print('p2: ' + ('%.4f' % p2))
        # print('MRR: ' + ('%.4f' % MRR))
        return precision,MRR

# 提取每个session各个item的回归分数
def extract_score_by_session(y_predict, test_dic_data):
    session_score_dic_data = list()
    session_item_dic_data = list()
    test_sessions = list()
    idx = 0
    for dic in test_dic_data:
        session = list(dic.keys())[0]
        item = dic[session]
        if len(test_sessions) == 0:
            cur_dic_1 = dict()  # 对应一个session各个item的分数
            cur_dic_2 = dict()  # 对应一个session各个item
            cur_dic_1[session] = list()
            cur_dic_2[session] = list()
            test_sessions.append(session)
        elif session != test_sessions[-1]:
            # 来了一个新的session ID，存放旧的session ID
            session_score_dic_data.append(cur_dic_1)
            session_item_dic_data.append(cur_dic_2)
            # 来了一个新的session ID
            cur_dic_1 = dict()  # 对应一个session各个item的分数
            cur_dic_2 = dict()  # 对应一个session各个item
            cur_dic_1[session] = list()
            cur_dic_2[session] = list()
            test_sessions.append(session)
        cur_dic_1[session].append(y_predict[idx])
        cur_dic_2[session].append(item)
        idx += 1
    # 最后一个session
    session_score_dic_data.append(cur_dic_1)
    session_item_dic_data.append(cur_dic_2)
    return session_score_dic_data, session_item_dic_data


# 提取每个session各个item的回归分数（将两个dic的内容整合到一起，并进行随机扰动和排序）
def extract_score_by_session2(y_predict, test_dic_data):
    session_item_score_dic = dict()
    session_list = list()
    idx = 0
    for dic in test_dic_data:
        session = list(dic.keys())[0]
        score = y_predict[idx]
        item = dic[session]
        # 第一个session
        if len(session_list) == 0:
            session_item_score_dic[session] = list()
            session_list.append(session)
        # 来了一个新的session
        elif session != session_list[-1]:
            # 处理上一个session
            pre_session = session_list[-1]
            item_score = session_item_score_dic[pre_session]
            # 对session中各个【商品、分数】数据进行随机扰动。防止存在各个item的预测分数值相等的情况（实际上确实存在）
            random.shuffle(item_score)
            # 对session中各个商品的分数进行排序
            item_score.sort(key=lambda x: x[1], reverse=True)

            session_item_score_dic[session] = list()
            session_list.append(session)
        # 还是原来的session；或者来了一个新的session（此时item_set已先重置为空）
        session_item_score_dic[session].append([item, score])
        idx += 1
    # 对最后一个session的分数进行随机扰动和排序
    item_score = session_item_score_dic[session]
    # 对session中各个【商品、分数】数据进行随机扰动。防止存在各个item的预测分数值相等的情况（实际上确实存在）
    random.shuffle(item_score)
    # 对session中各个商品的分数进行排序
    item_score.sort(key=lambda x: x[1], reverse=True)
    return session_item_score_dic


def calc_precision_at_1(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic):
    p = 0.0
    i = 0
    for cur_dic_1 in session_score_dic_data:
        session = list(cur_dic_1.keys())[0]
        cur_dic_2 = session_item_dic_data[i]
        test_items = cur_dic_2[session]

        scores = cur_dic_1[session]
        # 防止存在各个item的预测分数值相等的情况（实际上确实存在）
        if max(scores) == min(scores):
            max_score_index = random.randint(0, len(scores)-1)
            max_score_item = test_items[max_score_index]
        else:
            max_score = max(scores)
            max_score_index = scores.index(max_score)
            max_score_item = test_items[max_score_index]

        # groundtruth（注意groundtruth和测试数据的"idx"有所不一样。）
        idx = session_idx_dic[session]
        groundtruth_buy_items = session_item_data[idx][1]
        if max_score_item in groundtruth_buy_items:
            p += 1.0
        i += 1
    p /= i
    return p


def calc_precision_at_2(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic):
    p = 0.0
    i = 0
    for cur_dic_1 in session_score_dic_data:
        session = list(cur_dic_1.keys())[0]
        cur_dic_2 = session_item_dic_data[i]
        test_items = cur_dic_2[session]

        scores = copy.deepcopy(cur_dic_1[session])
        max_score = max(scores)
        max_score_index = scores.index(max_score)
        max_score_item = test_items[max_score_index]

        scores[max_score_index] = 0
        sub_max_score = max(scores)
        sub_max_score_index = scores.index(sub_max_score)
        sub_max_score_item = test_items[sub_max_score_index]
        # groundtruth（注意groundtruth和测试数据的"idx"有所不一样。）
        idx = session_idx_dic[session]
        groundtruth_buy_items = session_item_data[idx][1]
        if max_score_item in groundtruth_buy_items:
            p += 0.5
        if sub_max_score_item in groundtruth_buy_items:
            p += 0.5
        i += 1
    p /= i
    return p


def calc_MRR(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic):
    MRR = 0.0
    i = 0
    for cur_dic_1 in session_score_dic_data:
        session = list(cur_dic_1.keys())[0]
        cur_dic_2 = session_item_dic_data[i]
        test_items = cur_dic_2[session]

        # groundtruth（注意groundtruth和测试数据的"idx"有所不一样。）
        idx = session_idx_dic[session]
        groundtruth_buy_items = session_item_data[idx][1]

        scores = copy.deepcopy(cur_dic_1[session])
        for j in range(len(scores)):
            cur_max_score = max(scores)
            cur_max_score_index = scores.index(cur_max_score)
            cur_max_score_item = test_items[cur_max_score_index]
            if cur_max_score_item in groundtruth_buy_items:
                MRR += 1.0 / (j + 1)
                break
            else:
                scores[cur_max_score_index] = 0

        i += 1

    MRR /= i
    return MRR

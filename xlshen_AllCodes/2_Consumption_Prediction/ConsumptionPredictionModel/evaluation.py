#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 计算评价指标：p@n(n=session中的购买商品数)，MRR

class Evaluation:

    # 根据数据集购买了几个商品，分别计算对应的precision
    @staticmethod
    def calc_precision_at_n(session_item_data, session_item_prob_dic ,n):
        if n == 1:
            precision = calc_precision_at_1(session_item_data, session_item_prob_dic)
        elif n == 2:
            precision = calc_precision_at_2(session_item_data, session_item_prob_dic)
        elif n == 3:
            precision = calc_precision_at_3(session_item_data, session_item_prob_dic)
        elif n == 4:
            precision = calc_precision_at_4(session_item_data, session_item_prob_dic)
        elif n == 5:
            precision = calc_precision_at_5(session_item_data, session_item_prob_dic)
        elif n == 6:
            precision = calc_precision_at_6(session_item_data, session_item_prob_dic)
        else:
            print("calc_precision_at_n parameter error!!!!!!!!")
            exit()
        return precision

    @staticmethod
    def calc_MRR(session_item_data, session_item_prob_dic):
        MRR = calc_MRR_help(session_item_data, session_item_prob_dic)
        return MRR


def calc_precision_at_1(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        if cur_item_prob[0][0] in cur_buy_items:
            p += 1.0
    p /= len(session_item_data)
    return p


def calc_precision_at_2(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        if cur_item_prob[0][0] in cur_buy_items:
            p += 0.5
        if cur_item_prob[1][0] in cur_buy_items:
            p += 0.5
    p /= len(session_item_data)
    return p


def calc_precision_at_3(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        if cur_item_prob[0][0] in cur_buy_items:
            p += 0.34
        if cur_item_prob[1][0] in cur_buy_items:
            p += 0.33
        if cur_item_prob[2][0] in cur_buy_items:
            p += 0.33
    p /= len(session_item_data)
    return p


def calc_precision_at_4(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        for i in range(4):
            if cur_item_prob[i][0] in cur_buy_items:
                p += 0.25
    p /= len(session_item_data)
    return p


def calc_precision_at_5(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        for i in range(5):
            if cur_item_prob[i][0] in cur_buy_items:
                p += 0.2
    p /= len(session_item_data)
    return p


def calc_precision_at_6(session_item_data, session_item_prob_dic):
    p = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        for i in range(6):
            if cur_item_prob[i][0] in cur_buy_items:
                p += 1/6
    p /= len(session_item_data)
    return p



def calc_MRR_help(session_item_data, session_item_prob_dic):
    MRR = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        for i in range(len(cur_item_prob)):
            if (cur_item_prob[i][0]) in cur_buy_items:
                MRR += 1.0/(i+1)
                break
    MRR /= len(session_item_data)
    return MRR
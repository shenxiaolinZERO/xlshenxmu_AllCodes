#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 计算评价指标：p@n(n=session中的购买商品数)，MRR


# precision 是看前partnum个商品里面是购买商品的比例，
# MRR 是看前partnum个商品里面第一个商品的位置

class EvaluationAllData:

    # 根据数据集购买了几个商品，分别计算对应的precision
    @staticmethod
    def calc_precision_at_n(session_item_data, session_item_prob_dic,n):
        precision = 0.0
        for cur_data in session_item_data:

            session = cur_data[0]
            # n=len(cur_data[1])
            cur_buy_items = cur_data[1]
            cur_item_prob = session_item_prob_dic[session]
            print("计算precision中的购买个数",n)
            for i in range(n):
                if cur_item_prob[i][0] in cur_buy_items:
                   precision += 1/n
        #precision/= len(session_item_data) #计算session item data里面所有session的precision然后取平均
                   return precision

    @staticmethod
    def calc_MRR(session_item_data, session_item_prob_dic,n):
        MRR = calc_MRR_help(session_item_data, session_item_prob_dic,n)
        return MRR

# precision 是看前partnum个商品里面是购买商品的比例，
# mrr 是看前 partnum 个商品里面第一个商品的位置
def calc_MRR_help(session_item_data, session_item_prob_dic,n):
    MRR = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]

        for i in range(n):
        #for i in range(len(cur_item_prob)):
            if cur_item_prob[i][0] in cur_buy_items:
                MRR += 1.0/(i+1)
                break
            # MRR /= len(session_item_data)
            # return MRR
                #return MRR
            # return MRR #结果是0
        # return MRR  #no…结果都是0.5
    # return MRR  #no…结果是506.2262、532.7429、527
    MRR /= len(session_item_data)  #原来也是算平均的……（have break ：0.7922，0.8337，0.8262）
    return MRR                     #（have no break ：1.1382，1.1927，1.1817...no）


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


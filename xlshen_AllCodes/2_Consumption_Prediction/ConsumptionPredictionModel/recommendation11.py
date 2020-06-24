#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import print_to_file as p2f
import math
import csv
from evaluation import Evaluation
from evaluation_alldata import EvaluationAllData

# 原始：对每个session，对所有点击商品计算一遍各个商品的购买概率，然后从高到低排序。



class Recommendation11:

    @staticmethod
    def generate(U, V, theta, aspects_num, session_item_data, dic, item_session_times_dic,
                              session_click_stream, res_dir, part_num):

        # calculate favorite aspect of each session in test data
        session_aspect_dic = calc_favorite_aspect(V, aspects_num, session_item_data)

        # session_aspect_dic = calc_favorite_aspect2(U, session_item_data)
        # 加入item ICR
        # item_list = get_item_list(click_file_path)
        # item_ICR_dic = get_item_ICR(click_file_path, buy_file_path, item_list)

        # 计算各个session里浏览商品的购买概率，并进行排序
        # session_item_prob_dic = calc_item_prob(V, theta, aspects_num, session_item_data, session_aspect_dic)
        # 说明：calc_item_prob2：在结对比较的时候，如果一个商品在session里出现了超过两次，他的结对也是都需要出现的
        # 说明：calc_item_prob3：计算商品为skyline object概率的时候，每个项目的值前面都乘以 1+exp(商品在该session里出现次数)

        # 将结果输出到文件中
        res_file_path = res_dir + r'\Recommendation11.csv'
        file = open(res_file_path, 'a', newline='')
        writer = csv.writer(file)
        data = list()
        try:
            # 原始预测方法
            session_item_prob_dic = calc_item_prob(V, theta, aspects_num, dic, session_aspect_dic,
                                                   item_session_times_dic)

            print("计算评价指标时输出的partnum : ",part_num)

            MRR = EvaluationAllData.calc_MRR(session_item_data, session_item_prob_dic, part_num)
            precision = EvaluationAllData.calc_precision_at_n(session_item_data, session_item_prob_dic, part_num)
            data += [str('%.4f' % part_num), str('%.4f' % precision), str('%.4f' % MRR), ""]

            # # calc_item_prob2：在结对比较的时候，如果一个商品在session里出现了超过两次，他的结对也是都需要出现的
            session_item_prob_dic = calc_item_prob2(V, theta, aspects_num, dic, session_aspect_dic,
                                                    item_session_times_dic)
            precision = EvaluationAllData.calc_precision_at_n(session_item_data, session_item_prob_dic, part_num)
            MRR = EvaluationAllData.calc_MRR(session_item_data, session_item_prob_dic, part_num)
            data += [str('%.4f' % part_num),str('%.4f' % precision), str('%.4f' % MRR), ""]

            # # calc_item_prob3：计算商品为skyline object概率的时候，每个项目的值前面都乘以 1+exp(商品在该session里出现次数)
            session_item_prob_dic = calc_item_prob3(V, theta, aspects_num, dic, session_aspect_dic,
                                                    item_session_times_dic)
            precision = EvaluationAllData.calc_precision_at_n(session_item_data, session_item_prob_dic, part_num)
            MRR = EvaluationAllData.calc_MRR(session_item_data, session_item_prob_dic, part_num)
            data += [str('%.4f' % part_num), str('%.4f' % precision), str('%.4f' % MRR)]

            writer.writerow(data)

        except Exception as e:
            print(e)
        finally:
            file.close()


# 按照老师session推荐里favorite aspect的计算方式
def calc_favorite_aspect(V, aspects_num, session_item_data):
    session_aspect_dic = dict()
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_items = list()
        for v in cur_data[1]:
            cur_items.append(v)
        for v in cur_data[2]:
            cur_items.append(v)

        sum_of_vk = list()
        for k in range(aspects_num):
            s = 0
            for item in cur_items:
                s += V[item][k]
            sum_of_vk.append(s)

        max_val = max(sum_of_vk)
        favorite_aspect = sum_of_vk.index(max_val)

        session_aspect_dic[session] = favorite_aspect

    return session_aspect_dic


def calc_item_prob(V, theta, aspects_num, dic, session_aspect_dic, item_session_times_dic):
    session_item_prob_dic = dict()

    for session in dic.keys():
        cur_items = dic[session]

        # # 随机扰动session中各个item的位置（以便后续的排序更合理）
        # random.shuffle(cur_items)

        favorite_aspect = session_aspect_dic[session]

        session_item_prob_dic[session] = list()
        for w in cur_items:
            if sum(V[w]) == 0:
                prob = 0.0
                session_item_prob_dic[session].append([w, prob])
            else:
                temp_product = 1
                for v in cur_items:
                    if v != w:
                        for k in range(aspects_num):
                            if k == favorite_aspect:
                                if (V[w][k] + theta * V[v][k]) == 0:
                                    print('division error')
                                temp_product *= V[w][k] / (V[w][k] + theta * V[v][k])
                            else:
                                temp_product *= (theta * V[w][k]) / (V[v][k] + theta * V[w][k])
                # # 加入item ICR
                # item_ICR = item_ICR_dic[w]
                # temp_product *= item_ICR
                # 该session中商品w为skyline object的概率
                session_item_prob_dic[session].append([w, temp_product])

        cur_item_prob = session_item_prob_dic[session]
        # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
        s = 0
        for i in range(len(cur_item_prob)):
            s += cur_item_prob[i][1]
        if s != 0:
            normalize(cur_item_prob)
        # 对当前session中各个商品的分数进行排序
        cur_item_prob.sort(key=lambda x: x[1], reverse=True)

    # prob_file_path = r'E:\ranking aggregation\code\result\yoochoose\Full\sampling@0.01@partition\train\prob.txt'
    # p2f.print_list_dict_to_file()

    # for cur_data in session_item_data:
    #     session = cur_data[0]
    #     cur_item_prob = session_item_prob_dic[session]
    #     print(cur_item_prob)

    return session_item_prob_dic


# 在结对比较的时候，如果一个商品在session里出现了超过两次，他的结对也是都需要出现的
def calc_item_prob2(V, theta, aspects_num, dic, session_aspect_dic, item_session_times_dic):
    session_item_prob_dic = dict()

    for session in dic.keys():
        cur_items = dic[session]

        # # 随机扰动session中各个item的位置（以便后续的排序更合理）
        # random.shuffle(cur_items)

        favorite_aspect = session_aspect_dic[session]

        session_item_prob_dic[session] = list()
        for w in cur_items:
            w_times = item_session_times_dic[(w, session)]
            if sum(V[w]) == 0:
                prob = 0.0
                session_item_prob_dic[session].append([w, prob])
            else:
                temp = 1
                for v in cur_items:
                    if v != w:
                        v_times = item_session_times_dic[(v, session)]
                        temp2 = 1
                        for k in range(aspects_num):
                            if k == favorite_aspect:
                                if (V[w][k] + theta * V[v][k]) == 0:
                                    print('division error')
                                temp2 *= V[w][k] / (V[w][k] + theta * V[v][k])
                            else:
                                temp2 *= (theta * V[w][k]) / (V[v][k] + theta * V[w][k])
                            temp2 *= (w_times * v_times)
                        temp *= temp2
                # # 加入item ICR
                # item_ICR = item_ICR_dic[w]
                # temp_product *= item_ICR
                # 该session中商品w为skyline object的概率
                session_item_prob_dic[session].append([w, temp])

        cur_item_prob = session_item_prob_dic[session]
        # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
        s = 0
        for i in range(len(cur_item_prob)):
            s += cur_item_prob[i][1]
        if s != 0:
            normalize(cur_item_prob)
        # 对当前session中各个商品的分数进行排序
        cur_item_prob.sort(key=lambda x: x[1], reverse=True)

    # prob_file_path = r'E:\ranking aggregation\code\result\yoochoose\Full\sampling@0.01@partition\train\prob.txt'
    # p2f.print_list_dict_to_file()

    # for cur_data in session_item_data:
    #     session = cur_data[0]
    #     cur_item_prob = session_item_prob_dic[session]
    #     print(cur_item_prob)

    return session_item_prob_dic


# 计算商品为skyline object概率的时候，每个项目的值前面都乘以 1+exp(商品在该session里出现次数)
def calc_item_prob3(V, theta, aspects_num, dic, session_aspect_dic, item_session_times_dic):
    session_item_prob_dic = dict()

    for session in dic.keys():
        cur_items = dic[session]

        # # 随机扰动session中各个item的位置（以便后续的排序更合理）
        # random.shuffle(cur_items)

        favorite_aspect = session_aspect_dic[session]

        session_item_prob_dic[session] = list()
        for w in cur_items:
            w_times = item_session_times_dic[(w, session)]
            w_factor = 1 + math.exp(w_times)
            if sum(V[w]) == 0:
                prob = 0.0
                session_item_prob_dic[session].append([w, prob])
            else:
                temp_product = 1
                for v in cur_items:
                    if v != w:
                        v_times = item_session_times_dic[(v, session)]
                        v_factor = 1 + math.exp(v_times)
                        for k in range(aspects_num):
                            if k == favorite_aspect:
                                if (V[w][k] + theta * V[v][k]) == 0:
                                    print('division error in def calc_item_prob_help ')
                                    exit()
                                temp_product *= (w_factor * V[w][k]) / (w_factor * V[w][k] + theta * v_factor * V[v][k])
                            else:
                                temp_product *= (theta * w_factor * V[w][k]) / (v_factor * V[v][k] + theta * w_factor * V[w][k])
                # # 加入item ICR
                # item_ICR = item_ICR_dic[w]
                # temp_product *= item_ICR
                # 该session中商品w为skyline object的概率
                session_item_prob_dic[session].append([w, temp_product])

        cur_item_prob = session_item_prob_dic[session]
        # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
        s = 0
        for i in range(len(cur_item_prob)):
            s += cur_item_prob[i][1]
        if s != 0:
            normalize(cur_item_prob)
        # 对当前session中各个商品的分数进行排序
        cur_item_prob.sort(key=lambda x: x[1], reverse=True)

    # prob_file_path = r'E:\ranking aggregation\code\result\yoochoose\Full\sampling@0.01@partition\train\prob.txt'
    # p2f.print_list_dict_to_file()

    # for cur_data in session_item_data:
    #     session = cur_data[0]
    #     cur_item_prob = session_item_prob_dic[session]
    #     print(cur_item_prob)

    return session_item_prob_dic


# 形如[[1, 0.3], [2, 0.3]]， 归一化为[[1, 0.5], [2, 0.5]]
def normalize(ls):
    s = 0
    for i in range(len(ls)):
        s += ls[i][1]
    for i in range(len(ls)):
        ls[i][1] = ls[i][1]/s


def evaluate(session_item_data, session_item_prob_dic):
    p1 = calc_precision_at_1(session_item_data, session_item_prob_dic)
    p2 = calc_precision_at_2(session_item_data, session_item_prob_dic)
    MRR = calc_MRR(session_item_data, session_item_prob_dic)
    print('p1: ' + ('%.4f' % p1))
    print('p2: ' + ('%.4f' % p2))
    print('MRR: ' + ('%.4f' % MRR))
    return p1, p2, MRR


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


def calc_MRR(session_item_data, session_item_prob_dic):
    MRR = 0.0
    for cur_data in session_item_data:
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        cur_item_prob = session_item_prob_dic[session]
        for i in range(len(cur_item_prob)):
            if (cur_item_prob[i][0]) in cur_buy_items:
                MRR += 1.0/(i+1)
                break
    MRR /= len(session_item_data)  #原来也是算平均的……
    return MRR


if __name__ == '__main__':
    ls = [[1, 0.3], [2, 0.3]]
    normalize(ls)
    print(ls)

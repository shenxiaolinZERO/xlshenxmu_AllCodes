#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import print_to_file as p2f
import feature4
from matplotlib import pyplot
import math
import recommendation11
import csv


# 用于early predict

class Recommendation44:

    @staticmethod
    def generate(click_file_path, buy_file_path, test_file_path,
                 U, V, theta, aspects_num, session_item_data, dic, item_session_times_dic, session_click_stream, res_dir):

        # early predict策略之一——采用滑动窗口策略进行截断时的窗口大小设置
        WIN = 3

        # calculate favorite aspect of each session in test data
        # session_aspect_dic = recommendation11.calc_favorite_aspect(V, aspects_num, session_item_data)
        # session_aspect_dic = calc_favorite_aspect2(U, session_item_data)

        # early predict策略之一——求“购买该商品的确定性”（即decisiveness）
        # 获得商品的ICR
        item_list = get_item_list(click_file_path)
        item_ICR_dic = feature4.get_item_ICR(click_file_path, buy_file_path, item_list)
        # 获得每个商品的属性值的和（表示该商品在这个属性上的总体水平）
        item_aspectValue_dic = calc_item_aspectValue(V, aspects_num)
        # 计算ICR*aspectValue并进行归一化（用于表示购买该商品的确定性）
        item_decisiveness_dic = calc_decisiveness(item_ICR_dic, item_aspectValue_dic)
        # for item in item_decisiveness_dic.keys():
        #     print(item, item_decisiveness_dic[item])

        # 将结果输出到文件中
        res_file_path = res_dir + r'\Recommendation44.csv'
        file = open(res_file_path, 'a', newline='')
        writer = csv.writer(file)
        data = list()
        try:

            # 开始预测
            # 计算每个session里各个浏览商品的购买概率，并进行排序
            session_idx_item_prob_dic, session_item_prob_dic, session_cutoff_rate_dic, session_flag_dic = \
                calc_item_prob(V, theta, aspects_num, session_item_data, dic, item_session_times_dic
                               , session_click_stream, WIN, item_decisiveness_dic, 1)
            # 预测结果指标评估
            # precision@1
            p1 = evaluate(session_item_data, session_item_prob_dic)
            # earliness
            earliness = print_earliness(session_cutoff_rate_dic)
            data += ["", str('%.4f' % p1), str('%.4f' % earliness), ""]

            # 计算每个session里各个浏览商品的购买概率，并进行排序
            session_idx_item_prob_dic, session_item_prob_dic, session_cutoff_rate_dic, session_flag_dic = \
                calc_item_prob(V, theta, aspects_num, session_item_data, dic, item_session_times_dic
                               , session_click_stream, WIN, item_decisiveness_dic, 2)
            # 预测结果指标评估
            # precision@1
            p1 = evaluate(session_item_data, session_item_prob_dic)
            # earliness
            earliness = print_earliness(session_cutoff_rate_dic)
            data += ["", str('%.4f' % p1), str('%.4f' % earliness), ""]

            # 计算每个session里各个浏览商品的购买概率，并进行排序
            session_idx_item_prob_dic, session_item_prob_dic, session_cutoff_rate_dic, session_flag_dic = \
                calc_item_prob(V, theta, aspects_num, session_item_data, dic, item_session_times_dic
                               , session_click_stream, WIN, item_decisiveness_dic, 3)
            # 预测结果指标评估
            # precision@1
            p1 = evaluate(session_item_data, session_item_prob_dic)
            # earliness
            earliness = print_earliness(session_cutoff_rate_dic)
            data += ["", str('%.4f' % p1), str('%.4f' % earliness), ""]

            writer.writerow(data)

        except Exception as e:
            print(e)
        finally:
            file.close()

        # 预测结果辅助分析
        # 列出每个session的各个点击流在预测中排在“第一位”的所有商品
        # session_item1s_dic = proving_session_item1s(dic, session_idx_item_prob_dic, session_click_stream)

        # 采取滑动窗口进行early predict截断时预测结果辅助分析
        # sliding_result_analysis(session_item_data, session_item_prob_dic, session_item1s_dic)


        # 预测结果辅助分析
        # 每个session每次浏览预测排在第一位的商品中"最早出现购买商品的位置"
        # print_buyItemFirstRank1Idx(session_item_data, session_item1s_dic)


#
def sliding_result_analysis(session_item_data, session_item_prob_dic, session_item1s_dic):
    # 表示测试集中能通过"滑动窗口"找到截断位置的session数
    success_count = 0
    # 点击流长度统计
    all_click_stream_num = 0
    all_success_click_stream_num = 0
    all_fail_click_stream_num = 0
    # 预测正确的success
    correct_session = list()
    correct_num = 0
    # 预测错误的session
    wrong_session = list()
    wrong_num = 0
    all_item1_same_num = 0
    for cur_data in session_item_data:
        session = cur_data[0]
        buy_items = cur_data[1]
        # cur_click_stream = session_click_stream[session]
        # cur_click_stream_num = len(cur_click_stream)
        # all_click_stream_num += cur_click_stream_num
        # predict
        # 该session各点击流预测排在第一位的商品
        cur_item1_list = session_item1s_dic[session]
        # 截断/最终预测购买商品
        [item, prob] = session_item_prob_dic[session][0]
        if item in buy_items:
            correct_session.append([session, cur_item1_list, buy_items])
            correct_num += 1
        else:
            wrong_num += 1
            if (len(set(cur_item1_list))) == 1:
                all_item1_same_num += 1
            else:
                print('f', cur_item1_list, '预测', item, '购买', buy_items, '--session:', session)

                # flag = session_flag_dic[session]
                # if flag == 1:
                #     all_success_click_stream_num += cur_click_stream_num
                #     success_count += 1
                # else:
                #     all_fail_click_stream_num += cur_click_stream_num
                #     print('fail:', session, ':', cur_item1_list, 'buy:', buy_items)
    print(correct_session)
    print('correct_num', correct_num)
    print('wrong_num', wrong_num)
    print('in wrong_num, all_item1_same_num:', all_item1_same_num)
    # all_count = len(dic)
    # print('#####')
    # print('all test session num', all_count)
    # print('success_count of judge_by_sliding_window', success_count)
    # fail_count = all_count - success_count
    # print('fail_count of judge_by_sliding_window', fail_count)
    # print('#####')
    # print('average length of all click stream:', all_click_stream_num/all_count)
    # print('average length of success click stream:', all_success_click_stream_num/success_count)
    # print('average length of fail click stream:', all_fail_click_stream_num/fail_count)


# 获取训练集里所有的item
def get_item_list(file_path):
    item_set = set()
    # 训练集里的所有item
    file = open(file_path)
    try:
        for line in file:
            tmp = line.split(',')
            item_str = tmp[2]
            item = int(item_str)
            item_set.add(item)
    except Exception as e:
        print(e)
    finally:
        file.close()
    return list(item_set)


# 获得每个商品的属性值的和（表示该商品在这个属性上的总体水平）
def calc_item_aspectValue(V, aspects_num):
    item_aspectValue_dic = dict()
    for item in V.keys():
        s = 0
        for k in range(aspects_num):
            s += V[item][k]
        item_aspectValue_dic[item] = s
    return item_aspectValue_dic


# 计算ICR*aspectValue并进行归一化（用于表示购买该商品的确定性）
def calc_decisiveness(item_ICR_dic, item_aspectValue_dic):
    item_decisiveness_dic = dict()
    # 所有item的decisiveness值的和，用于进行归一化
    s = 0
    # 所有item的decisiveness值的最大、最小值，用于进行归一化
    MIN = 1000
    MAX = 0
    for item in item_ICR_dic.keys():
        # 当前item的decisiveness值的和
        v = item_ICR_dic[item] * item_aspectValue_dic[item]
        item_decisiveness_dic[item] = v
        s += v
        if v < MIN:
            MIN = v
        if v > MAX:
            MAX = v
    # 归一化
    # for item in item_ICR_dic.keys():
    #     v = item_decisiveness_dic[item]
    #     item_decisiveness_dic[item] = v/s
    # 另一种归一化方式
    for item in item_ICR_dic.keys():
        v = item_decisiveness_dic[item]
        item_decisiveness_dic[item] = (v-MIN) / (MAX-MIN)
    return item_decisiveness_dic


def calc_item_prob(V, theta, aspects_num, session_item_data, dic, item_session_times_dic, session_click_stream, WIN,
                   item_decisiveness_dic, para):

    # 计算各个点击流长度下中商品为skyline object的概率
    session_idx_item_prob_dic = dict()
    for cur_data in session_item_data:
        session = cur_data[0]
        # 计算当前session各个点击流长度下中商品为skyline object的概率
        # 点击流
        full_click_stream = session_click_stream[session]
        full_click_stream_num = len(full_click_stream)
        # 点击流长度为1时商品为skyline object的概率
        key = (session, 1)
        session_idx_item_prob_dic[key] = list()
        # 第一次只有一个点击商品，其概率为1
        session_idx_item_prob_dic[key].append([full_click_stream[0], 1])
        # 点击流长度为2,3，...，cur_click_stream_num，依次计算各个点击流中商品为skyline object的概率
        # （这里的2不可变动）
        for i in range(2, full_click_stream_num+1):
            # 计算当前session点击流长度为i时，各个商品为skyline object的概率
            key = (session, i)
            session_idx_item_prob_dic[key] = list()
            # #说明# calc_item_prob_help: 初始计算方式
            # #说明# calc_item_prob_help2: 在结对比较的时候，如果一个商品在session里出现了超过两次，他的结对也是都需要出现的
            # #说明# calc_item_prob_help3: 计算商品为skyline object概率的时候，每个项目的值前面都乘以 1+exp(商品在该session里出现次数)
            if para==1:
                calc_item_prob_help(V, theta, aspects_num, full_click_stream[0:i]
                                     , session_idx_item_prob_dic[key], session, item_session_times_dic)
            elif para==2:
                calc_item_prob_help2(V, theta, aspects_num, full_click_stream[0:i]
                                    , session_idx_item_prob_dic[key], session, item_session_times_dic)
            elif para == 3:
                calc_item_prob_help3(V, theta, aspects_num, full_click_stream[0:i]
                                    , session_idx_item_prob_dic[key], session, item_session_times_dic)

    # early_predict截断
    session_item_prob_dic, session_cutoff_rate_dic, session_cutoff_length_dic, session_flag_dic = \
        early_predict(V, aspects_num, dic, session_idx_item_prob_dic, item_session_times_dic, session_click_stream, WIN,
                      item_decisiveness_dic)

    # 根据预测结果计算准确率与召回率
    recommendation33.calc_precision_and_recall(session_click_stream, session_item_data, session_cutoff_length_dic, session_flag_dic)
    return session_idx_item_prob_dic, session_item_prob_dic, session_cutoff_rate_dic, session_flag_dic


# 计算当前session已点击i个商品时，各个商品为skyline object的概率
def calc_item_prob_help(V, theta, aspects_num, cur_click_stream, cur_item_prob, session, item_session_times_dic):

    # 根据当前session的点击商品计算该session最喜欢的属性
    favorite_aspect = calc_favorite_aspect(V, aspects_num, cur_click_stream)

    # 获取非重复item
    cur_items = list()
    for item in cur_click_stream:
        # 按商品在点击流里面第一次出现的顺序放置
        if item not in cur_items:
            cur_items.append(item)

    for w in cur_items:
        if sum(V[w]) == 0:
            prob = 0.0
            cur_item_prob.append([w, prob])
        else:
            temp_product = 1
            for v in cur_items:
                if v != w:
                    for k in range(aspects_num):
                        if k == favorite_aspect:
                            if (V[w][k] + theta * V[v][k]) == 0:
                                while True:
                                    print('division error in def calc_item_prob_help ')
                            temp_product *= V[w][k] / (V[w][k] + theta * V[v][k])
                        else:
                            temp_product *= (theta * V[w][k]) / (V[v][k] + theta * V[w][k])

            # 该session中商品w为skyline object的概率
            cur_item_prob.append([w, temp_product])

    # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
    s = 0
    for i in range(len(cur_item_prob)):
        s += cur_item_prob[i][1]
    if s != 0:
        normalize(cur_item_prob)
    # 对当前session中各个商品的分数进行排序
    cur_item_prob.sort(key=lambda x: x[1], reverse=True)


# 计算当前session已点击i个商品时，各个商品为skyline object的概率
# 在结对比较的时候，如果一个商品在session里出现了超过两次，他的结对也是都需要出现的
def calc_item_prob_help2(V, theta, aspects_num, cur_click_stream, cur_item_prob, session, item_session_times_dic):

    # 根据当前session的点击商品计算该session最喜欢的属性
    favorite_aspect = calc_favorite_aspect(V, aspects_num, cur_click_stream)

    # 计算当前点击流中各个item出现的次数
    item_times_dic = dict()

    # 获取非重复item
    cur_items = list()
    for item in cur_click_stream:
        # 按商品在点击流里面第一次出现的顺序放置
        if item not in cur_items:
            cur_items.append(item)
            item_times_dic[item] = 1
        else:
            item_times_dic[item] += 1

    for w in cur_items:
        w_times = item_times_dic[w]
        if sum(V[w]) == 0:
            prob = 0.0
            cur_item_prob.append([w, prob])
        else:
            temp = 1
            for v in cur_items:
                if v != w:
                    v_times = item_times_dic[v]
                    temp2 = 1
                    for k in range(aspects_num):
                        if k == favorite_aspect:
                            if (V[w][k] + theta * V[v][k]) == 0:
                                print('division error')
                            temp2 *= V[w][k] / (V[w][k] + theta * V[v][k])
                        else:
                            temp2 *= (theta * V[w][k]) / (V[v][k] + theta * V[w][k])
                        temp2 *= w_times * v_times
                    temp *= temp2

            # 该session中商品w为skyline object的概率
            cur_item_prob.append([w, temp])

    # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
    s = 0
    for i in range(len(cur_item_prob)):
        s += cur_item_prob[i][1]
    if s != 0:
        normalize(cur_item_prob)
    # 对当前session中各个商品的分数进行排序
    cur_item_prob.sort(key=lambda x: x[1], reverse=True)


# 计算当前session已点击i个商品时，各个（不同）商品为skyline object的概率
# 计算商品为skyline object概率的时候，每个项目的值前面都乘以 1+exp(商品在该session里出现次数)
def calc_item_prob_help3(V, theta, aspects_num, cur_click_stream, cur_item_prob, session, item_session_times_dic):

    # 根据当前session的点击商品计算该session最喜欢的属性
    favorite_aspect = calc_favorite_aspect(V, aspects_num, cur_click_stream)

    # 计算当前点击流中各个item出现的次数
    item_times_dic = dict()

    # 获取非重复item
    cur_items = list()
    for item in cur_click_stream:
        # 按商品在点击流里面第一次出现的顺序放置
        if item not in cur_items:
            cur_items.append(item)
            item_times_dic[item] = 1
        else:
            item_times_dic[item] += 1

    for w in cur_items:
        w_times = item_times_dic[w]
        w_factor = 1 + math.exp(w_times)
        if sum(V[w]) == 0:
            prob = 0.0
            cur_item_prob.append([w, prob])
        else:
            temp_product = 1
            for v in cur_items:
                if v != w:
                    v_times = item_times_dic[v]
                    v_factor = 1 + math.exp(v_times)
                    for k in range(aspects_num):
                        if k == favorite_aspect:
                            if (V[w][k] + theta * V[v][k]) == 0:
                                print('division error in def calc_item_prob_help3 ')
                                exit()
                            temp_product *= (w_factor * V[w][k]) / (w_factor * V[w][k] + theta * v_factor * V[v][k])
                        else:
                            temp_product *= (theta * w_factor * V[w][k]) / (v_factor * V[v][k] + theta * w_factor * V[w][k])

            # 该session中商品w为skyline object的概率
            cur_item_prob.append([w, temp_product])

    # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
    s = 0
    for i in range(len(cur_item_prob)):
        s += cur_item_prob[i][1]
    if s != 0:
        normalize(cur_item_prob)
    # 对当前session中各个商品的分数进行排序
    cur_item_prob.sort(key=lambda x: x[1], reverse=True)


# 计算当前session已点击i个商品时，各个商品为skyline object的概率
# 加入对流行商品计算的“惩罚”（未完成）
def calc_item_prob_help4(V, theta, aspects_num, cur_click_stream, cur_item_prob, session, item_session_times_dic):

    # 根据当前session的点击商品计算该session最喜欢的属性
    favorite_aspect = calc_favorite_aspect(V, aspects_num, cur_click_stream)

    # 计算当前点击流中各个item出现的次数
    item_times_dic = dict()

    # 获取非重复item
    cur_items = list()
    for item in cur_click_stream:
        # 按商品在点击流里面第一次出现的顺序放置
        if item not in cur_items:
            cur_items.append(item)
            item_times_dic[item] = 1
        else:
            item_times_dic[item] += 1

    for w in cur_items:
        w_times = item_times_dic[w]
        w_factor = 1 + math.exp(w_times)
        if sum(V[w]) == 0:
            prob = 0.0
            cur_item_prob.append([w, prob])
        else:
            temp_product = 1
            for v in cur_items:
                if v != w:
                    v_times = item_times_dic[v]
                    v_factor = 1 + math.exp(v_times)
                    for k in range(aspects_num):
                        if k == favorite_aspect:
                            if (V[w][k] + theta * V[v][k]) == 0:
                                print('division error in def calc_item_prob_help3 ')
                                exit()
                            temp_product *= (w_factor * V[w][k]) / (w_factor * V[w][k] + theta * v_factor * V[v][k])
                        else:
                            temp_product *= (theta * w_factor * V[w][k]) / (v_factor * V[v][k] + theta * w_factor * V[w][k])

            # 该session中商品w为skyline object的概率
            cur_item_prob.append([w, temp_product])

    # 当当前session各个商品的分数不全为0时，对各个商品分数进行归一化
    s = 0
    for i in range(len(cur_item_prob)):
        s += cur_item_prob[i][1]
    if s != 0:
        normalize(cur_item_prob)
    # 对当前session中各个商品的分数进行排序
    cur_item_prob.sort(key=lambda x: x[1], reverse=True)


# 根据当前session的点击商品计算该session最喜欢的属性
def calc_favorite_aspect(V, aspects_num, click_stream):
    sum_of_vk = list()
    for k in range(aspects_num):
        s = 0
        for item in click_stream:
            s += V[item][k]
        sum_of_vk.append(s)

    max_val = max(sum_of_vk)
    favorite_aspect = sum_of_vk.index(max_val)
    return favorite_aspect


# early_predict
def early_predict(V, aspects_num, dic, session_idx_item_prob_dic, item_session_times_dic, session_click_stream, WIN,
                  item_decisiveness_dic):
    session_item_prob_dic = dict()
    session_cutoff_length_dic = dict()
    session_cutoff_rate_dic = dict()

    # 表示是否能找到截断位置
    session_flag_dic = dict()

    vote3_vote4_count = 0
    for session in session_click_stream.keys():
        # 当前session的点击流
        full_click_stream = session_click_stream[session]
        # 当前session的点击流长度
        full_click_stream_num = len(full_click_stream)

        # 提取当前session各个点击流长度下预测时排在第一位的商品
        item1_list = list()
        # 当前session各个点击流长度下各个商品对应的购买概率
        all_item_prob_list = list()
        for i in range(1, full_click_stream_num + 1):
            key = (session, i)
            # 当前session点击流长度为i时预测时排在“第一位”的商品及其概率（概率值经过归一化）
            [item1, prob1] = session_idx_item_prob_dic[key][0]
            item1_list.append(item1)
            # 当前session点击流长度为i时各个商品的购买概率
            item_prob_list = list()
            # 点击流长度为i时，该点击流中的不同商品个数
            item_num = len(session_idx_item_prob_dic[key])
            for j in range(item_num):
                [item, prob] = session_idx_item_prob_dic[key][j]
                item_prob_list.append([item, prob])
            all_item_prob_list.append(item_prob_list)

        # # 这种集成方法有点问题--效率方面
        # result = list()
        #
        # # 通过滑动窗口策略进行截断
        # cutoff_lenth, cutoff_item, flag = judge_by_sliding_window(item1_list, WIN)
        # result.append([cutoff_lenth, cutoff_item, flag])
        #
        # # 通过项目在某个点击流长度下比例大于一定阈值（设为1/2）时进行截断
        # cutoff_lenth, cutoff_item, flag = judge_by_item_percentage(item1_list, 1/2)
        # result.append([cutoff_lenth, cutoff_item, flag])
        #
        # # 通过商品购买概率比较取阈值进行截断
        # cutoff_lenth, cutoff_item, flag = judge_by_comparing_threshold(all_item_prob_list, 0.9)
        # result.append([cutoff_lenth, cutoff_item, flag])
        #
        # # 通过decisiveness取阈值进行截断（即“购买该商品的确定性”）
        # cutoff_lenth, cutoff_item, flag = judge_by_decesiveness_threshold(all_item_prob_list, item_decisiveness_dic,
        #                                                                   0.05*(1/9356))       # 改前：0.03*(1/9356)
        # result.append([cutoff_lenth, cutoff_item, flag])
        #
        # # 几种early-predict截断策略结果的集成
        # cutoff_lenth, cutoff_item, flag = ensemble2(result)

        # # 新的集成框架——利用集成方法进行截断
        cutoff_lenth, cutoff_item, flag, vote1, vote2, vote3, vote4 = new_ensemble(all_item_prob_list, item_decisiveness_dic)
        if vote3+vote4 == 2:
            vote3_vote4_count += 1

        # 通过最喜欢的属性的值大于其他属性的值的和进行截断
        # 计算当前session各个点击流下的最喜欢属性
        # favorite_aspect_list = list()
        # for i in range(1, full_click_stream_num+1):
        #     a = calc_favorite_aspect(V, aspects_num, full_click_stream[0:i])
        #     favorite_aspect_list.append(a)
        # 进行截断
        # cutoff_lenth, cutoff_item, flag = judge_by_aspect_percentage(item1_list, favorite_aspect_list, V, aspects_num)

        # 通过最喜欢的属性的值大于某一阈值时进行截断
        # cutoff_lenth, cutoff_item, flag = judge_by_v_threshold(item1_list, favorite_aspect_list, V, 1/9356)

        # 若能找到截断位置
        session_flag_dic[session] = flag

        # 输出结果
        session_item_prob_dic[session] = list()
        # 注意这里一个session只有一个预测为skyline object的商品，并人为设置其概率为1
        session_item_prob_dic[session].append([cutoff_item, 1])
        # 截断时点击流的长度
        session_cutoff_length_dic[session] = cutoff_lenth
        session_cutoff_rate_dic[session] = cutoff_lenth/full_click_stream_num

    print(vote3_vote4_count, len(dic.keys()))

    return session_item_prob_dic, session_cutoff_rate_dic, session_cutoff_length_dic, session_flag_dic


# 新的集成框架——利用集成方法进行截断
def new_ensemble(all_item_prob_list, item_decisiveness_dic):       # 输入all_item_prob_list：当前session各个点击流长度下各个商品及对应的购买概率

    # 参数
    # 滑动窗口方法相关
    WIN = 3
    # comparing_threshold方法相关
    comparing_threshold = 0.9
    # decisiveness方法相关
    decisiveness_threshold = 0.05 * (1 / 9356)      # decisiveness方法更改之前阈值为0.03 * (1 / 9356)

    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 截断时点击流的长度
    cutoff_lenth = 0
    # 完整点击流的长度
    full_click_stream_num = len(all_item_prob_list)

    # 点击流长度为1时
    item_prob_list = all_item_prob_list[0]
    item = item_prob_list[0][0]

    # 滑动窗口方法相关
    # 滑动窗口初始化
    sliding_item = item
    # “第一位”商品连续相同时的次数
    count = 1

    # item_percentage相关
    # 各个点击流长度下预测概率第一位商品的出现次数
    item_times_dic = dict()

    # 当点击流长度大于等于2时开始判断
    for i in range(1, full_click_stream_num):
        # 当前点击流长度下，各种策略判断是否进行截断的结果（初始化）
        vote1 = 0
        vote2 = 0
        vote3 = 0
        vote4 = 0

        # 当点击流长度为i+1时，此时各个商品的购买概率（当点击流长度大于等于2时开始判断）
        item_prob_list = all_item_prob_list[i]
        # 此时预测第一位商品
        item = item_prob_list[0][0]
        # 此时预测第一位商品的概率
        prob = item_prob_list[0][1]

        # 滑动窗口方法
        # 当前“第一位”商品与上一个“第一位”商品相同
        if item == sliding_item:
            count += 1
        else:       # 当前“第一位”商品与上一个“第一位”商品不同
            sliding_item = item
            count = 1
        # 截断
        if count == WIN:
            # 表示该session能找到截断位置
            vote1 = 1

        # item_percentage方法
        if item not in item_times_dic.keys():
            item_times_dic[item] = 1
        else:
            item_times_dic[item] += 1
        # 获取当前点击流长度下出现次数最多的那个“第一位商品”（key）及其次数（maxValue）
        maxValue, key = get_dic_maxValue(item_times_dic)
        # 判断该“出现最多次”item的计数值是否超过当前长度的1/2
        if maxValue/(i+1) > 1/2:
            vote2 = 1
            # 注意此处是“截断处”预测第一位商品而非“出现最多次”item
            # cutoff_item = item

        # comparing_threshold方法
        if prob >= comparing_threshold:
            vote3 = 1

        # decisiveness方法
        # 获取该商品的decisiveness值
        decisiveness = item_decisiveness_dic[item]
        if decisiveness*math.exp(1/(i+1)) >= decisiveness_threshold:    # 改前：prob*decisiveness >= decisiveness_threshold
            vote4 = 1

        # 集成方法截断
        if vote1 + vote2 + vote3 + vote4 >= 2:
            cutoff_item = item
            cutoff_lenth = i+1
            flag = 1
            print(vote1, vote2, vote3, vote4, "cutoff_lenth/full_click_stream_num:", cutoff_lenth, "/", full_click_stream_num,
                  "=", cutoff_lenth/full_click_stream_num)
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = full_click_stream_num
        cutoff_item = all_item_prob_list[-1][0][0]
    return cutoff_lenth, cutoff_item, flag, vote1, vote2, vote3, vote4


# 几种early-predict截断策略结果的集成
def ensemble(result):
    # 当前截断策略结果
    cur_cutoff_lenth = result[0][0]
    cur_cutoff_item = result[0][1]
    cur_flag = result[0][2]
    # 集成结果初始化（若几种截断策略flag均为0，则它们的cutoff_lenth和cutoff_item必相同；若有flag为1的，则该集成方法能够达到目的。）
    cutoff_lenth = cur_cutoff_lenth
    cutoff_item = cur_cutoff_item
    flag = cur_flag
    for i in range(1, len(result)):
        # 当前截断策略结果
        cur_cutoff_lenth = result[i][0]
        cur_cutoff_item = result[i][1]
        cur_flag = result[i][2]
        if cur_flag==1 and cur_cutoff_lenth < cutoff_lenth:     # 注意：flag=1的cutoff_lenth必定比flag=0的cutoff_lenth小
            cutoff_lenth = cur_cutoff_lenth
            cutoff_item = cur_cutoff_item
            flag = cur_flag
    return cutoff_lenth, cutoff_item, flag


def ensemble2(result):
    # 记录几种策略结果中flag=1的结果的item对应出现次数
    item_times_dic = dict()
    for i in range(len(result)):
        # 当前截断策略结果
        cur_cutoff_item = result[i][1]
        cur_flag = result[i][2]
        # 只计数flag=1的结果的item
        if cur_flag == 1 or cur_flag == 0:
            if cur_cutoff_item not in item_times_dic.keys():
                item_times_dic[cur_cutoff_item] = 1
            else:
                item_times_dic[cur_cutoff_item] += 1
    # 若几种截断策略结果中都没有flag==1的(若几种截断策略flag均为0，则它们的cutoff_lenth和cutoff_item必相同)
    if len(item_times_dic.keys()) == 0:
        # 当前截断策略结果
        cur_cutoff_lenth = result[0][0]
        cur_cutoff_item = result[0][1]
        cur_flag = result[0][2]
        # 集成结果初始化
        cutoff_lenth = cur_cutoff_lenth
        cutoff_item = cur_cutoff_item
        flag = cur_flag
    else:
        # 若有flag为1的，则该集成方法能够达到目的。
        sorted_item_times_list = sorted(item_times_dic.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_item_times_list)
        # 出现次数最多的item有哪些（防止有多个item出现次数相同）
        item_candidates = set()
        # 出现次数最多的商品
        first_item = sorted_item_times_list[0][0]
        first_itemTimes = sorted_item_times_list[0][1]
        item_candidates.add(first_item)
        for i in range(1, len(sorted_item_times_list)):
            item = sorted_item_times_list[i][0]
            itemTimes = sorted_item_times_list[i][1]
            if itemTimes == first_itemTimes:
                item_candidates.add(item)
        # 找出这些item candidates中cutoff_lenth最小的
        # cutoff_lenth = result[0][0]     # 若result[0]的flag=0，没问题；若result[0]的flag=1，这个设置有问题
        cutoff_lenth = 10000       # 设置为一个认为是无穷大的数
        for i in range(len(result)):
            # 当前截断策略结果
            cur_cutoff_lenth = result[i][0]
            cur_cutoff_item = result[i][1]
            cur_flag = result[i][2]
            # 若存在flag=0，cutoff_item也在item_candidates的；那么必也存在flag=1，cutoff_item与其相同的情况，因此以下处理没有问题。
            if cur_cutoff_item in item_candidates and cur_cutoff_lenth < cutoff_lenth:
                cutoff_lenth = cur_cutoff_lenth
                cutoff_item = cur_cutoff_item
                flag = cur_flag

    return cutoff_lenth, cutoff_item, flag


# 通过一个固定长度的滑动窗口，当商品值连续WIN次不再变化时截断，返回该商品及截断时的长度；若没有满足该窗口条件的，返回最后一个商品及该商品序列的长度。
def judge_by_sliding_window(item_sequence, WIN):

    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 商品序列的长度
    sequence_length = len(item_sequence)
    # “第一位”商品连续相同时的次数
    count = 1
    # 截断时点击流的长度
    cutoff_lenth = 0

    # 计算当前session每个商品在每个浏览次数下“第一位”商品出现的次数
    for i in range(sequence_length):
        # 此时“第一位”的商品及其概率
        item = item_sequence[i]
        # 判断“第一位”商品连续相同时的次数
        if i == 0:
            cutoff_item = item
            continue
        else:
            # 当前“第一位”商品与上一个“第一位”商品相同
            if item == cutoff_item:
                count += 1
            # 当前“第一位”商品与上一个“第一位”商品不同
            else:
                cutoff_item = item
                count = 1
        # 截断
        if count == WIN:
            # 表示该session能找到截断位置
            flag = 1
            cutoff_lenth = i+1
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = sequence_length

    return cutoff_lenth, cutoff_item, flag


# 设置一个阈值（这里为：1/2），当序列中某个项出现的次数“大于”该值时截断。
def judge_by_item_percentage(item_sequence, threshold):

    # 商品序列的长度
    sequence_length = len(item_sequence)
    # 截断时点击流的长度
    cutoff_lenth = 0
    # 各个点击流长度下预测概率第一位商品的集合
    streaming_item_set = set()
    # 各个点击流长度下预测概率第一位商品的出现次数
    item_times_dic = dict()
    # 表示是否能通过该方法找到截断位置
    flag = 0
    for i in range(sequence_length):
        # 此时“第一位”的商品及其概率
        item = item_sequence[i]
        if item not in streaming_item_set:
            streaming_item_set.add(item)
            item_times_dic[item] = 1
        else:
            item_times_dic[item] += 1

        # 从点击流长度为3（包括3）以后开始判断
        if i >= 2:
            # if len(set(streaming_item_set)) == 1:
            #     continue
            # 获取当前点击流长度下出现次数最多的那个“第一位商品”（key）及其次数（maxValue）
            maxValue, key = get_dic_maxValue(item_times_dic)
            # 判断该“出现最多次”item的计数值是否超过当前长度的1/2
            if maxValue/(i+1) > threshold:
                cutoff_lenth = i+1
                # 注意此处是“截断处”预测第一位商品而非“出现最多次”item
                cutoff_item = item
                flag = 1
                break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = sequence_length
        cutoff_item = item_sequence[sequence_length-1]

    return cutoff_lenth, cutoff_item, flag


# 通过商品购买概率比较取阈值进行截断（即当前session若在某个点击流长度下，预测购买概率最高的商品的概率值大于某一阈值）
def judge_by_comparing_threshold(all_item_prob_list, threshold):
    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 完整点击流的长度
    full_click_stream_num = len(all_item_prob_list)
    # 截断时点击流的长度
    cutoff_lenth = 0

    # 当点击流长度大于等于2时开始判断
    for i in range(1, full_click_stream_num):
        # 当点击流长度为i+1时，此时各个商品的购买概率（当点击流长度大于等于2时开始判断）
        item_prob_list = all_item_prob_list[i]
        # 此时预测第一位商品
        item = item_prob_list[0][0]
        # 此时预测第一位商品的概率
        prob = item_prob_list[0][1]
        if prob >= threshold:
            cutoff_lenth = i+1
            cutoff_item = item
            flag = 1
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = full_click_stream_num
        cutoff_item = all_item_prob_list[-1][0][0]
    return cutoff_lenth, cutoff_item, flag


# 通过decisiveness取阈值进行截断
def judge_by_decesiveness_threshold(all_item_prob_list, item_decisiveness_dic, threshold):
    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 完整点击流的长度
    full_click_stream_num = len(all_item_prob_list)
    # 截断时点击流的长度
    cutoff_lenth = 0

    # 当点击流长度大于等于2时开始判断
    for i in range(1, full_click_stream_num):
        # 当点击流长度为i+1，此时各个商品的购买概率
        item_prob_list = all_item_prob_list[i]
        # 此时预测第一位商品
        item = item_prob_list[0][0]
        # 此时预测第一位商品的概率
        prob = item_prob_list[0][1]
        # 获取该商品的decisiveness值
        decisiveness = item_decisiveness_dic[item]
        if decisiveness * math.exp(1/(i+1)) >= threshold:       # 改前：prob*decisiveness >= threshold
            cutoff_lenth = i+1
            cutoff_item = item
            flag = 1
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = full_click_stream_num
        cutoff_item = all_item_prob_list[-1][0][0]
    return cutoff_lenth, cutoff_item, flag


# 通过最喜欢的属性的值大于其他属性的值的和进行截断
def judge_by_aspect_percentage(item_sequence, favorite_aspect_list, V, aspects_num):
    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 商品序列的长度
    sequence_length = len(item_sequence)
    # 截断时点击流的长度
    cutoff_lenth = 0
    for i in range(sequence_length):
        # 此时“第一位”的商品及其概率
        item = item_sequence[i]
        # 此时的最喜欢属性
        a = favorite_aspect_list[i]
        # 该商品在最喜欢属性上的值
        favorite_v = V[item][a]
        # 该商品在其他属性上的值的和
        other_v_sum = 0
        for k in range(aspects_num):
            if k != a:
                other_v_sum += V[item][k]
        if favorite_v > other_v_sum:
            cutoff_lenth = i+1
            cutoff_item = item
            flag = 1
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = sequence_length
        cutoff_item = item_sequence[sequence_length - 1]
    return cutoff_lenth, cutoff_item, flag


# 通过最喜欢的属性的值大于某一阈值
def judge_by_v_threshold(item_sequence, favorite_aspect_list, V, threshold):
    # 标记该session是否能找到截断位置。是为1，不是为0。
    flag = 0
    # 商品序列的长度
    sequence_length = len(item_sequence)
    # 截断时点击流的长度
    cutoff_lenth = 0
    for i in range(sequence_length):
        # 此时“第一位”的商品及其概率
        item = item_sequence[i]
        # 此时的最喜欢属性
        a = favorite_aspect_list[i]
        # 该商品在最喜欢属性上的值
        favorite_v = V[item][a]
        if favorite_v > threshold:
            cutoff_lenth = i + 1
            cutoff_item = item
            flag = 1
            break

    # 如果未能满足截断条件，令截断长度为点击流的长度（此时的预测购买商品为"完整长度"点击流预测排在“第一位”的商品）
    if cutoff_lenth == 0:
        cutoff_lenth = sequence_length
        cutoff_item = item_sequence[sequence_length-1]
    return cutoff_lenth, cutoff_item, flag


# 获取一个dict中最大的值及其对应的key
def get_dic_maxValue(dic):
    maxValue = 0
    for k in dic.keys():
        if dic[k] > maxValue:
            maxValue = dic[k]
            key = k
    return maxValue, key


# 分别计算当前session各个点击流的最喜欢属性
def calc_all_favorite_aspects(V, aspects_num, full_click_stream):
    all_favorite_aspects = list()
    cur_click_stream_num = len(full_click_stream)
    # 分别计算各个点击流长度下商品（各个属性值的和）
    for i in range(1, cur_click_stream_num+1):

        # 根据当前session的点击商品计算该session最喜欢的属性
        favorite_aspect = calc_favorite_aspect(V, aspects_num, full_click_stream[0:i])
        all_favorite_aspects.append(favorite_aspect)

    return all_favorite_aspects


# 计算序列的indecisiveness（加入失败，因为发现favorite aspect“基本不变”。）
def calc_all_indecisiveness(all_favorite_aspects):
    all_indecisiveness = list()

    cur_click_stream_num = len(all_favorite_aspects)
    # 分别计算各个点击流长度下商品（各个属性值的和）
    for i in range(1, cur_click_stream_num+1):
        # 根据当前session的点击商品计算该session最喜欢的属性
        indecisiveness = calc_indecisiveness(all_favorite_aspects[0:i])
        all_indecisiveness.append(indecisiveness)

    return all_indecisiveness


# calculate earliness
def print_earliness(session_cutoff_rate_dic):
    all_cutoff_rate = 0
    session_num = len(session_cutoff_rate_dic)
    for session in session_cutoff_rate_dic.keys():
        cur_cutoff_rate = session_cutoff_rate_dic[session]
        all_cutoff_rate += cur_cutoff_rate
    avg_cutoff_rate = all_cutoff_rate/session_num
    print('earliness(the smaller, the earlier): ', ('%.4f' % avg_cutoff_rate))
    return avg_cutoff_rate


# 形如[[1, 0.3], [2, 0.3]]， 归一化为[[1, 0.5], [2, 0.5]]
# 用于归一化商品为skyline object的概率
def normalize(ls):
    s = 0
    for i in range(len(ls)):
        s += ls[i][1]
    for i in range(len(ls)):
        ls[i][1] = ls[i][1]/s


def evaluate(session_item_data, session_item_prob_dic):
    p1 = calc_precision_at_1(session_item_data, session_item_prob_dic)
    # p2 = calc_precision_at_2(session_item_data, session_item_prob_dic)
    # MRR = calc_MRR(session_item_data, session_item_prob_dic)
    print('p1: ' + ('%.4f' % p1))
    # print('p2: ' + ('%.4f' % p2))
    # print('MRR: ' + ('%.4f' % MRR))
    return p1


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
    MRR /= len(session_item_data)
    return MRR


# 根据groundtruth探索规律
# 列出每个session每个点击流预测排在“第一位”的所有skyline object（与recommendation33比较的不同1）
# 这里每个session每个点击流预测排在“第一位”的所有商品“没有去除重复的情况”（与recommendation33比较的不同2）
def proving_session_item1s(dic, session_idx_item_prob_dic, session_click_stream):
    session_item1s_dic = dict()
    for session in dic.keys():
        # 当前session的点击流
        cur_click_stream = session_click_stream[session]
        # 当前session的点击流长度
        cur_click_stream_num = len(cur_click_stream)

        # 当前session每个点击流计算中排在第一位的商品
        cur_item1_list = list()
        for i in range(1, cur_click_stream_num + 1):
            key = (session, i)
            # 此时“第一位”的商品及其概率
            [item1, prob1] = session_idx_item_prob_dic[key][0]
            cur_item1_list.append(item1)

        session_item1s_dic[session] = cur_item1_list

    return session_item1s_dic


# 根据groundtruth探索规律
# 每个session每次浏览预测排在第一位的商品中"最早出现购买商品的位置"
def print_buyItemFirstRank1Idx(session_item_data, session_item1s_dic):
    session_buyItemFirstRank1Idx_dic = dict()
    for cur_data in session_item_data:
        # groundtruth
        session = cur_data[0]
        cur_buy_items = cur_data[1]
        # predict
        # 当前session各个点击流预测排在第一位的商品
        cur_item1s = session_item1s_dic[session]
        cur_item1s_num = len(cur_item1s)
        # 初始化为0，防止存在当前session所有浏览商品数目下排在第一位的商品都不是购买商品的情况
        session_buyItemFirstRank1Idx_dic[session] = 0
        for i in range(cur_item1s_num):
            # 判断当前session当前点击流预测排在第一位的商品是否为购买商品
            if cur_item1s[i] in cur_buy_items:
                session_buyItemFirstRank1Idx_dic[session] = i+1
                break

    print(session_buyItemFirstRank1Idx_dic)


if __name__ == '__main__':
    pass

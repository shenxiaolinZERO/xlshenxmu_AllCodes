#!/usr/bin/env python
# -*- coding:utf-8 -*-

import print_to_file as p2f
import time
import datetime
import feature4


# 时间类特征的提取


class Feature5:

    @staticmethod
    def go(dataset_dir, feature_dir):

        yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
        groundtruth_dir = dataset_dir + r'\test'

        click_file_path = yoochoose_selected_dir + r'\yoochoose-clicks-selected.dat'
        buy_file_path = yoochoose_selected_dir + r'\yoochoose-buys-selected.dat'
        test_file_path = yoochoose_selected_dir + r'\yoochoose-test-selected.dat'
        groundtruth_file_path = groundtruth_dir + r'\session_item.txt'

        # data_para_list为1和2分别表示训练数据集和测试数据集的特征输出
        data_para_list = [1, 2, 2]
        # print_para_list为1和0分别表示是、否输出数据的session ID, item ID信息
        print_para_list = [0, 1, 0]

        # 获取训练数据和测试数据中所有的item
        # item_list = get_item_list(click_file_path, test_file_path)

        for i in range(len(data_para_list)):
            data_para = data_para_list[i]
            print_para = print_para_list[i]

            if data_para == 1:
                write_file_path = feature_dir + r'\click-buy-train.arff'
            else:
                if print_para == 1:
                    write_file_path = feature_dir + r'\click-buy-test-BR.txt'
                else:
                    write_file_path = feature_dir + r'\click-buy-test.arff'

            # if data_para == 2:
            #     click_file_path = test_file_path
            #     buy_file_path = groundtruth_file_path
            Feature5.print_feature(click_file_path, buy_file_path, test_file_path, groundtruth_file_path, write_file_path, data_para, print_para)

    @staticmethod
    def print_feature(click_file_path, buy_file_path, test_file_path, groundtruth_file_path, write_file_path, data_para, print_para):

        # 提取训练数据或测试数据的特征
        if data_para == 1:
            # 训练数据
            # 获取month, week, timePoint特征。
            session_month_dic, session_week_dic, session_timePoint_dic = get_session_pointFeature(click_file_path)
            # 该session总共进行了多少次点击。（多次点击相同item算作多次点击。）（即该session在数据集中有多少“行”。）
            session_clickTime_dic = get_session_clickTime(click_file_path)
            # 持续时间（单位：秒）。该session从开始到结束总共持续了多长时间。
            session_lastTime_dic = get_session_lastTime(click_file_path)
            # 用来求每个商品在各个session的出现次数——{(item, session):times}
            item_session_times_dic = feature4.get_item_session_times(click_file_path)
        else:
            # 测试数据
            # 获取month, week, timePoint特征。
            session_month_dic, session_week_dic, session_timePoint_dic = get_session_pointFeature(test_file_path)
            # 该session总共进行了多少次点击。（多次点击相同item算作多次点击。）（即该session在数据集中有多少“行”。）
            session_clickTime_dic = get_session_clickTime(test_file_path)
            # 持续时间（单位：秒）。该session从开始到结束总共持续了多长时间。
            session_lastTime_dic = get_session_lastTime(test_file_path)
            # 用来求每个商品在各个session的出现次数——{(item, session):times}
            item_session_times_dic = feature4.get_item_session_times(test_file_path)
        # 用于后面label的提取
        if data_para == 1:
            # 训练数据的购买数据：每个商品被哪些session购买
            buys_item_session_dic = feature4.get_item_clicked(buy_file_path)
        else:
            # 测试数据的购买数据：每个商品被哪些session购买
            buys_item_session_dic = feature4.get_test_item_bought(groundtruth_file_path)
        # 将发生了购买行为的(item,session)放入groundtruth_buys中
        groundtruth_buys = list()
        extract_groundtruth(buys_item_session_dic, groundtruth_buys)
        # 输出数据头
        print_head(write_file_path, print_para)
        session_list = list()
        # item_set表示当前session的所有item
        item_set = set()
        data_list = list()
        if data_para == 1:
            file = open(click_file_path)
        else:
            file = open(test_file_path)
        try:
            for line in file:
                tmp = line.split(',')
                session_str = tmp[0]
                session = int(session_str)
                item_str = tmp[2]
                item = int(item_str)
                # 第一个session
                if len(session_list) == 0:
                    session_list.append(session)
                # 来了一个新的session
                elif session != session_list[-1]:
                    session_list.append(session)
                    item_set.clear()
                if item not in item_set:
                    # feature
                    month = session_month_dic[session]
                    week = session_week_dic[session]
                    timePoint = session_timePoint_dic[session]
                    clickTime = session_clickTime_dic[session]
                    lastTime = session_lastTime_dic[session]
                    # feature, 每个商品在当前session的出现次数
                    item_session_times = item_session_times_dic[(item, session)]
                    label = 0
                    if (item, session) in groundtruth_buys:
                        label = 1
                    if print_para == 1:
                        data = [session, item]
                    else:
                        data = []
                    # 选择使用哪些特征
                    data.append(month)
                    data.append(week)
                    data.append(timePoint)
                    data.append(clickTime)
                    data.append(lastTime)
                    data.append(item_session_times)
                    # label
                    data.append(label)
                    # all sample feature
                    data_list.append(data)
                    item_set.add(item)
        except Exception as e:
            print(e)
        finally:
            file.close()
        p2f.print_lists_to_file(data_list, write_file_path)


# 获取month, week, timePoint特征。
def get_session_pointFeature(file_path):
    session_month_dic = dict()
    session_week_dic = dict()
    session_timePoint_dic = dict()
    session_lists = list()
    file = open(file_path)
    try:
        for line in file:
            tmp = line.split(',')
            session_str = tmp[0]
            session = int(session_str)
            time_str = tmp[1]
            month = int(time_str[5])*10 + int(time_str[6])
            week = calc_week(time_str[0:19])
            timePoint = int(time_str[11])*10 + int(time_str[12])
            # 第一个session
            if len(session_lists) == 0:
                session_lists.append(session)
                session_month_dic[session] = month
                session_week_dic[session] = week
                session_timePoint_dic[session] = timePoint
            # 来了一个新的session
            elif session != session_lists[-1]:
                session_month_dic[session] = month
                session_week_dic[session] = week
                session_timePoint_dic[session] = timePoint

                session_lists.append(session)

    except Exception as e:
        print(e)
    finally:
        file.close()
    return session_month_dic, session_week_dic, session_timePoint_dic


# 该session总共进行了多少次点击。（多次点击相同item算作多次点击。）（即该session在数据集中有多少“行”。）
def get_session_clickTime(file_path):
    session_clickTime_dic = dict()
    session_lists = list()
    clickTime = 0
    file = open(file_path)
    try:
        for line in file:
            tmp = line.split(',')
            session_str = tmp[0]
            session = int(session_str)

            # 若session_list为空
            if len(session_lists) == 0:
                session_lists.append(session)
            # 来了一个新的session
            elif session != session_lists[-1]:
                pre_session = session_lists[-1]
                session_clickTime_dic[pre_session] = clickTime

                session_lists.append(session)
                clickTime = 0
            # 还是原来的session；或者来了一个新的session（此时clickTime已先重置）
            clickTime += 1
        # 最后一个session
        session_clickTime_dic[session] = clickTime
    except Exception as e:
        print(e)
    finally:
        file.close()
    return session_clickTime_dic


# 根据时间字符串如：2014-04-06T12:18:36.915Z或2014-04-06T12:18:36或2014-04-06，计算该日期属于星期几
def calc_week(time1):

    time1 = time.strptime(time1, "%Y-%m-%dT%H:%M:%S")
    return (time.strftime("%w",time1))

# 获取lastTime特征——该session从开始到结束总共持续了多长时间（单位：秒）。
# imported by rlso/statistic.py
def get_session_lastTime(file_path):
    session_lastTime_dic = dict()
    session_lists = list()
    file = open(file_path)
    try:
        for line in file:
            tmp = line.split(',')
            session_str = tmp[0]
            session = int(session_str)
            time_str = tmp[1]
            # 取出其中包含时间的部分
            time1_str = time_str[0:19]
            # 第一个session
            if len(session_lists) == 0:
                session_lists.append(session)
                start_time_str = time1_str
            # 来了一个新的session
            elif session != session_lists[-1]:
                # 结算旧的session
                pre_session = session_lists[-1]
                lastTime = calcTime(start_time_str, end_time_str)
                session_lastTime_dic[pre_session] = lastTime
                # 新session数据
                session_lists.append(session)
                start_time_str = time1_str

            # 还是原来的session；或者来了一个新的session
            end_time_str = time1_str
        # 结算最后一个session
        lastTime = calcTime(start_time_str, end_time_str)
        session_lastTime_dic[session] = lastTime
    except Exception as e:
        print(e)
    finally:
        file.close()
    return session_lastTime_dic


# 计算两个格式形如“2014-04-06T21:19:51”的时间字符串的时间差
def calcTime(time1, time2):
    time1 = time.strptime(time1, "%Y-%m-%dT%H:%M:%S")
    time2 = time.strptime(time2, "%Y-%m-%dT%H:%M:%S")
    time1 = datetime.datetime(time1[0], time1[1], time1[2], time1[3], time1[4], time1[5])
    time2 = datetime.datetime(time2[0], time2[1], time2[2], time2[3], time2[4], time2[5])
    return (time2-time1).seconds


# (item_session_dic:训练/测试数据每个商品被哪些session购买) 从item_session_dic中提取发生购买行为的(item,session)
def extract_groundtruth(item_session_dic, groundtruth_buys):
    for item in item_session_dic.keys():
        for session in item_session_dic[item]:
            cur_buy = (item, session)
            groundtruth_buys.append(cur_buy)


def print_head(file_path, print_para):
    f = open(file_path, 'w')
    try:
        f.write('@relation data' + '\n')
        if print_para == 1:
            f.write('@attribute sessionID integer' + '\n')
            f.write('@attribute itemID integer' + '\n')
        f.write('@attribute month integer' + '\n')
        f.write('@attribute week integer' + '\n')
        f.write('@attribute timePoint integer' + '\n')
        f.write('@attribute clickTime integer' + '\n')
        f.write('@attribute lastTime integer' + '\n')
        f.write('@attribute itemClickTime integer' + '\n')
        f.write('@attribute class {1, 0}' + '\n')
        f.write('@data' + '\n')
    except Exception as e:
        print(e)
    finally:
        f.close()


if __name__ == '__main__':
    # setting
    dataset_para = 'sampling@0.01@partition'
    dataset_dir = r'E:\ranking aggregation\dataset\yoochoose\Full' + '\\' + dataset_para
    # 生成特征文件的路径
    feature_dir = r'E:\recsyschallenge2015\mycode\result-data'

    Feature5.go(dataset_dir, feature_dir)

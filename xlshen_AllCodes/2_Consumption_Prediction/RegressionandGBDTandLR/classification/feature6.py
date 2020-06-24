#!/usr/bin/env python
# -*- coding:utf-8 -*-
import print_to_file as p2f
import feature4
import feature5

# 提取时间类特征+新特征


class Feature6:

    @staticmethod
    def go(dataset_dir, feature_dir):

        yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
        groundtruth_dir = dataset_dir + r'\test'

        click_file_path = yoochoose_selected_dir + r'\yoochoose-clicks-selected.dat'
        buy_file_path = yoochoose_selected_dir + r'\yoochoose-buys-selected.dat'
        test_file_path = yoochoose_selected_dir + r'\yoochoose-test-selected.dat'
        groundtruth_file_path = groundtruth_dir + r'\session_item.txt'

        # data_para_list为1和2分别表示训练数据集和测试数据集的特征输出
        #zero：so对应于click-buy-train.arff，click-buy-test-BR.txt，click-buy-test.arff
        data_para_list = [1, 2, 2]
        # print_para_list为1和0分别表示是、否输出数据的session ID, item ID信息
        print_para_list = [0, 1, 0]

        # 获取训练数据和测试数据中所有的item（用于提取item ICR）
        item_list = feature4.get_item_list(click_file_path, test_file_path)

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
            #       click_file_path = test_file_path
            #       buy_file_path = groundtruth_file_path
            Feature6.print_feature(click_file_path, buy_file_path, test_file_path, groundtruth_file_path, write_file_path,
                                   data_para, print_para, item_list)

    @staticmethod
    def print_feature(click_file_path, buy_file_path, test_file_path, groundtruth_file_path, write_file_path,
                      data_para, print_para, item_list):

        # 提取训练数据或测试数据的特征
        if data_para == 1:
            # 训练数据
            # 获取month, week, timePoint特征。
            #session_month_dic, session_week_dic, session_timePoint_dic = feature5.get_session_pointFeature(click_file_path)
            # 该session总共进行了多少次点击。（多次点击相同item算作多次点击。）（即该session在数据集中有多少“行”。）
            session_clickTime_dic = feature5.get_session_clickTime(click_file_path)
            # 持续时间（单位：秒）。该session从开始到结束总共持续了多长时间。
            #session_lastTime_dic = feature5.get_session_lastTime(click_file_path)
            # 用来求每个商品在各个session的出现次数——{(item, session):times}
            item_session_times_dic = feature4.get_item_session_times(click_file_path)
            # 用来求每个session长度（不同item，不是按照click算）——{session:item list length}
            session_len_dic = feature4.get_session_len(click_file_path)
        else:
            # 测试数据
            # 获取month, week, timePoint特征。
            #session_month_dic, session_week_dic, session_timePoint_dic = feature5.get_session_pointFeature(test_file_path)
            # 该session总共进行了多少次点击。（多次点击相同item算作多次点击。）（即该session在数据集中有多少“行”。）
            session_clickTime_dic = feature5.get_session_clickTime(test_file_path)
            # 持续时间（单位：秒）。该session从开始到结束总共持续了多长时间。
            #session_lastTime_dic = feature5.get_session_lastTime(test_file_path)
            # 用来求每个商品在各个session的出现次数——{(item, session):times}
            item_session_times_dic = feature4.get_item_session_times(test_file_path)
            # 用来求每个session长度（不同item，不是按照click算）——{session:item list length}
            session_len_dic = feature4.get_session_len(test_file_path)
        # 接下来的两个特征都是从训练数据提取得到
        # 用来求每个商品被哪些session点击——{item:session list clicked by}——借此可求出：每个商品的总出现session次数
        item_session_dic = feature4.get_item_clicked(click_file_path)
        # Item conversion rate——{item:ICR}
        item_ICR_dic = feature4.get_item_ICR(click_file_path, buy_file_path, item_list)
        # 用于后面label的提取
        if data_para == 1:
            # 训练数据的购买数据：每个商品被哪些session购买
            buys_item_session_dic = feature4.get_item_clicked(buy_file_path)
        else:
            # 测试数据的购买数据：每个商品被哪些session购买
            buys_item_session_dic = feature4.get_test_item_bought(groundtruth_file_path)
        # 将发生了购买行为的(item,session)放入groundtruth_buys中（zero：wai
        groundtruth_buys = list()
        feature5.extract_groundtruth(buys_item_session_dic, groundtruth_buys)
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
                    # time feature
                    #month = session_month_dic[session]
                    #week = session_week_dic[session]
                    #timePoint = session_timePoint_dic[session]
                    clickTime = session_clickTime_dic[session]
                    #lastTime = session_lastTime_dic[session]
                    # feature, 每个商品在当前session的出现次数
                    item_session_times = item_session_times_dic[(item, session)]
                    # new feature
                    session_len = session_len_dic[session]
                    item_all_session_len = len(item_session_dic[item])
                    item_ICR = item_ICR_dic[item]
                    # label
                    label = 0
                    if (item, session) in groundtruth_buys:
                        label = 1
                    if print_para == 1:
                        data = [session, item]
                    else:
                        data = []
                    # 选择使用哪些特征
                    #data.append(month)
                    #data.append(week)
                    #data.append(timePoint)
                    data.append(clickTime)
                    #data.append(lastTime)
                    data.append(item_session_times)
                    # new feature
                    data.append(item_all_session_len)
                    data.append(session_len)
                    data.append(item_ICR)
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


def print_head(file_path, print_para):
    f = open(file_path, 'w')
    try:
        f.write('@relation data' + '\n')
        if print_para == 1:
            f.write('@attribute sessionID integer' + '\n')
            f.write('@attribute itemID integer' + '\n')
        #f.write('@attribute month integer' + '\n')
        #f.write('@attribute week integer' + '\n')
        #f.write('@attribute timePoint integer' + '\n')
        f.write('@attribute clickTime integer' + '\n')
        #f.write('@attribute lastTime integer' + '\n')
        f.write('@attribute itemClickTime integer' + '\n')
        # 新特征（有一个和时间类特征相同）
        f.write('@attribute item_all_session_len integer' + '\n')
        f.write('@attribute session_len integer' + '\n')
        f.write('@attribute item_ICR integer' + '\n')
        # label
        f.write('@attribute class {1, 0}' + '\n')
        f.write('@data' + '\n')
    except Exception as e:
        print(e)
    finally:
        f.close()

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import os
from matplotlib import pyplot
from input import Input
import print_to_file as p2f
from rlso5 import RLSO5
import read_from_file as rff
from recommendation11 import Recommendation11
from recommendation44 import Recommendation44
import real_data
import feature4                              # 位于classification文件夹中
from preprocess11 import Preprocess11
import calcCorrelation
import csv
from recommendation22_aggregate import Recommendation22_aggregate


def TestRLSO():

    # # 模型训练输入例子
    # # 用户、session数据——判断session属于哪个用户
    # user_sessions_data = [[100, 101, 102],]
    # # 每个session购买的商品与点击不购买的商品，按照在数据集中出现的顺序放置
    # session_item_data = [[100, [10, 11], [12, 13]],
    #                      [101, [11, 12], [10, 14]],
    #                      [102, [10, 13, 14], [11, ]]]
    # # 每个商品被哪些session购买以及被哪些session点击但不购买（item_session_data由session_item_data决定）
    # item_session_data = [[10, [100, 102], [101, ]],
    #                      [11, [100, 101], (102, ]],
    #                      [12, [101, ], [100, ]],
    #                      [13, [102, ], [100, ]],
    #                      [14, [102, ], [101, ]]]
    # # parameter：the number of aspects(That's K)
    # aspects_num = 5

    # K
    aspects_num = 5

    # \result\yoochoose\Full\D1_partition\sampling@x@2@partition\train中likelihood.txt，有0-199，
    # \result\yoochoose\Full\D2_partition\sampling@x@2@partition\train 中likelihood.txt，有0-149，
    # 模型训练迭代次数(大约迭代完成次数：D1_partition:200,D2_partition:150,D3_partition:100,D4_partition:100,D5_partition:100,D6_partition:50)
    ITERATION =50 # ？？？？？？？

    # 当前数据样本路径
    main_dir = r"I:\Papers\consumer\codeandpaper"

    # 模型参数及实验结果输出路径
    out_file_dir = r"I:\Papers\consumer\codeandpaper\code\result\yoochoose\Full"

    # 当前所有已设计的aggregate方法的数目（当改变了aggregate方法时才需设置）
    aggregate_num = 9    ##### aggregate方法的数目 ？？？？？

    # 选择使用哪部分的数据集——取决于session中的购买商品数据
    #part_para_list = ['D1_partition', 'D2_partition', 'D3_partition', 'D4_partition', 'D5_partition', 'D6_partition']
    part_para_list = ['D1_partition']
    # 最终实验数据集(Zero:使用的数据集，不是50个都用，每个【D1-D6】选了其中的10个)
    selection_index = [0, # 类似于数组的索引从0开始，但是我们不用它
                       [2,8,18,19,22,28,36,40,44,49],
                       [3,7,12,18,21,35,37,44,46,49],
                       [1,6,8,9,11,24,33,45,47,50],
                       [3,4,5,18,20,25,29,39,45,49],
                       [18,19,25,28,34,38,39,40,45,47],
                       [5,6,9,12,24,28,33,39,40,49]]

    for part_para in part_para_list:
        # 实验结果输出表格初始化
        init_flag = 0

        # 当前数据集所属数据类型，决定了计算precision@N时N的大小
        part_num = int(part_para[1])
        # 当前数据集选择用于实验的数据编号
        selection = selection_index[part_num]

        for i in selection:
            number = i
            print("part_para:", part_para, ",   number:", number)
            dataset_para = "sampling@x" + '@' + str(number) + '@partition'
            dataset_dir = main_dir + r"\Full" + "\\" + part_para + "\\" + dataset_para
            #即为：\Full\D1_partition\sampling@x@1@partition

            # 原始yoochoose数据路径,Full文件夹里的D1,D2……
            yoochoose_data_dir = main_dir + r"\Full"
            # 当前数据样本的训练数据路径, \Full\D1_partition\sampling@x@1@partition \train
            train_file_dir = dataset_dir + r"\train"
            # 当前数据样本的测试数据路径
            test_data_dir = dataset_dir + r"\test"
            # 当前训练数据和测试数据的点击数据文件
            yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
            if not os.path.exists(yoochoose_selected_dir):
                os.makedirs(yoochoose_selected_dir)

            # 计算训练数据的item ICR时用到 （ICR：商品购买转化率，就是购买该商品了的人除以看了该商品的人）
            # 此ICR 原始模型不用用到，策略用到的

            click_file_path = yoochoose_selected_dir + r'\yoochoose-clicks-selected.dat'
            buy_file_path = yoochoose_selected_dir + r'\yoochoose-buys-selected.dat'

            # 模型参数路径
            write_file_dir = out_file_dir + "\\" + part_para + "\\" + dataset_para + r"\train"
            if not os.path.exists(write_file_dir):
                os.makedirs(write_file_dir)

            # 实验结果路径
            res_dir = out_file_dir + "\\" + part_para + r"\experiment result"
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            # # 训练过程（若未有训练好的模型参数文件时，重新训练模型）
            # # 生成对应数据的点击数据文件（只需生成一次即会保存下来。若已生成，下次可强制关闭，以节省运行时间。）
            # print("注意，已强制关闭extract_yoochoose_selected_data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # # print("extract_yoochoose_selected_data..")
            # # Preprocess11.extract_data(train_file_dir, test_data_dir, yoochoose_data_dir, yoochoose_selected_dir)
            # # print("finish extract_yoochoose_selected_data..")
            #
            # user_sessions_data, session_item_data, item_session_data = Input.get_data(train_file_dir)
            # print("finish getting data")
            # # 开始计时
            # start = time.time()
            # U, V, theta, likelihood = RLSO5.go(user_sessions_data, session_item_data, item_session_data, aspects_num, ITERATION)
            # c = time.time() - start
            # print("程序运行总耗时:%0.2f" % c, 's')
            #
            # # 假如输出文件夹不存在，则创建文件夹
            # if not os.path.exists(write_file_dir):
            #     os.makedirs(write_file_dir)
            # print2file_list = [[theta], likelihood]
            # # 输出结果到文件中
            # file_name = ["theta.txt", "likelihood.txt"]
            # idx = 0
            # for cur_list in print2file_list:
            #     cur_file_path = write_file_dir + "\\" + file_name[idx]
            #     p2f.print_list_to_file(cur_list, cur_file_path)
            #     idx += 1
            # U_file_path = write_file_dir + "\\" + "U.txt"
            # p2f.print_list_dict_to_file(U, U_file_path)
            # V_file_path = write_file_dir + "\\" + "V.txt"
            # p2f.print_list_dict_to_file(V, V_file_path)
            # # 画图——似然迭代过程
            # # pyplot.plot(range(len(likelihood)), likelihood)
            # # pyplot.show()


            #  write_file_dir：模型参数路径

            # （已经训练好模型）从文件中读取已经训练好的模型参数
            theta_file_path = write_file_dir + "\\" + "theta.txt"
            [theta] = rff.get_float_list(theta_file_path)   # rff ：read from file
            U_file_path = write_file_dir + "\\" + "U.txt"
            U = rff.get_float_list_dict(U_file_path)
            V_file_path = write_file_dir + "\\" + "V.txt"
            V = rff.get_float_list_dict(V_file_path)

            # 测试过程
            # 测试数据
            # test data/groundtruth

            #  test_data_dir = dataset_dir + r"\test" 当前数据样本的测试数据路径
            #  dataset_dir=I:\Papers\consumer\codeandpaper\Full\D1_partition\sampling@x@1@partition
            test_data_path = test_data_dir + r'\session_item.txt'
            session_item_data = rff.get_data_lists(test_data_path)
            # 测试数据的点击流数据（考虑商品的重复点击）
            test_click_stream_path = test_data_dir + r'\session_click_stream.txt'
            # 没有找到 “session_click_stream.txt”——后面，，，没有找到会生成。
            # 测试数据的点击数据文件
            test_file_path = yoochoose_selected_dir + r'\yoochoose-test-selected.dat'
            # 获取测试数据中各个session点击的item(item按点击顺序存放)（只考虑点击的不同商品，不考虑商品的重复点击——这是与session_click_stream的区别）
            dic, sessions, items_set = real_data.get_session_itemList(test_file_path)
            # 每个商品在各个session的出现次数（原静态特征，点击流场景下不会用到）
            item_session_times_dic = feature4.get_item_session_times(test_file_path)
            if os.path.exists(test_click_stream_path):
                session_click_stream = rff.get_int_list_dict(test_click_stream_path)
            else:
                session_click_stream = calcCorrelation.extract_click_stream(test_file_path)

            # res_path = res_dir + '\\' + dataset_para + '.txt'

            # 开始时初始化实验结果表格：输出行名、列名等信息
            if init_flag == 0:
                init_excel(res_dir, aggregate_num)
                # 表示表格已经初始化一次了，不用再初始化了
                init_flag = 1

            # 非early predict部分的实验（各种种整合策略）
            # 非ealry predict就是我们的论文里面的实验，本来还想做个early predict的东西，但是效果不好，就放弃了
            Recommendation22_aggregate.generate(click_file_path, buy_file_path, test_file_path,
                                                U, V, theta, aspects_num, session_item_data, dic, item_session_times_dic,
                                                session_click_stream, res_dir, part_num, aggregate_num)

            # （非early predict部分的实验（模型原始计算方法））
            # 重复加载一遍下面的数据的原因是Recommendation22_aggregate.generate（）方法貌似会对其函数参数造成一定改变，导致后面的程序运行时结果发生
            # （续）一定的改变（已知知道会发生改变）。
            # 测试过程
            # 测试数据
            # test data/groundtruth
            test_data_path = test_data_dir + r'\session_item.txt'
            session_item_data = rff.get_data_lists(test_data_path)
            # 测试数据的点击流数据（考虑商品的重复点击）
            test_click_stream_path = test_data_dir + r'\session_click_stream.txt'
            # 测试数据的点击数据文件
            test_file_path = yoochoose_selected_dir + r'\yoochoose-test-selected.dat'
            # 获取测试数据中各个session点击的item(item按点击顺序存放)（只考虑点击的不同商品，不考虑商品的重复点击——这是与session_click_stream的区别）
            dic, sessions, items_set = real_data.get_session_itemList(test_file_path)
            # 每个商品在各个session的出现次数（原静态特征，点击流场景下不会用到）
            item_session_times_dic = feature4.get_item_session_times(test_file_path)
            if os.path.exists(test_click_stream_path):
                session_click_stream = rff.get_int_list_dict(test_click_stream_path)
            else:
                session_click_stream = calcCorrelation.extract_click_stream(test_file_path)

            # 非early predict部分的实验（原始方法）
            Recommendation11.generate(click_file_path, buy_file_path, test_file_path,
                                      U, V, theta, aspects_num, session_item_data, dic, item_session_times_dic,
                                      session_click_stream, res_dir, part_num)


# 开始时初始化实验结果表格：输出行名、列名等信息
def init_excel(res_dir, aggregate_num):
    # 实验结果路径 res_dir = out_file_dir + "\\" + part_para + r"\experiment result"

    res_file_path = res_dir + r'\SimpleComparison.csv'
    file = open(res_file_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["", "p1", "precision", "MRR"])
    file.close()

    res_file_path = res_dir + r'\Recommendation11.csv'
    file = open(res_file_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["", "calc_item_prob", "", "", "", "calc_item_prob2", "", "", "", "calc_item_prob3"])
    writer.writerow(["", "precision", "MRR", "", "", "precision", "MRR", "", "","precision", "MRR"])
    file.close()

    for i in range(1, aggregate_num+1):
        res_file_path = res_dir + '\\' + 'Recommendation22_aggregate' + str(i) + '.csv'
        file = open(res_file_path, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(["", "calc_item_prob", "", "", "calc_item_prob2", "", "", "calc_item_prob3"])
        writer.writerow(["", "precision", "", "", "precision", "", "", "precision"])
        file.close()

    # res_file_path = res_dir + r'\Recommendation44.csv'
    # file = open(res_file_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(["", "calc_item_prob", "", "", "", "calc_item_prob2", "", "", "", "calc_item_prob3"])
    # writer.writerow(["", "precision", "ealiness", "", "", "precision", "earliness", "",  "", "precision", "earliness"])
    # file.close()


if __name__ == '__main__':
    TestRLSO()

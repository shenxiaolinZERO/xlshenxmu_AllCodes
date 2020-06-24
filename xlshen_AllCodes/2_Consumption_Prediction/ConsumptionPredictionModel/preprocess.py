#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import read_from_file as rff
import print_to_file as p2f
import random
import real_data
import preprocess2

## Zero:

# 1 对真实数据的采样
def sample_patition(rate, origin_file_dir, sampling_file_dir):

    origin_file_path = origin_file_dir + r"\session_item.txt"

    if not os.path.exists(sampling_file_dir):
        os.makedirs(sampling_file_dir)
    data_write_path = sampling_file_dir + r"\session_item.txt"
    items_write_path = sampling_file_dir + r"\items.txt"

    # 读取完整数据
    all_data = rff.get_data_lists(origin_file_path)
    # 进行采样
    sample_data, sample_items = sample_partition_help(all_data, rate)
    # 输出采样数据
    p2f.print_data_lists_to_file(sample_data, data_write_path)
    p2f.print_list_to_file(sample_items, items_write_path)


def sample_partition_help(all_data, rate):
    idx = 0
    sample_idx_list = list()
    sample_data = list()
    sample_items_set = set()
    for cur_data in all_data:
        rd = random.uniform(0, 1)
        if rd < rate:
            sample_idx_list.append(idx)
            sample_data.append(cur_data)
            for elem in cur_data[1]:
                sample_items_set.add(elem)
            for elem in cur_data[2]:
                sample_items_set.add(elem)
        idx += 1
    sample_items = list(sample_items_set)
    print(len(cur_data))
    print(len(sample_data))
    print(sample_idx_list)
    return sample_data, sample_items


# 2 将数据中购买了超过2个物品的session去掉（即只保留购买1个商品的session）
def data_selection(in_file_path, out_file_dir):
    out_data_file_path = out_file_dir + r'\session_item.txt'
    out_items_file_path = out_file_dir + r'\items.txt'
    data = rff.get_data_lists(in_file_path)
    selected_data = list()
    for cur_data in data:
        buy_items = cur_data[1]
        if len(buy_items) < 2:
            selected_data.append(cur_data)
    selected_items = extract_items(selected_data)
    p2f.print_data_lists_to_file(selected_data, out_data_file_path)
    p2f.print_list_to_file(selected_items, out_items_file_path)


# 3（废弃） 将测试数据中含有cold item的session（即测试集session中的item未在训练集中出现）移除（即只保留不含cold item的session）
def test_data_selection(out_file_dir, in_test_file_path, out_test_file_dir):
    train_items_file_path = out_file_dir + r'\items.txt'
    train_items = rff.get_int_list(train_items_file_path)
    test_data = rff.get_data_lists(in_test_file_path)
    test_data_selected = list()

    for cur_test_data in test_data:
        cur_items = cur_test_data[1] + cur_test_data[2]
        selection = True
        for item in cur_items:
            if item in train_items:
                continue
            else:
                selection = False
                break
        if selection:
            test_data_selected.append(cur_test_data)
    test_items_selected = extract_items(test_data_selected)
    out_test_data_file_path = out_test_file_dir + r'\session_item.txt'
    out_test_items_file_path = out_test_file_dir + r'\items.txt'
    p2f.print_data_lists_to_file(test_data_selected, out_test_data_file_path)
    p2f.print_list_to_file(test_items_selected, out_test_items_file_path)


# 提取session_item_data里的item
def extract_items(data):
    items_set = set()
    for cur_data in data:
        buy_items = cur_data[1]
        unbuy_items = cur_data[2]
        for item in buy_items:
            items_set.add(item)
        for item in unbuy_items:
            items_set.add(item)
    return list(items_set)


# 提取session_item_data里的item（session_item_data数据用2list2_dict表示）
def extract_items2(data):
    items_set = set()
    for session in data.keys():
        buy_items = data[session][0]
        unbuy_items = data[session][1]
        for item in buy_items:
            items_set.add(item)
        for item in unbuy_items:
            items_set.add(item)
    return list(items_set)


# ###划分数据集，使测试集不出现cold item（by nisheng）
def split_data_without_cold_start_item(session_item, setting):
    # 划分完以后的测试集
    test_session_item = {}
    # 总的Session数目
    session_num = len(session_item)
    # 存储所有session id
    session_list = list(session_item.keys())

    # 设置随机种子
    # random.seed(0)

    while len(test_session_item) < (int(setting * session_num)):
        session_id = session_list[random.randint(0, session_num - 1)]
        if session_id in test_session_item.keys():
            continue
        else:
            test_session_item[session_id] = session_item.pop(session_id)

    # 将测试集中含有冷启动商品的session数据移至训练集中
    split_data_help(session_item, test_session_item)

    # 下面这段代码已放入split_data_help函数中
    # # 训练集中的所有item集合
    # item_in_train = set()
    # for session_id in session_item:
    #     click_items = session_item[session_id][0] + session_item[session_id][1]
    #     for click_item in click_items:
    #         item_in_train.add(click_item)
    #
    # remove_session = []
    # # 如果测试集中有某个session的某个item在训练集中不出现，则将该session移至训练集
    # for session_id in test_session_item:
    #     click_items = test_session_item[session_id][0] + test_session_item[session_id][1]
    #     # 如果某个item不在训练集中，则标记为false
    #     mark = True
    #     for click_item in click_items:
    #         if click_item not in item_in_train:
    #             mark = False
    #             break
    #     if mark == False:
    #         remove_session.append(session_id)
    #
    # for session_id in remove_session:
    #     session_item[session_id] = test_session_item.pop(session_id)

    # 注意，此处返回的session_item已由输入数据变成所要求的训练数据
    print("训练集session数目：", len(session_item))
    print("测试集session数目：", len(test_session_item))
    print("训练集session数目/测试集session数目：", len(session_item)/len(test_session_item))

    return session_item, test_session_item


# 将测试集中含有冷启动商品的session数据移至训练集中
def split_data_help(train_session_item, test_session_item):
    # 训练集中的所有item集合
    item_in_train = set()
    for session_id in train_session_item:
        click_items = train_session_item[session_id][0] + train_session_item[session_id][1]
        for click_item in click_items:
            item_in_train.add(click_item)

    remove_session = []
    # 如果测试集中有某个session的某个item在训练集中不出现，则将该session移至训练集
    for session_id in test_session_item:
        click_items = test_session_item[session_id][0] + test_session_item[session_id][1]
        # 如果某个item不在训练集中，则标记为false
        mark = True
        for click_item in click_items:
            if click_item not in item_in_train:
                mark = False
                break
        if mark == False:
            remove_session.append(session_id)
    for session_id in remove_session:
        train_session_item[session_id] = test_session_item.pop(session_id)


# 将测试集中含有冷启动商品的session数据删去，并返回测试集中那些被删去的session
def split_data_help2(train_session_item, test_session_item):
    # 训练集中的所有item集合
    item_in_train = set()
    for session_id in train_session_item:
        click_items = train_session_item[session_id][0] + train_session_item[session_id][1]
        for click_item in click_items:
            item_in_train.add(click_item)

    remove_session = []
    # 如果测试集中有某个session的某个item在训练集中不出现，则将该session数据移除
    for session_id in test_session_item:
        click_items = test_session_item[session_id][0] + test_session_item[session_id][1]
        # 如果某个item不在训练集中，则标记为false
        mark = True
        for click_item in click_items:
            if click_item not in item_in_train:
                mark = False
                break
        if mark == False:
            remove_session.append(session_id)
    for session_id in remove_session:
        del test_session_item[session_id]

    return remove_session


# 划分数据集test
def split_data_without_cold_start_item_test(all_data_path, out_dir, setting):

    data_pro = rff.get_2lists_dict(all_data_path)
    train_session_item, test_session_item = split_data_without_cold_start_item(data_pro, setting)

    # 输出数据
    print_partition(out_dir, train_session_item, test_session_item)

    # train_items_list = extract_items2(train_session_item)
    # test_items_list = extract_items2(test_session_item)
    #
    # # 输出文件路径
    # out_train_dir = out_dir + r'\train'
    # if not os.path.exists(out_train_dir):
    #     os.makedirs(out_train_dir)
    # out_train_data_path = out_train_dir + r'\session_item.txt'
    # out_train_item_path = out_train_dir + r'\items.txt'
    #
    # out_test_dir = out_dir + r'\test'
    # if not os.path.exists(out_test_dir):
    #     os.makedirs(out_test_dir)
    # out_test_data_path = out_test_dir + r'\session_item.txt'
    # out_test_item_path = out_test_dir + r'\items.txt'
    #
    # # 输出数据
    # p2f.print_2lists_dict_to_file(train_session_item, out_train_data_path)
    # p2f.print_list_to_file(train_items_list, out_train_item_path)
    # p2f.print_2lists_dict_to_file(test_session_item, out_test_data_path)
    # p2f.print_list_to_file(test_items_list, out_test_item_path)

    return train_session_item, test_session_item


# 此处的train_data和test_data指的是no_cutoff数据的划分数据，在此处会得到“修正”
def extract_final_partition(train_data, test_data, all_data_path, cutoff_data_path):
    # print("###########################################")
    # print("temp测试集session数目：", len(test_session_item))
    # 根据“非截断”划分数据获取“截断”划分数据
    data_pro = rff.get_2lists_dict(cutoff_data_path)
    # 获取“截断”划分数据的训练数据
    train_session_list = list(train_data.keys())
    cutoff_train_data = extract_data_by_session(data_pro, train_session_list)
    # 将此时测试集中含有冷启动商品的session数据删去，并返回测试集中那些被删去的session
    removed_session = split_data_help2(cutoff_train_data, test_data)
    # print("最终测试集session数目：", len(test_session_item))
    # print("训练集session数目/测试集session数目：", len(cutoff_train_session_item) / len(test_session_item))
    # 将测试集中被删去的session的数据放入训练集中（非截断、截断数据需要分开处理，因为训练集的商品假设不同）
    # 非截断
    no_cutoff_data_pro = rff.get_2lists_dict(all_data_path)
    removed_data = extract_data_by_session(no_cutoff_data_pro, removed_session)
    # combine_data：将session_item中的数据放入train_session_item数据中
    combine_data(train_data, removed_data)
    # print("训练集session数目/测试集session数目：", len(train_session_item) / len(test_session_item))
    # 截断
    removed_data = extract_data_by_session(data_pro, removed_session)
    combine_data(cutoff_train_data, removed_data)
    # print("训练集session数目/测试集session数目：", len(cutoff_train_session_item) / len(test_session_item))

    return cutoff_train_data


# 辅助。从原始数据中提取出某些session的数据
def extract_data_by_session(data_pro, session_list):
    session_item = dict()
    for session in session_list:
        session_item[session] = data_pro[session]
    return session_item


# 辅助。combine_data：将session_item2中的数据放入session_item1数据中
def combine_data(session_item1, session_item2):
    for key in session_item2:
        session_item1[key] = session_item2[key]


# 提取data数据的点击流数据
def extract_click_stream(click_file_path, session_item):
    all_session = extract_session_of_click_file(click_file_path)
    current_session = extract_session_of_data_pro(session_item)
    # 标记一个session是否出现在当前data数据里，是为1，否为0（节省时间）
    session_flag_dic = dict()
    for session in all_session:
        session_flag_dic[session] = 0
    # 出现在当前data数据中的session
    for session in current_session:
        session_flag_dic[session] = 1
    # 提取data数据的点击流数据
    dic = dict()
    # 当前点击文件中存在于data中的session数据
    data_session_list = list()
    f = open(click_file_path)
    try:
        for line in f:
            tmp = line.split(',')
            session = int(tmp[0])
            item = int(tmp[2])
            # if len(session_list) == 0:
            #     session_list.append(session)
            # else:
            #     if session == session_list[-1]:

            # 判断当前session是否为data的session
            if session_flag_dic[session] == 1:
                # 判断该data session是否出现过
                # 是否为第一个data session
                if len(data_session_list) == 0:
                    data_session_list.append(session)
                    dic[session] = list()
                # 是否来了一个新的session
                elif session != data_session_list[-1]:
                    data_session_list.append(session)
                    dic[session] = list()
                dic[session].append(item)
    except Exception as e:
        print(e)
    finally:
        f.close()

    return dic


# 提取data数据里面的session
def extract_session_of_data_pro(data_pro):
    session_list = list()
    for session in data_pro.keys():
        session_list.append(session)
    return session_list


# 提取点击文件里面的session
def extract_session_of_click_file(file_path):
    session_list = list()
    f = open(file_path)
    try:
        for line in f:
            tmp = line.split(',')
            session = int(tmp[0])
            session_list.append(session)
    except Exception as e:
        print(e)
    finally:
        f.close()
    return session_list


# ###取出提取数据中第一个购买商品前的点击数据（包含第一个购买商品）（注意要舍去第一个浏览商品就是购买商品的session）
# 并获取符合条件的数据中每个session第一个购买的商品
# 这里click_file_path为提取数据（extracted）点击文件，data_pro为提取数据session_item_data
def get_early_predict_cutoff_data(click_file_path, data_pro, write_file_path):
    f = open(click_file_path)
    # extracted点击数据中的所有session
    all_session_lists = list()
    write_f = open(write_file_path, 'w')
    # 判断该session是否要抛弃（即该session第一个点击item是购买商品）
    abandom_session_flag = 0
    # 是否遇到session的第一个购买商品标记
    fisrt_buy_item_flag = 0
    # 当前session输出停止标记（当session第一个购买商品输出结束时输出停止）
    print_stop_flag = 0
    # 选出的符合条件的数据
    # 符合条件的数据session_flag_dic的值为1，不符合条件的为0
    session_flag_dic = dict()
    # 符合条件的数据中，每个session的第一个购买商品
    session_fisrt_buy_item_dic = dict()
    try:
        for line in f:
            tmp = line.split(',')
            session = int(tmp[0])
            item = int(tmp[2])
            # 每次来了一个新的session，判断该session是否要抛弃（即该session第一个点击item是购买商品）
            # 若all_session_lists为空
            if len(all_session_lists) == 0:
                all_session_lists.append(session)
                buy_items = data_pro[session][0]
                # 新的session的第一个item
                if item in buy_items:
                    abandom_session_flag = 1
            # 来了一个新的session
            elif session != all_session_lists[-1]:
                abandom_session_flag = 0
                fisrt_buy_item_flag = 0
                print_stop_flag = 0
                all_session_lists.append(session)
                buy_items = data_pro[session][0]
                # 新的session的第一个item
                if item in buy_items:
                    abandom_session_flag = 1

            # 还是原来的session；或者来了一个新的session
            # 该session的第一个点击item即为购买商品，跳过该session的所有item
            if abandom_session_flag == 1:
                session_flag_dic[session] = 0
                continue
            # 该session的第一个点击item不是购买商品，则符合条件，准备进行输出
            if abandom_session_flag == 0:
                session_flag_dic[session] = 1
                # 若点击商品到了第一个购买商品
                if fisrt_buy_item_flag == 0 and item in buy_items:
                    fisrt_buy_item_flag = 1
                    cur_first_buy_item = item
                    session_fisrt_buy_item_dic[session] = cur_first_buy_item
                # 若点击商品到了第一个购买商品后的第一个其他商品
                if print_stop_flag == 0 and fisrt_buy_item_flag == 1 and item != cur_first_buy_item:
                    print_stop_flag = 1
                # 输出
                if fisrt_buy_item_flag == 0:
                    write_f.write(line)
                elif print_stop_flag == 0:
                    write_f.write(line)

    except Exception as e:
        print(e)
    finally:
        f.close()
        write_f.close()

    # 返回值用于从购买数据中提取满足条件的数据
    return session_flag_dic, session_fisrt_buy_item_dic


# 在取出early predict截断点击数据后，提取目标购买数据
def get_early_predict_buy_data(buy_file_path, session_flag_dic, session_fisrt_buy_item_dic, write_file_path):
    f = open(buy_file_path)
    write_f = open(write_file_path, 'w')
    try:
        for line in f:
            tmp = line.split(',')
            session = int(tmp[0])
            item = int(tmp[2])
            if session_flag_dic[session] == 1 and item == session_fisrt_buy_item_dic[session]:
                write_f.write(line)
    except Exception as e:
        print(e)
    finally:
        f.close()
        write_f.close()


# 分别输出划分数据的训练数据和测试数据
def print_partition(out_dir, train_session_item, test_session_item):

    train_items_list = extract_items2(train_session_item)
    test_items_list = extract_items2(test_session_item)

    # 输出文件路径
    out_train_dir = out_dir + r'\train'
    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir)
    out_train_data_path = out_train_dir + r'\session_item.txt'
    out_train_item_path = out_train_dir + r'\items.txt'

    out_test_dir = out_dir + r'\test'
    if not os.path.exists(out_test_dir):
        os.makedirs(out_test_dir)
    out_test_data_path = out_test_dir + r'\session_item.txt'
    out_test_item_path = out_test_dir + r'\items.txt'

    # 输出数据
    p2f.print_2lists_dict_to_file(train_session_item, out_train_data_path)
    p2f.print_list_to_file(train_items_list, out_train_item_path)
    p2f.print_2lists_dict_to_file(test_session_item, out_test_data_path)
    p2f.print_list_to_file(test_items_list, out_test_item_path)

    '''
if __name__ == '__main__':

    # # 将数据中购买了超过2个物品的session去掉（即只保留购买1个商品的session）
    # data_para = 'sampling@0.01@partition'
    # out_data_para = 'sampling@0.01@partition@selection'
    # main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
    # in_file_dir = main_dir + '\\' + data_para + r'\train'
    # in_file_path = in_file_dir + r'\session_item.txt'
    # out_file_dir = main_dir + '\\' + out_data_para + r'\train'
    # if not os.path.exists(out_file_dir):
    #     os.makedirs(out_file_dir)
    # data_selection(in_file_path, out_file_dir)

    # ##对数据进行采样
    rate = 0.48
    # （原来）
    main_dir = r"E:\ranking aggregation\dataset\yoochoose\Full"

    # Zero

    # 输入数据集选择
    dataset_para = "D6"
    # 采样前数据所在路径
    origin_file_dir = main_dir + "\\" + dataset_para
    # 采样后、划分后数据所在路径
    out_dir = main_dir + "\\" + dataset_para + "_partition"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 采样与数据划分
    for i in range(1,51):
        # 数据序号number（用于生成重复采样率的数据时文件夹的命名）
        number = i
        # 采样后得到的数据所在路径
        data_para = "sampling@x" + '@' + str(number)
        sampling_file_dir = out_dir + "\\" +data_para
        # 原始数据的采样
        sample_patition(rate, origin_file_dir, sampling_file_dir)

        # 对原始数据采样后，再调整数据集（将测试集中的cold start session移到训练集中），使测试集不出现cold item（by nisheng）
        # 划分前(采样后)的采样数据集所在路径
        all_data_path = sampling_file_dir + r'\session_item.txt'
        # 划分后的数据结果所在路径
        out_data_para = data_para + '@partition'
        out_data_dir = out_dir + '\\' + out_data_para
        # setting: 训练集、测试集初始划分比例。为使最终目标划分数据集比例大概为4比1，0.1数据时设置为0.22   0.001数据时设置为0.4
        #  完整数据要设为4/1的话就是0.25

        setting = 0.32
        temp_train_data, temp_test_data = \
            split_data_without_cold_start_item_test(all_data_path, out_data_dir, setting)
    '''
    # zero ：
if __name__ == '__main__':

        # # 将数据中购买了超过2个物品的session去掉（即只保留购买1个商品的session）
        # data_para = 'sampling@0.01@partition'
        # out_data_para = 'sampling@0.01@partition@selection'
        # main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
        # in_file_dir = main_dir + '\\' + data_para + r'\train'
        # in_file_path = in_file_dir + r'\session_item.txt'
        # out_file_dir = main_dir + '\\' + out_data_para + r'\train'
        # if not os.path.exists(out_file_dir):
        #     os.makedirs(out_file_dir)
        # data_selection(in_file_path, out_file_dir)

        # ##对数据进行采样
        rate = 0.48
        # （原来）
        # main_dir = r"E:\ranking aggregation\dataset\yoochoose\Full"

        # Zero

        # 输入数据集选择
        # dataset_para = "D6"
        # 采样前数据所在路径
        #origin_file_dir = main_dir + "\\" + dataset_para
        # 采样后、划分后数据所在路径
        # out_dir = main_dir + "\\" + dataset_para + "_partition"
        #if not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
    # 采样与数据划分
    #for i in range(1, 51):
        # 数据序号number（用于生成重复采样率的数据时文件夹的命名）
        #number = i
        # 采样后得到的数据所在路径
       # data_para = "sampling@x" + '@' + str(number)
       # sampling_file_dir = out_dir + "\\" + data_para
        # 原始数据的采样
        #origin_file_dir=r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata'
        #sampling_file_dir=
        #sample_patition(rate, origin_file_dir, sampling_file_dir)

        # 对原始数据采样后，再调整数据集（将测试集中的cold start session移到训练集中），使测试集不出现cold item（by nisheng）
        # 划分前(采样后)的采样数据集所在路径
        #all_data_path = sampling_file_dir + r'\session_item.txt'
        all_data_path = r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\originAllData\session_item.txt'

        # 划分后的数据结果所在路径
        #out_data_para = data_para + '@partition'
        #out_data_dir = out_dir + '\\' + out_data_para
        out_data_dir=r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\sampling@alldata@partition'

        # setting: 训练集、测试集初始划分比例。为使最终目标划分数据集比例大概为4比1，0.1数据时设置为0.22   0.001数据时设置为0.4
        #  完整数据要设为4/1的话就是0.25，但是结果是3/1 ?
        #  设为了0.2，结果是4/1
        #  是setting=测试：训练，不过因为得将测试集中有冷启动商品的session移到训练集中，所以setting小于0.25
        setting = 0.2
        temp_train_data, temp_test_data = \
            split_data_without_cold_start_item_test(all_data_path, out_data_dir, setting)

    # ####以下程序与截断数据/非截断数据有关###########################################################

    # # ##对数据进行采样
    # rate = 0.1
    # # 数据序号number（用于生成重复采样率的数据时文件夹的命名）
    # number = 0
    # main_dir = r"E:\ranking aggregation\dataset\yoochoose\Full"
    # origin_file_dir = main_dir + r"\early predict\no_cutoff"
    #
    # if number == 0:
    #     # 对于之前的数据，将这里origin_file_dir换成main_dir
    #     partition_file_dir = origin_file_dir + r"\sampling@" + str(rate)
    # # 当有指定数据序号时
    # else:
    #     partition_file_dir = origin_file_dir + r"\sampling@" + str(rate) + '@' + str(number)
    # # 原始数据的采样
    # sample_patition(rate, origin_file_dir, partition_file_dir)
    #
    # # ###划分数据集，使测试集不出现cold item（by nisheng）
    # main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
    # data_para = r'early predict\no_cutoff\sampling@0.1'
    # all_data_path = main_dir + '\\' + data_para + r'\session_item.txt'
    # # 产生‘temp’数据的原因：此处得到“非截断”的划分数据后，在后面获取（相同训练和测试session的）“截断”的划分数据中仍可能出现冷启动项，因此此处得到的“非截断”的划分数据仍只是一个中间结果
    # out_data_para = r'early predict\no_cutoff\sampling@0.1@partition@temp'
    # out_dir = main_dir + '\\' + out_data_para
    # # setting: 训练集、测试集初始划分比例。为使最终目标划分数据集比例大概为4比1，0.1数据时设置为0.22   0.001数据时设置为0.4
    # setting = 0.3
    # temp_train_data, temp_test_data =\
    #     split_data_without_cold_start_item_test(all_data_path, out_dir, setting)


    # # # 根据“非截断”划分数据获取最终的“截断”划分数据和“修正后”的“非截断”划分数据（此时非截断的temp_train_data, temp_test_data也会经过“修正”）
    # cutoff_data_path = main_dir + r'\early predict\cutoff\session_item.txt'
    # cutoff_train_data = extract_final_partition(temp_train_data, temp_test_data, all_data_path, cutoff_data_path)
    # # 输出
    # # “修正后”的“非截断”划分数据
    # out_dir1 = main_dir + '\\' + r'early predict\no_cutoff\sampling@0.1@partition'
    # print_partition(out_dir1, temp_train_data, temp_test_data)
    # # “修正后”“截断”划分数据
    # out_dir2 = main_dir + '\\' + r'early predict\cutoff\sampling@0.1@partition'
    # print_partition(out_dir2, cutoff_train_data, temp_test_data)
    # # 提取测试集数据的点击流数据
    # click_file_path = main_dir + r'\yoochoose-clicks.dat'
    # session_click_stream = extract_click_stream(click_file_path, temp_test_data)
    # # 输出到文件中
    # write_file_path1 = out_dir1 + r'\test\session_click_stream.txt'
    # write_file_path2 = out_dir2 + r'\test\session_click_stream.txt'
    # p2f.print_list_dict_to_file(session_click_stream, write_file_path1)
    # p2f.print_list_dict_to_file(session_click_stream, write_file_path2)


    # # ###从“extracted”数据中提取early predict“截断”点击数据及购买数据（未分训练、测试）
    # main_dir = r'E:\ranking aggregation\dataset\yoochoose\Full'
    # extract_file_dir = main_dir + '\\' + 'extracted'    # 既包含购买商品，也包含点击不购买商品的session数据
    # # 输入：“extracted”数据
    # data_pro = rff.get_2lists_dict(extract_file_dir + r'\session_item.txt')
    # click_file_path = extract_file_dir + r'\yoochoose-selected\yoochoose-clicks-selected.dat'
    # write_file_path1 = main_dir + r'\early predict\cutoff\yoochoose-clicks-selected.dat'   # 目标的输出路径
    # # 取出“extracted”数据中第一个购买商品前的点击数据并输出到文件中（包含第一个购买商品）（注意要舍去第一个浏览商品就是购买商品的session）
    # session_flag_dic, session_fisrt_buy_item_dic = get_early_predict_cutoff_data(click_file_path, data_pro, write_file_path1)
    # buy_file_path = extract_file_dir + r'\yoochoose-selected\yoochoose-buys-selected.dat'
    # write_file_path2 = main_dir + r'\early predict\cutoff\yoochoose-buys-selected.dat'     # 目标的输出路径
    # # 在取出early predict截断点击数据后，提取目标购买数据
    # get_early_predict_buy_data(buy_file_path, session_flag_dic, session_fisrt_buy_item_dic, write_file_path2)
    # data_write_dir = main_dir + r'\early predict\cutoff'
    # # 根据cutoff点击数据及购买数据，获取session_item_data
    # real_data.extract_real_data1(write_file_path1, write_file_path2, data_write_dir)
    #
    # # 根据early predict“截断”数据获取“非截断”数据（当前做法不直接，更直接的做法是根据session直接到session_item_data中）
    # # （注：“截断”数据与“非截断”数据的关系：session相同，但非截断数据包含每个session的所有点击item）
    # # 提取“截断”数据session
    # cutoff_data_path = data_write_dir + r"\session_item.txt"
    # cutoff_session = set()
    # preprocess2.extract_session(cutoff_data_path, cutoff_session)
    # # 根据该数据session进行提取“非截断”点击数据和购买数据（包含每个session的所有item的点击和购买数据）
    # clicks_selected_path = main_dir + r'\early predict\no_cutoff\yoochoose-clicks-selected.dat'
    # buys_selected_path = main_dir + r'\early predict\no_cutoff\yoochoose-buys-selected.dat'
    # preprocess2.extract_and_print_data(click_file_path, cutoff_session, clicks_selected_path)
    # preprocess2.extract_and_print_data(buy_file_path, cutoff_session, buys_selected_path)
    # # 根据"非截断"点击和购买数据获取非截断session_item_data
    # data_write_dir2 = main_dir + r'\early predict\no_cutoff'
    # real_data.extract_real_data1(clicks_selected_path, buys_selected_path, data_write_dir2)


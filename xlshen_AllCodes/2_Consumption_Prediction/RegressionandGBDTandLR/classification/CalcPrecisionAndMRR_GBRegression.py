from input import Input
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from solution import Solution
from evaluate import Evaluate
from preprocess2 import Preprocess2
from feature2 import Feature2
from input2 import Input2
import random
import os
import csv
from feature3 import Feature3
from feature4 import Feature4
from feature5 import Feature5
from feature6 import Feature6
import read_from_file as rff
import sys
sys.path.append(r'I:\Papers\consumer\codeandpaper\RegressionandGBDTandLR\skyline_recommendation\rlso')
# from regreEvaluate import RegreEvaluate
import recommendation1
import time
# 综合了时间类特征和新特征的分类和回归实验
# 需要设置的路径：
# dataset_dir （实验数据）train\session_item.txt  test\session_item.txt
# yoochoose_data_dir （yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
# feature_dir 提取特征结果所在文件夹

def classifier_test():

    print("这是 GBRegression 回归方法")
    # setting
    # dataset_para = 'sampling@x@'+str(i)+'@partition'
    # 特征的选择：时间类特征：time； 新特征：new；  时间类特征+新特征: all
    feature = 'all'

    # 若用新特征，选择使用哪些特征
    feature_para = (1, 2, 3, 4)

    # file directory
    # feature_dir = dataset_dir + r'\feature1'

    # ###预处理：从原始数据yoochoose-data中提取出实验数据所需要部分数据（根据实验数据session进行提取）
    # 输入1：（实验数据）dataset_dir\train\session_item.txt  .\test\session_item.txt
    # 输入2：（yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
    # 输出：dataset_dir\yoochoose-selected\yoochoose-clicks-selected.dat  .\yoochoose-buys-selected.dat  .\yoochoose-test-selected.dat
    dataset_dir = r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\sampling@alldata@partition'
    #yoochoose_data_dir = r'E:\recsyschallenge2015\mycode\yoochoose-data'
    # 输出路径
    #yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
    # 假如输出文件夹不存在，则创建文件夹
    # if not os.path.exists(yoochoose_selected_dir):
    #     os.makedirs(yoochoose_selected_dir)
    # Preprocess2.extract_data(dataset_dir, yoochoose_data_dir, yoochoose_selected_dir)

    # ###提取特征
    # 输入：yoochoose selected data（及groundtruth）
    # 提取特征结果所在文件夹

    feature_dir = r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\sampling@alldata@partition\feature'
    # feature_dir = dataset_dir + r'\feature1'
    # 假如输出文件夹不存在，则创建文件夹
    """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    if feature == 'time':
        print('feature:', feature)
        Feature5.go(dataset_dir, feature_dir)
    elif feature == 'new':
        print('feature:', feature, 'feature_para:', feature_para)
        Feature4.go(dataset_dir, feature_dir, feature_para)
    else:
        print('feature:', feature)
        Feature6.go(dataset_dir, feature_dir)
    """
    # 读取特征
    # 训练文件路径
    train_file_path = feature_dir + r'\click-buy-train.arff'
    # 测试文件路径
    test_file_path = feature_dir + r'\click-buy-test-BR.txt'
    groundtruth_path = dataset_dir + r'\test\session_item.txt'

    X_train, y_train = Input2.read_train(train_file_path)
    X_test, y_test, test_dic_data = Input2.read_test(test_file_path, groundtruth_path)

    groundtruth_path = dataset_dir + r'\test\session_item.txt'
    session_item_data = rff.get_data_lists(groundtruth_path)

    # 模型训练

    # ########## LR 这个方法
    # print('model: LogisticRegressionClassifier')
    # model_LR = LogisticRegression()  # LogisticRegression()
    # model_LR.fit(X_train, y_train)
    # # 取第二列，即类为1的分数
    # score_LR = model_LR.predict_proba(X_test)[:, 1]
    # session_item_score_dic_LR = extract_score_by_session2(score_LR, test_dic_data)

    # ########## GB 这个方法
    # print('model: GradientBoostingClassifier')
    # model_GB = GradientBoostingClassifier()  # GradientBoostingClassifier()
    # model_GB.fit(X_train, y_train)
    # # 取第二列，即类为1的分数
    # score_GB = model_GB.predict_proba(X_test)[:, 1]
    # session_item_score_dic_GB = extract_score_by_session2(score_GB, test_dic_data)

    # ########## LinearRegression这个方法
    # print('model: LinearRegression')
    # model_LRegress = LinearRegression()
    # model_LRegress.fit(X_train, y_train)
    # # 取第二列，即类为1的分数
    # score_LRegress = model_LRegress.predict(X_test)
    # session_item_score_dic_LRegress = extract_score_by_session2(score_LRegress, test_dic_data)

    ########## GBRegression这个方法
    print('model: GBRegressor')
    model_GBRegressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    model_GBRegressor.fit(X_train, y_train)
    # 取第二列，即类为1的分数
    score_GBRegressor = model_GBRegressor.predict(X_test)
    session_item_score_dic_GBRegressor = extract_score_by_session2(score_GBRegressor, test_dic_data)


    # Zero：实验结果存放路径
    res_dir = r'I:\Papers\consumer\codeandpaper\PreprocessData\alldata\result_classifier&regression'
    if not os.path.exists(res_dir):
                os.makedirs(res_dir)

    init_flag = 0
    if init_flag == 0:
        init_excel(res_dir)
        # 表示表格已经初始化一次了，不用再初始化了
        init_flag = 1

    # 将结果输出到文件中
    res_file_path = res_dir + r'\GBRegression.csv'
    file = open(res_file_path, 'a', newline='')
    writer = csv.writer(file)
    data = list()

    # 开始计算precision和MRR
    for cur_data in session_item_data:

        precision4 = 0.0
        MRR4 = 0.0

        session = cur_data[0]
        n = len(cur_data[1])
        cur_buy_items = cur_data[1]
        print("计算购买个数:", n)

        cur_item_prob = session_item_score_dic_GBRegressor[session]
        for i in range(n):
            if cur_item_prob[i][0] in cur_buy_items:
                precision4 += 1 / n
        for i in range(len(cur_item_prob)):
            if cur_item_prob[i][0] in cur_buy_items:
                MRR4 += 1.0 / (i + 1)
                break

        data = [str('%.4f' % session), str('%.4f' % precision4), str('%.4f' % MRR4)]
        writer.writerow(data)

    file.close()


# 改进：
# 开始时初始化实验结果表格：输出行名、列名等信息
def init_excel(res_dir):
    # 实验结果路径 res_dir = out_file_dir + "\\" + part_para + r"\experiment result"
    res_file_path = res_dir + r'\GBRegression.csv'
    file = open(res_file_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow([ "", "GBRegression",""])
    writer.writerow(["sessionID", "precision", "MRR"])

    file.close()


        # """
        # print type(solution[0])
        # print solution[0]
        # Evaluate.go(solution, session_item_data)
        # """
        # #break

        
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


def setResgressionScore(lable, s0, s1):
    for i in range(len(lable)):
        if lable[i] == 0:
            lable[i] = s0
        else:
            lable[i] = s1


if __name__ == '__main__':

    start = time.clock()
    print("开始时间：",start)

    classifier_test()

    end = time.clock()
    print("结束时间：", end)
    c = end - start

    print("GBRegression 程序运行总耗时:%0.2f" % c, 's')
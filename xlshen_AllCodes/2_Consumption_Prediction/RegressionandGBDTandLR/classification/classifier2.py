from input import Input
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from solution import Solution
from evaluate import Evaluate
from preprocess2 import Preprocess2
from feature2 import Feature2
from input2 import Input2
import os
from feature3 import Feature3
from feature4 import Feature4


def classifier_test():

    # setting
    dataset_para = 'sampling@0.01@partition@selection'
    # 选择使用哪些特征
    feature_para = (1, 2, 3, 4)

    # file directory

    # feature_dir = dataset_dir + r'\feature1'

    # 预处理：从原始数据yoochoose-data中提取出实验数据所需要部分数据（根据实验数据session进行提取）
    # 输入1：（实验数据）dataset_dir\train\session_item.txt  .\test\session_item.txt
    # 输入2：（yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
    # 输出：dataset_dir\yoochoose-selected\yoochoose-clicks-selected.dat  .\yoochoose-buys-selected.dat  .\yoochoose-test-selected.dat
    dataset_dir = r'E:\ranking aggregation\dataset\yoochoose\Full' + '\\' + dataset_para
    yoochoose_data_dir = r'E:\recsyschallenge2015\mycode\yoochoose-data'
    # 输出路径
    yoochoose_selected_dir = dataset_dir + r'\yoochoose-selected'
    # 假如输出文件夹不存在，则创建文件夹
    # if not os.path.exists(yoochoose_selected_dir):
    #     os.makedirs(yoochoose_selected_dir)
    # Preprocess2.extract_data(dataset_dir, yoochoose_data_dir, yoochoose_selected_dir)

    # 提取特征
    # 输入：yoochoose selected data（及groundtruth）
    # 输出：特征
    feature_dir = r'E:\recsyschallenge2015\mycode\result-data'
    # feature_dir = dataset_dir + r'\feature1'
    # 假如输出文件夹不存在，则创建文件夹
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    print('feature_para:', feature_para)
    Feature4.go(dataset_dir, feature_dir, feature_para)

    # 读取特征
    X_train, y_train = Input2.read_train(feature_dir)
    X_test, y_test, test_dic_data, session_item_data, session_idx_dic = Input2.read_test(dataset_dir, feature_dir)

    groundtruth_path = dataset_dir + r'\test\session_item.txt'
    # 模型部分
    print('model: LogisticRegression')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # print(model)
    # make predictions
    y_predict = model.predict(X_test)
    # 结果评估
    solution = Solution.generate(test_dic_data, y_predict)
    Evaluate.go(solution, groundtruth_path)

    # 模型部分
    print('model: GaussianNB')
    model = GaussianNB()
    model.fit(X_train, y_train)
    # print(model)
    # make predictions
    y_predict = model.predict(X_test)
    # 结果评估
    solution = Solution.generate(test_dic_data, y_predict)
    Evaluate.go(solution, groundtruth_path)

    # 模型部分
    print('model: SVM')
    model = SVC()
    model.fit(X_train, y_train)
    # print(model)
    # make predictions
    y_predict = model.predict(X_test)
    # 结果评估
    solution = Solution.generate(test_dic_data, y_predict)
    Evaluate.go(solution, groundtruth_path)

if __name__ == '__main__':
    classifier_test()

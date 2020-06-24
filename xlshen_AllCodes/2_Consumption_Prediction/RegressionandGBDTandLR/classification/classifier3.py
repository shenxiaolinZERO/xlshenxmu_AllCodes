from input import Input
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from solution import Solution
from evaluate import Evaluate
from preprocess2 import Preprocess2
from feature2 import Feature2
from input2 import Input2
import os
from feature3 import Feature3
from feature4 import Feature4
from feature5 import Feature5
from feature6 import Feature6
import read_from_file as rff
# import sys
# sys.path.append(r'E:\ranking aggregation\code\regression')
# from regreEvaluate import RegreEvaluate


# 综合了时间类特征和新特征的分类和回归实验
# 需要设置的路径：
# dataset_dir （实验数据）train\session_item.txt  test\session_item.txt
# yoochoose_data_dir （yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
# feature_dir 提取特征结果所在文件夹

def classifier_test():

    for i in range(1,51):
        print(i)
        # setting
        dataset_para = 'sampling@x@'+str(i)+'@partition'
        # 特征的选择：时间类特征：time； 新特征：new；  时间类特征+新特征: all
        feature = 'all'
    
        # 若用新特征，选择使用哪些特征
        feature_para = (1, 2, 3, 4)
        #zero:看 feature4.py的go(dataset_dir, feature_dir, feature_para=(1, 2, 3, 4)):

        # file directory
        # feature_dir = dataset_dir + r'\feature1'
    
        # ###预处理：从原始数据yoochoose-data中提取出实验数据所需要部分数据（根据实验数据session进行提取）
        # 输入1：（实验数据）dataset_dir\train\session_item.txt  .\test\session_item.txt
        # 输入2：（yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
        # 输出：dataset_dir\yoochoose-selected\yoochoose-clicks-selected.dat  .\yoochoose-buys-selected.dat  .\yoochoose-test-selected.dat
        dataset_dir = r'I:\Papers\consumer\codeandpaper\PreprocessData\newfolder\sampling@0.01@partition'
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
        
        feature_dir = r'F:\skyline recommendation\data4\D6_partition' + '\\' + dataset_para+ '\\feature'
        # feature_dir = dataset_dir + r'\feature1'
        # 假如输出文件夹不存在，则创建文件夹
        
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
        print('model: LogisticRegression')
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # print(model)
        # make predictions
        
         
        y_predict = model.predict(X_test)
        # solution：根据预测结果，生成各个session其对应的购买商品
        solution = Solution.generate(test_dic_data, y_predict)
        Evaluate.go(solution, session_item_data)
       
        """
        # 模型训练
        """
        print('model: GaussianNB')
        model = GaussianNB()
        model.fit(X_train, y_train)
        # print(model)
        # make predictions
        y_predict = model.predict(X_test)
        # 结果评估
        solution = Solution.generate(test_dic_data, y_predict)
        Evaluate.go(solution, session_item_data)
        """
    # 模型训练
"""
    print('model: SVM')
    model = SVC()
    model.fit(X_train, y_train)
    # print(model)
    # make predictions
    y_predict = model.predict(X_test)
    # 结果评估
    solution = Solution.generate(test_dic_data, y_predict)
    Evaluate.go(solution, session_item_data)
"""
    # # for regression
    # # setting test dataset score for regression
    # setResgressionScore(y_test, 5, 10)
    # # 模型部分
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_predict = model.predict(X_test)
    # # 结果评估
    # RegreEvaluate.go(y_predict, test_dic_data, session_item_data, session_idx_dic)


def setResgressionScore(lable, s0, s1):
    for i in range(len(lable)):
        if lable[i] == 0:
            lable[i] = s0
        else:
            lable[i] = s1


if __name__ == '__main__':
    classifier_test()

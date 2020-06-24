from input import Input
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

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

# 综合了时间类特征和新特征的分类和回归实验
# 需要设置的路径：
# dataset_dir （实验数据）train\session_item.txt  test\session_item.txt
# yoochoose_data_dir （yoochoose-data）yoochoose_data_dir\yoochoose-clicks.dat  .\yoochoose-buys.dat  .\yoochoose-test.dat
# feature_dir 提取特征结果所在文件夹

def classifier_test():
    #[5,6,9,12,24,28,33,39,40,49]#
    L = [18,19,25,28,34,38,39,40,45,47];
    for i in L:
        

        """
        D1：
        2,8,18,19,22,28,36,40,44,49
        
        D2：
        3，7，12，18，21，35，37，44，46，49
        
        D3:
        1,6,8,9,11,24,33,45,47,50
        
        D4:
        3,4,5,18,20,25,29,39,45,49

        """
        #if not (i==3 or i==4 or i==5 or i==18 or i==20 or i==25 or i==29 or i==39 or i==45 or i==49):
        #     continue        
        print(i)
        # setting
        dataset_para = 'sampling@x@'+str(i)+'@partition'
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
        dataset_dir = r'F:\skyline recommendation\data4\D5_partition' + '\\' + dataset_para
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
        
        feature_dir = r'F:\skyline recommendation\data4\D5_partition' + '\\' + dataset_para+ '\\feature'
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
        print('model: GradientBoostingClassifier')
        model = LogisticRegression()  # LogisticRegression()
        model.fit(X_train, y_train)
        
        #取第二列，即类为1的分数
        score = model.predict_proba(X_test)[:,1]
         
        #y_predict = model.predict(X_test)
        # solution：根据预测结果，生成各个session其对应的购买商品
        #solution = Solution.generate(test_dic_data, y_predict)
        session_item_score_dic = extract_score_by_session2(score, test_dic_data)
        #print("test**************************************")
        p,MRR = recommendation1.evaluate(session_item_data, session_item_score_dic)
        # p1 = calc_precision_at_1(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # p2 = calc_precision_at_2(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # MRR = calc_MRR(session_score_dic_data, session_item_dic_data, session_item_data, session_idx_dic)
        # print('p1: ' + ('%.4f' % p1))
        # print('p2: ' + ('%.4f' % p2))
        # print('MRR: ' + ('%.4f' % MRR))
        #print precision
        f = open("F:\\skyline recommendation\\data4\\D5_partition\\D5_MRR.csv","a")
        
        writer = csv.writer(f)
    
        writer.writerow([i,('%.4f' % p),('%.4f' % MRR)])
        f.close()
        # 以上，只初始化一次，后面再次跑代码，结果是追加在上一次的结果之上的，

        """
        print type(solution[0])
        print solution[0]
        Evaluate.go(solution, session_item_data)
        """ 
        #break
        
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
    classifier_test()

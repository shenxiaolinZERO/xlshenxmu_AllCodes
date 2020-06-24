from input import Input
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from solution import Solution
from evaluate import Evaluate


def classifier_test():

    """
    dataset_para = 'sampling@0.01@partition'

    train_file_path = r'E:\recsyschallenge2015\mycode\result-data\click-buy-train.arff'
    X_train, y_train = Input.read_train(train_file_path)
    test_file_path = r'E:\recsyschallenge2015\mycode\result-data\click-buy-test-BR.txt'
    groundtruth_path = r'E:\recsyschallenge2015\mycode\ranking aggregation\classification\data' + '\\' + dataset_para + r'\test\session_item.txt'
    
    X_test, y_test, test_dic_data, session_item_data, session_idx_dic = Input.read_test(test_file_path, groundtruth_path)
    """
    dataset_para = 'sampling@0.01@1@partition'

    train_file_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\feature_time\click-buy-train.arff'
    X_train, y_train = Input.read_train(train_file_path)
    test_file_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\feature_time\click-buy-test-BR.txt'
    groundtruth_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\test\session_item.txt'
    X_test, y_test, test_dic_data, session_item_data, session_idx_dic = Input.read_test(test_file_path, groundtruth_path)

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
    # print('model: SVM')
    # model = SVC()
    # model.fit(X_train, y_train)
    # # print(model)
    # # make predictions
    # y_predict = model.predict(X_test)
    # # 结果评估
    # solution = Solution.generate(test_dic_data, y_predict)
    # Evaluate.go(solution, groundtruth_path)

if __name__ == '__main__':
    classifier_test()

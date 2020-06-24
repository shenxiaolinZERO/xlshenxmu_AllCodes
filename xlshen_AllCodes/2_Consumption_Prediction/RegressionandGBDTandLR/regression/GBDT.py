#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
# import read_from_file as rff
# from matplotlib import pyplot
# import copy
from input import Input
from evaluate import Evaluate


def GBDT(X_train, y_train, X_test, y_test):
    # X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
    # # X_train, X_test = X[:200], X[200:]
    # # y_train, y_test = y[:200], y[200:]
    # X_train = X[:200]
    # y_train = y[:200]
    # ls1 = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152]
    # ls2 = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152]
    # ls = [ls1, ls2]
    # X_test = array(ls)
    # y_test = array([10.1, 8.9])

    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
    # print(mean_squared_error(y_test, est.predict(X_test)))
    return est.predict(X_test)


def GBDT_test():

    dataset_para = 'sampling@0.01@1@partition'

    train_file_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\feature_time\click-buy-train.arff'
    X_train, y_train = Input.read_train(train_file_path)
    test_file_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\feature_time\click-buy-test-BR.txt'
    groundtruth_path = r'F:\skyline recommendation' + '\\' + dataset_para + r'\test\session_item.txt'
    X_test, y_test, test_dic_data, session_item_data, session_idx_dic = Input.read_test(test_file_path, groundtruth_path)
    y_predict = GBDT(X_train, y_train, X_test, y_test)
    # print(y_test, y_predict)
    # y = range(0, len(y_test), 1)
    # pyplot.plot(y, y_test, 'r', y, y_predict)
    # pyplot.show()
    Evaluate.go(y_predict, test_dic_data, session_item_data, session_idx_dic)


if __name__ == '__main__':
    GBDT_test()

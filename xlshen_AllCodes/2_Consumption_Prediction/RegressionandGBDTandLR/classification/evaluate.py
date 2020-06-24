#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_from_file as rff


class Evaluate:

    @staticmethod
    def go(solution, session_item_data):

        session_idx_dic = dict()
        extract_session(session_item_data, session_idx_dic)
        # print(session_item_data)
        # print(solution)
        p1 = calc_precision_at_1(session_item_data, session_idx_dic, solution)
        p2 = calc_precision_at_2(session_item_data, session_idx_dic, solution)
        #precision = calc_precision(session_item_data, session_idx_dic, solution)
        #recall = calc_recall(session_item_data, session_idx_dic, solution)
        print('precision@1: ' + ('%.4f' % p1))
        print('precision@2: ' + ('%.4f' % p2))
        print(('%.4f' % p1)+"\t"+('%.4f' % p2))
        # print('precision: ' + str(precision))
        # print('recall: ' + str(recall))


def extract_session(session_item_data, session_idx_dic):
    idx = 0
    for cur_list in session_item_data:
        session = cur_list[0]
        session_idx_dic[session] = idx
        idx += 1


# calculate precision@1
def calc_precision_at_1(session_item_data, session_idx_dic, solution):
    p1 = 0.0
    session_len = len(session_item_data)
    session_idx_dic_keys = session_idx_dic.keys()
    for cur_solution in solution:
        session = cur_solution[0]
        if session in session_idx_dic_keys:
            idx = session_idx_dic[session]
            groundtruth_buy_items = session_item_data[idx][1]
            solution_item1 = cur_solution[1][0]
            if solution_item1 in groundtruth_buy_items:
                p1 += 1.0
    # print(r'precision@1/len(solution): ' + str(p1/len(solution)))
    p1 /= session_len
    return p1


# calculate precision@2
def calc_precision_at_2(session_item_data, session_idx_dic, solution):
    p2 = 0.0
    session_len = len(session_item_data)
    session_idx_dic_keys = session_idx_dic.keys()
    for cur_solution in solution:
        session = cur_solution[0]
        if session in session_idx_dic_keys:
            idx = session_idx_dic[session]
            groundtruth_buy_items = session_item_data[idx][1]
            solution_items = cur_solution[1]
            solution_item1 = solution_items[0]
            if solution_item1 in groundtruth_buy_items:
                p2 += 0.5
            if len(solution_items) >= 2:
                solution_item2 = solution_items[1]
                if solution_item2 in groundtruth_buy_items:
                    p2 += 0.5
    # print(r'precision@2/len(solution): ' + str(p2 / len(solution)))
    p2 /= session_len
    return p2


# calculate precision
def calc_precision(session_item_data, session_idx_dic, solution):
    precision = 0.0
    session_len = len(session_item_data)
    session_idx_dic_keys = session_idx_dic.keys()
    for cur_solution in solution:
        session = cur_solution[0]
        if session in session_idx_dic_keys:
            idx = session_idx_dic[session]
            groundtruth_buy_items = session_item_data[idx][1]
            solution_items = cur_solution[1]
            solution_items_len = len(solution_items)
            for solution_item in solution_items:
                if solution_item in groundtruth_buy_items:
                    precision += 1.0/solution_items_len
    # print(r'precision/len(solution): ' + str(precision / len(solution)))
    precision /= session_len
    return precision


# calculate recall
def calc_recall(session_item_data, session_idx_dic, solution):
    recall = 0.0
    session_len = len(session_item_data)
    session_idx_dic_keys = session_idx_dic.keys()
    for cur_solution in solution:
        session = cur_solution[0]
        if session in session_idx_dic_keys:
            idx = session_idx_dic[session]
            groundtruth_buy_items = session_item_data[idx][1]
            groundtruth_buy_items_len = len(groundtruth_buy_items)
            solution_items = cur_solution[1]
            for solution_item in solution_items:
                if solution_item in groundtruth_buy_items:
                    recall += 1.0 / groundtruth_buy_items_len
    # print(r'recall/len(solution): ' + str(recall / len(solution)))
    recall /= session_len
    return recall


# 评估分类实验效果
def manual_evaluate():
    groundtruth_path = r'E:\recsyschallenge2015\mycode\ranking aggregation\classification\data\sampling@0.001\ranking aggregation\test\session_item.txt'
    solution_file = r'E:\recsyschallenge2015\mycode\result-data\solution.dat'

    session_item_data = rff.get_data_lists(groundtruth_path)
    session_idx_dic = dict()
    extract_session(session_item_data, session_idx_dic)
    solution = rff.get_solution(solution_file)
    # print(session_item_data)
    # print(solution)
    p1 = calc_precision_at_1(session_item_data, session_idx_dic, solution)
    p2 = calc_precision_at_2(session_item_data, session_idx_dic, solution)
    precision = calc_precision(session_item_data, session_idx_dic, solution)
    recall = calc_recall(session_item_data, session_idx_dic, solution)
    print('precision@1: ' + str(p1))
    print('precision@2: ' + str(p2))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))


if __name__ == '__main__':
    manual_evaluate()


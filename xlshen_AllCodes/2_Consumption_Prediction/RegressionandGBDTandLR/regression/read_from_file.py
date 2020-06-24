# -*- coding: utf-8 -*-
# 读取文件中的信息
# 输入文件格式每行如： 0;1,2,3;4,5,6;   (1,2,3或4,5,6可以为空)
# 输出格式如： 每行对应一个list: [0,[1,2,3],[4,5,6]]   所有行的数据append后构成一个"data lists"
# 适用于读取session_item_data和item_session_data
def get_data_lists(file_path):
    data_lists = list()
    f = open(file_path)
    try:
        for line in f:
            if line == '\n':
                print(r"warning: line == '\n'")
                continue
            # strip('\n')去掉每行行末换行符
            line = line.strip('\n')
            tmp = line.split(';')
            # x 为当前行第一个值
            x_str = tmp[0]
            x = int(x_str)
            # 初始化当前行的数据
            cur_list = [x, [], []]
            x_list1 = cur_list[1]
            x_list2 = cur_list[2]
            # tmp[1] 为当前行第二个值
            if tmp[1] != "":
                tmp1tmp = tmp[1].split(',')
                for elem_str in tmp1tmp:
                    elem = int(elem_str)
                    x_list1.append(elem)
            # tmp[2] 为当前行第三个值
            if tmp[2] != "":
                tmp2tmp = tmp[2].split(',')
                for elem_str in tmp2tmp:
                    elem = int(elem_str)
                    x_list2.append(elem)
            data_lists.append(cur_list)
    except Exception as e:
        print(e)
    finally:
        f.close()
    return data_lists


# 读取文件中的信息
# 输入文件格式每行如： 0;1,2,3
# 输出格式如： 每行对应一个list: [0,[1,2,3]]   所有行的数据append后构成一个"solution"
# 适用于读取solution.dat
def get_solution(file_path):
    solution = list()
    file = open(file_path)
    try:
        for line in file:
            if line == '\n':
                print(r"line == '\n']")
                continue
            # strip('\n')去掉每行行末换行符
            line = line.strip('\n')
            tmp = line.split(';')
            # x 为当前行第一个值
            x_str = tmp[0]
            x = int(x_str)
            cur_solution = [x, []]
            x_list1 = cur_solution[1]
            tmp1tmp = tmp[1].split(',')
            for elem_str in tmp1tmp:
                elem = int(elem_str)
                x_list1.append(elem)
            solution.append(cur_solution)
    except Exception as e:
        print(e)
    finally:
        file.close()
    return solution

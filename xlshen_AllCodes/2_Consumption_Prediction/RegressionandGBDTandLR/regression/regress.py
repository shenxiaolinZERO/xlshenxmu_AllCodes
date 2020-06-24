from input import Input
from evaluate import Evaluate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot
import csv

def regress_test():

    L = [5,6,9,12,24,28,33,39,40,49]#[18,19,25,28,34,38,39,40,45,47];
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
        D5
        18,19,25,28,34,38,39,40,45,47
        D6:
        5,6,9,12,24,28,33,39,40,49

        """
        #if not (i==2 or i==8 or i==18 or i==19 or i==22 or i==28 or i==36 or i==40 or i==44 or i==49):
        #    continue
        print(i)
        dataset_para = 'sampling@x@'+str(i)+'@partition'
    
        train_file_path = r'F:\skyline recommendation\data4\D6_partition' + '\\' + dataset_para + r'\feature\click-buy-train.arff'
        X_train, y_train = Input.read_train(train_file_path)
        test_file_path = r'F:\skyline recommendation\data4\D6_partition' + '\\' + dataset_para + r'\feature\click-buy-test-BR.txt'
        groundtruth_path = r'F:\skyline recommendation\data4\D6_partition' + '\\' + dataset_para + r'\test\session_item.txt'
        X_test, y_test, test_dic_data, session_item_data, session_idx_dic = Input.read_test(test_file_path, groundtruth_path)
    
        # 模型部分
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        # 结果评估
        p1,MRR1 = Evaluate.go(y_predict, test_dic_data, session_item_data)#, session_idx_dic
    
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        # 结果评估
        p2,MRR2 = Evaluate.go(y_predict, test_dic_data, session_item_data)#, session_idx_dic
        
        
        f = open("F:\\skyline recommendation\\data4\\D6_partition\\D6_result.csv","a")
        
        writer = csv.writer(f)
    
        writer.writerow([i,('%.4f' % p1),('%.4f' % MRR1),('%.4f' % p2),('%.4f' % MRR2)])
        f.close()
        print ("************************")
        # 模型部分
        # model = BayesianRidge()
        # model.fit(X_train, y_train)
        # y_predict = model.predict(X_test)
        # 结果评估
        # Evaluate.go(y_predict, test_dic_data, session_item_data, session_idx_dic)
        # 画图——似然迭代过程
        # y = range(0, len(y_test), 1)
        # pyplot.plot(y, y_test, 'r.', y, y_predict, 'go')
        # pyplot.show()

if __name__ == '__main__':
    regress_test()

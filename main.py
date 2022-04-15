from scipy.io import loadmat
import time
import sklearn
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
import xlrd
import random
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import numpy as np
import copy
from scipy import interpolate
import pandas as pd
import pylab as pl
'''
读取数据，随机，设置样本集，各种方法插值，对比error
'''

def save_as_csvfile(data, file_name,classname, headers,index=None):
    save_temp_data = pd.DataFrame(data=data)
    save_temp_data_name = './data/'+ file_name +'_' + classname + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk', header=headers,index=index)

def chazhi(data, times, random_list, kind):
    x = np.linspace(0, times-1, times)
    x = np.delete(x,random_list)
    x_new = random_list
    y = np.delete(data, random_list)
    #for kind in ["nearest", "zero", "slinear", "quadratic", "cubic"]:  # 插值方式
    # "nearest","zero"为阶梯插值
    # slinear 线性插值
    # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    f = interpolate.interp1d(x, y, kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    y_new = f(x_new)
    return y_new


def now_chazhi(data, times, random_list):
    x = np.linspace(0,times-1,times)
    x = np.delete(x,random_list)
    x_new = random_list
    y_old = data[random_list]
    y = np.delete(data, random_list)
    y_new = []
    error = []
    used_times = []
    for kind in ["slinear", "quadratic","cubic"]:  # 插值方式
        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        start_time = time.clock()
        f = interpolate.interp1d(x, y, kind=kind)
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        ynew = f(x_new)
        end_time = time.clock()
        used_time = abs((start_time - end_time)/random_list.shape[0]) *1000
        used_times.append(used_time)
        y_new.append(ynew)
        # bug1 = abs(y_old - ynew)
        # bug2 = bug1 / y_old
        # bug3 = np.sum(bug2)
        # print(bug3)
        error_rmse = sqrt(sklearn.metrics.mean_squared_error(y_old, ynew))
        error_mae = sklearn.metrics.mean_absolute_error(y_old, ynew)
        error_r2 = sklearn.metrics.r2_score(y_old, ynew)
        error.append([error_rmse, error_mae, error_r2])
    return y_new, error, used_times


def multi_chazhi(input_data, Y_location, times, random_list, kind, use_pca=False, use_kernelpca=False):
    data_corr = pd.DataFrame(input_data)
    data_corr = data_corr.corr()
    data_corr = data_corr.values
    X_Y_corr = data_corr[:, Y_location]
    line_corr = []
    line_index = []
    unline_corr = []
    unline_index = []
    used_times = []
    error = []
    start_time = time.clock()
    for i,corr in enumerate(X_Y_corr):
        if abs(X_Y_corr[i]) >0.5:
            line_corr.append(X_Y_corr[i])
            line_index.append(i)
        else:
            unline_corr.append(X_Y_corr[i])
            unline_index.append(i)
    Y_data = input_data[:, Y_location]
    unline_data = input_data[:, unline_index]
    X_data = copy.deepcopy(input_data)
    X_data[:, Y_location] = 0
    line_data = X_data[:, line_index]
    if use_pca == True and use_kernelpca == False:
        print('用pca')
        pca = PCA(n_components=1)
        pca.fit(unline_data)
        unline_data_pca = pca.transform(unline_data)
        A = np.linalg.pinv(unline_data)
        temp = np.matmul(A, unline_data)
        unline_corr = np.matmul(unline_corr, temp)
        unline_corr = np.mean(unline_corr)
        line_unline_data = np.hstack((line_data, unline_data_pca))
        line_unline_corr = np.hstack((line_corr, unline_corr))
        ys = np.matmul(line_unline_data, line_unline_corr)
    elif use_kernelpca == True:
        print('用核pca')
        kernalpca = KernelPCA(n_components=1, kernel='sigmoid', gamma=0.1)
        kernalpca.fit(unline_data)
        unline_data_pca = kernalpca.transform(unline_data)
        A = np.linalg.pinv(unline_data)
        temp = np.matmul(A, unline_data_pca)
        unline_corr = np.matmul(unline_corr, temp)
        line_unline_data = np.hstack((line_data, unline_data_pca))
        line_unline_corr = np.hstack((line_corr, unline_corr))
        ys = np.matmul(line_unline_data, line_unline_corr)
    else:
        ys = np.matmul(X_data, X_Y_corr)
    ya1 = chazhi(Y_data, times, random_list, kind)
    ya3 = chazhi(ys, times, random_list, kind)
    ya2 = ys[random_list]
   # print('ya3与ya2的插值误差：', np.sum(abs((ya2 - ya3)/ya2)))
    ya = ya1 - ((ya3 - ya2) / ya3) * ya1
    end_time = time.clock()
    use_time = abs((start_time - end_time)/random_list.shape[0]) * 1000
    used_times.append(use_time)
    y_old = Y_data[random_list]
    error_rmse = sqrt(sklearn.metrics.mean_squared_error(y_old, ya))
    error_mae = sklearn.metrics.mean_absolute_error(y_old, ya)
    error_r2 = sklearn.metrics.r2_score(y_old, ya)
    error.append([error_rmse, error_mae, error_r2])
    #print(error)
    return ya, error, used_times


def machinelearning_chazhi(data, Y_location, times, random_list, kind='random_forest'):
    used_times = []
    error = []
    test_data = data[random_list, :]
    test_y = test_data[:, Y_location]
    test_x = np.delete(test_data, Y_location, axis=1)
    train_data = np.delete(data, random_list, axis=0)
    train_y = train_data[:, Y_location]
    train_x = np.delete(train_data, Y_location, axis=1)
    start_time = time.clock()
    if kind == 'random_forest':
        rfr = RandomForestRegressor()
        # rfr = BayesianRidge()
    elif kind == 'adaboost':
        rfr = AdaBoostRegressor()
    elif kind == 'svr':
        rfr = SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                  tol=0.001, C=1.0, epsilon=0.1)
    else:
        pass
    rfr.fit(train_x, train_y)
    predict_y = rfr.predict(test_x)
    end_time = time.clock()

    use_time = abs((start_time - end_time) / random_list.shape[0]) * 1000
    used_times.append(use_time)
    error_rmse = sqrt(sklearn.metrics.mean_squared_error(test_y, predict_y))
    error_mae = sklearn.metrics.mean_absolute_error(test_y, predict_y)
    error_r2 = sklearn.metrics.r2_score(test_y, predict_y)

    error.append([error_rmse, error_mae, error_r2])


    return predict_y, error, used_times





def main(Y_location=12, missing_rate=0.05):
    data_old = loadmat('测试.mat')
    data = data_old['data']


    #leixing_number = data.shape[1]
    times = data.shape[0]
    #随机
    #random_list = random.randint(0,leixing_number-1)
    # 随机丢弃
    random_list = list(sorted(set(np.random.randint(0, int((times-1)/2), size=int(times * missing_rate)))))
    # 为了展示而丢弃
    # n_max = 100
    # n_min = 0
    # random_list = list(sorted(set(np.random.randint(n_min, n_max, size=int(n_max * 0.5)))))
    # random_list = []
    random_list = np.array(random_list)
    random_list = 2 * random_list

    print("随机丢弃的位置为：", random_list)
    print("随机丢弃的个数为：", random_list.shape[0])

    # 机器学习插值
    error = machinelearning_chazhi(data, Y_location, times, random_list)

    #原始插值法
    y_old_method, error_old_method, used_times_old_method = now_chazhi(data[:,Y_location], times, random_list)
    y_old_method = np.array(y_old_method)
    y_old_method = np.transpose(y_old_method)
    print('原始插值方法的误差率：')
    print(error_old_method)
    print('原始方法所需时间：')
    print(used_times_old_method)

    ##多元插值方法
    kind = 'slinear'  # "nearest", "zero", "slinear", "quadratic", "cubic"
    y_multi, error_multi, used_times_multi = multi_chazhi(data, Y_location, times, random_list, kind, use_pca=False, use_kernelpca=False)
    print('多元插值方法的误差率：')
    print(error_multi)
    print('多元插值方法所需时间：')
    print(used_times_multi)

    ## 主成分降维
    kind = 'slinear'  # "nearest", "zero", "slinear", "quadratic", "cubic"
    y_multi_pca, error_multi_pca, used_times_multi_pca = multi_chazhi(data, Y_location, times, random_list, kind, use_pca=True)
    print('非线性空间映射的插值方法的误差率：')
    print(error_multi_pca)
    print('非线性空间映射的插值方法所需时间：')
    print(used_times_multi_pca)

    ## 核主成分降维
    kind = 'slinear'  # "nearest", "zero", "slinear", "quadratic", "cubic"
    y_multi_kernel_pca, error_multi_kernel_pca, used_times_multi_kernel_pca = multi_chazhi(data, Y_location, times, random_list, kind, use_pca=True,use_kernelpca=True)
    print('核函数映射的插值方法的误差率：')
    print(error_multi_kernel_pca)
    print('核函数映射的插值方法所需时间：')
    print(used_times_multi_kernel_pca)


    # 机器学习插值
    y_machine_RF, error_machine_RF, used_times_machine_RF = \
        machinelearning_chazhi(data, Y_location, times, random_list, kind='random_forest')
    print('Random Forest的误差率：')
    print(error_machine_RF)
    print('Random Forest所需时间：')
    print(used_times_machine_RF)

    y_machine_ada, error_machine_ada, used_times_machine_ada = \
        machinelearning_chazhi(data, Y_location, times, random_list, kind='adaboost')
    print('adaboost的误差率：')
    print(error_machine_ada)
    print('adaboost所需时间：')
    print(used_times_machine_ada)

    y_machine_svr, error_machine_svr, used_times_machine_svr = \
        machinelearning_chazhi(data, Y_location, times, random_list, kind='svr')
    print('svr的误差率：')
    print(error_machine_svr)
    print('svr所需时间：')
    print(used_times_machine_svr)


    # 保存数据数值
    y_old = data[random_list, Y_location]
    headers = ['location','原始数据','slinear','quadratic','cubic','多元插值','主成分分析法','非线性空间映射','Random Forest','adaboost', 'SVR']
    save_data_finally = np.column_stack((random_list, y_old, y_old_method, y_multi, y_multi_pca, y_multi_kernel_pca, y_machine_RF, y_machine_ada, y_machine_svr))
    save_as_csvfile(save_data_finally, str(missing_rate)+'插值结果', str(Y_location), headers)

    # 保存运行时间和误差
    headers = [ 'RMSE', 'MAE', 'R2', '用时']
    index = ['slinear', 'quadratic', 'cubic', '多元插值', '主成分分析法', '非线性空间映射', 'Random Forest']
    error = np.array(error_old_method)
    error = np.append(error, error_multi, axis=0)
    error = np.append(error, error_multi_pca, axis=0)
    error = np.append(error, error_multi_kernel_pca, axis=0)
    error = np.append(error, error_machine_RF, axis=0)
    error = np.append(error, error_machine_ada, axis=0)
    error = np.append(error, error_machine_svr, axis=0)
    used_times = np.array(used_times_old_method)
    used_times = np.append(used_times, used_times_multi, axis=0)
    used_times = np.append(used_times, used_times_multi_pca, axis=0)
    used_times = np.append(used_times, used_times_multi_kernel_pca, axis=0)
    used_times = np.append(used_times, used_times_machine_RF, axis=0)
    used_times = np.append(used_times, used_times_machine_ada, axis=0)
    used_times = np.append(used_times, used_times_machine_svr, axis=0)
    save_error_time_finally = np.column_stack((error, used_times))
    save_as_csvfile(save_error_time_finally, str(missing_rate) + '误差和时间', str(Y_location), headers, index=index)


if __name__ == '__main__':
    print('*********开始************')
    for i in range(16, 17):
        for j in range(40, 45, 5):
            missing_rate = j/100
            print('***************Y_location = ', i, '*****************\n*************** missingrate =', missing_rate)
            main(Y_location=i, missing_rate=missing_rate)

    print('*********************结   束*********************')


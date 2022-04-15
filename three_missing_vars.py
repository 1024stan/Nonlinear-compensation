#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from scipy.io import loadmat
import numpy as np
from main import now_chazhi, multi_chazhi,save_as_csvfile
import sklearn
from math import sqrt


def frist_impute(x_data,y_linear_replace,y_true,Y,random_list, times):
    ###
    y_frist_time = []
    # 填补Y1
    kind = 'slinear'  # "nearest", "zero", "slinear", "quadratic", "cubic"
    # 其他维度缺失值的替换
    x_data[random_list[1], Y[1]] = y_linear_replace[1]
    x_data[random_list[2], Y[2]] = y_linear_replace[2]
    y_multi_kernel_pca, error_multi_kernel_pca, used_times_multi_kernel_pca = \
        multi_chazhi(x_data, Y_location=Y[0], times=times,
                     random_list=random_list[0], kind=kind, use_pca=True,use_kernelpca=True)
    # 记录Y1填补后的值
    x_data[random_list[0], Y[0]] = y_multi_kernel_pca
    y_frist_time.append(y_multi_kernel_pca)


    # 填补Y2
    y_multi_kernel_pca, error_multi_kernel_pca, used_times_multi_kernel_pca = \
        multi_chazhi(x_data, Y_location=Y[1], times=times,
                     random_list=random_list[1], kind=kind, use_pca=True, use_kernelpca=True)
    # 记录Y2填补后的值
    x_data[random_list[1], Y[1]] = y_multi_kernel_pca
    y_frist_time.append(y_multi_kernel_pca)



    # 填补Y3
    y_multi_kernel_pca, error_multi_kernel_pca, used_times_multi_kernel_pca = \
        multi_chazhi(x_data, Y_location=Y[2], times=times,
                     random_list=random_list[2], kind=kind, use_pca=True, use_kernelpca=True)
    # 记录Y3填补后的值
    x_data[random_list[2], Y[2]] = y_multi_kernel_pca
    y_frist_time.append(y_multi_kernel_pca)

    # 计算误差
    error_first_time = []
    for i in range(len(Y)):
        error_rmse = sqrt(sklearn.metrics.mean_squared_error(y_true[i], y_frist_time[i]))
        error_mae = sklearn.metrics.mean_absolute_error(y_true[i], y_frist_time[i])
        error_r2 = sklearn.metrics.r2_score(y_true[i], y_frist_time[i])
        error_first_time.append([error_rmse, error_mae, error_r2])



    return y_frist_time, error_first_time

def second_impute(x_data,y_frist_time,y_true,Y,random_list, times):
    y_second_time, error_second_time = frist_impute(x_data=x_data, y_linear_replace=y_frist_time,
                                y_true=y_true,Y=Y, random_list=random_list, times=times)
    return y_second_time, error_second_time








def main(Y, missing_rate=0.05):
    data_old = loadmat('测试.mat')
    data = data_old['data']

    # leixing_number = data.shape[1]
    times = data.shape[0]
    random_list = []
    for i in range(3):
        random_list_i = list(sorted(set(np.random.randint(0, int((times - 1) / 2), size=int(times * missing_rate)))))
        random_list_i = np.array(random_list_i)
        random_list_i = 2 * random_list_i
        random_list.append(random_list_i)

    print("随机丢弃的位置为：", random_list)
    print("随机丢弃的个数为：", len(random_list[0]))
    print("-------------------------------------------------------------------------")

    # 对这三个维度的缺失值进行初始化替代
    print("*****************替代开始**********************")
    y_true = []
    y_linear_replace =[]
    error_old = []
    for i in range(3):
        y_old_method, error_old_method, used_times_old_method = now_chazhi(data[:, Y[i]], times, random_list[i])
        y_linear_replace.append(y_old_method[0])
        print(error_old_method[0])
        y_true.append(data[random_list[i], Y[i]])
        error_old.append(error_old_method[0])
    print("*****************替代结束**********************")
    print("-------------------------------------------------------------------------")

    print("**********第一轮填补***********")
    y_frist_time, error_frist_time = frist_impute(x_data=data, y_linear_replace=y_linear_replace,
                                y_true=y_true, Y=Y, random_list=random_list, times=times)
    print(error_frist_time)
    print("********第一轮填补结束***********")
    print("-------------------------------------------------------------------------")

    print("**********第二轮填补***********")
    y_frist_time, error_second_time = second_impute(x_data=data, y_frist_time=y_frist_time,
                                y_true=y_true, Y=Y, random_list=random_list, times=times)
    print(error_second_time)
    print("********第二轮填补结束***********")
    print("-------------------------------------------------------------------------")

    error_all = error_old
    for i in range(3):
        error_all.append(error_frist_time[i])
    for i in range(3):
        error_all.append(error_second_time[i])
    save_as_csvfile(data=error_all, file_name='error/', classname='error_all', headers=None)










if __name__ == "__main__":
    Y = [14,12, 3]

    main(Y=Y)




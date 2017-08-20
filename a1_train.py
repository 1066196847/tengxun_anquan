# coding=utf-8
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import pickle
import cPickle
from math import ceil
import matplotlib.pyplot as plt
import csv

'''
函数说明：将3000个训练集的train.csv做好
'''
def make_train():
    print('mv to make_train function')
    train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    train.columns = ['id', 'guiji', 'target', 'label']
    train_2_col = train[['id', 'guiji']]

    for i in train_2_col['id'].unique():
        print 'a1_1:',i
        a = train_2_col.loc[train_2_col['id'] == i, 'guiji'][train_2_col.loc[train_2_col['id'] == i, 'guiji'].index[0]]
        a = a.split(';')[:-1]
        b = DataFrame()
        b['guiji'] = DataFrame(a)[0]
        b['id'] = i
        b = b.reset_index(drop=True)
        # 将 guiji 这一列，按照；分成3列
        b['guiji_x'] = b['guiji'].str.split(';').str[0].str.split(',').str[0].astype('int')
        b['guiji_y'] = b['guiji'].str.split(';').str[0].str.split(',').str[1].astype('int')
        b['guiji_t'] = b['guiji'].str.split(';').str[0].str.split(',').str[2].astype('int')
        if(i == 1821):
            b.to_csv('../data/train_feature/train.csv',index=False,mode='a')
        else:
            b.to_csv('../data/train_feature/train.csv', index=False ,header=None,mode='a')



'''
函数说明：假设这个用户的点迹只有2点，X方向就说明没法计算加速度，就执行这个函数
'''
def a1_equals_2_x(data,i,writer):
    '''先计算速度的5个特征'''
    # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
    list_x = list(data['guiji_x'])  # 按照顺序将guiji_x这一列 放在一个list列表里
    list_x_minus_1 = list_x[1:]  # 去掉list_x中第一个元素
    list_x = list_x[:-1]  # 去掉list_x中最后一个元素
    list_x_cha = list(
        DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])  # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
    # t 这一列也同上
    list_t = list(data['guiji_t'])
    list_t_minus_1 = list_t[1:]
    list_t = list_t[:-1]
    list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

    # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
    list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
    for ii in range(0, len(list_v)):
        list_v[ii] = round(list_v[ii], 6)
    v_min = min(list_v)
    v_max = max(list_v)
    v_mean = np.mean(list_v)
    v_variance = np.var(list_v)
    if(v_mean == 0):
        v_coe = np.std(list_v) / 0.3
    else:
        v_coe = np.std(list_v) / v_mean

    '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
    a_min = 0
    a_max = 0
    a_mean = 0
    a_variance = 0
    a_coe = 0
    writer.writerow([i, v_min, v_max, v_mean, v_variance, v_coe, a_min, a_max, a_mean, a_variance, a_coe])

'''
函数说明：假设这个用户的点迹只有1点，X方向就说明没法计算加速度，就执行这个函数
'''
def a1_equals_1_x(i, writer):
    writer.writerow([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

'''
函数说明：假设这个用户的点迹只有2点，Y方向就说明没法计算加速度，就执行这个函数
'''
def a1_equals_2_y(data, i, writer):
    '''先计算速度的5个特征'''
    # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
    list_x = list(data['guiji_y'])  # 按照顺序将guiji_x这一列 放在一个list列表里
    list_x_minus_1 = list_x[1:]      # 去掉list_x中第一个元素
    list_x = list_x[:-1]  # 去掉list_x中最后一个元素
    list_x_cha = list(
        DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])  # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
    # t 这一列也同上
    list_t = list(data['guiji_t'])
    list_t_minus_1 = list_t[1:]
    list_t = list_t[:-1]
    list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

    # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
    list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
    for ii in range(0, len(list_v)):
        list_v[ii] = round(list_v[ii], 6)
    v_min = min(list_v)
    v_max = max(list_v)
    v_mean = np.mean(list_v)
    v_variance = np.var(list_v)
    if(v_mean == 0):
        v_coe = np.std(list_v) / 0.3
    else:
        v_coe = np.std(list_v) / v_mean

    '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
    a_min = 0
    a_max = 0
    a_mean = 0
    a_variance = 0
    a_coe = 0

    writer.writerow([i, v_min, v_max, v_mean, v_variance, v_coe, a_min, a_max, a_mean, a_variance, a_coe])

'''
函数说明：假设这个用户的点迹只有1点，Y方向就说明没法计算加速度，就执行这个函数
'''
def a1_equals_1_y(i,writer):
    writer.writerow([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

'''
函数说明：x和y两个一维坐标下，可以得到一组速度和一组加速度，计算最大最小平均值和方差  变异系数，这是20个特征
'''
def a1():
    # train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    # train.columns = ['id', 'guiji', 'target', 'label']
    # train_2_col = train[['id', 'guiji']]
    # new = DataFrame() # 函数说明里面：放在一个变量里面 中的 变量
    # for i in train_2_col['id'].unique():
    #     print 'a1_1:',i
    #     a = train_2_col.loc[train_2_col['id'] == i, 'guiji'][train_2_col.loc[train_2_col['id'] == i, 'guiji'].index[0]]
    #     a = a.split(';')[:-1]
    #     b = DataFrame()
    #     b['guiji'] = DataFrame(a)[0]
    #     b['id'] = i
    #     new = new.append(b)
    #     new = new.reset_index(drop=True)
    #     # 将 guiji 这一列，按照；分成3列
    #     new['guiji_x'] = new['guiji'].str.split(';').str[0].str.split(',').str[0].astype('int')
    #     new['guiji_y'] = new['guiji'].str.split(';').str[0].str.split(',').str[1].astype('int')
    #     new['guiji_t'] = new['guiji'].str.split(';').str[0].str.split(',').str[2].astype('int')
    #     # # 这时候的格式 是这个样子
    #     #                   guiji    id  guiji_x  guiji_y  guiji_t
    #     # 0          353,2607,349     1      353     2607      349
    #     # 1          367,2607,376     1      367     2607      376
    #     # 2          388,2620,418     1      388     2620      418
    #     # 3          416,2620,442     1      416     2620      442
    #
    # new.to_csv('../data/train_feature/new.csv',index=False)

    new = pd.read_csv('../data/train_feature/train.csv')

    '''先计算x的这些'''
    f = open('../data/train_feature/x_10.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','x_v_min','x_v_max','x_v_mean','x_v_variance','x_v_coe','x_a_min','x_a_max','x_a_mean','x_a_variance','x_a_coe'])

    for i in new['id'].unique():
        print 'a1_2:',i
        data = new[new['id'] == i]
        if(len(data) == 2):
            a1_equals_2_x(data,i, writer)
            continue
        if (len(data) == 1):
            a1_equals_1_x(i, writer)
            continue


        '''先计算速度的5个特征'''
        # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
        list_x = list(data['guiji_x']) # 按照顺序将guiji_x这一列 放在一个list列表里
        list_x_minus_1 = list_x[1:] # 去掉list_x中第一个元素
        list_x = list_x[:-1]        # 去掉list_x中最后一个元素
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0]) # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
        # t 这一列也同上
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
        list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0,len(list_v)):
            list_v[ii] = round(list_v[ii],6)
        v_min = min(list_v)
        v_max = max(list_v)
        v_mean = np.mean(list_v)
        v_variance = np.var(list_v)
        if(v_mean == 0):
            v_coe = np.std(list_v) / 0.3
        else:
            v_coe = np.std(list_v) / v_mean

        '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
        list_x_cha_no_head = list_v[1:]   # 去掉list_x中第一个元素
        list_x_cha_no_tail = list_v[:-1]  # 去掉list_x中最后一个元素

        list_t_cha_no_head = list_t_cha[1:]   # 去掉list_x中第一个元素
        list_t_cha_no_tail = list_t_cha[:-1]  # 去掉list_x中最后一个元素

        # list_a = (DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]) / (   (DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0,0.3)   ) * 0.5
        oo = list( (DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]) )
        ooo = list( ( (DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0,0.3)  * 0.5  ))

        list_a = []
        for ii in range(0,len(oo)):
            list_a.append(oo[ii]*1.0/ooo[ii])

        for ii in range(0,len(list_a)):
            list_a[ii] = round(list_a[ii],6)

        a_min = min(list_a)
        a_max = max(list_a)
        a_mean = np.mean(list_a)
        a_variance = np.var(list_a)
        if(a_mean == 0):
            a_coe = np.std(list_a) / 0.3
        else:
            a_coe = np.std(list_a) / a_mean

        writer.writerow([i,v_min,v_max,v_mean,v_variance,v_coe,a_min,a_max,a_mean,a_variance,a_coe])
    f.close()


    '''再计算y的这些'''
    f = open('../data/train_feature/y_10.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(
        ['id', 'y_v_min', 'y_v_max', 'y_v_mean', 'y_v_variance', 'y_v_coe', 'y_a_min', 'y_a_max', 'y_a_mean',
         'y_a_variance', 'y_a_coe'])

    for i in new['id'].unique():
        print 'a1_3:',i
        data = new[new['id'] == i]
        if (len(data) == 2):
            a1_equals_2_y(data,i, writer)
            continue
        if (len(data) == 1):
            a1_equals_1_y(i, writer)
            continue


        '''先计算速度的5个特征'''
        # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
        list_x = list(data['guiji_y']) # 按照顺序将guiji_x这一列 放在一个list列表里
        list_x_minus_1 = list_x[1:] # 去掉list_x中第一个元素
        list_x = list_x[:-1]        # 去掉list_x中最后一个元素
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0]) # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
        # t 这一列也同上
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
        list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0,len(list_v)):
            list_v[ii] = round(list_v[ii],6)
        v_min = min(list_v)
        v_max = max(list_v)
        v_mean = np.mean(list_v)
        v_variance = np.var(list_v)
        if(v_mean == 0):
            v_coe = np.std(list_v) / 0.3
        else:
            v_coe = np.std(list_v) / v_mean


        '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
        list_x_cha_no_head = list_v[1:]   # 去掉list_x中第一个元素
        list_x_cha_no_tail = list_v[:-1]  # 去掉list_x中最后一个元素

        list_t_cha_no_head = list_t_cha[1:]   # 去掉list_x中第一个元素
        list_t_cha_no_tail = list_t_cha[:-1]  # 去掉list_x中最后一个元素

        #list_a = list( (DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]) / ( (DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0,0.3)  ) * 0.5 )

        oo = list((DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]))
        ooo = list(((DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0, 0.3) * 0.5))

        list_a = []
        for ii in range(0, len(oo)):
            list_a.append(oo[ii] * 1.0 / ooo[ii])


        for ii in range(0,len(list_a)):
            list_a[ii] = round(list_a[ii],6)

        a_min = min(list_a)
        a_max = max(list_a)
        a_mean = np.mean(list_a)
        a_variance = np.var(list_a)
        if(a_mean == 0):
            a_coe = np.std(list_a) / 0.3
        else:
            a_coe = np.std(list_a) / a_mean

        writer.writerow([i,v_min,v_max,v_mean,v_variance,v_coe,a_min,a_max,a_mean,a_variance,a_coe])

    f.close()












'''
函数说明：然后是二维坐标下，可以得到一组速度向量和一组加速度向量。一组速度向量可以先求 极径 的 最大、最小、平均值、方差、变异系数，还有 极角的5个，一共10个
         一组加速度向量也同样是10个！
'''
def a2_v_equals_1(i, writer):
    writer.writerow([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def a2_a_equals_2(i, writer):
    writer.writerow([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def a2_a_equals_1(i, writer):
    writer.writerow([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

import numpy as np

def cart2pol(x, y): # 这个小函数是“直角坐标系”=》“极坐标系”
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def a2():
    # train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    # train.columns = ['id', 'guiji', 'target', 'label']
    # train_2_col = train[['id', 'guiji']]
    # new = DataFrame() # 函数说明里面：放在一个变量里面 中的 变量
    # for i in train_2_col['id'].unique():
    #     # print i
    #     a = train_2_col.loc[train_2_col['id'] == i, 'guiji'][train_2_col.loc[train_2_col['id'] == i, 'guiji'].index[0]]
    #     a = a.split(';')[:-1]
    #     b = DataFrame()
    #     b['guiji'] = DataFrame(a)[0]
    #     b['id'] = i
    #     new = new.append(b)
    # new = new.reset_index(drop=True)
    # # 将 guiji 这一列，按照；分成3列
    # new['guiji_x'] = new['guiji'].str.split(';').str[0].str.split(',').str[0].astype('int')
    # new['guiji_y'] = new['guiji'].str.split(';').str[0].str.split(',').str[1].astype('int')
    # new['guiji_t'] = new['guiji'].str.split(';').str[0].str.split(',').str[2].astype('int')
    # # # 这时候的格式 是这个样子
    # #                   guiji    id  guiji_x  guiji_y  guiji_t
    # # 0          353,2607,349     1      353     2607      349
    # # 1          367,2607,376     1      367     2607      376
    # # 2          388,2620,418     1      388     2620      418
    # # 3          416,2620,442     1      416     2620      442
    new = pd.read_csv('../data/train_feature/train.csv')
    '''先计算"速度向量"的这些'''
    f = open('../data/train_feature/x_y_v_10.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','v_jijing_min','v_jijing_max','v_jijing_mean','v_jijing_variance','v_jijing_coe','v_jingjiao_min',
                     'v_jingjiao_max','v_jingjiao_mean','v_jingjiao_variance','v_jingjiao_coe'])

    for i in new['id'].unique():
        print 'a2:',i
        data = new[new['id'] == i]
        if (len(data) == 1):
            a2_v_equals_1(i, writer)
            continue


        '''先做出速度向量！'''
        # 后 - 前

        list_x = list(data['guiji_x']) # 按照顺序将guiji_x这一列 放在一个list列表里
        list_x_minus_1 = list_x[1:] # 去掉list_x中第一个元素
        list_x = list_x[:-1]        # 去掉list_x中最后一个元素
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0]) # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
        # t 这一列也同上
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])
        # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
        list_v_x = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0,len(list_v_x)):
            list_v_x[ii] = round(list_v_x[ii],6)

        list_x = list(data['guiji_y'])
        list_x_minus_1 = list_x[1:]
        list_x = list_x[:-1]
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])
        list_v_y = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0, len(list_v_y)):
            list_v_y[ii] = round(list_v_y[ii], 6)

        '''上面的 list_v_x list_v_y 一对应合并 就是一个速度向量 '''
        # # 但是也有实际情况就是：list_v_x list_v_y 中含有 nan inf -inf 这3个变量，使得没有办法进行 极坐标系的转换，所以需要先把他们转化成一个数字
        # # 我的选择是：99
        # for i in range(0, len(list_v_x)):
        #     if (list_v_x[i] in set([inf,-inf,nan])):
        #         list_v_x[i] = 99
        # for i in range(0, len(list_v_y)):
        #     if (list_v_y[i] in set([inf,-inf,nan])):
        #         list_v_y[i] = 99

        '''转换为 极坐标系'''
        rho = list(np.zeros(len(list_v_x)))
        phi = list(np.zeros(len(list_v_y)))
        for ii in range(0,len(list_v_x)):
            rho[ii],phi[ii] = cart2pol(list_v_x[ii], list_v_y[ii])

        for ii in range(0,len(rho)):
            rho[ii] = round(rho[ii],6)
        for ii in range(0,len(phi)):
            phi[ii] = round(phi[ii],6)

        v_jijing_min = min(rho)
        v_jijing_max = max(rho)
        v_jijing_mean = np.mean(rho)
        v_jijing_variance = np.var(rho)
        if(v_jijing_mean == 0):
            v_jijing_coe = np.std(rho) / 0.3
        else:
            v_jijing_coe = np.std(rho) / v_jijing_mean

        v_jingjiao_min = min(phi)
        v_jingjiao_max = max(phi)
        v_jingjiao_mean = np.mean(phi)
        v_jingjiao_variance = np.var(phi)
        if(v_jingjiao_mean == 0):
            v_jingjiao_coe = np.std(phi) / 0.3
        else:
            v_jingjiao_coe = np.std(phi) / v_jingjiao_mean

        writer.writerow([i,v_jijing_min, v_jijing_max, v_jijing_mean, v_jijing_variance, v_jijing_coe, v_jingjiao_min,
                         v_jingjiao_max, v_jingjiao_mean, v_jingjiao_variance, v_jingjiao_coe])
    f.close()


    '''再计算"加速度向量"的这些'''
    f = open('../data/train_feature/x_y_a_10.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','a_jijing_min','a_jijing_max','a_jijing_mean','a_jijing_variance','a_jijing_coe','a_jingjiao_min',
                     'a_jingjiao_max','a_jingjiao_mean','a_jingjiao_variance','a_jingjiao_coe'])

    for i in new['id'].unique():
        print 'a2:',i
        data = new[new['id'] == i]
        if(len(data) == 2):
            a2_a_equals_2(i, writer)
            continue
        if (len(data) == 1):
            a2_a_equals_1(i, writer)
            continue


        '''先做出速度向量！'''
        # 后 - 前

        list_x = list(data['guiji_x']) # 按照顺序将guiji_x这一列 放在一个list列表里
        list_x_minus_1 = list_x[1:] # 去掉list_x中第一个元素
        list_x = list_x[:-1]        # 去掉list_x中最后一个元素
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0]) # 这个里面第一个元素是原先的 第2行的guiji_x - 第1行的guiji_x，依次类推
        # t 这一列也同上
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])
        # 我们知道 刀塔x / 刀塔t 就是速度，所以先计算出来速度中的 最大、最小、平均值、方差、变异系数 这5个特征
        list_v_x = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0,len(list_v_x)):
            list_v_x[ii] = round(list_v_x[ii],6)

        list_x = list(data['guiji_y'])
        list_x_minus_1 = list_x[1:]
        list_x = list_x[:-1]
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])
        list_v_y = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0, len(list_v_y)):
            list_v_y[ii] = round(list_v_y[ii], 6)

        # 要将上面的速度向量，变成加速度向量
        list_v_x_nohead = list_v_x[1:]
        list_v_x_notail = list_v_x[:-1]
        list_v_y_nohead = list_v_y[1:]
        list_v_y_notail = list_v_y[:-1]
        list_t_cha_nohead = list_t_cha[1:]
        list_t_cha_notail = list_t_cha[:-1]

        # list_a_x = list( (DataFrame(list_v_x_nohead)[0] - DataFrame(list_v_x_notail)[0]) / (
        #     (DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0,0.3)   )*0.5 )
        oo = list((DataFrame(list_v_x_nohead)[0] - DataFrame(list_v_x_notail)[0]))
        ooo = list(((DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0,0.3)*0.5))
        list_a_x = []
        for ii in range(0, len(oo)):
            list_a_x.append(oo[ii] * 1.0 / ooo[ii])

        # list_a_y = list( (DataFrame(list_v_y_nohead)[0] - DataFrame(list_v_y_notail)[0]) / (
        #     (DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0,0.3)   ) * 0.5)
        oo = list((DataFrame(list_v_y_nohead)[0] - DataFrame(list_v_y_notail)[0]))
        ooo = list(((DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0,0.3) * 0.5))
        list_a_y = []
        for ii in range(0, len(oo)):
            list_a_y.append(oo[ii] * 1.0 / ooo[ii])

        '''转换为 极坐标系'''
        rho = list(np.zeros(len(list_a_x)))
        phi = list(np.zeros(len(list_a_y)))
        for ii in range(0,len(list_a_x)):
            rho[ii],phi[ii] = cart2pol(list_a_x[ii], list_a_y[ii])

        for ii in range(0,len(rho)):
            rho[ii] = round(rho[ii],6)
        for ii in range(0,len(phi)):
            phi[ii] = round(phi[ii],6)

        a_jijing_min = min(rho)
        a_jijing_max = max(rho)
        a_jijing_mean = np.mean(rho)
        a_jijing_variance = np.var(rho)
        if(a_jijing_mean == 0):
            a_jijing_coe = np.std(rho) / 0.3
        else:
            a_jijing_coe = np.std(rho) / a_jijing_mean

        a_jingjiao_min = min(phi)
        a_jingjiao_max = max(phi)
        a_jingjiao_mean = np.mean(phi)
        a_jingjiao_variance = np.var(phi)
        if(a_jingjiao_mean == 0):
            a_jingjiao_coe = np.std(phi) / 0.3
        else:
            a_jingjiao_coe = np.std(phi) / a_jingjiao_mean

        writer.writerow([i, a_jijing_min, a_jijing_max, a_jijing_mean, a_jijing_variance, a_jijing_coe, a_jingjiao_min,
                         a_jingjiao_max, a_jingjiao_mean, a_jingjiao_variance, a_jingjiao_coe])
    f.close()



'''
函数说明：再计算时刻t的间隔情况，最大最小平均值和方差变异系数，加上个数，这是6维特征，先用这36维训练一下看看效果
'''
def a3():
    # train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    # train.columns = ['id', 'guiji', 'target', 'label']
    # train_2_col = train[['id', 'guiji']]
    # new = DataFrame()  # 函数说明里面：放在一个变量里面 中的 变量
    # for i in train_2_col['id'].unique():
    #     # print i
    #     a = train_2_col.loc[train_2_col['id'] == i, 'guiji'][train_2_col.loc[train_2_col['id'] == i, 'guiji'].index[0]]
    #     a = a.split(';')[:-1]
    #     b = DataFrame()
    #     b['guiji'] = DataFrame(a)[0]
    #     b['id'] = i
    #     new = new.append(b)
    # new = new.reset_index(drop=True)
    # # 将 guiji 这一列，按照；分成3列
    # new['guiji_x'] = new['guiji'].str.split(';').str[0].str.split(',').str[0].astype('int')
    # new['guiji_y'] = new['guiji'].str.split(';').str[0].str.split(',').str[1].astype('int')
    # new['guiji_t'] = new['guiji'].str.split(';').str[0].str.split(',').str[2].astype('int')
    # # # 这时候的格式 是这个样子
    # #                   guiji    id  guiji_x  guiji_y  guiji_t
    # # 0          353,2607,349     1      353     2607      349
    # # 1          367,2607,376     1      367     2607      376
    # # 2          388,2620,418     1      388     2620      418
    # # 3          416,2620,442     1      416     2620      442
    new = pd.read_csv('../data/train_feature/train.csv')
    f = open('../data/train_feature/t_6.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id', 't_min', 't_max', 't_mean', 't_variance', 't_coe', 't_len'])

    for i in new['id'].unique():
        print 'a3:',i
        data = new[new['id'] == i]
        if(len(data) == 1):
            writer.writerow([i, 0, 0, 0, 0, 0, 0])
            continue
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        t_min = min(list_t_cha)
        t_max = max(list_t_cha)
        t_mean = np.mean(list_t_cha)
        t_variance = np.var(list_t_cha)
        if(t_mean == 0):
            t_coe = np.std(list_t_cha) / 0.3
        else:
            t_coe = np.std(list_t_cha) / t_mean
        t_len = len(list_t_cha)

        writer.writerow([i, t_min, t_max, t_mean, t_variance, t_coe, t_len])



'''
函数说明：将所有的特征连接在一起
'''
def a4():
    print('mv to a4 function')
    x_10 = pd.read_csv('../data/train_feature/x_10.csv')
    y_10 = pd.read_csv('../data/train_feature/y_10.csv')
    x_y_a_10 = pd.read_csv('../data/train_feature/x_y_a_10.csv')
    x_y_v_10 = pd.read_csv('../data/train_feature/x_y_v_10.csv')
    t_6 = pd.read_csv('../data/train_feature/t_6.csv')

    new = pd.merge(x_10, y_10, on='id', how='inner')
    new = pd.merge(new, x_y_a_10, on='id', how='inner')
    new = pd.merge(new, x_y_v_10, on='id', how='inner')
    new = pd.merge(new, t_6, on='id', how='inner')

    new.to_csv('../data/train_feature/train_new.csv',index=False)


'''
函数说明：再给 new 添加一列 Lable 列
'''
def a5():
    print('mv to a5 function')
    new = pd.read_csv('../data/train_feature/train_new.csv')
    train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    train.columns = ['id', 'guiji', 'target', 'label']
    train = train[['id','label']]
    # label0s2_15242 = pd.read_csv('../data/uesd2/label0s2_15242.csv')
    # label1s2 = pd.read_csv('../data/uesd2/label1s2.csv')
    # train = label0s2_15242.append(label1s2)

    new = pd.merge(new, train, on='id', how='inner')
    for i in new.columns:
        print new[i].isnull().sum()
    new.to_csv('../data/train_feature/train_new_with_label.csv', index=False)


'''
并非特征！不用运行
函数说明：做大熊老师说的那个 最多有多少个 连续的t相等，，我同时在做出来最多有 多少个连续的t递减
'''
def fenxi_1():
    train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    train.columns = ['id', 'guiji', 'target', 'label']
    train_2_col = train[['id', 'guiji']]

    new = DataFrame()  # 函数说明里面：放在一个变量里面 中的 变量
    for i in train_2_col['id'].unique():
        # print i
        a = train_2_col.loc[train_2_col['id'] == i, 'guiji'][train_2_col.loc[train_2_col['id'] == i, 'guiji'].index[0]]
        a = a.split(';')[:-1]
        b = DataFrame()
        b['guiji'] = DataFrame(a)[0]
        b['id'] = i
        new = new.append(b)
    new = new.reset_index(drop=True)

    new['guiji_x'] = new['guiji'].str.split(';').str[0].str.split(',').str[0].astype('int')
    new['guiji_y'] = new['guiji'].str.split(';').str[0].str.split(',').str[1].astype('int')
    new['guiji_t'] = new['guiji'].str.split(';').str[0].str.split(',').str[2].astype('int')

    f = open('../data/train_feature/tongji.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id', 'time_equal_flag', 'time_hou_bigger_qian_flag', 'time_equal_max', 'time_equal_num', 'time_hou_bigger_qian_max',
                     'time_hou_bigger_qian_num'])

    # new = new[new['id'] == 1]

    f_info = open('../data/train_feature/tongji_info.csv', 'ab')
    writer_info = csv.writer(f_info)


    for i in new['id'].unique():
        print i
        writer_info.writerow([i])

        data = new[new['id'] == i]
        if(len(data) == 1):
            writer.writerow([i, 0, 0, 0, 0, 0, 0])
            continue

        list_t_cha = list(data['guiji_t'])

        # 检查下 list_t_cha 中有几个0，就能判断出来有多少个 0，但是大熊老师要的是 最多有几个连续的t相等 ，所以我要计算的是有几个0相连
        # 时间相同的点有多少个，或者后面的点时间反而小于全面的点，这种异常情况有多少个
        time_equal_flag = 0             # 时间相同的的flag
        time_hou_bigger_qian_flag = 0   # 时间后面大于前面的flag

        num_time_equal_list = [] # 存储每一个元素后面有几个连续且相等的元素，假设原列表长度是10，num_time_equal_list的长度就是9
                                 # 然后求 num_time_equal_list 的最大值，就是这个样本中 最多有几个连续的t相等
                                 # num_time_equal_list 中 大于0的数字的个数 就是 时间相同的点有多少个
        for j in range(0,len(list_t_cha)):
            # print j,list_t_cha[j]
            num = 0
            if(j < len(list_t_cha)-2):
                for k in range(j+1,len(list_t_cha)):
                    # print('move to hear')
                    if(list_t_cha[j] == list_t_cha[k]):
                        print j,k
                        print list_t_cha[j],list_t_cha[k]
                        writer_info.writerow([j, k])
                        writer_info.writerow([list_t_cha[j],list_t_cha[k]])
                        time_equal_flag = 1
                        num += 1
                    else:
                        num_time_equal_list.append(num)
                        break
            else:
                for k in range(j+1,len(list_t_cha)):
                    # print('move to hear')
                    if(list_t_cha[j] == list_t_cha[k]):
                        print j, k
                        print list_t_cha[j], list_t_cha[k]
                        writer_info.writerow([j, k])
                        writer_info.writerow([list_t_cha[j], list_t_cha[k]])
                        time_equal_flag = 1
                        num += 1
                        num_time_equal_list.append(num)
                        break
                    else:
                        num_time_equal_list.append(num)
                        break

        num_num_hou_bigger_qian = [] # 存储每一个元素后面有“几个”连续且大于前面的元素，假设原列表长度是10，num_num_hou_bigger_qian的长度就是9
                                     # 然后求 num_num_hou_bigger_qian 的最大值，就是这个样本中 最多有几个连续的 t 后面的小于前面的
                                     # num_num_hou_bigger_qian 中 大于0的数字的个数 就是 后面的点时间反而小于全面的点 有多少个
        for j in range(0,len(list_t_cha)):
            # print j,list_t_cha[j]
            num = 0
            if(j < len(list_t_cha)-2):
                for k in range(j+1,len(list_t_cha)):
                    # print('move to hear')
                    if(list_t_cha[j] > list_t_cha[k]):
                        print j, k
                        print list_t_cha[j], list_t_cha[k]
                        writer_info.writerow([j, k])
                        writer_info.writerow([list_t_cha[j], list_t_cha[k]])
                        time_hou_bigger_qian_flag = 1
                        num += 1

                    else:
                        num_num_hou_bigger_qian.append(num)
                        break
            else:
                for k in range(j+1,len(list_t_cha)):
                    # print('move to hear')
                    if(list_t_cha[j] > list_t_cha[k]):
                        print j, k
                        print list_t_cha[j], list_t_cha[k]
                        writer_info.writerow([j, k])
                        writer_info.writerow([list_t_cha[j], list_t_cha[k]])

                        time_hou_bigger_qian_flag = 1
                        num += 1
                        num_num_hou_bigger_qian.append(num)
                        break
                    else:
                        num_num_hou_bigger_qian.append(num)
                        break


        time_equal_max = max(num_time_equal_list) + 1 # 就是这个样本中 最多有几个点连续相等
        time_equal_num = 0                            # 有多少组时间相同的点，如果某个索引位置的点的数字大于0的话，说明那个索引位置的数字 是一组 连续相等的点
        for ii in num_time_equal_list:
            if(ii>0):
                time_equal_num += 1
        time_hou_bigger_qian_max = max(num_num_hou_bigger_qian)  # 就是这个样本中 最多有几个连续的 t 后面的小于前面的
        time_hou_bigger_qian_num = 0                   # 同上，是有几组？
        for ii in num_num_hou_bigger_qian:
            if(ii>0):
                time_hou_bigger_qian_num += 1

        writer.writerow([i, time_equal_flag, time_hou_bigger_qian_flag, time_equal_max, time_equal_num, time_hou_bigger_qian_max, time_hou_bigger_qian_num])


'''
并非特征！不用运行
函数说明：对上面产生的两个结果 分析下数据情况，然后发送给大熊老师！
'''
def fenxi_1_add():

    tongji = pd.read_csv('../data/train_feature/tongji.csv')
    # >>> len(tongji[tongji['time_equal_flag'] == 1])
    # 463
    # >>> len(tongji[tongji['time_hou_bigger_qian_flag'] == 1])
    # 12
    # >>> tongji['time_equal_max'].max()
    # 9
    # >>> tongji['time_equal_num'].max()
    # 147
    # >>> tongji['time_hou_bigger_qian_max'].max()
    # 64
    # >>> tongji['time_hou_bigger_qian_num'].max()
    # 1

    # 3000条训练集样本中，有463条样本存在有 连续相等的t的轨迹点；这463条样本中，最多的一个样本中有9个点连续相等；
    # 这463条样本中，最多的一个样本中有147组t相等的点（一组中至少含有2个点）
    # 3000条训练集样本中，有12条样本存在后面点的时间 小于 前面点的时间；这12条样本中，最多的一个样本中有64个连续点小于前面的1个点；
    # 这12条样本中，每一条样本都只含有一次后面点的时间 小于 前面点的时间 这种情况

    # 接下来检查下 那463条样本中，有多少label为1 多少0
    a = tongji[tongji['time_equal_flag'] == 1]
    a = DataFrame(a['id'])
    a = a.reset_index(drop=True)
    train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    train.columns = ['id', 'guiji', 'target', 'label']
    train = train[['id', 'label']]
    train = pd.merge(a, train, on='id', how='left')

    print(len(train[train['label'] == 1])) # 463，刚好还都是 正样本

    a = tongji[tongji['time_hou_bigger_qian_flag'] == 1]
    a = DataFrame(a['id'])
    a = a.reset_index(drop=True)
    train = pd.read_csv('../data/train.txt', header=None, sep=' ')
    train.columns = ['id', 'guiji', 'target', 'label']
    train = train[['id', 'label']]
    train = pd.merge(a, train, on='id', how='left')

    print(len(train[train['label'] == 1])) # 12，刚好还也都是 正样本


if __name__ == "__main__":
    make_train()
    a1()
    a2()
    a3()
    a4()
    a5()













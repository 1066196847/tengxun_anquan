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


if __name__ == "__main__":
    make_train()
    a1()
    a2()
    a3()
    a4()
    a5()













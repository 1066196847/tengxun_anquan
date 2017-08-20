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
函数说明：当用户的点迹只有两点的时候，计算X方向 速度的中位数、加速度的中位数
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
    v_median = np.median(list_v)

    '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
    a_median = 0
    writer.writerow([i, v_median,a_median])


'''
函数说明：当用户的点迹只有1点的时候，计算X方向 速度的中位数、加速度的中位数
'''
def a1_equals_1_x(i, writer):
    writer.writerow([i, 0, 0])

'''
函数说明：当用户的点迹只有两点的时候，计算Y方向 速度的中位数、加速度的中位数
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
    v_median = np.median(list_v)

    '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
    a_median = 0

    writer.writerow([i, v_median, a_median])

'''
函数说明：当用户的点迹只有1点的时候，计算Y方向 速度的中位数、加速度的中位数
'''
def a1_equals_1_y(i,writer):
    writer.writerow([i, 0, 0])

'''
函数说明：x和y两个一维坐标下，计算 速度的中位数、加速度的中位数
'''
def a1():
    new = pd.read_csv('../data/train_feature/train.csv')

    '''先计算x的这些'''
    f = open('../data/train_feature/x_10_add.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','x_v_median','x_a_median'])

    for i in new['id'].unique():
        print 'a1_1:',i
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
        v_median = np.median(list_v)


        '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
        list_x_cha_no_head = list_v[1:]   # 去掉list_x中第一个元素
        list_x_cha_no_tail = list_v[:-1]  # 去掉list_x中最后一个元素

        list_t_cha_no_head = list_t_cha[1:]   # 去掉list_x中第一个元素
        list_t_cha_no_tail = list_t_cha[:-1]  # 去掉list_x中最后一个元素


        # list_a = (DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]) / (   (DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0,0.3)   ) * 0.5
        oo = list((DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]))
        ooo = list(((DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0, 0.3) * 0.5))

        list_a = []
        for ii in range(0, len(oo)):
            list_a.append(oo[ii] * 1.0 / ooo[ii])

        for ii in range(0,len(list_a)):
            list_a[ii] = round(list_a[ii],6)

        a_median = np.median(list_a)


        writer.writerow([i,v_median,a_median])
    f.close()


    '''再计算y的这些'''
    f = open('../data/train_feature/y_10_add.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(
        ['id', 'y_v_median', 'y_a_median'])

    for i in new['id'].unique():
        print 'a1_2:',i
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
        v_median = np.median(list_v)


        '''上面做出来了速度的5个特征，下面做出来 加速度的5个特征'''
        list_x_cha_no_head = list_v[1:]   # 去掉list_x中第一个元素
        list_x_cha_no_tail = list_v[:-1]  # 去掉list_x中最后一个元素

        list_t_cha_no_head = list_t_cha[1:]   # 去掉list_x中第一个元素
        list_t_cha_no_tail = list_t_cha[:-1]  # 去掉list_x中最后一个元素

        # list_a = list( (DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]) / ( (DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0,0.3)  ) * 0.5 )

        oo = list((DataFrame(list_x_cha_no_head)[0] - DataFrame(list_x_cha_no_tail)[0]))
        ooo = list(((DataFrame(list_t_cha_no_head)[0] + DataFrame(list_t_cha_no_tail)[0]).replace(0, 0.3) * 0.5))

        list_a = []
        for ii in range(0, len(oo)):
            list_a.append(oo[ii] * 1.0 / ooo[ii])

        for ii in range(0,len(list_a)):
            list_a[ii] = round(list_a[ii],6)

        a_median = np.median(list_a)

        writer.writerow([i,v_median,a_median])

    f.close()












'''
函数说明：然后是二维坐标下，计算 速度的中位数、加速度的中位数
'''
def a2_v_equals_1(i, writer):
    writer.writerow([i, 0, 0])

def a2_a_equals_2(i, writer):
    writer.writerow([i, 0, 0])

def a2_a_equals_1(i, writer):
    writer.writerow([i, 0, 0])

import numpy as np

def cart2pol(x, y): # 这个小函数是“直角坐标系”=》“极坐标系”
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def a2():
    print('mv to a2 function')
    new = pd.read_csv('../data/train_feature/train.csv')
    '''先计算"速度向量"的这些'''
    f = open('../data/train_feature/x_y_v_10_add.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','v_jijing_median','v_jingjiao_median'])

    for i in new['id'].unique():
        print 'a2_1:',i
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


        '''转换为 极坐标系'''
        rho = list(np.zeros(len(list_v_x)))
        phi = list(np.zeros(len(list_v_y)))
        for ii in range(0,len(list_v_x)):
            rho[ii],phi[ii] = cart2pol(list_v_x[ii], list_v_y[ii])

        for ii in range(0,len(rho)):
            rho[ii] = round(rho[ii],6)
        for ii in range(0,len(phi)):
            phi[ii] = round(phi[ii],6)

        v_jijing_median = np.median(rho)

        v_jingjiao_median = np.median(phi)

        writer.writerow([i,v_jijing_median, v_jingjiao_median])
    f.close()


    '''再计算"加速度向量"的这些'''
    f = open('../data/train_feature/x_y_a_10_add.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','a_jijing_median','a_jingjiao_median'])

    for i in new['id'].unique():
        print 'a2_2:',i
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
        ooo = list(((DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0, 0.3) * 0.5))
        list_a_x = []
        for ii in range(0, len(oo)):
            list_a_x.append(oo[ii] * 1.0 / ooo[ii])

        # list_a_y = list( (DataFrame(list_v_y_nohead)[0] - DataFrame(list_v_y_notail)[0]) / (
        #     (DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0,0.3)   ) * 0.5)
        oo = list((DataFrame(list_v_y_nohead)[0] - DataFrame(list_v_y_notail)[0]))
        ooo = list(((DataFrame(list_t_cha_nohead)[0] + DataFrame(list_t_cha_notail)[0]).replace(0, 0.3) * 0.5))
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

        a_jijing_median = np.median(rho)

        a_jingjiao_median = np.median(phi)

        writer.writerow([i, a_jijing_median, a_jingjiao_median])
    f.close()



'''
函数说明：再计算时刻t的中位数
'''
def a3():
    new = pd.read_csv('../data/train_feature/train.csv')
    f = open('../data/train_feature/t_6_add.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id', 't_median'])

    for i in new['id'].unique():
        print 'a3:',i
        data = new[new['id'] == i]
        if(len(data) == 1):
            writer.writerow([i, 0])
            continue
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        t_median = np.median(list_t_cha)

        writer.writerow([i, t_median])



'''
函数说明：将 a1_train.py 做出来的东西，和 前面做出来的许多 中位数 特征 连接在一起！
'''
def a4():
    print('mv to a4 function')
    train_new_with_label = pd.read_csv('../data/train_feature/train_new_with_label.csv')
    x_10_add = pd.read_csv('../data/train_feature/x_10_add.csv')
    y_10_add = pd.read_csv('../data/train_feature/y_10_add.csv')
    x_y_a_10_add = pd.read_csv('../data/train_feature/x_y_a_10_add.csv')
    x_y_v_10_add = pd.read_csv('../data/train_feature/x_y_v_10_add.csv')
    t_6_add = pd.read_csv('../data/train_feature/t_6_add.csv')

    new = pd.merge(x_10_add, y_10_add, on='id', how='inner')
    new = pd.merge(new, x_y_a_10_add, on='id', how='inner')
    new = pd.merge(new, x_y_v_10_add, on='id', how='inner')
    new = pd.merge(new, t_6_add, on='id', how='inner')
    new = pd.merge(train_new_with_label, new, on='id', how='inner')
    col = list(new.columns)
    col.remove('label')
    col.append('label')
    new = new[col]

    new.to_csv('../data/train_feature/train_new_with_label_with_midian.csv',index=False)



'''
函数说明：这可能是个强力特征。我做上两个特征：一个是 后面（t时间分布，后1/5吧）一些点x方向的方差，（之前已经把所有点x方向的方差做出来了），另一个是 两者的差值，
'''
def qiang_one_div_five_x_v(i, writer):
    writer.writerow([i, 0])

def qiang():
    new = pd.read_csv('../data/train_feature/train.csv')

    f = open('../data/train_feature/one_div_five_x_v.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id','one_fifth_x'])

    for i in new['id'].unique():
        print 'qiang:',i
        data = new[new['id'] == i]

        # 大熊老师的建议：t时间分布，后1 / 5吧；先找出 最后面 1/5 的时间段，然后求出来这段时间 有几个点，如果点数小于1，就continue
        list_t = list(data['guiji_t'])
        t_all = min(list_t) + (max(list_t) - min(list_t)) * 4 / 5.0
        data_t = data[data['guiji_t'] > t_all]

        if (len(data_t) < 2):
            qiang_one_div_five_x_v(i, writer)
            continue

        '''直接计算 data_t 中x方向上的速度的方差'''
        # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
        list_x = list(data_t['guiji_x'])
        list_x_minus_1 = list_x[1:]
        list_x = list_x[:-1]
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])
        # t 这一列也同上
        list_t = list(data_t['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        # 我们知道 刀塔x / 刀塔t 就是速度
        list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0,0.3) * 1.0))
        for ii in range(0,len(list_v)):
            list_v[ii] = round(list_v[ii],6)

        v_variance = np.var(list_v)
        writer.writerow([i,v_variance])
    f.close()


'''
函数说明：将 one_fifth_x 这个特征连接在原先特征上面，然后求出 差值
'''
def a5():
    print('mv to a5 function')
    train_new_with_label_with_midian = pd.read_csv('../data/train_feature/train_new_with_label_with_midian.csv')
    one_div_five_x_v = pd.read_csv('../data/train_feature/one_div_five_x_v.csv')

    new = pd.merge(train_new_with_label_with_midian, one_div_five_x_v, on='id', how='left')
    new['cha_zhi'] = new['one_fifth_x'] - new['x_v_variance']

    col = list(new.columns)
    col.remove('label')
    col.append('label')
    new = new[col]
    new.to_csv('../data/train_feature/train_new_with_label_with_midian_new.csv', index=False)


'''
函数说明：第一个点 到 第三个点之间的 时间段 在 整个时间段 所占有的比例
'''
def a6_1(i, writer):
    writer.writerow([i, 0])

def a6():
    new = pd.read_csv('../data/train_feature/train.csv')

    f = open('../data/train_feature/a6.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id', 'a6_bili'])

    for i in new['id'].unique():
        print 'a6:',i

        data = new[new['id'] == i]
        data = data.reset_index(drop=True)
        if (len(data) < 3):
            a6_1(i, writer)
            continue
        t1 = data.ix[2,'guiji_t'] - data.ix[0,'guiji_t']
        t2 = data.ix[len(data)-1,'guiji_t'] - data.ix[0,'guiji_t']

        writer.writerow([i, t1*1.0/t2])
    f.close()


'''
函数说明：2701~2900 最后一个点x方向上的速度 和 总体均值的差值
'''
def a7_1(i, writer):
    writer.writerow([i, 0])
def a7():
    new = pd.read_csv('../data/train_feature/train.csv')

    f = open('../data/train_feature/a7.csv', 'ab')
    writer = csv.writer(f)
    writer.writerow(['id', 'a7_cha'])

    for i in new['id'].unique():
        print 'a7:',i
        data = new[new['id'] == i]

        if (len(data) < 2):
            a7_1(i, writer)
            continue

        '''直接计算 data 中x方向上的速度的方差'''
        # 按照 本子上 的记录来写这个流程算法：第一步，做出来 x1~x6 的列表
        list_x = list(data['guiji_x'])
        list_x_minus_1 = list_x[1:]
        list_x = list_x[:-1]
        list_x_cha = list(DataFrame(list_x_minus_1)[0] - DataFrame(list_x)[0])
        # t 这一列也同上
        list_t = list(data['guiji_t'])
        list_t_minus_1 = list_t[1:]
        list_t = list_t[:-1]
        list_t_cha = list(DataFrame(list_t_minus_1)[0] - DataFrame(list_t)[0])

        # 我们知道 刀塔x / 刀塔t 就是速度
        list_v = list(DataFrame(list_x_cha)[0] / (DataFrame(list_t_cha)[0].replace(0, 0.3) * 1.0))
        for ii in range(0, len(list_v)):
            list_v[ii] = round(list_v[ii], 6)
        v_mean = np.mean(list_v)

        writer.writerow([i, list_v[len(list_v)-1] - v_mean])
    f.close()


def a8():
    print('mv to a8 function')
    train_new_with_label_with_midian_new = pd.read_csv('../data/train_feature/train_new_with_label_with_midian_new.csv')
    a6 = pd.read_csv('../data/train_feature/a6.csv')
    a7 = pd.read_csv('../data/train_feature/a7.csv')

    new = pd.merge(train_new_with_label_with_midian_new, a6, on='id', how='left')
    new = pd.merge(new, a7, on='id', how='left')

    col = list(new.columns)
    col.remove('label')
    col.append('label')
    new = new[col]
    new.to_csv('../data/train_feature/train_merge_7.csv', index=False)




if __name__ == "__main__":
    a1()
    a2()
    a3()
    a4()

    qiang()
    a5()

    a6()
    a7()

    a8()






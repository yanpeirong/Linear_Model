#coding=utf-8
import numpy as np
import csv
import matplotlib.pyplot as plt

def likelihood(x, y, beta):
    sum = 0
    m, n = np.shape(x)
    for i in range(m):
        j = -y[i] * np.dot(beta,x[i].T)+ np.math.log(1 + np.math.exp(np.dot(beta,x[i].T)))
        sum = sum + j
    return float(sum)


def diff1(x,y,beta):
    m, n = np.shape(x)
    sum = np.zeros((n))
    for i in range(m):
        p1 = np.math.exp(np.dot(beta, x[i].T)) / (1 + np.math.exp(np.dot(beta , x[i].T)))
        sum += -x[i] * (y[i] - p1)
    sum.shape = (n,1)
    return sum


def diff2(x, beta):
    m, n = np.shape(x)
    sum = np.zeros((n, n));
    for i in range(m):
        p1 = np.math.exp(np.dot(beta, x[i].T)) / (1 + np.math.exp(np.dot(beta, x[i].T)))
        xi = np.matrix(x[i])
        j = xi.T * xi * p1* (1 - p1)
        sum += j
    return sum


def newton(x, y):
    m, n = np.shape(x)
    beta = np.array([[0,0,1]])
    old_l = 0
    number = 0
    while 1:
        new_l = likelihood(x, y, beta)
        if abs(new_l-old_l) <= 0.00001:
            break
        number += 1
        old_l = new_l
        temp = np.dot(np.linalg.inv(diff2(x, beta)) , diff1(x, y, beta))
        temp.shape=(1,n)
        beta = beta - temp
    return beta

def drawimage(x,y,beta):
    m,n = np.shape(x)
    f1=plt.figure(1)
    ax1 = f1.add_subplot(111)
    ax1.scatter(x[y==1,0],x[y==1,1],c='g',s=100,label='good')
    ax1.scatter(x[y==0,0],x[y==0,1],c='k',s=100,label='bad')
    ax1.set_title('watermelon3.0a')
    ax1.set_xlabel('density')
    ax1.set_ylabel('sugar level')
    x = np.linspace(-0.2,1,100)
    y = -(beta[0][2]+beta[0][0]*x)/beta[0][1]
    ax1.plot(x,y)
    plt.legend()
    plt.show()

if __name__=='__main__':#主函数
    watermelon_data = []
    watermelon_file='E:/data/watermelon3.0.csv'
    with open(watermelon_file) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        watermelon_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到watermelon_data中
            watermelon_data.append(row)
    watermelon_data = [[float(x) for x in row] for row in watermelon_data]  # 将数据从string形式转换为float形式
    watermelon_data = np.array(watermelon_data)  # 将list数组转化成array数组便于查看数据结构
    watermelon_header = np.array(watermelon_header)
    dataset = watermelon_data[:,1:3]
    label = watermelon_data[:,3]
    dataset = np.column_stack((dataset,np.ones(dataset.shape[0])))
    beta = newton(dataset,label)
    print beta
    drawimage(dataset,label,beta)




#coding=utf-8
import numpy as np
import csv
import matplotlib.pyplot as plt
watermelon_data = []
with open('E:/data/watermelon3.0.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    watermelon_header = next(csv_reader)
    for row in csv_reader:
        watermelon_data.append(row)
watermelon_data = [[float(x) for x in row]for row in watermelon_data]
watermelon_data= np.array(watermelon_data)
dataset = watermelon_data[:,1:3]
label = watermelon_data[:,3]
mean =[]
for i in range(2):
    mean.append(np.mean(dataset[label==i],axis=0))
#计算sw
m,n = np.shape(dataset)
sw = np.zeros((n,n))
for i in range(m):
    if label[i]==0:
        w = (dataset[i] - mean[0]).reshape(n,1)
    else:
        w = (dataset[i] - mean[1]).reshape(n,1)
    sw += np.dot(w,w.T)
sw = np.matrix(sw)
u,sigma,vt = np.linalg.svd(sw)
sw_inv = vt.T*np.linalg.inv(np.diag(sigma))*u.T #sigma返回的是向量，应创建对角矩阵
w = np.dot(sw_inv ,(mean[0]-mean[1]).reshape(n,1))
print w
#画图
f1=plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.scatter(dataset[label==1,0],dataset[label==1,1],c='g',s=50,label='good')
ax1.scatter(dataset[label==0,0],dataset[label==0,1],c='k',s=50,label='bad')
ax1.set_title('watermelon3.0a')
ax1.set_xlabel('density')
ax1.set_ylabel('sugar level')
k = -w[0,0]/w[1,0]
x = np.arange(0,1,0.01)
y = k * x
ax1.plot(x,y,'b-')
for i in range(m):
    x0 = dataset[i,0]
    y0 = dataset[i,1]
    x1 = (x0 + k * y0)/(k*k +1)
    y1 = k * x0
    if label[i]==1:
        ax1.scatter(x1,y1,c='g',s=50)
        ax1.plot([x0,x1],[y0,y1],'g--')
    else:
        ax1.scatter(x1, y1, c='k', s=50)
        ax1.plot([x0,x1],[y0,y1],'k--')
plt.legend()
plt.show()










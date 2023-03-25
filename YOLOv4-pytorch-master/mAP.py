# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:49:57 2021

@author: 1952430
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

data=pd.read_excel('./AP.xlsx')

classes=[]
dic={}
predicted_class=data.predicted_class
score=data.score

for i in predicted_class:
    if not i in classes:
        classes.append(i)

for i in classes:
    value=[]
    for j in range(len(score)):
        if predicted_class[j]==i:
            value.append(score[j])
        dic[i]=np.mean(value)*100

# for i in dic:
#     plt.bar(i,dic[i],color='b')
# plt.xlabel(dic.keys())
# plt.ylabel(dic.values())
# plt.show()


# 中文乱码和坐标轴负号处理。
matplotlib.rc('font', family='Times New Roman', weight='bold')
plt.rcParams['axes.unicode_minus'] = False

#城市数据。
element_name1 = []
im1=[]
for i in dic:
    element_name1.append(i)
    im1.append(dic[i])
    
df=pd.DataFrame(dict(x=element_name1,y=im1))
df.to_csv('mAP.csv',index=False,header=True)
#数组反转。

df1=pd.read_excel('mAP.xlsx')
element_name=df1.x
im=df1.y
#element_name.reverse()
 
#绘图。
fig, ax = plt.subplots()
b = ax.barh(range(len(element_name)), im, color='#6699CC')
 
#为横向水平的柱图右侧添加数据标签。
for rect in b:
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2, '%d' %
            int(w), ha='left', va='center',fontsize=20)
 
#设置Y轴纵坐标上的刻度线标签。
ax.set_yticks(range(len(element_name)))
ax.set_yticklabels(element_name,fontsize=20)
 
a=np.arange(0,110,10)
plt.xticks(a,fontsize=18)
plt.xlabel('mAP of each category',fontsize=20)
 
#plt.title('XGBoost feature importance', loc='center', fontsize='25',fontweight='bold', color='red')
 
plt.show()

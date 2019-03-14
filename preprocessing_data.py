
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import csv
import json
import os
import pandas as pd
import multiprocessing


# In[2]:


raw_train_data = pd.read_csv('train_all.csv')


# In[3]:


# Columns (4,5,20,21) have mixed types.是由于\\N 故找出训练集里的\\N
print(raw_train_data[raw_train_data[raw_train_data.columns[4]].isin(['\\N'])].index)
print(raw_train_data[raw_train_data[raw_train_data.columns[5]].isin(['\\N'])].index)
print(raw_train_data[raw_train_data[raw_train_data.columns[20]].isin(['\\N'])].index)
print(raw_train_data[raw_train_data[raw_train_data.columns[21]].isin(['\\N'])].index)


# In[4]:


# 可以直接扔掉训练集里的\\N
print(raw_train_data.shape)
raw_train_data.drop(raw_train_data.index[[600438,642033,140569,232620]],inplace=True)
print(raw_train_data.shape)


# In[5]:


# 去除重复数据
raw_train_data.drop_duplicates(subset=['online_time','1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic',
                                       'last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time',
                                       'service2_caller_time'], inplace=True)
print(raw_train_data.shape)


# In[6]:


#训练集数据和标签 pandas
raw_x = raw_train_data.iloc[:,0:25].astype(float)
raw_y = raw_train_data.iloc[:,25].astype(int)


# #### 特征工程

# In[8]:


# 原始类别特征
raw_class_feature = ['service_type', 'complaint_level', 'contract_type', 'gender', 'is_mix_service',
                    'is_promise_low_consume',
                    'many_over_bill', 'net_service']
# 原始数值特征
raw_num_feature = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
                    'age', 'contract_time',
                    'former_complaint_fee', 'former_complaint_num',
                    'last_month_traffic', 'local_caller_time', 'local_trafffic_month', 'month_traffic',
                    'online_time', 'pay_num', 'pay_times', 'service1_caller_time', 'service2_caller_time']


# In[9]:


# 新增！差值特征
diff_feature_list = ['diff_total_fee_1', 'diff_total_fee_2', 'diff_total_fee_3', 'last_month_traffic_rest',
                     'rest_traffic_ratio',
                     'total_fee_mean', 'total_fee_max', 'total_fee_min', 'total_caller_time', 'service2_caller_ratio',
                     'local_caller_ratio',
                     'total_month_traffic', 'month_traffic_ratio', 'last_month_traffic_ratio', 'pay_num_1_total_fee',
                     '1_total_fee_call_fee', '1_total_fee_call2_fee', '1_total_fee_trfc_fee']

def add_diff_feature(data):
    data['diff_total_fee_1'] = data['1_total_fee'] - data['2_total_fee']
    data['diff_total_fee_2'] = data['2_total_fee'] - data['3_total_fee']
    data['diff_total_fee_3'] = data['3_total_fee'] - data['4_total_fee']

    data['pay_num_1_total_fee'] = data['pay_num'] - data['1_total_fee']

    data['last_month_traffic_rest'] = data['month_traffic'] - data['last_month_traffic']
    data['last_month_traffic_rest'][data['last_month_traffic_rest'] < 0] = 0
    data['rest_traffic_ratio'] = (data['last_month_traffic_rest'] * 15 / 1024) / data['1_total_fee']

    total_fee = []
    for i in range(1, 5):
        total_fee.append(str(i) + '_total_fee')
    data['total_fee_mean'] = data[total_fee].mean(1)
    data['total_fee_max'] = data[total_fee].max(1)
    data['total_fee_min'] = data[total_fee].min(1)

    data['total_caller_time'] = data['service2_caller_time'] + data['service1_caller_time']
    data['service2_caller_ratio'] = data['service2_caller_time'] / data['total_caller_time']
    data['local_caller_ratio'] = data['local_caller_time'] / data['total_caller_time']

    data['total_month_traffic'] = data['local_trafffic_month'] + data['month_traffic']
    data['month_traffic_ratio'] = data['month_traffic'] / data['total_month_traffic']
    data['last_month_traffic_ratio'] = data['last_month_traffic'] / data['total_month_traffic']

    data['1_total_fee_call_fee'] = data['1_total_fee'] - data['service1_caller_time'] * 0.15
    data['1_total_fee_call2_fee'] = data['1_total_fee'] - data['service2_caller_time'] * 0.15
    data['1_total_fee_trfc_fee'] = data['1_total_fee'] - (
    data['month_traffic'] - 2 * data['last_month_traffic']) * 0.3


# In[10]:


# 训练集 新增 差值特征：
print(raw_x.shape)
add_diff_feature(raw_x)
num_features=43
print(raw_x.shape)
print(raw_x.columns)
raw_x.head()


# In[11]:


#训练集数据和标签 从pandas到numpy array
x = np.array(raw_x,dtype=np.float64)
y = np.array(raw_y,dtype=np.int64)


# In[12]:


#筛选出service_type==1的训练数据
x_1 = x[np.where(x[:,0]==1)]
x_1 = x_1[:,1:num_features]
y_1 = y[np.where(x[:,0]==1)]
#筛选出service_type==4的训练数据
x_4 = x[np.where(x[:,0]==4)]
x_4 = x_4[:,1:num_features]
y_4 = y[np.where(x[:,0]==4)]

print('type=1 标签分布情况：')
print(pd.value_counts(y_1))
print('type=4 标签分布情况：')
print(pd.value_counts(y_4))


# In[13]:


#划分出训练集和验证集
from sklearn.model_selection import train_test_split
x_train_1, x_validation_1, y_train_1, y_validation_1 = train_test_split(x_1, y_1, test_size=0.33, random_state=2)
x_train_4, x_validation_4, y_train_4, y_validation_4 = train_test_split(x_4, y_4, test_size=0.33, random_state=2)


# #### 处理测试数据

# In[14]:


#读入测试数据
raw_test_data = pd.read_csv('republish_test.csv')


# In[15]:


# Columns (4,5) have mixed types.是由于\\N 不能直接扔掉，替换为999
raw_test_data = raw_test_data.replace('\\N', 999)


# In[16]:


#划分测试数据和id pandas
raw_test = raw_test_data.iloc[:,0:25].astype(float)
raw_id = raw_test_data.iloc[:,25]


# In[17]:


# 测试集 新增 差值特征：
print(raw_test.shape)
add_diff_feature(raw_test)
print(raw_test.shape)
print(raw_test.columns)
raw_test.head()


# In[18]:


x_test = np.array(raw_test,dtype=np.float64)
id_test = np.array(raw_id)


# #### 输出到文件

# In[19]:


#输出到文件
data_dir = 'preprocessed_add_diff'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

np.save(os.path.join(data_dir,'x_train_1.npy'),x_train_1)
np.save(os.path.join(data_dir,'x_validation_1.npy'),x_validation_1)
np.save(os.path.join(data_dir,'y_train_1.npy'),y_train_1)
np.save(os.path.join(data_dir,'y_validation_1.npy'),y_validation_1)

np.save(os.path.join(data_dir,'x_train_4.npy'),x_train_4)
np.save(os.path.join(data_dir,'x_validation_4.npy'),x_validation_4)
np.save(os.path.join(data_dir,'y_train_4.npy'),y_train_4)
np.save(os.path.join(data_dir,'y_validation_4.npy'),y_validation_4)

np.save(os.path.join(data_dir,'x_test.npy'),x_test)
np.save(os.path.join(data_dir,'id_test.npy'),id_test)


# In[20]:


#从文件load到numpy array
data_dir = 'preprocessed_add_diff'
#读入service_type==1的训练数据
x_train_1 = np.load(os.path.join(data_dir,'x_train_1.npy'))
x_validation_1 = np.load(os.path.join(data_dir,'x_validation_1.npy'))
y_train_1 = np.load(os.path.join(data_dir,'y_train_1.npy'))
y_validation_1 = np.load(os.path.join(data_dir,'y_validation_1.npy'))
#读入service_type==4的训练数据
x_train_4 = np.load(os.path.join(data_dir,'x_train_4.npy'))
x_validation_4 = np.load(os.path.join(data_dir,'x_validation_4.npy'))
y_train_4 = np.load(os.path.join(data_dir,'y_train_4.npy'))
y_validation_4 = np.load(os.path.join(data_dir,'y_validation_4.npy'))
#读入测试数据
x_test = np.load(os.path.join(data_dir,'x_test.npy'))
id_test = np.load(os.path.join(data_dir,'id_test.npy'))

print(x_train_1[0])
print(x_test[0])
print(id_test[0])


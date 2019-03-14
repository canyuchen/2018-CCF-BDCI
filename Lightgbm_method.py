
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import csv
import json
import os
import xgboost.sklearn as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[2]:


#从文件load到numpy array
#读入service_type==1的训练数据
x_train_1 = np.load('preprocessed_add_diff/x_train_1.npy')
x_validation_1 = np.load('preprocessed_add_diff/x_validation_1.npy')
y_train_1 = np.load('preprocessed_add_diff/y_train_1.npy')
y_validation_1 = np.load('preprocessed_add_diff/y_validation_1.npy')
#读入service_type==4的训练数据
x_train_4 = np.load('preprocessed_add_diff/x_train_4.npy')
x_validation_4 = np.load('preprocessed_add_diff/x_validation_4.npy')
y_train_4 = np.load('preprocessed_add_diff/y_train_4.npy')
y_validation_4 = np.load('preprocessed_add_diff/y_validation_4.npy')
#读入测试数据
# csv_file = csv.reader(open('preprocessed_add_diff/id_test.csv','r'))
# id_test = []
# for line in csv_file:
#     id_test.append(line[0])
id_test = np.load('preprocessed_add_diff/id_test.npy')
    
x_test = np.load('preprocessed_add_diff/x_test.npy')

print(x_test[0])
print(id_test[0])

label_1 = {90063345:0, 90109916:1, 90155946:2}
label_4 = {89950168:0, 89950166:1, 99999828:2, 99999827:3, 99999830:4, 89950167:5,99999826:6, 99999825:7}

for index in range(len(y_train_1)):
    y_train_1[index] = label_1[y_train_1[index]]

print(y_train_1[101])

for index in range(len(y_validation_1)):
    y_validation_1[index] =label_1[ y_validation_1[index]]

print(y_validation_1[101])

for index in range(len(y_train_4)):
    y_train_4[index] = label_4[y_train_4[index]]
     
print(y_train_4[101])

for index in range(len(y_validation_4)):
    y_validation_4[index] = label_4[y_validation_4[index]]

print(y_validation_4[101])


# In[3]:

#7
cat_cols = ['contract_type', 
            'net_service', 
            'gender',
            'complaint_level',       
            'is_mix_service',  
            'many_over_bill', 
            'is_promise_low_consume',     
    ]

#24
old_cols = ['is_mix_service',
            'online_time',
            '1_total_fee',
            '2_total_fee',
            '3_total_fee',

            '4_total_fee',
            'month_traffic',
            'many_over_bill',
            'contract_type',
            'contract_time',

            'is_promise_low_consume',
            'net_service',
            'pay_times',
            'pay_num',
            'last_month_traffic',

            'local_trafffic_month',
            'local_caller_time',
            'service1_caller_time',
            'service2_caller_time',
            'gender',

            'age',
            'complaint_level',
            'former_complaint_num',
            'former_complaint_fee'
    ]

#42
all_cols = ['is_mix_service', 
            'online_time', 
            '1_total_fee',
            '2_total_fee', 
            '3_total_fee', 
            
            '4_total_fee', 
            'month_traffic',
            'many_over_bill', 
            'contract_type', 
            'contract_time',

            'is_promise_low_consume', 
            'net_service', 
            'pay_times', 
            'pay_num',
            'last_month_traffic', 
            
            'local_trafffic_month', 
            'local_caller_time',
            'service1_caller_time', 
            'service2_caller_time', 
            'gender', 
            
            'age',
            'complaint_level', 
            'former_complaint_num', 
            'former_complaint_fee',
            'diff_total_fee_1', 

            'diff_total_fee_2', 
            'diff_total_fee_3',
            'pay_num_1_total_fee', 
            'last_month_traffic_rest', 
            'rest_traffic_ratio',

            'total_fee_mean', 
            'total_fee_max', 
            'total_fee_min', 
            'total_caller_time',
            'service2_caller_ratio', 
            
            'local_caller_ratio', 
            'total_month_traffic',
            'month_traffic_ratio', 
            'last_month_traffic_ratio',
            '1_total_fee_call_fee', 
            
            '1_total_fee_call2_fee',
            '1_total_fee_trfc_fee']

num_features = 43


# In[4]:


from sklearn.metrics import f1_score
targetlabel_1 = [0, 1, 2]

def evalerror_1(preds, lgtrain):
    preds = np.argmax(preds.reshape(3, -1),axis=0)
    y_true = lgtrain.get_label()
    result = f1_score(y_true,preds,average = 'macro')
    return 'f1_score',result,True


import lightgbm as lgb
lgb_train = lgb.Dataset(x_train_1,label = y_train_1,feature_name=all_cols, categorical_feature = cat_cols)
lgb_eval = lgb.Dataset(x_validation_1,label = y_validation_1,feature_name=all_cols, categorical_feature = cat_cols)
#lgb_train = lgb.Dataset(x_train_1,label = y_train_1)
#lgb_eval = lgb.Dataset(x_validation_1,label = y_validation_1)

print("begin build")        
params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':3,
            'max_depth': 8,
            'num_leaves': 150,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 66,
            'num_threads' : -1,
            'metric': 'multi_error',
         }
print("begin fit")    
model_1 = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    verbose_eval=10,
                    feval=evalerror_1,
                    early_stopping_rounds=100,
                   )



# In[5]:


# plt.figure(figsize=(12,6))
# lgb.plot_importance(model_1, max_num_features=30)
# plt.title("Featurertances")
# plt.show()

### make prediction for test data
y_predicted_1_prob = model_1.predict(x_validation_1, num_iteration=model_1.best_iteration)

y_predicted_1 = np.argmax(y_predicted_1_prob,axis = 1)   #转化为类别

accuracy_1 = accuracy_score(y_validation_1, y_predicted_1)
print("Accuracy: %.2f%%" % (accuracy_1 * 100.0))

from sklearn.metrics import f1_score
 
targetlabel_1 = [0, 1, 2]
score_1 = f1_score(y_validation_1, y_predicted_1, labels = targetlabel_1,average='macro')
score_1_array = f1_score(y_validation_1, y_predicted_1, labels = targetlabel_1,average=None)
print(score_1)
print(label_1)
print(score_1_array)


# In[6]:


import lightgbm as lgb
from sklearn.metrics import f1_score
targetlabel_4 = [0, 1, 2, 3, 4, 5, 6, 7]

def evalerror_2(preds, lgtrain):
    preds = np.argmax(preds.reshape(8, -1),axis=0)
    y_true = lgtrain.get_label()
    result = f1_score(y_true,preds,average = 'macro')
    return 'f1_score',result,True

lgb_train = lgb.Dataset(x_train_4,y_train_4,feature_name=all_cols, categorical_feature = cat_cols)
lgb_eval = lgb.Dataset(x_validation_4,y_validation_4,feature_name=all_cols, categorical_feature = cat_cols)
#lgb_train = lgb.Dataset(x_train_4,y_train_4)
#lgb_eval = lgb.Dataset(x_validation_4,y_validation_4)

print("begin build")        
params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':8,
            'max_depth': 8,
            'num_leaves': 150,
            # 'learning_rate': 0.05,
            # 'learning_rate': 0.01,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 66,
            'num_threads' : -1,
            'metric': 'multi_error',

            'reg_alpha' : 1,
            'reg_lambda' : 1e-5,
         }
print("begin fit")    
model_4 = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    verbose_eval=10,
                    feval=evalerror_2,
                    early_stopping_rounds=100,
                   )


# In[7]:


# plt.figure(figsize=(12,6))
# lgb.plot_importance(model_4, max_num_features=30)
# plt.title("Featurertances")
# plt.show()

### make prediction for test data
y_predicted_4_prob = model_4.predict(x_validation_4, num_iteration=model_4.best_iteration)

y_predicted_4 = np.argmax(y_predicted_4_prob,axis = 1)   #转化为类别

accuracy_4 = accuracy_score(y_validation_4, y_predicted_4)
print("Accuracy: %.2f%%" % (accuracy_4 * 100.0))

from sklearn.metrics import f1_score
 
targetlabel_4 = [0, 1, 2, 3, 4, 5, 6, 7]
score_4 = f1_score(y_validation_4, y_predicted_4, labels = targetlabel_4,average='macro')
score_4_array = f1_score(y_validation_4, y_predicted_4, labels = targetlabel_4,average=None)
print(score_4)
print(label_4)
print(score_4_array)


# In[8]:


### make prediction for test data
import warnings
warnings.filterwarnings("ignore")
y_predicted_test = []
line_predicted = []
result_label =0
temp = 0
dict_label_1 = {0:90063345,1:90109916, 2:90155946}
dict_label_4 = {0:89950168,1:89950166, 2:99999828,3: 99999827, 4:99999830, 5:89950167,6: 99999826, 7:99999825}
print("begin predict")
y_predicted_test_1_prob = model_1.predict(x_test[:,1:num_features], num_iteration=model_1.best_iteration)
y_predicted_test_1 = np.argmax(y_predicted_test_1_prob,axis = 1)   #转化为类别
y_predicted_test_4_prob = model_4.predict(x_test[:,1:num_features], num_iteration=model_4.best_iteration) 
y_predicted_test_4 = np.argmax(y_predicted_test_4_prob,axis = 1)   #转化为类别
index = 0
for line in x_test:
    if (line[0]==1):
        line_predicted = dict_label_1[y_predicted_test_1[index]]
    elif (line[0]==4):
        line_predicted = dict_label_4[y_predicted_test_4[index]]
    else:
        line_predicted = dict_label_4[y_predicted_test_4[index]]
    y_predicted_test.append(line_predicted)
    index += 1
print(y_predicted_test[0])


# In[9]:


print(x_test[0][1])
print(x_test[1])
print(x_test[2])


# In[10]:


header = [('user_id','current_service')]

with open('lgb_submission.csv', 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(header)
    for i in range(len(id_test)):
        writer.writerows([(id_test[i],y_predicted_test[i])])

print('done')




# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:07:25 2023

@author: Hugo
"""
import numpy as np
import re
import pandas as pd
from pandas import read_csv 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# data_path = 'E:\Kaggle\predict-student-performance-from-game-play\train.csv'

### loading data
train_data = read_csv('E:\Kaggle\predict-student-performance-from-game-play/train.csv')
train_label = read_csv('E:\Kaggle\predict-student-performance-from-game-play/train_labels.csv')

## clean train labels
session_label = train_label.session_id.str.split('_').str[0].tolist()
level_label = train_label.session_id.str.split('_').str[1].tolist()

level_num_label = list(range(len(level_label)))
for i in range(len(level_label)):
    level_num_label[i] = re.sub('\D','',level_label[i])

train_label_final = train_label
train_label_final['session']= session_label
train_label_final['level']= level_num_label

## find the number of NaN and get rid of columns with more 60% NaN entry
NaN_Count = pd.isnull(train_data).sum()
NaN_Percentage = NaN_Count/len(train_data)

Col_drop = NaN_Percentage[NaN_Percentage >= 0.6]
Col_Keep = NaN_Percentage[NaN_Percentage < 0.6]

Index_keep = Col_Keep.index 
train_keep = train_data[Index_keep]

## manipulate categorical data
# x = train_keep.head()
Count_event_name = train_keep['event_name'].value_counts()
Count_name = train_keep['name'].value_counts()
Count_fqid = train_keep['fqid'].value_counts()
Count_room_fqid = train_keep['room_fqid'].value_counts()

# train label only has level 1 - 18
# Drop level 0 and levels above 18
level_range = np.arange(1,19)
train_with_label = train_keep[train_keep.level.isin(level_range) == True]

# aggregate fqid, room_fqid, Sum through dummies event_name, name
dummy_train = pd.get_dummies(train_with_label, columns = ['event_name','name'])
# x=dummy_train.head()

name_columns = [col for col in dummy_train.columns if 'name_' in col]
coor_columns = [col for col in dummy_train.columns if '_coor_' in col]

agg_train = []
# agg_train = train_with_label.groupby(['session_id','level']).agg({'fqid':['nunique']})
# agg_train = agg_train.concat(agg_train,axis = 1)
# agg_train.name = agg_train.name + '_nunique'

temp = train_with_label.groupby(['session_id','level']).agg({'fqid':['nunique']})
agg_train.append(temp)

temp = train_with_label.groupby(['session_id','level']).agg({'room_fqid':['nunique']})
agg_train.append(temp)

for names in name_columns:
    temp = dummy_train.groupby(['session_id','level']).agg({names:['sum']})
    agg_train.append(temp)

for coors in coor_columns:
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['first']})
    agg_train.append(temp)
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['last']})
    agg_train.append(temp)
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['median']})
    agg_train.append(temp)
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['std']})
    agg_train.append(temp)
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['min']})
    agg_train.append(temp)
    temp = train_with_label.groupby(['session_id','level']).agg({coors:['max']})
    agg_train.append(temp)

# temp = train_with_label.groupby(['session_id','level']).agg({'elapsed_time':['sum']})
# temp = temp/1000
# agg_train.append(temp)
temp = train_with_label.groupby(['session_id','level']).agg({'elapsed_time':['mean']})
temp = temp/1000
agg_train.append(temp)
temp = train_with_label.groupby(['session_id','level']).agg({'elapsed_time':['std']})
temp = temp/1000
agg_train.append(temp)

final_train = pd.concat(agg_train,axis = 1)
final_train.columns = final_train.columns.map(''.join)
final_train=final_train.reset_index()


# level_count = []
# for i in np.arange(1,19):
#     temp = agg_train.loc[agg_train.level == i].count()
#     level_count.append(temp)
# # Missing Level = 15 (index 14), drop Level = 14;
# # Missing Level = 7 (index 6), but only missing one...

Row_drop_train = final_train
Row_drop_train = Row_drop_train[Row_drop_train.level!=15]
Row_drop_train = Row_drop_train[Row_drop_train.level!=7]
Row_drop_train = Row_drop_train.reset_index()
Row_drop_train = Row_drop_train.drop(['index'],axis = 1)

Col_names = Row_drop_train.columns
# min_max_scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler() 
# scaler = preprocessing.QuantileTransformer(random_state = 42)
normal_train_temp = pd.DataFrame(scaler.fit_transform(Row_drop_train.iloc[:,2:len(Col_names)].values))
### basically normal_train_temp stores all the features we want
# normal_train_temp = Row_drop_train.iloc[:,2:len(Col_names)]

# normal_train = Row_drop_train
# normal_train.iloc[:,2:len(Col_names)] = normal_train_temp
# normal_final = normal_train.reset_index()


Row_drop_label = train_label_final.astype({'level':'int'})
Row_drop_label = Row_drop_label[Row_drop_label.level!=15]
Row_drop_label = Row_drop_label[Row_drop_label.level!=7]
Row_drop_label = Row_drop_label.reset_index()
Row_drop_label = Row_drop_label.drop(['index'],axis = 1)

temp = Row_drop_label.sort_values(by = ['session','level'])
label_final = temp.reset_index()
label_final = label_final.drop(['index','session_id'],axis = 1)
# label_final = label_final.reset_index()

### turn data into tensorts
import torch 
X = torch.tensor(normal_train_temp.values).type(torch.float)
y = torch.tensor(label_final['correct'].values).type(torch.float)

## split into train test
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size = 0.2,
                                                 random_state = 42)



### Model with Decision tree ####

# from sklearn import tree 
# from sklearn import metrics
# clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth = 15, random_state = 42) 
# clf = clf.fit(X_train, y_train) 
# # tree.plot_tree(clf)     

# tree_predict = clf.predict(X_train)
# tree_test = clf.predict(X_test)

# print("Train Accuracy:", metrics.accuracy_score(y_train,tree_predict),"Test Accuracy:",metrics.accuracy_score(y_test,tree_test))


# # def acc_tree (y_true, y_pred):
# #     correct = np.equal(y_true,y_pred).sum().item()
# #     acc = (correct/len(y_pred)) * 100 
# #     return acc 
    
# # tree_accuracy = acc_tree(y_train,tree_predict)
# # test_tree_accuracy = acc_tree(y_test,tree_test)
# # tree_depth = clf.get_depth()



### Model with Random forest ####
# from sklearn.ensemble import RandomForestClassifier 

# RFClf = RandomForestClassifier(criterion = "gini",max_depth = 20, random_state = 42) 
# RFClf.fit(X_train,y_train) 
# RF_predict = RFClf.predict(X_train) 
# RF_test = RFClf.predict(X_test) 

# print("Train Accuracy:", metrics.accuracy_score(y_train,RF_predict),"Test Accuracy:",metrics.accuracy_score(y_test,RF_test))



### Adabosst classifier ###
# from sklearn.ensemble import AdaBoostClassifier 
# ABClf = AdaBoostClassifier(n_estimators = 400, learning_rate =0.1, random_state = 42) 

# ABClf.fit(X_train,y_train) 
# Ada_predict = ABClf.predict(X_train) 
# Ada_test = ABClf.predict(X_test) 
# print("Train Accuracy:", metrics.accuracy_score(y_train,Ada_predict),"Test Accuracy:",metrics.accuracy_score(y_test,Ada_test))




import torch.nn as nn 

### Model with neural network
# class ModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.layer_1 = nn.Linear(in_features = len(Col_names)-2,
#                                  out_features = 100)
#         self.layer_2 = nn.Linear(in_features = 100,
#                                  out_features = 100)
#         self.layer_3 = nn.Linear(in_features = 100,
#                                  out_features = 10)
#         self.layer_4 = nn.Linear(in_features = 10,
#                                  out_features =1)
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu((self.layer_1(x))))))))

# model_0 = ModelV0()


# ### setup the loss function
# loss_fn = nn.BCEWithLogitsLoss()

# ## Adam or SGD
# optimizer = torch.optim.SGD(params = model_0.parameters(),
#                             lr = 0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100 
    return acc 

# y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)))

# ### Train
# torch.manual_seed(42)

# epochs = 100 

# # loops
# for epoch in range(epochs):
#     ### Training
#     model_0.train()
    
#     y_logits = model_0(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) 
#     # y_pred = torch.round(torch.special.expit(y_logits)) 
    
#     loss = loss_fn(y_logits,y_train) 
#     acc = accuracy_fn(y_true = y_train, y_pred=y_pred) 
    
#     optimizer.zero_grad()
    
#     loss.backward()

#     optimizer.step()

#     ###Testing
#     model_0.eval()
#     with torch.inference_mode():
#         test_logits = model_0(X_test).squeeze() 
#         test_pred = torch.round(torch.sigmoid(test_logits)) 
        
#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_true = y_test,
#                                y_pred = test_pred)
    
#     ## print
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss: .5f}, Acc: {acc: .2f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}% ")
        
# from sklearn.metrics import f1_score
# Explore_F1 = f1_score(y_train.detach().numpy(),y_pred.detach().numpy(),average = 'macro')



### Convlutional NN
class Model_CNN(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_units,
                 output_shape):
        super().__init__()
        self.Conv_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels = input_shape,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU())
            # nn.MaxPool1d(kernel_size = 2))
        self.classifier = nn.Sequential(
            nn.Linear(in_features = hidden_units,
                      out_features = output_shape))
    def forward(self,x):
        x = self.Conv_layer_1(x).squeeze(2)
        x = self.classifier(x)
        return x

model_1 = Model_CNN(input_shape = len(Col_names)-2,hidden_units = 30, output_shape = 1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_1.parameters(),lr = 0.1) 

### Train
torch.manual_seed(42)

## out of memory after epochs >= 20 
epochs = 15

def CNN_accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100 
    return acc 

CNN_pred = torch.rand(len(X_train))
CNN_test_pred = torch.rand(len(X_test))

# loops
for epoch in range(epochs):
    if epoch %5 == 0:
        print(f"Epoch: {epoch} \n----")
    ### Training
    train_loss = 0
    train_acc = 0
    for ind,X_in in enumerate(X_train) :
        model_1.train()
        
        y_logits = model_1(X_in.unsqueeze(0).unsqueeze(2)).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) 
        CNN_pred[ind] = y_pred 

        # y_pred = torch.round(torch.special.expit(y_logits)) 
        
        loss = loss_fn(y_logits,y_train[ind]) 
        # acc = accuracy_fn(y_true = y_train[ind], y_pred=y_pred) 
        
        train_loss += loss
        # train_acc += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        # if ind %2000 == 0:
        #     print(f"Looked at {ind} samples")
    train_loss/=len(X_train)
    
    train_acc = accuracy_fn(y_true = y_train, y_pred = CNN_pred)
    # train_acc /=len(X_train)
    ###Testing 
    test_loss,test_acc = 0,0
    model_1.eval()
    with torch.inference_mode():
        for ind,X_in_test in enumerate(X_test):
            test_logits = model_1(X_in_test.unsqueeze(0).unsqueeze(2)).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits)) 
            CNN_test_pred[ind] = test_pred
            
            test_loss += loss_fn(test_logits, y_test[ind])
            # test_acc += accuracy_fn(y_true = y_test[ind],
            #                        y_pred = test_pred)
        
        test_loss /= len(X_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=CNN_test_pred)
        # test_acc /= len(X_test)
    
    ## print
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {train_loss: .5f}, Train Acc: {train_acc: .2f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}% ")
        

from sklearn import metrics
print("Train Accuracy:", metrics.accuracy_score(y_train.detach().numpy(),CNN_pred.detach().numpy()),"Test Accuracy:",
      metrics.accuracy_score(y_test.detach().numpy(),CNN_test_pred.detach().numpy()))



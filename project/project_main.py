import numpy as np 
import pandas as pd
import os
import time
from sklearn import preprocessing
from sklearn import cluster
from datetime import datetime as dt
#for this problem we need to load test and training data first and then print some graph to get some extra features and delete some useless data.

#load data through pd.read
train = pd.read_csv('comp9417-master/train.csv')
test = pd.read_csv('comp9417-master/test.csv')

#show the shape of train and test datasets (how many data and how many features)
print("Train :",train.shape)
print("Test:",test.shape)

#change data type to datetime
train['Open Date'] = pd.to_datetime(train['Open Date'])
test['Open Date'] = pd.to_datetime(test['Open Date'])

# for these huge data sets, it is uncertain whether it contains null values
# get column with null values for train data set
train.columns[train.isna().any()].tolist()
# get column with null values for test data set
test.columns[test.isna().any()].tolist()

#we need to seperate categorical and numberical variables from whole dataset for analysis
 
#for this dataset some numerical features will be ['id','p1',......,'p37','revenue']
num_fea = train.select_dtypes([np.number]).columns.tolist()
#for this dataset some categorical features will be ['city','city group','type']
cate_fea = train.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()

print('\ncheck the target revenue:')
print(train['revenue'].describe())

#Preprocessing
#According to the graph for revenue, we found out that some data with revenue > 1*10^7 will be outlayers
#Drop those outpayers
train = train[train['revenue'] < 1*10000000]
train.reset_index(drop=True).head()

print('\nNew train data shape:',train.shape)

#print(trains[train['P35']  == 0].shape)
#print(trains[train['P36']  == 0].shape)

#record the length of train and test sets
train_n = train.shape[0]
test_n = test.shape[0]
#for convenience, we can concat train and test sets then process together
data = pd.concat((train.loc[:,'Id':'P37'],test.loc[:,'Id':'P37']), ignore_index=True)
#we need to convert other cities to clusters by the cities we already know
#at first we need a copy of train
train_c = train.copy()
rel_pv = ['P1', 'P2', 'P8', 'P11', 'P19', 'P20', 'P23','P24', 'P30']
train_c = train_c.loc[:,rel_pv]
means = cluster.KMeans(n_clusters=25)
means.fit(train_c)
# get the cluster
# classify the city of each data instance as one of the centers
data['Clu'] = means.predict(data.loc[:,rel_pv])
del data['City']
#process city group
data = data.join(pd.get_dummies(data['City Group'],prefix='CG'))
data = data.drop(['City Group'],axis=1)
#process type
data = data.join(pd.get_dummies(data['Type'],prefix='T'))
data = data.drop(['Type'],axis=1)
#show all data's type
print('\n',data.dtypes)

#seperate data to train and test sets
train_pro = data[:train_n]
test_pro = data[train_n:]
#show train and test
print('\nTrain(transition):',train_pro.shape)
print('Test(transition):',test_pro.shape)
#check the length of the train
train['revenue'] = [np.log(i) for i in train['revenue']]
print('length of train[revenue]:',len(train['revenue']))
train_pro['revenue'] = train['revenue'].values
#get processed train and test sets
train = train_pro
test = test_pro

print("Train(processed):",train.shape)
print("Test(processed):",test.shape)
#print(train['Open Date']) #use for test attribute in open date

#now we got processed train and test data sets
#we need to convert the data type to the one machine learning model need
def prepare(frame,aim):
    f = frame.copy()
    #at first we need to change datetime type to numerical
    f['Open Date Day']  = f['Open Date'].dt.day
    f['Open Date Month']  = f['Open Date'].dt.month
    f['Open Date Year']  = f['Open Date'].dt.year
    '''
    something wrong with one date minus another date here, but if we add both those 2 dates to extra features, we should get the same result.
    #we may need to know the date difference between open date and now
    date_diff = []
    for d in f['Open Date']:
        diff = dt.now() - d
        #change to int and let the number wont be too large
        insert = int(diff.days)
        date_diff.append(insert)
    f['Different'] = pd.Series(date_diff)
    '''
    #now we can drop open date
    f = f.drop(['Open Date'], axis=1)
    #get values for y
    if aim in f.columns:
        y = f[aim]
        f = f.drop([aim], axis=1)
    else:
        #y_test will be none here
        y = None
    return (f,y)
#now we can get prepared train data
X_train,Y_train = prepare(train,'revenue')
#Since test dont have revenue , so Y_test must be None
X_test,Y_test = prepare(test,'revenue')

#get the length of them
num_training = X_train.shape[0]
num_test = X_test.shape[0]
num_fea = X_train.shape[1]

from sklearn.metrics import accuracy_score, f1_score

#  Model1 decision tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=round(0.02*num_training))
dtree.fit(X_train,Y_train.astype('int'))
Y_pred = dtree.predict(X_test)
#print(Y_train)
Y_pred = np.exp(Y_pred)
print('\nMy DecisionTree prediction:',Y_pred)
#if we get the revenue of test sets we can do something like below to get accuracy
#Accuracy_s = accuracy_score(Y_test,Y_pred)
#S = f1_score(Y_test, Y_pred)
#print(f'Decision Tree Accuracy: {dtree.score(X_test,Y_test)}')

result = pd.DataFrame({
        "Id": test["Id"],
        "Prediction": Y_pred
    })
result.to_csv('DecisionTreePrediction.csv',header=True, index=False)

#  Model2 Neural Network
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(2*num_fea,), activation='tanh',  alpha = 0.0001, solver='sgd', learning_rate='adaptive', learning_rate_init=0.01, verbose = False, random_state = None)
nn.fit(X_train,Y_train.astype('int'))
Y_pred = nn.predict(X_test)
Y_pred = np.exp(Y_pred)
print('My NeuralNetwork prediction:',Y_pred)

result = pd.DataFrame({
        "Id": test["Id"],
        "Prediction": Y_pred
    })
result.to_csv('NeuralNetworkPrediction.csv',header=True, index=False)

#  Model3 Support Vector Machine
from sklearn.svm import SVC

svm = SVC(C=0.05, kernel='linear', gamma='scale')
svm.fit(X_train,Y_train.astype('int'))
Y_pred = svm.predict(X_test)
Y_pred = np.exp(Y_pred)
print('My SupportVectorMachine prediction:',Y_pred)

result = pd.DataFrame({
        "Id": test["Id"],
        "Prediction": Y_pred
    })
result.to_csv('SVMPrediction.csv',header=True, index=False)

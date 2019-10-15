#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#function to do the target encoding
def calc_smooth_mean(df1,df2,cat_name, target, weight):
    # Compute the global mean
    mean = data1[target].mean()

    # Compute the number of values and the mean of each group
    agg = data1.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
       return df1[cat_name].map(smooth)
    else:
       return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())

#Read the training data 
data1= pd.read_csv('D:/TCD/MACHINE LEARNING/KAGGLE COMPETITION/dataset backup/tcd ml 2019-20 income prediction training (with labels).csv')

#Fill the empty data values of training data
data1['Age'].fillna((data1['Age'].median()), inplace=True)
data1['Year of Record'].fillna((data1['Year of Record'].median()), inplace=True)
data1['Profession'].fillna(method='ffill', inplace=True)
data1['Gender'].fillna(method='ffill', inplace=True)
data1['University Degree'].fillna(method='ffill', inplace=True)
data1['Hair Color'].fillna(method='ffill', inplace=True)
data1['Country'].fillna(method='ffill', inplace=True)

#Read the testing data
data2= pd.read_csv('D:/TCD/MACHINE LEARNING/KAGGLE COMPETITION/dataset backup/tcd ml 2019-20 income prediction test (without labels).csv')

#Fill the empty data values of testing data
data2['Age'].fillna((data2['Age'].median()), inplace=True)
data2['Year of Record'].fillna((data2['Year of Record'].median()), inplace=True)
data2['Country'].fillna(method='ffill', inplace=True)
data2['Profession'].fillna(method='ffill', inplace=True)
data2['Gender'].fillna(method='ffill', inplace=True)
data2['University Degree'].fillna(method='ffill', inplace=True)
data2['Hair Color'].fillna(method='ffill', inplace=True)

#Apply target encoding to categorical values in both training and testing data with respect to 'Income in Euros'
(data1['Profession'],data2['Profession'])=calc_smooth_mean(df1=data1,df2=data2,cat_name='Profession',target='Income in EUR',weight=0)
(data1['University Degree'],data2['University Degree'])=calc_smooth_mean(df1=data1,df2=data2,cat_name='University Degree',target='Income in EUR',weight=0)
(data1['Gender'],data2['Gender'])=calc_smooth_mean(df1=data1,df2=data2,cat_name='Gender',target='Income in EUR',weight=0)
(data1['Hair Color'],data2['Hair Color'])=calc_smooth_mean(df1=data1,df2=data2,cat_name='Hair Color',target='Income in EUR',weight=0)
(data1['Country'],data2['Country'])=calc_smooth_mean(df1=data1,df2=data2,cat_name='Country',target='Income in EUR',weight=0)

#Divide the training data set to 80-20 ratio
validationData=data1.iloc[89594:,:]
trainingValues=data1.iloc[:89594,:]
predictionValues=data2

#Combine the training and testing data sets
Data_in_rows = pd.concat([trainingValues, validationData,predictionValues])

#Choose the dependent and independent variables
X_values = Data_in_rows[['Country','Age','Profession','Year of Record','University Degree','Gender','Hair Color','Body Height [cm]']]
Y_values = Data_in_rows[['Income in EUR']]
X_values=X_values.to_numpy()

# Choose the training and testing data
training_data=X_values[:89594,:]
training_data_target=Y_values.iloc[:89594,:]

#Choose the regression model
#number of estimators - ususally bigger the forest the better
#bootstrap = method for sampling data points
#max depth of each tree-default none leading to full tree 
#max_features = max number of features considered for splitting a node
#min_samples_leaf = min number of data points allowed in a leaf node
#min_samples_split = min number of data points placed in a node before the node is split
model=RandomForestRegressor(n_estimators=2000,bootstrap=True,max_depth=None,max_features='auto',min_samples_leaf=4,min_samples_split=10)

#Train the model
result=model.fit(training_data,training_data_target)

validation_data_Xvalues=X_values[89594:111993,:]
validation_data_Yvalues=Y_values.iloc[89594:111993,:]

#Find the score of the model
print(model.score(validation_data_Xvalues,validation_data_Yvalues))

#Predict the income of the remaining data
predict_data=X_values[111993:,:]
predicted_Income=model.predict(predict_data)

#Write Predicted Income to a CSV File
dfobj=pd.DataFrame(predicted_Income,columns=['Income']) 
dfobj.to_csv('D:/TCD/MACHINE LEARNING/KAGGLE COMPETITION/dataset backup/income.csv')

#Find the RMSE value-- Not Accurate
training_data_target=training_data_target.iloc[:73230,:]
rmse=sqrt(mean_squared_error(training_data_target,predicted_Income))
print("RMSE Value: ")
print(rmse)


# In[ ]:





# In[ ]:





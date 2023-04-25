#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets  import load_boston
boston =load_boston()


# In[3]:


data = pd.DataFrame(boston.data)


# In[4]:


data.columns = boston.feature_names

data.head()


# In[5]:


#adding target variable to dataframe
data['PRICE'] = boston.target


# In[6]:


data


# In[7]:


data.isnull().sum()


# In[8]:


#Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[9]:


#Plotting the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True, square= True, fmt='.1f',annot=True,annot_kws={'size':15}, cmap='Blues')


# In[10]:


#split dependent variable and independent variables
x = data.drop(['PRICE'],axis = 1)
y = data['PRICE']


# In[11]:


#splitting data to training and testing dataset.
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state = 0)


# In[12]:


#Use Linear regression(Train the Machine)to create Model
import sklearn
from sklearn.linear_model import LinearRegression
#Create a linear regressor
lm =LinearRegression()
# Train the model using the training sets
model=lm.fit(xtrain,ytrain)


# In[13]:


xtrain


# In[14]:


#Predict the y_pred for all values of train_x and test_x
ytrain_pred =lm.predict(xtrain)
ytest_pred =lm.predict(xtest)


# In[15]:


testdata=[[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0,15.3,396.90,4.98]]


# In[16]:


test_pred =lm.predict(testdata)
test_pred


# In[17]:


#Evaluate the performance of model for train_y and test_y
df1=pd.DataFrame(ytrain_pred,ytrain)
df2=pd.DataFrame(ytest_pred,ytest)
df1


# In[18]:


df2


# # Model Evaluation

# In[19]:


#Calculate Mean Square error for train_y and test_y
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(ytest, ytest_pred)
print('MSE on the test data:',mse)
mse1 = mean_squared_error(ytrain_pred,ytrain)
print('MSE on training data:',mse1)


# In[20]:


#from sklearn.metrics import mean_squared_error
#def linear_metrics ():
r2 = lm .score(xtest, ytest)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
print('r-squared:{}'.format(r2))
print('...................................')
print('root mean squared error: {}'.format(rmse))


# In[26]:


#Plotting the linear regression model
plt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest, ytest_pred ,c='lightgreen',marker='s' ,label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.tittle("True value vs Predicted value")
plt.legend(loc= 'upper left') #plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()


# In[23]:


testdata=[[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0,15.3,396.90,4.98]]


# In[ ]:


test_pred = lm.predict(testdata)
tet_pred


# In[ ]:





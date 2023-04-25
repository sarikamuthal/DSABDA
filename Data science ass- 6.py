#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


# * Loading the dataset

# In[2]:


data =  pd.read_csv("C:/Users/coeco/Downloads/Iris (1).csv")
data.head(5)


# * Checking Basic statistics of the dataset

# In[3]:


data.describe(include = 'all')


# In[4]:


data.info()


# * Displaying Shape of the dataset and The Types of Species to Classify

# In[5]:


print(data.shape)
data['Species'].unique()


# * Checking for Null values

# In[6]:


data.isnull().sum()


# * As we see there are no missing values so lets split our dataset into training(x) and testing(y) 

# In[7]:


x = data.iloc[:,1:5]
y = data.iloc[:,5:]


# * Encoding the Species column

# In[8]:


encode = LabelEncoder()
y = encode.fit_transform(y)


# * Spliting training and testing dataset by 70-30 

# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)


# ### Preparing Naive Bayes Model

# In[10]:


naive_bayes = GaussianNB()
naive_bayes.fit(x_train,y_train)
pred = naive_bayes.predict(x_test)


# In[11]:


pred


# In[12]:


y_test


# * Plotting Confusion Matrix 

# In[13]:


matrix =  confusion_matrix(y_test,pred,labels = naive_bayes.classes_)
print(matrix)

tp, fn, fp, tn = confusion_matrix(y_test,pred,labels=[1,0]).reshape(-1)


# In[14]:


conf_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=naive_bayes.classes_)
conf_matrix.plot(cmap=plt.cm.YlGn)
plt.show()


# * Evaluating our model and calculating TN,FN,TP,FP Accuracy,Recall,Precision,ErrorRate,

# In[15]:


print(classification_report(y_test,pred))


# In[16]:


print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test,pred)))
print('Error Rate: ',(fp+fn)/(tp+tn+fn+fp))
print('Sensitivity (Recall or True positive rate) :',tp/(tp+fn))
print('Specificity (True negative rate) :',tn/(fp+tn))
print('Precision (Positive predictive value) :',tp/(tp+fp))
print('False Positive Rate :',fp/(tn+fp))


# In[ ]:





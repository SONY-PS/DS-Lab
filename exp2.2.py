#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[2]:


dataset = pd.read_csv('diabetes.csv')
print(len(dataset))
print(dataset)


# In[3]:


zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
print(dataset['Insulin'])


# In[4]:


X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0 ,test_size=0.2)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


# In[5]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[6]:


import math
math.sqrt(len(y_test))


# In[7]:


classifier = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')


# In[8]:


classifier.fit(X_train, y_train)


# In[11]:


y_pred=classifier.predict(X_test)


# In[12]:


print(accuracy_score(y_test,y_pred))


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





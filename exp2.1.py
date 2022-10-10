#!/usr/bin/env python
# coding: utf-8

# In[1]:


weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']


# In[2]:


temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot',
'Mild']


# In[3]:


play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[4]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# In[5]:


temp_encoded=le.fit_transform(temp)
print(temp_encoded)
print(" ")
label=le.fit_transform(play)
print(label)


# In[6]:


features=list(zip(weather_encoded,temp_encoded))
print(features)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)


# In[8]:


model.fit(features,label)
predicted= model.predict([[0,1]])                                                                                                                                                               
print(predicted)                                               


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.

# Data Description:

# Murder -- Muder rates in different places of United States

# Assualt- Assualt rate in different places of United States

# UrbanPop - urban population in different places of United States

# # Rape - Rape rate in different places of United States


# In[2]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[5]:


# Import Dataset
crime=pd.read_csv(r'C:\Users\anupa\Downloads\crime_data.csv')
crime


# In[6]:


crime.info()


# In[7]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime


# In[8]:


# Normalize heterogenous numerical data using standard scalar fit transform to dataset
crime_norm=StandardScaler().fit_transform(crime)
crime_norm


# In[9]:


# DBSCAN Clustering
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(crime_norm)


# In[10]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[11]:


# Adding clusters to dataset
crime['clusters']=dbscan.labels_
crime


# In[12]:


crime.groupby('clusters').agg(['mean']).reset_index()


# In[13]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(crime['clusters'],crime['UrbanPop'], c=dbscan.labels_) 


# In[ ]:





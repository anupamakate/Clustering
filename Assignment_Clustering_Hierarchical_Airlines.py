#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# # Draw the inferences from the clusters obtained.


# In[1]:



# Data Description:
 
# The file EastWestAirlinescontains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers

# ID --Unique ID

# Balance--Number of miles eligible for award travel

# Qual_mile--Number of miles counted as qualifying for Topflight status

# cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
# cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
# cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:

# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000

# Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months

# Bonus_trans--Number of non-flight bonus transactions in the past 12 months

# Flight_miles_12mo--Number of flight miles in the past 12 months

# Flight_trans_12--Number of flight transactions in the past 12 months

# Days_since_enrolled--Number of days since enrolled in flier program

# Award--whether that person had award flight (free flight) or not


 


# In[2]:


# Using Normalize Function


# In[8]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[11]:


# Import Dataset
import xlsxwriter
airline=pd.read_csv(r'C:\Users\anupa\Downloads\EastWestAirlines.csv')
airline


# In[12]:


airline.info()


# In[13]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[14]:


# Normalize heterogenous numerical data 
airline2_norm=pd.DataFrame(normalize(airline2),columns=airline2.columns)
airline2_norm


# In[15]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(airline2_norm,'complete'))


# In[16]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
hclusters


# In[17]:


y=pd.DataFrame(hclusters.fit_predict(airline2_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[18]:


# Adding clusters to dataset
airline2['clustersid']=hclusters.labels_
airline2


# In[19]:


airline2.groupby('clustersid').agg(['mean']).reset_index()


# In[20]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clustersid'],airline2['Balance'], c=hclusters.labels_) 


# In[ ]:





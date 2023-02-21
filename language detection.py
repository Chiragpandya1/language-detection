#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


df=pd.read_csv("E:\Language Detection.csv")
df


# In[53]:


df['Language'].unique()


# In[4]:


# this data need to be clean so for cleaning we have to remove punctuations


# In[5]:


string.punctuation


# In[11]:


def remove_punct(text):
    for pun in string.punctuation:
        text= text.replace(pun,"")
    text=text.lower() # to make every character in lowercase
    return(text)


# In[12]:


df['Text'].apply(remove_punct)


# # splitting data 

# In[20]:


x=df.iloc[:,0]
y=df.iloc[:,1]


# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=20,random_state=5)


# In[38]:


from sklearn import feature_extraction


# In[39]:


vec = feature_extraction.text.TfidfVectorizer(ngram_range =(1,2),analyzer = 'char')


# In[40]:


from sklearn import pipeline
from sklearn import linear_model


# In[41]:


model_pipe = pipeline.Pipeline([('vec',vec),('clf',linear_model.LogisticRegression())])


# In[42]:


model_pipe.fit(x_train,y_train)


# In[47]:


y_pred = model_pipe.predict(x_test)


# In[48]:


from sklearn import metrics


# In[49]:


metrics.accuracy_score(y_test,y_pred)*100


# In[50]:


metrics.confusion_matrix(y_test,y_pred)


# In[ ]:


# evaluting model by different languages


# In[51]:


model_pipe.predict(['My name is jay'])


# In[57]:


model_pipe.predict(['God eftermiddag'])


# In[75]:


df[df['Language']=='Kannada']


# In[74]:


model_pipe.predict(['ನಾವೆಲ್ಲರೂ ಇಂಗ್ಲಿಷ್ನಲ್ಲಿ ಹೆಚ್ಚು ನಿರರ್ಗಳವಾಗಲು'])


# In[ ]:





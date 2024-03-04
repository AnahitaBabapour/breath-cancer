#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


data = pd.read_csv('data.csv')
data.drop('Unnamed: 32',axis=1, inplace=True)
data.drop('id',axis=1, inplace=True)
data


# In[3]:


data.describe()


# In[4]:


data.diagnosis


# In[5]:


data.diagnosis.unique()


# In[6]:


data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
data


# In[7]:


plt.hist(data.diagnosis, color='purple')
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


# In[8]:


#یادت باشه میتونی نمودار های کتابخونه سیبورن بزاری


# In[9]:


dt_x = data.iloc[:,1:31]
dt_y = data.iloc[:,0:1]
dt_x


# In[10]:


x = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0,1)).fit(dt_x).transform(dt_x),
                 columns=(['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_meanco','points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
]))
y = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0,1)).fit(dt_y).transform(dt_y),columns=(['diagnosis']))


# In[11]:


logreg = LogisticRegression()
kfold = KFold(10)
cross = cross_val_score(logreg, x, np.ravel(y), cv= kfold)
print(cross)


# In[12]:


print(np.mean(cross))


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

logreg.fit(x_train, np.ravel(y_train))
y_pred = logreg.predict(x_test)
y_pred = pd.DataFrame(y_pred,columns = (['y_pred'])) 
print('intercept is: ',logreg.intercept_)
print('weights are: ',logreg.coef_)


# In[16]:


print('accuracy: ',metrics.accuracy_score(y_test,y_pred))


# In[19]:


fpr,tpr,_ = metrics.roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
#plt.legend(loc=4)
plt.show()


# In[21]:


y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr, label = 'Data, AUC: '+str(auc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend(loc=4)
plt.show()


# In[ ]:


c1 = np.corrcoef(data.radius_mean , data.diagnosis)
c2 = np.corrcoef(data.texture_mean , data.diagnosis)
c3 = np.corrcoef(data.perimeter_mean , data.diagnosis)
c4 = np.corrcoef(data.area_mean , data.diagnosis)
#....


# In[22]:


compare = pd.DataFrame({'y_test': y_test.values.flatten(),
                       'y_pred': y_pred.values.flatten()})
compare
#compare.to_csv('C:/Users/SETUP CO/Desktop/compare.csv')


# In[ ]:





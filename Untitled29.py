#!/usr/bin/env python
# coding: utf-8

# Goal : To design an algorithm that determines the “type” of a Pokemon (i.e.: grass, electric, fire)
# 
# -> There are 898 of total pokemons and all belong to one of the 18 different types of pokemon. 
# 
# 
# There are 18 Types of Pokemon
# Normal, Fire, Water, Bug, Grass, Electric, Fighting, Poison etc. 
# 
# 
# https://pokemon.fandom.com/wiki/Types
# 
# 
# 
# 

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
import pydotplus


# # EXPLORATORY DATA ANALYSIS

# loading the data set to Pandas Dataframe using Pandas pd.read_CSV

# In[23]:


train = pd.read_csv('train_pokemon.csv')


# head() to see all the coloumn types and names and it displays only the first 5 rows of the data set

# In[24]:


train.head()


# In[25]:


train.shape


#  Describe() functions is used on the data fraem to see the mean, median, standard deviations, quartiles to see if there are any discrepencies in the data and if there are any outliers thats effecting the data. 

# In[26]:


train.describe()


# using heat map to see if there are any Null values in the data set. There are multiple ways to check the null values. 

# In[27]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# Prediction can be made based on the Color and the Body Type

#  I HAVE DROPPED THE COLOUMNS THAT WONT HELP IN MODEL TRAINING AND ALSO WITH TOO MANY NULL VALUES, WHERE WE CANNOT TO FILL THOSE. 

# In[28]:


train.drop(columns=['Egg_Group_2'], inplace=True)


# In[29]:


train.drop(columns=['Type_2'], inplace=True)


# In[30]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


#  FEATURE SELECTION - WE CAN SELECT THE FEATURES THAT WILL HELP IN MODEL TRAINING AND CREATE A NEW DATA FRAME

# In[69]:


selected_columns = train[["Name", "Type_1","Body_Style", "Color", "Egg_Group_1"]]
new_df = selected_columns.copy()
print(new_df)


#  USING ENCODING TO MAKE THE DATA MORE READABLE BY THE MODELS. 

# In[70]:


def preprocessor(new_df):
    res_df = new_df.copy()
    le = preprocessing.LabelEncoder()
    
    res_df['Name'] = le.fit_transform(res_df['Name'])
    res_df['Type_1']= le.fit_transform(res_df['Type_1'])
    res_df['Color'] = le.fit_transform(res_df['Color'])
    res_df['Egg_Group_1'] = le.fit_transform(res_df['Egg_Group_1'])
    res_df['Body_Style'] = le.fit_transform(res_df['Body_Style'])
    return res_df


# In[71]:


encoded_df = preprocessor(new_df)
x = encoded_df.drop(['Type_1'],axis =1).values
y = encoded_df['Type_1'].values


# SPLITTING THE DATA SET INTO TRAIN AND TEST. uSUALLY WE SELECT 70% OF THE DATA TO TRAIN THE MODEL AND REMAINING 30% TO TEST THE MODEL

# # MODEL TRAINING

# In[72]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3)


#  I HAVE USED MULTIPLE MODELS TO SEE WHICH ONE PERFORMS BETTER. 

#  KNN ALGORITHM IS A CLASSIFICATION REGRESSION ALGORITHM. IN KNN THE OUTPUT WE GET IS THE CLASS. 

# In[80]:


clf_KNN = KNeighborsClassifier(n_neighbors = 3)
clf_KNN.fit(x_train, y_train)
y_pred_knn = clf_KNN.predict(x_test)
acc_knn = round(clf_KNN.score(x_train, y_train) * 100, 2)
print (acc_knn)


# In[74]:


clf_DT = DecisionTreeClassifier()
clf_DT.fit(x_train, y_train)
y_pred_decision_tree = clf_DT.predict(x_test)
acc_decision_tree = round(clf_DT.score(x_train, y_train) * 100, 2)
print (acc_decision_tree)


# In[82]:


clf_RF = RandomForestClassifier(n_estimators=100)
clf_RF.fit(x_train, y_train)
y_pred_random_forest = clf_RF.predict(x_test)
acc_random_forest = round(clf_RF.score(x_train, y_train) * 100, 2)
print (acc_random_forest)


# In[85]:


models = pd.DataFrame({
    'Model' : ['KNN' , 'Decision Tree', 'Random Forest'],
    
    'Score': [ acc_knn, acc_decision_tree, acc_random_forest]
    
})

models.sort_values(by='Score', ascending=False)


# # Model performance and Accuracy 

# In[86]:


predictions = clf_RF.predict(x_test)
predictions


# In[ ]:





# In[93]:


y_actual_result = y_actual_result.flatten()
count = 0
for result in y_actual_result:
     if(result == 1):
        count=count+1

print ("true yes|predicted yes:")
print (count/float(len(y_actual_result)))


# In[94]:


print (confusion_matrix(y_test, predictions))


# In[95]:


accuracy_score(y_test, predictions)


# In[ ]:





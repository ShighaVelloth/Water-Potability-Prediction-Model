#!/usr/bin/env python
# coding: utf-8

# # WATER POTABILITY PREDICTIVE MODEL

# In[2]:


#importing neccasary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("C:/Users/shigha/Downloads/water_potability (1).csv")
print(df)


# ## UNDERSTANDING DATASET

# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.describe()


# In[13]:


df.info()


# In[14]:


df.index


# In[15]:


df.columns


# In[16]:


df.size


# In[17]:


df.dtypes


# In[18]:


df.ndim


# In[19]:


df.shape


# ## DATA PREPROCESSING

# In[4]:


#To detect null values
df.isnull().sum()


# In[5]:


df["ph"].fillna(df["ph"].median(), inplace=True)
df["Sulfate"].fillna(df["Sulfate"].median(), inplace=True)
df["Trihalomethanes"].fillna(df["Trihalomethanes"].median(), inplace=True)


# In[6]:


#Checking if null value exists again
df.isnull().sum()


# ## DATA VISUALIZATION

# In[25]:


sns.pairplot(data = df,hue = 'Potability',palette=['teal','coral'])


# In[29]:


#visualising count of potability
sns.countplot(data = df, x = df.Potability)
df.Potability.value_counts()


# In[30]:


#Plotting pie chart of count of potability
data=df.value_counts('Potability')
data.plot(kind='pie')
plt.ylabel('Count')
plt.title('Count of Potability')


# In[31]:


corr_mat = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_mat,annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')


# ## Data Modelling

# Applying Non- Parametric Algorithm
# 
# 1. Random Forest
# 2. Support Vector Machine
# 3. K-Nearest Neighbor

# In[7]:


#Splitting data into test set and training set
X = df.drop("Potability", axis=1)
y = df["Potability"]


# In[8]:


#NORMALISATION
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# In[ ]:





# In[10]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

models =[('KNN',KNeighborsClassifier(n_neighbors=10)),
         ('RF',RandomForestClassifier(random_state=0)), ("SVC", SVC())]

results = []
names = []
finalResults = []

for name,model in models:
    model.fit(X_train, y_train)
    model_results = model.predict(X_test)
    score = accuracy_score(y_test, model_results)
    results.append(score)
    names.append(name)
    finalResults.append((name,score))

finalResults.sort(key=lambda k:k[1],reverse=True)
finalResults


# ##  Best Model : Random Forest

# In[11]:


#CONFUSION MATRIX FOR BEST  MODEL RANDOM FOREST
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
Rfc = RandomForestClassifier()
Rfc.fit(X_train, y_train)
y_Rfc = Rfc.predict(X_test)
classification_report = metrics.classification_report(y_test, y_Rfc)
print(classification_report)

modelAccuracy = []  # Define the list modelAccuracy
modelAccuracy.append(metrics.accuracy_score(y_test, y_Rfc))
print(modelAccuracy)

sns.heatmap(confusion_matrix(y_test, y_Rfc), annot=True, fmt='d')


# Parametric Algorithm
# 
# 1. Naive Bayes
# 2. Logistic Regression

# In[12]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[13]:


from sklearn.metrics import accuracy_score

Y_nb_pred = nb.predict(X_test)
nb_val_score = accuracy_score(y_test, Y_nb_pred) * 100
nb_train_score = accuracy_score(y_train, nb.predict(X_train)) * 100


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test, Y_nb_pred))


# In[15]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, Y_nb_pred)
print(conf_matrix)


# In[16]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)


# In[17]:


print(classification_report(y_test, y_pred))


# In[ ]:





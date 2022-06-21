# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:54:09 2022

this is a script for assessment 1 (data scientist course) regarding 
cardiovascular disease (CVD)

credit :
    http://archive.ics.uci.edu/ml/datasets/Heart+Disease
    
@author: Afiq Sabqi
"""


import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#%%                           STATIC

DATA_PATH=os.path.join(os.getcwd(),'dataset','heart.csv')

LE_PATH=os.path.join(os.getcwd(),'LE_model.pkl')
BEST_PIPELINE_PATH=os.path.join(os.getcwd(),'best_pipeline.pkl')




def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%%                          DATA LOADING

df=pd.read_csv(DATA_PATH)

#%%                          DATA INSPECTION

df.info()
df.describe().T

df.isna().sum()
# no NaNs value

df.duplicated().sum()
# only one duplicated file
df=df.drop_duplicates()
# since only one data duplicated will do instant.


df.columns

continuous_data=['age','trtbps','chol','thalachh','oldpeak']

categorical_data=['sex','cp','fbs','restecg','exng','slp',
                  'caa','thall','output']

fig , ax = plt.subplots(ncols=1,nrows=len(continuous_data))
for i in range(len(continuous_data)):
    sns.boxplot(df[continuous_data[i]],ax=ax[i])

plt.subplots_adjust(hspace=2)
plt.show()
# have outliers that is not important

cor = df.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cor,annot=True, cmap=plt.cm.Blues)
plt.show()

for con in continuous_data:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

for cat in categorical_data:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()


#%%                           DATA CLEANING

df['thall']=df['thall'].replace(0, np.nan)
df['thall'].fillna(df['thall'].mode()[0], inplace=True)
# in column thall there is a value = 0 which is null.
# hence replace with zero then impute using mode

# drop an outlierfrom cholestrol column
df = df[df['chol']<350]
# drop an outlierfrom oldpeak column
df = df[df['oldpeak']<=4]
# drop an outlierfrom trtbps column
df = df[df['trtbps']<170]
# drop an outlierfrom thalachh column
df = df[df['thalachh']>80]

fig , ax = plt.subplots(ncols=1,nrows=len(continuous_data))
for i in range(len(continuous_data)):
    sns.boxplot(df[continuous_data[i]],ax=ax[i])

plt.subplots_adjust(hspace=2)
plt.show()
#%%                           FEATURES SELECTION

continuous_data.append('output')
for i in df.columns:
    if i not in continuous_data:
        df[i] = df[i].astype('category')
        
df.info()

for i in continuous_data:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['output']))


for j in categorical_data:
    print(j)
    confusion_mat = pd.crosstab(df[j],df['output']).to_numpy()
    print(cramers_corrected_stat(confusion_mat))



y=df['output']
X=df.drop(['output'],axis=1)

nb_class=len(np.unique(df['output']))  # this is only for deep learning

ss=StandardScaler()
X=ss.fit_transform(X)

# Need to save SS model
SS_FILE_NAME=os.path.join(os.getcwd(),'standard_scaler.pkl')

with open(SS_FILE_NAME,'wb') as file:
     pickle.dump(ss,file)

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size=0.3,
                                                  random_state=123)


#%%                            PIPELINE


# To determine which ML approach is the best

# LR
step_mms_lr=[('mmscaler',MinMaxScaler()),
              ('lr',LogisticRegression())]

step_ss_lr=[('sscaler',StandardScaler()),
            ('lr',LogisticRegression())]

# RF
step_mms_rf=[('mmscaler',MinMaxScaler()),
              ('rf',RandomForestClassifier())]

step_ss_rf=[('sscaler',StandardScaler()),
            ('rf',RandomForestClassifier())]

# TREE
step_mms_tree=[('mmscaler',MinMaxScaler()),
                ('tree',DecisionTreeClassifier())]

step_ss_tree=[('sscaler',MinMaxScaler()),
              ('tree',DecisionTreeClassifier())]

# KNN
step_mms_knn=[('mmscaler',MinMaxScaler()),
              ('knn',KNeighborsClassifier())]
            
step_ss_knn=[('sscaler',MinMaxScaler()),
              ('knn',KNeighborsClassifier())]

# Create a list for the pipeline so that we can iterate them
pipelines=[Pipeline(step_mms_lr),Pipeline(step_ss_lr),
            Pipeline(step_mms_rf),Pipeline(step_ss_rf),
            Pipeline(step_mms_tree),Pipeline(step_ss_tree),
            Pipeline(step_mms_knn),Pipeline(step_ss_knn)]


# fitting the data
for pipe in pipelines:
    pipe.fit(X_train,y_train)

best_accuracy=0
pipeline_scored=[]

# Score/model evaluation
for k, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    pipeline_scored.append(pipeline.score(X_test,y_test))

best_pipeline=pipelines[np.argmax(pipeline_scored)]
best_accuracy=pipeline_scored[np.argmax(pipeline_scored)]
print('The best combination of pipeline will be {} with accuracy of {}'
      .format(best_pipeline.steps,best_accuracy))



step_ss_rf=[('sscaler',StandardScaler()),
            ('rf',RandomForestClassifier())]

pipeline_ss_rf=Pipeline(step_ss_rf)

#number of trees
grid_param=[{'rf':[RandomForestClassifier()],
             'rf__n_estimators':[10,100,1000],
             'rf__max_depth':[None,5,15]}]

grid_search=GridSearchCV(pipeline_ss_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model=grid_search.fit(X_train,y_train)
best_model.score(X_test,y_test)


##       Retrain the selected parameters
step_ss_rf=[('sscaler',StandardScaler()),
            ('rf',RandomForestClassifier(n_estimators=10,
                                         min_samples_leaf=4,
                                         max_depth=100))]


#%%                        SAVING MODEL

##              Saving Model
BEST_MODEL_PATH=os.path.join(os.getcwd(),'best_model.pkl')
with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)

##              Saving Pipeline
with open(BEST_PIPELINE_PATH,'wb') as file:
    pickle.dump(pipeline,file)


#%%                       MODEL ANALYSIS

print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

y_true=y_test
y_pred=best_model.predict(X_test)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))


#%%                           DISCUSSION

'''

    *This model has an accuracy f1 = 85.5%. 
    
    *In features selections, there is no features is filtered because 
    non filtered features gives more accuracy. from logistic regression
    and cramer's V shows the correlation of all features
    
    *StandardScaler + RandomForestClassifier is selected as machine learning
    model approach since it gives the highest accuracy with 83.1%


'''



































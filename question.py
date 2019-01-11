#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:12:56 2019

@author: jluo
"""

import pandas as pd
temp=pd.read_csv('../data/train.csv')
positive=temp[temp['target']==1].copy()
negative=temp[temp['target']==0].sample(len(positive))
data=pd.concat([negative, positive])

from sklearn.feature_extraction.text import TfidfVectorizer
model=TfidfVectorizer(binary=True, use_idf=False, norm=None,
                      max_df=0.1, min_df=5, max_features=None)
X=model.fit_transform(data['question_text'])

print('X shape',X.shape)
temp=X[0,:].todense()
print(temp.sum())


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.svm import SVC

#clf=SVC(C=1.0, kernel='rbf')
clf=RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1)
train_x, test_x, train_y, test_y=train_test_split(X, data['target'], 
                test_size=0.25, stratify=data['target'], random_state=8)



clf.fit(train_x, train_y)

print('training performance',
      precision_recall_fscore_support(train_y, clf.predict(train_x)))

pred_train=clf.predict(train_x)
proba_train=clf.predict_proba(train_x)[:,1]

pred=clf.predict(test_x)
proba=clf.predict_proba(test_x)[:,1]

print(precision_recall_fscore_support(test_y, pred))
print(roc_auc_score(test_y,proba))

#%%

test_data=pd.read_csv('../data/test.csv')
test_X=model.transform(test_data['question_text'])
result=clf.predict(test_X)
test_data['prediction']=result

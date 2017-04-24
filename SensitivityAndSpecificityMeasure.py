print(__doc__)

import numpy as np
from numpy import*
from scipy import interp
import copy
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_actual[i]==y_hat[i]:
           TP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_actual[i]!=y_hat[i]:
            print("yactual:",y_actual[i])
            print("yhat",y_hat[i])
            FN += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==0 and y_actual[i]==y_hat[i]:
           TN += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==0 and y_actual[i]!=y_hat[i]:
           FP += 1

    return(TP, FP, TN, FN)

#classifier = svm.SVC(kernel='linear', probability=True,random_state=random_state)
#methodName='SVM'

#classifier =GaussianNB()
#methodName='Naive Bayes'

classifier = RandomForestClassifier(n_estimators=10)
methodName='Random Forest' 
X=[[1],[1],[2],[2],[3],[3],[4],[4]]
Y=[1,1,1,1,1,1,0,0]
y=[0,1,1,1,1,1,1,1]
probas_ = classifier.fit(X,Y ).predict_proba(X)
print probas_
TPR=TP/(float)(TP+FN)
TPR2=TP/6.0
print("TPR",TPR)
print("TPR2",TPR2)
TP,FP,TN,FN=perf_measure(Y,y)
print(TP,FP,TN,FN)




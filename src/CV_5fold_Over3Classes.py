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
## Run classifier with cross-validation and plot ROC curves
random_state = np.random.RandomState(0)
#classifier = svm.SVC(kernel='linear', probability=True,random_state=random_state)
#methodName='SVM'

#classifier =GaussianNB()
#methodName='Naive Bayes'

classifier = RandomForestClassifier(n_estimators=10)
methodName='Random Forest' 



iris = datasets.load_iris()
X = iris.data
Y=iris.target

#n_classes = y.shape[1]
#y = label_binarize(y, classes=[0, 1, 2])
#kf = KFold(X.shape[0], n_folds=5)
n_classes=np.unique(Y).shape
print n_classes
FPR=dict()
TPR=dict()
ROC_AUC=dict()
PROBAS=[]
YTEST=[]
MEANTPR=[]
ALLFPR=[]
fig, ax = plt.subplots()
ax.set_color_cycle(['red', 'grey'])
#plt.figure(figsize=(2,1))

for n in range(0,n_classes[0]):
    iris = datasets.load_iris()
    X = iris.data
    Y=iris.target
    #add noise to better test the performance
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]    
#    print n    
    y = copy.copy(Y)
    for k in range(0,len(y)):
        if y[k]==n:
            y[k]=1
        else:
            y[k]=0
#    print y            

    fpr = dict()
    tpr = dict()
    roc_auc = dict()    
    cv = StratifiedKFold(y, n_folds=5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr = []
    
    for i, (train, test) in enumerate(cv):
        Ytest=y[test]
        for m in range(0,len(Ytest)):
            YTEST.append(Ytest[m])
        
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        PRO=probas_[:, 1]
        
#        print '========================================================='
        for k in range(0,len(PRO)):
            PROBAS.append(PRO[k])
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#        YTEST.append(y[test])        
#        PROBAS.append(probas_)
        for i in range(0,len(fpr)):
            ALLFPR.append(fpr[i])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
    #    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(cv)
    for x in range(0,len(mean_tpr)):
        MEANTPR.append(mean_tpr[x])
    
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    color='black'    
    if(n%5==0):
        color='green'
    elif(n%5==1):
        color='blue'
    elif(n%5==2):
        color='yellow'
    elif(n%5==3):
        color='brown'
    else:
        color='pink'
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC class %d (area = %0.2f)' % (n,mean_auc),color=color, lw=2) 
#             plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

#print YTEST#y_score
FPR["micro"], TPR["micro"], _ = roc_curve(YTEST, PROBAS)
ROC_AUC["micro"] = auc(FPR["micro"], TPR["micro"])
plt.plot(FPR["micro"], TPR["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(ROC_AUC["micro"]),
         linewidth=1.5)
       



#=================MACRO AVERAGE(not fully tested) 
MEANTPR=np.array(MEANTPR)
MeanTPR=[]
mean_tpr=list(mean_tpr)
while(len(mean_tpr)>len(all_fpr)):
    mean_tpr.pop()

ROC_AUC["macro"] = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr,
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(ROC_AUC["macro"]),
         linewidth=1.5)
#=========================================================        
         
        
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Perfectly calibrated')    
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(methodName)
plt.legend(loc="lower right")           
plt.legend(loc="lower right",prop={'size':10})
plt.show()
#mean_tpr = np.zeros_like(all_fpr)
#print mean_tpr
#mean_tpr=mean_tpr+MEANTPR

#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
## Finally average it and compute AUC
#mean_tpr /= n_classes
#print all_fpr.shape
#print "======================="
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



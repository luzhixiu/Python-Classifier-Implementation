print(__doc__)
import copy
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

# Run classifier with cross-validation and plot ROC curves

random_state = np.random.RandomState(0)

classifier = svm.SVC(kernel='linear', probability=True,random_state=random_state)
methodName='SVM'
#
#classifier =GaussianNB()
#methodName='Naive Bayes'

#classifier = RandomForestClassifier(n_estimators=10)
#methodName='Random Forest' 

iris = datasets.load_iris()
X = iris.data
Y=iris.target

y = copy.copy(Y)

#y = label_binarize(y, classes=[0, 1, 2])


#y = label_binarize(y, classes=[0, 1, 2])print(shape(X))
#X, y = X[y != 2], y[y != 2]
#original  
for i in range(0,len(y)):#for 0,0,1  
    if (y[i]==1):
        y[i]=0
    elif (y[i]==2):
        y[i]=1
print y
cv = StratifiedKFold(y, n_folds=5)
#random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    #print(probas_)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Perfectly calibrated')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',color='blue',
         label='Mean ROC class 0(area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(methodName)
plt.legend(loc="lower right")

#==============================================================================
#for i in range(0,len(y)):#for 0,0,1  
#    if (y[i]==1):
#        y[i]=0
#    if (y[i]==2):
#        y[i]=1
y = copy.copy(Y)
cv = StratifiedKFold(y, n_folds=5)
for i in range(0,len(y)):#for 1，0,0  
    if (y[i]==0):
        y[i]=1
    else:
        y[i]=0
   

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    #print(probas_)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',color='green',
         label='Mean ROC class 1 (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right")        
         
         
         
#=========================================================================================  
y = copy.copy(Y)
cv = StratifiedKFold(y, n_folds=5)     
for i in range(0,len(y)):#for 0,1,0  
    if (y[i]==2):
        y[i]=0                  


#Plot all ROC curves


for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    #print(probas_)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

  
#    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',color='red',
         label='Mean ROC class 2 (area = %0.2f)' % mean_auc, lw=2)
plt.plot(mean_fpr, mean_tpr, 'k--',color='red',
         label='Mean ROC class 2 (area = %0.2f)' % mean_auc, lw=2)         
         
plt.legend(loc="lower right")    
         
plt.show()
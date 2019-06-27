import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(seed=2017)
#reading data set
dataset=pd.read_csv('loan-prediction.csv')
dataset.dtypes

#histogram of null values
values=list(dataset.apply(lambda x:sum(x.isnull()),axis=0))
colors=['r', 'g', 'b', 'c', 'm','y','r', 'g', 'b', 'c', 'm','y','r']
fig= plt.figure(figsize=(22,6))
plt.bar(['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
                         'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',
                        'Loan_Status'], values, color= colors)
fig.show()

#non numarical to numarical conversion
d=dataset
d.Property_Area.replace(['Urban','Semiurban','Rural'],[2,1,0],inplace=True)
d.Married.replace(['Yes','No'],[1,0],inplace=True)
d.Gender.replace(['Male','Female'],[1,0],inplace=True)
d.Education.replace(['Graduate','Not Graduate'],[1,0],inplace=True)
d.Self_Employed.replace(['Yes','No'],[1,0],inplace=True)
d.Dependents.replace(['0','1','2','3+'],[0,1,2,3],inplace=True)
d.Loan_Status.replace(['Y','N'],[1,0],inplace=True)
dataset=d
dataset['LoanAmount'].fillna(np.mean(dataset.LoanAmount),inplace=True)
dataset['Loan_Amount_Term'].fillna(np.median(dataset.Loan_Amount_Term),inplace=True)
dataset['Credit_History'].fillna(np.median(dataset.Credit_History),inplace=True)
dataset['Dependents'].fillna(np.median(dataset.Dependents),inplace=True)
dataset.dropna(how='any',inplace=True)
dataset.to_csv('loan-prediction1.csv',index=False)
dataset.shape
X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]
print(X.shape,y.shape)
#feature selection
#https://scikit-learn.org/stable/modules/feature_selection.html
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',random_state=5)
rfe = RFE(dt, 8)
rfe = rfe.fit(X,y)
print(rfe.support_)
print(rfe.ranking_)
X.head()
X_new=rfe.fit_transform(X,y)
import pandas as pd
X_df=pd.DataFrame(X_new)
X_df.head()
X_df.to_csv('featured_X_dataframe.csv',index=False)
X_df=pd.read_csv('featured_X_dataframe.csv')
X=X_df
X.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2017)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#learning curve plotter
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt
#cross validation and accuracy measures
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
#support vector machine
#https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm 
svm_cls = svm.SVC(gamma=0.05,C=100,kernel='rbf')
svm_cls.fit(X_train_std, y_train)

print("Test set score svm_cls: %f" % svm_cls.score(X_test_std, y_test))
print("Classification report for on test:\n%s\n"% (classification_report(y_test, svm_cls.predict(X_test_std))))
print("Confusion matrix for svm_cls_rbf on test:\n%s" % (confusion_matrix(y_test, svm_cls.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(svm_cls, X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(svm_cls,"Learning Curves (svm Classifier)" , X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#dt
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',random_state=5)
dt.fit(X_train_std, y_train)

print("Test set score dt: %f" % dt.score(X_test_std, y_test))
print("Classification report for dt on test:\n%s\n"% (classification_report(y_test, dt.predict(X_test_std))))
print("Confusion matrix for dt on test:\n%s" % (confusion_matrix(y_test, dt.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(dt , X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(dt,"Learning Curves (dt Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()


from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image  

import os
import sys

def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths)
        
        
from sklearn import tree

buffer = StringIO()
tree.export_graphviz(dt, out_file=buffer, 
                         feature_names=X.columns,  
                         class_names=X.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(buffer.getvalue())
conda_fix(graph)
graph.write_pdf("loan_tree.pdf") 
Image(graph.create_png())
#ada-booster #boosting
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=1000,learning_rate=0.01, random_state=0)
ada.fit(X_train_std, y_train)

print("Test set score ada: %f" % ada.score(X_test_std, y_test))
print("Classification report for ada on test:\n%s\n"% (classification_report(y_test, ada.predict(X_test_std))))
print("Confusion matrix for ada on test:\n%s" % (confusion_matrix(y_test, ada.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(ada , X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(ada,"Learning Curves (ada Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#random forest #baggibg
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=5,max_features='auto')
rf.fit(X_train_std, y_train)

print("Test set score rf: %f" % rf.score(X_test_std, y_test))
print("Classification report for rf on test:\n%s\n"% (classification_report(y_test, rf.predict(X_test_std))))
print("Confusion matrix for rf on test:\n%s" % (confusion_matrix(y_test, rf.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(rf, X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(rf,"Learning Curves (rf Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#perceptron #feed forward
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,2), activation='logistic',max_iter=500,learning_rate_init=0.005,tol=1e-41,solver='adam')
mlp.fit(X_train_std, y_train)

print("Test set score mlp: %f" % mlp.score(X_test_std, y_test))
print("Classification report for mlp on test:\n%s\n"% (classification_report(y_test, mlp.predict(X_test_std))))
print("Confusion matrix for mlp on test:\n%s" % (confusion_matrix(y_test, mlp.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(mlp, X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(mlp,"Learning Curves (mlp Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#KNN 5
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier as KNN
knn=KNN(n_neighbors=3)
knn.fit(X_train_std, y_train)

print("Test set score knn: %f" % knn.score(X_test_std, y_test))
print("Classification report for knn on test:\n%s\n"% (classification_report(y_test, knn.predict(X_test_std))))
print("Confusion matrix for knn on test:\n%s" % (confusion_matrix(y_test, knn.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(knn, X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(knn,"Learning Curves (knn Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#logistic regression
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression as LR
lr=LR(C=1.0,solver='newton-cg',penalty='l2',multi_class='ovr',max_iter=500)
lr.fit(X_train_std, y_train)

print("Test set score lr: %f" % lr.score(X_test_std, y_test))
print("Classification report for lr on test:\n%s\n"% (classification_report(y_test, lr.predict(X_test_std))))
print("Confusion matrix for lr on test:\n%s" % (confusion_matrix(y_test, lr.predict(X_test_std))))
print('5 fold cross validation')
cvs=cross_val_score(lr, X_std, y, cv=5)
print(cvs,np.mean(cvs))

plot_learning_curve(lr,"Learning Curves (lr Classifier)", X_train_std,y_train,ylim=None,n_jobs=4)
plt.show()

#saving models
from sklearn.externals import joblib #dumping and loading model
joblib.dump(svm_cls,'svm_cls_model.joblib')
joblib.dump(dt,'dt_model.joblib')
joblib.dump(ada,'ada_model.joblib')
joblib.dump(mlp,'mlp_model.joblib')
joblib.dump(knn,'knn_model.joblib')
joblib.dump(lr,'lr_model.joblib')
joblib.dump(rf,'rf_model.joblib')

import seaborn as sns
from sklearn import metrics
sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
def plotAUC(truth, pred, lab):
    fpr, tpr, _ = metrics.roc_curve(truth,pred)
    roc_auc = metrics.auc(fpr, tpr)
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve') #Receiver Operating Characteristic 
    plt.legend(loc="lower right")

svm_cls_pred=svm_cls.predict(X_test_std)
mlp_pred=mlp.predict(X_test_std)
dt_pred=dt.predict(X_test_std)
ada_pred=ada.predict(X_test_std)
knn_pred=knn.predict(X_test_std)
rf_pred=rf.predict(X_test_std)
lr_pred=lr.predict(X_test_std)


svm_cls_pred_proba=svm_cls.decision_function(X_test_std)
mlp_pred_proba=mlp.predict_proba(X_test_std)[:,1]
dt_pred_proba=dt.predict_proba(X_test_std)[:,1]
ada_pred_proba=ada.predict_proba(X_test_std)[:,1]
knn_pred_proba=knn.predict_proba(X_test_std)[:,1]
rf_pred_proba=rf.predict_proba(X_test_std)[:,1]
lr_pred_proba=lr.predict_proba(X_test_std)[:,1]


plotAUC(y_test,svm_cls_pred_proba,'SVM cls')
plotAUC(y_test,mlp_pred_proba,'MLP')
plotAUC(y_test,dt_pred_proba,'DT')
plotAUC(y_test,ada_pred_proba,'ADA')
plotAUC(y_test,knn_pred_proba,'KNN')
plotAUC(y_test,rf_pred_proba,'RF')
plotAUC(y_test,lr_pred_proba,'LR')
plt.show()

svm_cls_pred=svm_cls.predict(X_train_std)
mlp_pred=mlp.predict(X_train_std)
dt_pred=dt.predict(X_train_std)
ada_pred=ada.predict(X_train_std)
knn_pred=knn.predict(X_train_std)
rf_pred=rf.predict(X_train_std)
lr_pred=lr.predict(X_train_std)


svm_cls_pred_proba=svm_cls.decision_function(X_train_std)
mlp_pred_proba=mlp.predict_proba(X_train_std)[:,1]
dt_pred_proba=dt.predict_proba(X_train_std)[:,1]
ada_pred_proba=ada.predict_proba(X_train_std)[:,1]
knn_pred_proba=knn.predict_proba(X_train_std)[:,1]
rf_pred_proba=rf.predict_proba(X_train_std)[:,1]
lr_pred_proba=lr.predict_proba(X_train_std)[:,1]


plotAUC(y_train,svm_cls_pred_proba,'SVM cls')
plotAUC(y_train,mlp_pred_proba,'MLP')
plotAUC(y_train,dt_pred_proba,'DT')
plotAUC(y_train,ada_pred_proba,'ADA')
plotAUC(y_train,knn_pred_proba,'KNN')
plotAUC(y_train,rf_pred_proba,'RF')
plotAUC(y_train,lr_pred_proba,'LR')
plt.show()

svm_cls_pred=svm_cls.predict(X_std)
mlp_pred=mlp.predict(X_std)
dt_pred=dt.predict(X_std)
ada_pred=ada.predict(X_std)
knn_pred=knn.predict(X_std)
rf_pred=rf.predict(X_std)
lr_pred=lr.predict(X_std)


svm_cls_pred_proba=svm_cls.decision_function(X_std)
mlp_pred_proba=mlp.predict_proba(X_std)[:,1]
dt_pred_proba=dt.predict_proba(X_std)[:,1]
ada_pred_proba=ada.predict_proba(X_std)[:,1]
knn_pred_proba=knn.predict_proba(X_std)[:,1]
rf_pred_proba=rf.predict_proba(X_std)[:,1]
lr_pred_proba=lr.predict_proba(X_std)[:,1]


plotAUC(y,svm_cls_pred_proba,'SVM cls')
plotAUC(y,mlp_pred_proba,'MLP')
plotAUC(y,dt_pred_proba,'DT')
plotAUC(y,ada_pred_proba,'ADA')
plotAUC(y,knn_pred_proba,'KNN')
plotAUC(y,rf_pred_proba,'RF')
plotAUC(y,lr_pred_proba,'LR')
plt.show()

print("Confusion matrix for svm_cls on test:\n%s" % (confusion_matrix(y_test, svm_cls.predict(X_test_std))))
print("Confusion matrix for dt on test:\n%s" % (confusion_matrix(y_test, dt.predict(X_test_std))))
print("Confusion matrix for ada on test:\n%s" % (confusion_matrix(y_test, ada.predict(X_test_std))))
print("Confusion matrix for mlp on test:\n%s" % (confusion_matrix(y_test, mlp.predict(X_test_std))))
print("Confusion matrix for rf on test:\n%s" % (confusion_matrix(y_test, rf.predict(X_test_std))))
print("Confusion matrix for knn on test:\n%s" % (confusion_matrix(y_test, knn.predict(X_test_std))))
print("Confusion matrix for lr on test:\n%s" % (confusion_matrix(y_test, lr.predict(X_test_std))))

print("Confusion matrix for svm_cls on train:\n%s" % (confusion_matrix(y_train, svm_cls.predict(X_train_std))))
print("Confusion matrix for dt on train:\n%s" % (confusion_matrix(y_train, dt.predict(X_train_std))))
print("Confusion matrix for ada on train:\n%s" % (confusion_matrix(y_train, ada.predict(X_train_std))))
print("Confusion matrix for mlp on train:\n%s" % (confusion_matrix(y_train, mlp.predict(X_train_std))))
print("Confusion matrix for rf on train:\n%s" % (confusion_matrix(y_train, rf.predict(X_train_std))))
print("Confusion matrix for knn on train:\n%s" % (confusion_matrix(y_train, knn.predict(X_train_std))))
print("Confusion matrix for lr on train:\n%s" % (confusion_matrix(y_train, lr.predict(X_train_std))))

print("Confusion matrix for svm_cls on total:\n%s" % (confusion_matrix(y, svm_cls.predict(X_std))))
print("Confusion matrix for dt on total:\n%s" % (confusion_matrix(y, dt.predict(X_std))))
print("Confusion matrix for ada on total:\n%s" % (confusion_matrix(y, ada.predict(X_std))))
print("Confusion matrix for mlp on total:\n%s" % (confusion_matrix(y, mlp.predict(X_std))))
print("Confusion matrix for rf on total:\n%s" % (confusion_matrix(y, rf.predict(X_std))))
print("Confusion matrix for knn on total:\n%s" % (confusion_matrix(y, knn.predict(X_std))))
print("Confusion matrix for lr on total:\n%s" % (confusion_matrix(y, lr.predict(X_std))))

svm_cls_pred_proba_test=ada.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=ada.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=ada.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'ADA cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'ADA cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'ADA cls train')
plt.show()



svm_cls_pred_proba_test=dt.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=dt.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=dt.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'dt cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'dt cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'dt cls train')
plt.show()

svm_cls_pred_proba_test=knn.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=knn.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=knn.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'knn cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'knn cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'knn cls train')
plt.show()

svm_cls_pred_proba_test=mlp.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=mlp.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=mlp.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'mlp cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'mlp cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'mlp cls train')
plt.show()

svm_cls_pred_proba_test=lr.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=lr.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=lr.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'lr cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'lr cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'lr cls train')
plt.show()

svm_cls_pred_proba_test=rf.predict_proba(X_test_std)[:,1]
svm_cls_pred_proba_train=rf.predict_proba(X_train_std)[:,1]
svm_cls_pred_proba=rf.predict_proba(X_std)[:,1]

plotAUC(y,svm_cls_pred_proba,'rf cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'rf cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'rf cls train')
plt.show()

svm_cls_pred_proba_test=svm_cls.decision_function(X_test_std)
svm_cls_pred_proba_train=svm_cls.decision_function(X_train_std)
svm_cls_pred_proba=svm_cls.decision_function(X_std)

plotAUC(y,svm_cls_pred_proba,'svm cls total')
plotAUC(y_test,svm_cls_pred_proba_test,'svm cls test')
plotAUC(y_train,svm_cls_pred_proba_train,'svm cls train')
plt.show()



import threading
from queue import Queue
q=Queue()
NO_OF_THR=7
JOB_NUM=[1,2,3,4,5,6,7]

def prelims():
    global X_train_std
    global y_train
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    np.random.seed(seed=2017)
    #reading data set
    dataset=pd.read_csv('loan-prediction.csv')
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
    dataset.to_csv("loan-prediction1.csv")
    X = dataset.iloc[:,1:-1]
    y = dataset.iloc[:,-1]
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(criterion='entropy',random_state=5)
    rfe = RFE(dt, 8)
    rfe = rfe.fit(X,y)
    X_new=rfe.fit_transform(X,y)
    import pandas as pd
    X_df=pd.DataFrame(X_new)
    X_df.head()
    X_df.to_csv('featured_X_dataframe.csv',index=False)
    X_df=pd.read_csv('featured_X_dataframe.csv')
    X=X_df
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X_std=sc.transform(X)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2017)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    #cross validation and accuracy measures
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score
    from sklearn.externals import joblib #dumping and loading model
def func_svm():
    global X_train_std
    global y_train
    from sklearn import svm 
    svm_cls = svm.SVC(gamma=0.05,C=100,kernel='rbf')
    svm_cls.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(svm_cls,'svm_cls_model.joblib')
def func_dt():
    global X_train_std
    global y_train
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(criterion='gini',random_state=5)
    dt.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(dt,'dt_model.joblib')
def func_ada():
    global X_train_std
    global y_train
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(criterion='gini',random_state=5)
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=1000,learning_rate=0.01, random_state=0)
    ada.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(ada,'ada_model.joblib')
def func_rf():
    global X_train_std
    global y_train
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=5,max_features='auto')
    rf.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(rf,'rf_model.joblib')
def func_mlp():
    global X_train_std
    global y_train
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(20,2), activation='logistic',max_iter=500,learning_rate_init=0.005,tol=1e-41,solver='adam')
    mlp.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(mlp,'mlp_model.joblib')
def func_knn():
    global X_train_std
    global y_train
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn=KNN(n_neighbors=3)
    knn.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(knn,'knn_model.joblib')
def func_lr():
    global X_train_std
    global y_train
    from sklearn.linear_model import LogisticRegression as LR
    lr=LR(C=1.0,solver='newton-cg',penalty='l2',multi_class='ovr',max_iter=500)
    lr.fit(X_train_std, y_train)
    from sklearn.externals import joblib #dumping and loading model
    joblib.dump(lr,'lr_model.joblib')
def create_workers():
    for x in range(NO_OF_THR):
        t=threading.Thread(target=work)
        t.daemon=True
        t.start()
def work():
    while True:
        x=q.get()
        if x==1:
            func_svm()
        if x==2:
            func_dt()
        if x==3:
            func_ada()
        if x==4:
            func_rf()
        if x==5:
            func_mlp()
        if x==6:
            func_knn()
        if x==7:
            func_lr()
        q.task_done()
def create_jobs():
    for x in JOB_NUM:
        q.put(x)
    q.join()
prelims()
create_workers()
create_jobs()

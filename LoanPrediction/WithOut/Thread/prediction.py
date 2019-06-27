#using the trained model
from sklearn.externals import joblib
import pandas as pd
svm_cls=joblib.load('svm_cls_model.joblib')
dt=joblib.load('dt_model.joblib')
mlp=joblib.load('mlp_model.joblib')
ada=joblib.load('ada_model.joblib')
knn=joblib.load('knn_model.joblib')
lr=joblib.load('lr_model.joblib')
rf=joblib.load('rf_model.joblib')
import statistics as stat

import pandas as pd

df=pd.DataFrame(columns=['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
                         'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',
                        'Loan_Status'])

dataset=pd.read_csv('loan-prediction1.csv')

for x in dataset.columns:
    df[x]=df[x].astype(dataset[x].dtypes)
X = pd.read_csv('featured_X_dataframe.csv')
y = dataset.iloc[:,-1]
def max_count(f):
    a,b,c=0,0,0
    a=f.count(1)
    b=f.count(0)
    if a>b:
        c=1
    else:
        c=0
    return c
def probab_count(f):
    a,b,c,d=0,0,0,0
    a=f.count(1)
    b=f.count(0)
    c=((a/7)*100)
    d=((b/7)*100)
    print("%f percent probability of getting loan"%(c))
    print("%f percent probability of not getting loan"%(d))
    
cc=0
def pred_preparation(V,a):
    global cc
    count=0
    arr1=[]
    f=[]
    arr=a
    score=[]
    model=[]
    df.loc[cc]=['predicted_'+str(cc)]+V+[0]
    for i in X.columns:
        p=[]
        for j in range(X.shape[0]):
            p.append(X[i][j])
        p.append(arr[count])
        s=stat.stdev(p)
        u=stat.mean(p)
        if s!=0:
            arr[count]=((arr[count]-u)/s)
        else:
            arr[count]=(arr[count]-u)
        count+=1
    
    model=[svm_cls,dt,ada,rf,lr,knn,mlp]
    for cou in model:
        f.extend(cou.predict([arr])) 
    probab_count(f)
    print(f)
    f_pred=max_count(f)
    df['Loan_Status'][cc]=f_pred
    cc+=1
    print(f_pred)
def arr_prep(V):
    a=[]
    for i in range(len(V)):
        if i==0:
            pass
        elif i==1:
            pass
        elif i==3:
            pass
        else:
            a.append(V[i])
    pred_preparation(V,a)    
    
arr_prep([1,0,0,1,0,5849,0.0,146.412162,360.0,1.0,2])#y
arr_prep([1,1,1,1,0,12841,10968.0,349.000000,360.0,1.0,1])#n
arr_prep([1,0,0,1,0,5849,0.0,146.412162,360.0,1.0,2])#y
arr_prep([1,1,1,1,0,12841,10968.0,349.000000,360.0,1.0,1])#n
arr_prep([1,1,0,1,0,1809,1868.0,90.000000,360.0,1.0,2])#y
arr_prep([1,1,0,1,0,2083,3150.0,128.000000,360.0,1.0,1])#y


df.to_csv('predicted_loan_status.csv')

#map function from interface
def set_list(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,
                         CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
    x=[]
    x.append(Gender)
    x.append(Married)
    x.append(Dependents)
    x.append(Education)
    x.append(Self_Employed)
    x.append(ApplicantIncome)
    x.append(CoapplicantIncome)
    x.append(LoanAmount)
    x.append(Loan_Amount_Term)
    x.append(Credit_History)
    x.append(Property_Area)
    pred_preparation(x)
    

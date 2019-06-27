import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
m,n=np.shape(array)
shape=m*n
m=m-1
n=n-1
k=0
l=0
j=1
while j<=shape:
    for i in range(l,n+1):
        print(array[k][i])
    k=k+1
    for i in range(k,m+1):
        print(array[i][n])
    n=n-1
    for i in range(n,l-1,-1):
        print(array[m][i])
    m=m-1
    for i in range(m,k-1,-1):
        print(array[i][l])
    l=l+1
    j=j+1

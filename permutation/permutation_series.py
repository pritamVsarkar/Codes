word=input("enter the number :")
length=len(word)

# value count pair
string = []
count = []
ll=[]

result=['' for i in range(len(word))] #output array

for i in word:
    if i not in ll:
        ll.append(i)
        string.append(i)
        count.append(word.count(i))
        
def permutation(string,count,result,level):
    if level==len(result):
        print(result)
        return
    for i in range(len(string)): #next stage left to right traversal #same stage oth to l->r traversal
        if count[i]==0:
            continue;
        result[level]=string[i]
        count[i]-=1
        permutation(string,count,result,level+1)
        count[i]+=1
    return


permutation(string,count,result,0)

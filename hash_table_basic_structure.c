#include <stdio.h>
struct pair{
    int value;
    int key;
};
void display(struct pair ar[],int size){
    int i;
    printf("key\tvalue\n");
    for(i=0;i<size;i++){
        printf("%d %d \n",ar[i].key,ar[i].value);
    }
}
int main(){
    int size,i,temp;
    printf("Enter the size of the table :\n");
    scanf("%d",&size);
    struct pair hash[size];
    for(i=0;i<size;i++){
        scanf("%d",&temp);
        hash[temp % size].value=temp;
        hash[temp%size].key=temp % size;
    }
    printf("\n");
    display(hash,size);
    return 0;
}

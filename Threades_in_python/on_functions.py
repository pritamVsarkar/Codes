import threading
from time import sleep
from queue import Queue
queue=Queue()
NO_OF_THREADES=3
JOBS=[1,2,3]

def func1():
    for i in range(10):
        print("P"+str(i))
def func2():
    for i in range(10):
        print("Q"+str(i))
def func3():
    for i in range(10):
        print("R"+str(i))

def create_workers(): #creating threades to perform each tasks
    for _ in range(NO_OF_THREADES):
        t=threading.Thread(target=work)
        t.daemon=True
        t.start()
        sleep(1)
        
def work(): #which work will be done by which thread
    while True:
        x=queue.get()
        if x==1:
            func1()
        if x==2:
            func2()
        if x==3:
            func3()
        queue.task_done()
        
def create_jobs(): #which puts the list into array
    for i in JOBS:
        queue.put(i)
    queue.join()
    
create_workers()
create_jobs()

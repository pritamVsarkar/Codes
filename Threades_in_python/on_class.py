from time import sleep
from threading import *
class C1(Thread):
    def run(self):
        for i in range(0,5):
            print('pritam')
            sleep(1)
class C2(Thread):
    def run(self):
        for i in range(0,5):
            print('sarkar')
            sleep(1)
c1=C1()
c2=C2()
c1.start()
sleep(0.2)
c2.start()

import numpy as np
import random

def generator():
    path = []
    res = 0
    pos = random.randint(0,35)
    path.append(pos)
    while(pos!=1 and pos!=35):
        direct = random.randint(0,3)
        if(direct==0):
            if(pos not in [0,1,2,3,4,5]):
                pos-=6
        if(direct==1):
            if(pos not in [0,6,12,18,24,30]):
                pos-=1
        if(direct==2):
            if(pos not in [30,31,32,33,34,35]):
                pos+=6
        if(direct==3):
            if(pos not in [5,11,17,23,29,25]):
                pos+=1
        if(pos!=1 and pos!=35):
            res-=1
        path.append(pos)
        # print(pos,end=" ")
    return path,res

# a,b=generator()
# print(a,b)
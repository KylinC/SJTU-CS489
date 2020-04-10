import numpy as np
import random
from self_gridworld import generator

sheld_times = 100000
state_value = np.zeros(36)
access_time = np.zeros((36,), dtype = np.int) 

for i in range(sheld_times):
    sequence,g=generator()
    sequence_set = set(sequence)
    for item in range(36):
        if(item in sequence_set):
            access_time[item]+=1
            state_value[item]=state_value[item]+(1/access_time[item])*(g-state_value[item])

for i in range(36):
    print(state_value[i],end="  ")
    if(i%6==5):
        print('')
import numpy as np
import random
from self_gridworld import generator

sheld_times = 1000000
state_value = np.zeros(36)
alpha=0.1
gamma=0.7

for i in range(sheld_times):
    sequence,g=generator()
    for item_index in range(len(sequence)-1):
        if(sequence[item_index+1]==1 or sequence[item_index+1]==35):
            next_value = 0 + gamma*state_value[sequence[item_index+1]]
        else:
            next_value = -1 + gamma*state_value[sequence[item_index+1]]
        state_value[sequence[item_index]]=state_value[sequence[item_index]]+alpha*(next_value-state_value[sequence[item_index]])
        

for i in range(36):
    print(state_value[i],end="  ")
    if(i%6==5):
        print('')
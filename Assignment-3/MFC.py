import numpy as np
import random
import matplotlib.pyplot as plt

alpha = 0.5
epsilon = 0.1
gamma = 1
x_length = 12
y_length = 4
episodes = 500
batch_size = 20

def observe(x,y,a):
    goal = 0
    if x == x_length - 1 and y == 0:
        goal = 1
    if a == 0:
        y += 1
    if a == 1:
        x += 1
    if a == 2:
        y -= 1
    if a == 3:
        x -= 1
        
    x = max(0,x)
    x = min(x_length-1, x)
    y = max(0,y)
    y = min(y_length-1, y)

    if goal == 1:
        return x,y,-1
    if x>0 and x<x_length-1 and y==0:
        return 0,0,-100
    return x,y,-1

def epsilon_policy(x,y,q,eps):
    t = random.randint(0,3)
    if random.random() < eps:
        a = t
    else:
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
    return a

def max_q(x,y,q):
    q_max = q[x][y][0]
    a_max = 0
    for i in range(4):
        if q[x][y][i] >= q_max:
            q_max = q[x][y][i]
            a_max = i
    a = a_max
    return a

actionRewards = np.zeros((x_length, y_length, 4))
actionRewards[:, :, :] = -1.0
actionRewards[1:x_length-1, 1, 2] = -100.0
actionRewards[0, 0, 1] = -100.0

actionDestination = []
for i in range(0, x_length):
    actionDestination.append([])
    for j in range(0, y_length):
        destination = dict()
        destination[0] = [i, min(j+1,y_length-1)]
        destination[1] = [min(i+1,x_length-1), j]
        if 0 < i < x_length-1 and j == 1:
            destination[2] = [0,0]
        else:
            destination[2] = [i, max(j - 1, 0)]
        destination[3] = [max(i-1,0), j]
        actionDestination[-1].append(destination)
actionDestination[0][0][1] = [0,0]


def sarsa_on_policy(q):
    rewards = np.zeros([episodes])
    for j in range(batch_size):
        for i in range(episodes):
            reward_sum = 0
            x = 0
            y = 0
            a = epsilon_policy(x,y,q,epsilon)
            while True:
                [x_next,y_next] = actionDestination[x][y][a]
                reward = actionRewards[x][y][a]
                reward_sum += reward
                a_next = epsilon_policy(x_next,y_next,q,epsilon)
                q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next]-q[x][y][a])
                if x == x_length - 1 and y==0:
                    break
                x = x_next
                y = y_next
                a = a_next
            rewards[i] += reward_sum
    rewards /= batch_size
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards

def q_learning(q):
    rewards = np.zeros([500])
    for j in range(batch_size):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            while True:
                a = epsilon_policy(x,y,q,epsilon)             
                x_next, y_next,reward = observe(x,y,a)
                a_next = max_q(x_next,y_next,q)
                reward_sum += reward
                q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next]-q[x][y][a])
                if x == x_length - 1 and y==0:
                    break
                x = x_next
                y = y_next
            rewards[i] += reward_sum
    rewards /= batch_size
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards


def OptimalPath(q):
    x = 0
    y = 0
    path = np.zeros([x_length,y_length]) - 1
    end = 0
    exist = np.zeros([x_length,y_length])
    while (x != x_length-1 or y != 0) and end == 0:
        a = max_q(x,y,q)
        path[x][y] = a
        if exist[x][y] == 1:
            end = 1
        exist[x][y] = 1
        x,y,r = observe(x,y,a)
    for j in range(y_length-1,-1,-1):
        for i in range(x_length):
            if i == x_length-1 and j == 0:
                print("G",end = "\t")
                continue
            a = path[i,j]
            if a == -1:
                print("0",end = "\t")
            elif a == 0:
                print("↑",end = "\t")
            elif a == 1:
                print("→",end = "\t")
            elif a == 2:
                print("↓",end = "\t")
            elif a == 3:
                print("←",end = "\t")
        print("")

def generate_heatmap(q_table):
    import seaborn as sns; sns.set()
    data = np.mean(q_table, axis = 2)
    data=np.swapaxes(data,1,0)
    f, ax = plt.subplots(figsize=(9, 6))
    ax = sns.heatmap(np.array(data)).invert_yaxis()
    plt.show()
    return ax

if __name__ == '__main__':
    SSq = np.zeros([12,4,4])
    QLq = np.zeros([12,4,4])

    sarsa_rewards = sarsa_on_policy(SSq)
    q_learning_rewards = q_learning(QLq)

    plt.plot(range(len(sarsa_rewards)),sarsa_rewards,label="Sarsa")
    plt.plot(range(len(sarsa_rewards)),q_learning_rewards,label="Q-learning")
    plt.ylim(-100,0)
    plt.legend(loc="lower right")
    plt.show()

    print("optimal travel path by Sara:")
    OptimalPath(SSq)
    print("optimal travel path by Q-Learning:")
    OptimalPath(QLq)

    generate_heatmap(SSq)
    generate_heatmap(QLq)


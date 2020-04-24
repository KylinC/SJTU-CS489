

# RL Assignment 1

> 陈麒麟 517030910155



[TOC]

**作业要求：**

- 实现GridWorld类
- 用策略迭代和策略评估优化随机策略

**作业完成：**

- 实现Sutton版RL提供的GirdWorld类
- 按要求用策略迭代和策略评估优化随机策略
- 实现值迭代并进行迭代次数与终态对比



## 代码实现

### GridWorld类

> 代码见 *gridworld.py*

该类实现了一个可以实例化的格子世界，概率转矩阵P、reward=$-1$已在其中声明，通过以下方法实例化：

```python
env = GridworldEnv([m,n])
```

(m,n为自定义格子世界尺寸，默认出口在左上和右下)



### PolicyIteration.py

> 代码见 *PolicyIteration.py*

- 首先使用GridWorld类实例化6x6的gridworld，之后通过函数 *policy_iteration* 集成策略迭代和策略评估：

```python
def policy_iteration(env, theta=0.001, discount_factor=1.0):
    """
    Policy Iteration Algorithm.
    
    Args:
        env: gridWorld
```

- 通过 *one_step_lookahead(state, V)* 按0.25的等概率计算更新后的值函数：

```python
def one_step_lookahead(state, V):

	A = 0.0
  for a in range(env.nA):
  for prob, next_state, reward, done in env.P[state][a]:
  A += 0.25 * (reward + discount_factor * V[next_state])
  return A
```

- 在theta=0.001的更新阈值下不断迭代

```python
V = np.zeros(env.nS)
Vtmp = np.zeros(env.nS)
iteration_step = 0
while True:
  iteration_step += 1
  # Stopping condition
  delta = 0
  # Update each state...
  for s in range(env.nS):
    # Calculate the new value
    new_action_value = one_step_lookahead(s, V)
    # Calculate delta across all states seen so far
    delta = max(delta, np.abs(new_action_value - V[s]))
    # Update the value function
    Vtmp[s] = new_action_value        
      # Check if we can stop 
      if delta < theta:
        print("iterations:",iteration_step)
        break
      else:
        V = Vtmp
```

- 通过 *greedy_policy_choose(state, V)* 进行一步的greedy策略选择：

```python
def greedy_policy_choose(state, V):
	A = np.zeros(env.nA)
	for a in range(env.nA):
		for prob, next_state, reward, done in env.P[state][a]:
			A[a] = V[next_state]
  return np.argmax(A)
```



### ValueIteration.py

> 代码见 *ValueIteration.py*

与策略迭代类似，但是在每一次值迭代时选取最优策略，详见代码。



## 测试

### 6x6 Policy Iteration 结果

（最优策略、状态矩阵）

> theta=0.001 下迭代次数为259

![截屏2020-04-02 下午11.42.19](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-02-154232.png)



### 6x6 Value Iteration 结果

（最优策略、状态矩阵）

> theta=0.001 下迭代次数为6

![截屏2020-04-02 下午11.44.12](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-02-154420.png)



### 4x4 Policy Iteration 结果（验证）

（最优策略、状态矩阵）

> theta=0.001 下迭代次数为89

![截屏2020-04-02 下午11.45.39](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-02-154559.png)

与textbook上结果一致，故验证通过：

![截屏2020-04-02 下午11.47.31](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-02-154738.png)

## 结论

Policy迭代与评估方法实现正确，而且其迭代次数比value迭代要多。


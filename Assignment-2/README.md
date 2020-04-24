# RL Assignment 2

> 陈麒麟 517030910155



[TOC]

**作业要求**：

- 利用MC-first visit\every visit、TD0算法估计6x6 gridworld(1、35为出口) 的状态值



## 代码实现

### generator方法

> 代码见 *self_gridworld.py* 

产生一个6x6 gridworld的随机episode，调用如下：

```python
sequence,g=generator()
```



### MC1文件

> 代码见 *MC1.py* 

文件为MC-first visit方法实现并输出最终状态矩阵，其中迭代次数可以控制，更新公式为：

![截屏2020-04-09 下午10.54.13](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-145426.png)



### MC2文件

> 代码见 *MC2.py* 

文件为MC-every visit方法实现并输出最终状态矩阵，其中迭代次数可以控制，更新公式同上，区别在于每一个episode中可以更新同一个状态两次。



### TD0文件

> 代码见 *TD0.py* 

文件为TD(0)方法实现并输出最终状态矩阵，其中迭代次数/alpha=0.1/gamma=0.7可以控制，更新公式为：

![截屏2020-04-09 下午10.57.43](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-145752.png)



## 测试

### MC-First Visit

10000次迭代后结果：

![截屏2020-04-09 下午10.14.20](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141438.png)

100000次迭代后结果1：

![截屏2020-04-09 下午10.15.19](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141551.png)

100000次迭代后结果2：

![截屏2020-04-09 下午10.15.44](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141558.png)

可以看出，10000次迭代时已经收敛，而且其最优方向是与直观一致。



###MC-Every Visit 

10000次迭代后结果：

![截屏2020-04-09 下午10.18.56](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141926.png)

100000次迭代后结果1：

![截屏2020-04-09 下午10.19.09](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141933.png)

100000次迭代后结果2：

![截屏2020-04-09 下午10.19.19](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-141940.png)

相比MC-first visit，其状态值偏大，因为每个episode可以计入多次同一状态值。且在10000次迭代时已经收敛，其最优方向是与直观一致。



### TD0

> alpha=0.1   gamma=0.7

10000次迭代后结果：

![截屏2020-04-09 下午10.33.53](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-150227.png)

100000次迭代后结果1：

![截屏2020-04-09 下午10.34.33](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-150234.png)

100000次迭代后结果2：

![截屏2020-04-09 下午10.39.41](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2020-04-09-150243.png)

相比MC方法，其收敛较慢，因为每次更新都是由非episode-compete进行更新，存在误差大于MC。
---
layout:     post
title:      多智能体强化学习算法
subtitle:   VDN：Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward
date:       2020-08-07
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: false
tags:
    - RL
    - MARL
    - RL advanced algorithms
    - Learning Cooperation
---


------

论文链接：<a href="https://arxiv.org/pdf/1706.05296.pdf">Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward, AAMAS 2018</a>


## 一、问题

多智能体强化学习的传统解决方案有两种：
  - 集中式：通过将各个智能体的状态空间与动作空间组合为联合状态空间与联合动作空间进行集中式训练，将多智能体训练视为单智能体训练
  - 独立式：每个智能体使用单智能体的方法独立训练

这两种方法各自具有缺陷：

  - 集中式的方法在训练两个智能体时往往会产生"lazy agent"的问题，即只有一个智能体活跃，而另一个变得很懒。这是因为当一个智能体学到有效的策略后，另一个智能体会因为不想干扰已经学好的智能体的策略导致整个团队的收益降低，从而而不愿意学习。
   - 独立式的方法在训练时很困难，在许多简单环境中都没有效果。因为每个智能体面对的环境是动态的（另一个智能体的行为成为了环境中的动态部分），而且每个智能体可能会收到由其他智能体执行且自己观察不到的动作所产生的假的奖励信号。

为了解决现有两类方法的缺陷，本文提出了值函数分解的方法，通过learning的方式将团队整体奖励中分解为每个智能体自己的奖励。

## 二、解法

如果想要使用基于价值的强化学习模型（value-based），就需要对系统的联合动作-价值函数（joint action-value function，联合Q函数）建模。假设系统中有$d$个智能体，则联合$Q$函数可以表示为$Q\left(\left(h^{1}, h^{2}, \ldots, h^{d}\right),\left(a^{1}, a^{2}, \ldots, a^{d}\right)\right)$，其中$h_{i}$表示智能体的局部信息，$a_{i}$表示动作。可以看出，如果使用一般的方法建模，Q函数的输出维度是$d_{n}$，$n$是动作空间的维度。

Value-Decomposition Networks（VDN）的基本假设是，系统的联合$Q$函数可以近似为多个单智能体的Q函数的和：$$
Q\left(\left(h^{1}, h^{2}, \ldots, h^{d}\right),\left(a^{1}, a^{2}, \ldots, a^{d}\right)\right) \approx \sum_{i=1}^{d} \tilde{Q}_{i}\left(h^{i}, a^{i}\right)$$

其中$Q_i$之间是独立的，只取决于局部观策和动作$h_i$,$a_i$。这样可以保证，最大化每个单智能体的$\tilde{Q}\_{i}$函数得到动作，与通过最大化联合$Q$函数得到的结果是一样的，即
$$\begin{array}{c}{\max_{a} Q=\max_{a} \sum_{i=1}^{d} \tilde{Q}_{i}=\sum_{i=1}^{d} \max_{a_{i}} \tilde{Q}_{i}} \\ {\operatorname{argmax}_{a} Q=\left(\operatorname{argmax}_{a_{i}} \tilde{Q}_{i}\right)}\end{array}$$

我们可以通过全局的奖励函数，间接地训练每个单智能体的$Q$函数。并且只要对每个智能体选择最大化$\tilde{Q}_{i}$的动作，就能使得全局$Q$值最大。

同时，使用以下的技巧，VDN能够得到更好的效果：
  * 使用DRQN作为Q函数。DRQN是一个用来处理POMDP（部分可观马尔可夫决策过程）的一个算法，其采用LSTM替换DQN卷基层后的一个全连接层，来达到能够记忆历史状态的作用，因此可以在部分可观的情况下提高算法性能。经试验证明，使用LSTM作为Q函数网络的输出层在PODMP环境下有更强的鲁棒性和泛化能力。
  * 参数共享（weight sharing）和角色信息（role information）。即所有的单智能体使用同一个Q函数网络，同时在输入中加入一个one-hot 向量来区别不同角色的智能体。
  * 智能体间的通信（information channels）。本文试验了两种通信方式，一种是在输入时就包括了其他智能体的信息（low level），另一种是结合Q网络的高层隐藏变量（high level）再分别输出。

## 三、实验内容 

实验对比了不同训练方式（集中训练分步执行、集中式、独立式）和不同通信方式（VDN,VDN+low level,VDN+high level的网络结构）的效果。不同网络结构中在low level做通信的结构学习速度比在high level通信更快。集中训练分步执行效果好于传统方式。三种通信方式结构如下图所示：

![aWeJHA.png](https://s1.ax1x.com/2020/08/07/aWeJHA.png)

![aWetAI.png](https://s1.ax1x.com/2020/08/07/aWetAI.png)

![aWeGBd.png](https://s1.ax1x.com/2020/08/07/aWeGBd.png)



## 四、缺点

  * 多智能体实验环境简单
  * VDN可行的原因可以总结为一个公式：$\operatorname{argmax}\_{a} Q=\left(\operatorname{argmax}\_{a_{i}} \tilde{Q}\_{i}\right)$。VDN中联合函数的表达形式（求和）满足这个条件，但求和这种方式表现力有限，并不能涵盖更加复杂的组合情况，比如非线性组合。

## 五、优点

对于value-based方法，设计了中心化计算系统的$Q$函数，单智能体的$Q_{i}$函数去中心化。这种结构使得策略在训练的时候可以利用全局信息，同时每个智能体仍然只接受局部信息作为输入。

---
layout:     post
title:      多智能体强化学习算法
subtitle:   QMIX：Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
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

论文链接：<a href="https://arxiv.org/pdf/1803.11485.pdf">QMIX：Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning, ICML 2018</a>


## 一、问题

IQL（independent Q-learning）是非常暴力的给每个智能体执行一个Q-learning算法，因为共享环境，并且环境随着每个智能体策略、状态发生改变，对每个智能体来说，环境是动态不稳定的，因此这个算法很难收敛。

Value-Decomposition Networks（VDN）的基本假设是，系统的联合$Q$函数可以近似为多个单智能体的$Q$函数的和。但VDN中联合函数的求和形式表现力有限，并不能涵盖更加复杂的组合情况，比如非线性组合。

## 二、解法

事实上，VDN可行的原因是满足如下条件：

$$
\underset{\mathbf{u}}{\operatorname{argmax}} Q_{t o t}(\tau, \mathbf{u})=\left(\begin{array}{c}{\operatorname{argmax}_{u^{1}} Q_{1}\left(\tau^{1}, u^{1}\right)} \\ {\vdots} \\ {\operatorname{argmax}_{u^{n}} Q_{n}\left(\tau^{n}, u^{n}\right)}\end{array}\right)
$$


为了满足这个条件，只要满足$Q_{t o t}$对于任意一个$Q$是单调增的（monotonic）即可：

$$
\frac{\partial Q_{t o t}}{\partial Q_{a}} \geq 0, \forall a \in A
$$

可以看出，VDN中的求和形式是该条件的一个特例$（\frac{\partial Q_{t o t}}{\partial Q_{a}} = 1, \forall a \in A）$。QMIX 模型的核心思想就是在$Q$和$Q_{i}$之间构造一个单调映射。

QMIX模型由两部分组成，agent networks输出单智能体的$Q_{i}$函数，mix network 以$Q_{i}$作为输入，用于计算联合$Q$函数。为了满足单调的限制，mix network每一层的参数均由一个hypernetwork计算生成。hypernetworks以当前状态s作为输入，输出一个与mix network当前层形状相匹配的向量作为参数。hypernetworks隐藏层的激活函数是非负的（比如ReLU），这样得到的mix network每一层的参数也都是非负的，从而保证了单调的条件。同时使用hypernetwork相比于直接在网络中做约束会更灵活，没有bias限制。QMIX模型如下：

<img src="https://s1.ax1x.com/2020/08/07/aWmLZj.png" alt="aWmLZj.png" style="zoom:80%;" />

与VDN相比，QMIX改进了联合$Q$函数的形式，融合了部分全局信息$s_{t}$，使得模型的性能有很大的提升。这里需要注意的是，agent networks并没有使用全局信息，所以这还是一个POMDP环境。

QMIX的优化目标是最小化如下loss：

$$\mathcal{L}(\theta)=\sum_{i=1}^{b}\left[\left(y_{i}^{t o t}-Q_{t o t}(\tau, \mathbf{u}, s ; \theta)\right)^{2}\right]$$

由于满足上文的单调性约束，对$Q_{t o t}$进行$argmax$操作的计算量就不再是随智能体数量呈指数增长了，而是随智能体数量线性增长，极大的提高了算法效率。

## 三、实验内容 

实验在星际二上进行，实验效果为QMIX好于VDN好于IQL

<img src="https://s1.ax1x.com/2020/08/07/aWmOds.png" alt="aWmOds.png" style="zoom:80%;" />



## 四、缺点

每个智能体观察到的都是自己的历史，而没有考虑其他智能体的观察和动作，对于困难得到合作任务可能无法将联合$Q$函数完美地分解为独立的$Q_{i}$

## 五、优点

* QMIX提出了一种通用性更强的值函数分解方案，改进了VDN联合Q函数的形式
* 在mix network融合了部分全局信息提高了效果
* 实验环境难度有较大提升
* 使用hypernetwork是一种很巧妙的方式，值得借鉴。

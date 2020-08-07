---
layout:     post
title:      单智能体强化学习算法
subtitle:   Dueling DQN：Dueling Network Architectures for Deep Reinforcement Learning
date:       2020-08-03
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - RL advanced algorithms
    - Model-Free RL
    - Deep Q-Learning
---
---

论文链接：<a href="https://arxiv.org/pdf/1511.06581.pdf">Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016</a>

## 一、问题 
智能体在与环境做互动的过程中，有些状态对应的动作对环境没任何影响。即在某些状态$s_t$下，无论做什么动作$a_t$，对下一个状态$s_{t+1}$都没多大影响，当前状态动作函数也与当前动作选择不太相关。如下图所示：从第一行的红光区域可以看出，state function专注于路面情况，特别是水平面方向，因为水平面方向极可能会出现汽车，而advantage function未专注于路面情况，因为当路面没有汽车时，它选的任何动作所获得的reward都差不多；从第二行的红光区域可以看出，advantage function专注于前方的小车，此时它所选的动作非常重要。

<img src="https://s1.ax1x.com/2020/08/03/aa16JA.png" alt="1" style="zoom: 40%;" />

## 二、解法
Dueling DQN网络结构与DQN相似，如下图所示，它有2个分支，1个用于预测state value，它是一个标量；另1个用于预测与状态相关的action advantage value，它是1个矢量，矢量的每个值对应着1个动作。这2个分支最后输出了每个动作的Q值。

<img src="https://s1.ax1x.com/2020/08/03/aa1Tij.png" alt="1" style="zoom:67%;" />

Dueling DQN可直接学习哪些状态是有价值的。Dueling DQN从Q function中剥离出state function和advantage function，state function只用于预测state的好坏，而advantage function只用于预测在该state下每个action的重要性，这样一来，各个分支各司其职，预测效果更好。

Dueling DQN把Q function拆分为state function和advantage function，所以有：

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+A(s, a ; \theta, \alpha)$$

其中$V(s ; \theta, \beta)$是state function，输出一个标量， $A(s, a ; \theta, \alpha)$是advantage function，输出一个矢量，矢量长度等于动作空间大小； $\theta$指网络卷积层的参数； $\alpha$和$\beta$分别是2个分支的全连接层的参数。

上述公式存在unidentifiable问题，也就是从Q中无法唯一地分离出V和A。举个例子，把一个常量加到V中，并且从A中减去该常量，那么Dueling DQN仍输出相同的Q值。这个unidentifiable问题会严重地降低网络性能。也就是说，V不能反映state值，A不能反映advantage值。

如何使V反映state值，以解决unidentifiable问题呢？文中的方法是强制把advantage function的输出矢量的和设置为0。那么Q function将改写为：

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+\left(A(s, a ; \theta, \alpha)-\max _{a} A\left(s, a^{\prime} ; \theta, \alpha\right)\right)$$

由于$a^{*}=\arg \max _{a^{\prime} \in \mathcal{A}} Q\left(s, a^{\prime} ; \theta, \alpha, \beta\right)=\arg \max _{a^{\prime} \in \mathcal{A}} A\left(s, a^{\prime} ; \theta, \alpha\right)$，由此可得$V(s ; \theta, \beta)=Q(s, a * ; \theta, \alpha, \beta)$，即上述公式使V反映了state值。

实际中为了训练的稳定性，使用如下公式作为替代：

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+\left(A(s, a ; \theta, \alpha)-\sum_{a^{\prime}} A\left(s, a^{\prime} ; \theta, \alpha\right) /|A|\right)$$

## 三、实验内容
在多个Atari游戏上效果优于DQN。

## 四、缺点
如果状态$s_{1}$比状态$s_{2}$总体要好，那么每个$Q(s_1, a)$相对每个$Q(s_2, a)$要高，而需要$Q(s, a)$的每项都去拟合这种“低频分量”，会在某种程度上费去神经网络的“容量”，不是最优的办法；而将$Q(s, a)$分解为$V(s)$及$A(s, a)$的和就没有这个问题。该假设偏于直觉，并不一定是真实情况。

## 五、优点
Dueling DQN仅仅涉及神经网络的中间结构的改进，现有的DQN算法可以在使用Duel DQN网络结构的基础上继续使用现有的算法。

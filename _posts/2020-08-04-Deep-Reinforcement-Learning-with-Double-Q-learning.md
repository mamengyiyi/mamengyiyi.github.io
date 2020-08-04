---
layout:     post
title:      单智能体强化学习算法
subtitle:   Double DQN: Deep Reinforcement Learning with Double Q-learning
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

论文链接：<a href="https://arxiv.org/pdf/1509.06461.pdf">Deep Reinforcement Learning with Double Q-learning, AAAI 2016</a>

## 一、问题 
在DDQN之前，基本上所有的目标Q值都是通过贪婪法直接得到的，无论是Q-Learning， DQN(NIPS 2013)还是 Nature DQN，都是如此。比如对于Nature DQN,虽然用了两个Q网络并使用目标Q网络计算Q值，其第$j$个样本的目标Q值的计算还是贪婪法得到的，计算如下式：

$$y_{j}=\left\{\begin{array}{ll}
R_{j} & \text {is_end}_{j} \text { is true} \\
R_{j}+\gamma \max _{a^{\prime}} Q^{\prime}\left(\phi\left(S_{j}^{\prime}\right), A_{j}^{\prime}, w^{\prime}\right) & \text {is_end}_{j} \text { is false }
\end{array}\right.$$

使用max虽然可以快速让Q值向可能的优化目标靠拢，但是很容易过犹不及，导致过度估计(Over Estimation)，所谓过度估计就是最终我们得到的算法模型有很大的偏差(bias)。为了解决这个问题， DDQN通过解耦目标Q值动作的选择和目标Q值的计算这两步，来达到消除过度估计的问题。

## 二、解法 

DDQN和Nature DQN一样，也有一样的两个Q网络结构。在Nature DQN的基础上，通过解耦目标Q值动作的选择和目标Q值的计算这两步，来消除过度估计的问题。

Nature DQN对于非终止状态，其目标Q值的计算式子是：

$$y_{j}=R_{j}+\gamma \max _{a^{\prime}} Q^{\prime}\left(\phi\left(S_{j}^{\prime}\right), A_{j}^{\prime}, w^{\prime}\right)$$

在DDQN这里，不再是直接在目标Q网络里面找各个动作中最大Q值，而是先在当前Q网络中先找出最大Q值对应的动作，即

$$a^{\max }\left(S_{j}^{\prime}, w\right)=\arg \max _{a^{\prime}} Q\left(\phi\left(S_{j}^{\prime}\right), a, w\right)$$

然后利用这个选择出来的动作$a^{\max }\left(S_{j}^{\prime}, w\right)$在目标网络里面去计算目标Q值。即：

$$y_{j}=R_{j}+\gamma Q^{\prime}\left(\phi\left(S_{j}^{\prime}\right), a^{\max }\left(S_{j}^{\prime}, w\right), w^{\prime}\right)$$

综合起来写就是：

$$y_{j}=R_{j}+\gamma Q^{\prime}\left(\phi\left(S_{j}^{\prime}\right), \arg \max _{a^{\prime}} Q\left(\phi\left(S_{j}^{\prime}\right), a, w\right), w^{\prime}\right)$$

除了目标Q值的计算方式以外，DDQN算法和Nature DQN的算法流程完全相同。

## 三、实验内容

效果超越了DQN：

![1](https://s1.ax1x.com/2020/08/04/aBn6QP.png)

## 四、缺点
暂无评价。

## 五、优点
DDQN算法出来以后，在处理高偏差方面取得了比较好的效果，因此得到了比较广泛的应用。

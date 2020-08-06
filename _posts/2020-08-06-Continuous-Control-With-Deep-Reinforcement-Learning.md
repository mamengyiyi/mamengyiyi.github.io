layout:     post
title:      单智能体强化学习算法
subtitle:   DDPG：Continuous Control With Deep Reinforcement Learning
date:       2020-08-06
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:

- RL
    - RL advanced algorithms
    - Model-Free RL
    - Deterministic Policy Gradients

---

论文链接：<a href="https://arxiv.org/pdf/1509.02971.pdf">Continuous Control With Deep Reinforcement Learning, ICLR 2016</a>

## 一、问题

  - DQN只能处理离散的、低维的动作空间。DQN不能直接处理连续的原因是它依赖于在每一次最优迭代中寻找 动作值函数的最大值(表现为在Q神经网络中输出每个动作的值函数)，针对连续动作空间DQN没有办法输出每个动作的动作值函数。
  - 解决上述连续动作空间问题的一个简单方法是将动作空间离散化，但是动作空间是随着动作的自由度呈指数增长的（论文中举了一个机械臂的例子，自由度是指机械臂的关节，即使将每个关节的动作离散化{-k,0,k}、自由度为7, 3^7这个数字也是很大的）。所以针对大部分任务来说这个方法是不现实的。
  因此，本文基于David Sliver在2014年提出的DPG（Deterministic Policy Gradient），设计了DDPG算法，将深度Q学习（DQN）的成功引入到连续动作空间中。DDPG使用确定性策略之后，**值函数期望与当前策略无关，只与环境有关**，因此DDPG算法是off-policy的，即$Q$值由$$Q^{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{r_{t}, s_{t+1} \sim E}\left[r\left(s_{t}, a_{t}\right)+\gamma \mathbb{E}_{a_{t+1} \sim \pi}\left[Q^{\pi}\left(s_{t+1}, a_{t+1}\right)\right]\right]$$变为了$$Q^{\mu}\left(s_{t}, a_{t}\right)=\mathbb{E}_{r_{t}, s_{t+1} \sim E}\left[r\left(s_{t}, a_{t}\right)+\gamma Q^{\mu}\left(s_{t+1}, \mu\left(s_{t+1}\right)\right)\right]$$

## 二、解法

### 2.1 actor-critic架构 & DPG for actor & TD error for critic

虽然DDPG借鉴了DQN的思想（memory replay 和 target netwotrk），但是却不能直接使用Q-learning算法框架，因为在连续动作空间无法简单、快速地实现Q-learning的贪婪策略。所以使用的是基于确定动作策略的actor-critic算法框架。并且在actor部分采用DPG的确定性策略方式。

### 2.2 soft target updates

使用类似DQN的架构，但是不同于DQN直接将Q网络的参数定期复制到target network，DDPG通过”soft” target updates的方式来保证参数可以缓慢的更新，从而达到和DQN定期复制参数相类似的提升学习稳定性的效果。
操作方式为：$\theta^{\prime} \leftarrow \tau \theta+(1-\tau) \theta^{\prime}$ with $\tau \ll 1$

### 2.3 batch normalization

使用batch normalization 解决不同的inputs特征有不同量级的单位及数据范围问题。比如不同传感器分别测量的是角度、距离等，显然不能将它们当做同一种数据进行处理。值得注意的是，即使不是传感器的input，比如在mujoco环境各种也可对obs等使用batch normalization

### 2.4 explore effectively

DDPG是off-policy的，所以行为策略和评估策略的不同可以增加探索。另外在DDPG中，通过在行为策略的确定性策略上添加噪声来使算法结构高效“探索”。
$$\mu^{\prime}\left(s_{t}\right)=\mu\left(s_{t} | \theta_{t}^{\mu}\right)+\mathcal{N}$$

### 2.5 算法

在DDPG中，因为采用actor-critic架构，所以有actor和critic两个部分。此外，因为借鉴DQN的思想，所以有四个神经网络。即：critic部分有两个神经网络，target network $Q^{'}$ 和 critic network $Q$；actor部分有两个神经网络：target network $\mu^{'}$ 和 actor network $\mu$ 。

![ageVM9.png](https://s1.ax1x.com/2020/08/06/ageVM9.png)

为了便于理解DDPG算法，结合算法框架和之前的解释，将四个神经网络的公式、输入、输出以及相互之间的关系制作成表格如下：

![ageAxJ.png](https://s1.ax1x.com/2020/08/06/ageAxJ.png)

## 三、实验内容

在连续控制任务上效果很好：
![ageZrR.png](https://s1.ax1x.com/2020/08/06/ageZrR.png)



## 四、缺点

暂无评价。

## 五、优点

  * 使用卷积神经网络来模拟策略函数和Q函数，并用深度学习的方法来训练，证明了在RL方法中，非线性模拟函数的准确性和高性能、可收敛；而DPG中，可以看成使用线性回归的机器学习方法：使用带参数的线性函数来模拟策略函数和Q函数，然后使用线性回归的方法进行训练。
  * experience replay memory的使用：actor同环境交互时，产生的transition数据序列是在时间上高度关联(correlated)的，如果这些数据序列直接用于训练，会导致神经网络的overfit，不易收敛。DDPG的actor将transition数据先存入experience replay buffer, 然后在训练时，从experience replay buffer中随机采样mini-batch数据，这样采样得到的数据可以认为是无关联的。
  * target网络的使用，使学习过程更加稳定，收敛更有保障。
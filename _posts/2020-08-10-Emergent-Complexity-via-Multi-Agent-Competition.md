---
layout:     post
title:      多智能体强化学习算法
subtitle:   Emergent Complexity via Multi-Agent Competition
date:       2020-08-10
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: false
tags:
    - RL
    - MARL
    - RL advanced algorithms
    - Analysis of Emergent Behaviors
---


------

论文链接：<a href="https://arxiv.org/pdf/1710.03748.pdf">Emergent Complexity via Multi-Agent Competition, ICLR 2018</a>

github链接：<a href="https://github.com/openai/multiagent-competition">论文环境与算法代码</a>

视频链接：<a href="https://goo.gl/eR7fbX">论文实验效果</a>


## 一、问题

强化学习智能体的复杂度通常只和环境有关，要想训练一个复杂的智能体，通常需要复杂的环境。本文提出多智能体设置下，在竞争环境中的self-play可以训练出远比环境复杂的智能体。AlphaGo和Dota 2上已经有相关工作，本文则想在Mujoco环境上验证了这种self-play的方法在连续控制任务上的有效性。

## 二、解法

本文使用分布式PPO训练，采用Clip方法。

为了解决稀疏reward的问题，本文采用exploration curriculum方法，在训练的初始阶段使用一些exploration reward使智能体学会站立或行走等基础动作，以便增加探索到后续competition reward的可能性。exploration reward在整体reward中的比重会随着训练的进行而降低，从而使得后续训练的重点放在赢下智能体的竞争。本文通过以下公式来实现reward的调整：

$$r_{t}=\alpha_{t} s_{t}+\left(1-\alpha_{t}\right) \mathbb{I}[t==T] R$$

其中$s_{t}$是exploration reward，$R$是competition reward。

为了解决对手策略变化引起的环境变化问题，本文采用了oppenent sampling方法。文章尝试了两种采样方法：1. 使用最新训练的对手模型作为训练环境的一部分，这会导致一方智能体变得越来越强而另一方一蹶不振。 2. 使用随机采样的对手旧版本模型，训练会更加稳定，策略更具有鲁棒性。

## 三、实验内容 

本文使用了四个环境。分别是Run to Goal，You Shall Not Pass，Sumo和Kick and Defend（足球射门），对每个环境都设计了不同的exploration reward。对于前两个环境，使用MLP作为策略网络；对于后两个环境，使用LSTM作为策略网络（实验结果效果更好，无理论分析）。智能体学会了奔跑、跳跃、躲避、阻拦等动作，甚至还会做假动作欺骗对手。同时，在Sumo上训练的智能体在面对Wind Attack时也具有很好的泛化性。对于exploration curriculum，实验对比了不降低exploration reward和降低exploration reward的智能体的对战胜率，证明了上文公式的有效性。对于opponent samping，实验对比了采样不同时期对手模型进行训练的智能体的对战胜率，证明了上文公式的有效性。

## 四、缺点

对于模型的选择和对手采样缺乏理论上的分析。

## 五、优点

本文的实验环境可以作为多智能体的经典benchmark使用；分段式reward的设计可以借鉴用于多种稀疏reward的RL任务；未来工作中可以尝试使用LSTM作为DRL的网络并给出一些直觉或理论上的分析。

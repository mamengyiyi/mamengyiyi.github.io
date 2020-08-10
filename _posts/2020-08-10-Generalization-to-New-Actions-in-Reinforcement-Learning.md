---
layout:     post
title:      单智能体强化学习算法
subtitle:   Generalization to New Actions in Reinforcement Learning
date:       2020-08-10
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - RL advanced algorithms
    - Generalized RL
---
---

论文链接：<a href="https://proceedings.icml.cc/static/paper_files/icml/2020/29-Paper.pdf">Generalization to New Actions in Reinforcement Learning, ICML 2020</a>

代码链接：<a href="https://github.com/clvrai/new-actions-rl">github链接</a>

环境链接：<a href="https://clvrai.com/create">CREATE: Chain REAction Tool Environment</a>

## 一、问题 

许多问题的action set可能会变化，比如机器人用没见过的工具完成任务，推荐系统中对一个新的物品如何进行推荐的冷启动等。本文研究的问题是如何让RL在不同的action set上有一个好的泛化表现。这个问题具有如下两个挑战：
  - agent需要观察action的影响或者与action实际交互来理解新的action
  - 如何从对action影响的观察中提取出有效的信息

## 二、解法 
针对上述问题，本文的提出一个两阶段的优化框架，如下图所示：
  - 对action observation学习表征，得到一个action representation space；
  - 将action representation space输入到策略网络中并使用RL进行训练

    

<img src="https://s1.ax1x.com/2020/08/10/aH8FDe.png" alt="aH8FDe.png" style="zoom:80%;" />

### 2.1 问题定义

本文的目标是在给定的动作集$\mathbb{A}=\left\{a_{1}, \dots, a_{N}\right\}$上训练，在一个训练中未见过的动作集$\mathbb{A}^{\prime}$中采样的动作子集$\mathcal{A} \subset \mathbb{A}^{\prime}$上进行测试，以最大化该动作子集$\mathcal{A}$上的累积收益：



$$R=\mathbb{E}_{\mathcal{A} \subset \mathbb{A}^{\prime}, a \sim \pi(a \mid s, \mathcal{A})}\left[\sum_{t=1}^{T} \gamma^{t-1} \mathcal{R}\left(s_{t}\right)\right]$$



### 2.2 无监督动作表征 

本文使用一个hierarchical variational autoencoder (HVAE)来表征action observations。给定一个训练动作$a_{i} \in \mathbb{A}$，HVAE将该action关联的observation进行编码得到$c_{i}$，$c_{i}$则用于决定encoder $q_{\psi}\left(z_{i, j} \mid o_{i, j}, c_{i}\right)$和decoder $p\left(o_{i, j} \mid z_{i, j}, c_{i}\right)$。HVAE使用重建的loss加KL散度正则化来训练：

$$\begin{aligned}
\mathcal{L}=& \sum_{\mathcal{O} \in \mathbb{O}}\left[\mathbb{E}_{q_{\phi}(c \mid \mathcal{O})}\left[\sum_{o \in \mathcal{O}} \mathbb{E}_{q_{\psi}(z \mid o, c)} \log p(o \mid z, c)\right.\right.\\
&\left.\left.-D_{K L}\left(q_{\psi} \| p(z \mid c)\right)\right]-D_{K L}\left(q_{\phi} \| p(c)\right)\right]
\end{aligned}
$$


### 2.3 自适应策略网络

给定动作集$\mathcal{A}=\left\{a_{1}, \ldots, a_{k} \right\}$与对应的动作表征$\left\{c_{1}, \ldots, c_{k} \right\}$作为策略网络的输入，计算每个动作在当前状态下的得分，再经过softmax得到动作的概率分布：

$$\pi\left(a_{i} \mid s, \mathcal{A}\right)=\frac{e^{f_{\nu}\left[c_{i}, f_{\omega}(s)\right]}}{\sum_{j=1}^{k} e^{f_{\nu}\left[c_{j}, f_{\omega}(s)\right]}}$$

其中$f_{\omega}(s)$是state encoder，$f_{\nu}$是用以计算得分的效用函数。整个网络使用policy gradient类的方法进行端到端的训练。

### 2.4 泛化目标与训练流程

对于RL来说，如果当某个策略过度利用了某些训练时的动作子集时，会使得策略非常偏向这些过度使用的动作，不利于泛化。针对这个问题，本文的处理方式为：

  - 对动作空间进行降采样
  - 增加随机策略的熵$\mathcal{H}\left[\pi_{\theta}(a \mid s)\right]$进行正则化辅助
  - 在验证集上验证选表现最好的模型

经过上述调整，最终的训练目标为：

$$\max _{\theta} \mathbb{E}_{\mathcal{A} \subset \mathbb{A}, a \sim \pi_{\theta}(. \mid s, \mathcal{A})}\left[R(s)+\beta \mathcal{H}\left[\pi_{\theta}(a \mid s, \mathcal{A})\right]\right]$$



训练流程如下图所示；

<img src="https://s1.ax1x.com/2020/08/10/aH8iuD.png" alt="aH8iuD.png" style="zoom:50%;" />



测试时方法如下：

<img src="https://s1.ax1x.com/2020/08/10/aH89gK.png" alt="aH89gK.png" style="zoom:50%;" />

## 三、实验内容
本文在如下环境上进行测试：

<img src="https://s1.ax1x.com/2020/08/10/aH8226.png" alt="aH8226.png" style="zoom: 80%;" />



HVAE可以提取出action observation中的有效信息：

<img src="https://s1.ax1x.com/2020/08/10/aH8y5R.png" alt="aH8y5R.png" style="zoom:67%;" />



对比baselines有很好的效果：

<img src="https://s1.ax1x.com/2020/08/10/aH8g8x.png" alt="aH8g8x.png" style="zoom: 80%;" />



动作空间降采样和熵正则化也有明显帮助：

<img src="https://s1.ax1x.com/2020/08/10/aH8cP1.png" alt="aH8cP1.png" style="zoom:80%;" />

## 四、缺点
暂无评价。 

## 五、优点
* 第一个在动作集上进行泛化研究的工作
* 提出了一个很酷的benchmark
* 使用HVAE对动作进行表征的方式值得学习
* 在动作集上进行泛化是一个具有现实意义的工作

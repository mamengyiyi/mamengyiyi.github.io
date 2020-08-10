---
layout:     post
title:      单智能体强化学习算法
subtitle:   Dynamic Weights in Multi-Objective Deep Reinforcement Learning
date:       2020-08-10
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - RL advanced algorithms
    - Transfer and Multitask RL
---
---

论文链接：<a href="http://proceedings.mlr.press/v97/abels19a/abels19a.pdf">Dynamic Weights in Multi-Objective Deep Reinforcement Learning, ICML 2019</a>

## 一、问题 

许多现实问题中同时存在多个优化目标，如果可以预先知道每个目标的权重，那么多目标优化的问题可以通过线性叠加的方式转换为单目标优化问题。但是实际中很多问题的不同目标的权重会动态变化，此时线性叠加的方式便不适用了。为了解决这个问题，本文提出一个多目标Q网络。

## 二、解法 
本文采取类似于Dueling Network的结构，把动态变化的权重也作为Q网络的一部分输入：

<img src="https://s1.ax1x.com/2020/08/10/abEoxs.png" alt="abEoxs.png" style="zoom:67%;" />

算法如下：

<img src="https://s1.ax1x.com/2020/08/10/abEHrq.png" alt="abEHrq.png" style="zoom:80%;" />



每个step，agent会收到环境的多目标权重变量$w_{t}$，如果该权重与上一个权重$w_{t-1}$不同，则把它加到$\mathcal{W}$中。给定一个minibatch的transition，每个transition的loss为当前step收到的权重变量$w_{t}$与从$\mathcal{W}$采样出的权重的Loss之和：

$$\begin{array}{l}
\frac{1}{2}\left[\left|\mathbf{y}_{\mathbf{w}_{t}}^{(j)}-\mathbf{Q}_{C N}\left(a_{j}, s_{j} ; \mathbf{w}_{t}\right)\right|+\left|\mathbf{y}_{\mathbf{w}_{j}}^{(j)}-\mathbf{Q}_{C N}\left(a_{j}, s_{j} ; \mathbf{w}_{j}\right)\right|\right] \\
\mathbf{y}_{\mathbf{w}}^{(j)}=\mathbf{r}_{j}+\gamma \mathbf{Q}_{C N}^{-}\left(\underset{a \in A}{\operatorname{argmax}} \mathbf{Q}_{C N}\left(a, s_{j+1} ; \mathbf{w}\right) \cdot \mathbf{w}, s_{j+1} ; \mathbf{w}\right)
\end{array}$$

这样在两个不同的权重上训练，有利于网络区分不同权重的Q值，有利于泛化性。

为了匹配动态环境设置下的不稳定性，本文还设计了一种Diverse Experience Replay：

<img src="https://s1.ax1x.com/2020/08/10/abEI2j.png" alt="abEI2j.png" style="zoom:80%;" />



## 三、实验内容
本文在一个环境权重动态变化的环境上进行了实验：

<img src="https://s1.ax1x.com/2020/08/10/abE7Mn.png" alt="abE7Mn.png" style="zoom:80%;" />



## 四、缺点
该方法默认已知环境权重会动态变化，而在很多情况中，我们并不能知道多个目标之间的权重关系，更不知道权重会不会随着时间关系而如何变化。若能使用learning的方法获取多目标之间的权重关系会更有意义。

## 五、优点
将权重作为输入的一部分进行感知是一种比较符合认知的做法。
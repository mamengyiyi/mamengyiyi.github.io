---
layout:     post
title:      多智能体强化学习算法
subtitle:   From Few to More：Large-scale Dynamic Multiagent Curriculum Learning
date:       2020-08-14
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

论文链接：<a href="https://arxiv.org/pdf/1909.02790.pdf">From Few to More: Large-scale Dynamic Multiagent Curriculum Learning, AAAI 2020</a>

代码链接：<a href="https://github.com/starry-sky6688/StarCraft">github链接</a>


## 一、问题

课程学习（Curriculum Learning）由Montreal大学的Bengio教授团队在2009年的ICML上提出，主要思想是模仿人类学习的特点，由简单到困难来学习课程（在机器学习里就是容易学习的样本和不容易学习的样本），这样容易使模型找到更好的局部最优，同时加快训练的速度。将课程学习思想应用到强化学习中的研究主要集中在单智能体领域，同时多数研究使用的是利用先验知识手工设计课程。本文提出DyMA-CL，设计了三种课程之间的迁移机制来加速大规模多智能体的训练。将课程学习应用到多智能体强化学习的示例如下图：

![dC5Nm8.png](https://s1.ax1x.com/2020/08/14/dC5Nm8.png)

## 二、解法

### 2.1 Partially Observable Stochastic Games

Partially Observable Stochastic Games（POSG）是MDP在多智能体设置下的自然扩展，定义为$\left\langle{\mathcal{N}}, \mathcal{S}, \mathcal{A}^{1}, \cdots, \mathcal{A}^{n}, T, \mathcal{R}^{1}, \cdots, \mathcal{R}^{n}, \mathcal{O}^{1}, \cdots, \mathcal{O}^{n}\right\rangle$：

  * $\mathcal{N}$：n个智能体
  * $\mathcal{S}$：状态集合
  * $\mathcal{A}^{i}$：智能体$i$的动作集合，$\mathcal{A}=\mathcal{A}^{1} \times \mathcal{A}^{2} \times \cdots \times \mathcal{A}^{n}$是联合动作空间。
  * $T$：转移函数
  * $\mathcal{R}^{i}$：智能体$i$的奖励
  * $\mathcal{O}^{1}$：智能体$i$的观察，$\\{o_{t}^{i, e n v}, m_{t}^{i}, o_{t}^{i, 1}, \cdots, o_{t}^{i, i-1}, o_{t}^{i, i+1}, \cdots, o_{t}^{i, n}\\}$是智能体$i$在时间$t$的观察，其中$o_{t}^{i, e n v}$是对周围环境的描述；$ m_{t}^{i}$是智能体的私有性质，如位置、健康状态等；$o_{t}^{i, i-1}$是智能体$i$对智能体$i-1$的观察，如两个智能体之间的相对位置等。

### 2.2 大规模多智能体系统的性质

大规模多智能体系统有如下三个性质：

  * 部分可观察性：在MAS中，智能体根据他们自己的观察来做出决策，从而可以将大规模问题减少为相对独立但又相关的子问题。
  * 稀疏交互：从全局角度来看，在同一时刻，每个智能体仅与MAS中的某些智能体交互，并且这种交互并非始终发生。
  * 状态语义：每个状态都包含语义信息，可用于衡量状态之间的相似性。例如，在《星际争霸2》中，随着游戏的进行，如果士兵的任何一方在战斗中死亡，智能体的数量将会减少，这种情况下的学习与小规模战斗中的学习类似。

### 2.3 知识迁移



![dC5aTg.png](https://s1.ax1x.com/2020/08/14/dC5aTg.png)



本文设计了三种知识迁移的方法：

  * Buffer Reuse
  * Curriculum Distillation
  * Model Reload (DyAN)

#### 2.3.1 Buffer Reuse

借鉴DQN from demonstration的思想，设计了针对于off-policy的RL算法的迁移机制Buffer Reuse。假设智能体按照$\tau_{1}, \tau_{2}, \cdots, \tau_{k}$的顺序学任务，每个任务$\tau_{i}$有一个replay buffer $\mathcal{D}\_{i}$。保留replay buffer序列$\mathcal{D}\_{1}, \mathcal{D}\_{2}, \cdots, \mathcal{D}\_{k-1}$并从每个buffer中采样$b$条经验作为expert demonstration，而当前任务的buffer为 $\mathcal{D}\_{k}$。在训练中最小化如下Loss：

$$\text { Loss }=\sum_{i=1}^{k} \sum_{j=1}^{b}\left[\left(r_{i}^{j}+\gamma \max _{a_{i}^{\prime} j} q_{\tau_{i}}\left(s_{i}^{\prime}, a_{i}^{\prime}\right)-q_{\tau_{i}}\left(s_{i}^{j}, a_{i}^{j}\right)\right)^{2}\right]$$

其中$\left(s_{i}^{j}, a_{i}^{j}, s_{i}^{j}, r_{i}^{j}\right)$是从buffer $\mathcal{D}_{i}$中采样的任务$i$的第$j$条经验。对于不同任务state space大小不一致的问题，通过补零等方式将它们reshape到同一维度。

#### 2.3.2 Curriculum Distillation

Curriculum Distillation对于on-policy和off-policy的RL算法都适用，具有做法就是在RL的loss之外加入一个KL散度loss，即$Loss =L_{\mathrm{RL}}+L_{\text {Dixtil. }}$，以最小化不同任务上策略的相似程度。具体如下：

$$\begin{array}{c}
L_{\text {Distil }}=\sum_{i=1}^{k-1} \mathrm{KL}\left(\pi_{\tau_{i}} \| \pi_{\tau_{k}}\right) \quad \text { or } \\
L_{\text {Distil }}=\sum_{i=1}^{k-1} \sum_{j=1}^{\left|\mathcal{D}_{k}\right|} \operatorname{softmax}\left(\frac{\mathbf{q}_{\tau_{i}}\left(s_{j}\right)}{\omega}\right) \ln \frac{\operatorname{softmax}\left(\frac{\mathbf{q}_{\tau_{1}}\left(s_{j}\right)}{\omega}\right)}{\operatorname{softmax}\left(\mathbf{q}_{\tau_{k}}\left(s_{j}\right)\right)}
\end{array}$$

对于不同任务state space大小不一致的问题，通过补零等方式将它们reshape到同一维度。

#### 2.3.3 Model Reload (DyAN) 

每个智能体的观察的定义里面包含着对于其他智能体的观察，因此维度会随着其他智能体数量的变化而变化。根据性质3，大型环境中的某些状态通常包含与小型环境中相似的语义信息，因此可以给出从每个智能体观察中提取语义信息并映射到同一隐变量空间的函数$\Phi(\cdot)$：

> 给定三个具体不同状态维度的任务$\tau_{e}$，$\tau_{f}$与$\tau_{g}$，如果$S_{e}^{\tau_{e}}$与$S_{f}^{\tau_{f}}$包含相似的语义信息，而$S_{g}^{\tau_{g}}$中没有，那么应该满足$\operatorname{dis}\left(\Phi\left(s_{e}^{\tau_{e}}\right), \Phi\left(s_{f}^{\tau_{f}}\right)\right)<\operatorname{dis}\left(\Phi\left(s_{e}^{\tau_{e}}\right), \Phi\left(s_{g}^{\tau_{g}}\right)\right)$与$\operatorname{dis}\left(\Phi\left(s_{e}^{\tau_{e}}\right), \Phi\left(s_{f}^{\tau_{f}}\right)\right)<\operatorname{dis}\left(\Phi\left(s_{f}^{\tau_{f}}\right), \Phi\left(s_{g}^{\tau_{g}}\right)\right)$



DyAN使用GNN来解决状态维度不一致的问题。如下图所示，左半部分使用普通的DNN来处理智能体本身的观察，右半部分使用GNN来处理智能体对于其他智能体的观察，将左右两边输出进行拼接再次输入DNN用以求出Q值或概率分布：

<img src="https://s1.ax1x.com/2020/08/14/dC5U0S.png" alt="dC5U0S.png" style="zoom:80%;" />



## 三、实验内容 

在星际2上与IQL、VDN进行了对比，对原始环境的攻击敌人进行了修改，改为攻击grid，然后看这个grid里面有没有敌人，以解决action space会变化的问题，有利于knowledge transfer或者distillation：

<img src="https://s1.ax1x.com/2020/08/14/dC50Yj.png" alt="dC50Yj.png" style="zoom:80%;" />

在MAgent环境上与PPO、A2C、ACER进行了对比：

<img src="https://s1.ax1x.com/2020/08/14/dC5YOf.png" alt="dC5YOf.png" style="zoom:80%;" />

只有SUM、MEAN和MAX的线表示从头开始学习。所有方法中使用SUM聚合的效果最好：

<img src="https://s1.ax1x.com/2020/08/14/dC5wkQ.png" alt="dC5wkQ.png" style="zoom:80%;" />

## 四、缺点

*  论文本身还是使用了手动设计的课程

## 五、优点

* 论文对于多智能体系统性质的总结非常到位，对于多智能体系统的认知是很好的补充
* 方法具有很直接很可信的insight，有一种大道至简的感觉

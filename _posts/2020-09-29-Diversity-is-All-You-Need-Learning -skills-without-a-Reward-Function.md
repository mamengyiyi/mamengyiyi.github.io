---
layout:     post
title:      单智能体强化学习算法
subtitle:   Diversity is All You Need：Learning Skills without a Reward Function
date:       2020-09-29
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: false
tags:
    - RL
    - RL advanced algorithms
    - Representation Learning in RL
---


------

论文链接：<a href="https://openreview.net/pdf?id=SJx63jRqFm">Diversity is All You Need：Learning Skills without a Reward Function, ICRL 2019</a>

代码链接：<a href="https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md">github链接</a>


## 一、问题

强化学习一般需要通过奖励函数的引导来学习只能用于特定的任务的policy。受人类学习过程的启发，这篇文章研究，agent如何在没有task-dependent reward的情况下，也能学习到多样化的skill，在后续遇到特定任务时，也可以利用现有的skill进行解决。skill定义为一种latent-conditioned的policy，可以以一致的方式改变环境状态。为了使学习到的skills能够尽可能涵盖状态空间，作者希望skill都互相不同并且尽可能多样化。基于这个想法，本文提出了“Diversity is All You Need”(DIAYN)方法，在没有奖励函数的情况下为agent学习有用的skills。

## 二、解法

#### 2.1 优化目标设计

在该算法中，策略被定义成$\pi(a \mid s, z)$。与普通的策略相比，多了一个隐变量$z$做为条件。 $z$一般从均匀分布$z \sim p(z)$采样。DIYAN 希望采样不同的$z$时，策略能够对应于不同的skill。例如，当$z=z_{1}$时，skill是奔跑；当$z=z_{2}$时，skill是跳跃...

如何在没有reward的setting学习不仅diverse，而且有含义、有用的skill，文章提出的DIAYN构建于三个idea上：
  * 不同的skill对应于不同的状态空间，相当于探索不同的区域，从而使得skill是distinguishable的
  * 通过状态而不是动作来区分skill，因为许多动作可能导致环境发生相同的转移；同时通过当前的状态，模型应该能够轻松的推断出当前的skill是什么
  * 让skill尽可能diverse，尽可能随机的act，但始终要保持distinguishable

DIYAN中，这种学习目标表述为“最大化状态$s$和对应skill $z$的互信息”。具体的，DIYAN最大化的目标如下：

$$\mathcal{F}(\theta)=I(S ; Z)+H(A \mid S)-I(A ; Z \mid S)$$

其中，
  * 第一项是$S$和$Z$的互信息，是目标函数的核心
  * 第二项是将所有skill混合起来的mixed policy的熵，本文希望动作具有多样性，这也是RL中常用的目标，如A3C, SAC。
  * 第三项是在已知状态$S$的情况下， $Z$和$A$的互信息。DIYAN希望$Z$和状态$S$的关系密切，同时尽量减少在已知状态下$Z$和动作$A$的关系，强调不能以动作来区分skill

展开上式可得：

$$\begin{aligned}
\mathcal{F}(\theta) & \triangleq I(S ; Z)+\mathcal{H}[A \mid S]-I(A ; Z \mid S) \\
&=(\mathcal{H}[Z]-\mathcal{H}[Z \mid S])+\mathcal{H}[A \mid S]-(\mathcal{H}[A \mid S]-\mathcal{H}[A \mid S, Z]) \\
&=\mathcal{H}[Z]-\mathcal{H}[Z \mid S]+\mathcal{H}[A \mid S, Z] \\
&=H(A \mid S, Z)+\mathbb{E}_{s \sim \pi(z), z \sim p(z \mid s)}[\log p(z \mid s)]-\mathbb{E}_{z \sim p(z)}[\log p(z)]
\end{aligned}$$

根据如下定理：

<img src="https://s1.ax1x.com/2020/09/29/0eZhxs.png" alt="0eZhxs.png" style="zoom: 67%;" />

可以将$z \sim p(z \mid s)$写成$z \sim p(z)$，从而进一步得到：

$$\mathcal{F}(\theta)=H(A \mid S, Z)+\mathbb{E}_{s \sim \pi(z), z \sim p(z)}[\log p(z \mid s)]-\mathbb{E}_{z \sim p(z)}[\log p(z)]$$

由于无法对所有state和skill上对$p(z \mid s)$进行计算，本文基于Jensen不等式提出用一个discriminator来拟合这个后验分布$q(z \mid s)$：

$$\begin{aligned}
\mathcal{F}(\theta) &=H(A \mid S, Z)+\mathbb{E}_{s \sim \pi(z), z \sim p(z)}\left[\log \frac{p(z \mid s)}{q(z \mid s)}+\log q(z \mid s)\right]-\mathbb{E}_{z \sim p(z)}[\log p(z)] \\
&=H(A \mid S, Z)+D_{K L}[p(z \mid s) \| q(z \mid s)]+\mathbb{E}_{s \sim \pi(z), z \sim p(z)}[\log q(z \mid s)]-\mathbb{E}_{z \sim p(z)}[\log p(z)] \\
& \geq H(A \mid S, Z)+\mathbb{E}_{s \sim \pi(z), z \sim p(z)}[\log q(z \mid s)-\log p(z)]
\end{aligned}$$

这里的$\geq$是因为KL散度大于零。由此 DIYAN 获得了一个下界，最大化该下界相当于最大化原始的目标函数。

#### 2.2 算法实现

DIYAN使用SAC训练，SAC本身在策略优化过程中，包括了对policy entropy的maximization，也就是对应优化目标中的第一项。我们仅考虑后面两项$\mathbb{E}_{s \sim \pi(z), z \sim p(z)}[\log q(z \mid s)-\log p(z)]$。由于$p(z)$一般是一个固定的分布，因此关键在于最大化$\mathbb{E}[\log q(z \mid s)]$。

在训练时分为两个部分：
  * 使用神经网络$q_{\phi}(z \mid s)$代表该概率分布，输入为$s$，输出为隐变量$z$。神经网络的参数为$\phi$。对于交互过程中采样的序列$\left(s_{1}, z_{1}, a_{1}\right),\left(s_{2}, a_{2}, z_{2}\right), \ldots$, 直接使用监督学习的方式，来最大化输出$z$的似然。例如当 latent 是离散变量时，该监督学习问题是一个标准的分类问题。
  * 强化学习使用该目标函数构造的奖励来训练策略，需要最大化的项可以设置为奖励的形式。期望可以使用蒙特卡罗方法来采样进行估计。奖励为$r_{z}(s, a) \triangleq \log q_{\phi}(z \mid s)-\log p(z)$。该奖励的直观含义是，agent应该按照这种最大化累积该奖励的方式更新policy/skill，使得agent在特定skill下到达的state能够容易地推断出对应的skill。该奖励使用SAC来最大化。

#### 2.3 算法流程

<img src="https://s1.ax1x.com/2020/09/29/0eZoq0.png" alt="0eZoq0.png" style="zoom:80%;" />

## 三、实验内容 

Figure 2 中的几个task中，表现出train得skills的确可以涵盖不同的状态空间，高度可区分且具有随机性；另外，在inverted pendulum和mountain car这两个classic control tasks中，unsupervised的DIAYN以及有一部分skill能够解决特定任务了：

<img src="https://s1.ax1x.com/2020/09/29/0eZIrq.png" alt="0eZIrq.png" style="zoom:80%;" />

DIAYN在mujoco tasks中学习到了多种运动模式，例如向前跑，向后跑，扑倒等：

<img src="https://s1.ax1x.com/2020/09/29/0eZ7ZV.png" alt="0eZ7ZV.png" style="zoom:80%;" />

学习到的skill可以用于作为policy的初始化：对每个mujoco task选择奖励最高的skill作为policy和value network的初始化，对比随机初始化得到了更好的结果：

<img src="https://s1.ax1x.com/2020/09/29/0eZ5Mn.png" alt="0eZ5Mn.png" style="zoom:80%;" />

用DIAYN学到的skills作为分层强化学习的low-level，用一个meta-controller训练high-level policy（HRL常规设定），在多个任务中，获得了很好的表现：

<img src="https://s1.ax1x.com/2020/09/29/0eZHaT.png" alt="0eZHaT.png" style="zoom:80%;" />

基于learned skills对expert policy进行模仿学习，假如给定一个专家的state trajectory $\tau^{\*}=\left\{\left(s_{i}\right)\right\}\_{1 \leq i \leq N}$，使用训练好的discriminator对最优的可能生成这样的trajectory的skill进行esimate，然后返回control policy进行imitation。这样的imitation成功模仿了4个expert中的3个：

<img src="https://s1.ax1x.com/2020/09/29/0eZbIU.png" alt="0eZbIU.png" style="zoom:80%;" />

## 四、缺点

一般问题中可能没有diverse skill这种天然形成的概念，对应的训练难度和真正的效果不能确定。

## 五、优点

对互信息的使用非常具有insight，介绍也很清晰；实验验证了skills作为一种表征的方式可以用于多种强化学习的任务。



### 参考

1. [汤宏垚的知乎](https://zhuanlan.zhihu.com/p/72403124)
2. [白辰甲的知乎](https://zhuanlan.zhihu.com/p/150434344)

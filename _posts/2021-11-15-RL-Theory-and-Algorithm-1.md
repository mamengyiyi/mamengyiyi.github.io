---
layout:     post
title:      Reinforcement Learning Theory and Algorithm
subtitle:   Fundamentals
date:       2021-11-15
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - DL
    - Combinatorial Optimization
typora-root-url: ..\post_pic
typora-copy-images-to: ..\post_pic
---
---

# RL Theory and Algorithm (1): Fundamentals

输出倒逼输入！本系列准备硬啃Alekh Agarwal，Nan Jiang，Sham M. Kakade和Wen Sun所写的《Reinforcement Learning: Theory and Algorithm》，强化一下自己的RL理论储备。本篇主要介绍一下基础知识。

## 一、Markov Decision Process

### 1.1 Discounted (Infinite-Horizon) Markov Decision Processes

在RL中，agent与环境的交互通常被描述为无穷时域（Infinite-horizon）且带折扣（discounted）的MDP，一般用$M=(\mathcal{S}, \mathcal{A}, P, r, \gamma, \mu)$来定义，其中：

* 状态空间$\mathcal{S}$，可能是无限的或者有限的。本书一般假设它是有限的或者可数无限的（比如正整数集就是可数无限的）。
* 动作空间$\mathcal{A}$，可能是无限的或者离散的。本书一般假设它是有限的。
* 转移函数$P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$，其中$\Delta(\mathcal{S})$是关于$\mathcal{S}$的概率分布空间，$P\left(s^{\prime} \mid s, a\right)$是在状态$s$采取动作$a$转移到状态$s'$的概率。本书用$P_{s,a}$表示向量$P(\cdot \mid s, a)$。
* 奖励函数$r: \mathcal{S} \times \mathcal{A} \rightarrow[0,1]$，$r(s,a)$是在状态$s$采取动作$a$得到的即时奖励。
* 折扣因子$\gamma \in[0,1)$，定义了问题的时域。
* 初始状态分布$\mu \in \Delta(\mathcal{S})$，定义了初始状态$s_{0}$是如何生成的。

#### 1.1.1 The objective, policies, and values

**策略 Policies**

agent与环境交互会产生交互记录$\tau_{t}=\left(s_{0}, a_{0}, r_{0}, s_{1}, \ldots, s_{t}, a_{t}, r_{t}\right)$，被称为trajectory。

一般来说，策略指定了一种决策的策略，其中agent根据观察历史自适应地选择动作。 准确地说，策略是从轨迹到动作的映射（可能是随机的映射），即$\pi: \mathcal{H} \rightarrow \Delta(\mathcal{A})$，其中$\mathcal{H}$是所有可能的轨迹的集合，而$\Delta(\mathcal{A})$是 $\mathcal{A}$上的概率分布空间。而平稳（stationary）策略$\pi: \mathcal{S} \rightarrow \Delta(\mathcal{A})$指定的决策策略中，agent仅根据当前的状态选择动作，即$a_{t} \sim \pi\left(\cdot \mid s_{t}\right)$。 而确定性的、平稳的策略的形式为$\pi: \mathcal{S} \rightarrow \mathcal{A}$。



**值函数 Values**

给定一个固定的策略和初始状态$s_{0}=s$，值函数定义为未来折扣收益之和：
$$
V_{M}^{\pi}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right) \mid \pi, s_{0}=s\right]
$$
其中期望是关于轨迹的随机性，包括状态转移的随机性和策略的随机性。假设$r(s,a)$处于0和1之间，那有如不等式成立：
$$
0 \leq V_{M}^{\pi}(s) \leq 1 /(1-\gamma)
$$

> **证明：**
>
> 首先，由于$r(s,a) \ge 0$$，V_{M}^{\pi}(s)$显然大于等于0。
>
> 
>
> 其次，由于$\gamma \in[0,1)$，所以有：
>
> 
> $$
> \begin{equation}
> \begin{aligned}
> V_{M}^{\pi}(s)&=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right) \mid \pi, s_{0}=s\right] \\& \leq \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} \mid \pi, s_{0}=s\right]\\&=\mathbb{E}\left[\lim_{t \to\infty}^{} (\gamma^{0} + \gamma^{1} + \dots + \gamma^{t}) \mid \pi, s_{0}=s\right] \\ &=\mathbb{E}\left[\lim_{t \to\infty}^{} (\frac{1-\gamma^{t}}{1-\gamma}) \mid \pi, s_{0}=s\right] \\&=\frac{1}{1-\gamma}
> \end{aligned}
> \end{equation}
> $$
> 
>
> 其中第一行到第二行是因为$\gamma^{t}<1$，第三行到第四行是等比数列求和公式，第四行到第五行是求极限。

类似的，动作值函数$Q_{M}^{\pi}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$定义为：


$$
Q_{M}^{\pi}(s, a)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right) \mid \pi, s_{0}=s, a_{0}=a\right]
$$


$Q_{M}^{\pi}(s, a)$上界同样为$1 /(1-\gamma)$。



**目标 Goal**

给定一个状态$s$，agent的目标是找到一个策略$\pi$来最大化值函数：


$$
\max _{\pi} V_{M}^{\pi}(s)
$$


其中最大值是值在所有（可能是非平稳和随机的）策略中寻找其值函数最大的策略。 接下来的章节会证明，**存在一个确定性的平稳策略，对所有的初始状态$s$都是最优的。**

#### 1.1.2 Bellman Consistency Equations for Stationary Policies

平稳策略满足以下一致性条件：

**Lemma 1.4**：假设$\pi$是一个平稳策略，那么$V^{\pi}$和$Q^{\pi}$满足如下贝尔曼一致性方程：对于任意的$s \in \mathcal{S}, a \in \mathcal{A}$，


$$
\begin{aligned}
V^{\pi}(s) &=Q^{\pi}(s, \pi(s)) \\
Q^{\pi}(s, a) &=r(s, a)+\gamma \mathbb{E}_{a \sim \pi(\cdot \mid s), s^{\prime} \sim P(\cdot \mid s, a)}\left[V^{\pi}\left(s^{\prime}\right)\right]
\end{aligned}
$$



用矩阵形式来描述值函数，如将$V^{\pi}$视为长度为$|\mathcal{S}|$的向量，$Q^{\pi}$和$r$视为长度为$\|\mathcal{S}\|\cdot \|\mathcal{A}\|$的向量，则概率矩阵$P$的大小为$(\|\mathcal{S}\| \cdot\|\mathcal{A}\|) \times\|\mathcal{S}\|$，其中$P_{(s, a), s^{\prime}}$等于$P\left(s^{\prime} \mid s, a\right)$。则policy evaluation求解值函数可以使用如下矩阵形式表示：


$$
\begin{aligned}
&Q^{\pi}=r+\gamma P V^{\pi} \\
&Q^{\pi}=r+\gamma P^{\pi} Q^{\pi}
\end{aligned}
$$


**Corollary 1.5**：假设$\pi$是一个平稳策略，则policy evaluation 值函数的解为：


$$
Q^{\pi}=\left(I-\gamma P^{\pi}\right)^{-1} r
$$


其中$I$是一个单位矩阵。



要保证$Q^{\pi}$有解，还需要保证矩阵$I-\gamma P^{\pi}$是可逆的。接下来我们证明其是可逆的。要证明矩阵$I-\gamma P^{\pi}$是可逆的，只需证明其对于任意非0向量$x$，$\left(I-\gamma P^{\pi}\right) x$向量不为0，即只需证明$\left\|\left(I-\gamma P^{\pi}\right) x\right\|_{\infty} $不为0即可：

> **证明：**
>
> 
> $$
> \begin{equation}
> \begin{aligned}
> \left\|\left(I-\gamma P^{\pi}\right) x\right\|_{\infty} 
> &=\left\|x-\gamma P^{\pi} x\right\|_{\infty} \\& \geq\|x\|_{\infty}-\gamma\left\|P^{\pi} x\right\|_{\infty} \\
> & \geq \|x\|_{\infty}-\gamma \|P^{\pi}\|_{\infty}\|x\|_{\infty} \\
> &=(1-\gamma \|P^{\pi}\|_{\infty})\|x\|_{\infty}>0
> \end{aligned}
> \end{equation}
> $$
>
> 
>
> 其中第一行到第二行是范数的三角不等式性质，第二行到第三行是算子范数的相容性。
>
> 
>
> 使用上述方法证明的原因为，例如，向量$Ax$范数如果为无穷范数，则该范数为向量元素绝对值的最大值。该范数只有在$Ax$每个元素都为0的情况下才会为0。所以既然$Ax$的范数大于0，说明$Ax$一定不为零向量，所以$Ax$一定不等于0。





## 二、Sample Complexity with a Generative Model



## 三、Linear Bellman Completeness



<img src="https://z3.ax1x.com/2021/10/21/5sdyNj.png" width="90%" height="60%" align=center />

<div align = "center">图1 整体框架</div>

## 四、Fitted Dynamic Programming Methods

## 五、Statistical Limits of Generalization





## 参考








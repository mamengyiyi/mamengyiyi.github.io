---
layout:     post
title:      多智能体强化学习算法
subtitle:   Mean Field Multi-Agent Reinforcement Learning
date:       2020-09-17
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

论文链接：<a href="https://arxiv.org/pdf/1802.05438.pdf">Mean Field Multi-Agent Reinforcement Learning, ICML 2018</a>


## 一、问题

解决大规模智能体之间的交互及计算困难。由于多智能体强化学习问题不仅有环境交互问题，还有智能体之间的动态影响，因此为了得到最优策略，每个智能体都需要考察其他智能体的动作及状态得到联合动作值函数。由于状态空间跟动作空间随着智能体数量的增多而迅速扩大，这给计算以及探索带来了非常大的困难。

MFMARL算法借用了平均场论（Mean Field Theory，MFT）的思想，其对多智能体系统给出了一个近似假设：对某个智能体，其他所有智能体对其产生的作用可以用一个均值替代。这样就就将一个智能体与其邻居智能体之间的相互作用简化为两个智能体之间的相互作用（该智能体与其所有邻居的均值）。这样极大地简化了智能体数量带来的模型空间的增大。应用平均场论后，学习在两个智能体之间是相互促进的：单个智能体的最优策略的学习是基于智能体群体的动态；同时，集体的动态也根据个体的策略进行更新。

## 二、解法

MFMARL算法主要解决的是集中式多智能体强化学习中，联合动作的维度随智能体数量n的增多极速扩大的情况。因为每个智能体是同时根据联合策略估计自身的值函数，因此当联合动作空间很大时，学习效率及学习效果非常差。为了解决这个问题，算法将值函数$Q^{j}(s, a)$转化为只包含邻居之间相互作用的形式：
$$Q^{j}(s, a)=\frac{1}{N^{j}} \sum_{k \in \mathcal{N}(j)} Q^{j}\left(s, a^{j}, a^{k}\right)$$
其中$N^{j}$表示智能体j邻居智能体的标签集， $N_{j}=|N(j)|$表示邻居节点的个数。上式对智能体之间的交互作用进行了一个近似，降低了表示智能体交互的复杂度，并且保留了部分主要的交互作用（近似保留邻居之间的交互，去掉了非邻居之间的交互）。虽然对联合动作$a$做了近似化简，但是状态信息$s$依然是一个全局信息。

#### 2.1 Mean Field 近似

下面就是将平均场论（Mean Field Theory，MFT）的思想引入上式中。该算法假定所有智能体都是同构的，其动作空间相同，并且动作空间是离散的。每个智能体的动作采用one-hot编码方式， 如智能体$j$的动作$a_{j}=\left[a_{j}^{1}, a_{j}^{2}, \cdots, a_{j}^{D}\right]$表示共有$D$个动作的动作空间每个动作的值，若选取动作$i$，则$a_{j}^{i}=1$，其余为0。定义$\bar{a}\_{j}$为智能体$j$邻居$N(j)$的平均动作，其邻居$k$的one-hot编码动作$a_{k}$可以表示为$\bar{a}\_{j}$与一个波动$\delta a_{j, k}$的和的形式：

$$a_{k}=a_{j}+\delta a_{j, k}, \quad \text { where } \bar{a}_{j}=\frac{1}{N_{j}} \sum_{k} a_{k}$$

利用泰勒公式展开：

$$\begin{array}{c}
Q^{j}(s, a)=\frac{1}{N^{j}} \sum_{k} Q^{j}\left(s, a^{j}, a^{k}\right) \\
\begin{aligned}
=\frac{1}{N^{j}} & \sum_{k}\left[Q^{j}\left(s, a^{j}, \bar{a}^{j}\right)+\nabla_{\bar{a}^{j}} Q^{j}\left(s, a^{j}, \bar{a}^{j}\right) \cdot \delta a^{j, k}\right.\\
&\left.+\frac{1}{2} \delta a^{j, k} \cdot \nabla_{\tilde{a}^{j, k}} Q^{j}\left(s, a^{j}, \tilde{a}^{j, k}\right) \cdot \delta a^{j, k}\right]
\end{aligned} \\
=Q^{j}\left(s, a^{j}, \bar{a}^{j}\right)+\nabla_{\bar{a}^{j}} Q^{j}\left(s, a^{j}, \bar{a}^{j}\right) \cdot\left[\frac{1}{N^{j}} \sum_{k} \delta a^{j, k}\right] \\
+\frac{1}{2 N^{j}} \sum_{k}\left[\delta a^{j, k} \cdot \nabla_{\tilde{a}^{j}, k}^{2} Q^{j}\left(s, a^{j}, \tilde{a}^{j, k}\right) \cdot \delta a^{j, k}\right]
\end{array}$$

其中第二个等号后面的第二项求和为0，第三项为$R_{s, a^{j}}^{j}\left(a^{k}\right) \triangleq \delta a^{j, k} \cdot \nabla_{\tilde{a}^{j, k}}^{2} Q^{j}\left(s, a^{j}, \tilde{a}^{j, k}\right) \cdot \delta a^{j, k}$是泰勒展开的余项，具有如下性质：若值函数$Q_{j}\left(s, a_{j}, a_{k}\right)$是一个$M$阶导数联系函数，则$R_{s, j}\left(a_{k}\right) \in[-2 M,-2 M]$

因此，两两作用求和的形式进一步化简为中心智能体$j$与一个虚拟智能体$\bar{a}\_{j}$的相互作用，虚拟智能体是智能体$j$所有邻居作用效果的平均。因此得到MF-Q函数$Q_{j}\left(s, a_{j}, \bar{a}\_{j}\right)$。 假设有一段经验$\left[s,\left\\{a_{j}\right\\},\left\\{r_{j}\right\\}, s^{\prime}\right]$，MF-Q可以通过下式循环更新：

$$Q_{j, t+1}\left(s, a_{j}, \bar{a}_{j}\right)=(1-\alpha) Q_{j, t}\left(s, a_{j}, \bar{a}_{j}\right)+\alpha\left[r_{j}+\gamma v_{j, t}\left(s^{\prime}\right)\right]$$

对于为什么不取max而是选取MF-v函数的情况，其一是因为取max需要邻居智能体策略的配合，中心智能体不能直接改变邻居智能体的策略；其二取max贪心获取动作，如果每个智能体都贪心获取动作则会因为环境的动态不稳定性而造成算法最终无法收敛。MF-v函数$v_{j, t}\left(s^{\prime}\right)$可以定义为如下形式：

$$v_{j, t}\left(s^{\prime}\right)=\sum_{a_{j}} \pi_{j, t}\left(a_{j} | s, \bar{a}_{j}\right) E_{\bar{a}_{j}\left(a_{-j} \sim \pi_{-j, t}\right)}\left[Q_{j, t}\left(s^{\prime}, a_{j}, \bar{a}_{j}\right)\right]$$

在每一时刻的阶段博弈中， $\bar{a}\_{j}$是通过上一时刻邻居$k$的策略$\pi_{k, t}$得出的，其策略参数中的$\bar{a}\_{k-}$也是使用的上一时刻的平均动作，更新过程如下：

$$\bar{a}_{j}=\frac{1}{N_{j}} \sum_{k} a_{k}, \quad \text { where } a_{k} \sim \pi_{k, t}\left(\cdot | s, \bar{a}_{k-}\right)$$

通过上式可以计算出邻居平均动作$\bar{a}\_{j}$，然后通过玻尔兹曼分布得到新的策略如下形式：

$$\pi_{j, t}\left(a_{j} | s, \bar{a}_{j}\right)=\frac{\exp \left(-\beta Q_{j, t}\left(s, a_{j}, \bar{a}_{j}\right)\right)}{\sum_{a j \in A_{j}} \exp \left(-\beta Q_{j, t}\left(s, a_{j^{\prime}}, \bar{a}_{j}\right)\right)}$$

通过上两式不断迭代更新，能够提升策略的效果而获得较大的累积回报值。原文中证明$\bar{a}\_{j}$能够收敛到唯一的平衡点，并推得策略$\pi_{j}$收敛到纳什均衡策略。为了与Nash-Q算法对应，MF-Q算法给出下式：

$$\mathcal{H}^{M F} Q(s, a)=E_{s^{\prime} \sim p}\left[r(s, a)+\gamma v^{M F}\left(s^{\prime}\right)\right]$$

最终MF-Q的值函数将会收敛到Nash-Q的值函数。

#### 2.2 算法设计

<img src="https://s1.ax1x.com/2020/09/17/wW8I0I.png" alt="wW8I0I.png" style="zoom:80%;" />





## 三、实验内容 

不同的大规模实验上超越多个baselines：

<img src="https://s1.ax1x.com/2020/09/17/wW85nA.png" alt="wW85nA.png" style="zoom:80%;" />

<img src="https://s1.ax1x.com/2020/09/17/wW8hXd.png" alt="wW8hXd.png" style="zoom:80%;" />

## 四、缺点

MFMARL算法主要解决的是联合动作的维度随智能体数量增多的扩张问题，将$a$的维度缩减为$\left[a_{j}, \bar{a}\_{j}\right]$。但是各个智能体的策略还是需要直到全局的状态信息$s$，不算是一个分布式的算法，并且依赖于通信获取邻居智能体的动作$a_{k}$。

## 五、优点

虽然不是完全分布式的，但是该算法是一个解决大规模数量智能体强化学习的一个非常有效的算法，并且理论证明十分严格。

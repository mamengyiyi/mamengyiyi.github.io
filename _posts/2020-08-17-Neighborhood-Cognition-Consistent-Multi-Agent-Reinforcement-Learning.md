---
layout:     post
title:      多智能体强化学习算法
subtitle:   Neighborhood Cognition Consistent Multi-Agent Reinforcement Learning
date:       2020-08-17
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

论文链接：<a href="https://arxiv.org/pdf/1912.01160.pdf">Neighborhood Cognition Consistent Multi-Agent Reinforcement Learning, AAAI 2020</a>


## 一、问题

受社会心理学领域内非常流行的认知一致性理论（Congnitive Consistency Theory）启发，本文作者发现这发现多智能体协作也是适用的：
  * 对环境形成一致性的认知是实现良好协作的必要条件；
  * 智能体只与具有局部感知的邻居智能体直接交互的事实表明，只要邻居智能体之间相互保持认知一致性通常足以保证系统级的协作。
受上述事实启发，本文首次提出了邻域认知一致性（Neighborhood Cognitive Consistency）的概念应用于多智能体研究。

## 二、解法

本文将认知一致性中的“认知”定义为每个智能体对于自己局部环境的理解，包括该智能体领域内所有智能体的局部观察以及从这些局部观察中抽象出的高维表征。

针对离散动作和连续动作，作者设计了NCC-Q和NCC-AC。

### 2.1 NCC-Q

#### 2.1.2 网络结构

<img src="https://s1.ax1x.com/2020/08/17/dZ4Pd1.png" alt="dZ4Pd1.png" style="zoom:80%;" />

如上图所示，NCC-Q的结构分为5个部分：

  - 全连接模块：将局部观察$o_i$编码为仅智能体特定认知的信息$h_i$
  - GCN模块：将邻域中所有的$h_i$进行聚合,得到高维抽象$H_{i}=\sigma\left(W \Sigma_{j \in N(i) \cap\{i\}} \frac{h_{j}}{\sqrt{\|N(j)\|\|N(i)\|}}\right)$。其中$N(i)$表示智能体$i$的邻居智能体。使用$\sqrt{\|N(j)\|\|N(i)\|}$对$h_j$做标准化是为了降低出度（入度）高的邻居的影响。同一邻域内的所有智能体都采用同一个$W$以期望更容易达到邻域认知一致性。
  - 认知模块：将$H_{i}$分解为两部分，分别是智能体特定认知表征$A_i$和邻域特定认知表征$\widehat{C_{i}}$。
  - Q-value模块：将$A_i$与$\widehat{C_{i}}$求和作为输入得到Q值$Q_{i}\left(o_{i}, a_{i}\right)$
  - 混合模块：使用VDN或QMIX的方式将所有$Q_{i}\left(o_{i}, a_{i}\right)$进行加权求和得到联合值函数$Q_{\text {total}}(\vec{o}, \vec{a})$

#### 2.1.3 邻域认知一致性

仅仅在GCN模块中使用同一个$W$无法保证可以达到邻域认知一致性。因此本文提出如下两个假设并进行了对应设计：

  - **对于每一个邻域都有一个真实的认知隐变量$C$，并且从$C$可以得到邻域中每个智能体的局部观察。**如下图所示，从邻域中的观察抽取出每个智能体的局部观察，相比于从全局观察中抽取每个智能体的局部观察，更符合多智能体设置的情理：

    <img src="https://s1.ax1x.com/2020/08/17/dZ4SsJ.png" alt="dZ4SsJ.png" style="zoom:80%;" />

    

    

  - 如果邻域中的智能体可以将$C$进行完整的表征，那么它们最终会达到认知一致性，即**学到的认知表征$\widehat{C_{i}}$要与真实的认知隐变量$C$相近**。

上述两个假设可以形式化定义为：

$$p\left(C | o_{i}\right)=\frac{p\left(o_{i} | C\right) p(C)}{p\left(o_{i}\right)}=\frac{p\left(o_{i} | C\right) p(C)}{\int p(x | C) p(C) d C}$$

但是该真实值很难直接计算，因此用另一个$q\left(C \| o_{i}\right)$分布去近似$p\left(C \| o_{i}\right)$，即最小化两个分布的KL散度：

$$\min K L\left(q\left(C | o_{i}\right) \| p\left(C | o_{i}\right)\right)$$

最小化这两个分布的KL散度等价于最大化如下目标：

$$\max \mathbb{E}_{q\left(C | o_{i}\right)} \log p\left(o_{i} | C\right)-K L\left(q\left(C | o_{i}\right) \| p(C)\right)$$

其中第一项代表重建该分布的可能性，第二项保证两个分布之间的相似性。

这个新分布的构建使用VAE来实现，如下图所示，VAE的enccoder学习将$o_i$映射到$\widehat{C_{i}}$的新分布$q\left(\widehat{C}\_{i} \| o_{i} ; w\right)$。在实际中，参考VAE的经典使用方法，不直接输出$\widehat{C_{i}}$，而是使用reparameterization trick从高斯分布中采样一个$\varepsilon$，并使用隐变量分布的均值和方差来修正该值，即$\widehat{C_{i}}=\widehat{C_{i}^{\mu}}+\widehat{C_{i}^{\sigma}} \odot \varepsilon$。而VAE的decoder部分则学习一个将$\widehat{C_{i}}$映射回$o_i$的分布$p(\widehat{\partial_{i}} \| \widehat{C_{i}} ; w)$。

<img src="https://s1.ax1x.com/2020/08/17/dZ4CZR.png" alt="dZ4CZR.png" style="zoom:80%;" />

训练该VAE的loss如下：

$$\min L 2\left(o_{i}, \widehat{o_{i}} ; w\right)+K L\left(q\left(\widehat{C_{i}} | o_{i} ; w\right) \| p(C)\right)$$

#### 2.1.4 NCC-Q的训练

训练NCC-Q时使用两个loss；

  * TD loss：鼓励所以智能体最大化$Q_{t o t a l}$，即$$\begin{aligned}
  L^{t d}(w) &=\mathbb{E}_{\left(\vec{o}, \vec{a}, r, \vec{o}^{\prime}\right)}\left[\left(y_{t o t a l}-Q_{t o t a l}(\vec{o}, \vec{a} ; w)\right)^{2}\right] \\
  y_{t o t a l} &=r+\gamma \max _{\vec{a}^{\prime}} Q_{t o t a l}\left(\vec{o}^{\prime}, \vec{a}^{\prime} ; w^{-}\right)
  \end{aligned}$$

  * 认知不一致loss：$$L_{i}^{c d}(w)=\mathbb{E}_{o_{i}}\left[L 2\left(o_{i}, \widehat{o_{i}} ; w\right)+K L\left(q\left(\widehat{C_{i}} | o_{i} ; w\right) \| p(C)\right)\right]$$
  因此，总的loss则为：$$L^{\text {total}}(w)=L^{t d}(w)+\alpha \Sigma_{i=1}^{N} L_{i}^{c d}(w)$$
  但是在实际中，我们无法获得真实的认知隐变量$C$及其分布$p(C)$，而且如何为每个邻域选择一个$p(C)$也是一件很复杂的事情。因此，本文使用每个智能体及其邻域内其他智能体的认知来作为替代，即**让同一邻域内智能体的认知达到一致即可，而不规定它们需要达到怎样的认知**，即替换认知不一致loss为：

  $$\begin{aligned}
  L_{i}^{c d}(w)=& \mathbb{E}_{o_{i}}\left[L 2\left(o_{i}, \widehat{\partial}_{i} ; w\right)+K L\left(q\left(\widehat{C}_{i} | o_{i} ; w\right) \| p(C)\right)\right] \\
  \approx & \mathbb{E}_{o_{i}}\left[L 2\left(o_{i}, \widehat{o}_{i} ; w\right)+\frac{1}{|N(i)|} \Sigma_{j \in N(i)} K L\left(q\left(\widehat{C_{i}} | o_{i} ; w\right) \| q\left(\widehat{C_{j}} | o_{j} ; w\right)\right)\right]
  \end{aligned}$$

### 2.2 NCC-AC

该方法思路与NCC-Q大致相同。其中，网络结构为：

<img src="https://s1.ax1x.com/2020/08/17/dZ4pL9.png" alt="dZ4pL9.png" style="zoom:80%;" />



训练的loss为：

$$\begin{aligned}
L_{i}^{\text {total}}\left(w_{i}\right)=& L_{i}^{t d}\left(w_{i}\right)+\alpha L_{i}^{c d}\left(w_{i}\right) \\
L_{i}^{t d}\left(w_{i}\right)=& \mathbb{E}_{\left(o_{i}, \vec{o}_{-i}, a_{i}, \vec{a}_{-i}, r, o_{i}^{\prime}, \vec{o}_{-i}^{\prime}\right) \sim D}\left[\left(\delta_{i}\right)^{2}\right] \\
\delta_{i}=& r+\left.\gamma Q_{i}\left(\left\langle o_{i}^{\prime}, a_{i}^{\prime}\right\rangle, \vec{o}_{-i}^{\prime}, \vec{a}_{-i}^{\prime} ; w_{i}^{-}\right)\right|_{a_{j}^{\prime}=\mu_{\theta}-\left(o_{j}^{\prime}\right)} - Q_{i}\left(\left\langle o_{i}, a_{i}\right\rangle, \vec{o}_{-i}, \vec{a}_{-i} ; w_{i}\right) \\L_{i}^{c d}\left(w_{i}\right) \approx & \mathbb{E}_{o_{i}}\left[L 2\left(o_{i}, \widehat{o_{i}} ; w_{i}\right)+L 2\left(a_{i}, \widehat{a}_{i} ; w_{i}\right)+\frac{1}{|N(i)|} \sum_{j \in N(i)} K L\left(q\left(\widehat{C_{i}} | o_{i}, a_{i} ; w_{i}\right) \| q\left(\widehat{C_{j}} | o_{j}, a_{j} ; w_{j}\right)\right)\right]
\end{aligned}$$





## 三、实验内容 

<img src="https://s1.ax1x.com/2020/08/17/dZ4ki6.png" alt="dZ4ki6.png" style="zoom:90%;" />

<img src="https://s1.ax1x.com/2020/08/17/dZ4iIx.png" alt="dZ4iIx.png" style="zoom:80%;" />

## 四、缺点

*  划分邻域的方式还是通过距离
*  不同邻域之间的智能体没有显式进行弱协调。

## 五、优点

从该文及之前的马尔萨斯强化学习一文中可以总结出一个道理，即在研究多智能体系统时，利用社会学、人口学、心理学等相关的知识对当前多智能体系统的不足进行理论上的改进是一个非常有insight的方式。可以考虑研究一下人类社群中相关的知识，在当前这些工作基础上进行深入研究。

当前多智能体研究的三个可切入点：
  * 怎么对所有智能体进行动态、自适应划域？
  * 怎么对同一邻域内的智能体进行强协调？
  * 怎么对不同邻域内的智能体进行适当的弱协调？

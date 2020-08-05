---
layout:     post
title:      单智能体强化学习算法
subtitle:   EC：Episodic Curiosity through Reachability
date:       2020-08-05
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - RL advanced algorithms
    - Exploration in RL
---
---

论文链接：<a href="https://openreview.net/pdf?id=SkeK3s0qKQ">Episodic Curiosity through Reachability, ICLR 2019</a>

## 一、问题 

许多基于好奇心的方法都是通过对预测难度越大的state给予越多的奖励，但这种方法在“看电视”的环境中会导致agent沉迷于其中，而不再进行迷宫中的有效探索。（迷宫环境中有个电视，agent可以控制电视换台，每次换台都会随机产生一个画面。agennt的目的是获取迷宫中某处的宝藏）。

基于这种问题，本文作者提出，如果agent可以意识到给电视换台得到的observation仅仅是执行一步操作即可达到的，即这种所谓新奇的observation很容易得到的话，那么一种直观的解决方法就是把intrinsic reward给到需要花费很多step才能获取到的observation。

## 二、解法 
本文的解法框架如下：在一个episode的开始阶段，agent有一个空的episodic memory；在每个step，agent将当前的observation与meomory中的observations进行比较来判断当前的observation是否是“新奇”的：如果需要比阈值step $k$花费更多step才能从memory中的obseravtions到达当前observation，那么认为该observation是“新奇”的，并给予一个intrinsic reward （如下图中，橙色部分的observations为新奇的）。考虑到obs之间的transition无法完整获得，本文使用一个神经网络来预测一个observation与memory中的observations的step距离是否大于阈值$k$。

![1](https://s1.ax1x.com/2020/08/05/arBPnf.png)

### Episodic Curiosity算法框架

具体来说，算法流程如下：

![1](https://s1.ax1x.com/2020/08/05/arBuj0.png)

首先，当前的observation $\mathbf{o}$经过embedding网络产生一个embedding向量$\mathbf{e}=E(\mathbf{o})$，该observation的embedding则会与存在memory buffer中的所有embedding向量$\mathbf{M}=\left\langle\mathbf{e}_{1}, \ldots,\mathbf{e}_{\|\mathbf{M}\|}\right\rangle$进行比较，比较的方式是通过一个通过逻辑回归loss训练得到的comparator network $C$：如下图所示，如果两个observation在$k$步内可达的概率很低，则输出接近0，否则输出接近1。针对每个memory中observation比较得到的分数，存到reachability buffer中。

![1](https://s1.ax1x.com/2020/08/05/arBlHU.png)

其次，使用reachability buffer中的值来计算当前obseravtion与整个memory buffer的相似程度：

$$C(\mathbf{M}, \mathbf{e})=F\left(c_{1}, \ldots, c_{|\mathbf{M}|}\right) \in[0,1]$$

其中$F$理论上应该用max，但是实践中发现用90%分位数更好。

再次，计算curiosiity bonus作为intrinsic reward：

$$b=B(\mathbf{M}, \mathbf{e})=\alpha(\beta-C(\mathbf{M}, \mathbf{e}))$$

其中$\alpha$用来放缩intrinsic reward以便和外部reward一致；$\beta$用以决定bonus的正负，$\beta=0.5$适用于固定长度的episode，$\beta=1$适用于变长的episode。

最后，若当前bonus值大于bonus阈值大于$b_{\text {novelty}}$，则将当前obderavtion的embedding加入到memory buffer中（若每个observation embedding都被加到Buffer中，那么当前observation必然是reachable的，则bonus永远不会被计算）。若buffer满了，则随机替换其中的一条，以保证新旧memory都有，以保持buffer中样本的丰富性。

### 网络训练

预测网络的训练使用逻辑回归loss，使用到的正负样本如下图所示：

![1](https://s1.ax1x.com/2020/08/05/ar0bnK.png)

样本的采样则通过离线的随机采样或者online采样定时更新的方法进行。

## 三、实验内容
在VizDoom和DMLab上对比了PPO + ICM, PPO + Grid Oracle和本文提出的PPO + EC：

![1](https://s1.ax1x.com/2020/08/05/ar0gmT.png)

![1](https://s1.ax1x.com/2020/08/05/ar0R7F.png)

![1](https://s1.ax1x.com/2020/08/05/ar020U.png)

## 四、缺点
暂无评价。 
  
## 五、优点
这种通过定义难以到达的state的想法很直观，对于传统基于预测误差的方法陷入局部解走不出来的情况提出了一种解决方案。

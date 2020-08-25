---
layout:     post
title:      多智能体强化学习算法
subtitle:   Graph Convolutional Reinforcement Learning
date:       2020-08-25
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

论文链接：<a href="https://openreview.net/pdf?id=HkxdQkSYDB">Graph Convolutional Reinforcement Learning, ICLR 2020</a>


## 一、问题

本文考虑到邻居智能体之间的交互更具有影响力，因此将智能体之间的关系视为一个graph，利用graph来研究智能体之间的合作关系。

## 二、解法

### 2.1 graph convolution

如下图所示，本文提出的DGN使用三层结构进行构建：

<img src="https://s1.ax1x.com/2020/08/25/dgPFC6.png" alt="dgPFC6.png" style="zoom:67%;" />

由于智能体的位置和数量一直在变化，图卷积不能直接适用，因此本文把所有智能体的特征合并到一个尺寸为$N \times L$的特征矩阵$F^{t}$，其中$N$是智能体数量，$L$是特征向量的长度。同时对每个智能体$i$构造一个尺寸为$\left(\left\|\mathbb{B}\_{i}\right\|+1\right) \times N$的邻接矩阵$C^{t}\_{i}$，其中第一行是智能体$i$的onehot向量id，第$j$行为第$j-1$个邻居的onehot向量id。通过$C_{i}^{t} \times F^{t}$可以得到智能体$i$的邻域特征。

受DenseNet影响，本文将之前每一层的输出都作为Q网络的输入以利用不同感知域的抽象特征。

训练时使用的loss为：

$$\mathcal{L}(\theta)=\frac{1}{\mathrm{S}} \sum_{\mathrm{S}} \frac{1}{\mathrm{N}} \sum_{i=1}^{\mathrm{N}}\left(y_{i}-Q\left(O_{i, \mathcal{C}}, a_{i} ; \theta\right)\right)^{2}$$

其中$y_{i}=r_{i}+\gamma \max_{a^{\prime}} Q\left(O_{i, c}^{\prime}, a_{i}^{\prime} ; \theta^{\prime}\right)$，$O_{i, \mathcal{C}} \subseteq \mathcal{O}$表示由$\mathcal{C}$决定的智能体$i$的感知域内的所有智能体的观察。由于graph变化较快时Q网络不容易收敛，因此在计算Q-loss时，每两个step再更新一次$\mathcal{C}$

### 2.2 relation kernel

本文使用multi-head attention来聚合邻居智能体之间的特征：

$$\alpha_{i j}^{m}=\frac{\exp \left(\tau \cdot \mathbf{W}_{Q}^{m} h_{i} \cdot\left(\mathbf{W}_{K}^{m} h_{j}\right)^{\top}\right)}{\sum_{k \in \mathbb{B}_{+i}} \exp \left(\tau \cdot \mathbf{W}_{Q}^{m} h_{i} \cdot\left(\mathbf{W}_{K}^{m} h_{k}\right)^{\top}\right)}$$

将M个attention head的输出连接起来，再通过全连接和ReLU得到图网络层的输出：

$$h_{i}^{\prime}=\sigma\left(\text { concatenate }\left[\sum_{j \in \mathbb{B}_{+i}} \alpha_{i j}^{m} \mathbf{W}_{V}^{m} h_{j}, \forall m \in \mathbb{M}\right]\right)$$

流程如下图所示：

<img src="https://s1.ax1x.com/2020/08/25/dgPAgO.png" alt="dgPAgO.png" style="zoom: 67%;" />

### 2.3 Temporal Relation Regularization

考虑到合作一般都会持续一段时间，因此即使邻居智能体的特征发生变化，relation kernel也应该在短时间内保持一致。采用类似TD的方法，将下一时刻状态所对应的attention权重作为当前时刻状态对应的attention权重的目标，尽量最小化二者之间的KL散度。注意，在计算目标状态对应的attention权重时使用的是eval net而不是target net，因为target更新滞后，计算出的attention权重分布并不是真实的分布，因此训练的loss变为：

$$\mathcal{L}(\theta)=\frac{1}{\mathrm{S}} \sum_{\mathrm{S}} \frac{1}{\mathrm{N}} \sum_{i=1}^{N}\left(\left(y_{i}-Q\left(O_{i, \mathcal{C}}, a_{i} ; \theta\right)\right)^{2}+\lambda \frac{1}{\mathrm{M}} \sum_{m=1}^{\mathrm{M}} D_{\mathrm{KL}}\left(\mathcal{G}_{m}^{\kappa}\left(O_{i, \mathcal{C}} ; \theta\right) \| \mathcal{G}_{m}^{\kappa}\left(O_{i, \mathcal{C}}^{\prime} ; \theta\right)\right)\right.$$

其中$\mathcal{G}\_{m}^{\kappa}\left(O_{i, \mathcal{C}} ; \theta\right)$是智能体$i$的attention head $m$在第$\kappa$个卷积层的attention权重分布。



## 三、实验内容 

在三个环境battle,jungle和routing上面做了实验，超越了baselines：

<img src="https://s1.ax1x.com/2020/08/25/dgPk8K.png" alt="dgPk8K.png" style="zoom:67%;" />

<img src="https://s1.ax1x.com/2020/08/25/dgPP4x.png" alt="dgPP4x.png" style="zoom:67%;" />

<img src="https://s1.ax1x.com/2020/08/25/dgPCU1.png" alt="dgPCU1.png" style="zoom:67%;" />



## 四、缺点

由于论文时隔很久才发表，现在看来已经不新鲜了。感觉现在多智能体和图网络结合基本都是这个套路。

## 五、优点

让邻域内的智能体在一段时间内保持合作的这个想法比较符合人类社会的设定，值得借鉴。

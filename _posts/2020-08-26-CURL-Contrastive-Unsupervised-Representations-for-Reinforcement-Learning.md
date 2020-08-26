---
layout:     post
title:      单智能体强化学习算法
subtitle:   CURL：Contrastive Unsupervised Representations for Reinforcement Learning
date:       2020-08-26
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: false
tags:
    - RL
    - DL
    - RL advanced algorithms
    - Contrastive Learning
---


------

论文链接：<a href="https://arxiv.org/pdf/2004.04136.pdf">CURL: Contrastive Unsupervised Representations for Reinforcement Learning, ICML 2020</a>

代码链接：<a href="https://www.github.com/MishaLaskin/curl.">github链接</a>


## 一、问题

RL从pixel中直接端到端学习状态表征的效率非常低，受contrastive learning的思想影响，本文考虑用这种方式来提取pixel的高维特征。但是在RL中采用contrastive learning与在CV中采用contrastive learning有两个区别：（1）事先没有可用的大量地无标签的数百万张图像的数据集，因为在RL中该数据集是从智能体与环境的互动中在线收集的，并会随智能体的经验变化而动态变化； （2）智能体必须同时进行无监督学习和强化学习，而不是针对特定的下游任务微调预先训练的网络。本文借鉴何凯明提出的 Momentum Contrast设计了CURL来解决这些问题。

## 二、解法

### 2.1 整体框架

本文将Contrastive Loss作为RL训练的辅助Loss。从replay buffer中采样得到observation，将observation通过数据增强方法得到锚定数据$o_{q}$以及对应的正负样本$o_{k}$。训练中方向传播时只更新query encoder $q=f_{\theta_{q}}\left(o_{q}\right)$的参数$\theta_{q}$，而key encoder $k=f_{\theta_{k}}\left(o_{k}\right)$的参数$\theta_{k}$的参数则通过monoentum的方式进行更新，即$\theta_{k}=m \theta_{k}+(1-m) \theta_{q}$。整体方法框架如下图所示：

<img src="https://s1.ax1x.com/2020/08/26/dRXbAP.png" alt="dRXbAP.png" style="zoom:80%;" />



### 2.2 构造正负样本

CV中常用的方法是使用image的patches作为正负样本。考虑到RL的特性，使用这种需要设计如何获取pathces的较为复杂的方式会为RL的训练带来更大的难度，因此，CURL使用instance区分而不是patch区分。

类似于图像中的instance discrimination，锚定数据和正样本是同一图像的两种不同数据增强，而负样本则来自其他图像。 CURL采用随机裁剪的数据增强方式，即从原始图片中裁剪出一个随机的方块作为样本。过程如下图所示：

<img src="https://s1.ax1x.com/2020/08/26/dRX77t.png" alt="dRX77t.png" style="zoom:67%;" />

### 2.3 衡量样本相似性

与CV中的MoCo和SimCLR不同，本文衡量样本相似性时没有采用点积的方式，而是采用双线性数量积，即：

$$\operatorname{sim}(q, k)=q^{T} W k$$

其中$W$是学出来的参数矩阵。

### 2.4 方法代码

<img src="https://s1.ax1x.com/2020/08/26/dRXqtf.png" alt="dRXqtf.png" style="zoom:67%;" />



## 三、实验内容 

本文的设计的CURL可以与任何强化学习算法结合。本文采用了SAC与RAINBOW来分别验证CURL在连续动作空间与离散动作空间上的效果。本文分别在DMControl与Atari环境上进行了测试，效果超越了SOTA的RL方法：

<img src="https://s1.ax1x.com/2020/08/26/dRXLh8.png" alt="dRXLh8.png" style="zoom:80%;" />

<img src="https://s1.ax1x.com/2020/08/26/dRXX9S.png" alt="dRXX9S.png" style="zoom:80%;" />



## 四、缺点

* 本文只是在图像的角度上做了状态表征。
* 根据后续研究的说法，CURL的大部分优势来自图像增强的使用，而不是对比损失。

## 五、优点

  * 相比于之前的工作，在结合Contastive Learning的时候更多地考虑到了RL的特性
* 方法易于实现，方便集成在不同的RL算法中

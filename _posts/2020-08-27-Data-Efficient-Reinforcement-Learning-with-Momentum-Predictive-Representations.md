---
layout:     post
title:      单智能体强化学习算法
subtitle:   Data-Efficient Reinforcement Learning with Momentum Predictive Representations
date:       2020-08-27
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

论文链接：<a href="https://arxiv.org/pdf/2007.05929.pdf">Data-Efficient Reinforcement Learning with Momentum Predictive Representations, NIPS 2020 Under Riview</a>


## 一、问题

使用Contrastive Learning的方法主要的思想核心是提取agent交互环境的有效表征，使得RL可以更好地感知环境做出决策。本文认为如果给定未来的动作，对于未来观察的状态表征是较好预测，且对于数据增强的预测比较稳定，那说明对于环境的状态提取的特征是非常有效的。基于此想法，本文提出了Momentum Predictive Representations (MPR)。

## 二、解法

### 2.1 整体框架

MPR的整体方法框架如下图所示：



<img src="https://s1.ax1x.com/2020/08/27/d4Y4Xt.png" alt="d4Y4Xt.png" style="zoom:80%;" />



MPR主要包括四个组件：online and momentum encoder、Transition Model、Projection Heads和Prediction Loss

#### 2.1.1 online and momentum encoder

online and momentum encoder分别用于构造类似于contrastive中的锚定数据和正样本。假设$\left(s_{t: t+K}, a_{t: t+K}\right)$是从buffer里采样得到的$K+1$个状态动作对，其中$K$是想要预测未来的step数。online encoder $f_{o}$用于将state $s_{t}$转换为表征$z_{t} \triangleq f_{o}\left(s_{t}\right)$，而momentum encoder $f_{m}$采用何凯明的MoCO的方式进行参数更新$\theta_{\mathrm{m}} \leftarrow \tau \theta_{\mathrm{m}}+(1-\tau) \theta_{\mathrm{o}}$，不经过梯度下降更新

#### 2.1.2 Transition Model

给定动作$a_{t+k}$，Transition Model是一个用于迭代地预测未来state的状态表征$\hat{z}_{t+k+1} \triangleq h\left(\hat{z}_{t+k}, a_{t+k}\right)$的CNN模型。Transition Model和预测损失在得到的latent space中运行，可以避免了基于像素进行目标重建

#### 2.1.3 Projection Heads

本文使用online和momentum映射模块$g_{o}$和$g_{m}$将online表征和momentum表征映射到一个较小的latent space中，并额外使用一个预测模块$q$来通过online映射对momentum映射进行预测：

$$\hat{y}\_{t+k} \triangleq q\left(g_{o}\left(\hat{z}\_{t+k}\right)\right), \forall \hat{z}\_{t+k} \in \hat{z}_{t+1: t+K} ; \quad \tilde{y}_{t+k} \triangleq g_{m}\left(\tilde{z}\_{t+k}\right), \forall \tilde{z}\_{t+k} \in \tilde{z}\_{t+1: t+K}$$

#### 2.1.4 Prediction Loss

本文使用余弦相似度来比较未来$t+k$个step时，momentum部分表征的真实observation和online部分表征的预测observation之间的差异：

$$\mathcal{L}^{\mathrm{MPR}}\left(s_{t: t+K}, a_{t: t+K}\right)=-\sum_{k=1}^{K}\left(\frac{\tilde{y}_{t+k}}{\left\|\tilde{y}_{t+k}\right\|_{2}}\right)^{\top}\left(\frac{\hat{y}_{t+k}}{\left\|\hat{y}_{t+k}\right\|_{2}}\right)$$

得到Prediction loss后，将它作为RL训练的辅助loss：

$$\mathcal{L}_{\theta}^{\text {total }}=\mathcal{L}_{\theta}^{\mathrm{RL}}+\lambda \mathcal{L}_{\theta}^{\mathrm{MPR}}$$

### 2.2 算法流程

<img src="https://s1.ax1x.com/2020/08/27/d4VFsA.png" alt="d4VFsA.png" style="zoom:80%;" />



### 2.3 数据增强

本文的数据增强方法采用的是对图像进行随机位移以及色彩抖动。本文还验证了不适用数据增强时，在两个encoder的每一层中采用概率为0.5的dropout效果更好。

## 三、实验内容 

在Atari游戏上进一步超越包括CURL在内的SOTA方法：

<img src="https://s1.ax1x.com/2020/08/27/d4VkqI.png" alt="d4VkqI.png" style="zoom:80%;" />

同时验证了momentum encoder的重要性、数据增强的重要性、预测模型的重要性：

<img src="https://s1.ax1x.com/2020/08/27/d4VCxH.png" alt="d4VCxH.png" style="zoom:80%;" />

## 四、缺点

* 思路还是局限于对状态做表征，对于RL本身的结构没有进行改进
* 严格来说本文不属于Contrastive Learning的范畴

## 五、优点

  * 相比于之前的工作，增强对未来的预测更加提高了数据增强的效果
* 相比于contrastive learning，本文的方法无需负样本，但可能需要通过设计合适的contrastive task以及使用较大的batch size来弥补无负样本带来的问题
* 综合几篇论文的结论，Contrastive Learning本身的效果不一定好，起主要作用的可能是数据增强

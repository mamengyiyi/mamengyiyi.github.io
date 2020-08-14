---
layout:     post
title:      多智能体强化学习算法
subtitle:   Action Semantics Network：Considering the Effects of Actions in Multiagent Systems
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

论文链接：<a href="https://openreview.net/pdf?id=ryg48p4tPH">Action Semantics Network: Considering the Effects of Actions in Multiagent Systems, ICLR 2020</a>


## 一、问题

本文显式考虑多智能体环境中，智能体不同的动作会对其他智能体产生不同的影响，比如一部分动作会影响环境或自身，另一部分动作会直接影响其他智能体，这种性质称为动作语义。基于此，本文提出了动作语义网络ASN。

## 二、解法

### 2.1 ASN结构

![dCTiFg.png](https://s1.ax1x.com/2020/08/14/dCTiFg.png)

对于value-based方法来说，左侧棕色部分的网络输入全部的$o^{i}_{t}$，输出$Q\left(o_{t}^{i}, a_{t}^{i}\right)=f a\left(e_{t}^{i}, a_{t}^{i}\right)$；右侧蓝色部分，每个部分输入agent对其他一个agent的观察，输出棕色部分得到的embeddinh与自身embedding的内积$Q\left(o_{t}^{i}, a_{t}^{i, j}\right)=\mathcal{M}\left(e_{t}^{i}, e_{t}^{i, j}\right)$。

对于policy-based方法来说，输出则为$\pi\left(a_{t}^{i} | o_{t}^{i}\right)=\frac{\exp \left(f a\left(e_{t}^{i}, a_{t}^{i}\right)\right)}{Z^{\pi_{i}}\left(o_{t}^{i}\right)}, \pi\left(a_{t}^{i, j} | o_{t}^{i}\right)=\frac{\exp \left(\mathcal{M}\left(e_{t}^{i}, e_{t}^{i, j}\right)\right)}{Z^{\pi_{i}}\left(o_{t}^{i}\right)}$，其中$Z^{\pi_{i}}\left(o_{t}^{i}\right)$是用于正则化分布的函数。

最终组合每部分输出得到最终的Q值或者概率分布。

### 2.2 与值分解方法结合

ASN通过以下方式与IDL或JAL范式的MARL算法相结合：
  * ASN-PPO：使用ASN替换PPO中的策略网络

  * ASN-QMIX：使用ASN替换QMIX中的Q网络

    ![dCTkWj.png](https://s1.ax1x.com/2020/08/14/dCTkWj.png)

### 2.3 ASN的扩展与参数共享

考虑到一个agent可能含有多个对其他agent直接产生影响的动作（比如一个士兵有多种武器打击敌人），可以将ASN扩展为如下形式：



![dCTCTS.png](https://s1.ax1x.com/2020/08/14/dCTCTS.png)



对于同质或者异质的智能体，可用如下方式进行参数共享：

![dCTFYQ.png](https://s1.ax1x.com/2020/08/14/dCTFYQ.png)



## 三、实验内容 

在星际2与Neural MMO上进行了实验：

- 实验效果超过普通IQL、VDN、QMIX方法
- 证明了ASN在大规模智能体场景下依然适用（15m环境）
- 证明了ASN可以识别出不同动作的影响
- 证明了ASN可以提高对动作估计的准确度
- 证明了ASN并不是过拟合智能体死后补零的信息
- 证明了ASN可以在多个动作中识别出最好的动作

## 四、缺点

暂无评价。

## 五、优点

* 想法简洁有效；实验非常充分
* 这种将智能体不同类型的观察分开进行处理的方式，可以用于分别提取智能体之间的关系和智能体与环境之间的关系
* 该方法已在网易某游戏中进行线上测试，效果很好

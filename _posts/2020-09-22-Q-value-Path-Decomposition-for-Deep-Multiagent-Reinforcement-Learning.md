---
layout:     post
title:      多智能体强化学习算法
subtitle:   Q-value Path Decomposition for Deep Multiagent Reinforcement Learning
date:       2020-09-22
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

论文链接：<a href="https://arxiv.org/pdf/2002.03950.pdf">Q-value Path Decomposition for Deep Multiagent Reinforcement Learning, ICML 2020</a>


## 一、问题

VDN、QMIX和QTRAN等值函数分解的方法对每个智能体的Q函数和全局的Q函数之间的表征关系都有限制。本文利用积分梯度归因技术来将全局Q分解给每个智能体。

## 二、解法

#### 2.1 积分梯度

积分梯度法 (Integrated Gradient)的提出是为了解决传统基于梯度的可解释性方法的一个缺陷 -- 梯度饱和。在最原始的 Saliency map方法中，假设神经网络的分类结果线性依赖于输入图片中的每个像素或特征，表示为$y=x W+b$, 则输出$y$对输入$x$的梯度$W=\frac{\partial y}{\partial x}$能够直接用来量化每个像素对分类决策的重要程度。

然而，真正的神经网络高度非线性。某个像素或特征增强到一定程度后可能对网络决策的贡献达到饱和。李宏毅老师举过一个例子，大象的鼻子对神经网络将一个物体识别为大象的决策很重要，但当大象的鼻子长度增加到一定程度后（比如1米），继续增加不会带来决策分数的增加，导致输出对输入特征的梯度为0。

<img src="https://s1.ax1x.com/2020/09/22/wLQAoD.png" alt="wLQAoD.png" style="zoom: 50%;" />

对于鼻子长度大于等于1米的大象，为了正确捕捉鼻子长度的重要性，积分梯度法不是使用上面这张图中粉红色部分的梯度（基本为0），而是使用沿整条梯度线的积分值，作为鼻子长度对决策分类的重要程度。写成公式就是

<img src="https://s1.ax1x.com/2020/09/22/wLQCsx.png" alt="wLQCsx.png" style="zoom:50%;" />

困难的地方在于对于一张给定的图片，大象鼻子长度已定（比如=2 米), 如何得到鼻子长度小于2米时输出对输入的梯度呢？假设当前图像为$x$，如果知道鼻子长度= 0米时的基线图像$x'$，那倒可以做一个线性插值：$x^{\prime}+\alpha\left(x-x^{\prime}\right)$

当常数$\alpha=0$时，输入图像为基线图像$x'$, 当$\alpha=1$时, 就是当前图像，在中间即为其他图像。这种方法不能说得到了鼻子长度改变的梯度积分，只能说得到了图像所有像素变化时的梯度积分。

假设神经网络的输出为函数$f$, 则积分梯度法的最终公式为：

$$\phi_{i}^{I G}\left(f, x, x^{\prime}\right)=\overbrace{\left(x_{i}-x_{i}^{\prime}\right)}^{\text {Difference from baseline }} \times \int_{\alpha=0}^{1} \frac{\delta f\left(x^{\prime}+\alpha\left(x-x^{\prime}\right)\right)}{\delta x_{i}} d \alpha$$

注意第一项$x_{i}-x_{i}^{\prime}=$来自于后面积分变量$d\left[\alpha\left(x-x^{\prime}\right)\right]$。分母上的$\delta x_{i}$表示变分。这里整个偏导被换成了变分的形式，变分边界是基线图像和当前图像，变分路径可以任意选择。积分梯度法使用线性插值作为变分路径。

总结：直接使用输出对输入的梯度作为特征重要性会遇到梯度饱和问题。积分梯度法从通过对梯度沿不同路径积分，期望得到非饱和区非零梯度对决策重要性的贡献。积分路径一般选作线性插值。

#### 2.2 积分梯度应用于值函数分解

深度学习中很难知道特征如何从输入变到baseline的，而RL中天然存在一条路径，即状态-动作转移。终止状态的$Q\left(s_{T}, \varnothing\right)=0$可以作为baseline，则$Q_{t o t}$可以分解为：

$$Q_{t o t}\left(\vec{o}_{t}, \vec{a}_{t}\right)=\sum_{x_{j} \in \mathbb{X}_{1}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}\left(\vec{o}_{t}, \vec{a}_{t}\right)+\ldots+\sum_{x_{j} \in \mathbb{X}_{n}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}\left(\vec{o}_{t}, \vec{a}_{t}\right)$$

证明如下：

$$\begin{array}{l}
Q_{t o t}\left(\vec{o}_{t}, \vec{a}_{t}\right)=Q_{t o t}\left(\vec{x}_{t}\right)=Q_{t o t}\left(\vec{x}_{t}\right)-Q_{t o t}\left(\vec{x}_{T}\right)=Q_{t o t}\left(\vec{x}_{t}\right)-Q_{t o t}\left(\vec{x}_{t+1}\right) \\
+Q_{t o t}\left(\vec{x}_{t+1}\right)-Q_{t o t}\left(\vec{x}_{t+2}\right)+\ldots+Q_{t o t}\left(\vec{x}_{T-1}\right)-Q_{t o t}\left(\vec{x}_{T}\right) \\
=\sum_{j=1}^{\left|\vec{x}_{t}\right|} I G_{j}^{\tau_{t}^{t+1}}(\vec{x})+\sum_{j=1}^{\left|\vec{x}_{t}\right|} I G_{j}^{\tau_{t+1}^{t+2}}(\vec{x})+\ldots+\sum_{j=1}^{\left|\vec{x}_{t}\right|} I G_{j}^{\tau_{T-1}^{T}}(\vec{x}) \\
=\operatorname{Path} I G_{j=1}^{\tau_{t}^{T}}(\vec{x})+\operatorname{Path} I G_{j=2}^{\tau_{t}^{T}}(\vec{x})+\ldots+\operatorname{Path} I G_{j=\left|\vec{x}_{t}\right|}^{\tau_{t}^{T}}(\vec{x}) \\
=\sum_{x_{j} \in \mathbb{X}_{1}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}(\vec{x})+\sum_{x_{j} \in \mathrm{X}_{2}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}(\vec{x})+\ldots+\sum_{x_{j} \in \mathbb{X}_{n}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}(\vec{x}) \\
=\sum_{i=1}^{n} \sum_{x_{j} \in \mathrm{X}_{i}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}(\vec{x})=\sum_{i=1}^{n} \sum_{x_{j} \in \mathrm{X}_{i}} \operatorname{Path} I G_{j}^{\tau_{t}^{T}}(\vec{o}, \vec{a})
\end{array}$$

由于每两个相邻的联合观察和动作之间的路径是路径中的直线，因此$\sum_{x_{i} \in \mathbb{X}_{i}}$PathIG$_{j}^{\tau_{t}^{T}}\left(\vec{o}_{t}, \vec{a}_{t}\right)$可以通过如下方式计算：
$$\begin{array}{l}
\sum_{x_{j} \in \mathbb{X}_{i}} \operatorname{Path} I G_{j}^{\tau_{f}^{T}}\left(\vec{o}_{t}, \vec{a}_{t}\right)= \sum_{x_{j} \in \mathbb{X}_{i}} I G_{j}^{\tau_{f}^{t+1}}(\vec{o}, \vec{a})+\sum_{x_{j} \in \mathbb{X}_{i}} I G_{j}^{\tau_{i+1}^{t+2}}(\vec{o}, \vec{a})+\ldots+\sum_{x_{j} \in \mathbb{X}_{i}} I G_{j}^{\tau_{T-1}^{T}}(\vec{o}, \vec{a})
\end{array}$$

#### 2.3 架构设计

<img src="https://s1.ax1x.com/2020/09/22/wLQPL6.png" alt="wLQPL6.png" style="zoom: 67%;" />

<img src="https://s1.ax1x.com/2020/09/22/wLQFeK.png" alt="wLQFeK.png" style="zoom:80%;" />



#### 2.4 算法流程

<img src="https://s1.ax1x.com/2020/09/22/wLQkdO.png" alt="wLQkdO.png" style="zoom:80%;" />





## 三、实验内容 

在混合兵种的星际2环境上效果很好：

<img src="https://s1.ax1x.com/2020/09/22/wLQVFe.png" alt="wLQVFe.png" style="zoom:80%;" />

## 四、缺点

暂无评价。

## 五、优点

* 首次在多智能体中使用路径积分技术，也给出了对应的理论支撑
* 信度分配方式给人的感觉就是比较契合值函数分解的想法，可以从广告推荐那边的论文借鉴一些想法继续做下去。
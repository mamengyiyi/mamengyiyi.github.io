---
layout:     post
title:      单智能体强化学习算法
subtitle:   Behavior Regularized Offline Reinforcement Learning
date:       2020-09-15
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: false
tags:
    - RL
    - RL advanced algorithms
    - Offline Reinforcement Learning
---


------

论文链接：<a href="https://openreview.net/attachment?id=BJg9hTNKPH&name=original_pdf">Behavior Regularized Offline Reinforcement Learning, Arxiv  2019</a>

代码链接：<a href="https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl">github链接</a>


## 一、问题

Offline RL中常用的两类方法为：
  * 使用target Q-value的ensemble来减少预估误差以稳定Q的学习
  * 考虑到未见过的state-action对更可能产生过估计的Q值，因此将学习到的策略向行为策略进行规范和约束
但是现有方法中的这些设计对于Offline RL的学习效果的影响是未知的，本文着眼于上述第二种情况，设计了一个框架behavior regularized actor critic (BRAC)，将现有的一些约束学习策略的工作如BCQ、BEAR等涵盖在内，以对现有方法的不同组件的重要性进行分析。

## 二、解法

### 2.1 约束学习策略

对一个策略进行regularization的方法主要有两种：
  * value penalty (vp)：在值函数中加入惩罚项
  * policy regularization (pr)：在策略中加入正则项

value penalty的方式如下。定义penalized value function：

$$V_{D}^{\pi}(s)=\sum_{t=0}^{\infty} \gamma^{t} \mathbb{E}_{s_{t} \sim P_{t}^{\pi}(s)}\left[R^{\pi}\left(s_{t}\right)-\alpha D\left(\pi\left(\cdot \mid s_{t}\right), \pi_{b}\left(\cdot \mid s_{t}\right)\right)\right]$$

其中$D$是动作分布之间的散度函数（比如MMD与KL散度）。在actor-critic框架下，Q值的更新目标则为：

$$\min _{Q_{\psi}} \mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathcal{D}}\left[\left(r+\gamma\left(\bar{Q}\left(s^{\prime}, a^{\prime}\right)-\alpha \hat{D}\left(\pi_{\theta}\left(\cdot \mid s^{\prime}\right), \pi_{b}\left(\cdot \mid s^{\prime}\right)\right)\right)-Q_{\psi}(s, a)\right)^{2}\right]$$

其中$\bar{Q}$为target Q function，$\hat{D}$是对散度函数$D$的采样估计。对应的策略的更新目标为：

$$\max _{\pi_{\theta}} \mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathcal{D}}\left[\mathbb{E}_{a^{\prime \prime} \sim \pi_{\theta}(\cdot \mid s)}\left[Q_{\psi}\left(s, a^{\prime \prime}\right)\right]-\alpha \hat{D}\left(\pi_{\theta}(\cdot \mid s), \pi_{b}(\cdot \mid s)\right)\right]$$

可以看到，当$\hat{D}\left(\pi_{\theta}\left(\cdot \mid s^{\prime}\right), \pi_{b}\left(\cdot \mid s^{\prime}\right)\right):=\log \pi\left(a^{\prime} \mid s^{\prime}\right)$时，上述算法框架就是SAC算法。

在policy regularization中，只需令value penalty更新Q时的$\alpha=0$，更新策略时的$\alpha \neq 0$即可。可以看到，当$\hat{D}=\pi_{\theta}$时，上述算法框架类似于A3C算法中的正则项。



### 2.2 散度函数$D$的选择与散度函数采样估计$\hat{D}$的实现

CV中常用的方法是使用image的patches作为正负样本。考虑到RL的特性，使用这种需要设计如何获取pathces的较为复杂的方式会为RL的训练带来更大的难度，因此，CURL使用instance区分而不是patch区分。

类似于图像中的instance discrimination，锚定数据和正样本是同一图像的两种不同数据增强，而负样本则来自其他图像。 CURL采用随机裁剪的数据增强方式，即从原始图片中裁剪出一个随机的方块作为样本。过程如下图所示：

<img src="https://s1.ax1x.com/2020/08/26/dRX77t.png" alt="dRX77t.png" style="zoom:67%;" />

#### 2.2.1 Kernel MMD

BEAR中使用的便是Kernel MMD：

$$\operatorname{MMD}_{k}^{2}\left(\pi(\cdot \mid s), \pi_{b}(\cdot \mid s)\right)=\underset{x, x^{\prime} \sim \pi(\cdot \mid s)}{\mathbb{E}}\left[K\left(x, x^{\prime}\right)\right]-2 \mathbb{E}_{x \sim \pi(\cdot \mid s)}[K(x, y)]+\underset{y \sim \pi_{b}(\cdot \mid s)}{\mathbb{E}}\left[\mathbb{E}_{y, y^{\prime} \sim \pi_{b}(\cdot \mid s)}\left[K\left(y, y^{\prime}\right)\right]\right.$$

其中$K$为核函数。但是由于无法获得产生offline data的行为策略$\pi_{b}$，可以通过offline数据的分布对该行为策略进行近似，并使用该近似行为策略$\hat{\pi}_{b}$来替代行为策略$\pi_{b}$：

$$\hat{\pi}_{b}:=\underset{\hat{\pi}}{\operatorname{argmax}} \mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathcal{D}}[\log \hat{\pi}(a \mid s)]$$

#### 2.2.2 KL Divergence

使用KL散度对行为进行约束如下：

$$D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot \mid s), \pi_{b}(\cdot \mid s)\right)=\mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)}\left[\log \pi_{\theta}(a \mid s)-\log \pi_{b}(a \mid s)\right]$$

但是直接计算KL散度需要同时知道学习策略$\pi_{\theta}$和行为策略$\pi_{b}$。因此，这里使用KL散度的对偶形式进行替代。理论上，任何$f-$散度都有一个对偶形式如下：

$$D_{f}(p, q)=\mathbb{E}_{x \sim p}[f(q(x) / p(x))]=\max _{g: \mathcal{X}_{\mapsto} \operatorname{dom}\left(f^{*}\right)} \mathbb{E}_{x \sim q}[g(x)]-\mathbb{E}_{x \sim p}\left[f^{*}(g(x))\right]$$

其中$f^{*}$是$f$的Fenchel对偶。在这种情况下，不再需要估计行为策略，而是需要学习一个鉴别函数$g$。在KL散度中，$f(x)=-\log x$，$f^{*}(t)=-\log (-t)-1$

#### 2.2.3 Wasserstein Distance

$$W(p, q)=\sup _{g:\|g\|_{L} \leq 1} \mathbb{E}_{x \sim p}[g(x)]-\mathbb{E}_{x \sim q}[g(x)]$$

同样也需要一个鉴别函数$g$

#### 2.2.4 BEAR

BEAR方法使用了基于采样的Kernel MMD和min-max形式的Q值ensemble。除此之外，此外，BEAR自适应地将正则化权重$\alpha$训练为Lagriagian乘数：它将Kernel MMD距离设置为$\epsilon>0$的阈值，如果当前平均散度高于阈值，则增大$\alpha$；如果低于阈值，则减小$\alpha$。

#### 2.2.5 BCQ

BCQ没有使用正则项，即vp和pr里的$\alpha$都为0.

## 三、实验内容 

针对已有方法的不同组件进行了各种实验。

BEAR中动态学习正则化权重的方式没有固定权重好：

<img src="https://s1.ax1x.com/2020/09/15/wyjs8P.png" alt="wyjs8P.png" style="zoom:80%;" />



整体上使用Q-ensemble效果更好：

<img src="https://s1.ax1x.com/2020/09/15/wyjrCt.png" alt="wyjrCt.png" style="zoom: 80%;" />



使用Q-ensemble时选取min Q比mix Q更好：

<img src="https://s1.ax1x.com/2020/09/15/wyj0UA.png" alt="wyj0UA.png" style="zoom:80%;" />



使用各种正则项都可以比行为策略好：

<img src="https://s1.ax1x.com/2020/09/15/wyjB4I.png" alt="wyjB4I.png" style="zoom:80%;" />

在VP上使用KL散度的BRAC框架超越了其他Offline RL方法：

<img src="https://s1.ax1x.com/2020/09/15/wyjwEd.png" alt="wyjwEd.png" style="zoom:80%;" />

## 四、缺点

* 提出的各种正则化方法，结果非常接近，这在某种程度上限制了本文作为RL研究的实用准则（虽然反向说明了使用不同的正则化方式区别不大）
* value penalty与policy regularization两种方法区别很小，且不是作者原创的贡献

## 五、优点

虽然本文因为创新性不足的原因被ICLR 2020拒稿了（初始评分6,6,6），但是还是有一些优点：
  * 基本上为约束行为策略这一类的方法构造了一个完备的算法框架
  * 实验充分对比了不同组件的重要程度，说明了Q-ensemble和动态调整正则化权重的作用则不是特别大
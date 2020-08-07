---
layout:     post
title:      单智能体强化学习算法
subtitle:   TD3：Addressing Function Approximation Error in Actor-Critic Methods
date:       2020-08-07
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:

    - RL
    - RL advanced algorithms
    - Model-Free RL
    - Deterministic Policy Gradients
---
---

论文链接：<a href="https://arxiv.org/pdf/1802.09477.pdf">Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018</a>

## 一、问题

在机器学习中广泛存在着bias和variance之间的矛盾。如下图所示，图中红心代表真实值，蓝点代表模型预测值：

<img src="https://s1.ax1x.com/2020/08/07/afJ0ld.png" alt="afJ0ld.png" style="zoom:67%;" />

从图中我们可以很形象的看到偏差（Bias）衡量的是模型输出值与真实样本之间的差异，也就是说偏差越低准确度就越高；而方差（Variance）则是衡量了模型输出的稳定性，比如很多时候模型在训练集上表现良好而在测试集上表现很差（过拟合），这样的模型往往方差较高。

如下图所示，很多时候低偏差与低方差难以两全，所以很多时候我们期望在其中进行trade off，来获得相对好的性能。

<img src="https://s1.ax1x.com/2020/08/07/afJafe.png" alt="afJafe.png" style="zoom: 80%;" />


对于Value-Based的方法，在Double Q-learning通过使用两个独立的目标值函数来解耦更新和action选择操作，以此防止过估计（over estimation）带来的高偏差（bias）。

在Policy gradient系列方法中同样也存在着累计误差带来的高偏差问题，然而Double DQN的做法在Actor-Critic中效果不是很明显。这是因为在连续动作空间中，策略变化缓慢，current Q与target Q变化不大，所以本文提出了TD3，沿用Double DQN之前的Double Q-learning的思想，使用两个独立的Critic来防止过估计。同时为了防止高方差（variance），又在其基础上提出了clipped Double Q-learning以及Delayed Policy Updates用于均衡。

## 二、解法

### 2.1 Clipped Double Q-learning for Actor-Critic

在上面已经提到过，使用current和target两个不同步的Critic在Actor-Critic方法中因为更新缓慢所以效果并不明显，所以沿用初始的double q-learning的思想，使用两个独立开来的Critic（但也不是完全“独立”，仍然公用同一个经验池），那么两个Critic用于更新的target值可以写作：

$$\begin{array}{l}
y_{1}=r+\gamma Q_{\theta_{2}^{\prime}}\left(s^{\prime}, \pi_{\phi_{1}}\left(s^{\prime}\right)\right) \\
y_{2}=r+\gamma Q_{\theta_{1}^{\prime}}\left(s^{\prime}, \pi_{\phi_{2}}\left(s^{\prime}\right)\right)
\end{array}$$

$Q_{\theta_{1}}$和$Q_{\theta_{2}}$总会有高有低，高的值难免会有过估计的可能，所以文中在其基础上进行了稍微的改变，取两者之间的最小值：

$$y=r+\gamma \min _{i=1,2} Q_{\theta_{i}^{\prime}}\left(s^{\prime}, \pi_{\phi_{1}}\left(s^{\prime}\right)\right)$$

尽管这么做会有可能导致低偏估计，但总比高偏要好许多。注意到公式中只使用了一个actor进行更新，它只与$Q_{\theta_{1}}$相关。在更新的时候$Q_{\theta_{1}}$和$Q_{\theta_{2}}$都是用$y$来进行更新的。

### 2.2 Addressing Variance 

在上面解决了过估计带来的偏差，但还有一个重要的问题需要解决，那就是方差，它会在策略更新时产生带有noisy的梯度，从而降低更新速度，导致performance不佳。

在进行TD更新时的每一步会产生一个小的误差$\delta(s, a)$（这对于近似估计更为明显）：

$$Q_{\theta}(s, a)=r+\gamma \mathbb{E}\left[Q_{\theta}\left(s^{\prime}, a^{\prime}\right)\right]-\delta(s, a)$$

当进行很多次更新之后，误差会被大量累计，最终导致Q值不准确：

$$\begin{array}{l}
Q_{\theta}\left(s_{t}, a_{t}\right)=r_{t}+\gamma \mathbb{E}\left[Q_{\theta}\left(s_{t+1}, a_{t+1}\right)\right]-\delta_{t} \\
=r_{t}+\gamma \mathbb{E}\left[r_{t+1}+\gamma \mathbb{E}\left[Q_{\theta}\left(s_{t+2}, a_{t+2}\right)-\delta_{t+1}\right]\right]-\delta_{t} \\
=\mathbb{E}_{s_{i} \sim p_{\pi}, a_{t} \sim \pi}\left[\sum_{i=t}^{T} \gamma^{i-t}\left(r_{i}-\delta_{i}\right)\right]
\end{array}$$

由此可见，估计的方差与未来reward和未来估计误差的方差成正比，当折扣因子$\gamma$很大时误差累积的速度也会非常快。

#### 2.2.1 Target Networks and Delayed Policy Updates

TD3中使用的第二个技巧就是对Policy进行延时更新，即使用target network。target network与critic并不同步更新，这样一来就可以减少之前我们提到的累计误差，从而降低方差。

具体的做法为以较高的频率更新价值函数，以较低的频率更新policy。因为actor-critic方法中参数更新缓慢，进行延时更新一方面可以减少不必要的重复更新，另一方面也可以减少在多次更新中累积的误差。

#### 2.2.2 Target Policy Smoothing Regularization

上面我们通过延时更新policy来避免误差被过分累积，接下来我们我们再思考能不能把误差本身变小呢？那么我们首先就要弄清楚误差的来源。

误差的根源是值函数估计产生的偏差。知道了原因我们就可以去解决它，在机器学习中消除估计的偏差的常用方法就是对参数更新进行正则化，同样的，我们也可以将这种方法引入强化学习中来：

在强化学习中一个很自然的想法就是：**对于相似的action，他们应该有着相似的value。**

所以我们希望能够对action空间中target action周围的一小片区域的值能够更加平滑，从而减少误差的产生。paper中的做法是对target action的Q值加入一定的噪声$\sigma$：

$$\begin{array}{l}
y=r+\gamma Q_{\theta^{\prime}}\left(s^{\prime}, \pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon\right) \\
\epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma),-c, c)
\end{array}$$

这里的噪声可以看作是一种正则化方式，这使得值函数更新更加平滑。

### 2.3 算法

## <img src="https://s1.ax1x.com/2020/08/07/afJB6A.png" alt="afJB6A.png" style="zoom:80%;" />



## 三、实验内容

在Mujoco环境上超越了多个baselines：
![afJwSH.png](https://s1.ax1x.com/2020/08/07/afJwSH.png)



## 四、缺点

暂无评价。

## 五、优点

总结来说TD3中一共使用了三个技巧来消除AC方法中的偏差问题：

- Clipped Double-Q Learning：使用两个Q函数进行学习，并在更新参数时使用其中最小的一个来避免value的过高估计。
- Delayed Policy Updates：对Target以及policy都进行延时更新，避免更新过程中的累积误差。
- Target Policy Smoothing：对target action增加噪音，对Q函数进行平滑操作，减少policy的误差。
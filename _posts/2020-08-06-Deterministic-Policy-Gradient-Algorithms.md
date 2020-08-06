---
layout:     post
title:      单智能体强化学习算法
subtitle:   DPG：Deterministic Policy Gradient Algorithms
date:       2020-08-06
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

论文链接：<a href="http://proceedings.mlr.press/v32/silver14.pdf">Deterministic Policy Gradient Algorithms, ICML 2014</a>

## 一、问题

Stochastic Policy Gradient (SPG) 是通过参数化的概率分布$\pi_{\theta}(a \| s)=\mathbb{P}[a \| s ; \theta]$，随机地选择动作，即$\pi_{\theta}(a \| s)$是一个动作的概率分布。本文提出的Deterministic Policy Gradient (DPG)与SPG不同之处是，这个方法会确定地选择一个动作：$a=\mu_{\theta}(s)$。DPG实际上是SPG在策略方差趋近于0的极限情况。相比SPG每次需要收集动作空间和状态空间，DPG只需收集更少的数据。但是由于DPG每次只探索一个动作，为了更多地探索，应该使用off-policy（离线）方法。

所以这篇论文旨在推导出基于DPG的离线actor-critic算法。

## 二、解法

### 2.1 背景

#### 2.1.1 问题模型 

  * MDP（马尔可夫决策过程）：$\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$
  * 奖励函数reward function）：$r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$
  * 策略（policy）：$\pi_{\theta}: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})$
  * 轨迹（trajectory）：$s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{1}, s_{3}, a_{3},, r_{3} \ldots, s_{T}, a_{T}, r_{T}$
  * 回报（return）：$r_{t}^{\gamma}=\sum_{k=t}^{\infty} \gamma^{k-t} r\left(s_{k}, a_{k}\right)$
  * 状态价值函数（state-value function）： $V^{\pi}(s)=\mathbb{E}\left[r_{1}^{\gamma} \| S_{1}=s ; \pi\right]$
  * 动作价值函数（action-value function）：$Q^{\pi}(s, a)=\mathbb{E}\left[r_{1}^{\gamma} \| S_{1}=s, A_{1}=a ; \pi\right]$
  * 过渡后的概率分布（density at state $s'$ after transitioning from state $s$）：$p\left(s \rightarrow s^{\prime}, t, \pi\right)$
  * 初始状态分布（an initial state distribution with density）：$p_{1}\left(s_{1}\right)$
  * 折扣状态分布（discounted state distribution）：$\rho^{\pi}\left(s^{\prime}\right):=\int_{\mathcal{S}} \sum_{t=1}^{\infty} \gamma^{t-1} p_{1}(s) p\left(s \rightarrow s^{\prime}, t, \pi\right) \mathrm{d} s$
  * 表现目标方程（performance objective）：$\begin{aligned} J\left(\pi_{\theta}\right) =\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \pi_{\theta}(s, a) r(s, a) \mathrm{d} a \mathrm{d} s =\mathbb{E}\_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[r(s, a)] \end{aligned}$

#### 2.1.2 Stochastic Policy Gradient理论

根据policy gradient理论(Sutton et al., 1999)，最基本的policy gradient计算方程为如下：

$$\begin{aligned}
\nabla J\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \nabla \pi_{\theta}(s, a) Q^{\pi}(s, a) \mathrm{d} a \mathrm{d} s \\
&=\mathbb{E}\_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a | s) Q^{\pi}(s, a)\right]
\end{aligned}$$

该定理将梯度转换为了期望的形式，使得求解变得简单。该变化成立的原因是$E_{x \sim P}[f(x)]=\int p(x) f(x) d x$利用这个方程，可以通过调整参数$\theta$来调整策略以优化表现。但是问题在于还需要估计$Q^{\pi}(s, a)$的值。

#### 2.1.3 Stochastic Actor-Critic 算法

Actor-critic是基于policy gradient理论的常用框架。Actor会通过计算上述方程来调整策略函数$\pi_{\theta}(s)$的参数$\theta$。由于真实的动作价值函数$Q^{\pi}(s, a)$是未知的，我们使用一个参数化的函数近似器$Q^{w}(s, a) \approx Q^{\pi}(s, a)$来代替，并且让critic使用策略评估算法来估量它。

由于$Q^{w}(s, a)$是近似，使用它会造成与真实值的偏差。为了消除偏差，$Q^{w}(s, a)$ 应满足两个条件：
  - $Q^{w}(s, a)=\nabla_{\theta} \log \pi_{\theta}(a \| s)^{\top} w$
  - 参数$w$应该最小化均方误差$\epsilon^{2}(w)=\mathbb{E}\_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\left(Q^{\pi}(s, a)-Q^{w}(s, a)\right)^{2}\right]$
  
则表现目标则变为了：
$$\nabla_{\theta} J\left(\pi_{\theta}\right)=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi e}\left[\nabla_{\theta} \log \pi_{\theta}(a | s) Q^{w}(s, a)\right]$$

#### 2.1.4 Off-Policy Actor-Critic

我们称已探索过的轨迹为已知策略（behaviour policy）$\beta(a \| s) \neq \pi_{\theta}(a \| s)$，未探索过的为目标策略（target policy） $\pi_{\theta}(a \| s)$。在 off-policy 中表现目标方程定义为：目标策略的状态价值函数在已知策略上的状态分布上取平均：

$$\begin{aligned}
J_{\beta}\left(\pi_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\beta} V^{\pi}(s) \mathrm{d} s \\
&=\int_{\mathcal{S}} \int_{\mathcal{A}} \rho^{\beta} \pi_{\theta} Q^{\pi}(s, a) \mathrm{d} a \mathrm{d} s
\end{aligned}$$

对其进行微分并进行近似即得到 off-policy policy-gradient：

$$\begin{aligned}
\nabla_{\theta} J_{\beta}\left(\pi_{\theta}\right) & \approx \int_{\mathcal{S}} \int_{\mathcal{A}} \rho^{\beta} \nabla_{\theta} \pi_{\theta}(a \| s) Q^{\pi}(s, a) \mathrm{d} a d s \\
&=\mathbb{E}\_{s \sim \rho^{\beta}, a \sim \beta}\left[\frac{\pi_{\theta}(a \| s)}{\beta_{\theta}(a \| s)} \nabla_{\theta} \log \pi_{\theta}(a \| s) Q^{\pi}(s, a)\right]
\end{aligned}$$

这个近似式去掉了一个依赖动作价值梯度的项$\nabla_{\theta} Q^{\pi}(s, a)$，这是一个很好的近似，因为它可以保留梯度上升收敛到的局部最优集。

### 2.2 确定性策略的梯度

#### 2.2.1 Action-Value Gradients

**大部分的model-free的RL算法本质上是基于generalised policy iteration的。**在policy evaluation阶段评估动作值函数$Q^{\pi}(s, a)$或者$Q^{\mu}(s, a)$，在policy improvement阶段根据动作值函数更新策略。通常，policy improvement使用的方法是贪心选择全局最大化动作值函数的动作。但是在连续动作空间里这样做并不可行。一种替代的方法是将策略朝$Q$的梯度方向移动，而不是使$Q$全局最大化。对于每一个state策略参数$\theta^{k+1}$都按照梯度$\nabla_{\theta} Q^{\mu^{k}}\left(s, \mu_{\theta}(s)\right)$正比例更新，因此对于整个state的分布可以用均值来进行参数的更新：

$$\theta^{k+1}=\theta^{k}+\alpha \mathbb{E}\_{s \sim \rho^{\mu} k}\left[\nabla_{\theta} Q^{\mu^{k}}\left(s, \mu_{\theta}(s)\right)\right]$$

使用链式法则，可以进一步得到：

$$\theta^{k+1}=\theta^{k}+\alpha \mathbb{E}\_{s \sim \rho^{\mu}}^{k}\left[\left.\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu^{k}}(s, a)\right|_{a=\mu \theta(s)}\right]$$

这样的梯度存在的问题是，随着策略的变化，状态分布也会变，按照这种方法进行policy improvement并不能看出来是否有效。

下面的理论表明，就像SPG一样，其实不需要计算state分布的梯度, 上面的直观描述是符合表现目标的梯度的。

#### 2.2.2 Deterministic Policy Gradient定理

Deterministic policy $\mu_{\theta}: \mathcal{S} \rightarrow \mathcal{A}, \theta \in \mathbb{R}^{n}$。和在SPG里讨论的类似，我们可以得到概率分布$p\left(s \rightarrow s^{\prime}, t, \mu\right)$，折扣状态分布$\rho^{\mu}$以及目标方程：

$$\begin{aligned}
J\left(\mu_{\theta}\right) &=\mathbb{E}\left[r_{1}^{\gamma} | \mu\right] \\
&=\int_{\mathcal{S}} \rho^{\mu} r\left(s, \mu_{\theta}(s)\right) \mathrm{d} s \\
&=\mathbb{E}\_{s \sim \rho^{\mu}}\left[r\left(s, \mu_{\theta}(s)\right)\right]
\end{aligned}$$

那么deterministic policy gradient也可由此推出：

$$\begin{aligned}
\nabla_{\theta} J\left(\mu^{\theta}\right) &=\left.\int_{\mathcal{S}} \rho^{\mu}(s) \nabla_{\theta} \mu_{\theta}(s) \nabla_{\theta} Q^{\mu}(s, a)\right|_{a=\mu_{\theta}(s)} \mathrm{d} s \\
&=\mathbb{E}\_{s \sim \rho^{\mu}}\left[\left.\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}(s, a)\right|_{a=\mu_{\theta}}\right]
\end{aligned}$$



证明如下：

> 如果MDP满足$p\left(s^{\prime} \| s, a\right), \quad \nabla_{a} p\left(s^{\prime} \| s, a\right), \quad \mu_{\theta}(s), \quad \nabla_{\theta} \mu_{\theta}(s), \quad r(s, a), \nabla_{a} r(s, a), \quad p_{1}(s)$，在参数$s, a, s^{\prime}, x$下都是连续的，那么$V^{\mu_{\theta}}(s)$和$\nabla_{\theta} V^{\mu_{\theta}}(s)$对于$\theta$和$s$都是连续的。对于任意$\theta$，状态空间$\mathcal{S}$都是紧凑的，因此$\left\|\nabla_{\theta} V^{\mu_{\theta}}(s)\right\|, \quad\left\|\nabla_{a} Q^{\mu_{\theta}}(s, a) \| a=\mu \theta(s)\right\|, \quad \nabla_{\theta} \mu_{\theta}(s)$都是$s$的有界函数。
>
> $$\begin{aligned}
> \nabla_{\theta} V^{\mu_{\theta}}(s)=& \nabla_{\theta} Q^{\mu_{\theta}}\left(s, \mu_{\theta}(s)\right) \\
> =& \nabla_{\theta}\left(r\left(s, \mu_{\theta}(s)\right)+\int_{S} \gamma p\left(s^{\prime} | s, \mu_{\theta}(s)\right) V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime}\right) \\
> =& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} r(s, a)\left|a=\mu \theta(s)+\nabla_{\theta} \int_{S} \gamma p\left(s^{\prime} | s, \mu_{\theta}(s)\right) V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime}\right.\\
> =& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} r(s, a) | a=\mu \theta(s) \\
> &+\int_{S} \gamma\left(p\left(s^{\prime} | s, \mu_{\theta}(s)\right) \nabla_{\theta} V^{\mu_{\theta}}\left(s^{\prime}\right)+\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} p\left(s^{\prime} | s, a\right) | a=\mu \theta(s) V^{\mu_{\theta}}\left(s^{\prime}\right)\right) d s^{\prime} \\
> =& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a}\left(r(s, a)+\int_{S} \gamma p\left(s^{\prime} | s, a\right) V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime}\right) | a=\mu \theta(s) \\
> &+\int_{S} \gamma p\left(s^{\prime} | s, \mu_{\theta}(s)\right) \nabla_{\theta} V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime} \\
> =& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu_{\theta}}(s, a) | a=\mu \theta(s)+\int_{S} \gamma p\left(s \rightarrow s^{\prime}, 1, \mu_{\theta}\right) \nabla_{\theta} V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime}
> \end{aligned}$$
>
> 迭代这一过程消去$V^{\mu_{\theta}}$：
>
> $$\begin{aligned}
> \nabla_{\theta} V^{\mu_{\theta}}(s)=& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu_{\theta}}(s, a) | a=\mu \theta(s)+\int_{\mathcal{S}} \gamma p\left(s \rightarrow s^{\prime}, 1, \mu_{\theta}\right) \nabla_{\theta} V^{\mu_{\theta}}\left(s^{\prime}\right) d s^{\prime} \\
> =& \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu_{\theta}}(s, a) | a=\mu \theta(s) \\
> &+\int_{S} \gamma p\left(s \rightarrow s^{\prime}, 1, \mu_{\theta}\right) \nabla_{\theta} \mu_{\theta}\left(s^{\prime}\right) Q^{\mu_{\theta}}\left(s^{\prime}, a\right) d s^{\prime} \\
> &+\int_{S} \gamma^{2} p\left(s \rightarrow s^{\prime}, 2, \mu_{\theta}\right) \nabla_{\theta} \mu_{\theta}\left(s^{\prime}\right) Q^{\mu_{\theta}}\left(s^{\prime}, a\right) d s^{\prime} \\
> &+\cdots \\
> =& \int_{\mathcal{S}} \sum_{t=0}^{\infty} \gamma^{t} p\left(s \rightarrow s^{\prime}, t, \mu_{\theta}\right) \nabla_{\theta} \mu_{\theta}\left(s^{\prime}\right) \nabla_{a} Q^{\mu_{\theta}}\left(s^{\prime}, a\right) | a=\mu \theta\left(s^{\prime}\right) d s^{\prime}
> \end{aligned}$$
>
> 因此我们有：
>
> $$\begin{aligned}
> \nabla \theta J\left(\mu_{\theta}\right) &=\nabla_{\theta} \int_{S} p_{1}(s) V^{\mu_{\theta}}(s) d s \\
> &=\int_{S} p_{1}(s) \nabla_{\theta} V^{\mu_{\theta}}(s) d s \\
> &=\int_{S} \int_{\mathcal{S}} \sum_{t=0}^{\infty} \gamma^{t} p_{1}(s) p\left(s \rightarrow s^{\prime}, t, \mu_{\theta}\right) \nabla_{\theta} \mu_{\theta}\left(s^{\prime}\right) \nabla_{a} Q^{\mu_{\theta}}\left(s^{\prime}, a\right) | a=\mu \theta\left(s^{\prime}\right) d s^{\prime} d s \\
> &=\int_{S} \rho^{\mu_{\theta}}(s) \nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu_{\theta}}(s, a) | a=\mu \theta(s) d s
> \end{aligned}$$



本质上，DPG确实是SPG的一种特殊（极限）情况：

$$\lim _{\sigma \downarrow 0} \nabla_{\theta} J\left(\pi_{\mu_{\theta}, \sigma}\right)=\nabla_{\theta} J\left(\mu_{\theta}\right)$$

### 2.3 确定性AC算法

与随机Actor-Critic算法类似，用一个可导的动作-值函数$Q^{w}(s, a)$来估计$Q^{\mu}(s, a)$

#### 2.3.1 On-Policy 确定性AC

对于On-Policy AC，critic使用Sarsa来估计动作-值函数，算法为：

$$\begin{aligned}
\delta_{t} &=r_{t}+\gamma Q^{w}\left(s_{t+1}, a_{t+1}\right)-Q^{w}\left(s_{t}, a_{t}\right) \\
w_{t+1} &=w_{t}+\alpha_{w} \delta_{t} \nabla_{w} Q^{w}\left(s_{t}, a_{t}\right) \\
\theta_{t+1} &=\theta_{t}+\left.\alpha_{\theta} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right) \nabla_{a} Q^{w}\left(s_{t}, a_{t}\right)\right|_{a=\mu_{\theta}(s)}
\end{aligned}$$

其中，On-Policy的确定性策略梯度为：

$$\nabla_{\theta} J\left(\mu_{\theta}\right)=E_{s \sim \rho^{\mu_{\theta}}}\left[\nabla_{\theta} \mu_{\theta}(s) \nabla_{\mu} Q^{\omega}(s, \mu)\right]$$

与SPG相比，区别如下：
  * $\pi_{\theta}$变成了$\mu_{\theta}$
  * 原来的$Q^{\pi}(s, a)$改成了$\left.Q^{\mu}(s, a)\right\|\_{a=\mu_{\theta}(s)}$
  * 原来的$s \sim \rho^{\pi}$变成了$s \sim \rho^{\mu}$
  * 去掉了对于动作的采样$a \sim \pi_{\theta}$，而改成确定性的动作$a=\mu_{\theta}(s)$
  * 原来对$\pi$的梯度，即$\nabla_{\theta} \log \pi_{\theta}(a \| s)$改成了对$\mu$的梯度$\nabla_{\theta} \mu_{\theta}(s)$
  * 对于$Q$也要求一次关于$a$的梯度，即：$\left.\nabla_{a} Q^{\mu}(s, a)\right\|\_{a=\mu_{\theta}(s)}$，即回报函数对动作的导数

#### 2.3.2 Off-Policy 确定性AC

对于off-policy来说，在生成样本轨迹时所用的策略可以使任意的随机行为策略$\beta(s, a)$，目标函数$J$为：

$$\begin{aligned}
J_{\beta}\left(\mu_{\theta}\right) &=\int_{\mathcal{S}} \rho^{\beta}(s) V^{\mu}(s) \mathrm{d} s \\
&=\int_{\mathcal{S}} \rho^{\beta}(s) Q^{\mu}\left(s, \mu_{\theta}(s)\right) \mathrm{d} s
\end{aligned}$$

梯度为：

$$\begin{aligned}
\nabla_{\theta} J_{\beta}\left(\mu_{\theta}\right) & \approx \int_{\mathcal{S}} \rho^{\beta}(s) \nabla_{\theta} \mu_{\theta}(a | s) Q^{\mu}(s, a) \mathrm{d} s \\
&=\mathbb{E}\_{s \sim \rho^{\beta}}\left[\left.\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}(s, a)\right|_{a=\mu_{\theta}(s)}\right]
\end{aligned}$$

与off policy的SPG进行对比，可以发现少了重要性权重，即$\frac{\pi \theta(a \| s)}{\beta \theta(a \| s)}$。因为重要性采样是用简单的概率分布估计复杂的概率分布，而确定性策略的动作是确定值；critic采用Q-learing的学习策略来估计动作-值函数，也就是用TD(0)估计动作值函数，并且忽略重要性权重：

$$\begin{aligned}
\delta_{t} &=r_{t}+\gamma Q^{w}\left(s_{t+1}, \mu_{\theta}\left(s_{t+1}\right)\right)-Q^{w}\left(s_{t}, a_{t}\right) \\
w_{t+1} &=w_{t}+\alpha_{w} \delta_{t} \nabla_{w} Q^{w}\left(s_{t}, a_{t}\right) \\
\theta_{t+1} &=\theta_{t}+\left.\alpha_{\theta} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right) \nabla_{a} Q^{w}\left(s_{t}, a_{t}\right)\right|_{a=\mu_{\theta}(s)}
\end{aligned}$$

前两行是利用值函数逼近的方法更新值函数参数$w$，使用的是TD，用Q-learning。第三行是用确定性策略梯度方法更新策略参数$\theta$

可以看出on policy和off policy的不同之处在于对$a_{t}$的生成，on policy用的是确定性策略，off policy则用的是一个任意的随机策略。on policy是选择状态$s_{t+1}$最大的$Q$，而off policy是选择状态$s_{t+1}$和动作$\mu\left(s_{t+1}\right)$的$Q$。

### 2.4 Compatible Function Approximation

这部分重点关于为什么可以用一个可微的函数近似器$Q^{w}(s, a)$来代替真实的动作价值函数$Q^{\mu}(s, a)$。和前面“Stochastic Actor-Critic 算法”部分类似，如果近似器$Q^{w}(s, a)$与$\mu_{\theta}(s)$和是兼容的，应满足如下两个条件：

  - $\left.\nabla_{a} Q^{w}(s, a)\right\|\_{q=\mu_{\theta}(s)}=\nabla_{\theta} \mu_{\theta}(s)^{\top} w$
  - $w$最小化均方误差$\operatorname{MSE}(\theta, w)=\mathbb{E}\left[\epsilon(s ; \theta, w)^{\top} \epsilon(s ; \theta, w)\right]$，其中$\epsilon(s ; \theta, w)=\left.\nabla_{a} Q^{w}(s, a)\right\|\_{a=\mu_{\theta}(s)}-\left.\nabla_{a} Q^{\mu}(s, a)\right\|\_{a=\mu_{\theta}(s)}$

一个 compatible off-policy deterministic actor-critic (COPDAC) 算法有两部分组成：Critic是一个线性函数近似器，通过特征$\phi(s, a)=a^{\top} \nabla_{\theta} \mu_{\theta}(s)$来评估动作价值函数。这个可以通过离线地学习已知策略的样本得到，比如用Q-learning 或者 gradient Q-learning。Actor则朝着 critic 的动作价值函数的梯度方向更新参数。例如，下列COPDAC-Q 算法中 critic 使用了简单Q-learning：

$$\begin{aligned} \delta_{t} &=r_{t}+\gamma Q^{w}\left(s_{t+1}, \mu_{\theta}\left(s_{t+1}\right)\right)-Q^{w}\left(s_{t}, a_{t}\right) \\ \theta_{t+1} &=\theta_{t}+\alpha_{\theta} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right)\left(\nabla_{\theta} \mu_{\theta}\left(s_{t}\right)^{\top} w_{t}\right) \\ w_{t+1} &=w_{t}+\alpha_{w} \delta_{t} \phi\left(s_{t}, a_{t}\right) \\ v_{t+1} &=v_{t}+\alpha_{t} \delta_{t} \phi\left(s_{t}\right) \end{aligned}$$

或者gradient Q-learning critic：

$$\begin{aligned}
\delta_{t} &=r_{t}+\gamma Q^{w}\left(s_{t+1}, \mu_{\theta}\left(s_{t+1}\right)\right)-Q^{w}\left(s_{t}, a_{t}\right) \\
\theta_{t+1} &=\theta_{t}+\alpha_{\theta} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right)\left(\nabla_{\theta} \mu_{\theta}\left(s_{t}\right)^{\top} w_{t}\right) \\
w_{t+1} &=w_{t}+\alpha_{w} \delta_{t} \phi\left(s_{t}, a_{t}\right)-\alpha_{w} \gamma \phi\left(s_{t+1}, \mu_{\theta}\left(s_{t+1}\right)\right)\left(\phi\left(s_{t}, a_{t}\right)^{\top} u_{t}\right) \\
v_{t+1} &=v_{t}+\alpha_{t} \delta_{t} \phi\left(s_{t}\right)-\alpha_{v} \gamma \phi\left(s_{t+1}\right)\left(\phi\left(s_{t}, a_{t}\right)^{\top} u_{t}\right) \\
u_{t+1} &=u_{t}+\alpha_{u}\left(\delta_{t}-\phi\left(s_{t}, a_{t}\right)^{\top} u_{t}\right) \phi\left(s_{t}, a_{t}\right)
\end{aligned}$$

## 三、实验内容

红色是本文结果：DAC off-policy > SAC off-policy > SAC on-policy
![img](https://s1.ax1x.com/2020/08/06/agFlG9.png)

## 四、缺点

暂无评价。

## 五、优点

确定性策略的优点在于：需要采样的数据少，算法效率高。确定性策略的动作是确定的，所以，如果存在确定性策略梯度，其求解不需要在动作空间采样，所以需要的样本数更少。对于动作空间很大的智能体（如多关节机器人），动作空间维数很大，有优势。

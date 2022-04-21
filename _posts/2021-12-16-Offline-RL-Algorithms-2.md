---
layout:     post
title:      离线强化学习（二）
subtitle:   policy constraint类方法简介
date:       2021-12-16
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - Offline RL
typora-root-url: ..\post_pic
typora-copy-images-to: ..\post_pic
---
---

# 离线强化学习（二）

本文为离线强化学习系列介绍的第二部分，包括部分policy constraint类方法简介。本系列持续更新，欢迎大家关注交流讨论~

本系列所有链接如下：

* [离线强化学习（一）：离线强化学习与部分policy constraint类方法简介](https://zhuanlan.zhihu.com/p/414497708)

* [离线强化学习（二）：部分policy constraint类方法简介](https://zhuanlan.zhihu.com/p/414497708)



## 一、Policy Constraint Methods

### 1.1 ABM: Keep Doing What Worked: Behavior Modelling Priors for Offline Reinforcement Learning

类似BCQ和BEAR这类方法，往往需要对产生离线数据的行为策略进行建模。这类方法通常会采用VAE等生成模型来重建行为策略，然而，对于一些随机策略或者带有噪声的策略所产生的数据，它们的行为很难被精确地预测出来。除此之外，当离线数据来自于多个不同的策略时，重建行为策略的难度也会增大。而行为策略重建的不精确性，会导致BCQ和BEAR这类通过最小化学习策略与行为策略之间差异作为约束的离线强化学习算法产生较差的表现。

针对这个问题，ICLR 2020的这篇论文ABM[1]便提出了一种叫做Advantage-weighted Behavior Model（ABM）的算法。ABM算法遵循Policy Iteration的流程：在Policy Evaluation阶段，对状态-动作值$Q^{\pi_{i}}(s, a) \approx \hat{Q}\left(s, a ; \phi_{i}\right)$进行评估；在Policy Improvement阶段，优化$\pi_{i+1}$时保证它与离线数据的状态-动作分布尽可能接近。

具体来说，Policy Evaluation时，ABM最小化TD error的均方：
$$
\begin{array}{c}
\phi_{i}=\arg \min _{\phi} \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\left(r\left(s_{t}, a_{t}\right)+\gamma \hat{V}^{\pi_{i}}\left(s_{t+1}\right)-\hat{Q}\left(s_{t}, a_{t} ; \phi\right)\right)^{2} \mid\left(s_{t}, a_{t}, s_{t+1}\right) \sim \tau\right] \\
\operatorname{with} \hat{V}^{\pi_{i}}(s)=\mathbb{E}_{a \sim \pi_{i}(\cdot \mid s)}\left[\hat{Q}\left(s, a ; \phi_{i-1}\right)\right] \approx \frac{1}{M}\left[\sum_{j=1}^{M} \hat{Q}\left(s, a_{j} ; \phi_{i-1}\right) \mid a_{j} \sim \pi_{i}(\cdot \mid s)\right]
\end{array}
$$
与BCQ不同的是，ABM提出当抽取足够的样本（$M = 20$）并且适当地约束该策略时，学习过程是稳定的，无需从采样出的多个动作中选取Q值最大的。

在Policy Improvement阶段求解如下带约束的优化问题：
$$
\begin{aligned}
\pi_{i+1}=\arg \max _{\pi} & \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\mathbb{E}_{a \sim \pi(\cdot \mid s)}\left[\hat{Q}^{\pi_{i}}(s, a)\right] \mid s \sim \tau\right] \\
\text { s.t. } & \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\mathrm{KL}\left[\pi(\cdot \mid s) \| \pi_{\text {prior }}(\cdot \mid s)\right] \mid s \sim \tau\right] \leq \epsilon
\end{aligned}
$$
其中$\mathcal{D}_{\mu}$是离线数据，$\pi_{\text {prior }}$是行为策略模型，即用于拟合产生离线数据的策略。$\pi_{\text {prior }}$有两种学习方式：

（1）直接从离线数据中学习。这种方法就类似于BCQ与BEAR等方法，根据离线数据通过最大似然估计来学习行为策略的模型，只不过本文衡量学习策略与行为策略差异用的是KL散度：
$$
\theta_{\mathrm{bm}}=\arg \max _{\theta_{\mathrm{bm}}} \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\sum_{t=1}^{|\tau|} \log \pi_{\theta_{\mathrm{bm}}}\left(a_{t} \mid s_{t}\right)\right.
$$
（2）有选择地从离线数据中学习。当离线数据是return比较高的数据时，直接从离线数据中学习是可行的。但是当离线数据是来自于多种任务、多种非完美策略时，离线数据中可能有互相冲突的一些数据，因此直接从离线数据中学习可能学习不到最优水平。针对这个问题，本文的做法是针对不同的轨迹使用不同权重的advantage，从而筛选过滤掉会导致学习策略表现变差的trajectory：
$$
\begin{array}{rl}
\theta_{\mathrm{abm}}=\arg \max _{\theta_{\mathrm{abm}}} & \mathbb{E}_{\mathcal{T} \sim \mathcal{D}_{\mu}}\left[\sum_{t=1}^{|\tau|} \log \pi_{\theta_{\mathrm{abm}}}\left(a_{t} \mid s_{t}\right) f\left(R\left(\tau_{t: N}\right)-\hat{V}^{\pi_{i}}(s)\right)\right] \\
& \operatorname{with} R\left(\tau_{t: N}\right)=\gamma^{N-t} \hat{V}^{\pi_{i}}\left(s_{N}\right)+\sum_{j=t}^{N-1} \gamma^{j-t} r\left(s_{j}, a_{j}\right)
\end{array}
$$
其中$f$是一个非负的递增函数，用于赋予不同的advantage以不同的权重；$R\left(\tau_{t: N}\right)-\hat{V}^{\pi_{i}}$是类似于n-step advantage的函数。本文的$f=1_{+}$，即当$x \ge 0$时，$f(x)=1$，否则为0。这种方法比较简单，超参数少，且在与其他函数的消融实验对比中没有什么明显区别。



上述两种行为策略模型的优化都是带约束的优化，针对带约束的优化，存在如下两种方式：

（1）EM-style optimization。在这种方式下，最优策略可以表示为$\hat{\pi}(a \mid s) \propto \pi_{\text {prior }}(a \mid s) \exp \left(\hat{Q}^{\pi} i(s, a) / \eta\right)$，其中$\eta$是温度系数。在这基础上，最小化KL散度的优化等价于最大化如下的公式：
$$
\theta_{i+1}=\arg \max _{\theta} \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\mathbb{E}_{a \sim \pi_{\text {pior }}(\cdot \mid s)}\left[\exp \left(\hat{Q}^{\pi_{i}}(s, a) / \eta\right) \log \pi_{\theta}(a \mid s) \mid s \sim \tau\right]\right]
$$
（2） Stochastic value gradient optimization。另一种常见的方式就是通过拉格朗日乘子法将约束也放入优化目标中，即：
$$
\left(\theta_{i+1}, \eta\right)=\arg \max _{\theta} \min _{\eta} \underset{\tau \sim \mathcal{D}_{\mu}}{\mathbb{E}}\left[\mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)}\left[\hat{Q}^{\pi_{i}}(s, a)\right]+\eta\left(\epsilon-\mathrm{KL}\left[\pi_{\theta}(\cdot \mid s) \| \pi_{\text {prior }}(\cdot \mid s)\right]\right)\right]
$$
ABM整体算法如下图所示：

<img src="/TPYfZ6.png" width="90%" height="60%" align=center />

<div align = "center">图1 ABM算法流程</div>

最终，ABM在多个随机种子下使用行为策略收集离线数据的DeepMind control suite环境与轮流产生不同任务的离线数据的SIMULATED ROBOT EXPERIMENTS环境中，表现超越了BCQ和BEAR。

据我了解，ABM这篇工作首次考虑了离线数据中存在多种分布的问题，并提供了一种直觉上非常合理的挑选优先程度高的分布数据进行学习的方法：本文设计的prior在最开始的时候覆盖完整的离线数据进行学习，并且随着时间的流逝，将滤除会导致性能不如当前策略的trajectory，直到最终收敛到数据中包含的最佳trajectory为止。遗憾的是，后续并没有工作与其在相同的多模态设置下进行对比，而我认为这种设定在真实世界中可能是非常现实的。比如在广告系统中，往往同时存在多种广告投放策略；自动驾驶中，不同的司机在相同情境下可能也会采取不同的驾驶策略。在这些情况下收集到离线数据自然是多模态的。

### 1.2 BRAC: Behavior Regularized Offline Reinforcement Learning

BCQ、BEAR等算法的做法本质上可以理解为是考虑到未见过的state-action对更可能产生过估计的Q值，因此将学习到的策略向行为策略进行规范和约束。由于不同的约束方法对于学习策略的约束松紧程度不同，自然会对学习策略的效果产生不同的影响。本文着眼于上述第二种情况，设计了一个框架behavior regularized actor critic (BRAC) [2]，将现有的一些约束学习策略的工作如BCQ、BEAR等涵盖在内，并对现有方法中用到的不同组件的重要性进行分析。

对一个策略进行regularization的方法主要有两种。一种是在值函数中加入惩罚项（value penalty, vp），一种是在策略中加入正则项（policy regularization,  pr）。首先，value penalty的方式如下。定义penalized value function：
$$
V_{D}^{\pi}(s)=\sum_{t=0}^{\infty} \gamma^{t} \mathbb{E}_{s_{t} \sim P_{t}^{\pi}(s)}\left[R^{\pi}\left(s_{t}\right)-\alpha D\left(\pi\left(\cdot \mid s_{t}\right), \pi_{b}\left(\cdot \mid s_{t}\right)\right)\right]
$$
其中$D$是动作分布之间的散度函数（比如MMD与KL散度）。那么在actor-critic框架下，Q值的更新目标则为：

$$
\min _{Q_{\psi}} \mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathcal{D}}\left[\left(r+\gamma\left(\bar{Q}\left(s^{\prime}, a^{\prime}\right)-\alpha \hat{D}\left(\pi_{\theta}\left(\cdot \mid s^{\prime}\right), \pi_{b}\left(\cdot \mid s^{\prime}\right)\right)\right)-Q_{\psi}(s, a)\right)^{2}\right]
$$
其中$\bar{Q}$为target Q function，$\hat{D}$是对散度函数$D$的采样估计。对应的策略的更新目标为：

$$
\max _{\pi_{\theta}} \mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim \mathcal{D}}\left[\mathbb{E}_{a^{\prime \prime} \sim \pi_{\theta}(\cdot \mid s)}\left[Q_{\psi}\left(s, a^{\prime \prime}\right)\right]-\alpha \hat{D}\left(\pi_{\theta}(\cdot \mid s), \pi_{b}(\cdot \mid s)\right)\right]
$$
可以看到，当$\hat{D}\left(\pi_{\theta}\left(\cdot \mid s^{\prime}\right), \pi_{b}\left(\cdot \mid s^{\prime}\right)\right):=\log \pi\left(a^{\prime} \mid s^{\prime}\right)$时，上述算法框架就是SAC算法。

在policy regularization中，只需令value penalty更新Q时的$\alpha=0$，更新策略时的$\alpha \neq 0$即可。可以看到，当$\hat{D}=\pi_{\theta}$时，上述算法框架类似于A3C算法中的正则项。

基于这个框架，我们便可以考虑使用不同的散度函数$D$以及其对应的采样估计$\hat{D}$进行实现。文中提出了Kernel MMD、KL Divergence、Wasserstein Distance等方式，同时BCQ和BEAR也可以归结到BRAC的这个框架下面。不过本文的实验结果证明，虽然在vp时使用KL散度的BRAC框架超越了BCQ和BEAR等Offline RL方法，但是采用不同约束条件的vp和pr方法之间没有特别明显的区别。我个人认为，这是由于在此框架下，绝大部分散度函数（除了KL散度）的使用都依赖于对行为策略的重建，而行为策略的好坏相比于约束条件的选择要有着更为显著的影响：

![img](http://localhost:8800/lib/exe/fetch.php?cache=&media=research:reinforcement_learning:single_agent:advanced_algorithms:batch_rl:pasted:20200914-112922.png)

虽然本文因为创新性不足的原因被ICLR 2020拒稿了（初始评分6,6,6），但是还是有不少优点：

  * 基本上为约束行为策略这一类的方法构造了一个比较完备的算法框架
  * 实验充分对比了不同组件的重要程度，其中比较重要的结论是说明了Q-ensemble在策略约束的框架下作用不是特别大。

### 1.3 PLAS: Latent Action Space for Offline Reinforcement Learning

BCQ、BEAR、BRAC等方法的做法是显式地约束策略可选的动作来避免OOD动作引起的外推误差。但是这种约束的设计同时还不能过于严格，否则会退化为对数据集的behaviour cloning。本文提出了一种与之不同的隐式方法PLAS [3]来约束策略：PLAS在动作隐空间中训练策略，以隐式地将策略限制为在数据集的support范围内输出动作。下面这张图用来描述了PLAS的隐式约束方法与BEAR等显式约束方法的区别。

![img](http://localhost:8800/lib/exe/fetch.php?cache=&media=research:reinforcement_learning:single_agent:advanced_algorithms:batch_rl:pasted:20210306-110659.png)

PLAS的具体做法如下。首先针对离线数据集使用CVAE进行行为策略的重建，CVAE的encoder部分输入状态-动作对，输出隐变量，而decoder部分则输入状态-隐变量，输出重建后的动作。端到端训练encoder和decoder之后，得到的decoder可以认为是学到了基于状态从隐空间到动作空间的映射。在使用时，使用一个隐策略来输入状态输出隐动作，再使用先前学好的decoder将动作解码回真实动作。

进一步地，PLAS提出当环境（转移概率和奖励函数）是平滑的并且数据集的质量和多样性比较有限时，OOD的动作有可能会提升表现。因此，PLAS在解码器的输出后面添加一个扰动层，该扰动层输出动作的一个残差值。如果认为学到的Q函数泛化性很强的话，可以让智能体适当地采取加上残差的动作来作为可控范围内的OOD动作，以进一步提高表现。具体使用过程如下图所示。

![img](http://localhost:8800/lib/exe/fetch.php?cache=&media=research:reinforcement_learning:single_agent:advanced_algorithms:batch_rl:pasted:20210306-110930.png)

PLAS在论文中报告的效果在Mujoco的medium-expert、medium-replay等环境上都优于BCQ、BEAR和BRAC。而实验发现，扰动层的重要性取决于数据集和环境。一般来说，允许进行OOD动作通常可以提高随机数据集上训练得到的智能体的性能。中等数据集上训练得到的智能体往往在扰动较小时可以达到更好的性能。而专家数据集上训练得到的智能体往往会因为扰动而降低表现。

总结来说，这种隐式约束的好处是可以自然而然地在构建隐式空间的过程中被满足，而不会影响算法其他部分的优化，也不受行为策略分布的限制。这种隐式约束的做法让我想起了[Decision Transformer](https://zhuanlan.zhihu.com/p/447031311)和[Trajectory Transformer](https://zhuanlan.zhihu.com/p/452082783)，二者对强化学习transitions进行建模并解码出动作与环境进行交互的做法，与PLAS的做法有着异曲同工之妙。

## 参考

1. Noah Y. Siegel, Jost Tobias Springenberg, Felix Berkenkamp, Abbas Abdolmaleki, Michael Neunert, Thomas Lampe, Roland Hafner, Nicolas Heess, Martin A. Riedmiller: Keep Doing What Worked: Behavior Modelling Priors for Offline Reinforcement Learning. ICLR 2020
2. Yifan Wu, George Tucker, Ofir Nachum: Behavior Regularized Offline Reinforcement Learning. CoRR abs/1911.11361 (2019)
3. Wenxuan Zhou, Sujay Bajracharya, David Held: PLAS: Latent Action Space for Offline Reinforcement Learning. CoRL 2020: 1719-1735




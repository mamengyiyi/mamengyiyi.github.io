---
layout:     post
title:      Offline Reinforcement Learning as One Big Sequence Modeling Problem
subtitle:   
date:       2021-12-30
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

# Offline Reinforcement Learning as One Big Sequence Modeling Problem

本文发表于NeurIPS 2021，论文链接详见https://openreview.net/forum?id=wgeK563QgSw

## 一、问题

标准的强化学习框架侧重于将复杂的长期问题分解为更小、更易处理的子问题，从而采用动态规划方法（如 Q -learning）和model-based等方法进行每个子问题的求解和优化。

然而，强化学习这种序列决策问题可以将强化学习视为一种序列生成问题。 本文介绍的这项工作旨在尝试用序列建模替换标准RL的方法，发挥序列模型的强大表征能力，以期可以达到媲美复杂RL算法的目的。

## 二、方法

本文方法的核心是采用 Trajectory Transformer拟合RL的状态、动作和奖励的分布，并使用beam search来对解码出的候选轨迹（策略）进行搜索和优化。

假设轨迹$\tau$由$𝑇$个状态、动作和奖励组成。 在连续状态和动作的情况下，本文的做法是将其每个维度进行离散化。 假设状态有$N$维，动作有$M$维，那么轨迹$\tau$可以变为长度为$ 𝑇(𝑁+𝑀+1)$ 的序列：
$$
\tau=\left(\ldots, \mathbf{s}_{t}^{1}, \mathbf{s}_{t}^{2}, \ldots, \mathbf{s}_{t}^{N}, \mathbf{a}_{t}^{1}, \mathbf{a}_{t}^{2}, \ldots, \mathbf{a}_{t}^{M}, r_{t}, \ldots\right) t=1, \ldots, T
$$
如下图所示，离散化后的状态、动作及奖励的分布将通过Trajectory Transformer进行建模。

<img src="/image-20211231141958386.png" alt="image-20211231141958386" style="zoom: 80%;" />

<div align = "center">图1 Trajectory Transformer整体框架</div>

本文采用了两种方式对状态和动作进行离散化。第一种做法是固定宽度的分箱。虽然固定宽度分箱非常容易计算，但如果统计值中有比较大的缺口，就会产生很多没有任何数据的空箱子。因此，第二种做法便是根据数据的分布特点，使用数据分布的分位数进行自适应的箱体定位，以解决固定宽度分箱的这个缺陷。两种分箱方式效果如下图所示。可以看到，左边的固定宽度分箱方法中存在多个箱体中不存在数据的情况，而右边展示的分位数分箱方法便不存在这个问题。

![image-20211231142549948](/image-20211231142549948.png)

<div align = "center">图2 两种离散化分箱方式</div>

那么很自然的一个疑问便是，为什么需要对状态和动作进行离散化呢？这个原因作者在Rebuttal阶段提到了，当使用具有输出参数化高斯分布的Transformer对状态和动作进行建模时，它的效果并不比单步 MLP 的方法要好。作者分析这是因为输出高斯分布的最大缺点之一是它不允许像离散化方法那样进行多模态预测，导致建模效果较差：

![image-20211231142748342](/image-20211231142748342.png)

<div align = "center">图3 离散化与未离散化的Transformer分布建模效果对比</div>

 本文训练Trajectory Transformer的方法是最大化如下目标：
$$
\mathcal{L}(\tau)=\sum_{t=1}^{T}\left(\sum_{i=1}^{N} \log P_{\theta}\left(\mathbf{s}_{t}^{i} \mid \mathbf{s}_{t}^{<i}, \tau_{<t}\right)+\sum_{j=1}^{M} \log P_{\theta}\left(\mathbf{a}_{t}^{j} \mid \mathbf{a}_{t}^{<j}, \mathbf{s}_{t}, \tau_{<t}\right)+\log P_{\theta}\left(\bar{r}_{t} \mid \mathbf{a}_{t}, \mathbf{s}_{t}, \tau_{<t}\right)\right)
$$
那么在获得训练好的Trajectory Transformer之后，该如何根据不同的RL场景进行使用呢？本文采用了beam search，根据使用用途的不同，通过调整搜索轨迹所使用的目标，引导模型采样多条轨迹，并在多条轨迹中保留目标值为Top $B$的轨迹作为候选。反复进行上述操作，最终选出找到的最优轨迹：

<img src="/image-20211231143228701.png" alt="image-20211231143228701" style="zoom: 80%;" />

<div align = "center">图4 Beam Search流程</div>

具体来说，在每个step $t$，beam search会构建一个候选集$C_{t}$，这个候选集$C_{t}$包括先前hypothesis set中的所有长度为$t-1$的序列与从$V$中选出的单个token的拼接。其中$B$条可能性最高的序列被拿出来更新hypothesis set $Y_{t}$。 论文中设置$B$为256，$V$的大小则为100（该$V$的大小其实就是离散化分箱中箱子的数量）。

针对不同的场景，只需调整输入到模型中的上下文信息即可。比如对于模仿学习，输入到$P_{\theta}(\mathrm{Y} \mid x)$中的$x$就是每一步的状态$s_{t}$；对于goal-conditioned RL，输入便是额外加入目标状态$s_{T}$的$P_{\theta}\left(\mathbf{s}_{t}^{i} \mid \mathbf{s}_{t}^{<i}, \tau_{<t}, \mathbf{s}_{T}\right)$；对于Offline RL，则引导的目标便是最大化累积奖励。注意，如果使用beam search对单步奖励之后最大化，有导致学到短视的贪心策略的风险。 为了解决这个问题，作者将搜索目标设置为 MC 值的估计，即目前为止最高的累积奖励与Reward-to-go之和。

## 三、实验结果

本文首先验证了Trajectory在轨迹预测方面的精准程度。下图为预测100个timestep内的轨迹的结果。可以看到，该方法比model-based方法中最先进的预测方法更好，对更远的轨迹预测更准确：

<img src="/image-20211231144505508.png" alt="image-20211231144505508" style="zoom: 80%;" />

<div align = "center">图5 轨迹预测效果</div>

本文将每个输出的状态、动作与之前轨迹中的每一项的attention权重进行了可视化，以探究Trajectory Transformer对状态和动作的预测与之前轨迹中的哪些状态动作有关。发现了两种比较典型的结果：

- 第一种（下图中的左图）：状态和动作都主要依赖于紧接在前的transition，也就是说在这种模式下Trajectory Transformer学到了马尔可夫特性
- 第二种（下图中的右图）：状态的每个维度强烈地依赖于多个先前时间步的同一维度； 而动作更多地取决于过去的动作而不是过去的状态

<img src="/image-20211231144533540.png" alt="image-20211231144533540" style="zoom:67%;" />

<div align = "center">图6 模型输出与输入的Attention权重</div>

最终在Offline RL和Goal Reaching的任务上，Trajectory Transformer取得了不错的效果：

<img src="/image-20211231145127074.png" alt="image-20211231145127074" style="zoom: 67%;" />

<div align = "center">图7 实验结果</div>

## 四、总结

- 缺点
  - 与基于模型的控制中经常使用的单步模型类型的预测相比，本文的方法预测速度更慢、需要消耗的资源显然要更多。
  - 由于使用最大似然目标来进行Trajectory Transformer的训练，与传统的动态规划算法相比，这种做法更依赖于训练数据的分布。

* 优点
  * 尽管缺乏强化学习算法的大多数常见的设计，但它的性能与复杂的RL算法在某些任务上可以达到相当的结果。
  * 这种可扩展的表征架构在实践中或许可以帮助强化学习解决一些本身解决不了的问题，如对复杂问题进行有效的表征提取，从而帮助强化学习算法更好地发挥自身的优势。








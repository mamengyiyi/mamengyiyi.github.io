---
layout:     post
title:      单智能体强化学习算法
subtitle:   Rainbow：Combining Improvements in Deep Reinforcement Learning
date:       2020-08-05
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - RL advanced algorithms
    - Model-Free RL
    - Deep Q-Learning
---
---

论文链接：<a href="https://arxiv.org/pdf/1710.02298.pdf">Rainbow: Combining Improvements in Deep Reinforcement Learning, AAAI 2018</a>

## 一、问题 
Rainbow主要探究了如下几种算法在不同方面对DQN的改进的作用，并进行了这些改进的组合。

## 二、解法 
### 2.1 DQN
#### 2.1.1 算法简介

强化学习算法的目标是根据当前的状态，选择能够使累计奖惩值最大的动作值给智能体执行。当使用非线性函数逼近器如神经网络拟合Q函数时，智能体的训练会产生不稳定甚至发散的情况，主要原因在于：  

  - 数字列表项目观察序列中存在相关性
  - 对当前Q的小步更新都可能显著影响策略，而今使更新后的策略采集的数据分布和更新前的不一致，进而改变Q值与目标值之间的相关性
  
在DQN中，作者提出两个解决方法：

  - Replay Buffer：强化学习训练时会将智能体的经验根据时间顺序连续保存到经验回放池中$(s,a,r,s')$，会造成相邻两个样本之间存在相关性，解决思路是利用随机均匀采样经验池的中样本数据供智能体训练
  - Fixed Q-Target：DQN使用定期迭代更新的方式更新目标值计算中的拟合网络参数，从而减少了与目标值之间的相关性，如果一直更新，会由于目标值一直改变不”稳定”而造成训练发散

DQN迭代更新的损失函数为：

$$\operatorname{Loss}_{i}\left(\theta_{i}\right)=\mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim U(D)}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i}^{-}\right)-Q\left(s, a ; \theta_{i}\right)\right)^{2}\right]$$

其中：
  * $\gamma$是折扣因子，用来决定目标值对于当前值的影响程度
  * $\theta_{i}$代表第i次迭代过程的神经网络参数
  * $\theta_{i}^{-}$代表目标值函数的神经网络参数
  * 目标值函数的网络参数是每隔$C$次训练才更新，$C$次训练期间保持目标值函数的网络参数固定不变

为了增加训练的稳定性，作者还提出了梯度裁剪(Gradient Clipping)的方法，将$\left(r+\gamma \max_{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta_{i}^{-}\right)-Q\left(s, a ; \theta_{i}\right)\right)$差值限定在了-1和1之间。

#### 2.1.2 代码实现
通常Python实现经验回放池会用到三种数据结构：

  * collections.deque: 底层实现是使用的双向链表，插入时间复杂度是O(1)，但是查询的时间复杂度是O(n)，会严重增加训练过程的时间
  * list：列表底层是数组，存储的是指向不同或者相同大小物体的指针，可以动态增加数据和支持随机访问，比使用deque队列的方式要快
  * numpy.ndarray：访问最快的数据结构，底层是存储的具有相同固定大小的同质数据，可以根据内存的访问局部性原理加快读取速度
  
![1](https://s1.ax1x.com/2020/08/05/astOIA.png)

![1](https://s1.ax1x.com/2020/08/05/astjPI.png)



### 2.2 Double DQN
#### 2.2.1 算法简介
在标准Q-learning和DQN算法中，取最大值的操作都是使用相同的Q值去选择和评估动作的好坏，这样会让智能体更加喜欢选择被过度估计的值，导致最终的价值估计过于乐观。

在DDQN中，通过对两个不同的值函数随机采样经验样本训练并更新其中一个值函数的方式，学习到两套参数集合。$\theta$和$\theta^{\prime}$。在每一次更新的过程中，其中一套参数被用来作为Greedy策略，另一套参数被选择用来决定它的值。

DQN的目标函数：

$$Y_{t}^{Q}=R_{t+1}+\gamma \max _{a} Q\left(S_{t+1}, \arg \max _{a} Q\left(S_{t+1}, a ; \theta_{t}\right) ; \theta_{t}\right)$$

DDQN的目标函数为：

$$Y_{t}^{\text {DoubleQ }}=R_{t+1}+\gamma \max _{a} Q\left(S_{t+1}, \text { arg } \max _{a} Q\left(S_{t+1}, a ; \theta_{t}\right) ; \theta_{t} \prime\right)$$

#### 2.2.2 代码实现

![1](https://s1.ax1x.com/2020/08/05/asNVGq.png)

### 2.3 Prioritized Experience Replay (PER)
#### 2.3.1 算法简介

使用经验回放池需要解决两个问题：

  * 存储哪些动作序列
  * 采样哪些动作序列
  
PER要解决的是第二个问题，即如何最有效地选择经验回放池中的样本数据进行学习，其核心思想是衡量每个动作序列样本的重要性程度。一种可行的方法是利用样本的TD-error来表示，因为这表明了这次样本的“惊喜”大小。该算法会存储经验回放池中每个样本的最后的TD-error值，然后根据最大绝对TD-error进行采样。Q-learning会利用采样到的样本进行训练和根据TD-error更新权重。当新的转换无法计算TD-error时，会将其误差设置为最大，从而保证所有的样本至少被采样一次。

计算TD-error的方式：

  - Greedy TD-Error Prioritization: 最大的缺点是会聚焦于一小部分经验子集，当使用函数逼近的方法时，误差减小的慢，意味着一开始高误差的样本会被频繁重复地采样，这样的欠多样性样本使得训练的智能体很容易就过拟合
  - Stochastic Prioritization：介于纯Greedy Prioritization和均匀随机采样之间:$P(i)=\frac{p_{i}^{\alpha}}{\sum_{k} p_{k}^{\alpha}}$。其中$p_{i}$代表样本$i$的优先级大小，指数$\alpha$代表使用优先级的影响程度，当$\alpha$为0时，代表均匀随机采样。在实际中，会添加一个正整数偏置常量$\epsilon$，$p_i=\|\sigma_i\| + \epsilon$，保证可以对所有的样本进行采样。

根据样本优先级采样的方式会带来偏差，因为其并不是均匀随机采样样本而是根据TD-error作为采样比例。为了修正该误差，作者引入了重要性采样(Importance-sampling, IS)权重：

$$w_{i}=\left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^{\beta}$$

当$\beta = 1$，上面的式子完全补偿非均匀概率$P_(𝑖)$，是的其最终为均匀概率。 通过使用$w_i\sigma_i$而不是$\sigma_i$，可以将这些权重嵌入到Q-learning的更新中。

在典型的强化学习场景中，更新的无偏性质在训练结束时接近收敛是最重要的，因此，我们利用随着时间衰减重要性采样校正量来达到当训练结束，指数$\beta=1$，从而满足无偏条件。

#### 2.3.2 代码实现

PrioritizedReplayBuffer的实现利用了Segment Tree数据结构，它在保存经验样本和采样样本的同时也负责管理样本的Priorities，是非常高效的实现方式。

![1](https://s1.ax1x.com/2020/08/05/asNUsO.png)

相比于DQNAgent，使用PrioritizedReplayBuffer，添加两个参数$\beta$计算权重和$prior_\epsilon$。新样本的默认TD-Error如下:

![1](https://s1.ax1x.com/2020/08/05/asNTWq.png)
  
计算DQN损失后，返回每一个样本的loss作为重要性采样的参数，更新网络参数后，需要更新相应采样样本的优先级

在训练过程中$\beta$随着迭代线性增加到1

Huber loss也就是通常所说的SmoothL1 loss，其对于异常点的敏感性不如MSE，在某些情况下防止了梯度爆炸

![1](https://s1.ax1x.com/2020/08/05/asUnfI.png)
  
### 2.4 Dueling Network
#### 2.4.1 算法简介
The Dueling Network中提出的网络体系结构明确地分开状态值函数的表示和Advantage函数的表示，可以直观地判断哪些状态是有价值的，而不必了解每个动作对每个状态的影响。这对于一些状态的动作无论如何都无法对环境产生影响的时候特别有用。

Dueling网络结构通过一个共享的深度神经网络表示了A: 状态值函数V(s)V(s)和B: Advantage函数 A(s,a)A(s,a)，该共享网络的输出将A与B网络的输出结合产生Q(s,a)值。

Advantage函数的定义为：

$$A^{\pi}(s, a)=Q^{\pi}(s, a)-V^{\pi}(s)$$

值函数$V(s)$表示状态s有多好，而$Q(s,a)$函数表示在状态$s$执行动作$a$有多好。通过上面的式子，可以得到：

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+A(s, a ; \theta, \alpha)$$

其中$\theta$代表前面的卷积神经网路参数， $\alpha$代表后面值函数$V(s)$的神经网络参数，$\beta$代表后面$Q(s,a$函数的神经网络参数。

上面的式子存在不可辩问题，即已知$Q(s,a)$，$V(s)$和$A(s,a)$有无穷的组合，无法确定它们各自的值。为了解决该问题，可以强制让Advantage函数的网络在状态$s$下选择的动作$a$对应的advantage值$C$为0，即将Advantage的值都减去$C$，该方法保证可以通过$Q$将$V$和$A$恢复出来，公式如下：

$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+\left(A(s, a ; \theta, \alpha)-\max _{a^{\prime} \in[.4]} A(s, a \prime ; \theta, \alpha)\right)$$

但是可以看出，上面的式子中Advantage函数需要拟合当任意的最优动作改变时对应的Advantage值，这会造成训练过程的不稳定，因此，改进方法是利用$A(s,a)$的平均值代替求最大值的操作，该方法的潜在意思是：相比于一直学习拟合最大的动作值对应的Advantage值，只需要拟合平均的动作值对应的Advantage值就好，公式如下：
$$Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+\left(A(s, a ; \theta, \alpha)-\frac{1}{\|\mathcal{A}\|} \sum_{a^{\prime}} A(s, a \prime ; \theta, \alpha)\right)$$

#### 2.4.2 代码实现

![1](https://s1.ax1x.com/2020/08/05/asUW1x.png)

### 2.5 Noisy Networks for Exploration
#### 2.5.1 算法简介
NoisyNet是一种探索方法，可以通过学习网络权重的扰动来促进策略的探索，主要认为对权重向量的微小更改会造成多个时间步中引发一致且可能非常复杂的状态相关的策略更改。

对于一个输入为$p$维，输出为$q$维的的线性神经元来说：

$$y=wx+b$$

其中$x \in \mathbb{R}^p$是神经元输入，$w \in \mathbb{R}^{q \times p}$，$b \in \mathbb{R}$是神经元输出。

对应的noisy线性层：

$$y=\left(\mu^{w}+\sigma^{w} \odot \epsilon^{w}\right) x+\mu^{b}+\sigma^{b} \odot \epsilon^{b}$$

其中$\mu^{w} \in \mathbb{R}^{q \times p}$, $\mu^{b} \in \mathbb{R}^{q}$, $\sigma^{w} \in \mathbb{R}^{q \times p}$和$\sigma^{b} \in \mathbb{R}^{q}$是可学习的参数，而$\epsilon^{w} \in \mathbb{R}^{q \times p}$和$\epsilon^{b} \in \mathbb{R}^{q}$是可以通过下面两种方式产生的随机噪声变量：
  
  * Independent Gaussian Noise: 每一个添加给权重和偏置的随机噪声都是从单位高斯分布中采样和独立的，对于每一个线性神经元来说，将会有$pq+q$个噪声变量
  * Factorised Gaussian Noise: 该方法对于运算来说非常高效。它通过生成2个随机高斯噪声向量$(p,q)$的外积来产生$pq+q$个随机噪声：$$\begin{array}{c}
\epsilon_{i, j}^{w}=f\left(\epsilon_{i}\right) f\left(\epsilon_{j}\right) \\
\epsilon_{j}^{b}=f\left(\epsilon_{i}\right) \\
f(x)=\operatorname{sgn}(x) \sqrt{|x|}
\end{array}$$

#### 2.5.2 代码实现
NoisyNet可以看做是$\epsilon$-greedy的一种，因此与$\epsilon$有关的代码都可以省去。

![1](https://s1.ax1x.com/2020/08/05/asdgQx.png)

### 2.6 Categorical DQN
#### 2.6.1 算法简介
在Categorical DQN论文中，作者认为DQN的训练目标是学习奖惩值的分布而不是某个状态下的奖惩值期望，核心思想是奖惩值的分布满足一个贝尔曼方程的变体，即在$S_t$状态下选择动作$A_t$，在最优策略的情况下，奖惩值的回报应该与在$S_{t+1}$的状态下，根据最优策略选择的$A_{t+1}$后产生的奖惩值分布有关系，即可根据折扣因子将其收缩至0，并通过奖惩（或者随机情况下的奖惩值分布）进行偏移。他们提出利用投射到一个离散支持向量$z$上的概率质量来建模奖惩值的分布。离散支持向量$z$具有$N_{atoms} \in \mathbb{N}^+$个原子, 其中$z_i = V_{min} + (i-1) \frac{V_{max} - V_{min}}{N-1}, i\in {1, …, N_{atoms}}$。

以概率分布的角度的Q-learning变体是通过对目标分布建立一个新的支持向量，然后最小化当前分布$d_t$和目标分布$d_t……{\prime}$之间的KL(Kullbeck-Leibler)散度来进行训练：

$$d_{t^{\prime}}=\left(R_{t+1}+\gamma_{t+1} z, p_{h} a t \theta\left(S_{t+1}, \hat{a}_{t+1}^{*}\right)\right)$$

最小化目标：

$$D_{K L}\left(\phi_{z} d_{t^{\prime}} | d_{t}\right)$$

其中$\phi_z$是一个目标分布在固定向量空间$z$上的一个L2投影, $\hat{a}^{\*}_{t+1} = \arg\max_{a} q_{\hat{\theta}} (S_{t+1}, a)$是在状态$S_{t+1}$下对应平均动作值函数$q_{\hat{\theta}} (S_{t+1}, a) = z^{T}p_{\theta}(S_{t+1}, a)$的最优动作。

#### 2.6.2 代码实现
通过神经网络来表示一个参数化的分布，在DQN中，输出的大小变成了atom_size * out_dim，然后对每一个输出的动作值都进行softmax操作，这样确保了标准化不同动作值之间的差异。

通过每一个动作的Softmax分布于支持向量的内积操作来估计$Q(s_t, a_t)$的值，其中支持向量为：

$$\begin{array}{c}
z_{i}=V_{\min }+i \Delta z: 0 \leq i<N, \Delta z=\frac{V_{\max }-V_{\min }}{N-1} \\
Q\left(s_{t}, a_{t}\right)=\sum_{i} z_{i} p_{i}\left(s_{t}, a_{t}\right)
\end{array}$$

其中$p_i$是$z_i$的概率分布，即Softmax函数的输出。

![1](https://s1.ax1x.com/2020/08/05/asarxP.png)

### 2.7 N-step Learning
#### 2.7.1 算法简介
Q-learning会利用一个单一的奖惩值和下一个状态下最优动作的Q值来进行自举(Bootstrap)，该方法只能利用下一个时刻的信息，改进思路是向前多看几步，利用接下来的N步信息，这就是在状态$S_t$下的Truncated N-Step Return，定义为：$$R_{t}^{(n)}=\sum_{k=0}^{n-1} \gamma_{t}^{(k)} R_{t+k+1}$$
DQN的变体N-Step Learning的目标是通过最小化下面的损失函数：

$$\left(R_{t}^{(n)}+\gamma_{t}^{(n)} \max _{a^{\prime}} q_{\theta}^{-}\left(S_{t+n}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}$$

利用调整后的N-Step Learning通常会加速智能体的训练过程。


#### 2.7.2 代码实现
这里和DQN不同的一点是利用deque来实现经验回放池：

![1](https://s1.ax1x.com/2020/08/05/asdusf.png)

## 三、实验内容

在Atari游戏上打败了多个DRL算法：

![1](https://s1.ax1x.com/2020/08/05/astJKS.png)

## 四、缺点
暂无评价。

## 五、优点
非常“实验主义”的一篇论文，对Deepp Q Network系列的工作各自的优势进行了详细的研究。

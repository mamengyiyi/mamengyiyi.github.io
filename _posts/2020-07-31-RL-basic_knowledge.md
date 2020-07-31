---
layout:     post
title:      强化学习基础知识
subtitle:   MDP
date:       2020-07-31
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - 强化学习基础知识
---
---
## MDP
在强化学习中，马尔可夫决策过程（Markov decision process, MDP）是对**完全可观测的环境**进行描述的，也就是说观测到的状态内容完整地决定了决策的需要的特征。几乎所有的强化学习问题都可以转化为MDP。本讲是理解强化学习问题的理论基础。
### 马尔可夫过程 Markov Process 
#### 马尔可夫性 Markov Property 
![1](https://s1.ax1x.com/2020/07/31/aQkZH1.png)

即该状态具有马尔可夫性的含义为：
  * 某一状态信息包含了所有相关的历史。
  * 只要当前状态可知，历史信息history就可以被丢弃。
  * 当前状态就可以决定未来。The future is independent of the past given the present
  
可以用下面的状态转移概率公式来描述马尔可夫性：

$$P_{ss’} = P[S_{t+1}=s’|S_t=s]$$

下面的状态转移矩阵定义了所有状态的转移概率：

$$\mathcal{P}=\left[\begin{array}{ccc}{P_{11}} & {\cdots} & {P_{1 n}} \\ {\vdots} & {} & {} \\ {P_{n 1}} & {\cdots} & {P_{n n}}\end{array}\right]$$

式中$n$为状态数量，矩阵中每一行元素之和为1。

#### 马尔可夫过程 Markov Process
![1](https://s1.ax1x.com/2020/07/31/aQkHV1.png)

马尔可夫过程又叫马尔可夫链(Markov Chain)，它是一个无记忆的随机过程，可以用一个元组表示，其中$S$是有限数量的状态集，$P$是状态转移概率矩阵。

如下图圆圈内是状态，箭头上的值是状态之间的转移概率。class是指上第几堂课，facebook指看facebook网页，pub指去酒吧，pass指通过考试，sleep指睡觉。例如处于class1有0.5的概率转移到class2，或者0.5的概率转移到facebook。

![1](https://s1.ax1x.com/2020/07/31/aQAWdI.png)

从而可以产生非常多的随机序列，例如C1 C2 C3 Pass Sleep或者C1 FB FB C1 C2 C3 Pub C1 FB FB FB C1 C2 C3 Pub C2 Sleep等。这些随机状态的序列就是马尔可夫过程。
### 马尔可夫奖励过程 Markov Reward Process

![1](https://s1.ax1x.com/2020/07/31/aQATSS.png)

**马尔可夫奖励过程在马尔可夫过程的基础上增加了奖励$R$和衰减系数$\gamma$。**
  * $R$是一个奖励函数。$S$状态下的奖励是某一时刻$t$处在状态$s$下在下一个时刻$t+1$能获得的奖励期望$R_{s} = E\[R_{t+1} \| S_{t} = s\]$
  * 衰减系数Discount Factor: $\gamma \in [0, 1]$，其远期利益具有一定的不确定性，符合人类对于眼前利益的追求等。
  
#### 回报 Return
定义：回报$G_{t}$为在一个马尔可夫奖励链上从$t$时刻开始往后所有的奖励的有衰减的总和。公式如下：

$$G_t = R_{t+1}+\gamma R_{t+2}+…=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

其中衰减系数体现了未来的奖励在当前时刻的价值比例，在$k+1$时刻获得的奖励R在t时刻的体现出的价值是$\gamma^k R$ ，$\gamma$接近0，则表明趋向于“近视”性评估；$\gamma$接近1则表明偏重考虑远期的利益。

#### 价值函数 Value Function
![1](https://s1.ax1x.com/2020/07/31/aQVlgU.png)
状态值函数给出了某一状态或某一动作的长期价值。

定义：一个马尔可夫奖励过程中的状态值函数为从该状态开始的马尔可夫链回报的期望：$v(s) = E [ G_{t} \| S_{t} = s ]$

#### 价值函数的推导
##### Bellman方程 - MRP
先尝试用价值的定义公式来推导看看能得到什么：

$$\begin{aligned} v(s) &=\mathbb{E}\left[G_{t} | S_{t}=s\right] \\ &=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots | S_{t}=s\right] \\ &=\mathbb{E}\left[R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\ldots\right) | S_{t}=s\right] \\ &=\mathbb{E}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right] \\ &=\mathbb{E}\left[R_{t+1}+\gamma v\left(S_{t+1}\right) | S_{t}=s\right] \end{aligned}$$

这个推导过程相对简单，仅在导出最后一行时,将$G_{t+1}$变成了$v(S_{t+1})$。

通过方程可以看出$v(s)$由两部分组成，一是该状态的即时奖励期望，即时奖励期望等于即时奖励，因为根据即时奖励的定义，它与下一个状态无关；另一个是下一时刻状态的价值期望，可以根据下一时刻状态的概率分布得到其期望。如果用s’表示s状态下一时刻任一可能的状态，那么Bellman方程可以写成：

$$\begin{aligned} v(s) &= E[G_t|S_t=s]= E[R_{t+1} + \gamma (R_{t+2}+\gamma R_{t+3}+…) | S_t=s] \\
& = E[R_{t+1} + \gamma G_{t+1} | S_t=s] = R_s + \gamma \sum_{s’\in S}P_{ss’}v(s’) \end{aligned}$$

下图已经给出了$\gamma=1$时各状态的价值（该图没有文字说明$\gamma=1$，根据视频讲解和前面图示以及状态方程的要求，$\gamma$必须要确定才能计算），状态$C_{3}$的价值可以通过状态Pub和Pass的价值以及他们之间的状态转移概率来计算：$4.3 = -2 + 1.0 ( 0.6 * 10 + 0.4 * 0.8 )$

![1](https://s1.ax1x.com/2020/07/31/aQVgUI.png)


##### Bellman方程的矩阵形式和求解
![1](https://s1.ax1x.com/2020/07/31/aQVban.png)

实际上，计算复杂度是$O(n^{3})$，$n$是状态数量。因此直接求解仅适用于小规模的MRPs。大规模MRP的求解通常使用迭代法。常用的迭代方法有：动态规划Dynamic Programming、蒙特卡洛评估Monte-Carlo evaluation、时序差分学习Temporal-Difference，后文会逐步讲解这些方法。

### 马尔可夫决策过程 Markov Decision Process
![1](https://s1.ax1x.com/2020/07/31/aQZMdA.png)

相较于马尔可夫奖励过程，马尔可夫决策过程多了一个动作（动作）集合$A$。看起来很类似马尔可夫奖励过程，但这里的$P$和$R$都与具体的动作$a$对应，而不像马尔可夫奖励过程那样仅对应于某个状态，$A$表示的是有限的动作的集合。具体的数学表达式如下：

$$P^a_{ss’} = P[S_{t+1}=s’|S_t=s,A_t=a]$$

$$R^a_{s} = E[R_{t+1} | S_{t} = s,A_t=a ]$$

下图给出了一个可能的MDP的状态转化图。图中红色的文字表示的是采取的动作，而不是先前的状态名。对比之前的学生MRP示例可以发现，即时奖励与动作对应了，同一个状态下采取不同的动作得到的即时奖励是不一样的。由于引入了Action，容易与状态名混淆，因此此图没有给出各状态的名称；此图还把Pass和Sleep状态合并成一个终止状态；另外当选择”去查阅文献”这个动作时，主动进入了一个临时状态（图中用黑色小实点表示），随后被动的被环境按照其动力学分配到另外三个状态，也就是说此时Agent没有选择权决定去哪一个状态。

![1](https://s1.ax1x.com/2020/07/31/aQZdds.png)

#### 策略 Policy $\pi$
策略$\pi$是概率的集合或分布，其元素$\pi(a|s)$为对过程中的某一状态$s$采取可能的动作$a$的概率。用$\pi(a|s)$表示。

一个策略完整定义了Agent的动作方式，也就是说定义了Agent在各个状态下的各种可能的动作方式以及其概率的大小。Policy仅和当前的状态有关，与历史信息无关；同时某一确定的Policy是静态的，与时间无关；但是Agent可以随着时间更新策略。

当给定一个MDP:$M = <S, A, P, R, \gamma>$和一个策略$\pi$,那么状态序列$S_1,S_2,...$是一个马尔可夫过程$<S,P^\pi>$。状态转移概率公式表示$P^\pi_{ss’}=\sum_{a\in A}\pi(a\|s)P^a_{ss’}$。在执行策略$\pi$时，状态从$s$转移至$s'$的概率等于一系列概率的和，这一系列概率指的是在执行当前策略时，执行某一个动作的概率与该动作能使状态从$s$转移至$s'$的概率的乘积。

同样的，状态和奖励序列$S_{1}, R_{2}, S_{2}, R_{3}, S_{3}, …$是一个马尔可夫奖励过程$<S, P^{\pi}, R^{\pi}, \gamma>$。奖励函数表示$R^\pi_{s}=\sum_{a\in A}\pi(a\|s)R^a_{s}$。当前态$s$下执行某一指定策略得到的即时奖励是该策略下所有可能动作得到的奖励与该动作发生的概率的乘积的和。

策略$\pi$在MDP中的作用相当于agent可以在某一个状态时做出选择，进而有形成各种马尔可夫过程的可能，而且基于策略产生的每一个马尔可夫过程是一个马尔可夫奖励过程，各过程之间的差别是不同的选择产生了不同的后续状态以及对应的不同的奖励。

#### 基于策略$\pi$的价值函数 Value Function
##### 状态值函数 STATE-VALUE FUNCTION $V$
定义$v_\pi(s)$是在MDP下的基于策略$\pi$的状态值函数，表示从状态$s$开始，遵循当前策略时所获得的收获的期望；或者说在执行当前策略$\pi$时，衡量个体处在状态$s$时的价值大小。数学表示如下：$$v_{\pi}(s) = E_{\pi}[G_t|S_t=s]$$

注意策略是静态的、关于整体的概念，不随状态改变而改变；变化的是在某一个状态时，依据策略可能产生的具体动作，因为具体的动作是有一定的概率的，策略就是用来描述各个不同状态下执行各个不同动作的概率。

##### 动作值函数 ACTION-VALUE FUNCTION $Q$
定义$q_{\pi}(s,a)$为动作值函数，表示在执行策略$\pi$时，对当前状态$s$执行某一具体动作$a$所能的到的收获的期望；或者说在遵循当前策略$\pi$时，衡量对当前状态执行动作$a$的价值大小。动作值函数一般都是与某一特定的状态相对应的。动作值函数的公式描述如下:

$$q_{\pi}(s,a)= E_{\pi}[G_t|S_t=s, A_t=a]$$

由于策略$\pi(a\|s)$是可以改变的，因此两个值函数的取值不像MRP一样是固定的，那么就能从不同的取值中找到一个最大值即最优值函数。MDP需要解决的问题并不是每一步到底会获得多少累积reward，而是找到一个最优的解决方案。

#### Bellman期望方程 Bellman Expectation Equation
![1](https://s1.ax1x.com/2020/07/31/aQZgL4.png)

根据这两个值函数的定义，它们之间的关系表示为:
  * $v_{\pi}(s) = \sum_{a\in A}\pi(a\|s)q_{\pi}(s,a)$
  * $q_{\pi}(s,a) = R_{s}^{a} + \gamma \sum_{s’\in S}P_{ss’}^a\sum_{a’\in A}\pi(a’\|s’)q_{\pi}(s’,a’)$
  
下图解释了红色空心圆圈状态的状态价值是如何计算的，遵循的策略随机策略，即所有可能的动作有相同的几率被选择执行。

![1](https://s1.ax1x.com/2020/07/31/aQZbOe.png)

和MRP类似的,我们也可以得到矩阵形式和求解。

![1](https://s1.ax1x.com/2020/07/31/aQeC6S.png)

#### 最优价值函数
最优状态值函数$v_{* }$指的是在从所有策略产生的状态值函数中，选取使状态$s$值最大的函数：

$$v_* = \max \limits_{\pi} v_{\pi}(s)$$

类似的，最优动作值函数$q(s,a)$指的是从所有策略产生的动作值函数中，选取是状态动作对$<s, a>$价值最大的函数：

$$q_* (s,a) = \max \limits_{\pi} q_{\pi}(s,a)$$

最优价值函数明确了MDP的最优可能表现，当我们知道了最优价值函数，也就知道了每个状态的最优价值，这时便认为这个MDP获得了解决。

#### 最优策略 
当对于任何状态$s$，遵循策略$\pi$的价值不小于遵循策略$\pi'$下的价值，则策略$\pi$优于策略$\pi'$：

$$\pi \geq \pi' \quad if \quad  v_{\pi}(s)\geq v_{\pi'}(s)  ,\forall s$$

定理:对于任何MDP，下面几点成立：
  * 存在一个最优策略，比任何其他策略更好或至少相等。
  * 所有的最优策略有相同的最优价值函数。
  * 所有的最优策略具有相同的动作值函数。
  
#### 寻找最优策略
可以通过最大化最优动作值函数来找到最优策略：

$$\pi_{*}(a | s)=\left\{\begin{array}{ll}{1} & {\text { if } a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} q_{*}(s, a)} \\ {0} & {\text { otherwise }}\end{array}\right.$$

对于任何MDP问题，总存在一个确定性的最优策略；同时如果我们知道最优动作值函数，则表明我们找到了最优策略。

#### Bellman最优方程 Bellman Optimality Equation
针对$v$，一个状态的最优价值等于从该状态出发采取的所有动作产生的动作价值中最大的那个动作价值：

$$v_{*}(s)=\max _{a} q_{*}(s, a)$$

针对$q$，在某个状态$s$下，采取某个动作的最优价值由2部分组成，一部分是离开状态$s$的即刻奖励，另一部分则是所有能到达的状态 $s'$的最优状态价值按出现概率求和：

$$
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)
$$

组合以上两个公式，可以得到Bellman最优方程：

$$
\begin{aligned} v_{*}(s) &=\max _{a} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right) \\ q_{*}(s, a) &=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in S} \mathcal{P}_{s s^{\prime}}^{a} \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right) \end{aligned}
$$

满足Bellman最优方程，意味着找到了最优策略。也就是$v_{\pi}(s)=\max_{a} q_{*}(s, a)$，也就是不需要在进行策略改进。

#### 求解Bellman最优方程
Bellman最优方程是非线性的，没有固定的解决方案，通过一些迭代方法来解决：价值迭代、策略迭代、Q-learning、Sarsa等。后续会逐步讲解展开。

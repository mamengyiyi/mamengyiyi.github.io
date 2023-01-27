



本文发表于ICML 2022，论文链接详见https://proceedings.mlr.press/v162/filos22a.html

## 一、方法

给定一个world model和value function，我们可以通过在该model上进行不同步数的rollout，通过bootstrapping来获取对同一个state的多个价值评估。基于此，如下图(c)所示，本文提出一个新的方式衡量不确定性Implicit Value Ensemble (IVE) ，即将这些对同一个state的多个价值评估视作一种ensemble来计算认知不确定性。而下图(a)和下图(b)分别展示的是value (function) ensemble和model ensemble的做法。

![image-20221019191129360](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019191129360.png)

本文这种思路背后的思想是真实的world model和value function是Bellman-consistent的，因此在world model和value function比较准确的区域，二者的self-inconsistency应该是比较低的。

本文首先定义了**k-steps model predicted value (k-MPV)**：将world model学到的Bellman operator应用不同的k步在学到的value function上，可以获得对同一个state不同的值评估：
$$
\hat{v}_{\hat{m}}^k \triangleq\left(\mathcal{T}_{\hat{m}}^\pi\right)^k \hat{v}
$$
进一步，通过变化k值，我们可以获得一个k-MPV的ensemble，记为IVE：

$$
\left\{\hat{v}_{\hat{m}}^i\right\}_{i=0}^n \triangleq \frac{\left\{\hat{v}, \mathcal{T}_{\hat{m}}^\pi \hat{v}, \ldots,\left(\mathcal{T}_{\hat{m}}^\pi\right)^n \hat{v}\right\}}{n+1 \text { value estimates }} .
$$
因此，根据IVE中不同的V值之间的disagreement (model-value inconsistency / self-inconsistency)，我们可以用该disagreement对不确定性进行评估的方式。由于k-MPVs为标量值，因此该disagreement的计算方式可以为每个值的标准差，记为$\sigma-IVE(n)$。同理，其均值可以记为$\mu-IVE(n)$。二者的加权求和可以记为：
$$
\mu+\beta * \sigma{-IVE }(n)
$$
策略可以在不同场景中进行该值的应用。当$\beta>0$，策略可以对不确定性高的地方进行探索，而当$\beta<0$，策略可以避免访问不确定性高的地方，是一种保守的体现，可以应用于离线强化学习场景。

在具体实现时，在tabular case下，k-MPVS可以精确计算出来。在神经网络函数近似时，采用Monte Carlo (MC) sampling来对其进行近似：
$$
\hat{\mathbf{v}}_{\hat{\mathbf{m}}}^{\mathbf{k}}(s)=\sum_{i=1}^{k-1} \gamma^{i-1} \mathbf{r}_{\hat{\mathbf{m}}}^{\mathbf{i}+1}+\gamma^k \hat{v}\left(\mathbf{s}_{\hat{\mathbf{m}}}^{\mathbf{k}}\right)
$$
同时，为了减少采样次数，本文直接复用采样的轨迹用来计算多个不同k的v。

## 二、实验

本文首先给出一个例子来验证提出的方法可以用于评估不确定性：

![image-20221019192940559](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019192940559.png)

接下来，在主实验部分，本文希望验证如下假设：

* H1：在分布内的区域self-inconsistency比较低
* H2：在OOD的区域self-inconsistency比较高
* H3：OOD区域的self-inconsistency会随着训练分布的接近而降低
* H4： self-inconsistency可以用于探索
* H5：避免self-inconsistency可以对分布偏移有较好的鲁棒性
* H6：IVE的ensemble averaging整体上比单个组件更加鲁棒

在grid-world上，本文的做法可以较好地区分出分布内与分布外的数据，验证了假设H1和H2：

![image-20221019193030980](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019193030980.png)

在procgen游戏上，本文验证了随着训练中见过的游戏关卡越多，测试时表现就越好，其对应的认知不确定性也越低，验证了H1，H2和H3：

![image-20221019193236973](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019193236973.png)

下图中左图证明了在乐观探索场景下，本文提出的IVE可以有效地鼓励策略访问OOD的状态，右图证明了在悲观学习场景下，该方法可以可以有效地防止策略访问OOD的状态，从而验证了假设H4和H5。
![image-20221019193252732](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019193252732.png)

本文还将验证了利用IVE的均值进行值规划要比只利用IVE的单个组件更有效，从而验证了假设H6。

![image-20221019193511433](C:\Users\马亿\AppData\Roaming\Typora\typora-user-images\image-20221019193511433.png)



## 三、总结

- 本文提出了一种新的认知不确定性的评估方法，在表格与图像输入的RL任务上都能很好地评估认知不确定性，该不确定性也可用于引导探索或保守学习等。
- 该方法将单点的world model和value function的估值转化成了值函数的多个估值，所需的计算资源更少；对于offline rl来说，这种做法由于是从bellman consistency的角度出发，因此可能更适合用于做offline rl任务
- 但这种做法得到的IVE的每个组件之间的diversity可能不够大








---
layout:     post
title:      Machine Learning: A probabilistic perspective学习笔记
subtitle:   Introduction
date:       2019-08-25
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - DL
    - Machine Learning: A probabilistic perspective
---

# 1.1 Machine Learning: what and why?
在阅读阿里《不止代码》时看到了师兄对这本书的极力推荐，因此在实习之余阅读学习。本书偏向于概率论的角度来理解和解释机器学习。

# 1.1.1 Types of Machine Learning
机器学习大概分成下面三种，
## 有监督学习
* predictive or supervised method
* 给定训练集，求从输入到输出的映射。
* 每个输入的维度都是一个特征(features, attributes, covariates)

## 无监督学习
* descriptive and unsupervised learning
* 从数据集中找到某些模式，但是数据集没有显式标注的数据和目标函数。
* 又叫知识发现（knowledge discovery）。

## 强化学习
* reinforcement learning
* agent，通过环境给予的reward or punishment来学习做出决策，即执行 action，
* decision theory 是强化学习的基础

# 1.2 Supervised Learning 有监督学习
## 1.2.1 Classification 分类
分类的目标是学习从输入$x$到输出 $y \in {1,...,C}$ 的一个映射（mapping），其中$C$表示类别的数量。若$C=2$，那么就是二分类问题（binary classification）；若$C>2$，就是多分类问题（multiclass classification）。二分类和多分类分类结果都是互斥的（mutually exclusive），即分类结果只有一个。如果同时有多个分类结果，比如一个物体既属于红色类又属于圆形类，那么这类问题叫做多标签分类问题（multi-label classification）。一个形式化（formalize）分类问题的方法就是函数逼近（function approximation）。假设分类问题要学习的映射是一个未知的函数$f$，真实的映射关系表示成$y=f(x)$，分类问题的目的为从训练集中学习一个函数$\hat{f}$来逼近或者拟合原始的函数$f$。我们的目标是用这个函数估计全新的输入（训练集中没有的样本）所对应的输出，叫做是泛化（generalization）。
<br>

从概率的角度来看预测分类的问题，即从所有的预测的类别中，选择条件概率最大的那个类别作为预测的结果。即，
@\hat{y}=\hat{f}(\mathbf{x})=\underset{c=1}{\operatorname{argmax}} p(y=c | \mathbf{x}, \mathcal{D})@
其中条件概率$p(y=c | \mathbf{x}, \mathcal{D})$表示给定输入$x$和训练集$D$之后，分类结果是$y=c$的概率。从概率分布的角度来看，$\hat{y}$就是离散概率分布$ p(y=c | \mathbf{x}, \mathcal{D})$的 mode 值。这种估计的方法叫作最大后验估计（MAP estimate, maximum a posteriori）。

分类问题的实际应用包括：
* document classification and email spam filtering 
* Classifying flowers 鸢尾花（iris）分类问题 
* Image classification and handwriting recognition 
* Face detection and recognition 


## 1.2.2 Regression 回归
回归与分类很像，区别在于分类的输出为离散的值，而回归的输出为连续值。<br>
回归问题的实际应用包括：
* 给定之前的股价，预测明天的股价
* 给定控制信息，预测机器人的位置信息
* 给定医学测量值，预测某些医学指数
* 给定日期，时间，大门传感器，预测大楼位置的温度

# 1.3 Unsupervised learning 无监督学习
目标是发现数据中的内在结构，也叫作是知识发现（knowledge discovery）。无监督学习可以形式化定义为密度估计（density estimation）问题，即建立形式为$p\left(\mathbf{x}\_{i} | \boldsymbol{\theta}\right)$的模型。

无监督学习和监督学习不同的包括：
* 非监督学习的模型为$p\left(\mathbf{x}\_{i} | \boldsymbol{\theta}\right)$，是unconditional density estimation；而监督学习的模型为$p\left(y_{i} | \mathbf{x}\_{i}, \boldsymbol{\theta}\right)$，是conditional density estimation
* 监督学习预测的是n-class问题，用单元概率模型（univariate probability models ）即可；而无监督学习则要从$n$维输入中建立多元概率模型（multivariate probability models）
* 无监督学习比监督学习更符合人类和动物的学习方式，应用更广泛。

## 1.3.1 Discovering clusters
无监督学习的典型例子就是聚类（clustering）问题。聚类的第一个目标是，确定数据中一共有多少类别；第二个目标是，确定每个数据属于什么分类。本书中讨论 model based clustering，而不是 hoc algorithm（自适应算法）。<br>
聚类问题的实际应用包括：
* 天文学中星星的分类
* 电商中用户的分类

## 1.3.2 Discovering latent factors
处理高维数据的一般方法是把数据从高维降低到低维dimensionality reduction），叫做降维。低维的变量，称为隐变量（latent factors）。主成分分析（PCA principal components analysis ）是用于降维的经典方法。它可以看作是无监督中的多标签的线性回归（linear regression with multi-outputs)。<br>
降维问题的实际应用包括：
* 使用PCA来解释基因微阵列数据
* NLP中的文档检索分析
* 信号处理中，将信号分离到它们的不同来源
* 计算机图形学中，将运动捕捉数据投影到低维空间，并用它来创建动画

## 1.3.3 Discovering graph structure
当某些变量之间会有直接的联系时，可以找到这些点之间的边连接，学习稀疏图模型（sparse graphical model）。<br>
稀疏图问题的实际应用包括：
* 解释数据的结构
* 建模实体间的关系并用于推测

## 1.3.4 Matrix completion
矩阵补全即修复或者补全缺失的部分数据。<br>
矩阵补全的实际应用包括：
* Image inpainting 图像修复
* Collaborative filtering 协同过滤，比如根据用户以前看的电影和其他人对电影的评价，给予推荐。因为一个用户几乎不可能对所有的电影评分，所以user-movie的评分矩阵中的绝大多数值为NAN
* Market basket analysis 市场购物篮分析，用于推荐系统等


# 1.4 Some basic concepts in machine learning
## 1.4.1 Parametric vs non-parametric models
参数模型和非参数模型的主要区别在于模型是具有固定数量的参数，还是参数的数量会随着训练数据量的增加而增加。前者称为参数模型，后者称为非参数模型。 参数模型具有通常更快使用的优点，但缺点是对数据分布的性质做出更强的假设。非参数模型更灵活，但对于大型数据集而言通常在计算上难以处理。 

## 1.4.2 A simple non-parametric classifier: K-nearest neighbors
KNN算法的思路是，寻找样本点最近的K个近邻，根据投票少数服从多数的原则得到预测的结果。

## 1.4.3 The curse of dimensionality 维度诅咒
由于维度诅咒，维度太高时，KNN等非参数模型性能会变得很差。维度诅咒可以这样理解：随着特征的增多，即低维（特征少）转向高维的过程中，样本会变的稀疏。这个稀疏可以有两种理解方式：(1)样本数目不变，样本彼此之间距离增大。(2)样本密度不变，所需的样本数目指数倍增长。举例来说，$p=1$，则单位球(简化为正值的情况）变为一条$[0,1]$之间的直线。如果我们有$N$个点，则在均匀分布的情况下，两点之间的距离为$1/N$。$p=2$，单位球则是边长为1的正方形，如果还是只有$N$个点，则两点之间的平均距离为$\sqrt{\frac{1}{n}}$。换言之，如果我们还想维持两点之间平均距离为$1/N$，那么则需$N^{2}$个点。以此类推，在$p$维空间，$N$个点两两之间的平均距离为$N^{-\frac{1}{p}}$，或者需要个$N^{p}$个点来维持$1/N$的平均距离。<br>

这种稀疏性会带来不好的影响：样本分布位于中心附近的概率，随着维度的增加会越来越低；而样本处在边缘的概率，则越来越高，即$N$个点在$p$维单位球内随机分布，则随着$p$的增大，这些点会越来越远离单位球的中心，转而往外缘分散。想象二维空间下的两个同心圆，假设$r_{1}=0.5, r_{2}=1$，那么面积之比为1/4；如果半径不变，在三维空间中，体积之比变成1/8；到了8维空间下，超球体的体积之比为1/256。当维数趋于无穷时，位于中心附近的概率趋于0。这种情况下，一些度量相异性的距离指标（如：欧式距离）效果会大大折扣，从而导致一些基于这些指标的分类器在高维度的时候表现不好。<br>

对于KNN来说，假设N个点均匀分布在$p$维的unit ball上，中心在原点，如果对中心原点进行一个k近邻估计，中心原点到近邻数据点的平均距离为$(1-(1/2^{N}))^{1/p}$。随着维数增加，这个值越来越接近1，大多数点接近于边界，极端来看，每个点距离中心原点有着很接近的距离，这时候k近邻的邻居就不再是local了。


## 1.4.4 Parametric models for classification and regression
对抗维度灾难的主要方法是对数据分布的性质做一些假设，归纳偏置（inductive bias）。

## 1.4.5 Linear regression
见第七章。

## 1.4.6 Logistic regression
见第八章。

## 1.4.7 Overfitting


## 1.4.8 Model selection
misclassification rate / generalization error / underfit / validation set / cross validation

## 1.4.9 no free lunch theorem
> All models are wrong, but some models are useful. – George Box




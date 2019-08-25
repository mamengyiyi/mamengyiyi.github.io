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
    -  Machine Learning: A probabilistic perspective


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
* 也叫知识发现（knowledge discovery）。

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


1.2.2 Regression 回归
1.3 Unsupervised learning 无监督学习
目标是发现数据中的有趣的结构，也叫作是知识发现（knowledge discovery）。我们可以把这个任务形式化定义为密度估计（density estimation）问题，即从 p(xi|θ)p(xi|θ) 中建立模型。

和监督学习不同的是：

p(xi|θ)p(xi|θ) 和 p(yi|xi,θ)p(yi|xi,θ) 的不同，是否基于条件概率分布
conditional density estimation && unconditional density estimation
监督学习预测的是n-class问题，用univariate probability models 即可，而无监督学习则要从n维输入xixi中建立multivariate probability models
无监督学习比监督学习更符合人类的动物的学习方式，应用也会更广泛。

1.3.1 Discovering clusters
无监督学习的一个典型的例子（a canonical example）就是聚类（clustering）问题。

hidden or latent variable 隐藏变量（潜变量）聚类的一个目标是，确定数据中一共有多少类别；第二个目标是，每个数据属于什么分类。本书中讨论 model based clustering，而不是 hoc algorithm（自适应算法）。应用：

天文学中星星的分类
电子商务中用户的分类
医学中。。。
1.3.2 Discovering latent factors
把数据从高维降低到低维乘降维（dimensionality reduction），在低维的变量，称为隐因子（latent factors），主成分分析（underlying PCA principal components analysis ）可以看作是无监督中的多标签的线性回归（linear regression with multi labels)。降维的应用也很多，如生物学；NLP中的文档查找；信号处理；计算机图形学

1.3.3 Discovering graph structure
因为某些变量之间会有直接的联系，因此可以考虑简历图模型，找到这些点之间的边连接。学习稀疏图模型（sparse graphical model）有两个主要的应用：

发现新知识；
get better joint probability density estimators；
1.3.4 Matrix completion
plausible 貌似可信的　imputation 修复？补全确实的部分数据

1.3.4.1 Image inpainting 图像的修复
1.3.4.2 Collaborative filtering 协同过滤　
比如根据用户以前看的电影和其他人队电影的比价，给予推荐。因为用户不可能对所有的电影都会评分，所以评分的那个矩阵应该是大部分为NAN

1.3.4.3 Market basket analysis 市场购物篮分析
association rules analysis 关联规则分析 
若有个矩阵，纵坐标是商品的id，横坐标是每次交易。预测商品的相关联性并给出推荐的商品。

# 1.4 Some basic concepts in machine learning
## 1.4.1 Parametric vs non-parametric models
分类的标准是参数的数量是固定还是随训练数据的增多而增多　computationally intractable　难以计算的

## 1.4.2 A simple non-parametric classifier: K-nearest neighbors
KNN算法的思路是，寻找样本点最近的K个近邻，投票得到预测的结果。KNN这种方法是一种 memory-based learning or instance-based learning

## 1.4.3 The curse of dimensionality 维度诅咒
由于维度诅咒，维度太高时，KNN性能太差。

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




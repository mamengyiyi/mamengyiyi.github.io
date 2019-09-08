---
layout:     post
title:      Machine Learning A probabilistic perspective学习笔记
subtitle:   Probability
date:       2019-09-08
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - DL
    - Machine Learning from probabilistic perspective
    
---

# 2.1 Introduction
对于“硬币正面朝上的概率是0.5”这句话，实际上至少有两种不同的概率解释。一种被称为频率解释。在该视角中，概率表示事件的长期执行的频率。例如，上述陈述意味着，如果我们多次翻转硬币，我们就是
期望它在大约一半的次数内正面朝上。另一种解释称为贝叶斯概率解释。在这种观点中，概率用于量化我们对某事物的不确定性，因此，它与信息相关，而不是重复试验。在贝叶斯观点中，上述陈述意味着我们相信硬币在下一次抛掷时正面或反面朝上的可能性是同样的。贝叶斯解释的一个重要优点是它可以用来模拟我们对没有长期频率的事件的不确定性。例如，我们可能想要计算极地冰盖在2020年前融化的概率。此事件将发生零次或一次，但不能重复发生。然而，我们应该能够量化我们对这一事件的不确定性。在本书中采用贝叶斯解释概率。

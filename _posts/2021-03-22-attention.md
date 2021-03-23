---
layout:     post
title:      Attention Mechanism 注意力机制
subtitle:   注意力的发明起源及各种注意力机制和模型介绍
date:       2021-03-22
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - DL
    - Attention
---
---

本文翻译、整理并扩充自[Lil's Log](https://lilianweng.github.io/lil-log/)与[Jay Alammar' Blog](https://jalammar.github.io/)，欢迎大家前往原文学习阅读！

# 一、背景

近年来，Attention（注意力）已经成为深度学习社区中一个相当流行的概念和实用的工具。 在这篇博客中，我将介绍Attention的发明以及各种Attention机制和模型，例如Transformer和SNAIL。

Attention在如下两个例子中有一定程度的体现：

- 人类对图像不同区域施以不同的关注程度来识别图像包含的内容。
- 如何将一个句子中不同的单词按照重要程度的不同将它们关联起来。 

以下图中的柴犬图片为例。人类的视觉注意力使我们能够将注意力集中在“高分辨率”的特定区域上（即我们可能会看着黄色方框中的尖耳朵），同时以“低分辨率”感知周围的图像（即白雪皑皑的背景和柴犬身上的衣服），然后进行相应的推断。 给定这一小块图像，图片中的其余像素可为我们提供线索，来推断应在此处显示的内容。 例如，我们期望在黄色方框中看到尖尖的耳朵，因为我们在红色方框的部分看到了狗的鼻子，又看到了尖尖的耳朵，还有眼睛。 但是，对于帮助我们推测黄色区域图像来说，图片底部的毛衣和毯子不会像柴犬的其他五官那样有用。

<img src="https://lilianweng.github.io/lil-log/assets/images/shiba-example-attention.png" alt="1" style="zoom:35%;" />

同样，我们可以解释一个句子或相关上下文中单词之间的关系。 如下图所示，当我们在一个句子中看到“吃”时，我们期望很快会再读到代表“美食”的词语（如图中的“苹果”）。而代表颜色的“绿色”一词描述的是食物，但对于“吃”这个动作来说可能关联关系并不强。

<img src="https://lilianweng.github.io/lil-log/assets/images/sentence-example-attention.png" alt="1" style="zoom:50%;" />



简而言之，深度学习中的Attention可以广义地解释为**重要性权重的向量**：为了预测或推断一个目标元素（例如图像中的像素或句子中的单词），我们使用Attention向量来估计目标元素与其他元素相关联的程度，并将这些元素的值乘以Attention向量进行加权后得到的总和作为目标元素的近似值。

Attention机制诞生于神经机器翻译领域。在Attention出现之前，seq2seq模型是该领域广泛使用的模型。seq2seq模型诞生于语言建模领域（[Sutskever, et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)）。 广义上讲，它旨在将输入序列（源）转换为新的序列（目标），并且两个序列都可以具有任意长度。例如，在文本或音频中的多种语言之间进行机器翻译、生成问答对话框，甚至将句子解析为语法树等，都属于这种转换任务。

seq2seq模型通常具有encoder-decoder（编码器-解码器）架构，该架构包括：

- encoder：处理输入序列，并将信息压缩为固定长度的上下文向量（也称为句子embedding），以期这种表征形式可以很好地概括整个源序列的含义。
- decoder：输入encoder处理得到的上下文向量初始化，以输出转换后的序列。 

<img src="https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-example.png" alt="1" style="zoom:30%;" />

然而，这种固定长度上下文向量设计的一个关键而明显的缺点是无法记住长句子。 通常，这种结构一旦完成了对整个输入的处理，便会忘记句子最开始部分。因此，为了解决这个问题，[Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)首次提出了初代的Attention机制，来帮助记忆神经机器翻译（NMT）中的长句子。

# 二、初代Attention简介

[Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)首次提出的包含Attention机制的encoder-decoder结构如下图所示：

<img src="https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-attention.png" alt="1" style="zoom:35%;" />

给定一个长度为$n$的源序列$\mathbf{x}$，并尝试输出一个长度为$m$的目标序列$\mathbf{y}$（粗体表示向量）：
$$
\begin{aligned}
\mathbf{x} &= [x_1, x_2, \dots, x_n] \\
\mathbf{y} &= [y_1, y_2, \dots, y_m]
\end{aligned}
$$
encoder是一个双向RNN（也可是其他RNN），该encoder将输入的每个部分$x_{i}$进行编码，使之具有前向隐状态$\overrightarrow{\boldsymbol{h}}\_i$和后向隐状态$\overleftarrow{\boldsymbol{h}}\_i$。直接将这两个隐状态进行拼接来表示$x_{i}$的状态。 这样做的动机是在一个单词的表征中将其前面和后面的单词的含义也包含进来：
$$
\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top, i=1,\dots,n
$$
decoder网络在位置$t=1,\dots,m$处的输出的词语具有隐状态$\boldsymbol{s}\_t=f(\boldsymbol{s}\_{t-1}, y_{t-1}, \mathbf{c}_t)$，其中上下文向量$c_{t}$是输入序列包含的所有$x_{i}$的隐状态的加权求和，而权重则为：
$$
\begin{aligned}
\mathbf{c}_t &= \sum_{i=1}^n \alpha_{t,i} \boldsymbol{h}_i & \small{\text{; 关于}y_t}\text{的上下文向量}\\
\alpha_{t,i} &= \text{align}(y_t, x_i) & \small{\text{; }y_t\text{与}x_i\text{匹配的好坏程度}}\\
&= \frac{\exp(\text{score}(\boldsymbol{s}\_{t-1}, \boldsymbol{h}\_i))}{\sum_{i'=1}^n \exp(\text{score}(\boldsymbol{s}\_{t-1}, \boldsymbol{h}\_{i'}))} & \small{\text{; 对上述好坏程度进行softmax得到概率值}}.
\end{aligned}
$$


align模型根据匹配程度如何，将分数$\alpha_{t,i}$分配给位置$i$处的输入和位置$t$处的输出$(y_t, x_i) $。$\{\alpha_{t, i}\}$是权重集合，用于定义对于每个输出，应该考虑源输入中的每个隐状态的比重。 在[Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)的论文中，匹配分数$\alpha$由具有单个隐藏层的前馈网络进行参数化，并且该网络与模型的其他部分共同训练。 因此，假定tanh被用作非线性激活函数，则匹配分数函数$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i)$为以下形式：
$$
\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])
$$
其中$\mathbf{v}_a$和$\mathbf{W}_a$都是要在align模型中学习的权重矩阵。

# 三、Attention家族简介

在Attention的帮助下，源序列和目标序列之间的依赖性不再受限于序列长度。 由于Attention在机器翻译方面有了很大的进步，因此很快就扩展到了计算机视觉领域（[Xu et al. 2015](http://proceedings.mlr.press/v37/xuc15.pdf)），人们开始探索各种形式的Attention机制（[Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf); [Britz et al., 2017](https://arxiv.org/abs/1703.03906); [Vaswani, et al., 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)）。

## 3.1 整体介绍

下表是几种流行的Attention机制和相应的匹配分数函数的总结：

| 方法               | 匹配分数函数$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i)$ | 论文                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Content-Base       | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$ | [Graves2014](https://arxiv.org/abs/1410.5401)                |
| Additive           | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$ | [Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)          |
| Location-Base      | $\alpha_{t,i} = \text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$ | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)            |
| General            | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$ | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)            |
| Dot-Product        | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\boldsymbol{h}_i$ | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)            |
| Scaled Dot-Product | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \frac{\boldsymbol{s}_t^\top\boldsymbol{h}_i}{\sqrt{n}}$ | [Vaswani2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) |

[Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)方法在[Luong2015](https://arxiv.org/pdf/1508.04025.pdf)中被称为“ concat”，在[Vaswani2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)中被称为“加法注意力”。[Vaswani2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)中在Dot-Product的基础上添加$1/\sqrt{n}$，基于的考虑是当输入较大时，softmax函数的梯度可能会非常小，难以进行有效学习，因此要使用该项对输入进行缩放。

从更广泛的角度，Attention机制可以如下分类：

| 方法           | 描述                                                         | 论文                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Self-Attention | 将相同输入序列的不同位置进行关联。 从理论上讲，Self-Attention可以采用上述任何得分函数，只需将目标序列替换为相同的输入序列即可。 | [Cheng2016](https://arxiv.org/pdf/1601.06733.pdf)            |
| Global/Soft    | 对整个输入状态空间进行关联，如整张图片。                     | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf)         |
| Local/Hard     | 对部分输入状态空间进行关联，如图片的某些部分。               | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf)；[Luong2015](https://arxiv.org/pdf/1508.04025.pdf) |

其中，Self-Attention又被称作Intra-Attention。

## 3.2 Self-Attention

Self-Attention（自我注意力），也称为Intra-Attention，是一种与单个序列的不同位置相关联的Attention机制，目的是计算同一序列的表征形式。Self-Attention已经在机器阅读、图像描述生成中展现出了强大的能力。

### 3.2.1 Self-Attention的整体介绍

在本节中，我们介绍Self-Attention的整体作用。在模型处理每个单词（输入序列中的每个位置）时，Self-Attention使其能够查看输入序列中的其他位置以寻找线索，从而有助于更好地对该单词进行编码。在声名大噪的Transformer中，Self-Attention便被用于将其他相关单词的“理解”关联到我们当前正在处理的单词中。

举例来说，假设下面的句子是我们要翻译的输入句子：“这只动物没有过马路，因为它太累了”。这句话中的“它”指的是什么？ 是指街道还是动物？ 对人类来说，这是一个简单的问题，但对算法而言却不那么简单。当模型处理“ 它”一词时，Self-Attention可以将“ 它”与“ 动物”相关联。

下图是另外一个例子。[long short-term memory network](https://arxiv.org/pdf/1601.06733.pdf)论文中利用Self-Attention做机器阅读。如下图所示，Self-Attention机制使我们能够学习当前单词和句子前一部分之间的相关性。图中当前单词为红色，蓝色阴影的大小表示之前的单词在理解当前单词时相关的程度。

<img src="https://lilianweng.github.io/lil-log/assets/images/cheng2016-fig1.png" alt="1" style="zoom:35%;" />

正式地来说，对于一个单词节点来说，它其他单词节点收到的每条信息的Value的权重取决于它自身的Query与其他单词的Key。如下图所示：

<img src="https://z3.ax1x.com/2021/03/22/6okCB4.png" alt="1" style="zoom:60%;" />

其中，单词节点$i$的Value, Query以及Key都是通过单词节点$i$的embedding $h_{i}$通过映射得到的。给定维度$d_{k}$和$d_{v}$按照如下方式定义：
$$
\mathbf{q}_{i}=W^{Q} \mathbf{h}_{i}, \quad \mathbf{k}_{i}=W^{K} \mathbf{h}_{i}, \quad \mathbf{v}_{i}=W^{V} \mathbf{h}_{i}
$$
其中$W^{V}$为$\left(d_{\mathrm{v}} \times d_{\mathrm{h}}\right)$的矩阵，$W^{Q}$与$W^{Q}$为$\left(d_{\mathrm{k}} \times d_{\mathrm{h}}\right)$的矩阵。通过Queries与Keys，我们可以计算节点$i$的Query $\mathbf{q}\_{i}$与节点$j$的Key $\mathbf{k}\_{j}$的关联性$u_{i j} \in \mathbb{R}$：
$$
u_{i j}=\left\{\begin{array}{ll}
\frac{\mathbf{q}\_{i}^{T} \mathbf{k}\_{j}}{\sqrt{d_{k}}} & \text { if } i \text { adjacent to } j \\
-\infty & \text { otherwise }
\end{array}\right.
$$
通过节点关联性$u_{i j}$，可以进一步使用sofemax计算Attention Weights $a_{i j} \in[0,1]$：
$$
a_{i j}=\frac{e^{u_{i j}}}{\sum_{j^{\prime}} e^{u_{i j^{\prime}}}}
$$
最终，单词节点$i$收到的Attetion信息组合为：
$$
\mathbf{h}_{i}^{\prime}=\sum_{j} a_{i j} \mathbf{v}_{j}
$$

### 3.2.2 Self-Attention的细节介绍

在本节中，我们介绍Self-Attention的细节计算过程。

首先，我们先介绍如何使用向量来计算Self-Attention，然后再来看如何使用矩阵来实现Self-Attention。

1. 第一步：从每个encdoer的输入向量（例如每个单词的embedding）中创建三个向量。对于每个单词，我们创建一个Query向量，一个Key向量和一个Value向量。 通过将embedding乘以我们在训练过程中训练的三个矩阵来创建这些向量。例如，$X_{1}$乘以$W^{Q}$权重矩阵会产生$q_{1}$，即与该单词相关联的Query向量。 我们最终为输入句子中的每个单词创建一个Query，一个Key和一个Value向量。这些新向量的维数一般小于embedding向量的维数。

   <img src="https://jalammar.github.io/images/t/transformer_self_attention_vectors.png" alt="1" style="zoom:70%;" />

2. 第二步：计算分数。 假设我们正在计算此例中第一个单词“Thinking”的Self-Attention。 我们需要根据该单词对输入句子的每个单词进行评分。分数决定了当我们在某个位置对单词进行编码时，将注意力集中在输入句子的其他部分上的程度。具体来说，通过计算Query向量与我们需要计算分数的各个单词的Key向量的点积来计算分数。 因此，如果我们正在计算位置#1的Self-Attention，则第一个分数将是$q_{1}$和$k_{1}$的点积。 第二个f分数将是$q_{1}$和$k_{2}$的点积。

   <img src="https://jalammar.github.io/images/t/transformer_self_attention_score.png" alt="1" style="zoom:90%;" />

   

3. 第三步：将分数除以$(\sqrt{d_k})$（一般中使用Key向量的维数$d_k$的平方根。这将使得梯度更稳定。可以使用其他值）。

4. 第四步：将结果通过softmax操作进行传递。 Softmax对分数进行归一化，使所有分数均为正且和为1。这个softmax分数确定每个单词在当前位置将被表达多少。 显然，当前位置的单词与当自身的softmax得分最高，但是也与单词“Machines”存在一定的关联关系。

   <img src="https://jalammar.github.io/images/t/self-attention_softmax.png" alt="1" style="zoom:70%;" />

5. 第五步：将每个Value向量乘以softmax分数。 这样做是为了保持我们要关注的单词的Value向量完整，并忽略无关的单词（例如，将它们乘以0.001之类的小数字）。

6. 第六步：对加权值向量求和，在当前位置（第一个单词“Thinking”）产生Self-Attention的输出。

   <img src="https://jalammar.github.io/images/t/self-attention-output.png" alt="1" style="zoom:80%;" />

   

通过以上六个步骤就完成了对单词“Thinking”的Self-Attention的计算。 生成的向量$z_{1}$是我们可以输入到前馈神经网络的向量。 但是，在实际实现中，我们可以把上面的向量计算变成矩阵的形式，从而一次计算出所有时刻的输出（同时计算多个单词的向量$z$），这样的矩阵运算可以充分利用硬件资源(包括一些软件的优化)，以加快处理速度，从而效率更高。具体过程如下：

- 首先是计算Query，Key和Value矩阵。 为此，我们将每个单词的embedding打包到矩阵$X$中，然后将其乘以我们在训练过程中训练的三个权重矩阵（$W^{Q}$，$W^{K}$，$W^{V}$）。$X$矩阵中的每一行对应于输入句子中的一个单词。

  <img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation.png" alt="1" style="zoom:70%;" />

  

- 最后，我们可以将步骤2到6压缩成一个公式，以计算Self-Attention的输出：

  <img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" alt="1" style="zoom:70%;" />
  
  

## 3.3 Soft vs Hard Attention

在 [show, attend and tell](http://proceedings.mlr.press/v37/xuc15.pdf) 论文中，对图像应用Attention机制来生成图像标题。图像首先由CNN编码以提取特征。然后LSTM解码器使用卷积特征逐个产生描述图像的单词，通过Attention学习权重。下图中对Attention权重的可视化显示了模型在输出特定的单词时所关注的图像区域。

<img src="https://lilianweng.github.io/lil-log/assets/images/xu2015-fig6b.png" alt="1" style="zoom:40%;" />

这篇文章根据Attention是关注整个图像还是仅关注图像的一个区域（patch），首次提出了Soft Attention与Hard Attention之间的区别：

- Soft Attention：是一种全局的attention，其中权重被softly地放在源图像的所有区域上，输出Attention分布的概率值；基本上与[Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)中的Attention类型相同。
  - 优点：模型是光滑的和可微的。
  - 缺点：当源输入很大时，成本很高。
- Hard Attention：一次只关注图像的一个区域，输出one-hot向量编码。
  - 优点：推理时计算量少。
  - 缺点：模型是不可微的，需要更复杂的技术，如方差缩减或使用强化学习来训练。（[Luong, et al., 2015](https://arxiv.org/abs/1508.04025)）

注意，在机器学习中soft 常常表示可微分，比如sigmoid和softmax机制；而hard常常表示不可微分。为了进一步阐述二者的区别，以翻译句子为例：“我是小明 --> I am XiaoMing”。对于 Hard Attention而言，在第$1$时刻翻译时，只关注“我”这个词，我们翻译得到“I”，在第$2$时刻翻译时，关注“是”这个词，翻译结果为“am”，以此直到$t$时刻结束。 它采用one-hot编码的方式对位置进行标记，比如第$1$时刻，编号信息就是$[1,0,0...]$， 第$2$时刻，编码信息就是$[0, 1, 0, ...]$， 以此类推。而对于Soft Attention 而言，在第$1$时刻翻译时， “我是小明” 三个单词都对 “I” 做出了贡献，只不过贡献有大小之分，也就是说，虽然“我”这个词很重要，但是我们也不能遗漏其他词所带来的信息。

## 3.4 Global vs Local Attention

[Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)提出了Global Attention和Local Attention。Global Attention类似于Soft Attention，而Local Attention将Hard Attention和Soft Attention进行结合，来对Hard Attention进行改进，从而使其可微。二者区别如下图所示：图中的五个蓝框表示encoder，两个红框表示decoder。对于左边的Global Attention来说，每个encoder都与上下文向量$c_{t}$相连，即上下文向量与所有的encoder输入状态都进行关联，所以名为global；对于右边的Local Attention来说，模型首先预测当前目标$h_{t}$的匹配位置（aligned position），然后使用围绕该匹配位置的选取一定大小的窗口，窗口内所有的encoder都用来计算上下文向量$c_{t}$，即上下文向量只关联右边模型里中间三个encoder的输出。

<img src="https://lilianweng.github.io/lil-log/assets/images/luong2015-fig2-3.png" alt="1" style="zoom:40%;" />

# 四、Attention相关经典模型

4.1 Neural Turing Machines

4.2 Pointer Network

4.3 Transformer

4.4 SNAIL



## 参考
1. [Lil's Log](https://lilianweng.github.io/lil-log/)
2. [Jay Alammar' Blog](https://jalammar.github.io/)




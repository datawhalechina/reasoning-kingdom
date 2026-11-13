# 第4章：流形假设——高维数据的隐秩序

> 你以为数据是随机散布在高维空间的。它不是。它被挤在一个薄薄的曲面上。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、维度的荒谬

先做一个思想实验。

在线段上随机撒 100 个点。平均而言，相邻两点之间的距离约为线段长度的 1/100——密集，采样充分。

现在把这 100 个点放进一个正方形。为了保持同样的采样密度，你需要 $100^2 = 10000$ 个点。

放进一个立方体呢？需要 $100^3 = 1000000$ 个点。

这就是**维度诅咒**（Curse of Dimensionality）——Bellman 在 1957 年命名这个现象时，大概没想到它会缠住整个机器学习领域。

:::details 维度诅咒：为什么高维空间里一切都稀疏？
**维度诅咒**说的是：随着数据维度的增长，以同等密度覆盖空间需要指数级增长的样本量。

一个直观的例子：
- 1 维线段（0 到 1）上的 100 个点：平均间距 = 0.01——仍然相当密集
- 要在 2 维正方形里保持同样的密度，你需要 100² = 10,000 个点
- 10 维超立方体需要 100¹⁰ = 10²⁰ 个点——比可观测宇宙中的原子还多

这不只是“需要更多数据”。更深层的问题是：在高维空间中，**距离这个概念本身崩溃了**——所有点之间的距离都收敛到同一个值，“最近邻”变得无法区分。

这就是为什么许多在低维下表现良好的算法（k-NN、聚类）在高维下表现很差。而流形假设（见下文）恰好提供了一个出路：真实数据并非均匀散布在高维空间中——它具有低维结构。
:::


随着维度增长，以同等密度覆盖空间所需的样本量呈指数爆炸。但这只是问题的表层。真正奇怪的事情发生在几何上。

在一个 $d$ 维单位超球内随机取两个点。它们的距离满足：

$$
\mathbb{E}[\|x - y\|] \to \sqrt{2}, \quad d \to \infty
$$

而距离的方差趋于零。

这意味着：在高维空间中，所有点之间的距离都收敛到同一个值。最近邻和最远邻变得几乎一样远。基于距离的相似性度量——余弦相似度、欧氏距离、$k$-NN 分类——在高维下全部失效，因为“近”失去了辨别力。

如果你随机初始化两个 1000 维向量，它们的余弦相似度几乎必然接近零。不是因为它们语义相反，而是因为在 1000 维中，随机向量几乎必然正交。

**正交意味着什么？** 两个向量正交，即 $\vec{u} \cdot \vec{v} = 0$，几何上意味着它们彼此垂直、互不相关——一个向量在另一个向量方向上的投影为零。在低维中，正交是一种必须刻意构造的特殊关系。在高维中，正交是默认状态，是随机性的必然产物。

这个事实在两个地方向我们发出回响。

**回响一：专家系统的概念空间。** 第 2 章中，专家系统用离散符号表示知识——`"是哺乳动物"`、`"有翅膀"`、`"会飞"`——每个符号是一个原子，概念之间的关系必须显式写进规则。从高维几何的视角重新看这件事：每个离散符号，相当于在某个正交基上放置了一个维度。`"猫"`和`"狗"`在符号系统里没有内在关联；它们是两个独立的 token，正如两个随机单位向量几乎必然正交。专家系统的根本困境——无法自动捕捉概念之间的连续相似性——从这个角度看，是因为它把世界强行塞进了一个**人工正交基**。世界不是正交的，但符号系统假装它是。

::: info 兔狲教授评
高维空间的正交性不是 bug，而是 feature。在极高维度中，随机向量几乎必然正交——这意味着它们天然解耦、彼此独立。这种解耦正是高维嵌入学习得以成功的前提：每个维度可以独立刻画不同的语义侧面，而不必像低维空间那样“黏在一起”。

专家系统的问题恰恰在于：它**刻意制造**了这种正交性——但用的是极其低效的维度数量。它把每个概念硬编码为一个正交基向量，创造出一种有限维的、人工的、离散的正交表示。这种表示在概念上干净，但表示能力极其低下——它无法捕捉概念之间的连续相似性，也无法利用高维空间天然正交性所提供的“免费午餐”。

真正的高维嵌入（例如词向量）之所以强大，是因为它们让数据自己决定如何利用高维正交性，同时保持语义连续性。训练并不强加正交性；它让统计约束把向量推到流形上——这个曲面既不是完全正交的随机沙漠，也不是完全相关的低维沼泽，而是一个精心折叠、语义丰富的曲面。
:::

**回响二：词向量的类比算术。** 第 3 章中，我们看到 $\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$ 这个漂亮的等式，并追问：这是推理，还是统计巧合？现在，以高维正交性为背景，我们可以更精确地说出它的限度：词向量之所以能做这个算术，恰恰是因为训练数据把它们推离了随机初始化的正交状态——语料中的共现约束把随机向量拉进了一个低维语义流形。类比算术之所以有效，功劳在于流形结构，而不在于算法的智慧。而当类比失败时（`doctor - man + woman ≈ nurse`，而非女医生），原因是训练语料的偏见扭曲了流形——语义空间在那个区域的曲率，不是“客观概念”的曲率，而是“语料统计偏见”的曲率。

这两个回响指向同一个问题：如果随机的高维空间是一片正交的沙漠，那么“有意义的数据”之所以可学习，恰恰是因为数据不是随机的——有某种东西把它压缩进了低维结构。这就是流形假设要讨论的事情。

<figure>
<p><img src="/figures/ch04_fig1_curse_of_dimensionality.png" alt="维度诅咒：随维度增长，随机点对距离分布的坍塌" /></p>
<figcaption>维度诅咒：随维度增长，随机点对距离分布的坍塌</figcaption>
</figure>

*图1：左图，随机点对在 2、10、100、1000 维空间中的距离分布。维度越高，分布越集中，方差 → 0——“距离”失去区分能力。右图，1000 维随机向量两两余弦相似度的直方图，几乎全部堆积在零附近。*

<div class="center">

------------------------------------------------------------------------

</div>

这给机器学习提出了一个根本问题：如果数据真的是高维随机的，那么任何基于距离或相似度的算法——包括神经网络——都不应该能工作。

但它们工作了。

这里有两种可能的解释：

第一，这些算法以某种我们尚未完全理解的方式，绕过了高维几何的限制。

第二，数据本身不是高维随机的。

第二种解释是对的。

<div class="center">

------------------------------------------------------------------------

</div>

## 停顿一下

等一下——在接受“第二种解释是对的”这个结论之前，让我们先慢下来。

为什么自然数据会聚集在低维流形上？这是一个需要解释的事实，而不是公理。

一种答案是：物理定律约束了可能状态的空间——人脸的变化受骨骼、肌肉和光照物理的支配，所以可能的人脸远少于可能的像素组合。

但这个答案偷偷塞进了一个假设：**世界是有规律的，而且这些规律是低维的。**

这个假设为什么应该成立？如果世界真正的规律是高维的、混沌的，流形假设就会崩溃——机器学习就不应该有效。

那它为什么事实上有效？是因为物理定律真的是低维的，还是因为我们恰好只在“流形假设成立的那些问题”上测试了它？

先把这个问题放在一边。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、流形假设：约束造就曲面

**流形假设**（Manifold Hypothesis）的核心主张是：

> 自然数据（图像、语言、声音）生活在高维空间中，但实际上聚集在一个光滑曲面（流形）附近，其内在维度远低于它所处空间的环境维度。

“流形”是一个数学术语，意思是局部看起来像欧氏空间的拓扑结构。嵌入在三维空间中的二维曲面就是一个流形——地球表面是流形，莫比乌斯带是流形，一根弯曲管道的表面是流形。

:::details 流形：用地球表面来理解
**流形**是一种几何对象，它的定义是：局部看起来“平坦”（像普通的欧氏平面），但整体可以是弯曲的。

最好的例子：**地球表面**。站在地球上的任意一点，只看脚下很小的一块，它看起来和平地没什么区别——这就是“局部像欧氏空间”。但整体来看，地球表面是球形的，不是平面。地球表面是一个二维流形，嵌入在三维空间中。

在机器学习语境中：
- 49152 维的人脸图像空间 = 三维空间的类比
- 人脸真实所在的几十维曲面 = 地球表面的类比

**流形假设为什么重要？** 因为它解释了机器学习为什么有效：数据不是高维随机的，而是聚集在低维曲面上，算法实际上是在学习这个低维曲面的结构。
:::


用一个具体的例子来建立直觉：

一张 $128 \times 128$ 的彩色人脸图像，作为向量，维度是 $128 \times 128 \times 3 = 49152$。在这个空间中，可能的像素组合有 $256^{49152}$ 种——比宇宙中的原子数多出无数个数量级。

但“真实的人脸”只占据这个空间中极小的一片区域。哪些参数决定了一张真实的人脸？

- 身份（谁的脸）

- 姿态（头转了多少度，仰角多少）

- 光照（光从哪里来，强度多大）

- 表情（在微笑？在皱眉？）

- 年龄

大约十几个到几十个参数就能描述一张人脸的大部分变化。这意味着人脸图像的内在维度是几十的量级——而不是 49,152。

图像数据被挤在一个嵌入于 49,152 维空间中、约几十维的流形上。

语言数据也是如此。“合法的英语句子”只占所有可能词序列的一小部分。“合理的语义”在所有合法英语句子中占的比例更小。每一层约束，都是对流形的一次折叠和压缩。

<figure>
<p><img src="/figures/ch04_fig2_manifold_embedding.png" alt="流形嵌入：低维曲面在高维空间中的结构" /></p>
<figcaption>流形嵌入：低维曲面在高维空间中的结构</figcaption>
</figure>

*图2：左图，一个二维流形（Swiss Roll 数据集）嵌入在三维空间中——三维坐标看起来复杂，但数据沿两个内在维度（展开后的 u、v 坐标）平滑分布。右图，对同一数据集做流形学习（UMAP），恢复了内在的二维结构。颜色编码内在坐标的连续性——局部邻居在高维空间中仍然是邻居。*

<div class="center">

------------------------------------------------------------------------

</div>

为什么自然数据会聚集在流形上？这不是巧合——它是**生成过程的必然结果**。

任何数据都由某个生成过程产生。人脸图像由生物过程产生——DNA、骨骼结构、皮肤纹理；这些参数是有限的，并且连续变化。语言由语法规则和语义约束产生。音乐由物理（声波）和文化惯例产生。

生成过程的**约束**就是流形的**曲率**。约束越强，流形越低维、越弯曲。

这是一个深刻的事实：数据的内在维度不是数据固有的属性，而是**生成它的世界结构**的投影。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、训练是折叠

现在我们到了这一章最重要的部分。

通常我们说神经网络在“学习”——学习识别猫，学习翻译语言，学习预测下一个词。这个说法很危险。“学习”暗示了某种理解，某种把知识内化的过程。

更准确的说法是：**神经网络在做流形变换。**

展开一个自动编码器的工作原理：

$$
\text{Input } x \in \mathbb{R}^D \xrightarrow{\text{Encoder } f} \text{Latent code } z \in \mathbb{R}^d \xrightarrow{\text{Decoder } g} \text{Reconstruction } \hat{x} \in \mathbb{R}^D
$$

其中 $d \ll D$。训练目标是最小化重建误差 $\|x - \hat{x}\|^2$。

这里发生了什么？编码器 $f$ 学会把高维输入映射到低维潜在空间 $z$，而这个低维空间恰好就是数据流形的坐标系。解码器 $g$ 学会从流形坐标重建高维输入。

换句话说，编码器学到的是流形的**参数化**——它找到一组能够描述流形上点的坐标。解码器学到的是这些坐标与原始高维表示之间的映射。

不只是自动编码器。分类网络也在做同样的事。当你训练一个 ResNet 识别图像类别时，最后一个全连接层之前的特征向量，正是网络学到的流形坐标——只不过这个坐标系被刻意设计成“对分类有用”。

反向传播就是调整这个坐标系的过程。每一次梯度更新都改变网络把高维输入映射到低维潜在空间的方式——也就是改变流形的参数化。

**训练的本质，是把过去见过的数据的流形结构折叠进参数的曲率中。**

“折叠”这个词很重要。不是“存储”，不是“记忆”——是折叠。训练数据不是逐条存储在网络中的——网络没有足够的参数来存储训练集的每一个样本。实际发生的是：数据的统计结构、数据所在流形的几何特征，被压缩进网络权重的配置中。

$10^6$ 张人脸图像，每张 49,152 维。ResNet-50 大约有 $2.5 \times 10^7$ 个参数。用 2500 万个数字，它记住了由 492 亿个像素值所描述的流形结构。压缩比约为 200:1。

这不是学习。这是拓扑压缩。

<figure>
<p><img src="/figures/ch04_fig3_compression_bottleneck.png" alt="训练：从高维数据到压缩流形参数的映射" /></p>
<figcaption>训练：从高维数据到压缩流形参数的映射</figcaption>
</figure>

*图3：上方，编码器将高维数据（散布在二维流形上的三维点云）逐步映射到一维潜在编码。下方，训练过程中潜在空间的演化——随着 epoch 增加，潜在表示从混乱走向结构化。颜色代表数据在原始流形上的连续坐标；颜色连续意味着局部结构被保留。*

<div class="center">

------------------------------------------------------------------------

</div>

这里有一个值得停下来思考的深层问题：参数中存储的是什么？

一种直觉是：参数存储的是“知识”——猫长什么样、语法规则、物理定律。

但更准确的说法是：参数存储的是**训练数据流形结构的压缩编码**。

一旦你知道了这一点，就能理解为什么预训练有效——同一个流形（比如“自然图像”）可以被多个任务共享，预训练中学到的流形参数化可以迁移。

你也能理解为什么微调有时会改变模型行为，有时不会——这取决于你新任务的流形是否是预训练流形的子集。

于是你开始理解为什么大模型会产生幻觉——第四幕会讲这一点。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、压缩的账单

任何压缩都有代价。这不是经验观察，而是信息论定理。

Shannon 的信源编码定理告诉我们：无损压缩的极限是信源的熵。超过这个极限，你必然会丢失信息。神经网络的压缩远远超过这个极限，所以它必然在丢失信息。

丢失的是什么？

**第一笔账：流形外的点**

训练结束后，网络的参数编码了训练数据所在的流形。当测试时它遇到一个不在这条流形上的点——来自不同分布的图像、从未见过的语言现象、反事实推理场景——网络会怎么做？

它没有“这个点不在我的流形上”的警报。它只能把这个点强行拖到最近的流形区域，然后按照那个区域的逻辑回答。

结果：**自信的错误**。不是“我不知道”，而是“它看起来像 X，所以我就回答 X”——即使 X 完全错误。

这就是分布偏移（Distribution Shift）问题的几何根源：模型学到的流形与真实世界的流形在训练分布之外不一致。偏移越大，误差越大——但置信度保持不变，因为置信度是流形坐标的函数，而不是“这个点离流形有多远”的函数。

**第二笔账：流形内部的等距性**

压缩时，流形的某些区域被“拉伸”，另一些被“挤压”。

训练数据密集的地方，网络学到的参数化分辨率高——相邻输入映射到相邻的潜在编码，局部几何被很好地保留。

训练数据稀疏的地方，网络参数化的分辨率低——相邻输入可能被映射到潜在空间中相距很远的位置，或者相距很远的输入可能被映射到相邻位置。

这意味着，网络在训练数据密集处泛化好，在稀疏处泛化差——不是随机的差，而是有规律的差：稀疏区域的流形结构被“脑补”出来，填充进去的是训练数据中最常见的模式，而不是那个区域的真实结构。

这种脑补填充的结果就是幻觉（hallucination）。

**第三笔账：Tishby 的信息瓶颈**

2017 年，Naftali Tishby 为这个过程提出了一个理论框架——**信息瓶颈**（Information Bottleneck）。

训练神经网络本质上是一个权衡：

$$
\min_{z} \left[ I(X; Z) - \beta \cdot I(Z; Y) \right]
$$

其中 $X$ 是输入，$Y$ 是标签，$Z$ 是中间表示，$I(\cdot; \cdot)$ 是互信息，$\beta$ 是权衡参数。

第一项 $I(X; Z)$ 衡量中间表示 $Z$ 保留了多少输入信息——越小越好，意味着压缩得越多。

第二项 $I(Z; Y)$ 衡量 $Z$ 保留了多少关于标签的信息——越大越好，意味着对任务越有用。

这个权衡的最优解是一个“充分统计量”——它保留预测标签所需的全部信息，丢弃所有冗余信息。

Tishby 的主张是：训练分两个阶段发生——先是“拟合”阶段，随着网络学习捕获与标签相关的信息，互信息 $I(Z; Y)$ 迅速上升；然后是“压缩”阶段，随着网络开始丢弃与标签无关的信息，$I(X; Z)$ 开始下降。

这个理论很漂亮。但它也有争议——后续研究（Saxe et al., 2018）发现，压缩阶段在许多情况下并不出现，这取决于激活函数的选择。信息瓶颈是一个有价值的分析框架，但不是完整的理论 `-> [Tishby & Schwartz-Ziv, 2017, arXiv:1703.00810]` `-> [Saxe et al., 2018, arXiv:1812.09881]`。

**最终的账**

把三笔账加起来，流形压缩的代价是：

1. 分布内：泛化好，置信度准确

2. 分布边缘：泛化开始退化，但置信度仍然很高（危险区）

3. 分布外：泛化差，但置信度仍可能非常高（最危险区）

网络不知道自己正站在流形的边缘。它只知道当前输入映射到了潜在空间中的某个坐标，然后按照那个坐标的逻辑输出。没有内置的“不确定性传感器”告诉它：你现在正在对一个你从未真正理解的区域进行推断。

这不是 bug；这是流形压缩不可避免的结构性后果。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、压缩的极限：流形能被压到多小？

流形假设说高维数据被挤压到低维流形上。那么“低”到底有多低？压缩能走多远？

这是一个信息论问题，而不仅仅是几何问题。

**速率-失真理论给出一个下界**

Shannon 的速率-失真理论告诉我们：要以失真 $D$ 重建一个信号，至少需要 $R(D)$ 比特的信息。这个 $R(D)$ 是一个**信息论下界**——没有任何压缩算法能突破它。

对于流形上的数据：

$$
R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(x, \hat{x})] \leq D} I(X; \hat{X})
$$

当你把神经网络隐藏层的维度从 $d$ 压缩到 $k$ 时，你实际做的是有损压缩——原始流形的某些细节不可避免地丢失了。

**内在维度**

流形的**内在维度**是描述它所需的最少坐标数。这个量可以从两两距离的分布中估计：

$$
d_{\text{int}} = -\frac{\log n}{\log (r / R)}
$$

其中 $r$ 是近邻半径，$R$ 是最大半径。

实验结果令人惊讶：ImageNet 图像的内在维度约为 40，远小于原始像素维度（$224 \times 224 \times 3 \approx 150{,}000$）。语言模型表示空间的内在维度约为 13–20，远小于隐藏维度（768 或更高）。

这表明真实世界的数据确实被挤压到了极低维的流形上。

**但存在一个硬下界**

压缩不是无限的。一旦压缩到内在维度以下，失真就会急剧上升——你开始丢失流形自身的结构，而不仅仅是噪声。

这就是为什么知识蒸馏（把大模型压缩成小模型）存在性能悬崖：超过某个压缩比，模型不是“差一点”，而是“彻底崩溃”。速率-失真曲线在拐点之后急剧下降。

这个硬下界，以及随机化如何突破确定性压缩的极限，是第 11 章的主题。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、一个小小的停顿

让我盘点一下这一章做了什么。

在高维空间中，随机数据无法被学习——距离失去意义，采样需要指数级多的数据点。机器学习之所以有效，不是因为克服了高维性，而是因为数据本身不是高维随机的。自然数据聚集在低维流形上，这是生成数据的物理过程、生物约束和语法规则共同作用的结果。

训练神经网络，就是把训练数据的流形结构压缩进参数的过程。不是存储——是压缩。压缩比在 100:1 到 1000:1 的量级。

压缩有不可避免的代价：流形外的点被强行投影，稀疏区域的结构被脑补，在分布偏移下置信度与准确率脱钩。这些代价不是工程缺陷；它们是信息论约束的结构性后果。

这引出了第 5 章的问题：如果模型学到的是流形结构而不是因果规律，那么它的行为为什么会在分布之外系统性地失败？统计相关性与真正的推理之间的差别究竟在哪里？

<div class="center">

------------------------------------------------------------------------

</div>


> 流形告诉我们数据在哪里，但没有告诉我们模型是否学到了真正的规律。下一章，我们将看到统计拟合如何成为推理的陷阱。

## 悬而未决

- 神经网络学到的流形，与真实世界数据的流形，究竟有多“吻合”？我们还没有好的工具来测量这种吻合程度。

- 流形的内在维度可以被估计吗？有哪些方法，各自假设了什么？在 LLM 的激活空间中，内在维度是多少？

- “压缩”丢失的信息中，是否包含推理所需的结构？还是说推理能力恰好存在于被保留的部分中？

- 如果用完全随机的数据训练一个神经网络，它会学到什么样的“流形”？这个思想实验的答案会告诉我们流形假设有多必要。

- 信息瓶颈的两阶段理论（拟合 + 压缩）在实验上仍有争议。如果压缩阶段不存在，该如何解释神经网络的泛化？

- **知识的流形结构**：知识图谱是离散的符号流形，词向量是连续的语义流形。这两种流形如何**相互映射、相互验证**？是否存在统一的“知识流形”理论？

- **多层次的流形**：物理系统（生命）的相空间流形、符号系统（知识）的逻辑空间流形、统计系统（学习）的表示空间流形——这些不同层次的流形之间是否存在**同构或映射关系**？

<div class="center">

------------------------------------------------------------------------

</div>

## 动手试试：压缩一个流形，然后找到它的账单

这一章的核心主张是：压缩有代价，而且代价发生的位置不是随机的——它有规律。你将用一个最小的自动编码器亲手验证这一点——不是为了得到一个好模型，而是为了精确指出压缩究竟在哪里撒了谎。

<div class="center">

------------------------------------------------------------------------

</div>

### 第一步：生成你的流形数据

不要使用真实数据集——干扰太多。从一个**你完全理解其生成过程**的流形开始。

生成以下之一：

**选项 A：嵌入三维空间的二维圆环**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 2000  # number of sample points

# Uniformly sample the two angular parameters of the torus
t = np.random.uniform(0, 2 * np.pi, N)   # angle around the main axis
phi = np.random.uniform(0, 2 * np.pi, N) # angle around the tube cross-section

r = 3.0    # main ring radius
tube = 1.0 # tube radius

# Torus parametric equations: embedded in 3D space
x = (r + tube * np.cos(phi)) * np.cos(t)
y = (r + tube * np.cos(phi)) * np.sin(t)
z = tube * np.sin(phi)

# Add a small amount of Gaussian noise to simulate measurement error
noise_std = 0.05
x += np.random.normal(0, noise_std, N)
y += np.random.normal(0, noise_std, N)
z += np.random.normal(0, noise_std, N)

# Stack the three coordinates into a data matrix of shape=(N, 3)
data_torus = np.stack([x, y, z], axis=1)
# Save intrinsic coordinates for later coloring (t is the main angle, phi is the tube angle)
intrinsic_t = t
intrinsic_phi = phi

# Visualize (colored by the t value)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=t, cmap='hsv', s=2, alpha=0.6)
plt.colorbar(sc, label='intrinsic coordinate t')
ax.set_title('Torus manifold (Option A)')
plt.tight_layout()
plt.show()
```

**选项 B：Swiss Roll（本章图示中的数据集）**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

np.random.seed(42)
N = 2000  # number of sample points

# Generate the standard Swiss Roll with sklearn
# t is the intrinsic coordinate (unfold angle), z is the intrinsic coordinate along the height
data_swiss, t_swiss = make_swiss_roll(n_samples=N, noise=0.1, random_state=42)
# data_swiss has shape=(N, 3); the columns are x, y, z (3D embedding coordinates)

# Or generate it manually (equivalent):
t_manual = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, N)  # unfold angle
x_manual = t_manual * np.cos(t_manual)
z_manual = np.random.uniform(0, 1, N)                        # height (second intrinsic dimension)
y_manual = t_manual * np.sin(t_manual)
# Add a small amount of Gaussian noise
x_manual += np.random.normal(0, 0.1, N)
y_manual += np.random.normal(0, 0.1, N)
z_manual += np.random.normal(0, 0.05, N)

# Visualize the manually generated Swiss Roll (colored by t)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_manual, y_manual, z_manual, c=t_manual, cmap='viridis', s=2)
plt.colorbar(sc, label='intrinsic coordinate t (unfold angle)')
ax.set_title('Swiss Roll manifold (Option B)')
plt.tight_layout()
plt.show()

# Use the sklearn version from here on (more standard)
data = data_swiss          # 3D coordinates, shape=(N, 3)
intrinsic_t = t_swiss      # intrinsic coordinate t, used for coloring
```

选项 B 的内在维度是 2（t 和 z），嵌入在三维空间中。

**你的第一个问题（写代码之前回答）：** 这个数据的“真实内在维度”是多少？如果你用 PCA 降维，你认为需要保留多少个主成分才能保留大部分方差？写下你的预测，之后与实际结果比较。

<div class="center">

------------------------------------------------------------------------

</div>

### 第二步：实现一个最小自动编码器

结构定义：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Autoencoder architecture ---
class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        # Encoder: 3 -> 16 -> d (bottleneck dimension)
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),      # 3D input (embedded data points)
            nn.ReLU(),             # non-linear activation
            nn.Linear(16, bottleneck_dim)  # d-dimensional latent code z
        )
        # Decoder: d -> 16 -> 3 (reconstruct the original coordinates)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)       # 3-dimensional reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)    # encode
        x_hat = self.decoder(z)  # decode
        return x_hat, z

# --- Training function ---
def train_autoencoder(data_tensor, bottleneck_dim, n_epochs=300, lr=1e-3):
    """
    Train the autoencoder until the reconstruction error converges.
    Returns the trained model.
    """
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = Autoencoder(bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # loss function: mean squared error (reconstruction error)

    for epoch in range(n_epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = loss_fn(x_hat, batch)  # reconstruction error
            loss.backward()
            optimizer.step()

    return model

# --- Example: use the Swiss Roll data, starting with bottleneck dimension d=2 ---
# (Assume data was generated in the previous step, shape=(N, 3))
data_tensor = torch.tensor(data, dtype=torch.float32)  # convert to a PyTorch tensor

# Train an autoencoder with bottleneck dimension d=2
model_d2 = train_autoencoder(data_tensor, bottleneck_dim=2)
print("Autoencoder for d=2 trained")
```

训练至重建误差收敛。不需要调超参数——默认设置即可。

<div class="center">

------------------------------------------------------------------------

</div>

### 第三步：系统地改变瓶颈维度，记录账单

用 d = 1, 2, 3, 4, 5 训练，并对每个 d 记录：

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Split the data into train and test sets (80% / 20%)
N = len(data_tensor)
split = int(0.8 * N)
train_data = data_tensor[:split]
test_data  = data_tensor[split:]

results = {}  # stores the results for each bottleneck dimension

for d in [1, 2, 3, 4, 5]:
    # 1. Train the autoencoder
    model = train_autoencoder(train_data, bottleneck_dim=d, n_epochs=300)
    model.eval()

    with torch.no_grad():
        # 2. Compute the average reconstruction error (MSE) on the train set
        train_hat, train_z = model(train_data)
        train_mse = nn.MSELoss()(train_hat, train_data).item()

        # 3. Compute the average reconstruction error on the test set
        test_hat, test_z = model(test_data)
        test_mse = nn.MSELoss()(test_hat, test_data).item()

    results[d] = {'train_mse': train_mse, 'test_mse': test_mse,
                  'train_z': train_z.numpy(), 'model': model}

    # 4. Visualize the latent space (colored by intrinsic coordinate t)
    train_t = intrinsic_t[:split]  # intrinsic coordinates of the train set
    if d == 1:
        # d=1: the latent space is a line, plot a 1D scatter
        fig, ax = plt.subplots(figsize=(8, 2))
        sc = ax.scatter(train_z.numpy()[:, 0],
                        np.zeros(len(train_z)),
                        c=train_t, cmap='viridis', s=5)
        plt.colorbar(sc, label='intrinsic coordinate t')
        ax.set_title(f'd={d} latent space (1D)')
    elif d == 2:
        # d=2: plot a 2D scatter directly
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(train_z.numpy()[:, 0],
                        train_z.numpy()[:, 1],
                        c=train_t, cmap='viridis', s=5)
        plt.colorbar(sc, label='intrinsic coordinate t')
        ax.set_title(f'd={d} latent space (2D)')
    plt.tight_layout()
    plt.show()

    # 5. Sample 20 points from the latent space, decode them, and plot them in 3D
    with torch.no_grad():
        # Uniformly sample the latent space (using the range of the train-set latent codes)
        z_min = train_z.min(dim=0).values
        z_max = train_z.max(dim=0).values
        z_samples = torch.rand(20, d) * (z_max - z_min) + z_min
        decoded_samples = model.decoder(z_samples).numpy()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:split, 0], data[:split, 1], data[:split, 2],
               c='lightblue', s=1, alpha=0.3, label='train data')
    ax.scatter(decoded_samples[:, 0], decoded_samples[:, 1], decoded_samples[:, 2],
               c='red', s=50, zorder=5, label='randomly decoded points')
    ax.set_title(f'd={d} random decoding (randomly sampled latent points)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"d={d}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}")

print("\nComparison across bottleneck dimensions:")
for d, r in results.items():
    print(f"  d={d}: train MSE={r['train_mse']:.4f}, test MSE={r['test_mse']:.4f}")
```

**你的第二个问题：** 当 d=1 时，编码器把一个二维流形压缩到一维。它保留了什么？丢失了什么？潜在空间中点的排列是否保留了原始流形的拓扑（例如圆环循环的关系）？

**你的第三个问题（核心问题）：** 当 d=2 时，如果用原始内在坐标（t 和 φ，或 t 和 z）给潜在空间的点染色，颜色会连续变化吗？如果是，说明什么？如果不是，又说明什么？

<div class="center">

------------------------------------------------------------------------

</div>

### 第四步：看看分布外点会怎样

生成一批**不在你的流形上**的点——例如三维空间中的随机高斯噪声点。

把这批点送进你的自动编码器：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Use the d=2 model for out-of-distribution testing
model = results[2]['model']
model.eval()

# OOD data: sample 100 points from a standard 3D Gaussian distribution
# Note: these points are scattered across the whole 3D space, not restricted to the manifold
ood_data = torch.randn(100, 3)  # standard normal with mean 0 and variance 1

with torch.no_grad():
    # Reconstruct with the autoencoder (encode -> decode)
    ood_reconstructed, ood_z = model(ood_data)

# Compute the reconstruction errors for both datasets
loss_fn = nn.MSELoss(reduction='none')  # compute the error for each sample individually
with torch.no_grad():
    train_hat, _ = model(train_data)
    # MSE of each train sample (averaged over the three dimensions)
    train_errors = loss_fn(train_hat, train_data).mean(dim=1).numpy()
    # MSE of each OOD sample
    ood_errors   = loss_fn(ood_reconstructed, ood_data).mean(dim=1).numpy()

# Compare the reconstruction errors
print(f"Average reconstruction error on train set: {train_errors.mean():.4f} ± {train_errors.std():.4f}")
print(f"Average reconstruction error on OOD points: {ood_errors.mean():.4f} ± {ood_errors.std():.4f}")

# Visualize: plot the reconstruction of the OOD points in 3D
# Observe where they land (whether they are "pulled" toward the manifold)
fig = plt.figure(figsize=(10, 5))

# Left panel: the original OOD points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:500, 0], data[:500, 1], data[:500, 2],
            c='lightblue', s=2, alpha=0.3, label='train manifold')
ax1.scatter(ood_data[:, 0], ood_data[:, 1], ood_data[:, 2],
            c='red', s=30, label='OOD points (original)')
ax1.set_title('OOD points (scattered in 3D space)')
ax1.legend(fontsize=8)

# Right panel: OOD points after reconstruction by the autoencoder
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(data[:500, 0], data[:500, 1], data[:500, 2],
            c='lightblue', s=2, alpha=0.3, label='train manifold')
ax2.scatter(ood_reconstructed[:, 0].numpy(),
            ood_reconstructed[:, 1].numpy(),
            ood_reconstructed[:, 2].numpy(),
            c='orange', s=30, label='OOD points (reconstructed)')
ax2.set_title('OOD points after reconstruction — "pulled" toward the manifold')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

**你的第四个问题：** 经过自动编码器之后，分布外的点被“拉”到了哪里？它们的重建结果是随机的，还是系统性地落在训练流形的某个区域？

这正是本章第 4 节所说的：“网络没有‘这个点不在我的流形上’的警报。它只能把这个点强行拖到最近的流形区域。”你刚刚亲手观察到了这个过程。

<div class="center">

------------------------------------------------------------------------

</div>

### 第五步：估计你的流形的内在维度

回到第一步——现在做一次数据驱动的内在维度估计：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Run PCA on the training data (a numpy array, not a tensor)
train_np = data[:split]  # shape=(N_train, 3)

pca = PCA()               # keep all components (no limit)
pca.fit(train_np)

# Variance ratio explained by each principal component
explained_ratio = pca.explained_variance_ratio_
# Cumulative explained variance curve
cumulative_ratio = np.cumsum(explained_ratio)

# Plot the cumulative variance curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: variance explained by each principal component
axes[0].bar(range(1, len(explained_ratio) + 1), explained_ratio, color='steelblue')
axes[0].set_xlabel('Principal component index')
axes[0].set_ylabel('Explained variance ratio')
axes[0].set_title('Variance explained by each principal component')

# Right panel: cumulative explained variance curve, find the "elbow"
axes[1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio,
             marker='o', color='tomato')
axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of principal components')
axes[1].set_ylabel('Cumulative explained variance ratio')
axes[1].set_title('Cumulative variance curve (find the elbow)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Find how many principal components are needed to explain 95% of the variance
n_components_95 = np.searchsorted(cumulative_ratio, 0.95) + 1
print(f"Principal components needed for cumulative variance >= 95%: {n_components_95}")
print(f"Variance explained by each component: {explained_ratio.round(4)}")
print(f"Cumulative explained variance: {cumulative_ratio.round(4)}")

# Compare with the autoencoder results
print(f"\nIntrinsic dimension estimated by PCA: {n_components_95}")
print("Test MSE of the autoencoder at each bottleneck dimension:")
for d, r in results.items():
    print(f"  d={d}: test MSE={r['test_mse']:.4f}")
print("(The point where the MSE jumps sharply corresponds to the intrinsic dimension estimated by the autoencoder)")
```

把这个数字与你在第一步写下的预测比较。

**你的第五个问题：** 你用 PCA 估计的内在维度，是否与自动编码器给出的“最小无损瓶颈维度”一致？如果不同，你认为原因是什么——PCA 的线性假设，还是自动编码器的非线性？

<div class="center">

------------------------------------------------------------------------

</div>

### 清单

完成这五步之后，你应该能够回答以下三个问题——并且用你自己的实验数据来回答，而不是用本章的文字：

1. 压缩到流形的内在维度以下，代价是什么？代价出现在哪里？

2. 分布外的点经过自动编码器之后，是如何被“幻觉式处理”的？

3. “训练数据密集的地方泛化好，稀疏的地方泛化差”——在你的实验中，稀疏区域对应流形的哪一部分？那里的重建误差更高吗？

如果你只做一件事，就做第四步。它是最短的实验，也是最能让你直观感受到“幻觉”如何发生的实验。

<div class="center">

------------------------------------------------------------------------

</div>

## 小结：从流形到知识结构——前四章的深层联系

在前四章结束之际，让我们回顾走过的路，以及第 1 章和第 2 章新增内容之间的深层联系。

### 三个层次的推理系统

前四章实际上描述了三个不同层次的推理系统，它们共享一个**元模式**：

1. **物理层（第 1 章新增）**：**生命系统**作为开放系统
   - 通过能量流实现局部熵减
   - 以环境熵增为代价维持秩序
   - 耗散结构：远离平衡态的自组织

2. **符号层（第 2 章新增）**：**知识系统**作为符号结构
   - 通过逻辑规则实现精确推理
   - 以构建成本为代价换来可解释性
   - 知识图谱：符号 AI 的规模化实现

3. **统计层（第 3–4 章）**：**学习系统**作为统计模型
   - 从数据流中学习模式识别
   - 以计算复杂度为代价换来泛化
   - 流形假设：高维数据的低维结构

### 统一视角：结构、代价、流动

这三个层次共享一个**统一框架**：

| 维度 | 物理层（生命） | 符号层（知识） | 统计层（学习） |
|------|----------------|----------------|----------------|
| **核心机制** | 能量流 | 逻辑推理 | 模式识别 |
| **有序形式** | 物理结构秩序 | 符号结构秩序 | 表示空间秩序 |
| **维持代价** | 环境熵 | 构建成本 | 计算复杂度 |
| **外部输入** | 能量 / 物质 | 知识 / 规则 | 数据 / 样本 |
| **结构假设** | 耗散结构 | 封闭世界 | 流形假设 |

### 从离散到连续：表示的演进

前四章也勾勒了**知识表示**的演进：

1. **离散符号**（第 2 章）：If-Then 规则、知识图谱三元组
2. **连续向量**（第 3 章）：词嵌入、语义空间
3. **低维流形**（第 4 章）：内在维度、数据的内部结构

**关键洞见**：知识图谱是**离散的符号流形**，词向量是**连续的语义流形**。两者都在试图捕捉世界的**结构**——只是使用了不同的数学语言。

### 元问题：在多个层次上推理

这引出一个**元问题**：什么是推理？

- **物理答案**：推理是生命用来对抗熵的认知策略
- **符号答案**：推理是符号系统遵循逻辑规则的推导
- **统计答案**：推理是统计模型在流形上的模式匹配

这三个答案**都对，但都不完整**。真正的推理可能是**三者的协同**：
- 物理层提供**存在基础**（为什么需要推理）
- 符号层提供**精确框架**（推理如何形式化）
- 统计层提供**实现机制**（推理如何计算）

## 思考题：边界在哪里？

每个层次都有自己的**边界**：
- 物理层：热力学极限（Landauer 原理）
- 符号层：知识边界（封闭世界假设）
- 统计层：泛化边界（流形假设）

**最终的问题**：当一个 AI 系统“推理”时，它在哪个层次上工作？还是说真正的智能需要一种**超越这些层次的整合**？

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Tishby, N. & Schwartz-Ziv, M. (2017). *Opening the Black Box of Deep Neural Networks via Information*——信息瓶颈理论的深度学习版本，提出两阶段训练假说 `-> [arXiv:1703.00810]`

- Saxe, A. et al. (2018). *On the Information Bottleneck Theory of Deep Learning*——对 Tishby 理论的批评与修正，表明压缩阶段取决于激活函数的选择 `-> [arXiv:1812.09881]`

- Fefferman, C., Mitter, S. & Narayanan, H. (2016). *Testing the Manifold Hypothesis*——对流形假设的数学检验：它何时成立、何时不成立 `-> [arXiv:1204.1423]`

- Bengio, Y., Courville, A. & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives*——表示学习综述，涵盖流形学习与深度表示之间的关系 `-> [arXiv:1206.5538]`

- Bellman, R. (1957). *Dynamic Programming*——维度诅咒的原始出处；Bellman 在优化语境中命名了这一现象

- 本书第 11 章第 5–9 节：Collins 随机化优化器——$O(1)$ 状态压缩与安全压缩比 $c_{\text{opt}} \approx 34$ 的信息论证明与实验（代码见第 11 章附录：`collins_adam.py`）

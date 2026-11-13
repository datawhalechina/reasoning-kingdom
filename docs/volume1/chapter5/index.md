# 第5章：拟合的陷阱——统计相关性不是推理

> 模型见过一百万只猫之后，它知道猫是什么吗？

<div class="center">

------------------------------------------------------------------------

</div>

## 一、一个让人不安的实验

2021年，Bender 等人发表了一篇论文，标题很直白：“论随机鹦鹉的危险”（On the Dangers of Stochastic Parrots）。

这个标题不是比喻，是诊断。

论文的核心论点是：大型语言模型——一个在海量文本上训练、用来预测下一个词的语言模型——无论训练在多少文本上，本质上都是在做**统计模式匹配**。它学会了哪些词序列在训练数据里经常一起出现，然后在生成时复现这些模式。就像一只鹦鹉，听到了足够多的对话之后，能够在合适的时机说出“你好”或“再见”，但它不理解这些词的意义。

这个比喻很刺耳，因为它触及了一个更深的问题：**统计相关性和因果理解之间，有一道鸿沟。**

让我给你看一个具体的例子。

2023年，Hodel 和 West 做了一个简单的测试。他们拿 GPT-3 做字母串类比谜题——这正是 Webb 等人（2023）声称 GPT-3 已经“涌现”出类比推理能力的任务。

原始任务是这样的：

    input: abc -> abd, kji -> ?
    expected output: kjj

这是一个简单的“最后一个字母后移一位”的规律。GPT-3 在这个任务上表现很好。

然后 Hodel 和 West 做了最简单的变体：把字母串的长度从 3 改成 4，或者把字母表稍微打乱。

**GPT-3 的表现立刻崩溃了。**

不是“稍微下降”，是崩溃。准确率从接近 100% 掉到接近随机猜测。而人类在这些变体上的表现几乎没有变化，因为人类理解的是**规律本身**，而不是绑定在特定长度、特定字母表上的表面模式。

这就是本章的核心问题：**当模型通过最小化训练误差来学习时，它实际学到的到底是什么？**

<div class="center">

------------------------------------------------------------------------

</div>

## 二、经验风险最小化的契约

让我们从数学上把这个问题说精确。

监督学习的标准范式是**经验风险最小化**（Empirical Risk Minimization，ERM）。给定训练数据 $(x_1, y_1), \ldots, (x_n, y_n)$，我们选择一个假设 $h$ 来最小化经验风险：

$$
\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)
$$

这里 $\ell$ 是损失函数，衡量预测 $h(x_i)$ 和真实标签 $y_i$ 之间的差距。

:::details 经验风险最小化（ERM）：训练模型在做什么？
**经验风险最小化**是大多数机器学习背后的引擎，用大白话说就是：**在训练数据上，让模型的预测误差尽可能小**。

几个关键术语：
- **假设 $h$**：模型本身（比如一个神经网络）。给定输入 $x$，它输出预测 $h(x)$
- **损失函数 $\ell$**：给“预测有多差”打分的函数。标签是“狗”却预测“猫”——损失很大；预测正确——损失很小
- **经验风险 $\hat{R}(h)$**：所有训练样本上的平均损失——最小化它，就是“让模型在训练集上尽可能多答对”

**核心问题**：最小化训练误差 ≠ 真正理解规律。模型可能只是在“背答案”（过拟合），也可能学到了表面模式而不是底层机制。这正是本章要追问的：ERM 学到的到底是什么？
:::


ERM 的理论保证来自一个简单的直觉：如果训练数据是从某个分布 $\mathcal{D}$ 上独立同分布（i.i.d.）采样得到的，那么当 $n$ 足够大时，经验风险 $\hat{R}(h)$ 会收敛到真实风险：

$$
R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}} [\ell(h(x), y)]
$$

这个收敛有多快？统计学习理论告诉我们：对于有限的假设空间，收敛速度是 $O(1/\sqrt{n})$。对于无限假设空间（比如神经网络），收敛速度取决于假设空间的“复杂度”——VC 维、Rademacher 复杂度，等等。

**但这里隐藏着一个前提：ERM 优化的东西，和我们真正关心的东西，真的是同一件事吗？**

让我把这一点拆开。

ERM 在做什么？它在寻找一个函数 $h$，使得 $h(x)$ 与 $y$ 之间的某种距离（损失）在训练数据上最小。

但“推理”需要什么？推理需要模型捕获 $x$ 和 $y$ 之间的**因果机制**——生成数据的那个底层过程。

这两者不是一回事。

统计学习理论的经典结果——例如 Vapnik 和 Chervonenkis 在 1970 年代建立的框架——讲的是**泛化界**：

$$
R(h) \leq \hat{R}(h) + \text{complexity penalty}(h, n)
$$

这个不等式说的是：如果假设空间不太复杂，那么训练误差低的模型，测试误差也会低。

但请注意其中的假设：**测试数据和训练数据来自同一个分布** $\mathcal{D}$。

在现实中，这个假设几乎从不成立。

你的训练数据是你能收集到的数据——也许来自某一家医院、某一个时间段、某一个人群。你的测试数据是模型部署后遇到的数据——来自不同的医院、不同的季节、不同的人群。

**分布偏移是常态，不是例外。**

而当分布偏移发生时，ERM 学到的那些“在训练数据上有效的相关性”可能完全失效。

<div class="center">

------------------------------------------------------------------------

</div>

## 停顿一下

模型记住了训练集——我们叫它过拟合，这是错误，我们会惩罚它。

但如果模型记住了整个互联网呢？

那我们该叫它什么？

GPT-4 的训练数据大致是人类有史以来写下的所有文本的一个大样本。如果它“记住”了这一切，那和过拟合的本质区别是什么？——是规模，还是别的什么？

还有一个问题：分布偏移发生时，ERM 会失效。但人类在分布偏移下有时也会失效——我们在陌生的文化里判断失误，在新的领域里犯低级错误。

那么，ERM 的缺陷是机器学习独有的，还是属于每一个归纳学习系统？包括你自己？

我们先把这个问题放在桌上。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、捷径学习：当相关性比因果更容易

这里有一个思想实验。

假设你要训练一个模型识别“奶牛”。训练集里有 1000 张奶牛照片：950 张的背景是草地，50 张的背景是沙滩（某个海边农场）。

ERM 会学到什么？

如果模型足够简单，它可能学到：“如果背景是草地，就预测奶牛。”这条规则在训练集上的准确率是 95%——相当不错。

但这条规则捕获了“奶牛”的本质吗？显然没有。把这个模型部署到沙滩农场，它会把每一头奶牛都分类错误。

这就是**捷径学习**——模型利用训练数据中的**虚假相关性**，而不是真正的因果特征。

Geirhos 等人在 2020 年系统地调查了这一现象。他们的结论是：捷径学习在深度学习中无处不在：

- **纹理偏见**：在 ImageNet 上训练的模型更多依据纹理而非形状来分类，而人类恰好相反
- **背景依赖**：目标检测模型把背景统计（比如“船通常出现在水面上”）当作捷径
- **数据集偏见**：情感模型依赖某些高频词（如“terrible”“amazing”），而忽略句子的整体语义

为什么会这样？

**因为 ERM 没有机制去区分“有用的相关性”和“虚假的相关性”。** 只要某个特征在训练数据中与标签相关，ERM 就会用它——不管这种相关性在训练分布之外是否仍然成立。

2023 年，Puli 等人揭示了一个更深层的原因：即使虚假特征（捷径）根本不提供任何额外信息——即使稳定特征已经完全决定了标签——**默认的 ERM 设置（梯度下降 + 交叉熵）仍然偏好捷径。**

为什么？因为梯度下降隐含地最大化分类边界。而在线性可分的情况下，最大边界解往往是那个同时使用稳定特征和捷径的解——即使仅靠稳定特征就能达到零训练误差。

这不是过拟合——训练误差已经是零了。这是**归纳偏置的问题**：梯度下降 + 交叉熵的组合天然偏好某一类解，而当捷径存在时，这类解会过度依赖捷径。

<figure>
<p><img src="/figures/ch05_fig1_gradient_analysis.png" alt="训练过程中捷径特征与稳定特征的梯度演化" /></p>
<figcaption>训练过程中捷径特征与稳定特征的梯度演化</figcaption>
</figure>

*图1：训练过程中，捷径特征的梯度范数迅速上升并保持高位，而稳定特征的梯度持续衰减。模型在优化过程中逐渐被捷径吸引，即使稳定特征已经足以完成任务。*

让我用一个更形式化的框架来说这件事。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、因果图视角：稳定特征与虚假特征

假设数据生成过程可以用一个因果图来描述。这里有三个变量：

- $Y$：标签（比如“这是不是一头奶牛”）
- $X_s$：稳定特征（比如奶牛的形状和纹理）
- $X_c$：捷径特征（比如背景是不是草地）

真实的因果关系是：

$$
Y \to X_s, \quad Y \to X_c
$$

也就是说，标签 $Y$ 同时决定了稳定特征和捷径特征。在训练数据中，由于某种采样偏差，$Y$ 和 $X_c$ 强相关——例如“奶牛”几乎总是出现在草地背景上。

但这种相关性不是因果的。$X_c$ 并不决定 $Y$；是 $Y$ 决定了 $X_c$ 在训练数据中的分布。

ERM 看到的是什么？ERM 看到的是联合分布 $P(X_s, X_c, Y)$，它学到一个预测函数：

$$
\hat{Y} = f(X_s, X_c)
$$

如果 $X_c$ 在训练数据中与 $Y$ 强相关，ERM 就会给 $X_c$ 很大的权重——因为这样能降低训练误差。

问题在于：当测试数据来自另一个分布、其中 $Y$ 与 $X_c$ 的相关性不再成立时（比如奶牛出现在沙滩上），模型的预测就会失败。

这不是模型“不够聪明”。这是 ERM 的结构性限制：**ERM 优化的是 $P(Y|X)$，而不是因果机制 $P(X|Y)$。**

Pearl 的因果阶梯告诉我们，要回答一个干预性问题（“如果我把背景换到沙滩上，模型还能认出奶牛吗？”），你需要的不只是条件概率——你需要一个因果模型。

但 ERM 只能看到观测数据。它无法区分相关与因果。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、随机鹦鹉假说：复读不是理解

现在让我们回到本章开头的问题：大型语言模型是随机鹦鹉吗？

这个问题的要害，不在于模型是否“有意识”或是否“理解”——那些是哲学问题，我们暂且搁置。要害在于：**模型的行为，有多大程度可以用“记忆 + 检索训练数据中的模式”来解释？**

2025 年，Zhao 等人提出了一个决定性的测试：思维链（CoT）推理——一步一步推理并把步骤写下来——究竟是真正的推理，还是训练数据分布的镜像？

他们的实验设计很巧妙。他们构建了一个完全可控的环境 DataAlchemy，从头训练语言模型，然后系统地改变训练数据与测试数据之间的分布差距：

- **任务分布偏移**：训练时见过加法，测试时做乘法
- **长度分布偏移**：训练时见过 3 位数运算，测试时做 5 位数运算
- **格式分布偏移**：训练时见过“step 1, step 2”的格式，测试时换成“first, second”

结果是毁灭性的：**CoT 推理在三种分布偏移下都显著退化。** 模型不是在“推理”，而是在“模式匹配”。它学到的是训练数据中什么样的输入对应什么样的思维链格式，然后在测试时复现那个格式。

当测试分布与训练分布不同时，这种复现就失败了。

这和第一节里 GPT-3 在字母串类比上的崩溃是同一个现象：模型学到的是表面的统计规律，而不是底层的抽象规则。

但这里还有一个更深的问题。

Bender 等人的“随机鹦鹉”批评隐藏着一个假设：如果模型只是在做统计模式匹配，那么它的能力就有上限——它无法超越训练数据的统计结构。

但这个假设对吗？

2023 年，Wei 等人提出了反驳：即使模型只是在做模式匹配，只要训练数据足够大、足够多样，单纯靠模式匹配也可能产生看起来像“推理”的行为。

这是一场关于**涌现**的争论：当模型规模和数据规模越过某个临界点，是否会出现某种质变？

目前的证据是混合的。

一方面，我们确实看到了令人惊讶的能力——比如 GPT-4 在某些推理任务上的表现已经接近人类平均水平。

另一方面，这些能力在分布偏移下的脆弱性表明，它们可能仍然是“复杂的模式匹配”，而不是真正的抽象推理。

决定性的测试是：**模型能否在训练中从未见过的组合上产生正确的行为？**

这就是下一节的主题：把分布偏移当作试金石。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、分布偏移：推理能力的试金石

如果一个模型真的“理解”自己在做什么，那么当输入分布发生变化时，它的性能应该优雅地退化——而不是崩溃。

这是一个可以精确检验的假设。

**分布内（In-Distribution，ID）性能**衡量模型在与训练数据相似的数据上的表现。这就是标准的测试集评估。

**分布外（Out-of-Distribution，OOD）性能**衡量模型在训练分布之外的数据上的表现。这才是真正的泛化能力。

让我给你看几个具体例子，说明分布偏移如何暴露模型的脆弱性。

**例子一：医学图像分割中的捷径**

2024 年，Woodland 等人研究了医学图像分割中深度学习模型的 OOD 检测。他们发现：在肝脏分割上训练的模型，遇到来自不同医院、不同扫描仪的图像时，性能显著下降。

问题不在图像质量——新图像完全没问题。问题在于模型把训练数据中的**设备特异性伪影**学成了捷径。

例如，某一种型号的 CT 扫描仪会在图像的一个角落产生特定的噪声模式。模型学会了利用这个噪声模式来辅助分割——因为在训练数据中，这个噪声模式与肝脏的位置强相关。

但当你换到另一种扫描仪型号时，噪声模式消失了，模型的分割准确率随之崩溃。

**例子二：自然语言理解中的虚假相关**

2023 年，Shuieh 等人系统评估了三种后训练算法（SFT、DPO、KTO）在虚假相关下的鲁棒性。

他们构建了三类任务——数学推理、指令遵循、文档问答——并把不同程度的虚假相关植入数据（10% 对 90%）。

结果是：**所有模型在高虚假相关下都显著退化。** 偏好学习方法（DPO/KTO）在数学推理上相对鲁棒，但在复杂、依赖上下文的任务上，监督微调（SFT）反而表现更好。

这告诉我们什么？没有任何一种训练方法能普遍抵抗捷径学习。最佳策略取决于任务类型和虚假相关的性质。

**例子三：CoT 推理的分布脆弱性**

回到 Zhao 的 DataAlchemy 实验。他们的核心发现是：**CoT 推理是训练数据分布的一面脆弱镜子。**

当任务、长度、格式这三个维度中的任何一个发生偏移，CoT 的有效性都会显著下降。这说明模型学到的不是“如何推理”，而是“训练数据中的推理链长什么样”。

更糟的是，模型在分布偏移下的失败是**系统性的**，不是随机的。它不是偶尔犯错——它在特定类型的输入上一致地失败，因为这些输入触发了训练数据中不存在的模式。

三个例子都指向同一个结论：**分布偏移不是边缘情况，而是核心测试。** 如果一个模型只在分布内表现良好，那么它学到的很可能是统计相关性，而不是因果机制。

<figure>
<p><img src="/figures/ch05_fig2_id_ood_performance.png" alt="不同捷径强度下的 ID 与 OOD 性能" /></p>
<figcaption>不同捷径强度下的 ID 与 OOD 性能</figcaption>
</figure>

*图2：左图：训练数据中的捷径相关性越强，ID 准确率越高，但 OOD 准确率越低。右图：泛化差距（ID − OOD）随捷径强度线性增长；一旦捷径相关性超过 80%，差距就进入危险区（\>20%）。这量化了捷径学习对分布外泛化的破坏。*

<div class="center">

------------------------------------------------------------------------

</div>

## 七、为什么 ERM 的归纳偏置不够

现在我们来到本章最重要的一点。

问题不在 ERM 本身——ERM 是一个合理的学习原则。问题在于：**ERM 搭配标准优化器（梯度下降）和标准损失（交叉熵），产生的归纳偏置不适合学习因果结构。**

让我把这个论证展开。

**归纳偏置**是学习算法中的隐含假设——当多个假设都能拟合训练数据时，它决定了算法会选哪一个。

梯度下降 + 交叉熵的归纳偏置是什么？

在线性可分的情况下，梯度下降收敛到**最大边界解**——使决策边界到最近训练样本的距离最大的那个解。

这通常是好事。最大边界往往意味着更好的泛化——因为它对训练数据的微小扰动更鲁棒。

但当捷径存在时，最大边界解往往是那个**同时使用稳定特征和捷径**的解。

为什么？因为如果你同时使用两个特征，就能把边界推得更远——即使其中一个特征（捷径）在分布外会失效。

Puli 等人 2023 年的研究精确刻画了这一现象。他们证明：在一个简单的线性感知机任务中，即使稳定特征已经完全决定标签，梯度下降仍然会给捷径分配非零权重——因为这样做能最大化边界。

那么，解法是什么？

一个方向是**改变归纳偏置**。例如，不追求最大边界，而是追求**均匀边界**——让所有训练样本到边界的距离尽可能相等。

Puli 的 MARG-CTRL（Margin Control）正是这个思路：通过重塑损失函数，鼓励模型产生均匀边界解，从而减少对捷径的依赖。

另一个方向是**显式地建模因果结构**。这意味着超越纯粹的观测数据——引入干预或反事实推理——这是第六章的主题。

但即使不借助因果推断，更聪明的训练策略也能缓解捷径学习。

<div class="center">

------------------------------------------------------------------------

</div>

## 八、缓解策略：从数据增强到对抗训练

如果我们知道捷径在哪里，能做些什么？

**策略一：数据增强**

最直接的方法：让训练数据更多样，打破虚假相关。

在奶牛的例子中，如果你能收集到足够多的“沙滩上的奶牛”图片，模型就不会过度依赖“草地背景”这个捷径。

但这个方法有两个问题。

第一，你需要知道捷径是什么。在真实场景中，捷径是隐藏的——你不知道模型在利用什么虚假相关。

第二，即使你知道捷径，收集多样化的数据也可能非常昂贵，或者根本不可行。

**策略二：重加权训练样本**

如果某些训练样本“太容易”——模型用捷径就能预测正确——那就降低它们的权重，迫使模型去学更难的样本。

这就是 Li 等人在 2020 年提出的 Tilted ERM 的思路。通过引入一个“倾斜”参数 $t$，调整每个样本的损失权重：

$$
\text{Tilted-ERM}: \quad \min_h \frac{1}{t} \log \left( \frac{1}{n} \sum_{i=1}^n e^{t \cdot \ell(h(x_i), y_i)} \right)
$$

当 $t > 0$ 时，高损失样本（难样本）获得更大权重；当 $t < 0$ 时，低损失样本（易样本）获得更大权重。

通过调节 $t$，你可以在鲁棒性和公平性之间做权衡。

**策略三：对抗训练**

另一个思路：显式地生成“对抗样本”——也就是模型会依赖捷径而失败的那些样本——然后在那些样本上训练。

Sricharan 和 Srivastava 在 2018 年提出：用 GAN 生成模型高置信度分类、但实际上是 OOD 的样本，然后最大化模型在这些样本上的熵（不确定性）。

这迫使模型不要在分布外输入上过度自信，从而减少对捷径的依赖。

**策略四：因果正则化**

如果我们对因果结构有先验知识，就可以把它编码进一个正则项。

例如，如果我们知道某个特征 $X_c$ 是捷径，就可以在损失中加入一个惩罚项，减少模型对 $X_c$ 的依赖：

$$
\min_h \hat{R}(h) + \lambda \cdot \text{Dependence}(h, X_c)
$$

这里 $\text{Dependence}(h, X_c)$ 衡量模型输出对 $X_c$ 有多敏感。它可以用梯度、互信息等方式定义。

但所有这些方法都有一个共同的局限：**它们需要某种形式的监督**——要么是关于捷径的先验知识，要么是 OOD 数据，要么是人工标注的难样本。

在完全无监督的设置下，检测并缓解捷径学习仍然是一个开放问题。

<figure>
<p><img src="/figures/ch05_fig3_confidence_calibration.png" alt="置信度校准曲线：ID 与 OOD 数据对比" /></p>
<figcaption>置信度校准曲线：ID 与 OOD 数据对比</figcaption>
</figure>

*图3：置信度校准曲线展示了捷径学习最危险的后果——过度自信。ID 数据（绿色）沿对角线校准良好，而 OOD 数据（红色）严重偏离：在高置信度（0.8–0.9）区间，模型的实际准确率只有 0.6–0.7。阴影区域标出了过度自信区，模型在这里自信地给出错误预测，却没有任何不确定性警告。*

<div class="center">

------------------------------------------------------------------------

</div>

## 九、伪代码：捷径检测与 OOD 泛化测试

让我把上面讨论的核心算法形式化。

**算法1：基于梯度的捷径检测**

```python
import torch

def shortcut_detection(model, dataloader, shortcut_indices, stable_indices, tau=0.5):
    """
    Gradient-based shortcut detection.
    shortcut_indices: column indices of the shortcut features in the input
    stable_indices:   column indices of the stable features in the input
    Returns: shortcut dependence score S (higher means stronger shortcut reliance)
    """
    model.eval()
    ratios = []
    criterion = torch.nn.CrossEntropyLoss()

    for x, y in dataloader:
        x = x.requires_grad_(True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        g = x.grad  # shape: (batch, n_features)
        g_c = g[:, shortcut_indices].norm(dim=1)   # norm of gradients w.r.t. shortcut features
        g_s = g[:, stable_indices].norm(dim=1)     # norm of gradients w.r.t. stable features
        ratio = g_c / (g_s + 1e-8)
        ratios.append(ratio.detach())

    S = torch.cat(ratios).mean().item()
    if S > tau:
        print(f"S={S:.3f} > τ={tau}: model relies significantly on shortcut features")
    else:
        print(f"S={S:.3f} ≤ τ={tau}: model mostly relies on stable features")
    return S
```

这个算法背后的直觉是：如果模型过度依赖捷径，损失对捷径特征的梯度就会很大——因为改变捷径特征会明显改变预测。

**算法2：分布偏移下的泛化测试**

```python
def generalization_test(model, id_loader, ood_loader, device="cpu"):
    """
    Evaluate the model on ID and OOD test sets and report the generalization gap.
    """
    def evaluate(loader):
        model.eval()
        correct, total, high_conf_wrong = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                conf = probs.max(dim=1).values
                correct += (pred == y).sum().item()
                # high-confidence but wrong samples (overconfidence detection)
                high_conf_wrong += ((conf > 0.8) & (pred != y)).sum().item()
                total += y.size(0)
        return correct / total, high_conf_wrong / total

    acc_id,  hcw_id  = evaluate(id_loader)
    acc_ood, hcw_ood = evaluate(ood_loader)
    gap = acc_id - acc_ood

    print(f"ID  accuracy:        {acc_id:.1%}")
    print(f"OOD accuracy:       {acc_ood:.1%}")
    print(f"Generalization gap: {gap:.1%}")
    if gap < 0.05:
        print("-> good generalization; the model may have learned stable features")
    elif gap < 0.20:
        print("-> moderate generalization; partial shortcut reliance")
    else:
        print("-> poor generalization; the model heavily relies on shortcuts")
    if hcw_ood > 0.1:
        print(f"⚠ OOD high-confidence error rate {hcw_ood:.1%}: overconfidence present")
    return acc_id, acc_ood, gap
```

**算法3：Tilted ERM 训练**

```python
import torch
import torch.nn as nn

def tilted_erm_train(model, dataloader, t=5.0, lr=1e-3, epochs=50):
    """
    Tilted ERM training: gives more weight to hard samples to reduce shortcut reliance.
    t > 0: larger values focus more on high-loss samples; t = 0 reduces to standard ERM.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            losses = criterion(model(x), y)          # per-sample losses
            # Tilted loss: log-sum-exp softens the max
            tilted_loss = (1.0 / t) * torch.log(
                (1.0 / len(losses)) * torch.sum(torch.exp(t * losses))
            )
            optimizer.zero_grad()
            tilted_loss.backward()
            optimizer.step()

    return model
```

当 $t > 0$ 时，这个算法给难样本更大的权重，迫使模型不能只依赖捷径。

<div class="center">

------------------------------------------------------------------------

</div>

## 十、一个小小的停顿

让我盘点一下这一章做了什么。

经验风险最小化是监督学习的标准范式。它的理论保证建立在一个关键假设上：训练数据和测试数据来自同一个分布。

但在现实中，这个假设几乎从不成立。分布偏移是常态，不是例外。

当分布偏移发生时，ERM 学到的那些“在训练数据上有效的相关性”可能完全失效。这不是 bug——这是 ERM 的结构性特征：**ERM 优化的是统计相关性，而不是因果机制。**

捷径学习是这个问题的具体形态：模型利用训练数据中的虚假相关性，而不是真正的因果特征。更糟的是，即使虚假特征根本不携带额外信息，梯度下降 + 交叉熵的归纳偏置仍然会让模型依赖它们——因为这样能最大化分类边界。

随机鹦鹉假说认为：大型语言模型可能只是在做复杂的统计模式匹配，而不是真正的推理。分布偏移下的脆弱性——比如 CoT 推理在任务、长度、格式偏移下崩溃——支持这一假说。

缓解方法包括数据增强、样本重加权、对抗训练和因果正则化。但它们都需要某种形式的监督。在完全无监督的设置下，检测并缓解捷径学习仍然是一个开放问题。

这引出了第六章的核心问题：如果统计相关性不够，我们需要什么？答案是因果推理——不只是观测 $P(Y|X)$，而是理解干预 $P(Y|do(X))$ 和反事实 $P(Y_x|X', Y')$。

> 相关性是捷径，不是推理。下一章，我们将正面回答：什么是因果？以及为什么观测数据永远无法告诉你答案。

<div class="center">

------------------------------------------------------------------------

</div>

## 悬而未决

- 大型语言模型的“涌现能力”是真正的质变，还是同类现象在数量上的继续堆积？随着模型不断变大，这个问题的答案会改变吗？
- 在完全无监督的设置下，是否存在检测捷径学习的通用方法？还是说，检测捷径本质上需要关于任务的先验知识？
- ERM 的归纳偏置（最大边界）在什么条件下有益，在什么条件下有害？是否存在一个统一的框架来刻画这种权衡？
- 如果我们用纯随机数据训练一个神经网络，它会学到什么样的“捷径”？这个思想实验能告诉我们捷径学习的什么本质？
- 人类学习也存在捷径学习吗？如果存在，人类如何克服它？这对设计更好的机器学习算法意味着什么？

<div class="center">

------------------------------------------------------------------------

</div>

## 动手试试：构造一个捷径陷阱，然后看着模型掉进去

本章有一个核心主张：ERM 会利用训练数据中的任何相关性，不管它是否因果。你即将亲手验证这一点——不是通过阅读，而是通过亲眼看着它发生。

<div class="center">

------------------------------------------------------------------------

</div>

### 第一步：设计你的捷径数据集

挑一个简单的二分类任务，然后**故意**在数据中植入一个捷径。

**选项 A：图像分类（推荐）**

任务：区分两种简单形状（比如圆形 vs. 方形）

捷径设计：
- 训练集：90% 的圆形放在图像左半边，90% 的方形放在右半边
- 测试集（ID）：保持同样的位置偏差
- 测试集（OOD）：翻转位置——圆形在右，方形在左

**选项 B：文本分类**

任务：情感分类（正面 vs. 负面）

捷径设计：
- 训练集：给 90% 的正面评论追加“recommended!”，给 90% 的负面评论追加“not recommended”
- 测试集（ID）：保留同样的标记
- 测试集（OOD）：移除所有标记，只保留评论正文

**你的第一个问题（在生成任何数据之前回答）：** 你认为模型会学到什么？它会依赖位置/标记（捷径），还是依赖形状/语义（稳定特征）？为什么？

<div class="center">

------------------------------------------------------------------------

</div>

### 第二步：训练一个标准 ERM 模型

使用最简单的架构：
- 图像：2–3 层卷积网络
- 文本：简单的 LSTM 或 Transformer

训练目标：最小化交叉熵损失，直到训练准确率 > 95%

**不要做任何特殊处理**——就是标准的梯度下降 + 交叉熵。

记录：
- 训练曲线（loss vs. epoch）
- 最终训练准确率

<div class="center">

------------------------------------------------------------------------

</div>

### 第三步：在三个测试集上评估

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# --- Helper: evaluate model accuracy -----------------------------------
def evaluate(model, dataloader, device='cpu'):
    """Return the accuracy (0~1) of the model on the given dataset."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return accuracy_score(all_labels, all_preds)

# 1. ID test set (same distribution as training; shortcut still present)
acc_id = evaluate(model, test_id_loader)
print(f"ID test set accuracy: {acc_id:.3f}")

# 2. OOD test set (shortcut broken: position/marker reversed)
acc_ood = evaluate(model, test_ood_loader)
print(f"OOD test set accuracy: {acc_ood:.3f}")

# 3. No-shortcut baseline (optional): retrain on data with a 50% shortcut correlation, then evaluate on OOD
# train_baseline = regenerate a training set with a 50% (random) shortcut correlation
# model_baseline = train a new model (same architecture, same hyperparameters)
# acc_baseline_ood = evaluate(model_baseline, test_ood_loader)
# print(f"No-shortcut baseline OOD accuracy: {acc_baseline_ood:.3f}")

# 4. Compute the generalization gap
gap = acc_id - acc_ood
print(f"\nGeneralization gap (ID - OOD): {gap:.3f}")
if gap < 0.05:
    print("-> good generalization; the model may have learned stable features")
elif gap < 0.20:
    print("-> partial shortcut reliance; moderate generalization")
else:
    print("-> heavy shortcut reliance; the model fell into the trap!")
```

**你的第二个问题：** 泛化差距有多大？如果差距 > 30%，说明模型严重依赖捷径。你的模型掉进陷阱了吗？

<div class="center">

------------------------------------------------------------------------

</div>

### 第四步：可视化模型在看什么

**对于图像：** 使用梯度加权类激活图（Grad-CAM）或简单的显著性图，看看模型做预测时关注图像的哪个区域。

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- Gradient-based saliency map (for image classification models) -------
def saliency_map(model, x_sample, target_class):
    """
    Compute the absolute gradients of the prediction w.r.t. the input pixels as the saliency.
    x_sample: tensor of shape=(1, C, H, W); requires_grad will be enabled
    target_class: index of the target class (0 or 1)
    Returns a saliency map of shape=(H, W) (max over channels)
    """
    model.eval()
    x = x_sample.clone().detach().requires_grad_(True)  # enable gradient tracking

    # Forward pass to obtain the prediction logits
    logits = model(x)
    y_hat = logits[0, target_class]  # logit of the target class

    # Backward pass: gradient of y_hat w.r.t. the input x
    model.zero_grad()
    y_hat.backward()

    # Take absolute gradients and max over channels to get a 2-D saliency map
    saliency = x.grad.data.abs()           # shape=(1, C, H, W)
    saliency, _ = saliency.max(dim=1)      # shape=(1, H, W)
    saliency = saliency.squeeze().numpy()  # shape=(H, W)
    return saliency

# Take one sample from the OOD test set and see where the model looks
sample_x, sample_y = next(iter(test_ood_loader))  # take one batch
x0 = sample_x[0:1]   # first image, shape=(1, C, H, W)
pred_class = model(x0).argmax(dim=1).item()

sal = saliency_map(model, x0, target_class=pred_class)

# Visualization: overlay the saliency on the original image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: original image (if it is a single-channel grayscale image)
axes[0].imshow(x0.squeeze().detach().numpy(), cmap='gray')
axes[0].set_title(f'Original image (true class={sample_y[0].item()}, prediction={pred_class})')
axes[0].axis('off')

# Right: saliency heatmap
im = axes[1].imshow(sal, cmap='hot')
axes[1].set_title('Saliency map (region the model attends to)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.suptitle('If the saliency concentrates on the left/right half, the model is using the position shortcut')
plt.tight_layout()
plt.show()

# If the model relies on the position shortcut, the saliency will concentrate
# on the left/right half instead of the shape edges.
```

**对于文本：** 计算每个词对预测的贡献（通过遮挡或梯度）。

如果模型依赖“recommended!”这个词，那么遮挡这个词会让预测翻转，而遮挡其他词几乎没有影响。

**你的第三个问题：** 模型的注意力/显著性集中在哪里？它看的是捷径，还是稳定特征？

<div class="center">

------------------------------------------------------------------------

</div>

### 第五步：测试置信度校准

模型在 OOD 数据上失败时，它知道自己正在失败吗？

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- Confidence analysis: compare max softmax probabilities on ID and OOD data ---
def get_confidence_and_accuracy(model, dataloader):
    """
    Return the max softmax probability (confidence) of each sample and whether the prediction is correct.
    """
    model.eval()
    confidences, corrects = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)          # convert to a probability distribution
            conf, pred = probs.max(dim=1)             # the max probability is the confidence
            corrects.extend((pred == y_batch).numpy())
            confidences.extend(conf.numpy())
    return np.array(confidences), np.array(corrects)

# Compute confidences on ID and OOD data separately
conf_id,  correct_id  = get_confidence_and_accuracy(model, test_id_loader)
conf_ood, correct_ood = get_confidence_and_accuracy(model, test_ood_loader)

# Print summary statistics
print(f"ID data  - accuracy: {correct_id.mean():.3f}, mean confidence: {conf_id.mean():.3f}")
print(f"OOD data - accuracy: {correct_ood.mean():.3f}, mean confidence: {conf_ood.mean():.3f}")

# Plot confidence histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(conf_id,  bins=20, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title(f'ID confidence distribution\n(accuracy={correct_id.mean():.2f})')
axes[0].set_xlabel('Max softmax probability')
axes[0].set_ylabel('Number of samples')
axes[1].hist(conf_ood, bins=20, color='tomato',    edgecolor='white', alpha=0.85)
axes[1].set_title(f'OOD confidence distribution\n(accuracy={correct_ood.mean():.2f})')
axes[1].set_xlabel('Max softmax probability')
plt.tight_layout()
plt.show()

# Diagnosis: overconfidence test
if conf_ood.mean() > 0.8 and correct_ood.mean() < 0.5:
    print("⚠ Overconfident! OOD data: high confidence + low accuracy -> the model is confidently wrong (dangerous!)")
elif conf_ood.mean() < 0.6:
    print("✓ The model knows it is uncertain (lower OOD confidence); relatively safe")
else:
    print("OOD confidence is moderate; judge it together with accuracy")

# Summary:
# - ID data: high accuracy + high confidence -> normal
# - OOD data: low accuracy + high confidence -> overconfidence (dangerous!)
# - OOD data: low accuracy + low confidence -> the model knows it is uncertain (relatively safe)
```

**你的第四个问题（核心问题）：** 模型在 OOD 数据上的平均置信度是多少？如果置信度仍然很高（> 0.8），而准确率很低（< 0.5），说明模型在自信地犯错——这是捷径学习最危险的后果。

<div class="center">

------------------------------------------------------------------------

</div>

### 第六步（可选）：尝试一种缓解方法

从第 8 节中选一种方法：

**方法 A：数据增强** 往训练集里加入一些“反捷径”样本（比如圆形在右、方形在左），把比例从 10% 逐步提高到 50%，观察泛化差距如何变化。

**方法 B：样本重加权** 实现一个简化版的 Tilted ERM：降低“太容易”的样本（模型已经高置信度预测正确的样本）的权重。

观察：缓解方法是否缩小了泛化差距？代价是什么（训练时间？ID 准确率下降？）？

<div class="center">

------------------------------------------------------------------------

</div>

### 检查清单

完成这个练习后，你应该能回答：

1. 你的模型掉进捷径陷阱了吗？泛化差距有多大？
2. 模型的注意力/显著性是否集中在捷径特征上？
3. 模型在 OOD 数据上是否过度自信？
4. 如果你尝试了缓解方法，它有效吗？代价是什么？

如果你只做一件事，就做第四步。它是最短的实验，也是最能让你对“模型在看什么”形成直觉的实验。

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Bender, E. M., et al. (2021). *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* —— “随机鹦鹉”假说的原始论文，引发了关于大型语言模型本质的广泛讨论
- Hodel, D. & West, J. (2023). *Response: Emergent analogical reasoning in large language models* —— 对 GPT-3“涌现推理能力”的批判性反驳，展示了分布偏移下的脆弱性 -> [arXiv:2308.16118]
- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks* —— 捷径学习的系统性综述，涵盖视觉、语言和语音
- Puli, A., et al. (2023). *Don't blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy* —— 揭示梯度下降 + 交叉熵的归纳偏置如何导致捷径学习 -> [arXiv:2308.12553]
- Zhao, C., et al. (2025). *Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens* —— DataAlchemy 实验，表明 CoT 推理是训练分布的一面脆弱镜子 -> [arXiv:2508.01191]
- Li, T., et al. (2020). *Tilted Empirical Risk Minimization* —— 通过对样本重加权来平衡鲁棒性和公平性 -> [arXiv:2007.01162]
- Sricharan, K. & Srivastava, A. (2018). *Building robust classifiers through generation of confident out of distribution examples* —— 用 GAN 生成 OOD 样本进行对抗训练 -> [arXiv:1812.00239]
- Woodland, M., et al. (2024). *Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation* —— 医学影像中的 OOD 检测与捷径学习 -> [arXiv:2408.02761]
- Vapnik, V. & Chervonenkis, A. (1971). *On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities* —— 统计学习理论的奠基工作，也是 ERM 的理论基础
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* —— 因果推理的经典教材；理解统计相关性与因果性之区别的必读之作

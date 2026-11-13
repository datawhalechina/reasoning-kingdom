# 第6章：因果的边界——观测数据永远不够

> 你看过一千次日出。这能证明明天太阳会升起吗？休谟说：不能。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、归纳的幽灵

1748年，大卫·休谟在《人类理解研究》里提出了一个让人不安的问题：

我们凭什么相信未来会像过去一样？

太阳每天升起，这是事实。我们看过一千次日出、一万次日出。但这能**证明**明天太阳会升起吗？

休谟的答案是：不能。

从逻辑上讲，从“过去太阳每天升起”推不出“明天太阳会升起”，二者之间没有必然的推导关系。你能想象一个世界，在那里太阳升起了一万次，然后在第一万零一次停止了——这个想象不包含逻辑矛盾。

这就是**归纳问题**：从有限的观测中，我们无法逻辑地推导出普遍规律。

休谟的论证震动了哲学史，因为它触及了科学方法的根基。科学依赖归纳——从实验数据推导出自然定律。如果归纳在逻辑上不可靠，科学的地位是什么？

三百年后，这个问题以一种更技术化的形式回来了：**从观测数据，我们能推导出因果关系吗？**

答案依然是：不能。

这一章要讲的，就是这个“不能”的精确含义，以及我们如何在这个限制下依然做出有用的推断。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、相关性的陷阱

让我从一个经典的例子开始。

1950年代，流行病学家观测到一个强烈的统计相关性：吸烟者得肺癌的比例远高于非吸烟者。

这能证明吸烟导致肺癌吗？

烟草公司的律师说：不能。他们提出了一个替代解释：也许存在某个**隐藏变量**——比如基因型——同时导致“喜欢吸烟”和“易患肺癌”。在这个解释下，吸烟和肺癌之间没有因果关系，它们只是共同原因的两个结果。

用因果图来表示：

**假说1（因果）：**

    Smoking -> Lung Cancer

**假说2（混淆）：**

    Genotype -> Smoking
    Genotype -> Lung Cancer

在假说2里，吸烟和肺癌之间的相关性是**虚假的**——它们在统计上相关，但没有直接的因果联系。

关键问题是：**仅凭观测数据，你能区分这两个假说吗？**

答案是：不能。

如果基因型是不可观测的（1950年代确实如此），那么这两个假说在所有观测数据上做出完全相同的预测。它们是**观测等价的**。

这不是数据量的问题。你可以观测一百万个样本，相关性会变得更显著，但你依然无法确定因果方向——因为观测数据只能告诉你联合分布 $P(X, Y)$，而因果关系是关于**干预**的：如果我强制 $X$ 改变，$Y$ 会如何变化？

观测和干预，是两个不同的问题。

![辛普森悖论：总体相关性与分层因果效应方向相反](/figures/ch06_fig2_simpsons_paradox.png)

*图2：辛普森悖论的几何。总体回归线（黑色虚线）斜率为负——治疗组的康复情况反而更差。但按性别分层后，女性内部（蓝线）和男性内部（红线）的治疗效应均为正。混淆变量（性别）同时影响了接受治疗的概率和基线康复率，扭曲了总体相关性的方向。通过后门调整消除混淆后，因果效应恢复正确的符号。*

<div class="center">

------------------------------------------------------------------------

</div>

## 三、贝叶斯网络：概率的因果外衣

在讲Pearl的因果理论之前，我们需要先理解贝叶斯网络——因为它是结构因果模型的前身，也是它的陷阱。

**贝叶斯网络**是一个有向无环图（DAG）：节点是随机变量，边表示条件依赖关系。

:::details 贝叶斯网络和 DAG：图上的概率是什么意思？
**有向无环图（DAG）**：图是节点和连接它们的边的集合。“有向”意味着每条边都有方向（A->B 和 B->A 不同）。“无环”意味着沿着边走，永远不可能回到出发的地方。

**贝叶斯网络**是在 DAG 上附加了概率的结构：
- 每个节点是一个随机变量（比如“是否下雨”，“是否带伞”）
- 有向边 A->B 表示“A 对 B 有条件影响”——知道 A 的值，会改变我们对 B 的概率估计

关键概念——**条件依赖**：如果给定 A 之后，B 和 C 变得独立（互不影响），那么 A 就是 B 和 C 的“分离点”。贝叶斯网络利用这些条件独立性，把复杂的联合概率分解成更小的局部概率的乘积，大大减少计算量。

**重要警告**：贝叶斯网络中的边表示的是统计依赖，**不一定是因果关系**——这正是本节要讲的陷阱。
:::

给定一个贝叶斯网络 $G = (V, E)$，联合概率分布分解为：

$$
P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
$$

这个分解有一个漂亮的性质：它把高维联合分布拆成一系列低维条件分布的乘积，大大减少了需要估计的参数数量。

但这里有一个微妙之处：**贝叶斯网络的边不一定是因果关系。**

贝叶斯网络只是对联合分布的一种分解方式。同一个联合分布可以对应多个不同的贝叶斯网络——只要它们编码了相同的条件独立性。

举个例子。考虑三个变量：$A$（警报响了），$B$（入室盗窃），$E$（地震）。

真实的因果结构是：

    B -> A ← E

入室盗窃或地震都可能触发警报。

但从纯粹的概率角度看，下面这个结构也编码了相同的条件独立性：

    A -> B, A -> E

警报响了，这提高了入室盗窃和地震的概率。

这两个网络在**观测数据**上是等价的——它们对应相同的联合分布 $P(A, B, E)$。但它们的**因果含义**完全不同。

第一个网络说：干预 $B$（强制发生盗窃）会影响 $A$（警报响）。

第二个网络说：干预 $A$（强制警报响）会影响 $B$（发生盗窃）——这显然是荒谬的；你不可能通过按警报按钮来制造盗窃。

**贝叶斯网络是概率模型，不是因果模型。** 它的边表示条件依赖，而不是因果流。

Pearl的贡献，是在贝叶斯网络之上加入**因果语义**——把边解释为因果关系，然后定义在这种解释下干预和反事实意味着什么。

<div class="center">

------------------------------------------------------------------------

</div>

## 停顿一下

贝叶斯网络的边不是因果关系——这已经够让人不安了。

但还有一个更深的问题：即使你有了Pearl的因果图，**你怎么知道这张图是对的？**

因果图是你画的。它是你对世界结构的先验假设。数据可以告诉你哪些变量是条件独立的，但这只能缩小可能因果图的范围——它永远无法确定唯一的一张图。

这意味着：因果推断的每一个结论，都建立在一个**无法从数据中验证的假设**之上——你画的那张图。

如果图画错了，用do算子计算出的干预效应也会是错的。你得到一个内部自洽的错误答案。

那么，我们凭什么相信自己的因果图？经验？直觉？领域知识？

这些都是答案——但它们都不是从数据里来的。

先把这个问题放在桌上。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、Pearl的因果阶梯

在2000年出版的《Causality》中，Judea Pearl提出了一个三层因果推理框架，后来被称为**因果阶梯**。

:::details 因果阶梯：观测、干预、反事实——三种不同的问题
Pearl的因果阶梯把“因果问题”分为三个层级，每一层都需要更强的假设才能回答：

**第一层（观测）**：*看到 X，Y 的概率是多少？*
纯粹的统计问题，只需要数据。例：“吸烟者中，有多大比例得肺癌？”

**第二层（干预）**：*如果我强制 X 发生，Y 的概率是多少？*
需要因果图。例：“如果我让所有人都吸烟（随机分配），肺癌率会变化多少？”这正是随机对照试验（RCT）所做的事。

**第三层（反事实）**：*如果过去 X 不同，Y 现在会是什么？*
需要完整的结构因果模型。例：“这个病人吸烟并且得了肺癌——如果他从未吸烟，他还会得肺癌吗？”这是关于某个具体个人的反事实，无法用实验直接回答。

**关键点**：你无法用低层的信息回答高层的问题。再多的观测数据，都无法直接回答干预或反事实问题——这就是“观测数据永远不够”的精确含义。
:::

**第一层：关联（观测）**

问题形式：$P(Y \mid X)$

“给定我看到了 $X$，$Y$ 的概率是多少？”

这是一个纯粹的统计问题。给定数据，你可以估计条件概率。机器学习的大部分工作都停留在这个层级：给定特征 $X$，预测标签 $Y$。

**第二层：干预**

问题形式：$P(Y \mid do(X))$

“如果我强制 $X$ 取某个值，$Y$ 的概率是多少？”

这里的 $do(X)$ 是Pearl引入的符号，表示对 $X$ 的外部干预——不是观测到 $X$ 自然发生，而是通过某种外部行动强制 $X$ 取某个值。

$P(Y \mid do(X))$ 和 $P(Y \mid X)$ 通常是不同的。

举个例子：

- $P(\text{康复} \mid \text{吃药})$ 可能很高——因为病情轻的人更可能吃药，而病情轻的人本来就更可能康复。

- $P(\text{康复} \mid do(\text{吃药}))$ 是药物的真实因果效应——如果我们随机分配谁吃药，吃药组的康复率是多少。

第一个是观测性的，第二个是干预性的。混淆两者，是医学研究中最常见的错误之一。

**第三层：反事实**

问题形式：$P(Y_x \mid X', Y')$

“如果过去 $X$ 的值是 $x$（而不是实际观测到的 $X'$），那么 $Y$ 会是什么（给定我们观测到 $Y'$）？”

这是最难的一层，因为它涉及对**没有发生的事情**的推理。

例子：

- 观测：这个病人吃了药，康复了。

- 反事实：如果这个病人没吃药，他会康复吗？

反事实推理需要的不只是数据，还需要一个**结构因果模型**——一个关于世界如何运作的机制性假设。

Pearl的核心主张是：**这三个层级不能互相替代。** 你不能用第一层的信息回答第二层的问题，也不能用第二层的信息回答第三层的问题。

每向上一层，都需要更强的假设和更多的结构。

![Pearl的因果阶梯：三个层级的推理——观测、干预、反事实](/figures/ch06_fig1_causal_ladder.png)

*图1：Pearl的因果阶梯。第一层（观测）只需要数据，回答“看到 X 之后 Y 的概率”。第二层（干预）需要因果图，回答“强制 X 之后 Y 的概率”。第三层（反事实）需要完整的 SCM 参数，回答“如果 X 曾经不同，Y 会如何不同”。每向上一层，都需要更强的结构假设。*

<div class="center">

------------------------------------------------------------------------

</div>

## 五、结构因果模型：世界的机制蓝图

要回答干预和反事实问题，我们需要**结构因果模型**（SCM）。

一个SCM由三部分组成：

1. **内生变量** $V = \{V_1, \ldots, V_n\}$：我们关心的变量

2. **外生变量** $U = \{U_1, \ldots, U_m\}$：外部的、不可观测的随机扰动

3. **结构方程** $\mathcal{F}$：每个内生变量由一个函数决定

$$
V_i = f_i(\text{Parents}(V_i), U_i)
$$

这里 $\text{Parents}(V_i)$ 是 $V_i$ 的直接原因（因果图中指向 $V_i$ 的节点），$U_i$ 是影响 $V_i$ 的外生噪声。

**例子：吸烟与肺癌**

变量：

- $S$：这个人是否吸烟
- $C$：他是否得肺癌
- $G$：基因型（不可观测）

结构方程：

    G = U_G                    (the genotype is set by exogenous factors)
    S = f_S(G, U_S)            (smoking tendency is influenced by genes and other factors)
    C = f_C(S, G, U_C)         (lung cancer is influenced by smoking, genes, and other factors)

因果图：

    G -> S -> C
    G -> C

这个模型编码了一个关于世界的假设：基因型 $G$ 同时影响吸烟倾向和肺癌风险，而吸烟 $S$ 也直接影响肺癌 $C$。

有了SCM，我们可以定义**干预**：

$$
do(S = s) \text{ means: replace the structural equation } S = f_S(G, U_S) \text{ with the constant } s
$$

干预后，模型变成：

    G = U_G
    S = s                      (forced)
    C = f_C(s, G, U_C)

现在 $S$ 不再依赖 $G$——我们切断了 $G \to S$ 这条边。这就是干预的图形表示：**删除所有指向被干预变量的边**。

干预后的联合分布 $P(C \mid do(S=s))$ 可以从修改后的模型中计算出来。

关键点：$P(C \mid do(S=s)) \neq P(C \mid S=s)$，因为后者包含 $G$ 对 $S$ 和 $C$ 的混淆，而前者切断了这个混淆。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、do 演算：从观测到干预的桥梁

现在我们来到一个核心问题：**给定观测数据，我们能计算干预分布吗？**

也就是说：能否从 $P(V)$（观测分布）推导出 $P(Y \mid do(X))$（干预分布）？

Pearl的**do 演算**给出了答案：在某些条件下，可以。

do 演算有三条规则，它们让你在因果图 $G$ 的约束下，操纵包含 $do(\cdot)$ 的概率表达式。

**规则1：插入 / 删除观测**

$$
P(Y \mid do(X), Z, W) = P(Y \mid do(X), W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}
$$

其中 $G_{\overline{X}}$ 是删除所有指向 $X$ 的边之后的图。

直觉：如果在干预 $X$ 之后，$Z$ 和 $Y$ 条件独立（给定 $W$），那么观测 $Z$ 不提供额外信息。

**规则2：行动 / 观测交换**

$$
P(Y \mid do(X), do(Z), W) = P(Y \mid do(X), Z, W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\underline{Z}}}
$$

其中 $G_{\overline{X}\underline{Z}}$ 是删除指向 $X$ 的边以及从 $Z$ 出发的边之后的图。

直觉：如果 $Z$ 不影响 $Y$（在干预 $X$ 之后），那么干预 $Z$ 等价于观测 $Z$。

**规则3：插入 / 删除行动**

$$
P(Y \mid do(X), do(Z), W) = P(Y \mid do(X), W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\overline{Z(W)}}}
$$

其中 $G_{\overline{X}\overline{Z(W)}}$ 是删除指向 $X$ 的边，以及删除指向 $Z$ 中不是 $W$ 祖先的节点的边之后的图。

直觉：如果 $Z$ 不影响 $Y$（在特定的图变换下），那么对 $Z$ 的干预可以忽略。

这三条规则看起来很技术化，但它们的威力在于：**它们是完备的**。

Pearl和同事证明：如果一个因果效应 $P(Y \mid do(X))$ 能从观测分布 $P(V)$ 和因果图 $G$ 中被识别，那么do 演算的三条规则就足以推导出它。

反过来，如果do 演算无法推导出 $P(Y \mid do(X))$，那么它是不可识别的：仅凭观测数据和图结构，无法确定因果效应。

<div class="center">

------------------------------------------------------------------------

</div>

## 七、后门准则与前门准则

do 演算在理论上是完备的工具，但在实践中，我们通常使用更简单、更直接的准则来判断因果效应是否可识别。

**后门准则**

给定因果图 $G$，如果存在一组变量 $Z$，满足：

1. $Z$ 阻断了所有从 $X$ 到 $Y$ 的**后门路径**（包含指向 $X$ 的边的路径）

2. $Z$ 中没有 $X$ 的后代

那么因果效应可以用调整公式计算：

$$
P(Y \mid do(X)) = \sum_z P(Y \mid X, Z=z) P(Z=z)
$$

这就是**调整**：通过对混淆变量 $Z$ 分层，消除虚假相关性。

回到吸烟与肺癌的例子：

因果图：

    G -> S -> C
    G -> C

$G$ 是位于后门路径 $S \leftarrow G \to C$ 上的混淆变量。如果我们能观测 $G$，那么：

$$
P(C \mid do(S)) = \sum_g P(C \mid S, G=g) P(G=g)
$$

这也是随机对照试验（RCT）有效的原因：随机分配 $S$ 等价于切断 $G \to S$，使得 $P(C \mid do(S)) = P(C \mid S)$。

**前门准则**

但如果混淆变量不可观测呢？

Pearl发现了一个巧妙的情形：即使混淆变量不可观测，如果存在一个**中介变量** $M$，满足：

1. $M$ 完全中介 $X$ 对 $Y$ 的效应（$X$ 只通过 $M$ 影响 $Y$）

2. $X$ 阻断了所有从 $M$ 到 $Y$ 的后门路径

3. 空集阻断了所有从 $X$ 到 $M$ 的后门路径

那么因果效应可以用前门公式计算：

$$
P(Y \mid do(X)) = \sum_m P(M=m \mid X) \sum_{x'} P(Y \mid M=m, X=x') P(X=x')
$$

这个公式不需要观测混淆变量——它通过中介变量 $M$ 绕开了混淆。

**例子：吸烟、焦油与肺癌**

    U -> S -> T -> C
    U -> C

$U$ 是不可观测的混淆变量（比如基因型），$T$ 是焦油沉积（可观测的中介变量）。

即使我们无法观测 $U$，我们依然可以通过前门准则计算 $P(C \mid do(S))$——因为 $T$ 完全中介了 $S$ 对 $C$ 的效应。

这是一个深刻的结果：**在某些结构下，即使存在不可观测的混淆，观测数据也足以识别因果效应。**

但“某些结构”是关键——不是每个因果图都满足后门准则或前门准则。

## 八、观测等价类：因果图何时不可区分

现在我们来到本章最重要的部分。即使有无限的观测数据，你依然无法确定唯一的因果图——因为多个不同的因果图可以产生相同的观测分布。

这些图被称为**马尔可夫等价类**。

**定义：** 两个DAG $G_1$ 和 $G_2$ 是马尔可夫等价的，如果它们编码相同的条件独立性关系。

等价的图共享相同的**骨架**（忽略箭头方向后得到无向图）和相同的**v-结构**（形如 $A \to C \leftarrow B$ 的结构，其中 $A$ 和 $B$ 不相邻）。

**例子：三个变量的等价类**

考虑三个变量 $X, Y, Z$。以下三个图是马尔可夫等价的：

    X -> Y -> Z
    X ← Y -> Z
    X ← Y ← Z

它们都编码相同的条件独立性：$X \perp\!\!\!\perp Z \mid Y$。

从观测数据中，你无法区分这三个图——因为它们对应相同的联合分布。

但它们的因果含义完全不同：

- 第一个图：$X$ 影响 $Y$，$Y$ 影响 $Z$

- 第二个图：$Y$ 同时影响 $X$ 和 $Z$

- 第三个图：$Z$ 影响 $Y$，$Y$ 影响 $X$

如果你想要 $P(Z \mid do(X))$，这三个图会给出不同的答案。

**不可识别性从何而来**

这不是数据量的问题，也不是算法的问题。这是**结构性的不可能**。

观测数据只能告诉你条件独立性——哪些变量在给定其他变量后是独立的。但因果关系不只是独立性，它是关于**干预下的行为**。

两个图可以有完全相同的观测分布，却在干预下表现完全不同。

这意味着：**仅凭观测数据，因果发现只能恢复到马尔可夫等价类——永远无法得到唯一确定的因果图。**

要打破等价类，你需要额外信息：

1. **干预数据**：如果你能做实验——强制改变某些变量并观察其他变量如何响应——你就能确定因果方向

2. **时间顺序**：如果你知道 $X$ 发生在 $Y$ 之前，那么 $Y \to X$ 不可能

3. **函数形式假设**：如果你假设因果机制是线性的，或者属于加性噪声模型，某些等价类可以被打破

4. **先验知识**：如果你知道某些边不可能存在（比如“年龄不可能被收入影响”），你可以排除某些图

但如果你只有观测数据、没有额外假设，因果图就是不可识别的。

![马尔可夫等价类：三个因果图，同一个观测分布](/figures/ch06_fig3_markov_equivalence.png)

*图3：马尔可夫等价类示例。三个不同的因果图（X->Y->Z、X←Y->Z、X←Y←Z）编码相同的条件独立性 X ⊥ Z | Y，因此在观测数据上完全等价。但它们的因果含义不同：第一个图中 do(X) 影响 Z；第二个图中不影响。仅凭观测数据无法区分它们。*

## 九、忠实性假设：一座脆弱的桥梁

因果发现算法（如PC算法和GES算法）通常依赖两个假设：

**假设1：因果马尔可夫条件**

给定其父节点，每个变量与其非后代独立。

这是贝叶斯网络的标准假设，通常是合理的。

**假设2：忠实性**

观测分布中的每一个条件独立性，都由因果图中的d-分离蕴含。

换句话说：如果 $X \perp\!\!\!\perp Y \mid Z$ 在数据中成立，那么 $X$ 和 $Y$ 在图中被 $Z$ d-分离。

这个假设看起来无害，但实际上很强。

**一个反例：参数的巧合抵消**

考虑以下因果图和结构方程：

    X -> Y -> Z
    X -> Z

    Y = a·X + U_Y
    Z = b·Y + c·X + U_Z

如果参数恰好满足 $b \cdot a + c = 0$，那么 $X$ 对 $Z$ 的总效应为零——直接效应 $c$ 和通过 $Y$ 的间接效应 $b \cdot a$ 相互抵消。

在这种情况下，观测数据会显示 $X \perp\!\!\!\perp Z \mid Y$——不是因为图结构，而是因为参数的巧合。

忠实性假设排除了这类巧合：它假设参数处于“一般位置”，不会恰好抵消。

但在真实世界中，这类巧合可能并不罕见。生物系统和经济系统充满了反馈和平衡机制，它们的参数可能恰好使某些效应相互抵消。

**忠实性失败的后果**

如果忠实性不成立，因果发现算法会推断出错误的图结构——它们会相信某些边不存在，因为对应的变量在数据中看起来独立，而实际上它们确实有因果关系——只是效应被抵消了。

这是一种深层的脆弱性：因果发现依赖一个关于参数的假设，而这个假设无法在数据中检验——因为你无法区分“真正的独立”和“巧合的抵消”。

<div class="center">

------------------------------------------------------------------------

</div>

## 十、反事实：最难的一层

Pearl因果阶梯的第三层——反事实——是最难的，因为它涉及对**没有发生的事情**的推理。

**定义反事实**

给定一个SCM和观测到的证据 $X=x', Y=y'$，反事实问题是：

$$
P(Y_x \mid X=x', Y=y')
$$

“如果 $X$ 的值是 $x$（而不是观测到的 $x'$），$Y$ 会是什么（给定我们观测到 $Y=y'$）？”

计算反事实需要三个步骤：

**步骤1：溯因**

根据观测到的证据，更新你对**外生变量** $U$ 的信念：

$$
P(U \mid X=x', Y=y')
$$

**步骤2：行动**

修改SCM，用 $X=x$ 替换 $X$ 的结构方程（一次干预）。

**步骤3：预测**

在修改后的模型和更新后的 $U$ 分布下，计算 $Y$ 的分布：

$$
P(Y_x \mid X=x', Y=y') = \sum_u P(Y_x \mid U=u) P(U=u \mid X=x', Y=y')
$$

**例子：药物的个体因果效应**

假设我们有以下SCM：

    X = U_X                    (whether the drug was taken; set by exogenous factors)
    Y = a·X + U_Y              (recovery outcome)

观测：某个病人吃了药（$X=1$），并且康复了（$Y=1$）。

反事实问题：如果这个病人没吃药（$X=0$），他会康复吗？

步骤1：从 $X=1, Y=1$ 推断 $U_Y$：

$$
1 = a \cdot 1 + U_Y \implies U_Y = 1 - a
$$

步骤2：干预 $X=0$。

步骤3：计算反事实结果：

$$
Y_0 = a \cdot 0 + U_Y = 1 - a
$$

如果 $a > 0$（药物有正效应），那么 $Y_0 < 1$——如果这个病人没吃药，他的康复会更差。

**反事实的不可识别性**

关键问题：反事实通常是**不可识别的**。

即使你知道因果图，即使有无限的观测数据，你依然无法从数据中唯一确定反事实分布——因为反事实依赖外生变量 $U$ 的分布，而 $U$ 不可观测。

在上面的例子里，如果我们不知道 $a$ 的值（药物效应的大小），我们就无法计算 $Y_0$。

反事实需要的不仅是因果图，还需要**参数化的结构方程**——这是比图结构更强的假设。

这就是为什么反事实推理在实践中很难：它需要一个完整的、参数化的世界模型，而这种模型通常无法从数据中完全学到。

## 十一、伪代码：因果推断的核心算法

让我把上面讨论的核心算法形式化。

**算法1：后门调整**

```python
import itertools
import numpy as np

def backdoor_adjustment(data, x_col, y_col, z_cols, x_val):
    """
    Estimate the causal effect P(Y | do(X=x_val)) via backdoor adjustment.
    data:    pandas DataFrame containing observational data
    x_col:   column name of the intervention variable
    y_col:   column name of the outcome variable
    z_cols:  list of adjustment-set column names (satisfying the backdoor criterion)
    x_val:   intervention value
    Returns: estimate of P(Y=1 | do(X=x_val))
    """
    if not z_cols:
        raise ValueError("Empty adjustment set; the causal effect is not identifiable via backdoor adjustment")

    # Enumerate all value combinations of Z
    z_values = [data[z].unique() for z in z_cols]
    result = 0.0

    for z_combo in itertools.product(*z_values):
        # Conditional probability P(Y | X=x_val, Z=z_combo)
        mask_xz = (data[x_col] == x_val)
        for z_col, z_val in zip(z_cols, z_combo):
            mask_xz &= (data[z_col] == z_val)
        if mask_xz.sum() == 0:
            continue
        p_y_given_xz = data.loc[mask_xz, y_col].mean()

        # Marginal probability P(Z=z_combo)
        mask_z = np.ones(len(data), dtype=bool)
        for z_col, z_val in zip(z_cols, z_combo):
            mask_z &= (data[z_col] == z_val)
        p_z = mask_z.mean()

        result += p_y_given_xz * p_z

    return result
```

## 十二、一个小小的停顿

让我盘点一下这一章做了什么。

休谟的归纳问题，在三百年后以因果推断的形式回来了：从观测数据，我们能推导出因果关系吗？答案是：不能——至少不能唯一确定。

观测数据只能告诉我们变量之间的联合分布 $P(X, Y)$，而因果关系是关于干预 $P(Y \mid do(X))$ 的。这两者通常是不同的，因为观测包含混淆。

Pearl的因果阶梯把推理分为三个层级：观测、干预、反事实。每一层都需要更强的假设。机器学习的大部分工作停留在第一层，而真正的因果推理需要第二层或第三层。

结构因果模型（SCM）提供了一个框架，用结构方程和因果图来编码世界的机制。有了SCM，我们可以定义干预（删除所有指向被干预变量的边）和反事实（溯因—行动—预测三步法）。

do 演算给出了从观测分布推导干预分布的完备规则。后门准则和前门准则是实践中更直接的工具——在某些图结构下，观测数据足以识别因果效应。

但核心的限制依然存在：

1. **马尔可夫等价类**：多个因果图可以产生相同的观测分布，仅凭观测数据无法区分它们

2. **忠实性假设**：因果发现依赖一个关于参数处于“一般位置”的假设，而这个假设无法在数据中检验

3. **反事实的不可识别性**：反事实推理需要参数化的结构方程，而不只是因果图

这引出了第七章的问题：如果观测数据不够，我们需要什么？答案是干预——主动改变世界，而不是被动地观察它。随机对照试验（RCT）是因果推断的黄金标准，不是因为它更精确，而是因为它打破了观测的牢笼。

<div class="center">

------------------------------------------------------------------------

</div>

> 因果推断揭示了观测的局限。下一章，我们将转向另一个前沿——计算本身内部的不对称性：为什么找到一个答案比验证一个答案难得多？

## 悬而未决

- 忠实性假设在真实世界中失败得有多频繁？有没有办法检测忠实性失败？

- 如果两个因果图属于同一个马尔可夫等价类，但给出不同的干预预测，我们应该相信哪一个？这是一个科学选择，还是一个哲学选择？

- 反事实推理需要一个完全参数化的SCM。在什么条件下我们可以从数据中学到这些参数？什么时候不可能？

- 大语言模型能做因果推理吗？它们从文本中学到的是 $P(Y \mid X)$ 还是 $P(Y \mid do(X))$？这个问题的答案决定了一个LLM能力的边界。

- 如果你用纯观测数据训练一个因果发现算法，然后在干预数据上测试它，它会在哪里系统性地失败？这种失败模式告诉我们什么？

## 动手试试：用辛普森悖论看穿观测的谎言

本章的核心主张是：观测数据会撒谎，因为它包含混淆。你要亲手构造一个辛普森悖论的例子，看着同一组数据在不同的分层下给出完全相反的结论。

<div class="center">

------------------------------------------------------------------------

</div>

### 第一步：生成带混淆的数据

构造以下因果结构：

    Causal graph:
    G -> T -> Y
    G -> Y

    where:
    G = gender (0=female, 1=male)
    T = whether the patient received treatment (0=no, 1=yes)
    Y = recovery outcome (continuous, higher is better)

真实的因果机制：

- 男性更常接受治疗（因为他们病得更重）
- 治疗有正效应（提高康复）
- 但男性的基线康复更低（因为他们病得更重）

```python
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000  # sample size

# Generate gender (confounder): 0=female, 1=male, 50% each
G = np.random.binomial(1, 0.5, N)

# Generate the treatment decision (affected by gender)
# Men are sicker and more likely to be treated
p_treatment = np.where(G == 1, 0.7, 0.3)  # men 70%, women 30%
T = np.random.binomial(1, p_treatment, N)

# Generate the recovery outcome (affected by both gender and treatment)
# Female baseline recovery score 80; male baseline 60 (men are sicker)
baseline = 80 - 20 * G        # vectorized: baseline score per sample
treatment_effect = 15 * T     # treatment adds 15 points (true causal effect)
noise = np.random.normal(0, 5, N)  # random noise
Y = baseline + treatment_effect + noise

# Arrange into a DataFrame for the following analysis
df = pd.DataFrame({'G': G, 'T': T, 'Y': Y})
print(f"Data generation complete: {N} samples")
print(df.describe().round(2))
print(f"\nTrue causal effect (treatment parameter) = +15 points")
```

**你的第一个问题（在生成任何数据之前回答）：** 在这个设定下，治疗的真实因果效应是多少？如果我们随机分配治疗（切断 G -> T），治疗组的康复分数会比对照组高多少？

<div class="center">

------------------------------------------------------------------------

</div>

### 第二步：计算观测到的相关性（不调整混淆）

生成数据后，直接计算：

```python
import pandas as pd

# Compare treatment vs. control directly, ignoring the gender confounder
treated_mean   = df[df['T'] == 1]['Y'].mean()   # mean of the treatment group
untreated_mean = df[df['T'] == 0]['Y'].mean()   # mean of the control group
observed_effect = treated_mean - untreated_mean  # observed effect (no confounder adjustment)

print(f"Treatment group mean: {treated_mean:.2f}")
print(f"Control group mean: {untreated_mean:.2f}")
print(f"Observed effect (unadjusted): {observed_effect:.2f}")
print(f"True causal effect: +15.00")
```

**你的第二个问题：** 观测到的效应是什么符号？它是正的（治疗组更好）还是负的（对照组更好）？它和真实因果效应一致吗？

如果不一致，你就遇到了辛普森悖论——总体相关性和因果效应的方向相反。

<div class="center">

------------------------------------------------------------------------

</div>

### 第三步：分层分析（调整混淆）

现在按性别分层，计算每一层内部的结果：

```python
import pandas as pd

# --- Treatment effect within the female group (G=0) ---------------------------
female_treated   = df[(df['G'] == 0) & (df['T'] == 1)]['Y'].mean()
female_untreated = df[(df['G'] == 0) & (df['T'] == 0)]['Y'].mean()
effect_female    = female_treated - female_untreated

# --- Treatment effect within the male group (G=1) -----------------------------
male_treated   = df[(df['G'] == 1) & (df['T'] == 1)]['Y'].mean()
male_untreated = df[(df['G'] == 1) & (df['T'] == 0)]['Y'].mean()
effect_male    = male_treated - male_untreated

# --- Backdoor adjustment: weighted average of the stratified effects by gender's marginal distribution ---
P_G0 = (df['G'] == 0).mean()  # proportion of female samples
P_G1 = (df['G'] == 1).mean()  # proportion of male samples
adjusted_effect = P_G0 * effect_female + P_G1 * effect_male  # weighted average

print(f"Effect within females (G=0): {effect_female:.2f}")
print(f"Effect within males (G=1): {effect_male:.2f}")
print(f"Adjusted causal effect (backdoor adjustment): {adjusted_effect:.2f}")
print(f"True causal effect: +15.00")
print(f"Observed effect (unadjusted): {observed_effect:.2f}")
```

**你的第三个问题（核心问题）：** 分层之后，每个性别内部的治疗效应是什么符号？它和第二步的观测效应相同吗？

如果第二步是负的，而两个性别内部的效应都是正的，你刚刚亲眼看到了辛普森悖论：总体相关性和分层因果效应方向相反。

<div class="center">

------------------------------------------------------------------------

</div>

### 第四步：可视化混淆的几何

画一个散点图，横轴是治疗（0 或 1），纵轴是康复分数：

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

fig, ax = plt.subplots(figsize=(9, 6))

# Scatter of the four subgroups (small jitter to avoid overlaps)
jitter = 0.04
groups = [
    (0, 0, 'royalblue',  'o', 'Females, control (G=0, T=0)'),
    (0, 1, 'royalblue',  '^', 'Females, treatment (G=0, T=1)'),
    (1, 0, 'tomato',     'o', 'Males, control (G=1, T=0)'),
    (1, 1, 'tomato',     '^', 'Males, treatment (G=1, T=1)'),
]
for g_val, t_val, color, marker, label in groups:
    mask = (df['G'] == g_val) & (df['T'] == t_val)
    t_jittered = df.loc[mask, 'T'] + np.random.uniform(-jitter, jitter, mask.sum())
    ax.scatter(t_jittered, df.loc[mask, 'Y'],
               c=color, marker=marker, alpha=0.5, s=20, label=label)

# Helper: draw an OLS regression line
def plot_regression(x, y, color, linestyle, label):
    coeffs = np.polyfit(x, y, 1)   # linear regression; coeffs=[slope, intercept]
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color=color, linestyle=linestyle, linewidth=2, label=label)
    return coeffs[0]  # return the slope

# 1. Overall regression line (ignoring gender) -- black dashed line
slope_overall = plot_regression(df['T'].values, df['Y'].values,
                                'black', '--', 'Overall regression line (ignoring gender)')

# 2. Within-female regression line -- blue solid line
female_mask = df['G'] == 0
slope_female = plot_regression(df.loc[female_mask, 'T'].values,
                               df.loc[female_mask, 'Y'].values,
                               'royalblue', '-', 'Within-female regression line')

# 3. Within-male regression line -- red solid line
male_mask = df['G'] == 1
slope_male = plot_regression(df.loc[male_mask, 'T'].values,
                             df.loc[male_mask, 'Y'].values,
                             'tomato', '-', 'Within-male regression line')

ax.set_xlabel('Received treatment (0=no, 1=yes)')
ax.set_ylabel('Recovery score Y')
ax.set_title("The geometry of Simpson's paradox\nOverall line slopes down, but within-group lines slope up")
ax.set_xticks([0, 1])
ax.set_xticklabels(['Control (T=0)', 'Treatment (T=1)'])
ax.legend(fontsize=8, loc='upper left')

print(f"Overall regression slope: {slope_overall:.2f} ({'positive' if slope_overall > 0 else 'negative'} -- treatment {'beneficial' if slope_overall > 0 else 'harmful'}?)")
print(f"Within-female regression slope: {slope_female:.2f}")
print(f"Within-male regression slope: {slope_male:.2f}")
plt.tight_layout()
plt.show()
```

**你的第四个问题：** 总体回归线的斜率是什么符号？女性内部和男性内部的斜率又如何？

如果总体线向下（负斜率），而两条组内线都向上（正斜率），这就是辛普森悖论的几何：混淆变量反转了相关性的方向。

<div class="center">

------------------------------------------------------------------------

</div>

### 第五步：反转因果图，看看会发生什么

现在假设你错误地认为因果图是：

    Wrong causal graph:
    T -> G -> Y
    (treatment affects gender? obviously absurd, but let's pretend we don't know)

在这个错误的图下，后门准则告诉你：不需要调整，因为没有后门路径。

```python
# Under the wrong causal graph T -> G -> Y:
# the graph says T is a cause of G, so G is not a confounder of T,
# the backdoor path T<-G->Y does not exist (the arrow direction is wrong),
# so we simply use the observed correlation as the "causal effect"

# "Causal effect" under the wrong graph = the direct observed effect (no adjustment)
wrong_causal_effect = observed_effect  # the unadjusted value computed in step 2

print(f"'Causal effect' under the wrong graph: {wrong_causal_effect:.2f}")
print(f"Correctly adjusted causal effect:       {adjusted_effect:.2f}")
print(f"True causal effect (generation setting): +15.00")
print(f"Deviation of the wrong inference from the truth: {abs(wrong_causal_effect - 15):.2f} points")
print()
print("Conclusion: a wrong causal graph assumption -> the inference deviates entirely from the true effect")
print("Observational data alone cannot tell you which graph is correct.")
```

**你的第五个问题：** 如果你使用了错误的因果图，你会得出什么结论？这个结论离真实因果效应有多远？

这就是第八节的核心论点：**因果图的结构假设是不可避免的。如果图错了，推断就是错的——而观测数据无法告诉你图是否正确。**

<div class="center">

------------------------------------------------------------------------

</div>

### 检验标准

完成这个练习后，你应该能够回答：

1. 辛普森悖论是如何发生的？混淆变量如何让观测相关性和因果效应方向相反？

2. 后门调整（分层分析）如何消除混淆？为什么分层后的效应更接近真实因果效应？

3. 如果因果图的结构假设错了，后门调整会给出什么结果？误差有多大？

如果你只做一件事，做第四步。那张图会让你一眼看清混淆是如何扭曲相关性的。

<div class="center">

------------------------------------------------------------------------

</div>

## 动手试试：手写 do 运算——咖啡真的让你更聪明吗？

在辛普森悖论的练习中，你看到了观测数据**为什么**会撒谎。这个练习更进一步：你要亲手计算 $P(Y \mid do(X))$，把“切断因果图中的箭头”从一个抽象操作变成可运行的代码。

**场景**：你在一家科技公司的数据团队工作。老板看到数据：喝咖啡的员工写出更多代码。他准备推行“每人每天三杯咖啡”的政策。你需要告诉他，观测到的相关性和强制喝咖啡的真实效应是不是同一回事。

**因果图**：

```
Work Stress (S) ─────> Coffee (C) ─────> Code Output (Y)
     │                              ↑
     └──────────────────────────────┘
```

- **S**（Stress，工作压力）：0 = 低压，1 = 高压
- **C**（Coffee，今天喝了咖啡）：0 = 没喝，1 = 喝了
- **Y**（代码产出，行/小时，连续值）

真实机制：**高压员工更想喝咖啡**（他们靠咖啡保持清醒），**但高压本身会降低产出**。咖啡本身对产出只有轻微的正面效应。

---

### 第一步：生成数据，感受混淆

```python
import numpy as np
import pandas as pd
np.random.seed(42)

N = 2000

# Work stress (confounder)
S = np.random.binomial(1, 0.5, N)          # 50% high stress

# Coffee intake (driven by stress)
p_coffee = np.where(S == 1, 0.80, 0.25)    # high stress: 80% drink coffee; low stress: 25%
C = np.random.binomial(1, p_coffee, N)

# Code output (dragged down by stress, slightly boosted by coffee)
Y = (50                    # baseline output
     - 20 * S              # high stress reduces output by 20 lines
     + 5  * C              # coffee raises output by 5 lines (true causal effect)
     + np.random.normal(0, 8, N))

df = pd.DataFrame({'S': S, 'C': C, 'Y': Y})
```

**问题1**：咖啡对产出的真实因果效应是多少？（直接从生成过程中读出来。）

---

### 第二步：计算观测到的相关性

```python
# Directly compare coffee drinkers vs. non-drinkers
obs_effect = df[df.C == 1]['Y'].mean() - df[df.C == 0]['Y'].mean()
print(f"Observed effect P(Y|C=1) - P(Y|C=0) = {obs_effect:.2f}")
```

**问题2**：观测到的效应是正的还是负的？它比真实因果效应更大还是更小？为什么？

（提示：高压员工既喝更多咖啡，产出又更低——这把咖啡和更低的产出绑在了一起。）

---

### 第三步：手写 do 运算——后门调整公式

现在你要亲手实现 $P(Y \mid do(C=1))$。

后门准则告诉我们：**S 位于 C 和 Y 之间的后门路径上**（$C \leftarrow S \rightarrow Y$）。阻断 S，因果效应就变得可识别。

公式是：

$$
P(Y \mid do(C=c)) = \sum_{s} P(Y \mid C=c, S=s) \cdot P(S=s)
$$

翻译成代码：

```python
def do_calculus(df, intervention_value):
    """
    Compute P(Y | do(C = intervention_value))

    Steps:
    1. Stratify by S; within each stratum, compute the mean of Y at C=intervention_value
    2. Weight-average using the marginal distribution of S

    This is the code implementation of "cutting the arrow C<-S."
    """
    result = 0.0

    for s_val in [0, 1]:
        # Step 1: expectation of Y when S=s and C=intervention_value
        # (in this subset, the value of C is what we "observed", not what we intervened on)
        subset = df[(df['S'] == s_val) & (df['C'] == intervention_value)]
        E_Y_given_C_S = subset['Y'].mean()

        # Step 2: marginal probability of S=s (estimated from the original data)
        P_S = (df['S'] == s_val).mean()

        result += E_Y_given_C_S * P_S

    return result

# Compute the intervention effects
E_Y_do_C1 = do_calculus(df, intervention_value=1)  # do(C=1)
E_Y_do_C0 = do_calculus(df, intervention_value=0)  # do(C=0)

causal_effect = E_Y_do_C1 - E_Y_do_C0
print(f"Expectation of Y given do(C=1): {E_Y_do_C1:.2f}")
print(f"Expectation of Y given do(C=0): {E_Y_do_C0:.2f}")
print(f"Causal effect P(Y|do(C=1)) - P(Y|do(C=0)) = {causal_effect:.2f}")
print(f"True causal effect (set at generation time) = 5.00")
```

**问题3**：`do_calculus` 函数与“先分层再取加权平均”做的是同一件事——与第二步的直接比较相比，哪一个更接近真实因果效应？

---

### 第四步：理解“切断箭头”意味着什么

do 运算的本质是构造一个**干预数据集**：强制 C 取某个值，同时让 S 保持不变（S 是 C 的原因，但 C 不是 S 的原因，所以强制喝咖啡不会改变工作压力）。

```python
# Build the intervention dataset: do(C=1) -- everyone drinks coffee
df_do_C1 = df.copy()
df_do_C1['C'] = 1   # force everyone to drink coffee
# Note: do not modify S! S is an external variable, unaffected by C

# On the intervention dataset, recompute Y (using the true generation formula)
df_do_C1['Y'] = (50
                 - 20 * df_do_C1['S']
                 + 5  * df_do_C1['C']   # C is now all 1s
                 + np.random.normal(0, 8, N))

# Build the intervention dataset: do(C=0) -- nobody drinks coffee
df_do_C0 = df.copy()
df_do_C0['C'] = 0
df_do_C0['Y'] = (50
                 - 20 * df_do_C0['S']
                 + 5  * df_do_C0['C']   # C is now all 0s
                 + np.random.normal(0, 8, N))

effect_intervention = df_do_C1['Y'].mean() - df_do_C0['Y'].mean()
print(f"Effect estimate from the direct intervention experiment: {effect_intervention:.2f}")
```

**问题4**：`do_calculus` 的结果（第三步）与“直接干预实验”的结果（第四步）应该很接近。如果它们接近，说明了什么？如果有差距，差距从哪里来？

---

### 第五步：告诉老板真相

```python
print("=" * 50)
print(f"Observed correlation (wrong basis): {obs_effect:.1f} lines/hour")
print(f"do-calculus causal effect (right basis): {causal_effect:.1f} lines/hour")
print(f"True causal effect (generation setting): 5.0 lines/hour")
print()
print("Conclusion:")
print(f"Observational data shows coffee-drinking employees output {'higher' if obs_effect > 0 else 'lower'} by {abs(obs_effect):.1f} lines")
print("But most of this gap comes from the confounder (work stress)")
print(f"The true effect of a mandatory coffee policy is only about 5 lines/hour")
```

**最终问题**：如果老板推行“三杯咖啡”政策，他期待的效果是多少？实际会发生什么？

这就是 do 运算的真正价值：**在没有随机对照试验的情况下，从观测数据中估计干预效应。**

---

### 扩展挑战（选做）

如果你想更深入，试着修改因果图，加入一个对撞节点：

```
S ───-> C ───> Y
           ↗
    Overtime (O)──-> Y
         ↑
         S
```

加班（O）也受工作压力影响，也影响产出。现在后门路径变了——你需要调整 S，但绝不能调整 O（O 是 C 的下游）。试着修改 `do_calculus` 来处理这个更复杂的图。

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.) — 因果推断的奠基之作；完整讨论 SCM、do 演算与反事实框架

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect* — 因果阶梯的通俗版本；因果革命的历史与哲学

- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* — 因果发现算法的经典教科书；PC 算法与 FCI 算法的原始出处

- Peters, J., Janzing, D. & Schölkopf, B. (2017). *Elements of Causal Inference* — 从机器学习角度讲因果推断，包括独立因果机制和加性噪声模型 `-> [MIT Press]`

- Hume, D. (1748). *An Enquiry Concerning Human Understanding* — 归纳问题的哲学起点；“过去不能证明未来”

- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If* — 流行病学与医学中的因果推断，强调干预与观测的区别 `-> [Free online]`

- Schölkopf, B. et al. (2021). *Toward Causal Representation Learning* — 因果推断与深度学习的交叉；如何从数据中学习因果结构 `-> [arXiv:2102.11107]`

- Bareinboim, E. & Pearl, J. (2016). *Causal inference and the data-fusion problem* — 如何从多个异质数据源中识别因果效应 `-> [PNAS]`

- Zhang, K. & Hyvärinen, A. (2009). *On the Identifiability of the Post-Nonlinear Causal Model* — 非线性因果模型的可识别性；打破马尔可夫等价类的一种途径 `-> [UAI 2009]`

- Uhler, C., Raskutti, G., Bühlmann, P. & Yu, B. (2013). *Geometry of the faithfulness assumption in causal inference* — 忠实性假设的几何分析及其失败条件 `-> [arXiv:1207.0547]`

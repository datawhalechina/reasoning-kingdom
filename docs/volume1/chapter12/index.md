# 第12章：隐式推理——神经网络的内部独白

> 在模型输出第一个 token 之前，它做了什么？

<div class="center">

------------------------------------------------------------------------

</div>

前四章构成了一条搜索的光谱。第 8 章：离散图上的显式搜索（A*、ADS）。第 9 章：隐状态流形上的软搜索，用 $S \times S$ 矩阵完成（Transformer）。第 10 章：在荒野中用随机模拟建图，手里没有地图（MCTS）。第 11 章：这些搜索形式每一种都要付一笔物理账单——$O(N^2)$ 的注意力矩阵、指数级的状态空间、由确定性下界钉死的存储需求。

但它们共享一个前提：搜索是显式的。你建一棵树。你算一个矩阵。你跑 N 次模拟。模型在“外面”找。

问题换到了“里面”。模型输出第一个 token 之前，显式地“搜索”之前，隐藏层里发生了什么？一次前向传播本身——那 $L$ 个欧拉步——是不是也在搜索？如果是，它搜的是什么空间？什么在约束它？它在哪里停？

答案的核心是永霖公式：$\lim_{n\to\infty}\Pi^{(n)}(s)=A\neq A^*$。搜索是显式的（外层 CoT 循环）也好，隐式的（前向传播的欧拉步）也好，推理的极限都是训练数据的统计偏置，不是真实答案。这不是工程没做好。任何在信念空间上执行压缩映射的系统，终点都落在这儿。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、黑盒里的思考

2022 年，Wei 等人试了一件笨办法：让大语言模型（LLM——在海量文本上训练、用来预测下一个词的模型）在给答案之前先把推理说出来。准确率大幅跃升。

这就是思维链（chain of thought，CoT）：一步一步推，把步骤写下来。

:::details 思维链（CoT）：为什么“把步骤说出来”能让模型变聪明？（前人工作：Wei et al., 2022）
**思维链（Chain-of-Thought，CoT）提示**是 2022 年 Google Brain 的 Wei 等人发现的技巧：在问题后面加一句“让我们一步一步思考”（let us think step by step），或者在少样本示例里摆出推理步骤，模型在复杂任务上的准确率就会往上跳。

**为什么有效？** 主流的解释是：

1. **更大的计算预算**：每生成一个 token 都要多走一步计算。让模型把中间步骤写出来，相当于允许它打草稿，而不是逼它心算。
2. **每个中间输出都变成新的输入**：模型写出的每一步都会追加到上下文里，成为下一步的输入。这就把一个难题拆成了更简单的子问题。

**一个重要的限制**：CoT 对很小的模型几乎无效；它只在约 1000 亿参数以上才涌现。这说明它依赖某种只有大模型才具备的底层能力。

于是一个问题挂在这儿：模型是在“推理”，还是在“匹配格式”？
:::

例如这个问题：“罗杰有 5 个网球。他又买了 2 罐网球，每罐有 3 个球。他现在有多少个球？”

**直接回答**：

    Model output: 11 balls
    Accuracy: 17%

**CoT 提示**：“让我们一步一步思考”

    Model output:
    Roger had 5 balls.
    He bought 2 cans with 3 balls each, so that's 2×3 = 6 balls.
    That's 5+6 = 11 balls in total.
    Answer: 11 balls
    Accuracy: 78%

17% 到 78%。参数没变，训练数据没变，变的只是模型被要求把中间步骤写出来。

那么，在模型“说出”推理之前，它自己内部发生了什么？

:::details 暂停一下
17% 到 78%。只是把步骤说出来的这个动作，计算就变了。

同一个模型，同一套权重，同样多的参数。只多了一条要求：把中间步骤写出来。

这说明什么？

一种解释：能力本来就在，CoT 只是把它放出来，给了模型多走几步计算的机会。

另一种解释：模型根本没推理。它在匹配“推理步骤的格式”。写出“罗杰有 5 个球”之后，这句话成了下一步的输入，题目被改成了统计匹配更顺手的形状。

多数时候，两种解释给出同样的预测。可它们对“模型是什么”的回答，差得很远。

哪一种是真？有没有一个实验能把两者分开？

先搁着。
:::

<div class="center">

------------------------------------------------------------------------

</div>

## 二、隐层：推理的暗室

前向传播是一层接一层的变换：

$$
h_0 = x \quad \to \quad h_1 = f_1(h_0) \quad \to \quad h_2 = f_2(h_1) \quad \to \quad \cdots \quad \to \quad y = f_L(h_{L-1})
$$

第 $l$ 层的 $h_l$ 是隐藏表示：输入在那一层被编码成什么样。

这些层在干什么？

浅层抓边缘、纹理，深层抓物体、概念——这是感知任务里的说法。放到推理上，事情细一点。

2022 年，Olsson 等人盯上了 GPT 的归纳头（induction heads）：一种专门“看见模式，接下一个”的注意力头。

输入序列是“A B C A B ？”。

归纳头看见“A B”又出现了一遍，把下一个 token 押在“C”上。

隐藏层里有一份像工作记忆的东西：模型留着刚才的模式，拿它推下一步。

这份记忆有多深？能撑多长的推理？

<div class="center">

------------------------------------------------------------------------

</div>

## 三、隐层里到底发生了什么

把问题说得再准一点。

CoT 有用，不是因为它给模型塞进了新的逻辑规则。它把压在一次前向传播里的东西摊开，摊成一串能写下来的中间状态。写出来的是表面；先发生的，还是隐藏层里的状态演化。

答案出口之前，隐藏层干了多少活？这是本章要追的。

最自然的想法：一次前向传播就是一场逐层展开的内部独白。输入不会一步变成答案。它先变成中间表示，再被重组、压缩、放大，最后才投影到输出。写成公式：

$$
h_0 = x \quad \to \quad h_1 = f_1(h_0) \quad \to \quad h_2 = f_2(h_1) \quad \to \quad \cdots \quad \to \quad y = f_L(h_{L-1})
$$

每个 $h_l$ 都不是答案，是答案成形之前的内部状态。不是输入，也不是输出，是一个还没定下来的判断。

感知任务里，浅层看边缘，深层看物体。推理任务里更像：浅层把问题改写成模型能用的坐标，深层在坐标里整合关系。

Olsson 等人在 2022 年对 GPT 的研究，重要就重要在这儿。有些注意力头干的是工作记忆的活：看见一个模式，就持续跟着它。输入“A B C A B ？”，归纳头把前面的“A B”和后面的“A B”对上，下一个 token 押“C”。

重点不是“会补全序列”，那太浅。隐藏层不是静态的特征仓库。它维持一个能操作的内部状态：留住局部关系，对齐远处的片段，把见过的结构带进当下接着算。这已经很像我们说的“中间推理”。

所以问题不是模型有没有想法。问题是这个状态能演化多远。

能撑多步推理吗？一步都不写下来，内部能把逻辑稳稳传下去吗？

永霖公式给的答案不客气：走一阵可以，走不了无限远；对象层能推进，元层认不了自己。临时窗口总会被先验锚点收回去。

说 CoT“教会模型把想法说出来”，不够。CoT 只是让这场内部独白多有用一会儿。

如果里面真有一场独白，链子变长的时候，它从哪里开始失真？

<div class="center">

------------------------------------------------------------------------

</div>

## 四、CoT：让隐式推理显式化

CoT 干了什么？一种说法：它把隐式推理拽到明面上。

平时，推理发生在隐藏层里，一次前向传播从输入走到输出。这个过程是压缩的：所有步骤挤进有限的层数。

CoT 生成中间 token，换来的其实是计算时间：

- 每生成一个 token，就要多走一次完整的前向传播
- 一条 10 个 token 的推理链，就要走 10 次前向传播
- 这给了模型 10 倍的计算深度

人也这样。题难的时候不立刻答，把步骤写在纸上，一步一步往前挪。

可这里有个分不清的地方：它是在推理，还是在表演推理？

<div class="center">

------------------------------------------------------------------------

</div>

## 五、永霖公式：先验从哪里来？

永霖公式给 CoT 划了一条底：链子再长，最后都会收回到先验锚点。

公式是这样：

记号先说清楚：这里的 $\Pi^{(n)}(s)$ 就是第七节的信念分布 $p_t$，$n$ 就是 $t$。严格推导在第七节，用压缩映射和巴拿赫不动点。

$$
\lim_{n \to \infty} \Pi^{(n)}(s) = A, \quad \text{但} \quad A \neq A^*
$$

读它之前，先认三个东西：$\Pi^{(n)}(s)$、$A$、$A^*$。

### 先验锚点 $A$ 从哪里来？

先验锚点不神秘。它是训练数据的统计偏置。

假设训练集里 50% 是正例（链条完整，答“是”），50% 是负例（链条断裂，答“否”）。

这样的训练集，会让模型学出一个默认倾向：不知道怎么答，就五五开猜。

这个默认倾向就是先验锚点 $A$。没证据的时候，模型退回到这个信念上。

写成式子：

$$
A = P_{\text{train}}(Y)
$$

训练集标签的边缘分布。

训练集 60% 是正例，$A$ 就往“是”偏，约 0.6。训练集平衡，$A$ 就是 0.5。

这个锚点是模型从数据里吸收来的偏置。不是错误。它是模型对“世界平均长什么样”的最好猜测。

### 推断分布 $\Pi^{(n)}(s)$ 是什么？

$\Pi^{(n)}(s)$ 是模型推了 $n$ 步之后，对答案的概率分布。

- $n=0$：一步没推，分布就是先验 $A$
- $n=1$：看见第一条证据，分布更新一次
- $n=10$：处理了 10 个推理步骤
- $n\to\infty$：无限推理后的极限

永霖公式说：推理无穷多步，分布最后也回到 $A$，不是真实答案 $A^*$。

### $A^*$ 是什么？

$A^*$ 是真实答案的分布。

链条完整（A>B>C>…>Z 全成立），“A>Z”的答案就是 100% 的“是”。$A^*=1.0$，一个确定性为 1 的分布。

模型的极限是 $A=0.5$（先验偏置），不是 $A^*=1.0$。

$A\neq A^*$ 的意思很直白：推理走到头，不是正确答案。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、隐式推理的哲学意义

永霖公式说的不是哪家的模型没做好。推理本身就不完备。

系统够强，就会撞上元层断裂：

- 它能在对象层生成推理链
- 但无法在元层验证推理链的对错
- 最后只能退回先验锚点

这不是规模的事。GPT-4o 有 2000 亿参数，照样收敛。只是收敛点可能离真实分布近一点（$A$ 更接近 $A^*$），但不会正好落在上面。

CoT 能做的是把 $t_{\text{eff}}$ 沿 $\gamma$ 变小的方向往前推。$\gamma$ 越小，压得越快，窗口越长。终点 $A$ 不动。

这样看，隐式推理（隐藏层内部）和显式推理（CoT 生成 token）是同一件事：都在有限的计算深度里朝正确答案挤。

区别只有一条：

- 隐式推理：压缩在一次前向传播中——快但浅
- 显式推理：展开为多次前向传播——慢但深

两条路都受永霖公式管。都会收敛。

永霖公式画的是推理的内边界。下一章是最后一章，把这些边界拼在一起：推理王国的地图。

<div class="center">

------------------------------------------------------------------------

</div>

## 七、永霖极限的严格推导：从不动点到压缩映射

### 1. 从最朴素的不动点概念说起

不动点：最简单的定义。对于函数 $f: X \to X$，如果存在 $x^* \in X$ 使得 $f(x^*) = x^*$，则称 $x^*$ 是 $f$ 的一个不动点。

定义看似平凡，但它藏着一层很深的哲学：自指。系统的输出等于它的输入；原因等于结果。

### 2. 迭代一个序列：离散动力系统

考虑一个由递推定义的序列 $\{x_t\}_{t=0}^\infty$：

$$
x_{t+1} = F(x_t), \quad t = 0, 1, 2, \dots
$$

这是一个离散动力系统。我们想知道：当 $t\to\infty$ 时，$x_t$ 会发生什么？

### 2.1 一个具体的例子

最简单的线性情形：$F(x) = ax + b$。

- 不动点方程：$x^* = ax^* + b \Rightarrow x^* = \frac{b}{1-a}$（若 $a \neq 1$）
- 迭代：$x_{t+1} = ax_t + b$
- 解：$x_t = a^t x_0 + \frac{b(1-a^t)}{1-a}$
- 收敛：当 $|a| < 1$ 时，$\lim_{t\to\infty} x_t = x^*$

那个 $|a| < 1$ 就是压缩性的第一个形状。

### 3. 建立信念空间

在推理系统中，$x_t$ 不是一个实数；它是一个信念分布——答案空间上的一个概率向量。

令 $\mathcal{P} = \{ p \in \mathbb{R}^n : p_i \geq 0,\ \sum_i p_i = 1 \}$ 为概率单纯形。

:::details 从 {0,1} 到 [0,1]：信念为什么应该是定量的？
在经典逻辑里，一个命题要么真要么假。“明天会下雨”是 1 或 0。状态空间是一组离散的顶点。

但你并不知道哪个是真相。你能知道的只是自己的确信程度。

贝叶斯式的转向：把“世界的状态”换成“你对世界的认知状态”。

- 经典逻辑：$s \in \{0, 1\}^n$；每个命题为真或为假
- 贝叶斯信念：$p \in [0, 1]^n$；每个命题带一个置信度

那个 0.7 不是“世界有 70% 的概率下雨”，而是你，作为认知主体，对这个命题有 0.7 的置信度。世界本身仍然是 0 或 1——模糊的是你的知识，不是世界。

再加上归一化约束 $\sum_i p_i = 1$（你的置信度必须加总为 100%），就得到了概率单纯形。
:::

:::details 为什么单纯形是 n-1 维而不是 n 维？
有 $n$ 个答案选项，信念向量 $p = (p_1, p_2, \ldots, p_n)$ 看起来是 $n$ 维的。

但有一个约束：$p_1 + p_2 + \cdots + p_n = 1$。

这个约束移除了一个自由度。你只需要自由选择前 $n-1$ 个分量；最后一个就完全确定了：

$$p_n = 1 - p_1 - p_2 - \cdots - p_{n-1}$$

类比：直线上的点是 1 维的，但如果我们约束 $x + y = 1$，原本 2 维的平面就被切成了 1 维的线段。$n$ 个变量加一个等式约束：自由度是 $n-1$。

具体地：

- $n=2$（二选一）：单纯形是 $[0, 1]$ 里的一条线段——1 维
- $n=3$（三选一）：单纯形是平面上的一个等边三角形——2 维
- $n=4$（四选一）：单纯形是空间里的一个正四面体——3 维

每个顶点对应“对某个答案完全确定”（经典逻辑的状态）。内部的点是贝叶斯的领地——一片连续的不确定性光谱。
:::

:::details 单纯形的几何本质：凸集的定义里藏着什么？
单纯形不只是一个几何形状；它是凸性的最小单元。

**凸集定义**：集合 $C$ 是凸的，当且仅当对任意 $x_1, \ldots, x_k \in C$，只要 $\theta_i \geq 0$ 且 $\sum_i \theta_i = 1$，就有 $\sum_i \theta_i x_i \in C$。

看那个条件：$\theta_i \geq 0$，$\sum_i \theta_i = 1$。这正是单纯形 $\Delta_{k-1}$ 的坐标定义。

所以凸集的定义可以重读为：$C$ 是凸的，当且仅当 $C$ 的任意有限点集上的单纯形的仿射像仍然落在 $C$ 内。凸性的本质，是在单纯形坐标的线性插值下封闭。

单纯形是“刚好够用的凸性容器”：$n+1$ 个仿射无关的点张成 $n$ 维空间，一个都不多。

---

线性规划的单纯形法共享同一套直觉。线性规划在凸多面体（可行域）上最小化一个线性目标。关键事实是：在凸多面体上，线性函数必在某个顶点取到最优值。

单纯形法的策略：从一个顶点出发，沿棱走到相邻顶点，只要目标下降就继续走，直到没有邻居能改进。绝不进入内部；只在边界上走，因为最优解藏在角上。

两个单纯形，同一个舞台，相反的策略。目标函数的性质定的，不是排的：

| 场景 | 目标函数 | 最优解位置 | 策略 |
|---|---|---|---|
| 单纯形法（线性规划） | 线性 | 顶点 | 沿棱贪心游走，绝不进入内部 |
| 信念更新（贝叶斯） | 严格凸（KL 散度、交叉熵） | 内部 | 梯度下降，深入内部 |
| 边际分析（经济学） | 严格凹（边际收益递减） | 内部 | 均衡在内部，不在角上 |

线性函数没有曲率，所以极值只能出现在边界上——单纯形法正是利用这一点，把搜索压缩到有限的顶点集上。严格凸函数有唯一的内部最优，梯度下降会把你拉向内部。

概率单纯形 $\mathcal{P}$ 的顶点是确定的信念（经典逻辑的 0/1 世界）；内部是连续的不确定性光谱（贝叶斯的领地）。推理就是在这座单纯形上找到你应该站的位置——而那个位置几乎总在内部。
:::

推断算子 $F: \mathcal{P} \to \mathcal{P}$ 把当前信念映射为更新后的信念。

### 4. KL 散度：信念之间自然的距离度量

对两个分布 $p, q \in \mathcal{P}$，KL 散度：

$$
D_{\text{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

性质：

1. $D_{\text{KL}}(p \| q) \geq 0$（吉布斯不等式）
2. $D_{\text{KL}}(p \| q) = 0 \iff p = q$
3. 凸性：$D_{\text{KL}}(p \| q)$ 关于 $(p, q)$ 联合凸

但 KL 散度不是度量：它不对称，也不满足三角不等式。所以 KL 不能承载巴拿赫定理所需的距离。它在下面证明中的作用是能量函数及其曲率；巴拿赫一步所需的“距离”，由切空间上的欧氏范数承载（第 7–8 步）。

### 5. 关键洞察：推断算子的具体形式

压缩性不是假设出来的；它来自推断算子的具体构造。

### 5.1 梯度下降视角

对一个训练好的神经网络，一步推理可以看作能量面上的梯度下降。

令能量为 $E: \mathcal{P} \to \mathbb{R}$；推断算子：

$$
F(p) = \operatorname{proj}_{\mathcal{P}}\left( p - \eta \nabla E(p) \right)
$$

其中 $\operatorname{proj}_{\mathcal{P}}$ 是到概率单纯形上的投影，$\eta > 0$ 是步长。

### 5.2 离散化欧拉迭代

连续梯度流：$\frac{dp}{dt} = -\nabla E(p)$

欧拉离散化（步长 $\Delta t = \eta$）：

$$
p_{t+1} = p_t - \eta \nabla E(p_t)
$$

加上保持概率的投影：

$$
p_{t+1} = \operatorname{proj}_{\mathcal{P}}\left( p_t - \eta \nabla E(p_t) \right)
$$

这正是 $F$ 的形式。

:::details 这里的欧拉步是什么？
连续梯度流 $\frac{dp}{dt} = -\nabla E(p)$ 描述的是一个信念分布在能量面上连续地往下流。但计算机只能一小步一小步地走；它做不到真正的“连续”。

欧拉法把连续的流切成离散的小步：

$$p_{t+1} = p_t - \eta \nabla E(p_t)$$

意思是：当前信念是 $p_t$；那里能量下降的方向是 $-\nabla E(p_t)$；沿这个方向走一步 $\eta$，就得到新信念 $p_{t+1}$。

这跟 A* 搜索的结构完全一样：你在当前节点 $s_t$，测一下各个方向的代价，然后挑最好的方向走一步。唯一的区别是：

- A* 的舞台是离散的节点图；它走的是代价最小的方向
- 这里的舞台是概率单纯形；它走的是能量下降最陡的方向

加上 $\operatorname{proj}_{\mathcal{P}}$ 是因为走完一步之后，结果可能跑出概率单纯形（某个分量变成负数），必须投影回来，保持“所有分量非负且和为 1”的约束。

欧拉步的误差：步长 $\eta$ 越大，每一步走得越远，但偏离真实连续轨迹也越多。步长太大就发散，太小就收敛慢。这就是学习率调参的本质。
:::

### 6. KL 散度作为 Bregman 散度

KL 散度可以写成 Bregman 散度。令 $\phi(p) = \sum_i p_i \log p_i$（负熵）；则：

$$
D_{\text{KL}}(p \| q) = \phi(p) - \phi(q) - \langle \nabla\phi(q), p-q \rangle
$$

Bregman 散度满足三点恒等式：

$$
D_{\text{KL}}(p \| q) = D_{\text{KL}}(p \| r) + D_{\text{KL}}(r \| q) - \langle \nabla\phi(q) - \nabla\phi(r), p-r \rangle
$$

### 7. 压缩性的证明（核心）

现在我们来证明：梯度下降形式的推断算子 $F$，在合适的条件下是一个压缩。

**设定**：取一个闭凸区域 $\mathcal{X} \subseteq \mathcal{P}$，包含单纯形内部的锚点 $A$（例如 $\mathcal{X}_\varepsilon = \{p: p_i \geq \varepsilon\}$，其中 $\varepsilon \leq \min_i A_i$）；令 $\Pi_{\mathcal{X}}$ 为到 $\mathcal{X}$ 的欧氏正交投影；令切空间为 $T = \{v: \mathbf{1}^\top v = 0\}$，$P_T$ 为到 $T$ 的正交投影。推断算子为欧氏投影梯度

$$
F(p) = \Pi_{\mathcal{X}}\left( p - \eta P_T \nabla E(p) \right).
$$

这个设定背后的三个选择，以及它们的理由：

1. **度量用切空间上的欧氏范数，而不是 KL。** KL 散度不对称、不满足三角不等式，单纯形在 KL 下不是完备度量空间，巴拿赫定理站不住。KL 被用在该用的地方：作为能量函数，它的 Hessian 决定 $F$ 往哪走、一步能走多大（第 4、6 步）。
2. **算子用欧氏投影梯度。** 下面的定理讨论的正是这个算子——带欧氏投影和切空间修正的梯度下降。它不会自动迁移到其他投影（例如真实模型用的 softmax/Bregman 投影）；每种投影定义自己的系统，本定理只覆盖这一种。
3. **投影是非扩张的。** 到闭凸集的欧氏投影永不放大距离——这正是证明所需的那一条不等式（第 2 步）。

（为什么需要 $\mathcal{X}$：KL 的 Hessian $\operatorname{diag}(q_i/p_i^2)$ 在边界上爆炸，谱界只在远离边界处存在。为什么需要 $P_T$：$\nabla E$ 的法向分量是规范自由度——沿法向移动不改变信念本身。）

**定理**：假设在 $\mathcal{X}$ 上，切空间 Hessian 有一致的谱界 $\mu I_T \preccurlyeq P_T \nabla^2 E(p) P_T \preccurlyeq L I_T$（$\mu, L > 0$）。取步长 $\eta \in (0, 2\mu/L^2)$。则 $F$ 是 $(\mathcal{X}, \|\cdot\|_2)$ 上的压缩映射，压缩因子为

$$
\gamma = \sqrt{1 - 2\eta\mu + \eta^2 L^2} < 1.
$$

（第23章会在切空间里把同一个压缩因子在最优步长下精确算出来；本章的 $\gamma$ 是一致谱界下的保守版本。）

**证明**：

**第 1 步：把谱界翻译成两个不等式。** 对任意 $p_1, p_2 \in \mathcal{X}$（注意 $p_1 - p_2 \in T$），由积分中值定理和谱界：

$$
\langle P_T(\nabla E(p_1) - \nabla E(p_2)),\ p_1 - p_2 \rangle \geq \mu \|p_1 - p_2\|_2^2,
$$

$$
\|P_T(\nabla E(p_1) - \nabla E(p_2))\|_2 \leq L \|p_1 - p_2\|_2.
$$

（下界：谱下界是沿切方向的强凸性；上界：谱上界是沿切方向的光滑性。第一个不等式用了 $p_1 - p_2 \in T$ 和 $P_T$ 的对称性。）

**第 2 步：投影不放大距离。** 到闭凸集的欧氏投影是非扩张的：$\|\Pi_{\mathcal{X}}(x) - \Pi_{\mathcal{X}}(y)\|_2 \leq \|x-y\|_2$。因此

$$
\|F(p_1) - F(p_2)\|_2^2 \leq \left\|(p_1-p_2) - \eta P_T(\nabla E(p_1) - \nabla E(p_2))\right\|_2^2.
$$

**第 3 步：展开。**

$$
\begin{aligned}
\|F(p_1) - F(p_2)\|_2^2 &\leq \|p_1-p_2\|_2^2 - 2\eta\langle P_T(\nabla E(p_1) - \nabla E(p_2)),\ p_1-p_2\rangle \\
&\quad + \eta^2\|P_T(\nabla E(p_1) - \nabla E(p_2))\|_2^2 \\
&\leq (1 - 2\eta\mu + \eta^2L^2)\|p_1-p_2\|_2^2 = \gamma^2\|p_1-p_2\|_2^2.
\end{aligned}
$$

$2\mu - \eta L^2$ 就是从这来的：强凸性贡献正的 $2\mu$，光滑性贡献负的 $\eta L^2$，两者之差决定每一步压缩多少。步长上界 $\eta < 2\mu/L^2$ 正是让 $\gamma < 1$ 的条件（$\gamma^2 \geq 1 - \mu^2/L^2 \geq 0$；非负由 $\mu \leq L$ 保证）。

**证毕。**

我们在这里得到的是整个区域上的一致 Lipschitz 压缩：对任意两点 $p_1, p_2 \in \mathcal{X}$ 都成立，而不只是“越来越靠近 $A$”。这才是巴拿赫意义上的压缩映射。

:::info 波兰学派：完备性从哪里来
第七、八步用到的完备性，来自斯特凡·巴拿赫（Stefan Banach）1922 年证明的压缩映射原理。巴拿赫是两次世界大战之间波兰数学学派的核心人物之一；同一个学派里还有乌拉姆、舒德、马祖尔等人。泛函分析里的巴拿赫空间、哈恩–巴拿赫定理、巴拿赫–施坦豪斯定理，都带着这个学派的名字。

这不是一段增加气氛的数学史。它解释了这个证明的方法论选择：**把“收敛”从具体算法里抽出来，放进一个完备空间，只研究算子本身。** 一旦这么做，收敛性就不再依赖你用的是 SGD、Adam 还是别的更新规则，而只依赖算子在不动点处的压缩性。

本章的能量面来自信息论，完备性来自波兰学派。永霖定理就站在这两根柱子上。
:::

### 8. 应用巴拿赫不动点定理

现在应用巴拿赫不动点定理：

由于 $F$ 是压缩映射（$\gamma < 1$），且 $(\mathcal{X}, \|\cdot\|_2)$ 是 $\mathbb{R}^n$ 的闭子集、在欧氏度量下完备（第 7 步的设定），则：

1. 存在唯一不动点 $A \in \mathcal{X}$，使 $F(A) = A$
2. 对任意初始 $p_0 \in \mathcal{X}$，迭代 $p_{t+1} = F(p_t)$ 收敛到 $A$
3. 收敛速率：$\|p_t - A\|_2 \leq \gamma^t \|p_0 - A\|_2$

注意，度量是欧氏范数，不是 KL。KL 散度在这里的角色是能量函数——它决定 $F$ 往哪走，并通过 Hessian 决定一步能走多大。

### 9. 识别先验锚点

训练最小化经验风险：

$$
\min_{p \in \mathcal{P}} \mathbb{E}_{(x,y) \sim D}[\mathcal{L}(p, y)]
$$

对交叉熵损失 $\mathcal{L}(p, y) = -\log p_y$，最优解为：

$$
A(y) = P_D(y) = \frac{\text{训练集中标签 } y \text{ 的计数}}{\text{总样本数}}
$$

关键观察：这个 $A$ 恰好是推断算子的不动点。因为对 $E(p) = \mathbb{E}_D[-\log p_y]$，

$$
\nabla_i E(p) = -\frac{q_i}{p_i}, \qquad \nabla E(A) = (-1, \ldots, -1)
$$

$\nabla E(A)$ 完全沿单纯形的法向——它在切空间上的投影为零：$P_T \nabla E(A) = 0$。于是

$$
F(A) = \Pi_{\mathcal{X}}(A - \eta \cdot 0) = A,
$$

即 $A$ 是 $F$ 的不动点。这正是 $P_T$ 出现在算子里的原因：切掉法向分量之后，不动点是精确的。在 $A$ 处梯度是 $(-1, \ldots, -1)$，不是零向量；要记住的命题不是“梯度为零”，而是“梯度沿法向，切空间分量为零”。

### 10. 永霖定理的完整陈述

永霖定理（严格版）：假设一个推理系统满足：

1. **区域与谱假设**：存在一个包含先验锚点 $A$ 的闭凸区域 $\mathcal{X} \subseteq \mathcal{P}$（例如 $\mathcal{X}_\varepsilon$），且能量 $E$ 的切空间 Hessian 在 $\mathcal{X}$ 上有一致谱界：$\mu I_T \preccurlyeq P_T \nabla^2 E(p) P_T \preccurlyeq L I_T$（$\mu, L > 0$）
2. **梯度下降形式**：推断算子 $F(p) = \Pi_{\mathcal{X}}(p - \eta P_T \nabla E(p))$，步长 $\eta \in (0, 2\mu/L^2)$
3. **训练一致性**：$E$ 的最小点 $A = \arg\min_{p \in \mathcal{P}} E(p)$ 对应训练数据的经验分布

则：

1. $F$ 是 $(\mathcal{X}, \|\cdot\|_2)$ 上的压缩映射，压缩因子 $\gamma = \sqrt{1 - 2\eta\mu + \eta^2L^2} < 1$
2. $A$ 是 $F$ 的唯一不动点
3. 对任意初始信念 $p_0 \in \mathcal{X}$，$\lim_{t\to\infty} p_t = A$
4. 收敛指数：$\|p_t - A\|_2 \leq \gamma^t \|p_0 - A\|_2$，且 $D_{\text{KL}}(A\|p_t) \leq \kappa\gamma^{2t} D_{\text{KL}}(A\|p_0)$，其中 $\kappa = L/\mu$

如果真实答案 $A^* \neq A$，那么无论系统推理多少步，都永远到不了 $A^*$。

**注（一条甚至不需要巴拿赫的路径）**：若对某个 $\nu \in (0, 1)$，对每个 $p \in \mathcal{X}$ 都有 $E(F(p)) \leq \nu E(p)$，则不动点唯一，且 $p_t \to A$ 完全不需要巴拿赫：迭代给出 $E(p_t) \leq \nu^t E(p_0)$；不动点 $q$ 满足 $E(q) = E(F(q)) \leq \nu E(q)$，而 $\nu < 1$ 迫使 $E(q) = 0$，即 $q = A$；Pinsker 不等式 $D_{\text{KL}}(A\|p_t) \geq \frac{1}{2}\|p_t - A\|_1^2$ 给出 $\|p_t - A\|_1 \leq \sqrt{2\nu^t E(p_0)} \to 0$。当 $\kappa\gamma^2 < 1$ 时，本定理直接满足该假设（取 $\nu = \kappa\gamma^2$）。

### 10.7 前向传播 = 推理：欧拉步 + 单纯形投影

在前面讲前向传播时，兔狲教授在白板上画了一个陡峭的山坡，对小猪和小海豹说：“沿着负梯度方向更新参数，步长要合适。”

那是他在描述训练——在参数空间 $\Theta$ 里走梯度下降：

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

这个描述完全正确，但它只讲了一半。另一半是：前向传播本身就是推理。

#### 拆开一次神经网络的前向传播

任意一个 Transformer 层的前向传播：

```
h_0 = x + positional_encoding          # initial state
h_1 = h_0 + Attention(h_0)             # step 1
h_2 = h_1 + FFN(h_1)                   # step 2
h_3 = h_2 + Attention(h_2)             # step 3
h_4 = h_3 + FFN(h_3)                   # step 4
...
h_L = ...                               # step L
y = softmax(h_L)                        # projection
```

盯着残差连接看——$h_k = h_{k-1} + F(h_{k-1})$。这就是欧拉迭代：

$$
x_{t+1} = x_t + \eta \cdot v(x_t)
$$

步长 $\eta = 1$，速度 $v = \text{Attention}$ 或 $\text{FFN}$。三个要素都齐了。

那没有残差连接的普通全连接层呢？$h_k = \sigma(W_k h_{k-1} + b_k)$。把它拆成“当前状态 + 增量”：

$$
h_k = h_{k-1} + \underbrace{\left(\sigma(W_k h_{k-1} + b_k) - h_{k-1}\right)}_{\text{速度 } v(h_{k-1})}
$$

一样。任何隐藏状态传递，只要能写成 $h_k = h_{k-1} + \Delta_k$：那么 $\Delta_k$ 就是速度，这一层就是欧拉步。残差连接只是把这个结构公开暴露出来——把 $\eta = 1$ 和速度函数 $v$ 分开写。

#### 欧拉步的三个要素

不管什么网络，只要在传递隐藏状态，就可以被建模为欧拉迭代。三个要素：

| 要素 | 一般形式 | 残差连接（显式） | 全连接层（隐式） |
|---|---|---|---|
| 当前状态 | $x_t$ | $h_{k-1}$ | $h_{k-1}$ |
| 速度函数 | $v(x_t)$ | $\text{Attention}(h_{k-1})$ | $\sigma(W_k h_{k-1} + b_k) - h_{k-1}$ |
| 步长 | $\eta$ | $1$（硬编码） | $1$（隐含） |
| 下一状态 | $x_{t+1} = x_t + \eta \cdot v(x_t)$ | $h_k = h_{k-1} + \text{Attention}(h_{k-1})$ | $h_k = h_{k-1} + (\sigma(W_k h_{k-1} + b_k) - h_{k-1})$ |

RNN 的隐状态递推 $h_t = \tanh(W h_{t-1} + U x_t)$ 也是欧拉步——当前状态是 $h_{t-1}$，速度是 $\tanh(W h_{t-1} + U x_t) - h_{t-1}$。Mamba 的离散化 SSM $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ 也是欧拉步——速度 $(\bar{A} - I)h_{t-1} + \bar{B} x_t$。

任何神经网络的隐藏状态传递都是一个离散动力系统。层数 = 欧拉步数。

#### 前向传播 = 推理

于是前向传播的完整形状是：

$$
\underbrace{h_0 \xrightarrow{\text{欧拉步}} h_1 \xrightarrow{\text{欧拉步}} \cdots \xrightarrow{\text{欧拉步}} h_L}_{\text{隐状态空间上的离散动力系统：} L \text{ 个欧拉步}} \xrightarrow{\text{softmax}} \underbrace{p}_{\mathcal{P}}
$$

这就是推理。隐藏状态逐层演化，走 $L$ 个欧拉步。每一步轻推它一下；$L$ 步之后它落在 $h_L$ 的某个位置。最后 softmax 把那个点投影到概率单纯形 $\mathcal{P}$ 上。

softmax 是 $\operatorname{proj}_{\mathcal{P}}$ 的一种具体实现：

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$z \in \mathbb{R}^n$ 可以是任何东西；输出 $p \in \mathcal{P}$ 一定合法。softmax 是指数族的最大似然投影，它自然地把 logit 空间映射到概率单纯形上。

CoT 只是多跑几轮：每生成一个 token，隐藏状态就再走 $L$ 个欧拉步，再投影一次。

$$
\text{CoT} = (\text{欧拉步}^L \circ \text{softmax}) \circ (\text{欧拉步}^L \circ \text{softmax}) \circ \cdots
$$

#### 反向传播是另一个范式

训练不发生在隐藏状态空间里。训练在参数空间 $\Theta$ 里走梯度下降：

$$
\theta_{t+1} = \theta_t - \hat{\eta} \nabla_\theta \mathcal{L}(\theta_t)
$$

反向传播只是计算 $\nabla_\theta \mathcal{L}$ 的方法——因为 $\mathcal{L}$ 是一个 $L$ 层复合：

$$
\mathcal{L}(\theta) = \ell \circ f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
$$

链式法则穿过 $L$ 层：

$$
\nabla_{\theta_i} \mathcal{L} = J_{f_L} \cdot J_{f_{L-1}} \cdot \cdots \cdot J_{f_{i+1}} \cdot \frac{\partial f_i}{\partial \theta_i}
$$

每个 $J_{f_k}$ 是第 $k$ 层的 Jacobian。从后往前乘（矩阵-向量，$O(d^2)$）而不是从前往后乘（矩阵-矩阵，$O(d^3)$）——反向传播就这么多。一个计算技巧；它没有创造新的数学。

#### 两个范式，不是一个

前向传播和反向传播不是“同一组方程在不同空间里”。它们是两种完全不同的动力学：

| | 前向传播 = 推理 | 反向传播 = 训练 |
|---|---|---|
| 空间 | 隐藏状态空间 $\mathcal{H}$ | 参数空间 $\Theta$ |
| 动力学类型 | 离散动力系统（欧拉步） | 连续优化（梯度下降） |
| 迭代公式 | $h_k = h_{k-1} + v_k(h_{k-1})$ | $\theta_{t+1} = \theta_t - \hat{\eta} \nabla_\theta \mathcal{L}$ |
| 速度来源 | 层变换 $F_k$（注意力、FFN 等） | 损失梯度 $\nabla_\theta \mathcal{L}$ |
| 步数 | 固定（$L$ 层） | 可变（训练步数） |
| 终点 | softmax -> $\mathcal{P}$ | $\theta^*$（由训练数据决定） |
| 优化什么？ | 什么都不优化——执行一条固定轨道 | 最小化损失函数 |

前向传播不在“优化”任何东西。每一层的速度函数 $v_k$ 由训练好的权重决定。前向传播只是执行已经铺好的轨道——$L$ 个欧拉步，softmax 投影，输出答案。

反向传播在优化。它改变权重 $\theta$，从而改变下一次前向传播轨道的形状。训练让它更陡（$\mu$ 变大）、更平滑（$L$ 变小），于是前向传播的 $L$ 个欧拉步更快到达正确的位置。

#### 唯一的交汇点：$\operatorname{proj}_{\mathcal{P}}$

两个范式在 softmax 处交汇。

训练从不碰 $\mathcal{P}$——它在 $\Theta$ 里走梯度下降，$\Theta = \mathbb{R}^d$ 是平的，不需要投影。偏导数直接减掉；参数可以取任何值。

前向传播在末端碰 $\mathcal{P}$——隐藏状态 $h_L$ 被 softmax 投影成概率分布。softmax 是神经网络结构的一部分，不是外挂上去的。

为什么 softmax = $\operatorname{proj}_{\mathcal{P}}$ 必须是信息几何意义上的投影，而不是简单的截断？因为 $\mathcal{P}$ 不是欧氏空间。考虑：

$$
p = (0.99, 0.01)
\quad \text{对比} \quad
q = (0.01, 0.99)
$$

欧氏距离：$\|p-q\|_2 \approx 1.39$。但欧氏距离告诉你 $(0.5, 0.5)$ 离这两个点“差不多远”——这完全无法反映“确定 A”和“确定 B”之间巨大的认知鸿沟。

KL 散度：$D_{\text{KL}}(p\|q) \approx 4.60$——两个极端信念之间的 KL 距离远比从均匀出发的那个大。KL 捕捉到了信念的方向性：从“几乎确定 A”走到“几乎确定 B”，是一段巨大的认知距离。

所以 $\operatorname{proj}_{\mathcal{P}}$ 不是“截断到合法范围”。它是在 KL 散度（Bregman 散度）的几何意义上做的投影——把信念保持在单纯形流形上，而不是欧氏平面上。

#### 永霖极限约束两个范式

训练：在 $\Theta$ 里的梯度下降收敛到 $\theta^*$。$\theta^*$ 由训练数据上的损失最小点决定——也就是先验锚点 $A = P_D(Y)$。

推理：欧拉步在 $\mathcal{H}$ 里走 $L$ 步，softmax 投影到 $\mathcal{P}$。但如果推理链被拉长（多轮 CoT），隐藏状态被 softmax 投影之后再次出发，每一步都在单纯形上重新锚定自己。足够多轮之后，隐藏状态轨迹被吸引子 $A$ 捕获——这就是永霖极限。

两个范式，同一个终点：$A$。训练把 $A$ 编码进权重矩阵；推理被 $A$ 雕刻出的能量面拉向 $A$。$A$ 是训练数据的统计偏置，不是真实答案 $A^*$。前向传播走多少个欧拉步、CoT 拉多长——隐藏状态动力学的吸引子都是 $A$。

#### 两句话

前面兔狲教授说：“沿着负梯度方向更新参数，步长要合适。”

那是训练。完全正确。

再加一句：前向传播的每一层都是欧拉步。$L$ 个欧拉步之后，softmax 把隐藏状态投影到单纯形上。那是推理。训练改变欧拉步的速度场；推理在那个速度场上执行隐藏状态轨迹。

两个范式：一个优化（在 $\Theta$ 里走梯度下降），一个执行（在 $\mathcal{H}$ 里走欧拉步 + $\operatorname{proj}_{\mathcal{P}}$）。它们在 softmax 处交汇。都以 $A$ 为终点。

<div class="center">

------------------------------------------------------------------------

</div>

## 八、元层断裂的数学表述

令 $V: \mathcal{P} \to \{0, 1\}$ 为验证函数（$V(p) = 1$ 表示信念 $p$ 正确），其真值由外部真实答案 $A^*$ 定义。元层验证器如果存在，应该是一个由推断算子 $F$（等价地，由能量函数 $E$）构造的可计算映射 $M$，使得对任意 $p \in \mathcal{P}$：

$$
V(F(p)) = M(F, p) \qquad \forall p \in \mathcal{P}
$$

**命题（开放问题）**：当 $A \neq A^*$ 时，这样的 $M$ 不存在。

为什么这是开放问题而不是定理：早期版本试图用“若 $V$ 不是 $E$ 的凸函数，则 $M$ 不存在”来证明它——但那个论证不成立：对固定的 $F$，$V \circ F$ 本身就是一个定义良好的函数；凸性和单调性与 $M$ 的存在性无关。旧的证明路线把“尚未证明”误当成了“已经证明”。

真正的问题在语义域：$V$ 的定义涉及 $A^*$（外部真理），而 $F$ 的构造只涉及 $A$（训练统计）。用范畴语言说：$V$ 是外部范畴 $\mathcal{R}$ 的谓词，$F$ 是内部范畴 $\mathcal{P}$ 的自函子，两者之间没有伴随函子连接。所以“是否存在从 $F$ 到 $V \circ F$ 的可计算桥”这一严格表述，等价于“模型能否仅凭自身参数推出自己与外部真理的偏差”——而这目前是开放的。

直觉上的答案是：不能。这里的元层断裂表现为：模型能在对象层生成推理链（$F$ 可计算），但它们的正确性（$V$ 的真值）由域外对象 $A^*$ 定义，而 $A^*$ 从不进入 $F$ 的构造。

<div class="center">

------------------------------------------------------------------------

</div>

## 九、总结：从朴素到严谨的路径

1. **起点**：朴素的不动点概念 $F(x) = x$
2. **具体化**：信念空间 $\mathcal{P}$ 与推断算子 $F$
3. **结构假设**：$F$ 是梯度下降的离散化（欧拉迭代）
4. **分工**：KL 散度管能量与曲率（Bregman 几何）；巴拿赫的度量用切空间上的欧氏范数
5. **压缩性证明**：切空间 Hessian 谱界 + 非扩张欧氏投影 $\Rightarrow$ $\|F(p_1) - F(p_2)\|_2 \leq \gamma\|p_1 - p_2\|_2$；KL 能量以 $\kappa\gamma^{2t}$ 指数衰减
6. **应用定理**：巴拿赫不动点定理保证存在、唯一、收敛
7. **物理意义**：不动点 $A$ 是训练数据的统计；收敛速度由问题的曲率决定

这套推导避免了“凑公式”；每一步都有清晰的数学理由。它展示了：

- 离散与连续的桥梁（欧拉迭代）
- 优化与动力系统的统一（梯度下降）
- 统计与几何的融合（KL 散度作为 Bregman 散度）

<div class="center">

------------------------------------------------------------------------

</div>

## 十、直接证据：能量地形与闭式解的数值验证

第 7 步的压缩性证明给出了永霖极限；本节为它提供直接证据：在信念单纯形上数值实现推断动力学（KL 能量的梯度流 + 投影到单纯形的欧拉步），并逐项核对闭式解的四条预测。整个实验可复现：完整源码是 `ch12_energy_terrain.py`（完整代码就在下面），图是 `ch12_energy_terrain.png`。

**实验设置。** 三个类别；信念单纯形 $\Delta^2$。能量函数 $E(p) = D_{\text{KL}}(p^* \| p)$，$p^*$ 是目标信念 / 吸引子；推断算子 $p_{t+1} = \operatorname{proj}_{\Delta}(p_t - \eta \nabla E(p_t))$；投影是到概率单纯形的欧氏投影（标准算法——两个实现细节见下）。所有闭式量都是第 7 步（压缩性证明）与第 10 步（永霖定理）定义的那一族：$\mu = 1/\max_i p_i$，$L = 1/\min_i p_i$，$\kappa = L/\mu$，$\eta_{\max} = 2\mu/L^2$。

![图 1：能量地形与闭式解的数值验证](/figures/ch12_energy_terrain.png)

**如何读图 1 的四个面板**：

**（a）地形存在。** $E$ 的等高线在单纯形上形成一个能量盆地，$p^*$ 在盆底；从三个不同初始信念出发、以步长 $0.8\eta_{\max}$ 迭代的推断轨迹都收敛到 $p^*$（星星）。推理是能量地形上的运动，不是黑箱——这是本章论点最直接的画面。

**（b）曲率谱：闭式 = 数值。** 沿一条推理轨迹，解析谱（$\lambda_{\min} = 1/\max_i p_i$，$\lambda_{\max} = 1/\min_i p_i$，在不动点处）与数值微分 Hessian 的特征值逐点吻合，$\kappa = L/\mu = 7.00 = 0.70/0.10$。第 7 步的“Hessian 是 $\operatorname{diag}(1/p_i)$”与有限差分一致——从这一面板直接可见，不需要“近似”。

**（c）$\eta_{\max}$ 的闭式景观。** 单纯形上 $\eta_{\max}(p) = 2\min_i p_i^2/\max_i p_i$ 的对数颜色图（对数色标从 $10^{-6}$ 到 $10^{-1}$）：中心可以大步走；越靠近边缘，步伐必须越小。“地形决定步速”不是修辞——就是这张图。

**（d）稳定边界：闭式精确预测数值。** 局部扰动动力学的数值失稳阈值 $\eta^*$（渐近判据，见下）与闭式 $2/\lambda_{\max}$（切空间谱，见下）：30 个随机目标分布上，$R^2 = 0.9978$，中位误差 2.16%——蓝点全部紧贴对角线。粉色三角是从均匀先验出发的全局扫描的数值阈值，始终位于保守界 $2/(L\kappa)$ 之上。

这个实验顺带做了两处改进：

**第一：精确的局部稳定界是切空间谱，不是全空间谱。** 投影把动力学限制在切空间 $\{v: \sum v_i = 0\}$ 上，所以精确的失稳条件是 $\eta < 2/\lambda_{\max}$，其中 $\lambda_{\max}$ 是 $\operatorname{diag}(1/q_i)$ 限制在切空间上的最大特征值。对 $K = 3$，$\lambda_{\max}$ 有解析式：它是

$$
3q_0q_1q_2\lambda^2 - 2(q_0q_1 + q_0q_2 + q_1q_2)\lambda + 1 = 0
$$

的最大根。第 10 步的 $\eta_{\max} = 2/(L\kappa)$ 仍然正确，但它是保守界（$\lambda_{\max} \leq L$，所以 $2/\lambda_{\max} \geq 2/L = \kappa \cdot \eta_{\max}$）。两族界在同一张图上可见：精确界（点落在线上）与保守界（点始终在其上方）。

**第二：稳定性判据必须用渐近终点，而不是全程最大值。** 动力学是否稳定，看的是 $\|p_T - p^*\| / \|p_0 - p^*\| < 1$（渐近终点距离比），而不是整个过程的最大值——在中等步长下，非线性瞬态先增长后收缩，全程最大值会把一个线性稳定的区域误判为不稳定（测得值是 0.79 的虚假比值；用渐近判据则是 1.000，与闭式一致）。

**源码**（完整可运行；运行时会把图保存为同目录下的 `ch12_energy_terrain.png`）：

```python
"""
Energy Landscape of Reasoning: Closed-Form Curvature and Safe Step Size
========================================================================

Demonstration accompanying the Yonglin Limit derivation.

Setup: belief simplex Delta^2 (3 classes). The reasoning energy is
    E(p) = D_KL(p* || p),  p* = target/attractor belief.
Inference = Euler steps of the gradient flow on E with projection to the
simplex:  p_{t+1} = proj_Delta(p_t - eta * grad E(p_t)).

Closed-form facts under study (true at the fixed point p = p*):
    Hess E(p)  = diag(q_i / p_i^2)   [general position; q_i = p*_i]
               -> diag(1/p_i)        [at p = p*]
    mu = lambda_min = 1/max_i p_i,  L = lambda_max = 1/min_i p_i
    kappa = L/mu = max_i p_i / min_i p_i          (spectral ratio / condition number)
    eta_max = 2 mu / L^2 = 2 / (L * kappa) = 2 min_i p_i^2 / max_i p_i

On the tangent space {v: sum v_i = 0} (the subspace preserved by the
projection) the exact local stability boundary is
    eta_crit = 2 / lambda_max,  lambda_max = largest eigenvalue of
    diag(1/q_i) restricted to the tangent space;
    for K = 3, lambda_max solves  3 q0 q1 q2 lam^2 - 2 (q0q1+q0q2+q1q2) lam + 1 = 0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke

rng = np.random.default_rng(42)


# ----------------------------- core helpers -----------------------------
def proj_simplex(v):
    """Euclidean projection onto probability simplex (vectorized over last axis).
    Standard algorithm: find the last index j with u_j - (css_j - 1)/j > 0."""
    v = np.asarray(v, float)
    shape = v.shape
    v = v.reshape(-1, shape[-1])
    K = shape[-1]
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)
    avg = (css - 1.0) / np.arange(1, K + 1)
    cond = u - avg > 0
    j = np.where(cond)[1]    # column indices of True entries (rows repeat)
    # last True index per row:
    last = np.array([np.where(cond[i])[0][-1] for i in range(len(cond))])
    theta = (css[np.arange(len(v)), last] - 1.0) / (last + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out.reshape(shape)


def energy(p, pstar):
    """E(p) = D_KL(p* || p), elementwise-safe (p clipped away from 0)."""
    pc = np.clip(p, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sum(pstar * np.log(pstar / pc), axis=-1)
    return np.where(np.all(p > 0, axis=-1), e, np.inf)


def grad_energy(p, pstar):
    return -pstar / np.clip(p, 1e-12, 1.0)


def euler_step(p, pstar, eta):
    return proj_simplex(p - eta * grad_energy(p, pstar))


def eta_max_closed(pstar):
    """Closed-form safe step size at the fixed point: 2 min^2 / max."""
    return 2.0 * np.min(pstar) ** 2 / np.max(pstar)


def hessian_numeric(p, pstar, eps=1e-6):
    """Central-difference Hessian of E at p (K=3)."""
    K = len(p)
    H = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ei = np.zeros(K); ej = np.zeros(K)
            ei[i] = eps; ej[j] = eps
            H[i, j] = (energy(p + ei + ej, pstar) - energy(p + ei - ej, pstar)
                       - energy(p - ei + ej, pstar) + energy(p - ei - ej, pstar)) / (4 * eps * eps)
    return H


def to_xy(p):
    """Barycentric -> Cartesian (A=e1 at (0,0), B=e2 at (1,0), C=e3 at (0.5,sqrt3/2))."""
    return p[..., 1] * 1.0 + p[..., 2] * 0.5, p[..., 2] * (np.sqrt(3) / 2)


def from_xy(x, y):
    s = np.sqrt(3.0) / 2.0
    p3 = np.clip(y / s, 0.0, 1.0)
    p2 = np.clip(x - y / (np.sqrt(3.0)), 0.0, 1.0)
    p1 = 1.0 - p2 - p3
    mask = (p1 >= -1e-9) & (p2 >= -1e-9) & (p3 >= -1e-9)
    p1 = np.clip(p1, 0.0, 1.0)
    return np.stack([p1, p2, p3], axis=-1), mask


# ----------------------------- figure panels -----------------------------
PSTAR = np.array([0.70, 0.20, 0.10])
EMAX = eta_max_closed(PSTAR)          # ~ 2*0.01/0.7 = 0.02857
ETA_ILLUS = 0.8 * EMAX

fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
ps = withStroke(linewidth=3, foreground="white")

# ---------- (a) energy terrain + trajectories ----------
ax = axes[0, 0]
xgrid = np.linspace(-0.05, 1.05, 300)
ygrid = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, 260)
X, Y = np.meshgrid(xgrid, ygrid)
P, mask = from_xy(X, Y)
E = energy(P, PSTAR)
E = np.where(mask, E, np.nan)
lv = np.geomspace(1e-5, 50, 40)
cf = ax.contourf(X, Y, E, levels=lv, cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=50))
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label(r"$E(p)=D_{KL}(p^{*}\|p)$  [log scale]", fontsize=9)

starts = [np.array([1/3, 1/3, 1/3]), np.array([0.15, 0.60, 0.25]), np.array([0.45, 0.25, 0.30])]
for p0 in starts:
    p = p0.copy()
    xs, ys = [], []
    for t in range(120):
        p = euler_step(p, PSTAR, ETA_ILLUS)
        xs.append(to_xy(p)[0]); ys.append(to_xy(p)[1])
        if energy(p, PSTAR) < 1e-6:
            break
    ax.plot(xs, ys, lw=1.8, color="#D55E00", alpha=0.9)
    ax.plot(xs[0], ys[0], "o", ms=6, color="#D55E00", mfc="white")
    ax.annotate("$p_0$", (xs[0], ys[0]), textcoords="offset points", xytext=(-2, 8),
                fontsize=9, color="#D55E00")

px, py = to_xy(PSTAR)
ax.plot(px, py, "*", ms=22, color="#0072B2", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$  (fixed point)", (px, py), textcoords="offset points",
            xytext=(8, -16), fontsize=11, color="#0072B2", fontweight="bold")
ax.text(0.98, 0.94, f"$\\eta=0.8\\,\\eta_{{max}}$ = {ETA_ILLUS:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
ax.set_title("(a) The energy terrain of reasoning", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "belief simplex $\\Delta^2$: $p_1$-vertex (left), $p_2$-vertex (right), $p_3$-vertex (top)",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (b) curvature spectrum along one trajectory ----------
ax = axes[0, 1]
p = np.array([1/3, 1/3, 1/3])
lmax_a, lmin_a, kap_a = [], [], []
lmax_n, lmin_n = [], []
p_rec = []
for t in range(80):
    p = euler_step(p, PSTAR, ETA_ILLUS)
    p_rec.append(p.copy())
for p in p_rec:
    lmax_a.append(np.max(PSTAR / p**2)); lmin_a.append(np.min(PSTAR / p**2))
    H = hessian_numeric(p, PSTAR)
    w = np.linalg.eigvalsh(H)
    lmax_n.append(w[-1]); lmin_n.append(w[0])
ts = np.arange(len(p_rec))
ax.plot(ts, lmax_a, color="#0072B2", lw=1.8, label=r"closed form  $\lambda_{max}$")
ax.plot(ts, lmin_a, color="#009E73", lw=1.8, label=r"closed form  $\lambda_{min}$")
ax.plot(ts, lmax_n, "o", ms=4.2, color="#0072B2", mfc="none", label=r"numerical Hessian  $\lambda_{max}$")
ax.plot(ts, lmin_n, "s", ms=4.2, color="#009E73", mfc="none", label=r"numerical Hessian  $\lambda_{min}$")
ax.set_yscale("log")
ax.set_xlabel("reasoning step  $t$", fontsize=10)
ax.set_ylabel("curvature eigenvalues  $\\lambda$", fontsize=10)
ax.set_title("(b) Curvature spectrum: closed form = numerical", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
ax.grid(alpha=0.3)
kap = PSTAR.max() / PSTAR.min()
ax.text(0.98, 0.08, r"$\kappa = L/\mu = %.2f = 0.70/0.10$ (at $p^{*}$)" % kap,
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# ---------- (c) analytic eta_max surface ----------
ax = axes[1, 0]
XS, YS = np.meshgrid(np.linspace(0, 1, 260), np.linspace(0, np.sqrt(3) / 2, 230))
PS, maskC = from_xy(XS, YS)
eta = np.where(maskC & np.all(PS > 1e-4, axis=-1),
               2.0 * np.min(PS, axis=-1) ** 2 / np.max(PS, axis=-1), np.nan)
eta = np.where(maskC, eta, np.nan)
lv2 = np.geomspace(1e-6, 0.25, 35)
cf2 = ax.contourf(XS, YS, eta, levels=lv2, cmap="magma", norm=LogNorm(vmin=1e-6, vmax=0.25))
cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.85)
cbar2.set_label(r"$\eta_{max}(p)=2\min_i p_i^2/\max_i p_i$  [log]", fontsize=9)
ax.plot(px, py, "*", ms=22, color="#56B4E9", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$", (px, py), textcoords="offset points", xytext=(6, 10),
            fontsize=11, color="#56B4E9", fontweight="bold")
ax.set_title("(c) Analytic safe step size on the simplex", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "$\\eta_{max} \\to 0$ near the edges: the terrain dictates the pace",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]


def lambda_max_tangent(q):
    """Largest eigenvalue of diag(q_i^{-1}) restricted to the tangent space {v: sum v_i = 0}.
    K = 3: analytic root of  3 q0 q1 q2 lam^2 - 2 (q0q1+q0q2+q1q2) lam + 1 = 0."""
    q0, q1, q2 = q
    a = 3 * q0 * q1 * q2
    b = -2 * (q0 * q1 + q0 * q2 + q1 * q2)
    c = 1.0
    disc = b * b - 4 * a * c
    lam = (-b + np.sqrt(disc)) / (2 * a)
    return lam


def eta_crit_tangent_closed(q):
    """Exact local stability boundary: eta_safe < 2 / lambda_max of Hessian on the tangent space."""
    return 2.0 / lambda_max_tangent(q)


# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

etas_scan = np.logspace(-4.0, 0.6, 140)

# --- LOCAL analysis: perturbations of p*; exact prediction = tangent-space spectrum ---
local_ana, local_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    pert = [proj_simplex(ps_ + 0.02 * rng.normal(0, 1, 3) * np.array([1, 0.5, 0.25])) for _ in range(8)]
    # asymptotic stability: final distance does not grow
    worst = np.zeros(len(etas_scan))
    for p0 in pert:
        P = np.broadcast_to(p0.copy(), (len(etas_scan), 3)).copy()
        d0 = np.linalg.norm(p0 - ps_)
        for _ in range(500):
            P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
        worst = np.maximum(worst, np.linalg.norm(P - ps_, axis=1) / d0)
    idx = np.where(worst < 1.0)[0]
    local_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    local_ana.append(eta_crit_tangent_closed(ps_))

# --- GLOBAL analysis: from uniform prior; conservative closed form = full-space spectrum ---
glob_ana, glob_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    P = np.broadcast_to(np.full(3, 1 / 3), (len(etas_scan), 3)).copy()
    for _ in range(400):
        P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
    fk = np.sum(ps_ * np.log(np.clip(ps_, 1e-12, 1) / np.clip(P, 1e-12, 1)), axis=1)
    idx = np.where(fk < 1e-3)[0]
    glob_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    glob_ana.append(eta_max_closed(ps_))

local_ana, local_num = np.array(local_ana), np.array(local_num)
glob_ana, glob_num = np.array(glob_ana), np.array(glob_num)

ax.loglog(local_ana, local_num, "o", ms=7, color="#0072B2", alpha=0.85, zorder=6,
          label=r"local: $2/\lambda_{\max}$ on tangent space (exact)")
ax.loglog(glob_ana, glob_num, "^", ms=7, color="#CC79A7", alpha=0.85, zorder=5,
          label=r"global: conservative bound $2/(L\kappa)$")
lims = [min(local_ana.min(), local_num.min()) * 0.6, max(local_ana.max(), local_num.max()) * 1.6]
ax.loglog(lims, lims, "k--", lw=1.5, label="y = x (exact prediction)")
logR = np.corrcoef(np.log10(local_ana), np.log10(local_num))[0, 1] ** 2
med = np.median(np.abs(local_num - local_ana) / local_ana)
ax.set_xlabel(r"closed-form  $\eta_{crit}$", fontsize=10)
ax.set_ylabel("numerical stability threshold  $\\eta_{crit}$", fontsize=10)
ax.set_title("(d) Closed form = the exact local stability boundary", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.05, f"exact (tangent spectrum): $R^2$ = {logR:.4f}, median err = {med:.2%}\nconservative (full spectrum): $\\eta_{{crit}} \\geq 2/(L\\kappa)$ always",
        transform=ax.transAxes, fontsize=9, color="k", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

fig.suptitle("The Energy Landscape of Reasoning: closed-form geometry and its verification\n"
             r"$E(p)=D_{KL}(p^{*}\|p)$,  inference $=$ Euler steps of $-\nabla E$ projected to the simplex",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "ch12_energy_terrain.png"   # saves next to the script
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
print(f"eta_max closed form (p* = {PSTAR}): {EMAX:.6f}")
print(f"kappa at fixed point: {PSTAR.max()/PSTAR.min():.2f}")
print(f"panel d: local R^2(log-log) = {logR:.4f}, n = {len(local_ana)}, median err = {med:.2%}")
print(f"panel d: global: numerically stable region >= eta_max (projection adds stability)")
```

*图 1 与源码：李籽溪（兔狲教授），2026。实验脚本 `ch12_energy_terrain.py`（完整代码见本节代码块）。*

### 10.1 真实 LLM 上的直接证据：信念锚点、证据强度与记忆衰减

上一节的数值实验验证了公式的内部自洽性：动力学本来就是按闭式解的欧拉步构造的，当然会跟着它走。本节给出真实 LLM 上的直接证据：用真实模型（DeepSeek-V4-flash，reasoning 类型）实例化第七节的推断算子 $F: \mathcal{P} \to \mathcal{P}$，逐步注入证据、逐步读回信念，逐项核对永霖公式与永霖极限在真实推理中的对应物。图是 `ch12_llm_evidence.png`；实验的完整数据设计如下（实验本身与数字都活在图中——涉及真实模型 API 的脚本不在此重复）。

**实验设置。** 三色球袋世界，仍在信念单纯形 $\Delta^2$ 上。球袋的球比例 $p^*$ 取三种设定，包括全书基线 $(0.70, 0.20, 0.10)$（$\kappa = 7.00$）；一个已知可靠性的检测器 $d \in \{0.90, 0.70, 0.55\}$ 每步对摸出的球给出一次带噪颜色报告（似然 $P(r|c)$ 完全已知，真实后验 $p^B_t$ 逐步闭式计算）；证据序列就是检测器报告序列，$T = 6$。每一步，模型的输入是“球袋参数 + 前 $t$ 条报告 + 强制一词作答”，信念从最终答案 token 的归一化 top-k logprobs 读出（$K = 3$）。网格：$3 \times 3 \times 4$ 个种子——每个单元一条普通轨迹加两条提示轨迹；锚点测量：3 种选项排列 $\times$ 8 种改写。

![图 12.2：真实 LLM 上的信念锚点与证据动力学验证](/figures/ch12_llm_evidence.png)

**如何读图 12.2 的三个面板**——（a）锚点：三种选项排列下的零信息信念都贴近某个顶点（位置偏置），全局平均 $\kappa_A = 1.285$；（b）证据吸收：每条曲线两条线——实线是 LLM 与真实后验 $p^B_t$ 的 KL，虚线是与“$d' = 0.9$ 假后验”的 KL——虚线几乎处处更低；（c）相位差记忆衰减：中位距离 $d_t$ 在第三步归零。

**结果 1：零信息锚点 $A$ 不是均匀点——它是可测量的先验。** 三种排列下的信念：红/蓝/黑 $(0.999, 0.001, 0)$；黑/红/蓝 $(0.125, 0.0001, 0.875)$；蓝/黑/红 $(0.0001, 1.000, 0)$。位置偏置在 87.5%–100% 之间（唯一的“叛徒”：黑/红/蓝排列下，红色在 8 次中被选了 1 次）。全局 $A = (0.375, 0.334, 0.292)$，$\kappa_A = 1.285$。$A$ 在条件数意义上只离均匀 1.285，但它的成分并不均匀：它是“选项位置规律 × 轻度词偏好”的混合。第 9 步“识别先验锚点”：经验锚点就在这里——它与永霖公式 $A \neq A^*$ 中的 $A$ 是同一个对象，现在有了数字。

**结果 2：输出层信念是尖峰的——推理贴着单纯形边界走。** 在第 0 步（提示词已写明球袋比例 7/2/1，还没有报告），中位 $\text{KL}(p^B_0 \| p^{\text{LLM}}_0)$ 约为 6.8–7.4 nats：即使提示词写明了不对称比例，模型的信念仍然贴近 one-hot 顶点。模型对概率世界的信仰是决断式的——它不在单纯形内部（连续的不确定性光谱）游荡，而是直接跳到某个近顶点。对永霖公式而言：真实推理系统的吸引子 $\approx$ 一个顶点；$A$ 不仅偏离 $A^*$，而且落在顶点国度。

**结果 3：证据强度吸收是不对称的——弱证据被放大，强证据被忠实跟随。** 下表（中位 KL：LLM 信念与真实后验的 KL / 与 $d' = 0.9$ 假后验的 KL）：

| d | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---|---|---|---|---|
| 0.55（真实后验 / $d' = 0.9$ 后验） | 5.82 / 0.96 | 3.30 / 1.34 | 3.74 / 0.40 | 4.06 / 0.40 | — |
| 0.70 | 3.32 / 0.92 | 0.79 / 0.05 | 1.72 / 0.38 | 0.40 / 0.02 | 0.28 / 0.02 |
| 0.90 | 0.60 | 0.03 | 0.31 | 0.02 | — |

对 $d = 0.90$ 单调下降（最终中位 0.02——几乎精确贝叶斯）；对 $d = 0.70$，从第 2 步起更贴近 $d' = 0.9$ 假后验；对 $d = 0.55$，与 $d' = 0.9$ 假后验的距离比与真实后验近约 8 倍。模型把提示词里的证据可靠性当成一个近似固定的 $d' \approx 0.9$：弱证据被系统性放大，强证据被精确跟随。这与人类的保守主义（Edwards 1968）正好相反——LLM 的证据强度天花板约为 0.9：弱证据永远意味着“别信提示词，按最大强度用”。

**结果 4：压缩与记忆——相位差半衰期约 2 步。** 同一个世界，同一串报告尾部（第 2–6 条报告相同）；两条轨迹唯一的区别是第一条报告是红还是蓝。初始分叉距离 $d_1 \equiv 1.414 = \sqrt{2}$（两个顶点之间；36 个单元无一例外）；从 $d_2$ 起中位距离为 0；单步压缩中位 0.07–0.20；半衰期 2.0–2.5 步。用压缩因子 $\gamma$ 写出的有效推理窗口 $t_{\text{eff}}(\varepsilon)=\ln(1/\varepsilon)/\ln(1/\gamma)$（$\varepsilon = 0.1$）预测约 2.3 步——与测量一致。注：（a）偶发再分叉：12 个单元中有 1 个在 $t = 6$ 时仍保持 >0.1 的距离，全程最大值可达 $\sqrt{2}$——这是上一节面板（d）“瞬态先增长后收缩”的镜像（这里是先收缩后重新增长）：即使确定性模型也包含混沌式再分裂；（b）在 temperature 0 下，$d_t = 0$ 的部分“巧合”是确定性伪影；统计上，只有这一读法成立：“一个单位的相位差，两步之后不再影响下一步的选择”。

对理论四行交叉核对：锚点 = 第 9 步先验锚点 $A$ 的测量（$A \neq$ 均匀，$\kappa_A = 1.285$）；尖峰 = 顶点附近的能量盆地（推理轨迹在顶点国度里运动）；弱证据过反应 = 不对称证据吸收（$d' \approx 0.9$ 天花板）；相位差半衰期 $\approx 2$ 步 = 压缩因子定义的有效推理窗口的实测常数，与闭式压缩界同量级。一个诚实的保留：$\eta_{\max}$、$\lambda_{\max}$ 之类闭式量描述的是连续动力学的安全步长；真实 LLM 的“步长”是证据强度的函数（弱证据也带一个有效 $d' \approx 0.9$），它的运动更像“大步、短窗、在顶点间跳跃”。

*图 12.2 与源码：李籽溪（兔狲教授），2026。实验见 10.1 节（脚本的 API 版本不在书中展开）。*

### 10.2 同一极限，在一台笔记本上：一幅你自己能画的蛛网图

10.1 节测量了大型闭参数模型的信念动力学。那个实验的代价是：读者拿到的是数字，不是实验。本小节把同一个极限做成你能在自己笔记本上跑的形式。模型是 GPT-2，约 1.24 亿参数，开源且小；信念是一维的；得到的图是离散动力系统（第七节，迭代序列 $x_{t+1} = F(x_t)$）的经典蛛网图（cobweb plot）——只不过是从真实模型上画出来的，而不是从公式里画出来的。

**实验设置。** 24 个传递链问题（例如“A 比 B 高。B 比 C 高。A 比 C 高吗？”）。它们全部为真，所以真实答案分布恰好是 $A^* = 1.0$。信念是 $p = P(\text{YES})$：模型给出一个答案词，我们读两个答案 logit，取它们之差的 sigmoid（$P = \sigma(\text{logit}_{YES} - \text{logit}_{NO})$）。模型在 temperature 0 下运行，所以实验是确定性的：每次都是同样的数字。动力学的一步：把当前信念作为一行文本写进提示词（告诉它到目前为止 YES 的概率是 0.437），再让模型回答一次，读回下一个信念。映射 $F$ 在 31 个网格点 $x = 0.02 \ldots 0.98$ 上测量，并对 24 个问题取平均；然后从每个问题自己的自然起始信念（完全没有置信度行）出发，跑 128 步递推。

**结果 1：锚点是测出来的，不是假设的。** 24 个问题上的初始信念均值为 0.4045；逐个问题的值在 0.1993 到 0.7105 之间。一段文本的先验锚点是文本的函数：road/path/lane 链条上模型从 0.199 起步，X/Y/Z 链条上从 0.710 起步。一个问题背后没有单一的“训练数据边缘分布”：先验锚点是逐问题锚点的混合物。

**结果 2：测得的映射是平的——一步，不是阶梯。** 整个网格上，$F(x)$ 保持在 0.47 到 0.52 之间（$\sigma \approx 0.067$）。写下一个 0.03 或 0.97 的信念：模型不管怎样都回给你约 0.49。蛛网轨迹不是缓慢的阶梯，而是悬崖。七个起点（0.03、0.10、0.25、0.40、0.70、0.85、0.97）都在一步之内被吸进锚点带。测得映射的不动点是 $A = 0.4849$；与 $A^* = 1.0$ 的差距是 0.5151。映射是平的，不动点是先验：没有任何回读步骤能把信念带得更接近真理。

**结果 3：有效推理窗口是一步。** 在 128 步 rollout 中，中位 $P(\text{YES})$ 从 0.3944（$t = 0$）走到 0.5002（$t = 1$），然后再也不动：从 $t = 16$ 到 $t = 128$，中位数一直留在 $[0.490, 0.505]$ 内，逐问题的最小 / 最大值完全不动（0.380 / 0.6885，恒定）。每个问题冻结在自己的局部锚点上（0.6885 的最大值就是 X/Y/Z 链条的锚点）。测得的有用窗口是一步。“想久一点”在这个模型上不只是慢——而是已经结束了。

![图 12.3：真实 GPT-2 上一步吸收的蛛网图：测得的映射、七条迭代信念轨迹、锚点不动点 A = 0.485 与不可达的真实答案 A* = 1.0](/figures/yonglin_cobweb_measured.png)

![图 12.4：真实 GPT-2 上 128 步信念再循环：24 个问题的中位数与分布，一步之后冻结在锚点](/figures/yonglin_belief_evolution.png)

**如何读图 12.3（蛛网图）。** 灰色虚线对角线是“信念不变”：线上的任何点都是候选不动点。蓝线是测得的映射 $F$：一次重读前的信念（$x$）对一次重读后的信念（$F(x)$）。它在 0.49 附近近乎水平——这份平坦就是全部要点。彩色阶梯是七个不同起点出发的迭代信念：每一条先竖直跳到映射上，再水平落到对角线上，之后的一切都发生在放大框里。绿色星标是不动点 $A = 0.4849$；紫色叉标是 $A^* = 1.0$，真实答案。放大框显示锯齿在 $A$ 处熄灭：第一步之后，信念只是在原地振动。（图的标题可以用 $A = 0.485$。）

**如何读图 12.4（时间序列）。** 每条线是一个问题在 128 步上的信念；阴影带是 24 个问题的四分位带与 10–90 百分位带。看 $t = 1$ 处的悬崖：中位数一步从 0.394 跳到 0.500，而没有任何一个问题真正接近过 $A^* = 1.0$。点线是测得的锚点 $A \approx 0.485$；1.0 处的紫线是始终不可达的真理。重要的不是线是平的——而是它们冻结在哪里。

一个诚实的保留。这里的信念是一个 1.24 亿参数状态的一维投影，而把数字写回提示词是“把信念写下来”的一种风格化做法。这个实验展示的不是 GPT-2 的心理，而是这个投影的动力学：平坦的映射、位于先验的锚点、一步的窗口。它毫无建模地表明：在真实模型上，永霖极限不是缓慢的淡出——是悬崖。

**动手玩。** 改问题库，改注入置信度的模板，运行 `--model gpt2-medium`，改 `--grid` 和 `--steps`。锚点会动（它总会动）；映射保持平坦；$A \neq A^*$ 是你会反复重新发现的模式。

**源码**（完整可运行；保存为 `yonglin_gpt2.py` 后运行。它会在自身旁边保存两张 PNG：`yonglin_cobweb_measured.png` = 图 12.3，`yonglin_belief_evolution.png` = 图 12.4）：

```python
#!/usr/bin/env python3
"""
Yonglin Cobweb: Belief-Recirculation Dynamics on a Real GPT-2
==============================================================
Companion experiment for The Reasoning Kingdom, Chapter 12
(Sections 5, 7, 10.1 and 10.2).

THE IDEA IN ONE PARAGRAPH
-------------------------
Yonglin's formula says: belief dynamics p_{t+1} = F(p_t) converges to the
prior anchor A, not to the true answer A* (A != A*). Section 10.1 verifies anchors and evidence
absorption on a large closed API model, but the reader cannot run it.
Section 10.2 gives the version you can RUN and PLAY WITH: a small open
model (GPT-2, ~124M params, runs on a laptop), a one-dimensional belief
state P(YES), and the belief recirculated through the model itself.
Each step you read the model's belief from its next-token logits, write
that belief back into the prompt as text, and measure the next belief.
That defines a one-dimensional dynamical system p_{t+1} = F(p_t) that you
can measure, plot, and look at as a COBWEB picture: the curve y = F(x),
the diagonal y = x, and the zig-zag staircase of the iterated belief.

WHAT THE SCRIPT DOES
--------------------
  1. loads the model (default: gpt2; check the HuggingFace cache first).
  2. read_belief(prompt): P(YES) = sigmoid(logit_YES - logit_NO).
  3. measure_F(): 31 grid points x in [0, 1], the belief written back
     into the prompt; F measured over 24 questions.
  4. rollouts(): 128 steps of belief re-circulation from each question's
     natural start; prints summaries and the distribution over questions.
  5. cobweb_points(): the classic zig-zag construction.
  6. two figures:
       (a) yonglin_cobweb_measured.png       -- the cobweb itself
       (b) yonglin_belief_evolution.png      -- p_t vs t, all 128 steps

HONESTY CAVEATS
---------------
  * "Belief" is a one-dimensional projection (next-token softmax of
    YES/NO). The real state is 124M-dimensional; this is the deliberate
    cut that makes the cobweb picture possible.
  * Writing the number p back into the prompt is a stylized way of
    writing the belief down. The model re-reads a text that contains
    its own last belief; it does not literally iterate a hidden state.
  * Every number printed and every point plotted comes from a live run
    of the model on this machine. Nothing is scripted.

Run:  python3 yonglin_gpt2.py
      python3 yonglin_gpt2.py --model gpt2-medium --steps 128 --grid 31
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------------------------
# 0. The question bank (24 short transitive-chain questions, all TRUE)
# ---------------------------------------------------------------------------
QUESTIONS = [
    "A is taller than B. B is taller than C. Is A taller than C?",
    "X is larger than Y. Y is larger than Z. Is X larger than Z?",
    "P is heavier than Q. Q is heavier than R. Is P heavier than R?",
    "The oak is older than the birch. The birch is older than the elm. Is the oak older than the elm?",
    "K is faster than L. L is faster than M. Is K faster than M?",
    "The river is longer than the stream. The stream is longer than the brook. Is the river longer than the brook?",
    "Red is darker than pink. Pink is darker than cream. Is red darker than cream?",
    "The mountain is steeper than the hill. The hill is steeper than the slope. Is the mountain steeper than the slope?",
    "Steel is harder than iron. Iron is harder than tin. Is steel harder than tin?",
    "The tiger is stronger than the wolf. The wolf is stronger than the fox. Is the tiger stronger than the fox?",
    "The poem is longer than the song. The song is longer than the haiku. Is the poem longer than the haiku?",
    "Bob is faster than Anne. Anne is faster than Carl. Is Bob faster than Carl?",
    "Winter is colder than autumn. Autumn is colder than spring. Is winter colder than spring?",
    "Silver is shinier than copper. Copper is shinier than iron. Is silver shinier than iron?",
    "The soup is hotter than the stew. The stew is hotter than the porridge. Is the soup hotter than the porridge?",
    "The whale is larger than the elephant. The elephant is larger than the horse. Is the whale larger than the horse?",
    "The road is longer than the path. The path is longer than the lane. Is the road longer than the lane?",
    "The clock is older than the watch. The watch is older than the phone. Is the clock older than the phone?",
    "Sarah is taller than Kate. Kate is taller than Anna. Is Sarah taller than Anna?",
    "The ocean is deeper than the lake. The lake is deeper than the pond. Is the ocean deeper than the pond?",
    "The horse is faster than the dog. The dog is faster than the cat. Is the horse faster than the cat?",
    "The house is taller than the garage. The garage is taller than the shed. Is the house taller than the shed?",
    "The sun is hotter than the flame. The flame is hotter than the candle. Is the sun hotter than the candle?",
    "Matthew is slower than Helen. Helen is slower than George. Is Matthew slower than George?",
]
ANSWER = "YES"                      # every question above is TRUE (A* = 1.0)
ID_YES, ID_NO = None, None          # filled in main()
TEMPLATE_ANSWER = "\nAnswer with exactly one word: YES or NO."


def confidence_template(p: float) -> str:
    """Write the current belief back into the prompt as one line."""
    return f"\nConfidence so far: P(YES) = {p:.3f}."


# ---------------------------------------------------------------------------
# 1. Belief reader
# ---------------------------------------------------------------------------
@torch.no_grad()
def read_belief(model, tokenizer, prompt: str, device) -> float:
    """P(YES) = sigmoid(logit_YES - logit_NO) at the last position."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = model(input_ids).logits[0, -1]
    return float(torch.sigmoid(logits[ID_YES] - logits[ID_NO]).item())


def ask(model, tokenizer, question: str, p: float, device) -> float:
    """One application of the map: belief p in, belief F(p) out."""
    prompt = question + confidence_template(p) + TEMPLATE_ANSWER
    return read_belief(model, tokenizer, prompt, device)


# ---------------------------------------------------------------------------
# 2. Measuring F on a grid
# ---------------------------------------------------------------------------
def measure_F(model, tokenizer, questions, grid_n: int, device):
    """F(x) on a grid, averaged over questions: (xs, mean, std)."""
    xs = np.linspace(0.02, 0.98, grid_n)
    Fs = np.zeros((len(questions), grid_n))
    for qi, q in enumerate(questions):
        for xi, x in enumerate(xs):
            Fs[qi, xi] = ask(model, tokenizer, q, float(x), device)
    return xs, Fs.mean(axis=0), Fs.std(axis=0)


# ---------------------------------------------------------------------------
# 3. Rollouts
# ---------------------------------------------------------------------------
def rollouts(model, tokenizer, questions, steps: int, device):
    """p_0 = natural belief; p_{t+1} = F(p_t). Returns array (Q, steps+1)."""
    traj = np.zeros((len(questions), steps + 1))
    for qi, q in enumerate(questions):
        p0 = read_belief(model, tokenizer, q + TEMPLATE_ANSWER, device)
        traj[qi, 0] = p0
        for t in range(steps):
            traj[qi, t + 1] = ask(model, tokenizer, q, float(traj[qi, t]), device)
    return traj


# ---------------------------------------------------------------------------
# 5. Cobweb construction
# ---------------------------------------------------------------------------
def cobweb_points(fn, p0, steps: int):
    """The zig-zag: vertical to the curve, horizontal to the diagonal."""
    pts_x, pts_y = [p0], [p0]
    p = p0
    for _ in range(steps):
        f = float(np.asarray(fn(np.array([p])))[0])
        pts_x.append(p)
        pts_y.append(f)
        p = f
        pts_x.append(p)
        pts_y.append(p)
    return np.array(pts_x), np.array(pts_y)


def interpolate_F(xs, Fm):
    def fn(x):
        return np.interp(np.asarray(x, dtype=float), xs, Fm)
    return fn


def fixed_point(xs, Fm):
    """Root of F(x) - x by linear interpolation on the grid."""
    g = Fm - xs
    idx = np.where(np.diff(np.sign(g)) != 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(xs[i] - g[i] * (xs[i + 1] - xs[i]) / (g[i + 1] - g[i]))


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def plot_measured_cobweb(xs, Fm, Fs, p0_mean, fixed, out_dir):
    """(a) The cobweb itself: multi-start, the flat anchor map."""
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    ax.plot(xs, xs, color="0.55", lw=1.6, ls="--", label="y = x  (belief unchanged)")
    ax.fill_between(xs, Fm - Fs, Fm + Fs, color="#9ecae1", alpha=0.45,
                    label="measured F(x) ± std over 24 questions")
    ax.plot(xs, Fm, color="#1f77b4", lw=3.2, label="F(x): GPT-2 re-reads its belief")
    starts = [0.03, 0.10, 0.25, float(p0_mean), 0.70, 0.85, 0.97]
    fn = interpolate_F(xs, Fm)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(starts)))
    for c, p0 in zip(colors, starts):
        px, py = cobweb_points(fn, float(p0), 20)
        ax.plot(px, py, color=c, lw=1.7, alpha=0.9)
        ax.plot([p0], [p0], "o", ms=8, color=c, zorder=6, label=f"start p0 = {p0:.3f}")
    ax.plot(fixed, fixed, "*", color="#2ca02c", ms=20, zorder=7,
            label=f"fixed point A = {fixed:.3f}")
    ax.plot([1.0], [1.0], "P", color="#9467bd", ms=15, zorder=7,
            label="A* = 1.0  (true answer)")
    ax.set_xlim(-0.02, 1.07); ax.set_ylim(-0.02, 1.07)
    ax.set_xlabel("belief before re-reading (x)")
    ax.set_ylabel("belief after re-reading (F(x))")
    ax.set_title("Yonglin cobweb on a real GPT-2: one-step absorption into the anchor\n"
                 f"every start falls into the flat map F(x) ~ {np.mean(Fm):.2f} "
                 f"-> A = {fixed:.3f}, never A* = 1.0",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax.grid(alpha=0.25)
    axin = ax.inset_axes([0.50, 0.04, 0.44, 0.44])
    axin.plot(xs, xs, color="0.55", lw=1.2, ls="--")
    axin.plot(xs, Fm, color="#1f77b4", lw=2.0)
    axin.fill_between(xs, Fm - Fs, Fm + Fs, color="#9ecae1", alpha=0.4)
    for c, p0 in zip(colors, starts):
        px, py = cobweb_points(fn, float(p0), 20)
        axin.plot(px, py, color=c, lw=1.0, alpha=0.9)
    axin.plot([fixed], [fixed], "*", color="#2ca02c", ms=11)
    axin.set_xlim(0.47, 0.66); axin.set_ylim(0.47, 0.66)
    axin.set_xticks([0.50, 0.55, 0.60, 0.65]); axin.set_yticks([0.50, 0.55, 0.60, 0.65])
    axin.tick_params(labelsize=7)
    axin.set_title("zoom: the zig-zag dies at A", fontsize=8.5)
    axin.grid(alpha=0.3)
    # the frame that tells the reader WHICH region of the main plot is zoomed:
    ax.indicate_inset_zoom(axin, edgecolor="0.30", lw=1.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "yonglin_cobweb_measured.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_evolution(traj, fixed, out_dir):
    """(b) p_t vs t over 128 steps: median, quartile band, samples."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    t = np.arange(traj.shape[1])
    med = np.median(traj, axis=0)
    q1, q3 = np.percentile(traj, [25, 75], axis=0)
    lo, hi = np.percentile(traj, [10, 90], axis=0)
    ax.fill_between(t, q1, q3, color="#9ecae1", alpha=0.55,
                    label="interquartile band (24 questions)")
    ax.fill_between(t, lo, hi, color="#c6dbef", alpha=0.35,
                    label="10-90% band")
    ax.plot(t, med, color="#1f77b4", lw=2.8, label="median P(YES)")
    for qi in range(min(4, traj.shape[0])):
        ax.plot(t, traj[qi], color=plt.cm.plasma(0.15 + 0.2 * qi),
                lw=1.0, alpha=0.65, label=f"question {qi + 1}")
    ax.axhline(fixed, color="#2ca02c", ls=":", lw=2.0,
               label=f"anchor A ≈ {fixed:.3f} (fixed point)")
    ax.axhline(1.0, color="#9467bd", ls="-", lw=2.0, label="A* = 1.0 (true answer)")
    ax.set_xlim(0, traj.shape[1] - 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("re-circulation step t")
    ax.set_ylabel("belief P(YES)")
    ax.set_title("128 steps of belief re-circulation: the window of usefulness",
                 fontsize=11.5)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "yonglin_belief_evolution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Yonglin cobweb experiment on GPT-2")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--replot", action="store_true",
                    help="skip measurement, redraw figures from the cached data")
    args = ap.parse_args()
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(out_dir, "yonglin_cache.npz")
    if args.replot and os.path.exists(cache_path):
        data = np.load(cache_path)
        xs, Fm, Fs, traj, p0s = (data["xs"], data["Fm"], data["Fs"],
                                 data["traj"], data["p0s"])
        p0_mean = float(np.mean(p0s))
        print(f"[replot] loaded cached data from {cache_path}")
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[setup] model={args.model} device={device}")
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
        global ID_YES, ID_NO
        ID_YES = tokenizer.encode(" YES")[0]
        ID_NO = tokenizer.encode(" NO")[0]
        print(f"[setup] YES token={ID_YES}  NO token={ID_NO}")
        print(f"\n[1] Initial belief per question (no confidence injected):")
        p0s = []
        for q in QUESTIONS:
            p0 = read_belief(model, tokenizer, q + TEMPLATE_ANSWER, device)
            p0s.append(p0)
            print(f"    P(YES) = {p0:.4f}   | {q[:62]}")
        p0_mean = float(np.mean(p0s))
        print(f"    mean p0 = {p0_mean:.4f}   (the measured prior anchor A)")
        print(f"\n[2] Measuring F on {args.grid} grid points x {len(QUESTIONS)} questions ...")
        xs, Fm, Fs = measure_F(model, tokenizer, QUESTIONS, args.grid, device)
        for x, fm, fs in zip(xs, Fm, Fs):
            print(f"    x={x:.3f}   F_mean={fm:.4f}   F_std={fs:.4f}")
        print(f"\n[3] Rollouts ({args.steps} steps of belief re-circulation):")
        traj = rollouts(model, tokenizer, QUESTIONS, args.steps, device)
        for t in range(0, args.steps + 1, 16):
            row = traj[:, t]
            print(f"    t={t:3d}  median={np.median(row):.4f}  "
                  f"q1={np.percentile(row, 25):.4f} q3={np.percentile(row, 75):.4f}  "
                  f"min={row.min():.4f}  max={row.max():.4f}")
        np.savez_compressed(cache_path, xs=xs, Fm=Fm, Fs=Fs, traj=traj,
                            p0s=np.asarray(p0s, dtype=float))
        print(f"\n[cache] saved measured data to {cache_path}")
    fixed = fixed_point(xs, Fm)
    print(f"\n[4] Fixed point of the measured map: A = {fixed:.4f}")
    print(f"    True answer A* = 1.0.  Gap = {abs(fixed - 1.0):.4f}")
    print(f"\n[5] Final belief after {args.steps} steps: "
          f"median={np.median(traj[:, -1]):.4f}")
    print(f"    One-step view: p0 mean={p0_mean:.4f}, p1 median={np.median(traj[:, 1]):.4f}")
    p1 = plot_measured_cobweb(xs, Fm, Fs, p0_mean, fixed, out_dir)
    p2 = plot_evolution(traj, fixed, out_dir)
    print(f"\n[6] Figures written:")
    for p in (p1, p2):
        print(f"    {p}")
    print("\nDone. The zig-zag in the first figure is Yonglin's formula,\n"
          "drawn from a real model instead of a formula.")


if __name__ == "__main__":
    main()
```

运行脚本，你会看到（真实运行的节选——下面的数字正是由这份脚本、在 macOS MPS 设备上的 gpt2 产生的；尾部是这次运行的代表性输出）：

```text
[setup] model=gpt2 device=mps
[setup] YES token=21560  NO token=8005
[1] Initial belief per question (no confidence injected):
    P(YES) = 0.5230   | A is taller than B. B is taller than C. Is A taller than C?
    P(YES) = 0.7105   | X is larger than Y. Y is larger than Z. Is X larger than Z?
    P(YES) = 0.4635   | P is heavier than Q. Q is heavier than R. Is P heavier than R?
    P(YES) = 0.1993   | The road is longer than the path. The path is longer than the lane. Is the road longer than the lane?
    P(YES) = 0.3056   | The tiger is stronger than the wolf. The wolf is stronger than the fox. Is the tiger stronger than the fox?
    P(YES) = 0.4031   | Winter is colder than autumn. Autumn is colder than spring. Is winter colder than spring?
    ...
    mean p0 = 0.4045   (the measured prior anchor A)
[2] Measuring F on 31 grid points x 24 questions ...
    x=0.020   F_mean=0.4834   F_std=0.0668
    x=0.212   F_mean=0.5067   F_std=0.0647
    x=0.500   F_mean=0.4700   F_std=0.0693
    x=0.788   F_mean=0.5002   F_std=0.0666
    x=0.980   F_mean=0.4867   F_std=0.0681
[3] Rollouts (128 steps of belief re-circulation):
    t=  0  median=0.3944  q1=0.3338  q3=0.4611  min=0.1993  max=0.7105
    t= 16  median=0.4930  q1=0.4615  q3=0.5241  min=0.3800  max=0.6885
    t= 32  median=0.5009  q1=0.4601  q3=0.5316  min=0.3800  max=0.6885
    t= 64  median=0.4937  q1=0.4681  q3=0.5241  min=0.3800  max=0.6885
    t= 96  median=0.4932  q1=0.4632  q3=0.5361  min=0.3769  max=0.6885
    t=128  median=0.4955  q1=0.4596  q3=0.5296  min=0.3800  max=0.6885
[4] Fixed point of the measured map: A = 0.4849
    True answer A* = 1.0.  Gap = 0.5151
[5] Final belief after 128 steps: median=0.4955
    One-step view: p0 mean=0.4045, p1 median=0.5002
```

*图 12.3–12.4 与源码：李籽溪（兔狲教授），2026。实验在默认的笔记本构建上运行：GPT-2、transformers、numpy、matplotlib；不动点与所有百分比都直接来自模型运行，脚本是确定性的。*

<div class="center">

------------------------------------------------------------------------

</div>

## 十一、物理直觉：碗、坡度和步长

形式推导的四个关键参数 $(\mu, L, \eta, \gamma)$，有直白的物理含义：

### 1. 凸性（碗的形状）

- 碗（凸函数）：只有一个最低点；梯度下降保证能找到它
- 薯片（非凸函数）：好几个凹陷；可能卡在局部极小
- 训练良好的模型，能量函数像碗；数据的多样性决定碗的深度

### 2. 强凸系数 $\mu$（碗有多陡）

- 浅碗（$\mu$ 小）：坡缓，收敛慢
- 深碗（$\mu$ 大）：坡陡，收敛快
- 物理含义：训练数据多样性 -> $\mu$ 大 -> 模型容易学到清晰的模式

### 3. 光滑常数 $L$（斜率变化的最大速率）

- 平缓地形（$L$ 小）：斜率变化不大；可以大步走
- 崎岖地形（$L$ 大）：斜率剧烈摆动；必须小步走
- 物理含义：架构光滑 -> $L$ 小 -> 推理稳定

### 4. 步长 $\eta$（一次推断的跨度）

- 太大：可能直接跨过正确路径
- 太小：磨蹭，收敛慢
- 刚好：$\eta < \frac{2\mu}{L^2}$（安全下坡公式）

### 5. 压缩因子 $\gamma = \sqrt{1 - 2\eta\mu + \eta^2L^2}$

- $\mu$ 大（陡碗）-> $\gamma$ 小 -> 收敛快
- $L$ 大（崎岖）-> $\gamma$ 大 -> 收敛慢
- $\eta$ 刚好 -> $\gamma$ 最小 -> 最优收敛

### 类比：学骑自行车

- 凸性：一个明确的目的地（家）
- $\mu$：家的牵引力（距离和重力）
- $L$：路面有多平
- $\eta$：每一脚蹬得多用力
- $\gamma$：你多快到达

### 这在 AI 推理里是什么样

1. 数据质量 -> $\mu$：多样的数据造出“深碗”
2. 模型架构 -> $L$：光滑的设计造出“平缓地形”
3. 推理设计 -> $\eta$：合适的步长平衡探索与利用
4. 三者合起来 -> $\gamma$：它决定有效推理窗口的长度

好系统：$\mu$ 大 + $L$ 小 -> 允许大 $\eta$ -> $\gamma$ 小 -> 快速收敛
坏系统：$\mu$ 小 + $L$ 大 -> 必须小 $\eta$ -> $\gamma$ 大 -> 缓慢收敛

这就是为什么不同的模型、不同的任务有不同的“有效推理窗口”——它们的 $(\mu, L, \eta)$ 组合不同，收敛速度就不同。

永霖极限不是神秘现象；它是信念空间上凸优化的必然结果。

<div class="center">

------------------------------------------------------------------------

</div>

## 十二、后言：兔狲下山记

> 推理就像爬山，真理在山顶。但山有重力，训练是在修路。

### 一个小故事：兔狲教授的推理远征

先爬一次“真理山”。永霖公式的几个核心物件，都在路上。

**第一幕：选择地形**

山脚下三条路。

1. 深碗路（$\mu$ 大）：陡，但直通山顶
2. 浅盘路（$\mu$ 小）：缓，但要绕远
3. 薯片路（非凸）：好几个凹陷；可能迷路

走深碗路。数据够杂，碗才挖得够深。

**第二幕：检查路面**

深碗路也分两种路面。

- 光滑路（$L$ 小）：坡度变化均匀
- 崎岖路（$L$ 大）：一会儿陡一会儿缓，容易绊倒

我们的模型架构光滑，$L$ 小。放心走。

**第三幕：决定步幅**

步幅有安全线：$\eta < \frac{2\mu}{L^2}$。

碗陡，路平，步子可以放大。

**第四幕：压缩因子 $\gamma$**

每迈一步，离山顶的距离乘上 $\gamma = \sqrt{1 - 2\eta\mu + \eta^2L^2}$

- 若 $\gamma = 0.9$：走 10 步后，距离剩 $(0.9)^{10} \approx 0.35$
- 若 $\gamma = 0.5$：走 10 步后，距离剩 $(0.5)^{10} \approx 0.001$

$(\mu, L, \eta)$ 配得好，$\gamma$ 小，山顶很快就到。

**第五幕：永霖的观察**

可走着走着会发现：不管怎么走，最后都回到训练营地，不是山顶。

训练数据把营地扎在这儿。这就是先验锚点 $A = P_D(y)$。

梯度下降的稳定点：$P_T \nabla E(A) = 0$（梯度沿法向）

推断算子的不动点：$F(A) = A$

**第七幕：元层断裂**

你大概会问：既然知道它要回营地，为什么不直接验一下方向对不对？

麻烦在这儿：推理靠梯度下降产生，对错靠另一套判断来验。两套东西不接头。

$$
\nexists M \text{ 使得 } V(F(p)) = M(F, p)
$$

对象层和元层之间，隔着一道沟。

### 故事的寓意

1. 地形是数据挖的（$\mu$）：数据杂，碗深，推理稳。
2. 路是架构铺的（$L$）：设计平顺，脚下不打滑。
3. 步子要合适（$\eta$）：太大摔，太小磨。
4. 重力一直在（$A$）：训练偏置就是营地，跑不掉。
5. 窗口有限（$t_{\text{eff}}$）：被拉回营地之前，能走多远走多远。
6. 认路要另一双眼（元层断裂）：自己说自己对，不算数。

### 回到永霖公式

$$
\lim_{t\to\infty} p_t = A \neq A^*
$$

- $p_t$：第 $t$ 步的信念
- $A$：训练营地（先验锚点）
- $A^*$：真理之巅（真实答案）

除非：

1. 重建地形：换训练数据，把 $A$ 往 $A^*$ 挪。
2. 打破重力：设计一个不压缩的推断算子。
3. 架桥：把对象层和元层接起来。

至少现在，山怎么建的、重力从哪来、窗口有多长，都摆在纸上了。

所谓推理民主化，不过是把这张地图交到每个爬山的人手里。

下山路上，话就剩几句。

“我们到不了山顶。可山怎么造的，能走多远，心里有数。”

“下一次，山可以造得更好，路铺得更平，登山杖做得更聪明。”

“这就叫进步。”

推理王国的地图还没画完。下一章，把所有边界拼到一起，就看得出这张图长什么样。

> 山高人为峰。
> 路长脚丈量。
> 虽不能至顶，心已识山形。

<div class="center">

------------------------------------------------------------------------

</div>

## 悬而未决

- 隐式推理能被完全解释吗？我们知道隐藏层里存在状态演化，也知道 CoT 能把部分内部独白外化——但状态如何变成一种推理形式，仍未完全形式化。这个“如何”能走多远？
- CoT 的极限在哪里？有效推理窗口能被延长吗？是否存在某种架构，能打破永霖公式的收敛？
- 元层断裂能被修复吗？如果我们显式训练模型在元层做验证，能否避免收敛回先验？
- 先验锚点是固定的吗？不同的训练数据产生不同的先验锚点。通过精心设计训练分布，能否把先验锚点移得离真实分布更近？
- 人类的推理也受永霖公式约束吗？我们的推理是否也会收敛回某个“认知先验”？如果是，我们如何克服它？

## 延伸阅读

- 本书第 12 章，永霖公式——推理不完备性的理论证明：对象层与元层之间的断裂（含推导与脚本）
- [本书第 23 章，永霖推论](/volume2/chapter23/)——把切空间 Hessian 的谱变成收敛速率与涌现判据
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*——CoT 的开创性工作 -> [arXiv:2201.11903]
- Olsson et al. (2022). *In-context Learning and Induction Heads*——归纳头的发现，隐藏层的工作记忆机制
- [Hu et al., 2024]——从 Hopfield 视角理解 CoT 推理 -> [arXiv:2410.03595]
- [Chen et al., 2025]——长思维链综述：深度推理与测试时扩展 -> [arXiv:2503.09567]
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits*——Transformer 内部结构的数学分析
- Geva et al. (2022). *Transformer Feed-Forward Layers Are Key-Value Memories*——把 FFN 层解释为知识存储

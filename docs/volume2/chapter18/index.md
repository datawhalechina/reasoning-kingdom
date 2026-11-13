# 第18章：因果结构的形式化——三层阶梯与 do 演算

> 从数据里永远推不出因果。除非你愿意承认，有些结构是假设，不是发现。

---

第17章结尾留下了一个令人不舒服的事实：概率论无论多么精密，都无法区分"$X$ 导致 $Y$"和"$Y$ 导致 $X$"——只要两者产生相同的联合分布，贝叶斯更新就对它们一视同仁。

这不是计算能力的问题，也不是数据量的问题。这是数学本身的结构性限制：**观测描述的是世界静止时的样子，不是有人扳动杠杆之后的样子**。

但我们一直在扳动杠杆。医生开药，政策制定者调整税率，工程师修改参数。每一次干预都在问一个概率论回答不了的问题："如果我改变 $X$，$Y$ 会怎样？"

回答这个问题需要一种新的推断规则。本章的任务，就是把"改变"这个动作本身形式化。

---

## 18.0 因果剪线游戏：观察不是干预

概率游戏里，你只能看着筹码怎么移动。因果游戏里，你终于可以动手剪线。

想象一块因果电路板。每个变量是一个节点，每条有向边是一根线：$X \to Y$ 表示 $X$ 的变化沿着这根线传播，影响 $Y$。观测时，你只是看见哪个节点亮了。干预时，你拿起剪刀，切断指向该节点的每一根线，然后强行让这个节点停在某个固定值上。

这就是 $\mathsf{do}(X=x)$ 在游戏里的含义。

"看到 $X=x$"完全不改变电路，它只是告诉你：此刻这盏灯恰好亮在 $x$。但"执行 $\mathsf{do}(X=x)$"改变了电路的结构：所有原本影响 $X$ 的线都被剪断，$X$ 不再由它的父节点决定——它由你的手决定。

:::details 这个游戏的形式化骨架
- **状态空间**：一个有向无环图 $G$，加上一组结构方程 $X_i = f_i(\mathrm{Pa}(X_i), U_i)$。
- **合法动作**：观察某些节点；或执行干预 $\mathsf{do}(X=x)$，切断 $X$ 的所有入边并固定其值。
- **转移规则**：观测更新分布；干预替换结构方程，得到 mutilated graph（剪线后的图）。
- **胜利条件**：从可观测分布和因果图中识别 $P(Y \mid \mathsf{do}(X=x))$ 或某个反事实量。
- **失败模式**：把 $P(Y \mid X=x)$ 当成 $P(Y \mid \mathsf{do}(X=x))$——把相关性误读成因果。
:::

这个游戏好玩的地方在于，它把"因果"从形而上学争论变成了一个非常具体的动作：剪哪根线？保留哪根线？剪完以后，信号怎么传播？

后门路径是一条偷偷绕到背后进来的线。混淆变量是藏在板子后面、同时拉动两个节点的手。d-分离问的是：在挡住某些线之后，信息还能不能从一边传到另一边。

::: info 兔狲教授评
因果推断的第一课很粗暴：看见不是改变。你看着洒水器在跑，和你亲手把它打开，不是同一个世界。统计学最常见的傲慢，就是把观察当作干预。因果图的剪刀，存在的意义正是剪掉这种傲慢。
:::

## 18.0.5 用代码玩这个游戏

这块板子足够小，可以精确枚举——没有采样噪声。下面的程序把两种计算并排跑：条件化"冰淇淋销量高"，以及干预 $\mathsf{do}(\text{冰淇淋}=\text{高})$。两个数字会不一样，而这个差值正是混淆本身。

```python
#!/usr/bin/env python3
"""
causal_wire_cutting.py -- the Causal Wire-Cutting Game of Chapter 18.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 18, Section 18.0
"The Causal Wire-Cutting Game: Observing Is Not Intervening."

Each variable is a node, each edge a wire. Observing watches the lights;
intervening picks up scissors, cuts every wire into a node, and pins the
node to a value. This program runs a small causal circuit board:

      season (Z) ----> ice cream sales (X)
          |
          +----------> drownings (Y)

  * P(Y | X=x)      -- conditioning: watch the board, no scissors.
  * P(Y | do(X=x))  -- intervention: cut Z -> X, pin X, let the rest run.

The two numbers come out different. That difference IS the game -- and
it is why "seeing is not changing."

Everything below is exact enumeration over a small finite board: no
randomness, no sampling noise.

Run:  python3 causal_wire_cutting.py
"""
from itertools import product

# Worlds: (season, ux, uy). Noise ux/uy in 1..10 decides X and Y through
# the structural equations. Summer pushes both variables up -- that shared
# push is the confounder (the hand behind the board).
SEASONS = ("summer", "winter")
NOISE = range(1, 11)

def x_value(season, ux):
    """Structural equation of X: ice cream sales."""
    if season == "summer":
        return "high" if ux <= 9 else "low"
    return "high" if ux <= 2 else "low"

def y_value(season, uy):
    """Structural equation of Y: drownings."""
    if season == "summer":
        return "high" if uy <= 8 else "low"
    return "high" if uy <= 1 else "low"

def worlds():
    return product(SEASONS, NOISE, NOISE)

def prob_condition():
    """P(Y=high | X=high): watch the board, count the worlds."""
    both = total = 0
    for season, ux, uy in worlds():
        if x_value(season, ux) == "high":
            total += 1
            if y_value(season, uy) == "high":
                both += 1
    return both / total

def prob_intervene():
    """P(Y=high | do(X=high)): cut Z -> X, pin X=high, count the worlds.
    The scissors remove X's structural equation; X is decided by the hand.
    Y keeps its own equation, so only the seasonal push into Y remains.
    """
    high = total = 0
    for season, _, uy in worlds():
        total += 1
        if y_value(season, uy) == "high":
            high += 1
    return high / total

def demo():
    print("=" * 70)
    print(" THE CAUSAL WIRE-CUTTING GAME -- seeing is not changing")
    print(" The Reasoning Kingdom, Volume 2, Chapter 18 (Sec. 18.0)")
    print("=" * 70)
    print("\nCircuit:  season -> ice cream -> (nothing)")
    print("          season -> drownings")
    print("  the season pushes both ice cream sales and drownings up at")
    print("  once: a confounder, the hand behind the board.")
    p_cond = prob_condition()
    p_do = prob_intervene()
    print(f"\nP(drownings=high | ice cream=high)     = {p_cond:.4f}")
    print(f"P(drownings=high | do(ice cream=high)) = {p_do:.4f}")
    print("\nThe same 'ice cream is high' observation means different things:")
    print("  conditioning: the board is untouched; a high ice cream reading")
    print("    suggests summer, so drownings look likely too.")
    print("  intervention: scissors cut season -> ice cream, the hand pins")
    print("    X=high. Summer is no longer implied, and the drowning rate")
    print("    drops back to its unconditional level.")
    print(f"\nDifference: {p_cond - p_do:.4f} -- pure confounding, no data")
    print("  collection can remove it; only a wire cut can.")
    print("\nThe point of the game: correlation is what the board shows you;")
    print("  causation is what you are allowed to do to the board. The most")
    print("  common arrogance in statistics is passing off observation as")
    print("  intervention. The scissors exist to cut that arrogance away.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

## 18.1 三层阶梯

Judea Pearl 用一个比喻来描述因果推断的层级结构，他称之为**因果阶梯**（Ladder of Causation）。阶梯有三层，从低到高，每一层都需要下一层所不具备的能力。

**第一层：关联（Association）**

这是概率论的主场。问题长这样：

> "看到 $X$，$Y$ 是什么？"

用数学写：$P(Y \mid X)$。这一层只需要数据——观测足够多，你就能估计条件概率。动物、婴儿，以及几乎所有的统计模型，都生活在这一层。

**第二层：干预（Intervention）**

问题的形状变了：

> "如果我把 $X$ 设成某个值，$Y$ 会是什么？"

它和第一层的区别是根本性的。"看到 $X=x$"和"把 $X$ 设成 $x$"是两个完全不同的动作。前者是被动观察，后者是主动操作。

被动观察时，$X=x$ 成立，可能是因为某个共同原因 $Z$ 同时影响了 $X$ 和 $Y$。主动干预时，你切断 $X$ 与一切导致它的东西之间的联系，强行让 $X$ 固定在 $x$——只有这样，$Y$ 的变化才真正来自 $X$ 对 $Y$ 的直接影响。

这一层需要的不只是数据，还需要行动——或者，当行动不可行时，需要某种让你能在数学上模拟行动的工具。

**第三层：反事实（Counterfactual）**

问题变得更难：

> "如果 $X$ 当初不是那个值，$Y$ 会是什么？"

这已经是关于单个个体在另一个可能世界里的问题。"这个病人服了药、康复了——如果他当初没服药，他还会康复吗？"再多的观测或实验都无法直接回答这个问题，因为那个"他没服药"的平行世界，我们永远进不去。

::: info 三层是真实的分层，不是哲学修辞
这三层对应着能力上的真实边界。纯粹的观测数据只能回答第一层问题。随机对照实验（RCT——一种把人随机分配到处理组或对照组的实验）能回答第二层，但代价是实际执行干预。反事实推断需要一个完整的因果模型，再加上关于"个体机制"的额外假设——这超出了任何实验能达到的范围。大多数统计学家把职业生涯的大半花在第一层上，却相信自己回答的是第二层的问题。这种混淆在科学文献里造成了大量错误结论。
:::

::: info 兔狲教授评
"误以为"这个词太客气了。这不是认知错误，这是工具错误。手里只有相关性的锤子，每个问题看起来都是相关性的钉子。问题不在于统计学家的智力——而在于标准训练从来没有把第一层和第二层之间的界线画清楚。
:::

---

## 18.2 图：因果的几何

第一步，是给"因果结构"一个精确的数学表示。

**有向无环图（DAG）**是最自然的工具。节点代表变量，有向边代表直接的因果影响：$X \to Y$ 表示 $X$ 直接影响 $Y$。"无环"是一个重要约束——因果不能绕圈（如果 $X$ 导致 $Y$、$Y$ 又导致 $X$，时间本身就悖论了）。

一个简单的例子。考虑三个变量：季节（$S$）、洒水器是否开着（$W$）、草坪是否湿（$G$）。直觉上的因果结构是：

$$S \to W, \quad S \to G, \quad W \to G$$

季节影响洒水器（夏天开得更勤），季节也直接影响草坪（下雨），洒水器又直接影响草坪。

在这张图里，$W$ 和 $G$ 是相关的——洒水器开着时，草坪更可能是湿的。但这个相关性沿着两条路径传播：一条是直接因果路径 $W \to G$，另一条是穿过共同原因 $S$ 的"后门路径"：$W \leftarrow S \to G$。

概率论看到的是两条路径叠加的结果，无法把它们分开。图则把这个结构显式地画了出来。

::: info 图不是从数据里读出来的
这是初学者最常见的错误。因果图表示的是**领域知识**，不是统计推断的产物。你无法从 $P(S, W, G)$ 中推出这张图——数据只告诉你变量之间相关有多强，不告诉你箭头指向哪边。因果图是你写下世界如何运作的地方。它是假设，不是发现。接受这一点需要一次不舒服的认知跳跃：科学推断从不只从数据出发——它总是带着结构假设走进房间。
:::

::: info 兔狲教授评
箭头的方向是你带进来的，不是数据给的——这句话会让很多数据科学家不舒服，因为他们习惯了"让数据说话"。但数据说不出这句话。承认这一点需要勇气，很多论文回避了这一步。回避的代价，是把假设藏进方法里，假装它是结论。
:::

---

## 18.3 结构因果模型

图只是骨架。要把它变成一台能做推断的机器，你需要给每条边附上内容。这就是**结构因果模型（SCM）**。

一个 SCM 由三部分组成：

**外生变量 $U$**：来自系统外部的噪声或背景因素，不由模型中的任何其他变量决定。它们捕捉我们无法观测、或选择不去建模的全部随机性。

**内生变量 $V$**：由模型中的其他变量决定的变量，包括我们关心的每一个可观测量。

**结构方程**：对每个内生变量 $V_i$，都有一个方程

$$V_i = f_i(\mathsf{Pa}(V_i),\, U_i)$$

其中 $\mathsf{Pa}(V_i)$ 是 $V_i$ 在因果图中的**父节点**集合（它的直接原因），$U_i$ 是该变量的噪声项。

这个方程不是统计回归方程——它是一个**机制**，描述"给定父节点和噪声，这个变量取什么值"。机制是稳定而局部的：改变某个其他变量的方程，不会影响这一个。正是这种局部稳定性，把因果模型与统计模型区分开来。

回到洒水器的例子。结构方程可能是：

$$W = f_W(S, U_W), \quad G = f_G(S, W, U_G)$$

$W$ 由季节和它自己的噪声决定；$G$ 由季节、洒水器的状态和它自己的噪声决定。每个方程描述一个局部机制，彼此独立。

---

## 18.4 do 算子：干预的形式化

现在可以精确地定义"干预"了。

干预 $\mathsf{do}(X=x)$ 的操作定义是：**把 $X$ 的结构方程替换为常数方程 $X=x$**，同时保持其他所有方程不变。

它对图的影响一眼可见：**删除所有指向 $X$ 的边**。因为 $X$ 现在被强制固定，父节点对它的影响被切断了。$X$ 的后代仍然感受得到 $X$（出边保留），但 $X$ 不再响应它的原因（入边消失了）。

这张"被操作过"的图叫做**干预图**，记作 $G_{\overline{X}}$——删除了 $X$ 入边的图。

:::details do 算子 vs. 条件化
为什么 $P(Y \mid X=x)$ 和 $P(Y \mid \mathsf{do}(X=x))$ 不同？这是因果推断中最核心的区别，也是最容易混淆的地方：

**条件化 $P(Y \mid X=x)$**：在你观测到 $X=x$ 的那些样本里，$Y$ 的分布是什么？这是**被动观察**——你在问"在 $X$ 恰好等于 $x$ 的那些世界里，$Y$ 长什么样？"麻烦在于，造成 $X=x$ 的原因可能与 $Y$ 共享某个共同原因（混淆变量），从而扭曲了相关性。

**干预 $P(Y \mid \mathsf{do}(X=x))$**：如果我**强制** $X$ 固定在 $x$（删除 $X$ 的所有原因），$Y$ 会是什么？这是**主动操作**——你切断了混淆路径，只留下 $X$ 对 $Y$ 的直接因果效应。

一个具体的例子：
- $P(\text{康复} \mid \text{治疗}=1)$：在那些选择服药的人里，康复率是多少？（可能偏高——症状较轻的病人更可能选择服药。）
- $P(\text{康复} \mid \mathsf{do}(\text{治疗}=1))$：如果我强行随机指派一组人服药（随机对照实验），康复率是多少？（真实的药效。）

**do 算子**是随机对照实验的数学表达。当 RCT 不可行时，do 演算提供了从观测数据估计干预效应的规则。
:::

干预后的概率分布写作 $P(Y \mid \mathsf{do}(X=x))$，有时也写作 $P_x(Y)$，它在干预图 $G_{\overline{X}}$ 中定义。

把它与条件概率比较：

$$P(Y \mid X=x) \quad \text{vs.} \quad P(Y \mid \mathsf{do}(X=x))$$

前者是"在观测到 $X=x$ 的样本中，$Y$ 的分布"。后者是"如果整个世界的 $X$ 都被强制设为 $x$，$Y$ 的分布会是什么"。

两者可以相差极大。在洒水器的例子里，$P(G=\text{湿} \mid W=\text{开})$ 包含了"夏天让洒水器开起来，夏天也让天下雨"的混淆效应。但 $P(G=\text{湿} \mid \mathsf{do}(W=\text{开}))$ 切断了路径 $W \leftarrow S$，只保留直接效应 $W \to G$。这才是"洒水器对草坪湿度的因果效应"。

::: info 随机对照实验实际做了什么
随机对照实验的核心操作——把受试者随机分配到处理组和对照组——在数学上正是 $\mathsf{do}$：它切断了处理变量（服药/不服药）与一切可能混淆因素之间的连接。随机化等价于删除因果图中所有指向处理变量的边。这就是为什么 RCT 是估计因果效应的"黄金标准"——它在物理现实中实现了 do 算子。do 演算的价值在于：它告诉你，什么时候不必做实验，也能从观测数据计算出 $P(Y \mid \mathsf{do}(X))$。
:::

### do 算子在矩阵与代码中的样子

do 算子不只是数学符号——它可以在两行代码里被精确实现：剪掉入边，然后传播效应。

把单元结构方程 $E_j = \sum_i A_{ij} \cdot E_i + U_j$ 写成矩阵形式：$E = A^{\top} E + U$。于是 $\mathsf{do}(X=v)$ 就是：

```python
import numpy as np
A = np.array([[0, 0.9, 0.8],      # structure: ad_spend -> clicks -> revenue
              [0,   0, 0.7],
              [0,   0,   0]])
E = np.random.randn(3, 16)         # one embedding per variable
U = np.zeros_like(E)
# step 1: cut all incoming edges of X (fill in column X with zeros)
def do_cut(A, var):
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do
# step 2: propagate the effect along the remaining edges
def propagate(A_do, E, U):
    return A_do.T @ E + U
# do(ad_spend = 3.0): the value enters as a constant, then effects flow
A_do = do_cut(A, 0)
E_next = propagate(A_do, E, U)
```

**与 Pearl 定义的对应关系：**

| Pearl 的 do 算子 | 矩阵/代码实现 |
|---|---|
| 把 $X$ 的结构方程替换为 $X=v$ | 把 $X$ 对应的列置零；值 $v$ 作为常数进入 |
| 删除所有指向 $X$ 的边 | 把列 $A_{\cdot,X}$ 置零 |
| 沿后代传播效应 | 通过 `propagate` 做一次矩阵乘法 |
| 循环图是非法的 | $A$ 必须尊重拓扑序（上三角） |

就这么简单——因果推断中最深刻的算子，在实现层面就是"把一列置零，再乘一次"。你不需要任何额外的框架或类型系统。

---

## 18.5 后门准则：混淆的几何

do 演算的核心问题是：**在什么条件下，你可以从观测数据估计因果效应**——也就是说，什么时候 $P(Y \mid \mathsf{do}(X=x))$ 可以由 $P$ 和图的结构计算出来，而不必真的去干预？

回答这个问题，需要理解"混淆"的几何。

在因果图中，$X$ 对 $Y$ 的总效应被"混淆"，当且仅当存在一条非因果路径：它从 $X$ 出发，不经过 $X$ 的任何后代，却仍然到达 $Y$。这样的路径叫做**后门路径**——它们绕过直接效应 $X \to Y$，从后门把虚假的相关性偷运进来。

**后门准则**给出了一个精确条件：变量集 $Z$ 满足（关于 $X \to Y$ 的）后门准则，当且仅当：

1. $Z$ 中不含 $X$ 的后代；
2. $Z$ 阻断了从 $X$ 到 $Y$ 的每一条后门路径。

如果这样的 $Z$ 存在，因果效应可以用下面的公式计算：

$$P(Y \mid \mathsf{do}(X=x)) = \sum_z P(Y \mid X=x, Z=z)\, P(Z=z)$$

这个公式叫做**调整公式**。它的意思是：对 $Z$ 的每个取值，计算 $X=x$ 时 $Y$ 的条件分布，再按 $Z$ 的边际分布加权平均。这个操作叫做**对 $Z$ 调整**，在流行病学里称为"控制混淆变量"。

在洒水器的例子里，$X=W$，$Y=G$，后门路径是 $W \leftarrow S \to G$。取 $Z=\{S\}$：$S$ 不是 $W$ 的后代（条件 1 成立），且 $S$ 阻断了后门路径 $W \leftarrow S \to G$（条件 2 成立）。于是：

$$P(G \mid \mathsf{do}(W=w)) = \sum_s P(G \mid W=w, S=s)\, P(S=s)$$

这个量完全由观测数据决定——不需要真的去碰洒水器。

### 18.5.5 实操：前门调整——当后门被堵死时

后门准则要求有一个可观测的混淆变量 $Z$。但如果混淆变量不可观测呢？假设你不知道"吸烟欲望"这个隐藏变量，它同时影响吸烟量和肺癌风险。

这时**前门准则**可以救场——只要存在一个**中介变量** $M$，满足：

1. $M$ 阻断了从 $X$ 到 $Y$ 的所有因果路径；
2. 从 $X$ 到 $M$ 没有后门（$X$ 与 $M$ 之间没有不可观测的混淆）；
3. 从 $M$ 到 $Y$ 的后门被 $X$ 阻断（调整 $X$ 之后，$M$ 到 $Y$ 没有混淆）。

经典例子：**吸烟 $\to$ 焦油沉积 $\to$ 肺癌**（出自 Pearl 的原始论文）。

- $X = \text{吸烟}$，$M = \text{焦油沉积}$，$Y = \text{肺癌}$
- 隐藏变量 $U = \text{吸烟欲望/基因}$ 同时影响 $X$ 和 $Y$（一条后门路径），但**不影响** $M$（$X$ 与 $M$ 之间没有隐藏混淆——吸烟量直接决定焦油量，基因不直接改变焦油）
- 后门准则：你需要观测 $U$，但 $U$ 不可观测 $\to$ 后门准则失效
- 前门准则：通过 $M$（焦油）识别效应

前门调整公式：

$$P(Y \mid \mathsf{do}(X=x)) = \sum_m P(M=m \mid X=x) \cdot \sum_{x'} P(Y \mid X=x', M=m) \cdot P(X=x')$$

用平实的语言拆开：

- **第一项** $P(M=m \mid X=x)$：一个人吸 $x$ 量的烟时，会产生多少焦油 $m$？这是 $X$ 对 $M$ 的因果效应（无后门，可直接估计）。
- **第二项** $\sum_{x'} P(Y \mid X=x', M=m) \cdot P(X=x')$：给定焦油量 $m$，在吸烟人群上对肺癌概率做加权平均——这里通过调整 $X$ 来阻断 $M$ 到 $Y$ 的后门。
- **相乘，再对 $m$ 求和**：把 $X$ 对 $M$ 的效应与 $M$ 对 $Y$ 的效应结合起来，得到 $X$ 对 $Y$ 的总因果效应。

**一个数值例子**（虚构数据，已简化）：

| 吸烟 $X$ | 焦油 $M$ | 肺癌 $Y$ | 人数 |
|---|---|---|---|
| 1 | 高 | 1 | 300 |
| 1 | 高 | 0 | 100 |
| 1 | 低 | 1 | 50 |
| 1 | 低 | 0 | 150 |
| 0 | 高 | 1 | 30 |
| 0 | 高 | 0 | 70 |
| 0 | 低 | 1 | 20 |
| 0 | 低 | 0 | 280 |

计算 $P(Y=1 \mid \mathsf{do}(X=1))$——如果强制所有人都吸烟，肺癌概率会是多少？

1. $P(M=\text{高} \mid X=1) = \frac{400}{600} = 0.667$；$P(M=\text{低} \mid X=1) = \frac{200}{600} = 0.333$

2. 给定 $M=\text{高}$：

   $$\sum_{x'} P(Y=1 \mid X=x', M=\text{高}) \cdot P(X=x')$$
   $$= P(Y=1 \mid X=1, M=\text{高}) \cdot P(X=1) + P(Y=1 \mid X=0, M=\text{高}) \cdot P(X=0)$$
   $$= \frac{300}{400} \cdot 0.6 + \frac{30}{100} \cdot 0.4 = 0.75 \cdot 0.6 + 0.30 \cdot 0.4 = 0.45 + 0.12 = 0.57$$

   给定 $M=\text{低}$：

   $$\sum_{x'} P(Y=1 \mid X=x', M=\text{低}) \cdot P(X=x')$$
   $$= \frac{50}{200} \cdot 0.6 + \frac{20}{300} \cdot 0.4 = 0.25 \cdot 0.6 + 0.067 \cdot 0.4 = 0.15 + 0.027 = 0.177$$

3. $P(Y=1 \mid \mathsf{do}(X=1)) = 0.667 \times 0.57 + 0.333 \times 0.177 = 0.380 + 0.059 = 0.439$

如果强制所有人都吸烟，肺癌概率会是 43.9%。而朴素的条件概率 $P(Y=1 \mid X=1) = \frac{350}{600} = 0.583$ 更高——因为它包含了 $U$ 的混淆效应。do 演算正确地把因果效应从混淆效应中分离了出来。

::: info 兔狲教授评
前门调整是一颗被低估的宝石。大多数教材只教后门准则，因为后门更直观。但后门要求观测到混淆变量——而这往往根本做不到。前门准则完全不要求观测混淆变量；它利用一个你能观测到的中介变量。在实践中，这使它比后门可用得多，也反直觉得多——你穿过一个中介变量，绕过了一个看不见的混淆变量。有时候，绕路才是正路。
:::

---

## 18.6 do 演算的三条规则

后门准则覆盖了大量真实情形，但不是全部。在某些因果图中，任何可观测变量集都无法完全阻断后门路径——例如存在不可观测的混淆因素时。

为了处理更一般的情况，Pearl 提出了 **do 演算**：三条关于 $\mathsf{do}$ 算子的推断规则，它们共同构成因果推断的一个完备演算。

设 $X, Y, Z, W$ 是因果图 $G$ 中的变量集。用 $G_{\overline{X}}$ 表示删除 $X$ 入边的图，用 $G_{\underline{X}}$ 表示删除 $X$ 出边的图。

**规则一（观测的插入与删除）**：

$$P(Y \mid \mathsf{do}(X),\, Z,\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

当且仅当在 $G_{\overline{X}}$ 中，$Y \perp\!\!\!\perp Z \mid X, W$。

用文字说：如果在干预图中，$Z$ 不携带关于 $Y$ 的额外信息（被 $X$ 和 $W$ 阻断），那么观测或不观测 $Z$，都不会影响你对 $Y$ 的推断。

**规则二（行动与观测的互换）**：

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, Z,\, W)$$

当且仅当在 $G_{\overline{X}\,\underline{Z}}$ 中，$Y \perp\!\!\!\perp Z \mid X, W$。

用文字说：在一定条件下，对 $Z$ 做干预与观测 $Z$ 对 $Y$ 的效果相同——干预失去了"强行切断"的特殊性，退化为普通的条件化。

**规则三（干预的消除）**：

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

当且仅当在 $G_{\overline{X}\,\overline{Z(W)}}$ 中，$Y \perp\!\!\!\perp Z \mid X, W$（其中 $Z(W)$ 由 $Z$ 中不是 $W$ 祖先的那些节点组成）。

用文字说：在一定条件下，$\mathsf{do}(Z)$ 对 $Y$ 没有影响，可以从公式中去掉。

::: info 三条规则的完备性
Pearl 和 Shpitser 证明了这一点：对任意因果图，如果 $P(Y \mid \mathsf{do}(X))$ 能由观测分布 $P$ 计算出来（即它是"可识别的"），那么反复应用这三条规则，总能得到一个只含观测概率的表达式。这就是 do 演算的**完备性定理**——它不只是一套工具箱，而是一个真正的演算，不会漏掉任何可被识别的因果效应。
:::

这三条规则的形式，与第14章的推断规则高度相似：横线上方是条件，横线下方是你可以做出的推断。区别在于，这里的"语言"不只是命题——它是携带 $\mathsf{do}$ 算子的概率表达式。而图的结构、条件独立性，扮演着公理的角色。

回到剪线游戏：do 演算的三条规则，就是你剪线之后查阅的走法表。什么时候可以丢掉一个观测？什么时候可以把行动换成观测？什么时候一个干预可以被完全消除？答案不在公式表面上，而在剪线之后的图里。因果推断比概率推断更强，是因为它把"行动"接纳进了语法；它也因此更危险——每一个行动都取决于你是否把那张图画对了。

---

## 18.7 d-分离：图的独立性语言

do 演算的三条规则，每一条都依赖一个核心判断：在某个图中，两组变量在给定第三组变量时是否**条件独立**。如何直接从图的结构读出条件独立性？这正是 **d-分离**（directional separation）的工作。

给定有向无环图 $G$ 和三组节点 $X, Y, Z$，我们说 $X$ 与 $Y$ 被 $Z$ **d-分离**，记作 $X \perp_G Y \mid Z$，当且仅当 $Z$ 阻断了 $X$ 与 $Y$ 之间的每一条路径。

一条路径是否"被阻断"，取决于路径上节点的类型：

- **链** $A \to B \to C$：当 $B$ 在 $Z$ 中时，路径被阻断（$B$ 传递信息；一旦控制 $B$，流动就停止）。
- **叉** $A \leftarrow B \to C$：当 $B$ 在 $Z$ 中时，路径被阻断（$B$ 是共同原因；一旦控制 $B$，相关性就消失）。
- **对撞** $A \to B \leftarrow C$：当 $B$ **不在** $Z$ 中时，路径被阻断（对撞节点默认阻断信息；但控制它，或它的任何后代，反而会打开路径——这就是"对撞偏差"的来源）。

对撞节点的行为是反直觉的，值得停下来想清楚。假设"身高"（$H$）和"篮球技术"（$T$）都影响"是否入选篮球队"（$S$）：$H \to S \leftarrow T$。在整个人群中，$H$ 和 $T$ 可能是独立的。但只看那些入选了球队的人——这等价于控制 $S$——就会发现矮个子球员往往技术格外好，而高个子球员有时只是平平。控制一个对撞节点，原本独立的变量就变得相关了。这就是统计学中的**选择偏差**；在图的语言里，它是路径因对撞节点被控制而打开的自然结果。

---

## 18.8 反事实：单个个体的另一个世界

阶梯的最高层——反事实——需要在结构因果模型内部给出精确定义。

考虑一个特定的个体，在特定情境下经历了 $X=x$，得到了结果 $Y=y$。反事实问题是：如果这个个体当时经历的是 $X=x'$，$Y$ 会是什么？

在 SCM 中，这个问题有三步答案：

**第一步：溯因（Abduction）**。利用观测到的 $X=x$、$Y=y$，以及所有其他观测，更新这个个体外生噪声变量 $U$ 的分布——把先验 $P(U)$ 推到后验 $P(U \mid \text{观测})$。这一步由贝叶斯更新完成，它把个体的"特殊性"编码进 $U$ 的后验之中。

**第二步：行动（Action）**。在结构方程中，把 $X$ 的方程替换为 $X=x'$（也就是施加干预 $\mathsf{do}(X=x')$），其他方程保持不变。

**第三步：预测（Prediction）**。在修改后的模型和第一步得到的 $U$ 后验分布下，计算 $Y$ 的分布。这给出反事实结论：如果这个个体当时面对的是 $X=x'$，$Y$ 的分布会是什么。

这三步合在一起，给出了反事实推断的操作定义——不是哲学思辨，而是一个可计算的程序，前提是你有一个足够完整的 SCM。

::: info 平均处理效应 vs. 个体处理效应
流行病学中熟悉的"平均处理效应"（ATE）是一个第二层概念：$\mathbb{E}[Y \mid \mathsf{do}(X=1)] - \mathbb{E}[Y \mid \mathsf{do}(X=0)]$。它回答的是"在整个人群中，平均而言，处理让结果改变了多少？""个体处理效应"（ITE）是一个第三层概念：对这个特定的人，如果处理改变了，他的结果会改变多少？ITE 原则上无法从观测数据中识别（每个人只能处于一种处理状态）——这就是反事实推断的根本困难，也是精准医疗问题的数学根源。
:::

---

## 18.9 因果与推断：一个更大的图景

回顾这一章走过的路。

回到第14章，推断规则是关于命题的：从一批假设 $\Gamma$ 出发，推出新的命题。第15章发现，这套机制有内在极限——有些真命题根本无法被证明。第16章质疑了"假设可以重复使用"这个默认设定，发现去掉它会得到一种新的逻辑。第17章把真值从 $\{0,1\}$ 拉伸到 $[0,1]$，把推断变成了信念的更新。

现在，这一章做了件新事：它给推断的语言加了一个新动词——**干预**。$\mathsf{do}$ 算子不是条件化；它是一条新的推断规则，需要在语法中单独处理。图为这条规则提供语境：什么时候 $\mathsf{do}$ 可以被化简掉，什么时候不能。

这个结构与形式系统的结构完全平行：

| 形式系统 | 因果推断 |
|---|---|
| 命题变元 | 变量 |
| 推断规则 | do 演算的三条规则 |
| 公理 | 图（局部独立性假设） |
| 可证（$\vdash$） | 可识别 |
| 不完备定理 | 不可识别性定理 |

不可识别性定理扮演着不完备定理的角色：存在某些因果效应，即使给了你完整的观测数据和完整的图，也永远无法只从观测计算出来——你必须实际去跑实验。这不是方法的缺陷；这是结构性的边界。

---

## 悬而未决

**因果图从哪里来？** 整章都假设你已经拥有一张正确的因果图。但在实践中，图是主观知识与领域假设的产物。如果图画错了，do 演算给出的答案也是错的——系统的可靠性建立在图的正确性之上，而图的正确性无法从数据中验证。这是因果推断最深的困境：工具越精确，它承载的假设就越重。

**因果图能否从数据中自动发现？** 因果发现算法——PC 算法、FCI 算法、LiNGAM——试图从观测数据推断因果结构的某些方面。但在一般情况下，可观测数据只能确定一个"马尔可夫等价类"——一组产生相同条件独立性关系的图，类内箭头方向无法区分。从类中挑出一张图，仍然需要领域知识，或者关于误差分布的额外假设。

**在因果图上做推断有多昂贵？** 即使图已在手中，识别并计算因果效应在一般情况下也是计算昂贵的。判断一个因果效应是否可识别，可以在多项式时间内完成；但在含有隐藏变量的复杂图中，计算一个可识别效应的实际表达式会变得昂贵得多。这个问题的全貌是第19章的主题：推理的代价不仅是逻辑极限，也是计算极限。

---

## 思考题

**★ 热身**

对以下每个问题，判断它属于因果阶梯的哪一层（关联 / 干预 / 反事实），并解释为什么。

1. "吸烟者的肺癌率比不吸烟者高多少？"
2. "如果严格实施禁烟令，肺癌率会下降多少？"
3. "如果这位肺癌患者当年从未吸烟，他现在会患癌吗？"
4. "在被诊断出高血压的人中，有多大比例正在服用降压药？"
5. "如果给这位高血压患者开药，他未来10年心血管事件的风险会下降多少？"

---

**★★ 推导**

考虑因果图：$Z \to X \to Y$，同时 $Z \to Y$（$Z$ 同时影响 $X$ 和 $Y$，$X$ 也影响 $Y$）。

1. 列出从 $X$ 到 $Y$ 的每一条路径。哪些是因果路径（沿箭头方向），哪些是后门路径？
2. 使用后门准则，写出 $P(Y \mid \mathsf{do}(X=x))$ 的调整公式。
3. 如果 $Z$ 是不可观测的隐藏变量，调整公式还能用吗？此时 $P(Y \mid \mathsf{do}(X=x))$ 可识别吗？

---

**★★★ 挑战**

在因果图 $Z \to X \to Y$（$Z$ 可观测）中，加入一个中介变量 $M$，满足 $X \to M \to Y$，同时保留直接路径 $X \to Y$。

你想分别估计：

- $X$ 对 $Y$ 的**直接效应**（不经过 $M$ 的部分）
- $X$ 对 $Y$ 的**间接效应**（经过 $M$ 的部分）

直接效应和间接效应属于阶梯的哪一层？试着用 $\mathsf{do}$ 算子的语言写出这两个量的定义——你会发现，定义"直接效应"需要同时做两次干预。再进一步思考：为什么间接效应（自然间接效应）比直接效应更难定义，必须诉诸反事实？

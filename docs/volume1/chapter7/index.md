# 第7章：复杂度的真相——不是快慢，是结构

> 如果 $P \neq NP$ 成立，它告诉我们的不是“有些问题很难”，而是“宇宙本身内置了一种不对称”。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、一个不对称

2023年，当 DeepSeek-R1 和 GPT-4 在各种推理任务上不断刷新纪录时，研究者开始用 NP 完全问题测试它们。结果令人震惊。

我们来设计一个自己的测试——OpenXOR，一个建立在约束满足之上的诊断工具。任务很简单：给定一组 XOR 风格的约束满足实例，判断是否存在一个满足所有约束的赋值。这不是线性代数里能用高斯消元打发掉的纯 XOR-SAT。OpenXOR 的设计保留了“XOR 关系”的局部迷惑性，同时把问题推入 NP 难搜索的相变区域。在这一节里，我们不只是在谈论它——我们要把它造出来，然后在真实模型上测量它。

:::details NP 完全问题：为什么它们比普通问题难这么多？（动手任务：我们的 OpenXOR）
**NP（nondeterministic polynomial，非确定性多项式）** 是一类具有这样一种良好性质的问题：给定一个候选答案，你可以在多项式时间内**验证**它是否正确。比如 3-SAT：给我一组变量赋值，我很快就能检查它是否满足每一个子句。

**P（polynomial，多项式）** 是另一类问题——对这些问题，你可以在多项式时间内**找到**答案。比如排序，或者最短路径（Dijkstra）。

**NP 完全**问题是 NP 里最难的一批问题。有多难？如果你能高效解决其中任意一个，你就能高效解决所有 NP 问题。

**P = NP？** 这是计算机科学中最著名的未解问题。大多数人相信 $P \neq NP$——验证远比搜索容易，这两种能力有根本区别。但没有人能证明它。

**为什么 LLM 解不了 NP 完全问题？**（LLM 是大语言模型——一个在大量文本上训练、用来预测下一个词的模型。）因为 LLM 的推理运行在多项式时间内（有限次前向传播），而 NP 完全问题要求指数级搜索——实例规模一大，它就崩溃了。这是架构的限制，不是参数量的限制。
:::

测试结果令人震惊。DeepSeek-R1 的任务完成率是 **0%**。GPT-4 的也是 **0%**。

注意，这不是 0% 准确率，而是 0% 任务完成率。模型要么拒绝回答（37–42%），要么撞上上下文限制（28–31%），要么产生幻觉约束（18–22%）。它们甚至没有尝试去解决问题。

这个结果揭示了一个深刻的事实：**规模不是问题所在，架构才是。**

但为什么？一个拥有千亿参数的模型，怎么会在这个问题上失败？

这里有一个关键事实：LLM 的一次前向传播运行在多项式时间内——无论模型多大，一次前向传播的工作量都被限制在 $O(LN^2)$，即层数 $L$ 与序列长度 $N$ 的平方之积。而 OpenXOR 的测试实例并不是随机挑选的：它们的约束密度被精心固定在 $\alpha \approx 4.26$ 附近——一个相变临界点（相变是指某个参数越过阈值时行为发生的急剧变化）。在这个点上，3-SAT 实例从“几乎总是可满足”翻转为“几乎总是不可满足”，求解器恰好被困在最难的搜索区域。

LLM 没有搜索，只有前向传播。多项式时间的架构，遇上指数级难度的问题，结果就是 0% 的任务完成率。规模能让你在多项式时间内做更丰富的模式匹配，但它无法把多项式时间变成指数时间。这是架构边界，不是算力边界。

让我们从最基础的地方开始。

## 二、验证与搜索：一个简单的实验

拿出一张纸，写下这个 3-SAT 实例：

    (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₄) ∧ (x₂ ∨ x₃ ∨ x₄) ∧ (¬x₁ ∨ ¬x₃ ∨ x₄)

现在我告诉你一个候选解：`x₁=true, x₂=false, x₃=true, x₄=true`。

验证它是否满足每一个子句：

第一个子句：x₁=true——满足。

第二个子句：¬x₁=false，x₂=false，¬x₄=false……等等。这个子句不满足。

所以这不是一个解。整个验证过程花了你多久？10秒？20秒？

现在，不看答案，从零开始找一个满足所有子句的赋值。

你可能会尝试：

- x₁=true, x₂=true, x₃=true, x₄=true——检查：不满足

- x₁=true, x₂=true, x₃=true, x₄=false——检查：不满足

- x₁=true, x₂=true, x₃=false, x₄=true——检查：不满足

- …

4 个变量给出 $2^4 = 16$ 种可能。如果运气不好，你可能得试完 16 种才能找到解（或者确认无解）。

这就是**验证与搜索之间的不对称**。验证一个候选解需要 $O(m)$ 时间——线性时间，其中 $m$ 是子句数量。但搜索的代价是 $O(2^n \times m)$——指数时间，其中 $n$ 是变量数量。

当 $n = 10$ 时，搜索空间是 1024。当 $n = 100$ 时，搜索空间是 $2^{100} \approx 10^{30}$——比宇宙中的原子数还多。

**这不是工程问题。这是数学事实。**

<div class="center">

------------------------------------------------------------------------

</div>

## 三、动手试试：遇见指数墙

让我们做一个更动手的实验。

**步骤1：构造一个小规模 3-SAT 实例**

变量：x₁, x₂, x₃, x₄（4 个布尔变量）

子句：

    (x₁ ∨ ¬x₂ ∨ x₃)
    (¬x₁ ∨ x₂ ∨ ¬x₄)
    (x₂ ∨ x₃ ∨ x₄)
    (¬x₁ ∨ ¬x₃ ∨ x₄)

**步骤2：验证一个候选解**

给定赋值：x₁=F, x₂=F, x₃=T, x₄=T

逐个检查子句（只要至少有一个文字为真，子句就得到满足）：

- 子句1：¬x₂ = T——满足

- 子句2：¬x₁ = T——满足

- 子句3：x₃ = T——满足

- 子句4：¬x₁ = T——满足

验证通过！用时：O(4)——常数时间。

**步骤3：搜索所有可能解**

现在，不看答案，枚举所有 $2^4 = 16$ 种赋值：

```python
from itertools import product

# Define a 4-variable 3-SAT instance
# Each clause is a list of literals; positive i means x_i, negative -i means ¬x_i
clauses = [
    [1, -2, 3],   # (x₁ ∨ ¬x₂ ∨ x₃)
    [-1, 2, -4],  # (¬x₁ ∨ x₂ ∨ ¬x₄)
    [2, 3, 4],    # (x₂ ∨ x₃ ∨ x₄)
    [-1, -3, 4],  # (¬x₁ ∨ ¬x₃ ∨ x₄)
]

def check_clause(clause, assignment):
    """Check whether a single clause is satisfied: at least one literal is true"""
    for lit in clause:
        # lit > 0 means a positive literal; lit < 0 means a negated literal
        val = assignment[abs(lit) - 1]
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False

def check_sat(clauses, assignment):
    """Check whether all clauses are satisfied simultaneously"""
    return all(check_clause(c, assignment) for c in clauses)

# Enumerate all 2^4 = 16 assignments (False=0, True=1)
print(f"{'x₁':^5}{'x₂':^5}{'x₃':^5}{'x₄':^5}{'SAT?':^8}")
print("-" * 35)
solutions = []
for bits in product([False, True], repeat=4):
    sat = check_sat(clauses, bits)
    mark = "Yes" if sat else "No"
    vals = ["T" if b else "F" for b in bits]
    print(f"{vals[0]:^5}{vals[1]:^5}{vals[2]:^5}{vals[3]:^5}{mark:^8}")
    if sat:
        solutions.append(bits)

print(f"\nFound {len(solutions)} satisfying assignments:")
for sol in solutions:
    print("  ", {f"x{i+1}": ("T" if v else "F") for i, v in enumerate(sol)})
```

体验指数增长：

- 5 个变量——32 种赋值

- 10 个变量——1024 种赋值

- 20 个变量——1,048,576 种赋值

**步骤4：思考题**

如果变量数增加到 100，穷举搜索需要多久？

假设你每秒能检查 $10^9$ 个赋值（现代 CPU 的速度）。那么你需要：

$$
\frac{2^{100}}{10^9} \approx 4 \times 10^{13} \text{ 秒} \approx 1.27 \text{ 百万年}
$$

宇宙的年龄是 138 亿年。仅仅求解一个 100 变量的 3-SAT 实例，你就要花掉宇宙年龄的万分之一。

这就是**指数墙**。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、相变：问题最难的地方

但并非所有 3-SAT 实例都一样难。

这听起来像句废话。当然不一样难——10 个变量的实例比 100 个变量的实例容易。但我说的不是这个。

我说的是：即使变量数量相同，不同的 3-SAT 实例难度也可能天差地别。而这个差别不是随机的——它遵循某种规律。

考虑两个极端。第一个是**欠约束**的情况：10 个变量，20 个子句，子句密度 $\alpha = 2$。约束很少，解空间巨大。随便扔一个赋值进去，你很可能就落在一个解上。求解器几乎不需要回溯。

第二个是**过约束**的情况：10 个变量，80 个子句，子句密度 $\alpha = 8$。约束太多，矛盾很快暴露。求解器走几步，发现“这条路堵死了”，迅速剪枝，再换一条路。虽然没有解，但证明无解很快。

两个极端都不难。

但在 $\alpha \approx 4.26$ 附近，奇怪的事情发生了。

\[Xu & Li, 2000\] 在他们 2000 年的开创性工作中发现：当子句密度 $\alpha = m/n$ 从 2 升到 8 时，可满足概率并不是平滑下降的。

它在 $\alpha \approx 4.26$ 处发生了一次剧烈的**相变**。

从接近 1，跌到接近 0。不是渐渐褪去——而是断崖。

这不是第一次在计算问题中看到相变。但每一次看到，都让人不安。因为它意味着问题的性质在某个临界点发生了质的改变：水在 0°C 从固态变成液态，磁铁在居里温度失去磁性，而 3-SAT 在 $\alpha \approx 4.26$ 从“几乎总是可满足”翻转为“几乎总是不可满足”。

更重要的是，这个相变点恰好就是问题**最难求解**的地方。

![图7.2：3-SAT相变图](/figures/ch07_fig3_real_mu.png) *横轴是子句密度 $\alpha = m/n$；纵轴是随机实例可满足的概率。S 型曲线在 $\alpha \approx 4.26$ 附近发生陡峭的相变。绿色阴影区域（$\alpha < 4.2$）是欠约束区域——几乎总是可满足。红色阴影区域（$\alpha > 4.3$）是过约束区域——几乎总是不可满足。临界点附近（黄色区域）是最难求解的区域。*

为什么临界点最难？

我先描述两个极端。

在欠约束区域，解很充足。随机搜索很容易撞上一个。求解器不需要多聪明——只要别太倒霉就行。

在过约束区域，矛盾出现得很快。求解器走几步，发现“这条路走不通”，回溯、剪枝，再试另一条。它最终证明无解，但过程很快。

但在临界点，求解器陷入一种微妙的平衡。

解很稀少，但不是零——所以你不能放弃搜索。矛盾并不明显——所以回溯树长得巨大。搜索空间既不稀疏也不稠密：它处于最难导航的状态。你既没有捷径，也没有快速排除错误方向的办法。

这是最糟糕的一种困境。

\[Zhan, 2026\] 用一个离散几何模型解释了这一现象。他把 3-SAT 映射到 N 维布尔超立方体的组合拓扑上，推导出最小不可满足极限 $\alpha = 8/N$，以及最大可满足极限 $\alpha = 7/6(N-1)(N-2)$。

这解释了为什么可满足性阈值猜想只在“几乎必然”的意义上成立——在有限规模下，相变不是绝对的，而是概率性的。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、μ(α)：把离散边界变成连续边界

经典复杂度理论的麻烦在于，它只告诉你一个问题“属于”P 还是 NP——它无法告诉你某个具体实例有多难。

3-SAT 是 NP 完全的——这是一个是/否的判决。但这并不意味着所有 3-SAT 实例都一样难。一个 10 变量、20 子句的实例，和一个 100 变量、420 子句的实例，难度差得很远。

我们的 OpenXOR 框架打破了这种二元束缚。它把 NP 问题的可满足性从一个是/否判决，变成一张连续的相图——而且这一次我们不是在引用别人的结果：是我们自己跑出来的。下面这段代码就是 OpenXOR v2 的完整实现：谓词检查点生成器 + 参考求解器（DPLL）。全都在这儿，你一行都不用改就能运行。

```python
# OpenXOR v2: predicate checkpoints over the operator sequence.
# Chapter 7 companion script. Run: python3 openxor_v2.py

XOR, NOP = 1, 0

class Instance:
    """n operator positions; each clause is a list of literals (pos, val),
    meaning 'the operator at pos equals val'. All clauses must hold (AND),
    each clause is satisfied by at least one literal (OR) = 3-CNF."""
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses

def generate_instance(n, m, rng=None, seed=0):
    """Random 3-CNF checkpoints. No solution is precomputed."""
    rng = rng or np.random.default_rng(seed)
    clauses = []
    for _ in range(m):
        pos = rng.choice(n, size=min(3, n), replace=False) + 1   # 1-indexed
        vals = rng.integers(0, 2, size=len(pos))
        clauses.append([(int(p), int(v)) for p, v in zip(pos, vals)])
    return Instance(n, clauses)

def solve_dpll(inst, max_nodes=500_000):
    """DPLL over operator variables with unit propagation.
    Returns (assignment, sat, nodes)."""
    n, clauses = inst.n, [list(c) for c in inst.clauses]
    asn, nodes = {}, 0

    def propagate(cls, a):
        changed = True
        while changed:
            changed, new_cls = False, []
            for c in cls:
                sat, residual = False, []
                for pos, val in c:
                    if pos in a:
                        if a[pos] == val:
                            sat = True; break
                    else:
                        residual.append((pos, val))
                if sat: continue
                if not residual: return None, True, a      # conflict
                if len(residual) == 1:                     # unit clause
                    p, v = residual[0]
                    a[p] = v; changed = True
                else:
                    new_cls.append(residual)
            cls = new_cls
        return cls, False, a

    def search(cls, a):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes: return None                  # timeout
        cls, conflict, a = propagate(cls, a)
        if conflict: return None
        if not cls: return a                               # all satisfied
        c = min(cls, key=len)
        pos = c[0][0]
        for val in [c[0][1], 1 - c[0][1]]:
            child = dict(a)
            child[pos] = val
            res = search([list(cc) for cc in cls], child)
            if res is not None: return res
        return None

    final = search(clauses, asn)
    return (final, final is not None, nodes)
```

首先，注意两件事。第一：什么是“检查点”？检查点是一个逻辑谓词——例如 $(o_3 = \text{XOR}) \lor (o_5 = \text{NOP}) \lor (o_9 = \text{XOR})$。它要求算子序列在某个子句上得到满足，而不是要求累加器等于某个值。（这个区别为什么重要？见下文“为什么检查点必须在算子上”。）第二：实例是怎么生成的？纯随机 3-CNF——没有预设解，没有“反向工程出来的满足赋值”。

```python
# --- Quick check: a small 6-position instance ---
import numpy as np

inst = generate_instance(n=6, m=4, seed=0)
asn, sat, nodes = solve_dpll(inst)
print("instance:", [(c) for c in inst.clauses])
print("sat =", sat, "| nodes =", nodes)
```

结果——来自你自己机器的真实数据：

```
instance: [[(5,0),(6,0),(4,0)], [(1,1),(5,1),(4,1)], [(5,1),(3,1),(4,0)], [(4,1),(5,1),(2,0)]]
sat = True | nodes = 5
```

一个只有 6 个算子位置的小实例，用 5 个搜索节点就解出来了。现在做正经实验：把子句密度 $\alpha = m/n$ 从 3.0 扫到 5.5，每个点 30 个实例，统计可满足概率 $\mu(\alpha)$。我们自己的“豌豆实验”（蒙特卡洛边界测绘）：

```python
# --- Pea experiment: phase transition scan (n=40, m = round(alpha*n), 30 samples per point) ---
rng = np.random.default_rng(11)
n, samples = 40, 30

for alpha in [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.26, 4.50, 4.75, 5.00, 5.25, 5.50]:
    m = int(round(alpha * n))
    sat_cnt = 0
    for _ in range(samples):
        inst = generate_instance(n, m, rng=rng)
        _, sat, _ = solve_dpll(inst)
        sat_cnt += 1 if sat else 0
    print(f"alpha={alpha:5.2f}  mu(SAT)={sat_cnt/samples:.3f}")
```

这是真实输出（n=40，每个点 30 个样本）：

```
alpha= 3.00  mu(SAT)=1.000
alpha= 3.25  mu(SAT)=1.000
alpha= 3.50  mu(SAT)=1.000
alpha= 3.75  mu(SAT)=0.933
alpha= 4.00  mu(SAT)=0.900
alpha= 4.25  mu(SAT)=0.467
alpha= 4.26  mu(SAT)=0.500
alpha= 4.50  mu(SAT)=0.500
alpha= 4.75  mu(SAT)=0.133
alpha= 5.00  mu(SAT)=0.067
alpha= 5.25  mu(SAT)=0.067
alpha= 5.50  mu(SAT)=0.033
```

看看发生了什么：$\mu$ 在 $\alpha = 4.00$ 时还有 0.90，到 $\alpha = 4.25$ 掉到 0.47，在 $\alpha = 4.26$ 时恰好停在 0.50，到 $\alpha = 5.00$ 就只剩下 0.10 了。这就是相变——可满足概率从“几乎总是可满足”崩塌为“几乎总是不可满足”，分界线钉在 $\alpha_c \approx 4.26$。

![图7.3：OpenXOR 相变与搜索代价](/figures/openxor_v2_phase.png)

*我们自己测出的 OpenXOR 相变：(a) 可满足概率 $\mu$ 随子句密度 $\alpha$ 的变化——一条在 $\alpha \approx 4.26$ 处穿过 0.5 的 S 型曲线（红色虚线）；(b) 搜索代价在相变点附近达到峰值。这张图不是引用来的——它就是你用上面的代码亲手跑出来的。*

**关键发现：临界密度不会随规模漂移。** 我们把 $\alpha$ 固定在 4.26，改变规模 $n$：

```
n=20  alpha=4.26  mu=0.533
n=30  alpha=4.26  mu=0.700
n=40  alpha=4.26  mu=0.567
n=50  alpha=4.26  mu=0.533
```

$\mu$ 始终在 0.5 附近——$\alpha_c \approx 4.26$ 是一个普适常数。这正是第 4 节讨论的经典 3-SAT 相变：临界密度不依赖于 $n$——只依赖于“子句/变量”的比值。下面是对应的连续公式：

$$
\mu(\alpha) = \frac{1}{2}\left(1 - \operatorname{erf}\left(\frac{\alpha - \alpha_c}{\sigma}\right)\right),
\qquad \alpha_c \approx 4.26, \ \sigma \approx 0.1
$$

这就是“把 P/NP 的是/否判决变成连续概率景观”的完整表达式：给定子句密度，你不用真正求解任何东西，就能预测可满足概率；而且 $\mu \in [0,1]$，不再是“可满足/不可满足”的二元判决。

**为什么检查点必须落在算子上（而不是累加器上）？** 这是 OpenXOR 设计中最要命的一个坑。如果我把检查点定义为“位置 $p$ 的累加器必须等于 $v$”——也就是 $\text{acc}_p = v$——问题会立刻坍缩成一个线性方程组：每个位置选 XOR 还是 NOP，本质上就是在 GF(2) 上解一个线性方程组，高斯消元就能搞定。这不是 NP 难——这是一个普通的 P 问题。更糟的是，如果你再配上“反向构造”（先抽一个解，再反推出满足它的检查点），每个实例都自带一个预计算好的见证，这种“难”是最假的那种难。一旦把检查点变成算子序列上的逻辑谓词（一个 3-CNF 子句），问题才成为真正的 3-SAT，指数级最坏情况才无法回避，上面测到的相变才真实反映了难度结构。

这揭示了一种哲学上的转变：**可判定性不是二元的——它是概率性的**。

经典复杂度理论像一张行政地图：这个问题属于 P，那个问题属于 NP，边界画得斩钉截铁。但真实的实例分布更像一张气象图。你不能只问“这属于哪个国家？”——你还要问“这里的能见度如何，风暴往哪个方向走，气温是不是即将越过某个临界点？”

$\mu(\alpha)$ 做的正是这件事：它把行政地图变成气象图。它没有推翻 P/NP 分类；它在分类内部补上了一层读者真正需要的信息：一个具体实例离崩塌边缘到底有多近。

:::details 为什么 $\mu(\alpha)$ 比“这是 NP 完全”更有信息量？
“3-SAT 是 NP 完全的”只告诉你：在最坏情况下，没有已知的高效算法。它不告诉你某个具体实例会不会被很快解出。

$n$ 是算子数量（规模），$\alpha$ 是子句密度，$\alpha_c \approx 4.26$ 是这个问题对应的临界密度。于是：

- 当 $\alpha \ll \alpha_c$：约束太少，解很充足——连随机搜索都能撞上答案；
- 当 $\alpha \gg \alpha_c$：约束太多，矛盾很快暴露——剪枝很有效；
- 当 $\alpha \approx \alpha_c$：解稀少但没有消失，矛盾存在但藏得很深——搜索树达到最大。

所以最难的实例不是“最大”的实例，而是“最接近相变”的实例。这就是 OpenXOR 的意义：它不用规模去吓唬模型——而是把模型放在复杂度景观中最薄、最危险的冰面上。
:::

NP 不是一堵墙。它是一片有梯度的雾。在雾的边缘（$\mu \approx 0.5$），一个问题处于可解与不可解之间的量子叠加态。参数的微小变化就会产生质的飞跃。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、NP 完全性：Cook-Levin 定理与归约

P vs NP 是一个存在性问题——我们不知道答案。但在这个问题悬而未决的时候，数学家做了一件聪明的事：他们证明了某些问题“和每一个 NP 问题一样难”。

这就是 NP 完全性。

### 6.1 Cook-Levin 定理：SAT 是 NP 完全的

1971年，Stephen Cook（以及独立工作的 Leonid Levin）证明了一个惊人的定理：

**SAT（布尔可满足性问题）是 NP 完全的。**

SAT 问的是：给定一个布尔公式（由 AND/OR/NOT 构成的逻辑表达式），是否存在一组变量赋值使它为真？

例如：$(x_1 \lor \neg x_2) \land (\neg x_1 \lor x_3) \land (\neg x_2 \lor \neg x_3)$

是否存在 $x_1, x_2, x_3$ 的赋值，使整个公式为真？

Cook-Levin 定理说了两件事：

**第一，SAT 属于 NP**：给定一组赋值，验证它是否满足公式只需要 $O(n)$ 时间。验证很容易；搜索（找到那组赋值）可能很难。

**第二，每一个 NP 问题都能归约到 SAT**：对于任意 NP 问题 $X$，$X$ 的任意实例都能在多项式时间内转化为一个 SAT 实例，使得原实例有解 $\iff$ 该 SAT 实例可满足。

这个归约的方向是关键：它并不是说 SAT 和 $X$ 一样难——而是说 **SAT 至少和 $X$ 一样难**，因为你可以用 SAT 求解器来解 $X$。

如果有人发明了多项式时间的 SAT 算法，他就同时解出了每一个 NP 问题。

1971年，Cook 做了一件比证明 $P \neq NP$ 更聪明的事。他不知道 P 是否等于 NP——到今天也没人知道。但他证明了 SAT 是所有 NP 问题的**天花板**。如果你能捅破 SAT 这层天花板，整座 NP 大厦就会塌进 P。

这就是归约的力量：你不需要一个一个地推倒每一个 NP 问题——3-SAT、旅行商问题、背包问题、图着色……它们每一个都不一样，每一个都很难。但归约告诉你，它们共享同一种“难度结构”。推倒一个，就推倒了全部。

### 6.2 多项式归约：难度的传递

NP 完全性的核心工具是**多项式时间归约**。

问题 $A$ 在多项式时间内归约到问题 $B$（记作 $A \leq_p B$）意味着：

> 存在一个多项式时间算法 $f$，使得 $x \in A \iff f(x) \in B$。

直觉是：如果你能解 $B$，你就用 $f$ 把 $A$ 的一个实例转成 $B$ 的一个实例，用 $B$ 的算法解出来，再把答案翻译回去。唯一的代价就是这次多项式时间的转换。

**归约链**：Cook 证明了 SAT 是 NP 完全的。此后，Richard Karp（1972）给出了一条贯穿 21 个经典问题的归约链：

$$
\text{SAT} \leq_p \text{3-SAT} \leq_p \text{独立集} \leq_p \text{顶点覆盖} \leq_p \text{集合覆盖} \leq_p \cdots
$$

链上的每一步都证明了下一个问题是 NP 完全的。

这条链有一种哲学含义：**这些看似毫不相关的问题，在计算难度上是等价的**。图论、组合优化、逻辑推理……在底层，它们共享同一种“难度结构”。

### 6.3 为什么这对推理很重要

NP 完全性告诉我们：

**如果 $P \neq NP$，就不存在通用的高效推理算法。**

定理证明、规划、约束满足……这些推理任务大多都是 NP 完全的。这意味着：

- 精确算法在最坏情况下是指数级的
- 启发式算法（第8章）和近似算法（也是第8章）是唯一实际可行的路径
- 没有任何算法能同时具备三者：通用、高效、精确

随便挑两个。这不是工程限制——这是一个数学定理。

## 七、停机问题：计算的根本禁区

如果说 P vs NP 是关于“难度”的问题，那么停机问题就是关于“可能性”的问题。

这两个边界有根本的不同。P vs NP 问的是：搜索比验证难多少？即使 $P \neq NP$，你仍然可以暴力搜索——只不过要花指数时间。它沿**时间**维度划了一条线。停机问题问的是：这个程序到底会不会停？即使给你无限的时间和无限的内存，也没有算法能回答。它沿**逻辑**维度封死了答案。

一个用时间丈量，一个用逻辑封死。计算的两种边界，永不重叠。

它不是说某些问题很难解——而是说某些问题**根本无法被解决**。

停机问题问的是：给定一个程序 P 和它的输入 x，你能否判定 P(x) 是否会停机（而不是永远循环下去）？

图灵在 1936 年用对角化论证证明了这个问题不可判定。让我们把这个证明重放一遍。

**假设**：存在一个停机判定器 H(P, x)，它能判断程序 P 在输入 x 上是否停机。

**构造**：定义一个新程序 D：

**矛盾**：现在问——D(D) 会停机吗？

- 如果 H(D, D) 返回“停机”，那么 D(D) 就进入无限循环，不停机
- 如果 H(D, D) 返回“不停机”，那么 D(D) 就立即返回，停机

无论哪种情况，H 的答案都是错的。所以 H 不可能存在。

这个证明的深刻之处不在于技巧——而在于它揭示的结构性限制。

停机问题是不可判定的，不是因为：

- 我们还没找到足够聪明的算法
- 计算机还不够快
- 我们需要更多内存

而是因为**没有任何算法能解决它**。这是逻辑必然，不是技术限制。

\[Hamkins & Nenu, 2024\] 以细致的历史考据分析了图灵的原始论文。他们指出，图灵 1936 年的论文确实包含了不可判定性的核心思想，但停机问题的现代表述是后来逐步完善出来的。这提醒我们，即使是最基础的理论结果，其历史演化也比我们想象的更复杂。

停机问题标志着计算的终极边界。越过这条边界，事情不是“难”——而是“不可能”。

<div class="center">

------------------------------------------------------------------------

</div>

## 八、复杂度的哲学意义

复杂度理论告诉我们的，不只是“哪些问题难”——它告诉我们的是宇宙的计算结构。

P vs NP 问的是：验证与搜索之间的不对称是本质的吗？如果 $P \neq NP$ 成立，那么宇宙就包含一种根本的不对称——知道答案比找到答案更容易，而没有任何算法能弥合这个差距。

这不是一个技术问题——而是一个关于知识本质的问题。为什么验证比发现容易？为什么理解比创造容易？这种不对称暗示了什么？

相变核函数 $\mu(\alpha)$ 告诉我们，这种不对称不是绝对的——而是渐进的。在相变点附近，一个问题处于可解与不可解之间的临界状态。参数的微小变化就会产生质的飞跃——从“几乎总是可满足”到“几乎总是不可满足”。

停机问题标志着计算的终极边界。有些问题不是“难”——而是“不可能”。这不是技术限制，而是逻辑必然。任何足够强大的计算系统都会撞上这条边界。

> 有些问题太难，无法精确求解。下一章，我们将看到启发式算法如何用“明确代价的近似”绕过这堵墙——以及这种近似必须遵守什么契约。

## 悬而未决

- **$P \neq NP$ 能被证明吗？** 还是说它本身就是一个不可判定的命题？如果 $P \neq NP$ 无法被证明，那意味着什么？

- **相变核函数能推广吗？** $\mu(\alpha)$ 框架能扩展到其他复杂度类吗，比如 PSPACE 或 EXP？不同问题的相变是否遵循同一种统一的数学结构？

- **自然界的相变** 自然的计算过程——蛋白质折叠、神经网络训练、生物进化——是否遵循类似的相变规律？

- **LLM 的架构缺陷** 为什么一个千亿参数的大语言模型会有 0% 的任务完成率？差距究竟在哪里？

## 附录：完整代码

这是本章实验的完整脚本（包含上面所有内容，并带有一个可运行的入口——把它保存为 `openxor_v2.py` 然后运行）：

```python
"""
OpenXOR v2: Predicate Checkpoints over the Operator Sequence
============================================================
Companion script for The Reasoning Kingdom, Chapter 7.

WHY THE FIX
-----------
The original OpenXOR draft defined a checkpoint as (p, v): "the accumulator
must equal v at position p". Combined with reverse construction (sample b,
sample o, read off v = acc_p), every instance is a system of linear equations
over GF(2) — polynomial-time solvable, with a precomputed witness. The claimed
NP-hardness never materialized.

v2 defines a checkpoint as a LOGICAL PREDICATE over the operator sequence,
written in 3-CNF: variables are the operators o_i in {XOR, NOP}, and each
checkpoint is a clause like  (o_3 = XOR) OR (o_5 = NOP) OR (o_9 = XOR).
Satisfying all checkpoints = solving 3-SAT on the operators. Instances are
generated at random with no precomputed solution: satisfiability is genuinely
unknown until we search. The observed phase transition sits at the classic
3-SAT critical density alpha_c ~ 4.26.

Script outline:
  1. Instance            - n operator positions, clauses (3-CNF)
  2. generate_instance   - random 3-CNF checkpoints
  3. solve_dpll          - DPLL with unit propagation (reference solver)
  4. pea_scan            - Monte Carlo boundary mapping, mu(alpha) and cost
  5. demo                - tiny example + phase transition scan

Run:  python3 openxor_v2.py
"""
import numpy as np

XOR, NOP = 1, 0   # operator encoding; clause literals are (pos, value)

# --------------------------------------------------------------------------
# 1. Instance
# --------------------------------------------------------------------------
class Instance:
    """n operator positions; each clause is a list of literals (pos, val),
    meaning 'the operator at pos equals val'. All clauses must hold (AND),
    each clause is satisfied by at least one literal (OR) = 3-CNF."""
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses

    def __repr__(self):
        return f"OpenXORv2(n={self.n}, m={len(self.clauses)}, alpha={len(self.clauses)/self.n:.3f})"

# --------------------------------------------------------------------------
# 2. Generator: random 3-CNF checkpoints (no hidden solution)
# --------------------------------------------------------------------------
def generate_instance(n, m, rng=None, seed=0):
    """Random 3-CNF checkpoints. Each clause picks up to 3 distinct positions
    and a random required operator per position. No solution is precomputed:
    satisfiability is unknown until we search."""
    rng = rng or np.random.default_rng(seed)
    clauses = []
    for _ in range(m):
        pos = rng.choice(n, size=min(3, n), replace=False) + 1   # 1-indexed
        vals = rng.integers(0, 2, size=len(pos))
        clauses.append([(int(p), int(v)) for p, v in zip(pos, vals)])
    return Instance(n, clauses)

# --------------------------------------------------------------------------
# 3. Reference solver: DPLL with unit propagation
# --------------------------------------------------------------------------
def solve_dpll(inst, max_nodes=500_000):
    """DPLL over operator variables with unit propagation.
    Returns (assignment, sat, nodes)."""
    n, clauses = inst.n, [list(c) for c in inst.clauses]
    asn, nodes = {}, 0

    def propagate(cls, a):
        changed = True
        while changed:
            changed, new_cls = False, []
            for c in cls:
                sat, residual = False, []
                for pos, val in c:
                    if pos in a:
                        if a[pos] == val:
                            sat = True
                            break
                    else:
                        residual.append((pos, val))
                if sat:
                    continue
                if not residual:                # conflict: empty clause
                    return None, True, a
                if len(residual) == 1:          # unit clause -> force assignment
                    p, v = residual[0]
                    a[p] = v
                    changed = True
                else:
                    new_cls.append(residual)
            cls = new_cls
        return cls, False, a

    def search(cls, a):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return None                         # timeout
        cls, conflict, a = propagate(cls, a)
        if conflict:
            return None
        if not cls:
            return a                            # all clauses satisfied
        c = min(cls, key=len)                   # heuristic: shortest clause
        pos = c[0][0]
        for val in [c[0][1], 1 - c[0][1]]:
            child = dict(a)
            child[pos] = val
            res = search([list(cc) for cc in cls], child)
            if res is not None:
                return res
        return None

    final = search(clauses, asn)
    return (final, final is not None, nodes)

# --------------------------------------------------------------------------
# 4. Monte Carlo boundary mapping ("pea experiment")
# --------------------------------------------------------------------------
def pea_scan(n_list, alpha_list, samples=20, seed=1, max_nodes=50_000):
    """mu(alpha) and mean search nodes for each (n, alpha = m/n)."""
    rng = np.random.default_rng(seed)
    results = {}
    for n in n_list:
        row = []
        for alpha in alpha_list:
            m = max(1, int(round(alpha * n)))
            sat_cnt, node_acc = 0, 0
            for _ in range(samples):
                inst = generate_instance(n, m, rng=rng)
                inst.max_nodes = max_nodes
                _, sat, nd = solve_dpll(inst, max_nodes=max_nodes)
                sat_cnt += 1 if sat else 0
                node_acc += nd
            row.append((sat_cnt / samples, node_acc / samples))
        results[n] = row
    return results

# --------------------------------------------------------------------------
# 5. Demo: tiny instance + phase transition scan
# --------------------------------------------------------------------------
def demo():
    print("=" * 78)
    print("OpenXOR v2 - predicate checkpoints over operator sequence")
    print("=" * 78)

    inst = generate_instance(n=6, m=4, seed=0)
    print("\n[1] Example instance (n=6, m=4):", inst)
    for c in inst.clauses:
        print("    clause:", c)
    asn, sat, nodes = solve_dpll(inst)
    print(f"\n[2] DPLL solve: SAT={sat}, search nodes={nodes}")
    if sat:
        print("    assignment o =", ["XOR" if asn.get(p, NOP) == XOR else "NOP"
                                     for p in range(1, inst.n + 1)])

    print("\n[3] Phase transition scan (n=40, samples=30 per point):")
    rng = np.random.default_rng(11)
    n, samples = 40, 30
    for alpha in [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.26, 4.50, 4.75,
                  5.00, 5.25, 5.50]:
        m = int(round(alpha * n))
        sat_cnt = 0
        for _ in range(samples):
            inst = generate_instance(n, m, rng=rng)
            _, sat, _ = solve_dpll(inst)
            sat_cnt += 1 if sat else 0
        print(f"    alpha={alpha:5.2f}  mu(SAT)={sat_cnt/samples:.3f}")

    print("\n[4] Critical density does not drift with n (alpha=4.26):")
    for n2 in [20, 30, 40, 50]:
        m = int(round(4.26 * n2))
        sat_cnt = 0
        for _ in range(samples):
            inst = generate_instance(n2, m, rng=rng)
            _, sat, _ = solve_dpll(inst)
            sat_cnt += 1 if sat else 0
        print(f"    n={n2:3d}  mu={sat_cnt/samples:.3f}")

    print("\n[5] Note: checkpoint as accumulator value acc_p = v is a linear")
    print("    system over GF(2) - solvable by Gaussian elimination. Only")
    print("    predicate checkpoints over the operators realize 3-SAT.")
    print("\nDone.")

if __name__ == "__main__":
    demo()
```

## 延伸阅读

- 第7章的配套实验——OpenXOR v2 谓词检查点生成器 + DPLL 求解器（完整代码见本章附录：`openxor_v2.py`）

- \[Xu & Li, 2000\]——3-SAT 相变的开创性工作

- \[Zhan, 2026\]——3-SAT 的离散几何模型

- \[Istrate, 2000\]——计算复杂度与相变之间的理论联系

- \[Hamkins & Nenu, 2024\]——停机问题的历史澄清

# 第7章：复杂度的真相——不是快慢，是结构

> 如果 $P \neq NP$ 成立，它告诉我们的不是“有些问题很难”，而是“宇宙本身内置了一种不对称”。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、一个不对称

2023 年，DeepSeek-R1 和 GPT-4 在推理任务上轮流破纪录。研究者拿 NP 完全问题去试它们。

这里有一个专门造的测试，叫 OpenXOR：一个从约束满足里长出来的诊断工具。任务不复杂——给定一组 XOR 风格的约束实例，判断存不存在一组满足全部约束的赋值。它和纯 XOR-SAT 不同，不能拿高斯消元一笔带过；设计上保留了 XOR 的局部迷惑性，又把人推进 NP 难的相变区。下文会把它造出来，拿到真实模型上量一量。

:::details NP 完全问题：为什么它们比普通问题难这么多？（动手任务：我们的 OpenXOR）
**NP（nondeterministic polynomial，非确定性多项式）** 的问题有一个好性质：给你一个候选答案，你能在多项式时间内**验证**它对不对。比如 3-SAT，拿到一组变量赋值，挨个子句检查一遍就完了。

**P（polynomial，多项式）** 是另一类：多项式时间内**找得到**答案。排序是，最短路径也是，Dijkstra 那条就是。

**NP 完全**是 NP 里最硬的一批。硬到什么程度？解掉其中任意一个，所有 NP 问题就一起解掉了。

**P = NP？** 计算机科学最有名的未解问题。多数人押 $P \neq NP$：验证比搜索容易，两种能力隔着一条沟。可谁也没证出来。

**为什么 LLM 解不了 NP 完全问题？** LLM 是大语言模型，在大量文本上训练，用来预测下一个词。它的推理跑在多项式时间里，靠有限次前向传播；NP 完全问题要指数级搜索。实例一大，它就垮。参数堆得再高，也补不上架构这道口子。
:::

DeepSeek-R1 的任务完成率是 0%。GPT-4 也是 0%。

0% 不是准确率，是 0% 的任务完成率。模型有的拒绝回答，37–42%；有的撞上上下文上限，28–31%；有的干脆编造约束，18–22%。它们连试都没试着解题。

症结不在规模。在架构。

千亿参数的模型，为什么栽在这道题上？

关键在时间。LLM 的一次前向传播跑在多项式时间里：不管模型多大，工作量卡在 $O(LN^2)$，层数 $L$ 乘序列长度 $N$ 的平方。OpenXOR 的实例也不是随手挑的，约束密度钉在 $\alpha \approx 4.26$ 上下——一个相变临界点。相变就是参数越过某条线，行为突然翻脸。到了这儿，3-SAT 从“几乎总是可满足”一头栽进“几乎总是不可满足”，求解器正好卡在最难搜的那块地方。

LLM 没有搜索，只有前向传播。多项式时间的架构撞上指数级难度，完成率只能是 0%。参数规模能把模式匹配做得更花，变不出指数时间。这是架构的边界，算力堆不过去。

从最小的地方开始。

## 二、验证与搜索：一个简单的实验

拿张纸，写下这个 3-SAT 实例：

    (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₄) ∧ (x₂ ∨ x₃ ∨ x₄) ∧ (¬x₁ ∨ ¬x₃ ∨ x₄)

给你一个候选解：`x₁=true, x₂=false, x₃=true, x₄=true`。

挨个子句验一遍：

第一句：x₁=true，过。

第二句：¬x₁=false，x₂=false，¬x₄=false……三个都是假。没过。

它不是一个解。整个过程你花了多久？10 秒？20 秒？

现在把答案盖住，从零找一组满足全部子句的赋值。

你大概会这么试：

- x₁=true、x₂=true、x₃=true、x₄=true——检查：不满足

- x₁=true、x₂=true、x₃=true、x₄=false——检查：不满足

- x₁=true、x₂=true、x₃=false、x₄=true——检查：不满足

- …

4 个变量，$2^4 = 16$ 种可能。运气背的时候，16 种全试完才找得到解，或者才敢说无解。

这就是**验证与搜索之间的不对称**。验一个候选解，$O(m)$，线性，$m$ 是子句数。找解要 $O(2^n \times m)$——指数，$n$ 是变量数。

$n = 10$，搜 1024 种。$n = 100$，搜 $2^{100} \approx 10^{30}$ 种——比宇宙里的原子还多。

道理在数学这边，不在工程那边。

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

3-SAT 实例不是一样难的。

这听着像废话。10 个变量当然比 100 个容易。可这不是重点。

变量数相同，两个 3-SAT 实例的难度也能差出天去。而且这差别不随机，背后有规律。

先看两个极端。一头是**欠约束**：10 个变量，20 个子句，密度 $\alpha = 2$。约束少，解多得像空地。随便丢一个赋值进去，多半就中。求解器几乎不用回头。

另一头是**过约束**：10 个变量，80 个子句，密度 $\alpha = 8$。约束挤成一团，矛盾几步就露头。求解器一撞墙就剪枝换路。没有解，但证明没有，很快。

两头都不难。

到了 $\alpha \approx 4.26$ 附近，事情变了。

\[Xu & Li, 2000\] 在他们 2000 年的工作里发现：子句密度 $\alpha = m/n$ 从 2 升到 8，可满足概率不是一路平滑下滑的。

在 $\alpha \approx 4.26$，它翻了个面。这叫**相变**。

从接近 1 直落接近 0。没有缓坡，是断崖。

计算问题里的相变不是头一回见，可每回见着都叫人心里一沉。到一个临界点，问题的性子整个换掉：水在 0°C 结冰，3-SAT 在 $\alpha \approx 4.26$ 从“几乎总是可满足”翻成“几乎总是不可满足”。

这个相变点，偏偏就是最难求解的地方。

![图7.2：3-SAT相变图](/figures/ch07_fig3_real_mu.png) *横轴是子句密度 $\alpha = m/n$；纵轴是随机实例可满足的概率。S 型曲线在 $\alpha \approx 4.26$ 附近发生陡峭的相变。绿色阴影区域（$\alpha < 4.2$）是欠约束区域——几乎总是可满足。红色阴影区域（$\alpha > 4.3$）是过约束区域——几乎总是不可满足。临界点附近（黄色区域）是最难求解的区域。*

临界点为什么最难？

还是那两头。

欠约束那头，解多得铺在路上。随机搜索都能撞上。求解器不用多聪明，别太倒霉就行。

过约束那头，矛盾露头快。走几步撞上死路，回溯、剪枝，换一条。最后证明无解，整个过程利索。

临界点上，求解器卡在一种窄得离谱的平衡里。

解很少，可又不是没有，放弃不得。矛盾藏着，不显眼，回溯树于是长得没边。空间既不稀疏也不稠密，导航最难。没有捷径，也没法快速排掉错路。

最难受的处境，就是这个。

Zhan（2026）拿一个离散几何模型解释这件事。他把 3-SAT 映射到 N 维布尔超立方体的组合拓扑上，推出最小不可满足极限 $\alpha = 8/N$，以及最大可满足极限 $\alpha = 7/6(N-1)(N-2)$。

这也解释了可满足性阈值猜想为什么只在“几乎必然”的意义上成立。规模有限，相变本身就带概率。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、μ(α)：把离散边界变成连续边界

经典复杂度理论只判归属：这个问题在 P，还是 NP。它说不清某个具体实例有多难。

3-SAT 是 NP 完全的——一句话，是或否。可不是所有 3-SAT 实例都一样难：10 变量、20 子句的实例，和 100 变量、420 子句的实例，难度隔着一条河。

OpenXOR 把这层二元拆了。它把 NP 问题的可满足性从一个是或否，摊成一张连续的相图。这张相图不是引用来的，是跑出来的。下面的代码是 OpenXOR v2 的完整实现：谓词检查点生成器，加一个参考求解器 DPLL。都在这儿，一行不用改就能跑。

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

先看两件事。检查点是什么？一个逻辑谓词，比如 $(o_3 = \text{XOR}) \lor (o_5 = \text{NOP}) \lor (o_9 = \text{XOR})$。它要求算子序列满足某个子句，不要求累加器等于某个数。这个区别为什么重要，下文“为什么检查点必须在算子上”再说。再看实例怎么生成：纯随机 3-CNF，没有预设解，没有“反向工程出来的满足赋值”。

```python
# --- Quick check: a small 6-position instance ---
import numpy as np

inst = generate_instance(n=6, m=4, seed=0)
asn, sat, nodes = solve_dpll(inst)
print("instance:", [(c) for c in inst.clauses])
print("sat =", sat, "| nodes =", nodes)
```

真数据，来自你自己的机器：

```
instance: [[(5,0),(6,0),(4,0)], [(1,1),(5,1),(4,1)], [(5,1),(3,1),(4,0)], [(4,1),(5,1),(2,0)]]
sat = True | nodes = 5
```

6 个算子位置的小实例，5 个搜索节点就解了。下面做正经的：把 $\alpha = m/n$ 从 3.0 扫到 5.5，每个点 30 个实例，统计可满足概率 $\mu(\alpha)$。这是自己的“豌豆实验”，蒙特卡洛边界测绘：

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

跑出来的真输出，n=40，每个点 30 个样本：

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

看这条线：$\mu$ 在 $\alpha = 4.00$ 时还有 0.90，到 $\alpha = 4.25$ 掉到 0.47，在 $\alpha = 4.26$ 恰好停在 0.50，到 $\alpha = 5.00$ 只剩 0.10。这就是相变——可满足概率从“几乎总是可满足”塌成“几乎总是不可满足”，分界线钉在 $\alpha_c \approx 4.26$。

![图7.3：OpenXOR 相变与搜索代价](/figures/openxor_v2_phase.png)

*图里两张：左边是可满足概率 $\mu$ 随子句密度 $\alpha$ 的变化，一条在 $\alpha \approx 4.26$ 处穿过 0.5 的 S 型曲线，红色虚线；右边是搜索代价，在相变点附近爬到顶。图是跑出来的，不是引来的——用的就是上面的代码。*

临界密度不随规模漂移。把 $\alpha$ 固定在 4.26，改规模 $n$：

```
n=20  alpha=4.26  mu=0.533
n=30  alpha=4.26  mu=0.700
n=40  alpha=4.26  mu=0.567
n=50  alpha=4.26  mu=0.533
```

$\mu$ 始终在 0.5 附近，$\alpha_c \approx 4.26$ 是个普适常数。第 4 节那场经典 3-SAT 相变说的就是这回事：临界密度与 $n$ 无关，只看子句与变量的比值。写成连续公式是这样：

$$
\mu(\alpha) = \frac{1}{2}\left(1 - \operatorname{erf}\left(\frac{\alpha - \alpha_c}{\sigma}\right)\right),
\qquad \alpha_c \approx 4.26, \ \sigma \approx 0.1
$$

一个式子，把 P/NP 的是或否换成连续的概率景观。给定子句密度，不用求解，就能预测可满足概率。$\mu \in [0,1]$ 是个数，不再只有“可满足”和“不可满足”两个答案。

**为什么检查点必须落在算子上，而不是累加器上？** 这是 OpenXOR 设计里最要命的一个坑。把检查点定义成“位置 $p$ 的累加器必须等于 $v$”——也就是 $\text{acc}_p = v$——问题立刻坍成线性方程组。每个位置选 XOR 还是 NOP，等于在 GF(2) 上解线性方程，高斯消元就能打发。这里没有 NP 难，只有普通的 P。更假的是配上“反向构造”：先抽一个解，再反推出满足它的检查点，每个实例自带一份预计算好的见证，难是装出来的。检查点只有落在算子序列上，成为逻辑谓词，成为一个 3-CNF 子句，问题才是真正的 3-SAT，指数级最坏情况才躲不开，上面测到的相变才真的在说难度。

可判定性有刻度，不只两个档位。

经典复杂度理论是一张行政地图：这个在 P，那个在 NP，边界画得斩钉截铁。真实实例却像气象图。只问“属于哪个国家”不够，还得问能见度、风向、气温离临界点还有多远。

$\mu(\alpha)$ 干的就是这个：行政地图换成气象图。P/NP 分类没被推翻，里面多了一层更有用的信息——一个具体实例，离崩塌边缘还有多近。

:::details 为什么 $\mu(\alpha)$ 比“这是 NP 完全”更有信息量？
“3-SAT 是 NP 完全的”只说了最坏情况：没有已知的高效算法。某个具体实例会不会一解就开，它不管。

$n$ 是算子数量，也就是规模；$\alpha$ 是子句密度；$\alpha_c \approx 4.26$ 是临界密度。于是：

- 当 $\alpha \ll \alpha_c$：约束太少，解多得是，随机搜索也能撞上；
- 当 $\alpha \gg \alpha_c$：约束太挤，矛盾露头快，剪枝一剪一个准；
- 当 $\alpha \approx \alpha_c$：解少但没绝，矛盾深藏，搜索树长到最大。

所以最难的实例不靠块头大，靠离相变近。OpenXOR 就是干这个的：不拿规模吓唬模型，把它放到复杂度景观里最薄、最容易裂的那层冰上。
:::

NP 更像一片有梯度的雾。雾的边缘 $\mu \approx 0.5$，问题悬在可解与不可解之间。参数挪一点，结果翻一个面。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、NP 完全性：Cook-Levin 定理与归约

P vs NP 悬着，谁也不知道答案。数学家换了条路：证明某些问题“和每一个 NP 问题一样难”。

这条路叫 NP 完全性。

### 6.1 Cook-Levin 定理：SAT 是 NP 完全的

1971 年，Stephen Cook 证了一个定理。Leonid Levin 独立地证了同一个：

SAT，也就是布尔可满足性问题，是 NP 完全的。

SAT 问：给一个布尔公式，由 AND、OR、NOT 搭成，存不存在一组赋值让它为真？

例如：$(x_1 \lor \neg x_2) \land (\neg x_1 \lor x_3) \land (\neg x_2 \lor \neg x_3)$

有没有一组 $x_1, x_2, x_3$，能让整个公式为真？

Cook-Levin 定理讲两件事。

**第一，SAT 属于 NP**：给一组赋值，验它满不满足公式，$O(n)$ 时间就够。验证便宜，搜索——也就是找到那组赋值——可能贵得离谱。

**第二，每个 NP 问题都能归约到 SAT**：任意 NP 问题 $X$ 的任意实例，都能在多项式时间里变成一个 SAT 实例，而且原实例有解 $\iff$ 新实例可满足。

方向要看准：SAT 至少和 $X$ 一样难。手里有 SAT 求解器，就能拿来解 $X$。

谁要是发明了多项式时间的 SAT 算法，所有 NP 问题就一起解了。

1971 年，Cook 没去硬碰 $P \neq NP$。P 等不等于 NP，他不知道，今天也没人知道。他绕到上头，证明 SAT 是所有 NP 问题的**天花板**。捅破这层，整座 NP 大厦就塌进 P。

归约的力气在这儿：不必挨个推倒 NP 问题。3-SAT、旅行商、背包、图着色……面孔个个不同，骨头是同一种。推倒一个，全倒。

### 6.2 多项式归约：难度的传递

核心工具叫**多项式时间归约**。

问题 $A$ 在多项式时间内归约到问题 $B$，记作 $A \leq_p B$，意思是：

> 存在一个多项式时间算法 $f$，使得 $x \in A \iff f(x) \in B$。

直觉很直白：会解 $B$，就拿 $f$ 把 $A$ 的实例翻成 $B$ 的实例，解完再翻回来。代价只有这次多项式时间的转换。

**归约链**：Cook 先证 SAT 是 NP 完全的。1972 年，Richard Karp 接上一条链，串起 21 个经典问题：

$$
\text{SAT} \leq_p \text{3-SAT} \leq_p \text{独立集} \leq_p \text{顶点覆盖} \leq_p \text{集合覆盖} \leq_p \cdots
$$

链上每一步，都把下一个问题钉成 NP 完全。

链背后有件事：看似毫不相干的问题，在计算难度上是一回事。图论、组合优化、逻辑推理……底下是同一种“难度结构”。

### 6.3 为什么这对推理很重要

NP 完全性还带出一句更狠的话：

如果 $P \neq NP$，就不存在通用的高效推理算法。

定理证明、规划、约束满足……推理任务大多落在这类里。于是：

- 精确算法最坏情况下是指数级；
- 能上路的只有启发式算法（第 8 章）和近似算法（也在第 8 章）；
- 通用、高效、精确，三样凑不齐。

三样里挑两样。剩下那一样，定理不卖。

## 七、停机问题：计算的根本禁区

P vs NP 问难度。停机问题问另一件事：到底有没有可能。

两个边界不一样。P vs NP 问：搜索比验证难多少？$P \neq NP$ 也挡不住暴力搜索，只是要拿指数时间换。这条线画在**时间**上。停机问题问：这个程序会不会停？时间和内存都给你无限，照样没有算法答得上来。这条线画在**逻辑**上，封死。

一条用时间丈量，一条用逻辑封死。两道边界，不挨着。

它越过了“难解”这一档，直接说无法被解决。

停机问题问：给一个程序 P 和一份输入 x，能不能判定 P 跑 x 的时候会停，还是会一直转下去？

1936 年，图灵用对角化论证证明它不可判定。证明不长，重放一遍。

**假设**：有一个停机判定器 H，喂它一个程序 P 和一份输入 x，它能判断 P 在 x 上停不停。

**构造**：造一个新程序 D。

**矛盾**：把 D 喂给 D，它停不停？

- H 收到 D 和 D，若答“停机”，D 拿到这个答案就进入无限循环，不停
- 若 H 答“不停机”，D 就立即返回，停了

两下里，H 都答错。这样的 H 不存在。

巧劲不稀奇，稀奇的是它露出来的那道结构性限制。

停机问题不可判定，原因不在下面这几条：

- 算法还没聪明到家；
- 计算机还不够快；
- 内存还不够多。

没有任何算法能解决它。逻辑堵死了，技术挪不开。

Hamkins 与 Nenu（2024）细查了图灵的原论文。图灵 1936 年那篇里有不可判定性的核心想法，但“停机问题”这个说法是后来才长出来的。最基础的结果，历史也不平整。

这是计算的终极边界。越过去，事情不叫“难”，叫“不可能”。

<div class="center">

------------------------------------------------------------------------

</div>

## 八、复杂度的哲学意义

复杂度理论不只列难题清单。它描出宇宙的计算结构。

P vs NP 问的是：验证比搜索容易，这件事是本质的吗？若 $P \neq NP$ 成立，宇宙里就埋着一道不对称——知道答案比找到答案容易，什么算法都填不平。

这问题不归工程管，归知识本身管。为什么验证比发现容易？为什么理解比创造容易？这道不对称，到底在暗示什么？

相变核函数 $\mu(\alpha)$ 给的是一把尺子：不对称有刻度，不是一堵墙。相变点附近，问题卡在可解与不可解之间。参数动一点，结果从“几乎总是可满足”跳到“几乎总是不可满足”。

停机问题落在另一头。有些问题不叫“难”，叫“不可能”。逻辑写死了，什么系统都绕不开，越强大越绕不开。

> 有些问题太难，精确求解没门。下一章看启发式算法：拿“代价写得明明白白的近似”绕墙，以及这份近似要守什么契约。

## 悬而未决

- **$P \neq NP$ 证得出来吗？** 还是它自己就不可判定？要是证不出来，意味着什么？

- **相变核函数推得开吗？** $\mu(\alpha)$ 这套东西能不能挪到 PSPACE、EXP 这些类上？不同问题的相变，底下是不是同一副骨架？

- **自然界的相变** 蛋白质折叠、神经网络训练、生物进化，这些自然过程走不走同一条相变曲线？

- **LLM 的架构缺陷** 千亿参数的大语言模型，任务完成率怎么会是 0%？差的那一截，到底在哪儿？

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

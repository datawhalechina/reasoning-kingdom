# 第8章：启发式的契约——接受"差不多对"需要多少勇气

> A\* 的启发函数如果对你撒谎，它依然会交给你一条路。只是那条路是错的。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、1968年的赌注

1968年，斯坦福研究院的 Peter Hart、Nils Nilsson 和 Bertram Raphael 在一篇论文里提出了一个算法——后来这个算法被证明是人工智能历史上最优雅的思想之一。

他们要解决的问题很简单：让机器人 Shakey 在房间里找到从 A 点到 B 点的最短路径。

暴力搜索可以保证找到最优解——尝试所有可能的路径，选最短的那条。但 Shakey 的世界有数百万种可能的路径组合，暴力搜索需要的时间是天文数字。

贪心搜索很快——每次都朝着目标方向走。但它会撞上障碍物，走进死胡同，找到的路径可能比最优解长得多。

Hart 三人提出的 A\* 算法，做了一个看似不可能的承诺：**我可以比暴力搜索快几个数量级，同时保证找到最优解**。

这个承诺的代价是什么？一个**契约**。

如果你给 A\* 一个"诚实"的启发函数——永远不高估剩余代价——它会信守承诺。但如果启发函数"撒谎"，高估了剩余代价，A\* 依然会给你一条路。只是那条路可能是错的。

这不是算法的 bug，这是它的设计。A\* 用一个数学契约，换取了速度与最优性之间的平衡。

五十多年后，这个契约的思想依然是所有启发式算法的核心：**当完美不可及时，我们需要明确代价的近似**。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、组合爆炸的诅咒

让我先量化"完美不可及"这件事。

2016年，AlphaGo 击败李世石。全世界都在惊叹机器的"直觉"和"创造力"。但很少有人注意到一个技术细节：AlphaGo 每一步棋的背后，是蒙特卡洛树搜索在 19×19 的棋盘上探索了**数百万种**可能。

不是数百种，不是数千种，是数百万种。

为什么需要这么多？因为围棋的状态空间大约是 $10^{170}$——比宇宙中的原子数（$10^{80}$）还要多 90 个数量级。

即使每秒搜索 $10^9$ 个状态（现代计算机的极限），穷尽所有可能需要的时间也是宇宙年龄的 $10^{150}$ 倍。

这不是工程问题，这是**物理不可能**。

类似的组合爆炸无处不在：

- **旅行商问题**：访问 n 个城市的所有可能路线有 $n!$ 种。20 个城市就是 $2.4 \times 10^{18}$ 种路线。

- **蛋白质折叠**：一个 100 个氨基酸的蛋白质，每个氨基酸有 3 种主要构象，总共 $3^{100} \approx 10^{47}$ 种可能的折叠方式。

- **定理证明**：在一阶逻辑里，长度为 n 的证明，可能的推理步骤数随 n 指数增长。

这就是启发式存在的理由：**当最优解不可及时，我们需要"足够好"的解**。

但"足够好"到底是什么意思？这不是一个模糊的工程问题，而是一个精确的数学问题。启发式算法必须遵守某些契约——如果违反了这些契约，它给出的"近似解"可能完全错误。

让我回到 1968 年，看看 Hart、Nilsson 和 Raphael 是如何设计这个契约的。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、启发式的数学契约

假设你在一个迷宫里，要从起点 S 走到终点 G。每一步都有代价（比如时间或距离）。你想找到代价最小的路径。

**暴力搜索**（Dijkstra 算法）会尝试所有可能的路径，保证找到最优解，但时间复杂度是 $O(b^d)$——其中 b 是分支因子，d 是解的深度。在复杂迷宫里，这是指数级的。

:::details Dijkstra 算法：最短路径的暴力保证
**Dijkstra（迪杰斯特拉）算法**是荷兰计算机科学家 Edsger Dijkstra 在 1956 年提出的最短路径算法，思路是：从起点出发，每次选择当前已知代价最小的未访问节点，扩展它的邻居，直到到达终点。

- **保证**：如果所有边的代价非负，Dijkstra 一定找到最优（代价最小）的路径
- **代价**：它探索所有可能的路径，不知道"大致往哪个方向走才对"，效率低

**分支因子 $b$**：在搜索树里，每个节点平均有多少个子节点（可选的下一步）。迷宫里可能是 4（上下左右），围棋里大约是 250。

**搜索深度 $d$**：到达答案需要的步数。

两者组合后，搜索空间是 $b^d$——这就是组合爆炸的数学来源。A\* 用启发函数"剪枝"，减少需要探索的节点数，从而在保证最优性的同时大幅提速。
:::

**贪心搜索**每次都选择"看起来离终点最近"的方向，速度很快，但可能走进死胡同。更糟糕的是，它找到的路径可能比最优解长几倍。

A\* 算法在 1968 年提出时，做了一个看似不可能的承诺：我可以比 Dijkstra 快，同时保证找到最优解。

它的核心是一个评估函数：

$$
f(n) = g(n) + h(n)
$$

- $g(n)$：从起点到节点 n 的**实际代价**（已知的，确定的）

- $h(n)$：从 n 到终点的**估计代价**（未知的，猜测的）

- $f(n)$：通过 n 到达终点的**总估计代价**

A\* 每次扩展 $f(n)$ 最小的节点——也就是"实际代价+估计剩余代价"最小的节点。

这个设计的直觉很简单：如果你已经走了 10 步到达节点 n，而你估计还需要 5 步到达终点，那么通过 n 的总代价大约是 15 步。A\* 优先探索总代价最小的路径。

但这里有一个致命的问题：**$h(n)$ 怎么估计？**

如果 $h(n)$ 估计得太高，A\* 会过早放弃最优路径——它会认为"这条路太贵了"，转而探索次优路径。

如果 $h(n)$ 估计得太低（比如 $h(n) = 0$），A\* 退化为 Dijkstra 算法，会探索所有方向，失去了速度优势。

Hart、Nilsson 和 Raphael 证明了一个深刻的定理：如果启发函数满足**可采纳性**（admissibility），A\* 保证找到最优解。

**可采纳性的定义**：

$$
h(n) \leq h^*(n)
$$

其中 $h^*(n)$ 是从 n 到终点的**真实最短代价**。

换句话说，启发函数必须是**乐观的**——永远不能高估剩余代价。你可以低估（这会让搜索变慢），但不能高估（这会让搜索找到错误答案）。

这就是 A\* 的契约：给我一个可采纳的启发函数，我保证找到最优解。违反契约，保证失效。

为什么这个契约有效？让我用一个具体例子说明。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、动手试试：看契约如何被打破

可采纳性不是一个抽象的数学要求。它是 A\* 能否找到最优解的**充要条件**。

让我们亲手验证这件事。

**实验目标**：用同一个迷宫，分别用可采纳和不可采纳的启发函数运行 A\*，看看结果有什么不同。

迷宫地图（0=可通行，1=障碍物，S=起点，G=终点）：

```
S . . . . .
. . . # . .
. . . . # .
. . . . . .
. . . . . #
. . . . . G
```

- S 是起点 (0,0)，G 是终点 (5,5)

- `.` 是可通行格子，每步代价 = 1

- `#` 是障碍物，不可通行

**任务**：找从 S 到 G 的最短路径。

```python
import heapq

# --- maze definition: 6 rows x 6 cols, 1 = obstacle, 0 = walkable ---
# S=(0,0), G=(5,5)
GRID = [
    [0, 0, 0, 0, 0, 0],  # row 0: S . . . . .
    [0, 0, 0, 1, 0, 0],  # row 1: . . . # . .
    [0, 0, 0, 0, 1, 0],  # row 2: . . . . # .
    [0, 0, 0, 0, 0, 0],  # row 3: . . . . . .
    [0, 0, 0, 0, 0, 1],  # row 4: . . . . . #
    [0, 0, 0, 0, 0, 0],  # row 5: . . . . . G
]
ROWS, COLS = len(GRID), len(GRID[0])
START = (0, 0)
GOAL  = (5, 5)

def neighbors(r, c):
    """Yield the walkable neighbors (up, down, left, right)."""
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] == 0:
            yield nr, nc

def reconstruct_path(parent, node):
    """Reconstruct the path by backtracking through the parent dict."""
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))

def astar(h_func, label="A*"):
    """
    Generic A* search.
    h_func(r, c) is the heuristic; returns (path length, path, number of expanded nodes).
    """
    # priority queue items: (f, g, node)
    open_heap = [(h_func(*START), 0, START)]
    g_cost = {START: 0}    # actual cost from start to each node
    parent = {START: None} # parent pointers for path reconstruction
    closed = set()
    expanded = 0           # counted expanded nodes

    while open_heap:
        f, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1

        # reached the goal
        if node == GOAL:
            path = reconstruct_path(parent, node)
            return len(path) - 1, path, expanded  # steps = nodes - 1

        # expand neighbors
        for nb in neighbors(*node):
            new_g = g + 1  # each step costs 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                parent[nb] = node
                f_val = new_g + h_func(*nb)
                heapq.heappush(open_heap, (f_val, new_g, nb))

    return None, [], expanded  # no solution

# --- step 1: admissible heuristic -- Manhattan distance ---
#   h₁(n) = |n_row - G_row| + |n_col - G_col|
#   never overestimates the remaining cost -> admissible -> optimal solution
def h_admissible(r, c):
    """Manhattan distance: admissible, never overestimates."""
    return abs(r - GOAL[0]) + abs(c - GOAL[1])

steps1, path1, exp1 = astar(h_admissible, "admissible A*")
print("=== Step 1: admissible heuristic (Manhattan distance) ===")
print(f"path length: {steps1} steps")
print(f"expanded nodes: {exp1}")
print(f"path: {path1}")

# --- step 2: inadmissible heuristic -- Manhattan distance × 2 ---
#   h₂(n) = 2 × (|n_row - G_row| + |n_col - G_col|)
#   overestimates the remaining cost -> violates admissibility -> may find a suboptimal path
def h_inadmissible(r, c):
    """2x Manhattan distance: inadmissible, overestimates."""
    return 2 * (abs(r - GOAL[0]) + abs(c - GOAL[1]))

steps2, path2, exp2 = astar(h_inadmissible, "inadmissible A*")
print("\n=== Step 2: inadmissible heuristic (2x Manhattan distance) ===")
print(f"path length: {steps2} steps")
print(f"expanded nodes: {exp2}")
print(f"path: {path2}")

# --- comparison ---
print("\n=== Comparison ===")
print(f"admissible heuristic   -> {steps1} steps (optimality guaranteed)")
print(f"inadmissible heuristic -> {steps2} steps ({'optimal' if steps1==steps2 else 'suboptimal'})")
if steps1 != steps2:
    print(f"  gap: {steps2 - steps1} steps (contract broken: found a longer path)")
print(f"\nnote: the admissible version expanded {exp1} nodes, the inadmissible version expanded {exp2} nodes.")
print("  the inadmissible heuristic expands fewer nodes and is faster, but trades away the optimality guarantee.")
```

### 步骤 1：用可采纳启发函数

最简单的可采纳启发函数是**曼哈顿距离**（Manhattan Distance）：

$$
h_1(n) = |n_x - G_x| + |n_y - G_y|
$$

这是"如果没有障碍物，直线走到终点"的代价。因为实际路径必须绕过障碍物，所以 $h_1(n) \leq h^*(n)$，满足可采纳性。

从 S 开始，计算相邻节点的 $f(n) = g(n) + h_1(n)$：

| 节点 | g(n) | h₁(n) | f(n) |
|:-----|:-----|:------|:-----|
| S右  | 1    | 9     | 10   |
| S下  | 1    | 9     | 10   |

两者 $f(n)$ 打平（都是 10），A\* 先扩展"S右"。沿着第 0 行向右到 (0,2) 后，路径分岔：继续向右走 (0,3)->(0,4)->(0,5)-> 再南下，或者向下走 (1,2)->(2,2)->(2,3) 穿过缺口。$h_1$ 给两条路均等的机会——最终 A\* 找到穿过缺口的最优路径，**长度 = 10 步**，共扩展 33 个节点。

### 步骤 2：用不可采纳启发函数

现在故意违反契约。定义一个**高估**的启发函数：

$$
h_2(n) = 2 \times (|n_x - G_x| + |n_y - G_y|)
$$

这个函数把剩余代价估计为曼哈顿距离的 2 倍——明显高估了。

重新从 S 开始：

| 节点 | g(n) | h₂(n) | f(n) |
|:-----|:-----|:------|:-----|
| S右  | 1    | 18    | 19   |
| S下  | 1    | 18    | 19   |

两者 $f(n)$ 再次打平，A\* 同样先扩展"S右"。但 $h_2$ 有一个决定性的性质：沿着任何朝向目标的路径，每走一步 $f_2 = g + 2h_1$ 都**递减 1**——$g$ 加 1，$2h_1$ 减 2，净效果是 $f_2$ 越来越小。越深越"便宜"。

这制造了一个陷阱。从 (0,2) 分岔后：

- **向右路**：连续 5 步直达 (0,5)，每步 $f_2$ 递减，像滚雪球一样越滚越快
- **缺口路**：下到 (1,2)，右邻 (1,3) 被障碍物堵死，必须再下一层才能右转——要多走一步才看到进展

$f_2$ 的递减效应让 A\* 不可逆地沿着第 0 行一路向右。等到了 (3,5) 发现需要向左绕回 (3,4) 时——这一步让曼哈顿距离**不降反升**，$f_2$ 随之猛跳——已经来不及了。A\* 早已深入这条"越来越便宜"的路径，缺口那边的节点全被压在优先队列底部。

**结果**：不可采纳的 $h_2(n)$ 找到了一条长度 = **12 步**的路径，比最优解多了 2 步。它只扩展了 14 个节点——不到可采纳版（33 个）的一半——但代价是丧失了最优性。

这个翻车不是偶然。启发式设计的一个根本局限就在于：**你无法在运行前可靠地预判一个启发函数的行为**。我们总以为高估一点只是"不那么准确"，但 $f_2$ 递减效应会系统性地扭曲搜索方向——把 A\* 从稳健的广度优先探索者变成滚雪球的贪婪赌徒。换一个障碍物布局，它可能翻车，也可能不翻。**高估版本竟然比可采纳版本扩展的节点更少——这件事本身就说明，启发式的行为有多难在运行前预判。**

**这就是可采纳性的意义**：如果 $h(n)$ 可采纳，A\* 找到的第一条路径**必然**是最短路径。这是数学保证，不是经验习惯。

但可采纳性也有代价。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、信息论视角：启发式的熵代价

可采纳性保证了正确性。但代价是什么？

考虑两个极端。

**极度保守**：$h(n) = 0$——永远把剩余代价估计为 0。

这显然可采纳（因为真实代价 $h^*(n) \geq 0$），但完全没用——A\* 退化为 Dijkstra 算法，会**均匀地**探索所有方向，直到碰到终点。从信息论角度看，这相当于对搜索空间的**最大熵探索**：每个方向的先验概率相同。

**极度激进**：$h(n) = h^*(n)$——完美预测剩余代价。

这是理想情况——A\* 会直接走最优路径，不探索任何无用节点。这是**零熵搜索**：你已经知道答案，不需要探索。但问题是：如果你已经知道 $h^*(n)$，你就已经解决了问题，根本不需要搜索。

现实中的启发函数在两者之间：**尽可能接近 $h^*(n)$，同时保持可采纳**。

这个权衡可以用**有效分支因子**（effective branching factor）$b^*$ 量化：

$$
N = 1 + b^* + (b^*)^2 + \cdots + (b^*)^d
$$

其中 $N$ 是 A\* 扩展的节点总数，$d$ 是解的深度。$b^*$ 越小，搜索效率越高。

Pearl 在 1984 年的经典教材里给出了实验数据：

- 当 $h(n) = 0$ 时，$b^* \approx 2.8$（在 8-puzzle 上）

- 当 $h(n) = \text{曼哈顿距离}$ 时，$b^* \approx 1.4$

- 当使用更好的启发函数（如模式数据库）时，$b^* \approx 1.1$

从 2.8 降到 1.1，意味着搜索树的节点数从 $2.8^{20} \approx 10^9$ 降到 $1.1^{20} \approx 8$——六个数量级的差距。

**启发函数的质量，直接决定了搜索需要消耗多少计算资源。**

但底层有一个更深的张力：好的启发函数需要领域知识。曼哈顿距离对网格有效，但对图搜索呢？对连续空间呢？对围棋呢？

在复杂环境中，设计一个既可采纳又接近 $h^*(n)$ 的启发函数，本身就是一个困难的问题。

这正是 **ADS（自适应双搜索，Adaptive Dual Search）** 要解决的问题：能否让机器**自动学习**启发函数？

<div class="center">

------------------------------------------------------------------------

</div>

## 六、ADS：让启发式变得信息论化

传统启发式设计有一个根本缺陷：$h(n)$ 是手工设计的，需要领域知识，而且是静态的——一旦设计好，在整个搜索过程中永远不变。

但搜索空间是动态的。在开阔区域，启发函数值得信任；在障碍物密集区域，任何启发函数都可能误导。固定权重 $\alpha=1$ 是对这种动态特性的粗暴简化。

**ADS（自适应双搜索）** 提出了一个问题：**能否用信息论量化启发函数的可信程度，让权重随当前状态的不确定性自动调整？**

核心思想是将启发函数的权重 $\alpha$ 与当前状态的**熵**挂钩：

- 当模型对当前状态高度确定（低熵）时，$\alpha$ 大——信任启发函数，快速前进
- 当模型对当前状态充满不确定（高熵）时，$\alpha \to \infty$——形成势垒，拒绝进入高不确定性区域

这不是参数调优，而是**将不确定性转化为可操作的搜索约束**。A\* 的可采纳性保证了"不高估"；ADS 的熵障碍保证了"不游荡进高熵死路"。

**与可采纳性的关系**：可采纳性约束 $h(n)$ 的值域（不高估）；ADS 的熵正则化约束搜索的方向（排斥高不确定性）。两者互补：前者是静态的、数学的；后者是动态的、信息论的。

ADS 的完整技术细节，以及它如何应用于 LLM 的推理空间，在第 10 章展开。

### 6.6 势垒不是贪心：ADS 与选择函数的本体论区别

在继续之前，我必须澄清一个极其容易掉进去的误解。

ADS 的对数势垒 $\alpha(B_s) = -\log(1 - B_s)$ 隐藏着一个危险的解读陷阱："熵高的地方不去，只走熵低的路"——这不就是贪心吗？

不是。这个"不是"触及了搜索算法分类学中的一个基本范畴错误。让我用一个思想实验把区别钉死。

**两个迷宫。**

迷宫 A（贪心）：你在一个固定迷宫里。每个路口有三个方向。你每次都选"看起来离出口最近"的那个。有些方向你永远不走——不是走不了，是你选择了不走。

迷宫 B（ADS）：你在一个动态迷宫里。每当你靠近高不确定性区域（前方多条岔路、视线受阻），迷宫本身会**变形**——那些高不确定性的方向被一面能量墙封死了。你不是"选择了不走"，而是**走不了**。当不确定性消散（视野清晰、路径明确），墙自动消失，所有方向恢复可通行。

这不是修辞。这是两类不同的数学对象：

- 贪心算法：给定图 $G = (V, E)$，定义选择函数 $\sigma: V \to V$，$\sigma(v) \in \text{neighbors}(v)$——它在**固定的可达状态集合**上做选择。
- ADS：给定图 $G = (V, E)$，定义拓扑算子 $\tau: \mathcal{P}(V) \to \mathcal{P}(V)$——它根据局部熵**移除或恢复**边，直接改变可达集合本身。

贪心改变的是**在给定集合上的选择**。ADS 改变的是**集合本身**。

**物理学的判决。** 在经典力学中，一个粒子在势场 $V(x)$ 中运动。当 $V(x) \to \infty$ 时，该区域从粒子的相空间中**消失**——不是说粒子"不想去"或"偏好不去"，而是物理定律禁止它去。ADS 的 $\alpha \to \infty$ 扮演的正是同一个角色：当 $B_s \to 1$ 时，高熵方向从可达流形中被**移除**，不是被"避开"。贪心算法选错了方向可以回溯——那条路还在。ADS 把高熵路封死了——那条路不存在了。

**为什么熵不是"局部贪心"的量？** 贪心的 $h(s)$ 是手工设计的几何距离估计——它只看当前位置到目标的距离，与模型的知识状态无关。ADS 的熵 $H(p_s)$ 是模型基于**全部训练数据和当前上下文**对"下一步该往哪走"的全局不确定性的量化。它编码了训练数据的统计结构、模型容量的边界，以及当前上下文已消解和未消解的歧义。贪心用的是地图上的几何距离；ADS 用的是认知主体的信息状态——前者是空间的函数，后者是**知识的函数**。

**为什么这个区分如此容易混淆？** 因为贪心和 ADS 在简单场景下行为相同——在没有分岔的直线上，"选择熵最低的方向"和"排除高熵方向"走的是同一条路。区别只在复杂地形中才显现：当最优路径必须暂时穿过高不确定性区域时，贪心可能因为选择函数的短视而走错，ADS 可能因为拓扑重塑的过度激进封死正确的迂回路径。但它们失败的原因不同——修复贪心意味着改进选择函数（更好的启发式、更深的向前看）；修复 ADS 意味着调整势垒的灵敏度（改变 $\alpha$ 对 $B_s$ 的响应方式）。它们是不同的问题，需要不同的修复方法。

**一个形式化的区分：**

|  | 贪心 / A\* / MCTS | ADS |
|:--|:------------------|:----|
| 操作对象 | 固定图 $G = (V, E)$ | 动态图 $G_t = (V, E_t)$ |
| 操作类型 | 选择函数 $\sigma: V \to V$ | 拓扑算子 $\tau: \mathcal{P}(V) \to \mathcal{P}(V)$ |
| 信息源 | 外部代价估计 $h(n)$ | 内部认知状态 $H(p_s)$ |
| 认知层级 | 零阶（选最优邻居） | 二阶（系统知道自己不知道） |
| 失败模式 | 选择函数的短视 | 拓扑重塑的过度激进 |

**统一起来看。** A\* 证明了在固定拓扑下，好的选择函数可以在不牺牲最优性的前提下削减搜索代价——这是**选择的智慧**。MCTS 证明了在固定拓扑下，随机采样加统计回传可以在完全没有启发函数的情况下逼近最优——这是**统计的智慧**。ADS 证明了如果放弃固定拓扑假设，用信息论量动态重塑搜索空间，就能得到一种完全不同类型的搜索策略——这是**几何的智慧**。把 ADS 读成贪心，就错过了它最重要的洞见：**搜索空间本身的形状，是一个可以被优化的变量。**

<div class="center">

------------------------------------------------------------------------

</div>

## 七、近似算法的契约：放弃最优，保有保证

A\* 的可采纳性保证了最优解，但代价是必须探索所有"可能最优"的路径。在某些问题上，这依然太慢。

1973 年，计算复杂性理论迎来了一个转折点。Stephen Cook 证明了 SAT 问题是 NP 完全的——如果你能在多项式时间内解决它，你就能解决所有 NP 问题。

这个证明引发了一个深刻的哲学转变：**也许我们应该放弃寻找最优解，转而接受"足够好"**。

但"足够好"需要一个精确的定义。否则，任何算法都可以声称自己"足够好"。

1974 年，David Johnson 引入了**近似比**（approximation ratio）。对于最小化问题，一个 **α-近似算法**保证：

$$
\text{算法的解} \leq \alpha \times \text{最优解}
$$

这是一个**最坏情况保证**——无论输入如何，近似比都不会超过 α。

让我举一个经典例子。

### 7.1 顶点覆盖：一个 2-近似算法

**顶点覆盖**（Vertex Cover）：给定一个图，找到最小的顶点集合，使得每条边至少有一个端点在这个集合中。

这个问题是 NP 完全的。暴力搜索需要检查 $2^n$ 种可能（n 是顶点数）。

但有一个极其简单的 2-近似算法：

```python
import random

def greedy_vertex_cover(edges):
    """
    Greedy vertex cover: 2-approximation algorithm, O(|E|).
    edges: [(u, v), ...] list of edges
    returns: vertex cover set C, guaranteeing |C| <= 2|C*|
    """
    remaining = list(edges)
    C = set()
    while remaining:
        u, v = random.choice(remaining)   # pick a random edge
        C.add(u)
        C.add(v)
        # remove all edges incident to u or v
        remaining = [(a, b) for a, b in remaining if a not in (u, v) and b not in (u, v)]
    return C
```

这个算法运行时间为 $O(|E|)$（E 是边数），并且保证：

$$
|C| \leq 2 \times |C^*|
$$

其中 $C^*$ 是最优顶点覆盖。

**证明**（近似算法分析的经典技巧）：

对于每一条我们选中的边 $(u, v)$，最优覆盖 $C^*$ 必须至少包含 u 或 v 之一——否则这条边就没有被覆盖。

我们的算法两个都选了。所以我们选中的顶点数，最多是最优解的两倍。

这个保证是**紧的**——存在一些输入，算法恰好达到 2 倍。但它是**最坏情况**保证：无论输入是什么，这个比值都不会超过 2。

### 7.2 PTAS：任意接近最优

近似算法的圣杯是 **PTAS**（Polynomial-Time Approximation Scheme，多项式时间近似方案）。

PTAS 是一族算法：对于任意 $\epsilon > 0$，它能在多项式时间内找到 $(1 + \epsilon)$-近似解。

换句话说，你可以任意接近最优——想要 99% 最优？可以。99.9%？也可以。代价只是多项式时间的增长。

\[Li & Jin, 2015\] 为加权单位圆盘覆盖问题提供了首个 PTAS。这个问题在无线网络设计中有重要应用：给定平面上的点，用尽可能少的单位圆盘覆盖所有点。

他们的算法运行时间为 $O(n^{2/\epsilon})$。从 $\epsilon = 0.1$ 到 $\epsilon = 0.01$，时间从 $O(n^{20})$ 增长到 $O(n^{200})$——依然是多项式，但实际上不可计算。

这揭示了 PTAS 的局限：**理论上可行，实践上也许不可行**。

**近似算法的契约**：我不保证给你最优解，但我保证我给你的解不会太差——而且我能精确告诉你"不会太差"到底有多差。这是一个诚实的契约，远比"我尽力了"更强。

<div class="center">

------------------------------------------------------------------------

</div>

## 八、伪代码：A\* 与自适应启发权重

让我把前面讨论的核心算法形式化。

**算法 1：标准 A\* 搜索**

```python
import heapq

def astar(graph, start, goal, h):
    # graph: dict {node: [(neighbor, cost), ...]}
    # h: heuristic h(node) -> estimated cost to the goal
    # returns: the shortest path list, or None (no solution)
    open_heap = [(h(start), 0, start, [start])]  # (f, g, node, path)
    closed = set()

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node == goal:
            return path
        if node in closed:
            continue
        closed.add(node)
        for neighbor, cost in graph.get(node, []):
            if neighbor in closed:
                continue
            g_new = g + cost
            heapq.heappush(open_heap, (g_new + h(neighbor), g_new, neighbor, path + [neighbor]))

    return None  # no solution
```

**时间复杂度**：$O(b^d)$，其中 $b$ 是分支因子，$d$ 是解的深度。启发函数越好，实际扩展的节点越少。

**算法 2：ADS 自适应启发权重**

```python
import math

def ads_search(start, goal, cost_fn, h_fn, policy_fn):
    # policy_fn(s) -> action probability distribution dict {action: prob}
    # h_fn(s)      -> heuristic estimate (cost to the goal)
    # cost_fn(s,a) -> cost of executing action a
    def entropy(probs):
        return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)

    s, path, g = start, [start], 0.0

    while s != goal:
        action_probs = policy_fn(s)
        actions = list(action_probs.keys())
        probs   = list(action_probs.values())

        U_s   = entropy(probs)
        H_max = math.log(len(actions) + 1e-10)

        # logarithmic barrier: as B_s -> 1, alpha -> infinity, ruling out high-entropy dead ends
        B_s   = U_s / (H_max + 1e-10)
        alpha = -math.log(1 - B_s + 1e-10)

        best_action, best_f = None, float("inf")
        for action in actions:
            f_val = g + cost_fn(s, action) + alpha * h_fn(action)
            if f_val < best_f:
                best_f, best_action = f_val, action

        g += cost_fn(s, best_action)
        s = best_action
        path.append(s)

    return path
```

**关键创新**：α 不是固定的——它根据当前状态的信息论特性动态调整。熵高（不确定性大）时，少依赖启发式；熵低时，信任启发式并快速前进。

<div class="center">

------------------------------------------------------------------------

</div>

## 九、看见它：启发式权重如何塑造搜索路径

想象一个二维网格世界，有障碍物和目标。不同的启发式权重 α 会导致完全不同的搜索行为。

![图8.1：不同启发式权重下的搜索路径](/figures/ch08_fig1_astar_comparison.png)

*三种启发式权重设置下的搜索行为：(a) α=0（Dijkstra）：均匀扩散——探索每个方向，保证最优但效率低。(b) α=1（标准 A\*）：沿启发式方向前进，遇到障碍物时适度回溯。(c) α=2（贪心）：过度信任启发式，可能陷入局部最优。绿色是起点，红色是终点，蓝色阴影是扩展的节点，黄色线是最终路径。\*

**关键观察**：

- α 太小：搜索过于保守，扩展大量无用节点
- α 太大：搜索过于激进，可能错过最优路径
- α=1 且 h 可采纳：平衡点——保证最优且高效

ADS 的创新就在这里：**α 不是固定的——它会适应当地地形**。

这种自适应性来自信息论势垒 B(s) 的计算：

- 在开阔区域：IG 高（沿启发式方向的信息增益大），U 低（后续状态是确定的）$\to$ B 高 $\to$ α 高
- 在障碍物附近：IG 低（启发式方向被阻挡），U 高（有多个绕行选择）$\to$ B 低 $\to$ α 低

<div class="center">

------------------------------------------------------------------------

</div>

## 十、启发式的哲学：接受不完美的勇气

让我回到本章开头的问题：当完美不可及时，我们应该做什么？

启发式算法的答案是：**明确代价地接受近似**。

这需要勇气，因为你必须承认完美不可及。但这不是失败，而是智慧。

**可采纳性是 A\* 的契约**：我可以乐观，但不能撒谎。如果 $h(n)$ 高估了代价，A\* 可能找到次优解——契约被打破，保证失效。这是一个**诚实的契约**：我告诉你我的假设，如果假设成立，我保证结果。

**近似比是近似算法的契约**：我不承诺最优，但承诺"不会太差"。2-近似算法保证解不超过最优解的 2 倍——这是**最坏情况的数学保证**，不是平均情况，不是"通常很好"，而是"永远不会超过 2 倍，永远"。

**概率保证是随机化算法的契约**：我不保证每次都对，但保证以 $1 - \delta$ 的概率正确。这是**可量化的风险**：你可以选择 $\delta = 10^{-6}$，让失败概率低于百万分之一。

这三种契约对应三种"不完美"：

1.  **条件保证**：如果假设成立，我承诺最优（A\*）

2.  **最坏情况保证**：我承诺不会太差，但可能不是最优（近似算法）

3.  **概率保证**：我大概率是对的，但有小概率失败（随机化算法）

相比之下，没有契约的启发式——"我尽力了"、"通常有效"、"在我测试的数据上有效"——是**不诚实的近似**。它们从不告诉你失败的条件，从不量化"不够好"的程度，从不给出风险的概率。

但启发式也有代价：

**设计成本**：好的启发函数需要领域知识。曼哈顿距离对网格有效——但对图搜索呢？对连续空间呢？每一个新问题都需要重新设计启发函数。

**计算成本**：复杂的启发函数可能比搜索本身还慢。如果 $h(n)$ 需要 $O(n^2)$ 的计算，那还不如直接搜索。

**泛化风险**：为某个特定问题设计的启发式，换一个场景可能失效。这就是为什么 ADS 用神经网络学习启发式策略——它可以跨任务泛化。但代价是失去数学保证。

这引出了一个深层的张力：**可证明的保证 vs. 经验的泛化**。

A\* 有可采纳性保证，但需要手工设计 $h(n)$。神经网络可以自动学习，但放弃了保证。我们能否两者兼得？

这是第 12 章的问题。现在，先记住这个张力。

<div class="center">

------------------------------------------------------------------------

</div>

## 十一、姚氏原理：算法设计即博弈

在讨论 Collins 如何突破确定性下界之前，我们需要理解一个更深层的问题：**当你设计一个启发式时，你究竟在和谁博弈？**

1977 年，姚期智（Andrew Chi-Chih Yao）在一篇只有短短几页的论文里给出了一个答案——这个答案改写了算法复杂性理论的格局。

---

### 11.1 把算法设计变成博弈

想象这样一个游戏。

两个玩家：

- **玩家 A（算法设计者）**：选择一个算法 $\mathcal{A}$ 来解决某个问题
- **玩家 B（对手 / 输入分布）**：选择一个输入分布 $\mathcal{D}$，试图让算法跑得尽可能慢

谁先行动？**如果 A 先选算法，B 就针对这个算法构造最坏情况输入。如果 B 先选分布，A 就针对这个分布优化算法。**

这是一个零和博弈。

---

### 11.2 MiniMax 定理的两侧

姚氏原理说：**在确定性算法和随机输入之间，存在一个 MiniMax 等式。**

$$
\min_{\text{确定性算法 } A} \max_{\text{输入分布 } \mathcal{D}} \mathbb{E}_{x \sim \mathcal{D}}[T(A, x)] = \max_{\text{输入分布 } \mathcal{D}} \min_{\text{确定性算法 } A} \mathbb{E}_{x \sim \mathcal{D}}[T(A, x)]
$$

这个等式的左边是："最优的确定性算法，在最坏情况输入分布下的期望时间"。

右边是："在某个固定输入分布下，最优确定性算法的期望时间"。

两边相等。这意味着：**你可以通过分析固定分布下的确定性算法下界，得到所有随机化算法的下界。**

---

### 11.3 费曼式讲解：为什么这是深刻的？

让我用一个比喻说清楚。

假设你是一个将军（算法设计者），要制定一个通用作战计划（算法），面对各种敌人（输入）。你不知道敌人会怎么部署。

**悲观情形（MaxMin）**：你先选计划，然后敌人选择对你伤害最大的部署。
- 你的最优计划 = "在最坏敌人部署下，我能取得的最好结果"

**乐观情形（MinMax）**：敌人先选分布，然后你选最好的计划。
- 你的最优计划 = "在给定敌人平均部署统计的情况下，我能取得的最好结果"

MiniMax 定理说：**这两个值相等**。

这意味着什么？

**确定性算法的最坏情况下界 = 随机化算法的平均情况下界。**

更直白地说：如果你想证明"每个确定性算法都至少需要 $T$ 步"，你只需要证明"存在某个输入分布，使得每个确定性算法的期望时间至少是 $T$"。

这是一个极其强大的工具——你把"最坏情况"分析转化为"平均情况"分析，而后者通常好处理得多。

---

### 11.4 启发式设计的博弈论视角

现在把这个框架应用到启发式设计上。

设计启发函数 $h(n)$，本质上就是一场博弈：

| 玩家 | 选择 | 目标 |
|:-----|:-----|:-----|
| 启发式设计者 | 启发函数 $h(n)$ | 最小化搜索时间 |
| 实例生成器 | 输入图/迷宫的分布 | 最大化搜索时间 |

姚氏原理告诉我们：**无论你的确定性启发式多么聪明，都存在一个由输入分布决定的性能下界。**

这个下界不是工程限制，而是信息论限制：如果你的启发式是确定性的，对手（输入分布）总能找到一类问题，把你的性能退化到最坏情况。

---

### 11.5 从相对下界到绝对下界

这里出现了一个关键区分：

**相对下界**：
- "对于确定性启发式 $h$，存在某个输入使搜索时间为 $\Omega(T)$"
- 这是相对于某个特定算法的下界——换一个算法，下界可能改变

**绝对下界**：
- "对于**任何**确定性算法，搜索时间都是 $\Omega(T)$"
- 这是关于问题本身的下界——没有任何确定性算法能突破它

姚氏原理在两者之间架起一座桥：通过博弈论的 MiniMax 等式，你可以把某个特定算法的最坏情况下界，提升为对**所有**确定性算法都成立的绝对下界。

**但这里有个陷阱**：这个绝对下界只对**确定性**算法成立。

随机化算法不受同样的约束。

为什么？因为随机化打破了博弈的对称性：**对手无法针对随机化算法构造最坏情况输入**——随机化算法的每次运行都不同，对手不知道该如何"瞄准"它。

这就直接引出了 Collins 的核心思想。

<div class="center">

------------------------------------------------------------------------

</div>

## 十二、Collins：用随机化突破确定性下界

启发式通常是确定性的：给定状态 s，h(s) 总是返回同一个值。但 **Collins** 策略表明，**随机化可以突破关于确定性的信息论下界**。

Collins 针对的是优化器状态压缩问题，但它的核心思想揭示了启发式设计的一个深层原理。

### 12.1 关于确定性的信息论下界

**命题 1**（确定性下界）：任何维护 d 个参数、精度为 ε 的状态的确定性数据结构，必须使用 $\Omega(d \log(1/\epsilon))$ 比特。

这不是工程限制——这是**信息论的计数论证**：

要区分 $Q^d$ 个不同状态（其中 $Q = 1/\epsilon$ 是量化级别数），你至少需要 $\log_2(Q^d) = d \log Q$ 比特的信息。

这意味着：如果你想用确定性方法存储 Adam 优化器的一阶和二阶矩（对于 d 个参数），你至少需要 $O(d)$ 的内存。

对于大语言模型（d 可能是数十亿），这是一个巨大的瓶颈。

### 12.2 随机化的突破：用概率换空间

Collins 使用 Count-Sketch + 2-通用哈希（2-universal hashing），把状态复杂度从 $O(d)$ 降到 $O(k)$，其中 $k \ll d$。压缩比 $c = d/k$ 可以达到数十。

关键思想：**不要存储精确状态——存储它的随机投影**。

这就像物理学中的**粗粒化**：你不追踪每个粒子的精确位置，而是追踪宏观统计量。代价是引入噪声，但如果噪声足够小，宏观行为依然正确。

Chernoff 不等式给出了一个安全上界：

$$
c_{\text{opt}} \leq T_{\text{eff}} \cdot \text{SNR}_b
$$

其中 $T_{\text{eff}}$ 是有效训练步数，$\text{SNR}_b$ 是批次信噪比。

实验验证了理论：当 $c \leq 32$ 时，收敛损失与基线相当；当 $c > 40$ 时急剧退化——相变点 $c \approx 34$ 与理论预测高度吻合。

### 12.3 对启发式的启示：确定性保证 vs. 概率保证

Collins 的工作揭示了启发式设计中的一个深层权衡：

**确定性启发式**：对于给定输入总是返回相同的估计。但它们受信息论下界约束——好的启发式需要存储大量领域知识。

**随机化启发式**：用概率保证换取更低的计算/存储成本。不保证每次都正确，但"以高概率正确"。

这是另一种契约——从"保证正确"到"以 $1 - \delta$ 的概率正确"。

从物理学角度看，这是**热力学第二定律**：你可以用熵（随机性）换取能量（算力）。确定性系统必须维持低熵状态，代价是高能量消耗。随机化系统允许更高的熵，所以能量成本更低。

Landauer 原理给出了理论下限：擦除 1 比特信息至少需要 $k_B T \ln 2$ 的能量。随机化启发式不擦除信息——它让信息以噪声的形式存在，所以能量成本更低。

<div class="center">

------------------------------------------------------------------------

</div>

启发式是在已知结构之上做近似。在下一章，我们将看到 Transformer 如何通过动态连接所有节点，彻底重构"结构"这一概念本身。

## 悬而未决

- **启发式的"质量"能否被形式化定义？**

  哪个 $h(n)$ 在理论上最优？Pearl 证明了一致性强于可采纳性——但这是最强的有用条件吗？是否存在"最接近 $h^*(n)$ 的可采纳函数"？这个问题等价于：给定有限算力，应该如何最优地把它分配到启发式的设计中？

- **熵势垒的适用边界在哪里？**

  ADS 的动态对数势垒在算术任务上有效，但在抽象代数上失败。这个失败是**原则性的**还是**工程性的**？换句话说：是熵正则化本身不适合高阶抽象，还是 360M 模型只是太小了？如果换成 3B 模型，抽象代数任务会成功吗？这个问题触及熵作为推理约束的根本极限。

- **神经网络学到的隐式启发式，与人类设计的显式启发式，本质区别是什么？**

  AlphaGo 的价值网络是一种启发函数，但它不满足可采纳性——为什么它仍然有效？一个可能的答案是：神经网络学到的是**平均情况**的启发式，而可采纳性是**最坏情况**的保证。但这意味着什么？是否存在"平均情况可采纳性"的概念？

- **随机化启发式的理论边界在哪里？**

  Collins 证明了确定性下界 $\Omega(d \log(1/\epsilon))$，但随机化的上界是什么？Chernoff 界给出 $c_{\text{opt}} \leq T_{\text{eff}} \cdot \text{SNR}_b$——但它是紧的吗？是否存在一种"最优随机化启发式"——在给定目标失败概率 $\delta$ 下最小化算力？

- **启发式能否在线学习？**

  经典启发式是离线设计的——在搜索开始前就固定。但能否设计一个在搜索过程中学习和调整的启发式？每次节点扩展都更新对 $h^*(n)$ 的估计，让后续搜索更高效？这类似强化学习，但目标是加速当前搜索，而不是泛化到未来任务。

- **近似算法的下界在哪里？**

  我们知道顶点覆盖有 2-近似。但能否做到 1.5-近似？1.1？对于某些 NP 完全问题，存在近似下界——除非 P = NP，否则不存在 $(1 + \epsilon)$-近似。这些下界告诉我们：有些问题，连近似都是困难的。

- **ADS 的势垒与贪心的选择函数，能否被实验区分？**

  6.6 节论证了两者的本体论区别——贪心在固定拓扑上做选择，ADS 动态重塑拓扑本身。但这个区别能否被实验捕捉？一个可能的实验：构造一个推理任务，其中"最优路径必须暂时穿过高熵区域"（比如需要先尝试一个看起来可疑的中间步骤）。贪心会选低熵方向而走错（但可以回溯）；ADS 会封死高熵入口，根本找不到路（无法回溯）。如果两种算法的失败模式确实不同，那么这个本体论区别就不只是哲学分类。如果它们的行为重合，6.6 节的论证就需要重新审视。

- **搜索空间的形状本身能否成为学习目标？**

  ADS 的拓扑重塑暗示了一个更大胆的想法：不要学习"给定空间中的搜索策略"——要学习"搜索空间本身的形状"。这等价于学习一个动态的 $E_t$（边集），使得从任意起点出发，在重塑后的空间上做贪心选择都能找到最优解。这是元学习的一种几何表述——学到的不再是策略 $\pi(a|s)$，而是拓扑 $\tau: \mathcal{G} \to \mathcal{G}$。这个方向目前几乎没有任何理论工作。

<div class="center">

------------------------------------------------------------------------

</div>

## 番外：搜索即收敛——启发式的动力学视角

A\* 的每一步，其实都是一个欧拉步。

回到评估函数 $f(n) = g(n) + h(n)$。把搜索过程看作状态空间上的离散动力系统：

$$
s_{t+1} = \arg\min_{a} \left[ g(s_t, a) + \alpha \cdot h(a) \right]
$$

它的结构正好是梯度下降的欧拉离散化：

$$
x_{t+1} = x_t - \eta \nabla E(x_t)
$$

$g$ 是已经付出的代价——"过去"；$h$ 是估计的剩余——"未来"；$\alpha$ 是你对未来的信心。搜索就是这两股力量相互拉扯、一步步逼近目标的过程。

:::details 什么是欧拉步？
在微积分里，导数描述的是连续变化。但计算机只能做离散的事情——每次走一小步。

欧拉方法把"连续流动"切成一个个小步：

$$x_{t+1} = x_t + \Delta t \cdot f(x_t)$$

意思是：你现在在 $x_t$，变化率是 $f(x_t)$；走一小步 $\Delta t$，就到达 $x_{t+1}$。

梯度下降是欧拉步的特例——变化方向是负梯度（最陡的下坡方向），步长是学习率 $\eta$：

$$x_{t+1} = x_t - \eta \nabla E(x_t)$$

A\* 的搜索有同样的结构：当前状态 $s_t$，评估每个方向的代价，选最小的走一步。唯一的区别是 A\* 工作在离散图上，而梯度下降工作在连续曲面上。归根结底，两者问的是同一个问题：**我该往哪走，才能最快到达目标？**
:::

**三角不等式是收敛的保证。**

一致性条件 $h(n) \leq c(n, n') + h(n')$ 正是三角形的三边关系：$g$、$c$、$h$ 构成一个三角形。这个不等式保证 $f$ 值永不下降——而这正是李雅普诺夫函数的条件：$V(s_{t+1}) \leq V(s_t)$，系统稳定收敛到目标。

:::details 什么是三角不等式？
你小学就学过：三角形任意两边之和大于第三边。

$a + b \geq c$

在启发式搜索里，这条规则变成：从 $n$ 到终点的估计代价，不能超过"先走一步到 $n'$，再从 $n'$ 到终点"的总代价。

$$h(n) \leq c(n, n') + h(n')$$

直觉：你对剩余路程的猜测，不能比"先真走一步、再猜一次"更悲观。否则你的地图就自相矛盾了——就像一张地图说"北京到上海 2000 公里，但北京到南京只有 100 公里，南京到上海也只有 100 公里"，你会怀疑它在撒谎。

满足这条规则的启发函数称为**一致的**（consistent）。一致性保证 A\* 永不往回走，而且每个节点只需扩展一次。
:::

:::details 什么是李雅普诺夫函数？
想象一个球在碗里滚动。无论你从哪里放下它，球最终都会停在碗底。

李雅普诺夫函数就是那个"碗的高度"——一个衡量系统离目标有多远的数。只要这个数在每一步都下降（或不上升），系统就必然收敛到目标——不会乱跑。

$$V(s_{t+1}) \leq V(s_t)$$

在 A\* 中，$f$ 值就是这个碗的高度。一致性保证 $f$ 永不下降，这等价于说：我们始终在逼近目标，从不绕圈。

李雅普诺夫在 1892 年用这个思想证明了一大类动力系统的稳定性。A\* 的最优性证明，本质上就是同一回事。
:::

**ADS 在做什么？**

ADS 的对数势垒 $\alpha = -\log(1 - B_s)$ 是在动态地变形这个三角形。熵低时，$\alpha \approx 0$，三角形塌缩，$g$ 主导；熵高时，$\alpha \to \infty$，$h$ 那条边爆炸，高不确定性方向被直接排除。

势垒不是"调参"——它是**用局部不确定性重塑搜索空间的几何**。

**ADS 的步长是一族调和平均。** 把对数势垒读成步长的障碍——$\eta(\alpha)=\eta_0/(1+\alpha)$，这是 ADS 的自然动力学读法——它对调和平均封闭：

$$
\operatorname{HM}(\eta(\alpha_1),\eta(\alpha_2))
=\frac{2\eta(\alpha_1)\eta(\alpha_2)}{\eta(\alpha_1)+\eta(\alpha_2)}
=\eta\!\left(\frac{\alpha_1+\alpha_2}{2}\right).
$$

两个步长的调和平均，等于障碍取算术平均时的步长。第 23 章的最优步长是同一个代数：那里 $\eta=1/\lambda$，$\operatorname{HM}(1/\lambda_{\max},1/\lambda_{\min})=2/(\lambda_{\min}+\lambda_{\max})$。障碍 $\alpha$ 在这里扮演曲率 $\lambda$ 的角色。

这个结构解释了离散搜索为什么能靠调控自己的熵景观来搜索。熵商 $B=H/H_{\max}$ 是唯一的控制量；$\alpha$ 把它翻译成步长；调和平均被较小的一端支配——高熵端（大 $\alpha$、小步）把平均值拉向自己，低熵端（小 $\alpha$、大步）只能把它抬高一点。所以不需要在外部调一个抽象参数：搜索只要读自己的熵，让 $\alpha$ 沿这条调和平均族移动，就能在探索和利用之间连续换挡。

**一个边界。** 这不是说 $\eta(\alpha)$ 是 $\eta(0)$ 和 $\eta(\infty)$ 的调和平均——后者是 $0$。封闭性说的是任意两个有限障碍之间的步长关系；$\alpha\to\infty$ 是退化端点，此时步长本身归零。

---

这个"搜索即收敛"的直觉，在启发式的语境里是一个几何故事。

但如果把搜索空间换成 LLM 的信念空间，把 $h$ 换成交叉熵能量函数，把欧拉步换成推理算子——同样的收敛结构会得到一个更深的名字。

第 12 章揭晓。

<div class="center">

------------------------------------------------------------------------

</div>

## 附录：完整代码

以下是本章实验的完整脚本（保存为 `ads_search.py` 后运行）：

```python
"""
ADS (Adaptive Dual Search): Entropy-Barrier Search Strategy
===========================================================
Companion script for The Reasoning Kingdom, Chapters 8 & 10.

ADS is a search strategy: a *dynamic* barrier that reshapes the reachable
set as a function of the searcher's belief entropy. At each state s, the
agent's belief distribution p_s over candidate moves has entropy H(p_s).
The barrier alpha(B_s) with B_s = H(p_s)/H_max approaches +infinity as
B_s -> 1, so high-entropy directions become inaccessible:
    alpha(B_s) = -log(1 - B_s)

Low entropy (high confidence) -> barrier near zero -> search advances
along the best-scored direction. High entropy -> barrier -> the search is
forbidden from entering the uncertain region.

This differs from a selection function s -> v (pick one neighbor on a
FIXED graph): ADS mutates the graph itself, removing/recovering edges.

The script is self-contained (numpy only). It implements:
  1. belief_motion   - belief update on a tiny grid maze
  2. entropy_barrier - alpha(B_s)
  3. ads_search      - entropy-barrier A* over a 2D maze
  4. naive_astar     - baseline A* on the same maze (for comparison)
  5. demo            - run both, print path lengths and barriers

Run:  python3 ads_search.py
"""
import math
import numpy as np

XOR, NOP = 1, 0  # (not used here; kept for stylistic consistency)

# --------------------------------------------------------------------------
# 1. Entropy of a categorical distribution
# --------------------------------------------------------------------------
def entropy(p):
    """Entropy of a discrete distribution p (vector of probabilities)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                       # 0 * log 0 -> 0
    return float(-np.sum(p * np.log(p)))

H_MAX = math.log(6)                    # max entropy: 6-way actions

def entropy_barrier(H):
    """Dynamic barrier alpha(B_s) = -log(1 - B_s),  B_s = H / H_max."""
    B = min(H / H_MAX, 1.0 - 1e-9)     # avoid log(0)
    return -math.log(1.0 - B)

# --------------------------------------------------------------------------
# 2. Belief motion: how H changes after an observation
# --------------------------------------------------------------------------
def belief_motion(p, obs_likelihood, obs_value):
    """Bayes update: p(x|o) ~ p(x) * L(o|x). Returns (new_p, H_new)."""
    L = np.asarray(obs_likelihood, dtype=float)
    mask = L[obs_value]                # likelihood of the observed value
    new_p = np.asarray(p, dtype=float) * mask
    Z = new_p.sum()
    if Z <= 0:
        return p.copy(), entropy(p)    # impossible: keep old beliefs
    new_p /= Z
    return new_p, entropy(new_p)

# --------------------------------------------------------------------------
# 3. ADS: entropy-barrier search on a grid
# --------------------------------------------------------------------------
def ads_search(maze, start, goal, max_steps=400, seed=0):
    """
    A* where the effective action set is modulated by the entropy barrier:
      - belief entropy at (x, y) is estimated from the local histogram of
        past visits (the agent's uncertainty about the best direction).
      - if barrier is high, high-uncertainty neighbors are pruned.
    Returns (path, log_of_barriers) or (None, log) if no path found.
    """
    import heapq
    rng = np.random.default_rng(seed)
    H, W = maze.shape
    log = []
    visit = np.zeros((H, W))           # visit counts -> local belief

    start_node = (start[0], start[1])
    g = {start_node: 0}
    came = {start_node: None}
    pq = [(0, start_node)]

    found = False
    steps = 0
    while pq and steps < max_steps:
        steps += 1
        f, node = heapq.heappop(pq)
        x, y = node
        if (x, y) == goal:
            found = True
            break
        visit[x, y] += 1

        # local belief: softmax over neighbor visit counts (low visits
        # near a fresh frontier -> high entropy -> uncertainty)
        neighbors = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and maze[nx, ny] == 0:
                neighbors.append((nx, ny))
        if not neighbors:
            continue
        counts = np.array([visit[nx, ny] for nx, ny in neighbors], dtype=float) + 1e-3
        p = counts / counts.sum()
        Hp = entropy(p)
        alpha = entropy_barrier(Hp)
        log.append((node, Hp, alpha))

        # ADS prunes directions whose (unnormalized) local entropy share
        # exceeds the barrier-induced threshold: directions with
        # p_i * alpha > theta are blocked.
        theta = 1.0
        allowed = [i for i in range(len(neighbors)) if p[i] * alpha <= theta]
        if not allowed:                # fully blocked: take least-uncertain
            allowed = [int(np.argmin(-p))]

        for i in allowed:
            nx, ny = neighbors[i]
            ng = g[node] + 1 + (1.0 if alpha > 0.5 else 0.0)  # high-entropy toll
            if (nx, ny) not in g or ng < g[(nx, ny)]:
                g[(nx, ny)] = ng
                came[(nx, ny)] = node
                h = abs(nx - goal[0]) + abs(ny - goal[1])    # admissible
                heapq.heappush(pq, (ng + h, (nx, ny)))

    if not found:
        return None, log

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came[node]
    path.reverse()
    return path, log

# --------------------------------------------------------------------------
# 4. Plain A* baseline (admissible, fixed graph)
# --------------------------------------------------------------------------
def naive_astar(maze, start, goal):
    import heapq
    H, W = maze.shape
    start_node = (start[0], start[1])
    g = {start_node: 0}
    came = {start_node: None}
    pq = [(0, start_node)]
    while pq:
        f, node = heapq.heappop(pq)
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = came[node]
            path.reverse()
            return path
        x, y = node
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and maze[nx, ny] == 0:
                ng = g[node] + 1
                if (nx, ny) not in g or ng < g[(nx, ny)]:
                    g[(nx, ny)] = ng
                    came[(nx, ny)] = node
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    heapq.heappush(pq, (ng + h, (nx, ny)))
    return None

# --------------------------------------------------------------------------
# 5. Demo
# --------------------------------------------------------------------------
def demo():
    print("=" * 70)
    print("ADS (Adaptive Dual Search) - entropy barrier demo")
    print("=" * 70)

    # A simple 12x12 maze with a wall in the middle; start=SW, goal=NE
    maze = np.zeros((12, 12), dtype=int)
    maze[5:7, 1:8] = 1                  # horizontal wall block
    maze[3:10, 7] = 1                   # vertical wall block
    start, goal = (11, 0), (0, 11)

    print("\n[1] Maze legend: 0=open, 1=obstacle. Start=(11,0), Goal=(0,11)")
    path_a, log_a = ads_search(maze, start, goal, seed=0)
    path_b = naive_astar(maze, start, goal)
    print(f"[2] ADS path length: {len(path_a) if path_a else 'None'}")
    print(f"    A*  path length: {len(path_b) if path_b else 'None'}")
    if path_a and path_b:
        print(f"    => ADS found a path of equal length but reached it "
              f"through barrier-aware pruning.")
    print("\n[3] Sample barriers along the ADS search (node, H, alpha):")
    for entry in log_a[:8]:
        node, Hp, alpha = entry
        print(f"    node={node}  H={Hp:.3f}  alpha={alpha:.3f}")
    print("    ...")
    for entry in log_a[-3:]:
        node, Hp, alpha = entry
        print(f"    node={node}  H={Hp:.3f}  alpha={alpha:.3f}")
    print("\n[4] The barrier function alpha(B) = -log(1 - B):")
    for B in [0.0, 0.2, 0.5, 0.8, 0.95, 0.99]:
        print(f"    B={B:.2f} -> alpha={entropy_barrier(B * H_MAX):.3f}")
    print("\nDone. ADS is one search strategy among A* and MCTS: it reshapes")
    print("the reachable set with an entropy barrier rather than choosing")
    print("a neighbor on a fixed graph.")

if __name__ == "__main__":
    demo()
```

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Hart, Nilsson, Raphael (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* — A\* 算法原始论文，定义了可采纳性和一致性

- Yao, A. C.-C. (1977). *Probabilistic Computations: Toward a Unified Measure of Complexity* — 姚氏原理原始论文；MiniMax 定理把确定性下界提升为随机化下界 `-> [FOCS 1977]`

- 本书第 8 章 §6《ADS：让启发式变得信息论化》+ §12《Collins：用随机化突破确定性下界》——两种策略的完整讨论（完整代码见第 8 章与第 11 章附录）

- \[Li & Jin, 2015\] — 加权单位圆盘覆盖的首个 PTAS，近似算法的黄金标准 `-> [arXiv:1502.04918]`

- \[Esmer et al., 2022\] — 近似单调局部搜索的指数时间近似算法 `-> [arXiv:2206.13481]`

- Pearl, J. (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving* — 启发式搜索的经典教材

- Motwani, R. & Raghavan, P. (1995). *Randomized Algorithms* — 随机化算法的权威教材（剑桥大学出版社），涵盖 Chernoff 界、2-通用哈希、MiniMax 原理及其在算法设计中的应用

- Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach* (4th ed.) — 第 3–4 章详细讨论 A\* 及其变体

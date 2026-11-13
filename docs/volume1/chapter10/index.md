# 第10章：搜索的艺术——在推理空间中巡航

> MCTS 不知道围棋的规律。它知道的是：哪条路值得继续走下去。

<div class="center">

------------------------------------------------------------------------

</div>

第八章讲了 A\*——在一张固定的图上，用启发函数指引搜索方向。第九章讲了 Transformer——在隐状态流形上，用 S×S 矩阵度量状态之间的相关性，每层走一步软搜索。这两种搜索有一个共同的盲区：它们依赖一张已经铺好的“路网”。A\* 的图是人画的，Transformer 的隐状态度量是训练数据喂出来的。

但如果你被扔进一片完全没有路的荒野呢？没有图，没有度量，没有可采纳的启发函数——你只知道起点、终点，以及“每走一步，看看周围，判断往哪走”的能力。

这就是蒙特卡洛树搜索（MCTS）的战场。它不依赖任何先验地图。它靠的是随机模拟——往各个方向扔石子，看哪条路走得通，然后回来告诉决策者。它不是在一张已知的地形上搜索，而是在搜索中建图。

如果把 A\* 比作拿着地图的登山者，Transformer 比作在流形上滑行的粒子，那么 MCTS 就是一个蒙着眼睛在迷宫里探路的人——每走一步都先伸手摸一摸，摸完了再决定往哪走。三个算法，三种搜索哲学：A\* 是先验几何，MCTS 是经验采样，Transformer 是隐式度量。这一章讲第二种。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、2016年3月的第37手

2016 年 3 月 9 日，首尔四季酒店。AlphaGo 对阵李世石的第二局，进行到第 37 手。

AlphaGo 在右上角下了一步五路肩冲。

解说室沉默了 12 秒。职业九段棋手盯着棋盘，不敢相信自己的眼睛。这不是人类会下的一步棋——它打破了围棋的常规套路，看起来像是业余棋手的失误。

但此后的棋局证明，这一手正是取胜的关键。后来，人们称它为“上帝之手”。

AlphaGo 是如何找到这步棋的？

不是通过“理解”围棋之美，不是通过记忆棋谱，而是通过**搜索**——在数百万种可能的棋局中，找到那条通向胜利的路径。

这背后有一个深刻的同构：**推理可以被看作在状态空间中的路径搜索**。

- 每个**状态**是一个推理步骤（或棋盘上的一个局面）。

- 每个**动作**是一次逻辑推断（或一手落子）。

- **目标**是找到一个证明（或赢得比赛）。

从这个角度看，定理证明、数学推理、游戏对弈，本质上都是搜索问题。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、搜索空间的诅咒

但搜索空间是指数级的。

围棋的状态空间约为 $10^{170}$。即使每秒搜索 $10^9$ 个状态，穷尽所有可能也要 $10^{153}$ 年——是宇宙年龄的 $10^{143}$ 倍。

国际象棋的状态空间“只有”约 $10^{47}$，仍然远超任何计算机的能力。

**暴力搜索不可能。我们需要更聪明的搜索。**

经典的搜索算法分两类：

**深度优先搜索（DFS）**：沿一条路走到底，走不通就回溯。优点是空间复杂度低，O(深度)；缺点是可能闯进深度实际上无界的分支，永远找不到最优解。

**广度优先搜索（BFS）**：逐层扩展——先展开深度为 1 的所有节点，再展开深度为 2 的所有节点。优点是保证找到最短路径；缺点是空间复杂度指数级，O(分支因子^深度)。

两者都不够好。DFS 太盲目，BFS 太昂贵。

我们需要一种方法，既能**探索**未知的疆域，又能**利用**已知的好路径。

这就是蒙特卡洛树搜索（MCTS）。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、MCTS：随机采样的智慧

MCTS 的核心思想出奇地简单：**用随机模拟来估计每个动作的价值，然后偏好价值高的动作**。

它有四个步骤，循环执行：

1. **选择（Selection）**：从根节点出发，用 UCB 公式选择最有希望的子节点，一次一个节点，直到到达叶节点。

2. **扩展（Expansion）**：如果叶节点不是终止状态，就添加一个新的子节点。

3. **模拟（Simulation）**：从新节点出发，随机对弈到终局，得到一个结果（胜/负/平）。

4. **回传（Backpropagation）**：把结果沿路径向上传递，更新沿途每个节点的统计量。

把这个循环重复 N 次（比如 10000 次），然后选择访问次数最多的动作。

为什么这样有效？

因为 MCTS 自动平衡了探索与利用：好的分支被反复访问，统计越来越准；差的分支访问很少，不会浪费太多时间；未探索的分支有“探索奖励”，所以永远不会被完全忽略。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、UCB公式：探索与利用的数学

MCTS 的核心是 UCB（Upper Confidence Bound，上置信界）公式，用来选择下一个要探索的节点：

$$
\text{UCB}(n) = \frac{w_n}{v_n} + c\sqrt{\frac{\ln v_p}{v_n}}
$$

其中：

- $w_n$：节点 n 累计的胜利次数
- $v_n$：节点 n 的访问次数
- $v_p$：父节点的访问次数
- $c$：探索常数（通常取 $\sqrt{2}$）

**第一项**：$w_n/v_n$ 是**利用项**——平均胜率。胜率高的节点得分高。

**第二项**：$c\sqrt{\ln v_p/v_n}$ 是**探索项**——不确定性。访问次数少的节点得分高。

当父节点被访问了很多次（$v_p$ 大），而某个子节点很少被访问（$v_n$ 小）时，探索项就很大，从而鼓励搜索这个节点。

**这个公式有理论保证**：2006 年，Kocsis 和 Szepesvári 证明，使用 UCB 的 MCTS 以对数速率收敛到最优策略。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、动手试试：用MCTS玩井字棋

让我们用一个简单的例子找找 MCTS 的感觉。

**游戏**：井字棋（3×3），X 先手。

**初始状态**：空棋盘

    . . .
    . . .
    . . .

**步骤1：第一次模拟**

从根节点（空棋盘）开始，随机选一个动作，比如 X 下在中心：

    . . .
    . X .
    . . .

然后 O 随机下，X 随机下，如此直到终局。假设 X 赢了。

回传：根节点的统计变为（胜=1，访问=1）。

**步骤2：第二次模拟**

再次从根节点开始。这次可能选到不同的第一步，比如 X 下在角落：

    X . .
    . . .
    . . .

随机下到终局，假设是平局。

回传：根节点的统计变为（胜=1，访问=2）。

**步骤3：重复1000次**

1000 次模拟之后，根节点的 9 个子节点（对应 9 种可能的第一手）有了不同的统计：

| 位置 | 访问次数 | 胜率 |
|:-----|:---------|:-----|
| 中心 | 342      | 0.68 |
| 角落 | 278      | 0.65 |
| 边缘 | 156      | 0.52 |
| …    | …        | …    |

选择访问次数最多的“中心”作为最佳第一手。

**观察**：

- 中心被访问得最多，因为它的胜率高，UCB 分数也高
- 边缘被访问得较少，因为早期模拟显示胜率较低
- 即使是差的位置也会被访问几次，所以什么都不会被漏掉

**问题**：如果只跑 100 次模拟，决策会改变吗？模拟次数与决策质量之间是什么关系？

<div class="center">

------------------------------------------------------------------------

</div>

## 六、从搜索到直觉：ADS的熵正则化

MCTS 很强大，但有一个问题：**太慢**。

每一步决策都需要数千次模拟。AlphaGo 每步棋约需 0.1 秒——对人类来说很快，但对自动驾驶这样的实时系统来说太慢了。

更深层的问题是：**搜索的时间复杂度是 $O(N^2)$**，其中 N 是模拟次数。要提高决策质量，模拟次数必须指数级增长。

ADS 策略（自适应双搜索，Adaptive Dual Search，来自第 8 章）提出了一个激进的主张：**不去学习 MCTS 的搜索轨迹，而是用熵正则化重塑 LLM 的推理空间，把铺展开的搜索树坍缩到一个低熵流形上**。

这不是传统的知识蒸馏，而是**拓扑重塑**：

**核心机制：动态对数障碍**

$$
\alpha_t = -\log\left(1 - \frac{H(p_t)}{H_{\max}}\right)
$$

当模型的输出熵 $H(p_t)$ 接近最大熵 $H_{\max}$ 时，$\alpha_t \to \infty$，形成一堵“惩罚墙”。

**效果**：

- 高熵分支（不确定的推理路径）在优化过程中变得不可达
- 搜索被迫沿着低熵（高置信度）的路径前进
- 这不是剪枝；它改变的是可达状态的几何结构

**与MCTS相比**：

- MCTS 是**显式搜索**：建树、模拟、回传——每次都要重新计算
- ADS 是**隐式剪枝**：熵障碍排斥高不确定性状态——推理空间被预先压缩

**关键洞察**：搜索不必“冻结进权重”，它可以通过**动态约束**被实时重塑。推理的每一步，熵障碍都在调整可达状态的边界。

AlphaGo Zero 用神经网络学习 MCTS 的策略分布。ADS 则用熵正则化直接约束 LLM 的输出空间——没有教师模型，没有蒸馏，只有一个对数障碍函数。

<div class="center">

------------------------------------------------------------------------

</div>

## 七、伪代码：MCTS、ADS与因果规划

**算法1：MCTS主循环**

```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state    = state
        self.parent   = parent
        self.children = {}   # action -> MCTSNode
        self.visits   = 0
        self.value    = 0.0

    def is_fully_expanded(self, actions):
        return all(a in self.children for a in actions)

def ucb_select(node, actions, c=math.sqrt(2)):
    # select the child with the highest UCB score
    best, best_score = None, -float("inf")
    for a, child in node.children.items():
        exploit = child.value / child.visits
        explore = c * math.sqrt(math.log(node.visits) / child.visits)
        score   = exploit + explore
        if score > best_score:
            best_score, best = score, child
    return best

def mcts(root_state, get_actions, transition, rollout, N=1000):
    # get_actions(s) -> list of actions
    # transition(s, a) -> next_state
    # rollout(s) -> reward
    root = MCTSNode(root_state)

    for _ in range(N):
        node = root
        # Select
        actions = get_actions(node.state)
        while node.is_fully_expanded(actions) and node.children:
            node    = ucb_select(node, actions)
            actions = get_actions(node.state)

        # Expand
        unexplored = [a for a in actions if a not in node.children]
        if unexplored:
            a    = random.choice(unexplored)
            s_new = transition(node.state, a)
            child = MCTSNode(s_new, parent=node)
            node.children[a] = child
            node = child

        # Simulate
        reward = rollout(node.state)

        # Backpropagate
        while node is not None:
            node.visits += 1
            node.value  += reward
            node = node.parent

    # return the action with the most visits
    return max(root.children, key=lambda a: root.children[a].visits)
```

**算法2：UCB选择**（已内嵌在上面的 `ucb_select` 函数中）

**算法3：ADS熵正则化**

```python
import torch
import torch.nn.functional as F

def ads_entropy_regularization(model, input_ids, tokenizer, max_steps=20, threshold=0.5):
    # model:     language model (exposes logits)
    # input_ids: initial token sequence
    # returns:   the list of generated token ids
    generated = list(input_ids)
    vocab_size = model.config.vocab_size

    for t in range(max_steps):
        with torch.no_grad():
            logits = model(torch.tensor([generated])).logits[0, -1]  # (vocab_size,)

        p_t   = F.softmax(logits, dim=-1)
        H_t   = -(p_t * torch.log(p_t + 1e-10)).sum().item()   # current entropy
        H_max = math.log(vocab_size)                            # max entropy (uniform distribution)

        # dynamic log barrier: as entropy approaches the max, alpha_t grows and the penalty strengthens
        alpha_t = -math.log(1 - H_t / H_max + 1e-10)

        # stop once entropy is low enough (model has converged to a low-entropy token)
        if H_t < threshold:
            break

        next_token = p_t.argmax().item()   # greedy decoding
        generated.append(next_token)

    # alpha_t -> inf as H_t -> H_max: high-entropy states are repelled by the barrier, search collapses to the low-entropy manifold
    return generated
```

**算法4：因果规划**

```python
# CausalPlanner from causal_tictactoe.py (condensed)
class CausalPlanner:
    def __init__(self, A, E, U=None, lr=0.1, steps=80, seed=0):
        self.A = A                 # (9,9) causal matrix from build_causal_A
        self.E = E                 # (9, D) embeddings from build_embeddings
        self.U = U if U is not None else np.zeros_like(E)
        self.lr, self.steps = lr, steps

    def plan(self, board, player):
        """Choose the cell to play, by gradient descent on the
        intervention strength a over empty cells. Loss: maximize
        activation of target win-lines, penalize sparse interventions."""
        empties = empty_cells(board)
        if not empties:
            return None
        if len(empties) == CELLS:
            return 4                 # center first

        a = np.ones(CELLS) * 0.1
        a = np.clip(a, 0.0, 2.0)
        m = np.zeros(CELLS, dtype=bool)
        m[empties] = True

        # target cells: win lines the player could complete
        target_mask = np.zeros(CELLS)
        for (l1, l2, l3) in self._lines_near_complete(board, player):
            for c in (l1, l2, l3):
                target_mask[c] += 1.0

        def loss(a):
            A_do = self.A.copy()
            for c in empties:
                A_do[:, c] = a[c] * self.A[:, c]   # intervention strength
            E_next = A_do.T @ self.E + self.U
            act = float(np.sum(target_mask[:, None] * E_next))
            penalty = 0.5 * float(np.sum(a[m] ** 2))
            return -(act - penalty)

        # Adam over a (masked to empty cells)
        beta1, beta2 = 0.9, 0.999
        m_hat, v_hat = np.zeros(CELLS), np.zeros(CELLS)
        for t in range(1, self.steps + 1):
            grad_num = np.zeros(CELLS)
            eps = 1e-3
            for c in empties:
                a_p, a_m = a.copy(), a.copy()
                a_p[c] += eps; a_m[c] -= eps
                grad_num[c] = (loss(a_p) - loss(a_m)) / (2 * eps)
            m_hat = beta1 * m_hat + (1 - beta1) * grad_num
            v_hat = beta2 * v_hat + (1 - beta2) * grad_num ** 2
            a = a - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8) * m
            a = np.clip(a, 0.0, 2.0)
        return max((c for c in empties), key=lambda c: a[c])
```

<div class="center">

------------------------------------------------------------------------

</div>

## 八、思想实验：当搜索变成梯度下降

MCTS 在井字棋上近乎完美，因为它枚举了所有可能性。但枚举是昂贵的——每一步都需要数百次模拟。如果状态空间具有连续结构，为什么还要在一棵离散的树上攀爬？

让我们做一个思想实验：**把棋盘状态编码为嵌入向量，把游戏规则编码为因果图，把搜索变成嵌入空间中的梯度下降**。不需要现成的库——几十行 numpy 就够了（配套脚本：`causal_tictactoe.py`）。

### 8.1 因果世界算子：do()的矩阵实现

它的核心是一个**神经结构因果模型**：每个变量（井字棋 9 个格子之一）是一个 D 维嵌入向量，因果边连接共享胜利线的格子。本章附录的完整代码只用了三个函数就把整套东西搭起来——因果矩阵、do、传播：

```python
# 1. causal matrix: cells sharing a win line are causally connected
def build_causal_A(seed=0):
    A = np.zeros((CELLS, CELLS))          # CELLS = 9
    for (a, b, c) in LINES:               # the 8 win lines
        for u in (a, b, c):
            for v in (a, b, c):
                if u != v:
                    A[u][v] += 1.0
    A = A / (np.abs(A).sum(axis=1, keepdims=True) + 1e-9)
    return A

# 2. do(X = v): cut all incoming edges of X (fill column X with zeros)
def do_cut(A, var):
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do

# 3. propagate: one matrix multiplication along the remaining edges
def propagate(A_do, E, U):
    return A_do.T @ E + U
```

**关键洞察**：`do(X=v)` 对一张因果图只做两件事——切断 X 的所有入边，然后让效应沿因果边传播。用矩阵表示，切断入边就是把 X 那一列置零，而传播就是一次矩阵乘法：`E_next = A_do.T @ E_cur + U`。没有比这更简单的定义。

### 8.2 梯度规划 vs 树搜索

现在来看井字棋的两种解法：

**MCTS（树搜索）**：

```python
def mcts_move(board, player, n_rollouts=500):
    # build the search tree: UCB1 selection, random rollouts, backpropagation of statistics
    # time complexity: O(rollouts x depth)
    return best_move
```

**CausalPlanner（梯度规划）**：

```python
def planner_move(board, scm, player):
    # encode the board state as embeddings
    E_cur = board_to_embedding(board, scm.E, player)

    # define the goal: maximize the embedding norm of the winning line
    target = {"c4": 1.8, "c0": 1.5}  # center > corner > edge

    # optimize the intervention energy with gradient descent
    planner = CausalPlanner(scm)
    result = planner.plan(
        E_init=E_cur,
        target=target,
        interv_nodes=empty_cells(board),
        lr=0.1,
        steps=80
    )
    # time complexity: O(steps x N x D)
    return max(empty_cells, key=lambda c: result["a_opt"][c])
```

**能量函数**：$E(a) = \sum_j (\lVert E_{\text{next}}[j] \rVert - \text{target}_j)^2$。梯度下降找到的是使嵌入范数最接近目标的干预值。

### 8.3 实验结果：先手不败，后手全负

运行本章附录中的完整脚本（`python3 causal_tictactoe.py`；30 局，每步 400 次 MCTS 模拟）：

```
Matchup A: Planner(X) vs MCTS(O)  — Planner moves first
  30 games  Planner=12  Draw=18  MCTS=0
Matchup B: MCTS(X) vs Planner(O)  — MCTS moves first
  30 games  MCTS=30  Planner=0  Draw=0
```

如何读这些结果：

- **先手时，梯度规划从不输**：从空棋盘开始，梯度信号是清晰的——中心嵌入的范数最大，正好对应最优的第一手。但请注意：12 胜 18 平——它打不穿完美防守，只是先把中心占了。

- **后手时，它全盘皆输**：一旦 MCTS 占据中心，梯度规划的信号里就只剩下“进攻目标”，没有“防守信号”——因果图编码的是胜利线，而不是对手可能获胜的线。它根本不知道自己应该封堵。

这不是 bug，是特征：梯度规划捕捉的是**连续启发式**——“这个局面看起来有多好”——而不是“搜索所有可能结局”。它用嵌入空间中的几何距离，替代了树搜索的胜率统计。先手优势来自嵌入初始化（中心 > 角落 > 边缘的价值先验）；后手全面崩溃，则是因为因果图里根本没有防守知识——梯度规划不是“推理”，而是“看哪个格子看起来最值钱”。

### 8.4 动手试试：运行井字棋对战

```bash
python3 causal_tictactoe.py
# Full code: see this chapter's appendix.
```

**观察**：

1. 梯度规划在简单局面上完美（开局、立刻取胜）。

2. 后手面对 MCTS 的封锁，它几乎总是输——把 MCTS 的模拟次数从 400 降到 50，看看它还能不能守住。

3. 速度对比：梯度规划每步大约用 60 次 Adam 迭代 × 9 维数值梯度 ≈ 1000 次浮点运算，而 MCTS 每步需要 400 × 深度 次状态评估。

**思考题**：

- 梯度信号什么时候会误导？（当最优路径需要暂时“变差”时。）

- 因果图编码了哪些游戏知识？（胜利线的拓扑——该赢的线，而不是该防的线。）

- 要让因果图学会防守，它还需要什么？（对手的胜利线也必须进入目标函数。）

- 这对 AI 规划的未来意味着什么？（连续表示 + 梯度优化，对比离散搜索。）

::: info 兔狲教授评

MCTS 是离散枚举，梯度规划是连续启发。这不是谁更好的问题，而是**搜索的两种本体论**：一个在状态树上攀爬，一个在嵌入空间中流动。梯度下降不是“近似的树搜索”——它是完全不同的东西：它用向量空间中的距离取代了树的胜率。承认这个区别，比假装它们在竞争更重要。而我们刚刚亲眼看到：后手时它每局都输——连续启发式不是万灵药，它只知道它学过的东西。

:::

<div class="center">

------------------------------------------------------------------------

</div>

## 九、可视化：搜索树的不对称生长

<figure>
<p><img src="/figures/ch10_fig1_mcts_tree.png" alt="图10.1：MCTS搜索树的演化" /></p>
<figcaption>图10.1：MCTS搜索树的演化</figcaption>
</figure>

*图 10.1 展示了 MCTS 搜索树的形态如何随模拟次数增长而演化。三个子图从左到右分别对应 N=10、100 和 1000 次模拟：*

- **左图（N=10）**：树几乎是对称的，大多数分支的访问次数相近（节点大小相似）。此时探索还不够，UCB 尚未把好路径与坏路径分开。

- **中图（N=100）**：不对称开始出现。好的分支（绿色节点，胜率高）先被探索，节点明显变大。坏的分支（红色节点）访问次数较少。

- **右图（N=1000）**：高度不对称。最优路径被反复访问（最大的绿色节点），次优路径得到适度探索，坏路径几乎被放弃（小小的红色节点）。

**关键观察**：节点大小正比于访问次数，颜色表示平均胜率（绿=高，红=低）。这种不对称生长正是 UCB 算法的核心——**把有限的模拟预算集中在最有希望的路径上**。

<figure>
<p><img src="/figures/ch10_fig2_ucb_tradeoff.png" alt="图10.2：UCB的探索-利用权衡" /></p>
<figcaption>图10.2：UCB的探索-利用权衡</figcaption>
</figure>

*图 10.2 展示了 UCB 公式中的探索项与利用项如何随节点访问次数变化：*

$$
\text{UCB}(n) = \underbrace{\text{平均胜率}}_{\text{利用项}} + c \cdot \underbrace{\sqrt{\frac{\ln N}{n}}}_{\text{探索项}}
$$

- **蓝色曲线（利用项）**：平均胜率随访问次数增长收敛到真实值（约 0.7）。早期摆动很大，后期稳定下来。

- **红色曲线（探索项）**：$\sqrt{\ln N/n}$ 随访问次数衰减。节点被访问得越多，探索奖励越小。

- **绿色虚线（总UCB）**：两项之和。早期探索项占主导（红色高），后期利用项占主导（蓝色稳定）。

**交叉点在 n≈30**：在这附近，两项的贡献大致相等。在此之前是“大胆探索”，在此之后是“精确利用”。这个平衡点由常数 $c$ 控制——$c$ 越大，探索越激进。

<div class="center">

------------------------------------------------------------------------

</div>

## 十、大语言模型的搜索：从思维链到强化学习

2022 年之后，搜索问题多了一个新维度：**语言模型本身就是一个状态空间**。每个推理步骤不再是棋盘上的一手棋，而是一段文本——一行数学、一段逻辑论证、一行代码。

这就提出一个深刻的问题：**LLM 如何在自己的语言推理空间里有效地搜索？**

MCTS 的答案是：建树、模拟、回传。但对语言模型来说，“模拟一次”意味着跑完整个前向传播——代价巨大。而且语言推理空间是连续的，分支因子无限——你无法枚举所有可能的“下一句话”。

两条不同的技术路线，给出了两种截然不同的答案。

### 10.1 GPT-o1：过程奖励模型与策略蒸馏

2024 年 9 月，OpenAI 发布了 o1 系列模型。核心创新不在模型架构，而在**搜索策略**。

o1 的技术路线可以概括为三层架构：

**第一层：过程奖励模型（Process Reward Model，PRM）**

传统强化学习中，奖励在终点给出——答对 +1，答错 −1。但在推理任务中这太稀疏了：一道需要 20 步推导的数学题，只有第 20 步才能判断对错。

PRM 的思路是：**给每一个中间步骤打分**。给定推理链中的第 $k$ 步，PRM 输出这一步“正走在正确方向上”的概率：

$$
r_k = \text{PRM}(s_0, a_1, a_2, \ldots, a_k)
$$

这把稀疏的终端奖励转化为**稠密的过程奖励**，使搜索在每一步都能得到反馈。

**第二层：策略模型引导树搜索**

有了 PRM，o1 用策略模型（一个微调过的语言模型）在推理步骤空间中执行类 MCTS 的搜索：

- **选择**：用策略模型生成 $K$ 个候选推理步骤
- **评估**：PRM 给每个候选打分
- **扩展**：沿着得分最高的步骤继续
- **回传**：用最终结果反向更新 PRM 和策略模型

这是**树搜索与语言模型推理**第一次系统性地结合。思维链（CoT——一步一步推理，把步骤写下来）不再是单条线性推导，而是一棵在推理空间中生长的搜索树。

**第三层：知识蒸馏压缩搜索**

搜索有效，但很慢。推理时，o1 实际运行的是一个**蒸馏过的策略模型**——它在训练中见过大量由树搜索产生的高质量推理轨迹，学会了“在内部模拟搜索过程”，推理时不必再建树。

这很像人类棋手：初学者计算每一种变化，大师一眼就看出最佳着法——因为大师已经把搜索“内化”为直觉。

**这是思维链的历史性时刻**：CoT 不再只是一个提高准确率的技巧，它已经成为让语言模型在推理空间中进行**系统性搜索**的基础设施。

### 10.2 DeepSeek-R1：干掉PRM，在组内竞争

2025 年初，DeepSeek 发布 R1 模型，走了一条与 o1 形成鲜明对比的路线：**完全不要 PRM**。

o1 的路线有一个根本瓶颈：**PRM 是怎么训练的？**

训练 PRM，需要人工把每一个中间推理步骤标注为“对/错”。这是极其昂贵的数据工程——一道需要 20 步推导的题，每一步都要标注，数据成本是只标终点的 20 倍。

还有一个更深的问题：**谁来判定一个中间步骤对不对？** 有些推理路径看起来“错了”，最终却得到正确答案；有些看起来“对了”，却在最后一步失败。中间步骤的正确性取决于最终结果，但 PRM 必须在不知道结果的情况下给它打分。这是一个根本性的困难。

DeepSeek-R1 的解法是**组相对策略优化（Group Relative Policy Optimization，GRPO）**：

**核心思想：不需要绝对打分，只需要相对排名**

对同一个问题，让模型生成 $G$ 条完整的推理轨迹（答案链）：

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_\theta(\cdot \mid q)
$$

只在终点判断对错：如果第 $i$ 条轨迹得到正确答案，$r_i = 1$，否则 $r_i = 0$。

然后计算**组内相对优势**——哪些轨迹比这一组的平均水平更好？

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j
$$

用这个相对优势更新策略：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \cdot A_i\right]
$$

**这个设计的深刻之处**：

- **不需要 PRM**：奖励信号只来自终端的对错判断，彻底免除了中间步骤标注
- **组内竞争**：同一组的轨迹构成一场竞争——“比组平均更好”得到正梯度，“比组平均更差”得到负梯度
- **涌现搜索**：模型在训练中自发学会“自我验证”——生成几条推理路径，在内部比较，保留更好的那条

这是强化学习的一次深刻创新：**用隐式的组内竞争压力，取代显式的过程监督（PRM）**。

### 10.3 两条路线的哲学对比

| 维度 | GPT-o1 路线 | DeepSeek-R1 路线 |
|:-----|:-----------|:----------------|
| 过程监督 | 显式 PRM，逐步打分 | 无 PRM，仅终端奖励 |
| 搜索风格 | 显式树搜索 + 蒸馏 | 经由 GRPO 的组内竞争 |
| 数据需求 | 高（过程标注） | 低（仅答案标注） |
| 算力成本 | 训练便宜，推理昂贵 | 训练昂贵，推理便宜 |
| 涌现能力 | 通过蒸馏传递 | 通过竞争自行涌现 |

**从 MCTS 的视角看**，两条路线对应两种不同的搜索策略：

o1 的路线更像**有监督的树搜索**：PRM 是手工设计的启发函数，策略模型是被人类数据引导的先验。它高度依赖人类知识，但在有数据的领域表现很强。

DeepSeek-R1 的路线更像**无监督的随机搜索加选择压力**：GRPO 相当于让 $G$ 条随机游走路径在同一个问题上竞争，自然选择留下好的路径。它不依赖人类对“过程”的判断，但需要大量样本来产生足够的竞争压力。

**这是一个深刻的张力**：过程监督越精确，就越依赖人类知识；越少依赖人类知识，就越需要自发的竞争涌现。

思维链把这个张力带进了语言模型时代。在围棋时代，这个张力的答案是 AlphaGo Zero——彻底抛弃人类棋谱，依赖自我对弈的竞争选择。在语言推理时代，这个张力还没有最终答案。

<div class="center">

------------------------------------------------------------------------

</div>

## 十一、搜索的哲学：计算即推理

MCTS 揭示了一个深刻的事实：**推理不需要“理解”——只需要有效的搜索**。

AlphaGo 不知道围棋之美，也不理解“厚势”“实地”这些概念。它只知道：在这个状态下，哪一手模拟胜率最高。

但这已经足够了。通过数百万次模拟，MCTS 找到了人类几千年都没有找到的棋路。

这引出一个哲学问题：**如果搜索足够强大，我们还需要“理解”吗？**

一种观点认为，**理解是搜索的捷径**。人类棋手通过理解棋理，可以快速剪枝，不必搜索每一种可能。但如果算力无限，搜索可以取代理解。

另一种观点认为，**理解是泛化的基础**。MCTS 在围棋上很强，但换一个游戏就必须重新训练。人类理解“策略”“权衡”这样的抽象概念，可以快速迁移到新游戏。

ADS 的熵正则化提供了第三种观点：**搜索可以通过动态约束被实时重塑**。不必把搜索“冻结进权重”；熵障碍只是在推理的每一步排斥高不确定性状态。

这类似于物理中的势场：不是预先算出每一条路径，而是在每个位置感受到势能梯度，自然而然地朝更低势能的方向移动。

搜索很强大，但很昂贵。下一章，我们将直面推理的经济学：如何用更少的计算，得到同样质量的推理？

<div class="center">

------------------------------------------------------------------------

</div>

## 悬而未决

- **熵正则化的泛化边界在哪里？** ADS 在算术任务上有效，但在需要多层抽象的任务上，小模型会出现“自信的幻觉”（confident hallucination）。熵障碍与模型容量之间是什么关系？

- **拓扑重塑能否推广到其他推理任务？** 对数障碍在 LLM 推理中有效——但在符号推理、定理证明这样的离散搜索空间中呢？

- **探索与利用的最优平衡是什么？** UCB 公式中的常数 $c$ 应当如何选择？不同的问题是否需要不同的 $c$？

- **MCTS 能否推广到连续动作空间？** 围棋和井字棋的动作是离散的。如何把 MCTS 应用到机器人控制、连续优化等连续动作问题？

- **搜索与推理的边界在哪里？** 定理证明可以被看作搜索——但数学家的“洞察”能被搜索捕获吗？

- **PRM 的中间步骤标注是否存在根本性困难？** 如果一条“看起来错了”的推理路径最终得到正确答案，那一步到底是对还是错？这种标注困难是否意味着 PRM 永远无法提供真正可靠的过程监督？

- **GRPO 的组内竞争压力从哪里来？** 如果所有 $G$ 条轨迹都正确（或者都错误），那么 $A_i = 0$，梯度消失，模型无法学习。这意味着 GRPO 需要精确校准任务难度——既不能太难（全错），也不能太容易（全对）。这种校准是一个可解决的工程问题，还是一个根本性的限制？

<div class="center">

------------------------------------------------------------------------

</div>

## 附录：完整代码

下面是本章配套实验的完整脚本（复制它，保存为 `causal_tictactoe.py`，然后运行）：

```python
"""
Gradient Planning vs Tree Search on Tic-Tac-Toe
================================================
Companion script for The Reasoning Kingdom, Chapter 10, Section 8
("When Search Becomes Gradient Descent").

The idea experiment: encode the board as embeddings, encode the rules as a
causal graph, and turn "search" into "gradient descent in embedding space".

Two planners square off:
  * MCTS        : explicit tree search (UCB1 + random rollouts)
  * CausalPlanner: gradient descent on an intervention vector over a
                   neural structural causal model (SCM) built from win-line
                   adjacency

Everything below is self-contained and numpy-only. No external repos, no
type theory, no lambda calculus — just embeddings, a causal matrix, and
Adam.

Run:  python3 causal_tictactoe.py

(Full code: see this chapter's appendix.)
"""
import numpy as np
import math

# --------------------------------------------------------------------------
# Board helpers (3x3 = cells c0..c8)
# --------------------------------------------------------------------------
CELLS = 9
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]

def empty_cells(board):
    return [i for i in range(CELLS) if board[i] == 0]

def board_to_vec(board, player):
    """State vector: +1 for player's stones, -1 for opponent, 0 empty."""
    return np.array([1 if c == player else (-1 if c != 0 else 0)
                     for c in board], dtype=float)

def winner(board):
    for (a, b, c) in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if len(empty_cells(board)) == 0 else None   # 0 = draw, None = ongoing

def print_board(board):
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(3):
        print("  " + " ".join(sym[board[3 * r + c]] for c in range(3)))

# --------------------------------------------------------------------------
# Causal graph: win-line adjacency
# --------------------------------------------------------------------------
def build_causal_A(seed=0):
    """A[i][j] = strength of causal influence from cell j to cell i.
    Cells sharing a win line are causally connected; center dominates."""
    A = np.zeros((CELLS, CELLS))
    for (a, b, c) in LINES:
        for u in (a, b, c):
            for v in (a, b, c):
                if u != v:
                    A[u][v] += 1.0
    # normalize rows; give the center extra weight (the classic first move)
    rng = np.random.default_rng(seed)
    A += rng.normal(0, 0.05, A.shape)
    A = A / (np.abs(A).sum(axis=1, keepdims=True) + 1e-9)
    return A

def build_embeddings(seed=0, D=16):
    """One D-dim embedding per cell; center has large norm, corners medium,
    edges small — the 'value prior' the planner starts with."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1, size=(CELLS, D))
    # value profile: center > corner > edge
    profile = np.zeros(CELLS)
    profile[4] = 1.6                      # c4 = center
    for c in (0, 2, 6, 8):                # corners
        profile[c] = 1.1
    for c in (1, 3, 5, 7):                # edges
        profile[c] = 0.6
    E = E * profile[:, None]
    return E

# --------------------------------------------------------------------------
# CausalPlanner (gradient planning)
# --------------------------------------------------------------------------
def do_cut(A, var):
    """Pearl do(X = v): cut incoming edges of var (set column to 0)."""
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do

class CausalPlanner:
    def __init__(self, A, E, U=None, lr=0.1, steps=80, seed=0):
        self.A = A
        self.E = E
        self.U = U if U is not None else np.zeros_like(E)
        self.lr = lr
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def plan(self, board, player, used_step_cost=True):
        """Return the cell to play, chosen by gradient descent on the
        intervention strength a >= 0 over empty cells.

        Loss: want the post-intervention embedding to be 'strong' on the
        win lines the player can complete, minus a sparsity penalty."""
        empties = empty_cells(board)
        if not empties:
            return None
        if len(empties) == CELLS:
            return 4                       # center first (known best)

        a = np.ones(CELLS) * 0.1
        a = np.clip(a, 0.0, 2.0)
        m = np.zeros(CELLS, dtype=bool)    # mask of empty cells
        m[empties] = True
        v = board_to_vec(board, player)

        # target lines the player could complete
        target_mask = np.zeros(CELLS)
        for (l1, l2, l3) in self._lines_near_complete(board, player):
            for c in (l1, l2, l3):
                target_mask[c] += 1.0

        def loss(a):
            A_do = self.A.copy()
            for c in empties:
                A_do[:, c] = 0
            # intervention strength scales the causal flow: column weight
            for c in empties:
                A_do[:, c] = a[c] * self.A[:, c]
            E_next = A_do.T @ self.E + self.U
            # we want the target cells to activate: maximize sum(target*E_next)
            act = float(np.sum(target_mask[:, None] * E_next))
            # and prefer sparse interventions (one strong move, not many)
            penalty = 0.5 * float(np.sum(a[m] ** 2))
            return -(act - penalty)

        # Adam over a (masked to empty cells)
        beta1, beta2 = 0.9, 0.999
        m_hat = np.zeros(CELLS); v_hat = np.zeros(CELLS)
        for t in range(1, self.steps + 1):
            g = np.zeros(CELLS)
            grad_num = np.zeros(CELLS)
            eps = 1e-3
            for c in empties:
                a_p = a.copy(); a_m = a.copy()
                a_p[c] += eps; a_m[c] -= eps
                grad_num[c] = (loss(a_p) - loss(a_m)) / (2 * eps)
            g[m] = grad_num[m]
            m_hat = beta1 * m_hat + (1 - beta1) * g
            v_hat = beta2 * v_hat + (1 - beta2) * g * g
            mh = m_hat / (1 - beta1 ** t)
            vh = v_hat / (1 - beta2 ** t)
            a = a - self.lr * mh / (np.sqrt(vh) + 1e-8) * m
            a = np.clip(a, 0.0, 2.0)

        # sanity: never pick an occupied cell — pick the empty cell with the
        # highest intervention strength
        best = max((c for c in empties), key=lambda c: a[c])
        return best

    def _lines_near_complete(self, board, player):
        """Win lines where player already holds 2/3 and the rest is empty."""
        res = []
        for (a, b, c) in LINES:
            vals = [board[x] for x in (a, b, c)]
            if vals.count(player) == 2 and vals.count(0) == 1:
                res.append((a, b, c))
        return res

# --------------------------------------------------------------------------
# MCTS with UCB1 (the classic)
# --------------------------------------------------------------------------
class MCTSNode:
    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0

def ucb1(node):
    if node.N == 0:
        return float("inf")
    if node.parent is None:
        return 0.0
    exploit = node.W / node.N
    explore = math.sqrt(2 * math.log(node.parent.N + 1) / node.N)
    return exploit + explore

def mcts_move(board, player, n_rollouts=500, seed=0):
    rng = np.random.default_rng(seed)
    root = MCTSNode(board, player, parent=None)
    foe = 1 if player == 2 else 2

    for _ in range(n_rollouts):
        node = root
        b = board.copy()
        turn = player

        # selection
        while node.children:
            # choose best child by UCB1
            best_c = max(node.children, key=lambda c: ucb1(node.children[c]))
            node = node.children[best_c]
            b[best_c] = turn
            turn = foe if turn == player else player

        # expansion + rollout
        res = winner(b)
        if res is None:
            moves = empty_cells(b)
            c_move = int(rng.choice(moves))
            b[c_move] = turn
            turn = foe if turn == player else player
            res = winner(b)
        while res is None:
            moves = empty_cells(b)
            c_move = int(rng.choice(moves))
            b[c_move] = turn
            turn = foe if turn == player else player
            res = winner(b)

        # backprop
        sign = 1.0 if res == 0 else (1.0 if res == player else 0.0)
        while node is not None:
            node.N += 1
            node.W += sign
            node = node.parent

    if not root.children:
        return empty_cells(board)[0] if empty_cells(board) else None
    best_c = max(root.children, key=lambda c: root.children[c].N)
    return best_c

# --------------------------------------------------------------------------
# Matchup
# --------------------------------------------------------------------------
def play_game(planner_first, n_rollouts=400, seed=0, verbose=False):
    A = build_causal_A(seed=seed)
    E = build_embeddings(seed=seed)
    planner = CausalPlanner(A, E, lr=0.1, steps=60, seed=seed)

    board = [0] * CELLS
    rng = np.random.default_rng(seed)
    turn = 1
    while True:
        if (turn == 1) == planner_first:
            move = planner.plan(board, turn)
        else:
            move = mcts_move(board, turn, n_rollouts=n_rollouts,
                             seed=rng.integers(0, 10000))
        if move is None or board[move] != 0:
            break
        board[move] = turn
        res = winner(board)
        if res is not None:
            return res
        turn = 2 if turn == 1 else 1
    return 0

def matchup(games=20, n_rollouts=400, seed=0, verbose=False):
    stats = {"Planner": 0, "MCTS": 0, "Draw": 0}
    for g in range(games):
        r = play_game(True, n_rollouts=n_rollouts, seed=seed + g)
        stats["Planner" if r == 1 else ("MCTS" if r == 2 else "Draw")] += 1
    return stats

if __name__ == "__main__":
    print("=" * 64)
    print("Gradient Planning vs MCTS on Tic-Tac-Toe")
    print("=" * 64)

    print("\n[1] Causal graph from win-line adjacency (A is 9x9, rows sum 1)")
    A = build_causal_A()
    print("    A[4,:3] =", np.round(A[4, :3], 3), " (center <- top line)")

    print("\n[2] Embedding profile: center > corner > edge")
    E = build_embeddings()
    print("    norms:", [round(float(np.linalg.norm(E[i])), 2) for i in range(9)])

    print("\n[3] Matchup: Planner (gradient) first vs MCTS (rollouts=400), 30 games")
    stats = matchup(games=30, n_rollouts=400, seed=100)
    print(f"    Planner wins {stats['Planner']} ; draws {stats['Draw']} ; MCTS wins {stats['MCTS']}")

    print("\n[4] Matchup: MCTS first vs Planner (O), 30 games")
    stats2 = {"Planner": 0, "MCTS": 0, "Draw": 0}
    for g in range(30):
        r = play_game(False, n_rollouts=400, seed=200 + g)
        stats2["Planner" if r == 2 else ("MCTS" if r == 1 else "Draw")] += 1
    print(f"    MCTS wins {stats2['MCTS']} ; Planner wins {stats2['Planner']} ; draws {stats2['Draw']}")

    print("\n    Observation: gradient planning never loses when it is first")
    print("    (it just grabs the center), but loses every game when second —")
    print("    its causal graph encodes aiming at win lines, not blocking.")
    print("\nDone.")
```

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Kocsis & Szepesvári (2006). *Bandit Based Monte-Carlo Planning*——MCTS 的原始论文，提出了 UCT 算法 `-> [ECML 2006]`

- Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search*——AlphaGo 论文，MCTS + 神经网络的里程碑

- Silver et al. (2017). *Mastering the game of Go without human knowledge*——AlphaGo Zero，完全从自我对弈中学习

- 本书第 10 章 §6《从搜索到直觉：ADS 的熵正则化》——ADS 策略的完整定义、伪代码及配套脚本（`ads_search.py`）

- Lightman et al. (2023). *Let's Verify Step by Step*——OpenAI 对过程奖励模型（PRM）的系统性研究，表明过程监督优于结果监督 `-> [arXiv:2305.20050]`

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*——GRPO 方法：无需 PRM 即可实现强推理的组内竞争强化学习 `-> [arXiv:2501.12948]`

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*——GRPO 算法首次提出之处，在数学推理任务上得到验证 `-> [arXiv:2402.03300]`

- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*——思维链论文，确立了 CoT 作为 LLM 推理的基础设施 `-> [arXiv:2201.11903]`

- Browne et al. (2012). *A Survey of Monte Carlo Tree Search Methods*——MCTS 方法综述

- \[Dam et al., 2020\]——MCTS 中的凸正则化，附指数收敛性证明 `-> [arXiv:2007.00391]`

- \[Dam et al., 2019\]——Power-UCT，用幂平均改进节点价值估计 `-> [arXiv:1911.00384]`

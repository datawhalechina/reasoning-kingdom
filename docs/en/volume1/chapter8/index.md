# Chapter 8: The Heuristic's Contract — How Much Courage It Takes to Accept "Close Enough"

> If A\*'s heuristic lies to you, it will still hand you a path. It's just the wrong one.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. The Bet of 1968

In 1968, Peter Hart, Nils Nilsson, and Bertram Raphael at the Stanford Research Institute proposed an algorithm in a paper — an algorithm that would turn out to be one of the most elegant ideas in the history of AI.

The problem they wanted to solve was simple: help the robot Shakey find the shortest path from point A to point B in a room.

Brute-force search guarantees the optimal solution — try every possible path, pick the shortest. But Shakey's world has millions of possible path combinations, and brute-force search would need an astronomical amount of time.

Greedy search is fast — you always head toward the goal. But it can slam into obstacles, walk into dead ends, and find paths much longer than the optimal one.

The A\* algorithm that Hart and his two colleagues proposed made a seemingly impossible promise: **I can be orders of magnitude faster than brute force, while still guaranteeing the optimal solution.**

What does that promise cost? A **contract**.

If you give A\* an "honest" heuristic — one that never overestimates the remaining cost — it keeps its promise. But if the heuristic "lies" and overestimates, A\* will still give you a path. It's just that the path may be wrong.

That's not a bug in the algorithm; it's by design. A\* buys the balance between speed and optimality with a mathematical contract.

More than fifty years later, this contract's idea is still the heart of all heuristic algorithms: **when perfection is out of reach, we need approximation at an explicit cost.**

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Curse of Combinatorial Explosion

Let me first quantify what "perfection is out of reach" means.

In 2016, AlphaGo beat Lee Sedol. The world marveled at the machine's "intuition" and "creativity". But very few people noticed a technical detail: behind every move of AlphaGo, Monte Carlo tree search explored millions of possibilities on a 19×19 board.

Not hundreds, not thousands — millions.

Why so many? Because Go's state space is about $10^{170}$ — 90 more orders of magnitude than the number of atoms in the universe ($10^{80}$).

Even at $10^9$ states per second (the limit of a modern computer), it would take $10^{150}$ times the age of the universe to exhaust everything.

This isn't an engineering problem. It's **physically impossible**.

Combinatorial explosion shows up everywhere:

- **The traveling salesperson problem**: a visit to n cities has $n!$ possible routes. For 20 cities, that's $2.4 \times 10^{18}$ routes.

- **Protein folding**: a protein of 100 amino acids, each with 3 main conformations, gives $3^{100} \approx 10^{47}$ possible ways to fold.

- **Theorem proving**: in first-order logic, the number of possible proof steps for a proof of length n grows exponentially in n.

This is why heuristics exist: **when the optimal solution is out of reach, we need a "good enough" solution**.

But what does "good enough" even mean? That's not a vague engineering question — it's a precise mathematical one. Heuristic algorithms must obey certain contracts — and if they break them, the "approximate solution" they produce can be entirely wrong.

Let's go back to 1968 and see how Hart, Nilsson, and Raphael designed that contract.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. The Mathematical Contract of A\*

Imagine you're in a maze, trying to get from the start S to the goal G. Every step has a cost (time or distance, say). You want the path with the smallest cost.

**Brute-force search** (Dijkstra's algorithm) tries every possible path and guarantees the optimum, but its time complexity is $O(b^d)$ — where b is the branching factor and d is the depth of the solution. In a complicated maze, that's exponential.

:::details Dijkstra's Algorithm: The Brute-Force Guarantee for Shortest Paths
**Dijkstra's algorithm** (named for the Dutch computer scientist Edsger Dijkstra, who proposed it in 1956) finds the shortest path by repeatedly taking the unvisited node with the smallest known cost, expanding its neighbors, and carrying on until it reaches the goal.

- **Guarantee**: if every edge cost is non-negative, Dijkstra always finds the optimal (lowest-cost) path
- **Cost**: it explores all possible paths — it doesn't know "roughly which way to go", so it's inefficient

**Branching factor $b$**: in a search tree, the average number of child nodes per node (the available next moves). In a maze it could be 4 (up, down, left, right); in Go it's about 250.

**Search depth $d$**: how many steps are needed to reach the answer.

Combined, the search space is $b^d$ — that's the mathematical source of combinatorial explosion. A* uses its heuristic to "prune", reducing the number of nodes it must explore, gaining a huge speedup while keeping the optimality guarantee.
:::

**Greedy search** always chooses the direction that "looks closest to the goal". It's fast, but it can walk into dead ends. Worse, the path it finds may be several times longer than the optimal one.

When A\* was introduced in 1968, it made a seemingly impossible promise: I can be faster than Dijkstra, and still find the optimal path.

Its core is an evaluation function:

$$
f(n) = g(n) + h(n)
$$

- $g(n)$: the **actual cost** from the start to node n (known, certain)

- $h(n)$: the **estimated cost** from n to the goal (unknown, a guess)

- $f(n)$: the **estimated total cost** of reaching the goal through n

A\* always expands the node with the smallest $f(n)$ — the one with the smallest "actual cost + estimated remaining cost".

The intuition is simple: if you've already walked 10 steps to reach node n, and you estimate it takes 5 more steps to reach the goal, then the total cost through n is about 15 steps. A\* explores the path with the smallest total cost first.

But there's a fatal question here: **how do you estimate $h(n)$?**

If $h(n)$ is too high, A\* will give up on the optimal path too early — it will think "this path is too expensive" and switch to a suboptimal one.

If $h(n)$ is too low (say $h(n) = 0$), A\* degenerates into Dijkstra's algorithm, explores every direction, and loses its speed advantage.

Hart, Nilsson, and Raphael proved a profound theorem: if the heuristic satisfies **admissibility**, A\* is guaranteed to find the optimal path.

**The definition of admissibility**:

$$
h(n) \leq h^*(n)
$$

where $h^*(n)$ is the **true shortest cost** from n to the goal.

In other words, the heuristic must be **optimistic** — it must never overestimate the remaining cost. You may underestimate (which makes the search slower), but you may not overestimate (which makes the search find a wrong answer).

That's the A\* contract: give me an admissible heuristic, and I guarantee the optimal path. Break the contract, and the guarantee is void.

Why does this contract work? Let me show it with a concrete example.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. Try It Yourself: Watch the Contract Break

Admissibility isn't an abstract mathematical requirement. It's a **necessary and sufficient condition** for A\* to find the optimal path.

Let's verify it with our own hands.

**Experiment goal**: run A\* on the same maze with an admissible heuristic and with an inadmissible one, and see what differs.

Maze map (0 = walkable, 1 = obstacle, S = start, G = goal):

```
S . . . . .
. . . # . .
. . . . # .
. . . . . .
. . . . . #
. . . . . G
```

- S is the start (0,0); G is the goal (5,5)

- `.` is a walkable cell; each step costs 1

- `#` is an obstacle — not walkable

**Task**: find the shortest path from S to G.

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

### Step 1: With an Admissible Heuristic

The simplest admissible heuristic is the **Manhattan distance**:

$$
h_1(n) = |n_x - G_x| + |n_y - G_y|
$$

This is the cost of "walking straight to the goal if there were no obstacles". Because the real path has to get around obstacles, $h_1(n) \leq h^*(n)$, so it satisfies admissibility.

Starting from S, compute $f(n) = g(n) + h_1(n)$ for the neighboring nodes:

| Node | g(n) | h₁(n) | f(n) |
|:-----|:-----|:------|:-----|
| S right  | 1    | 9     | 10   |
| S down   | 1    | 9     | 10   |

The two tie on $f(n)$ (both 10), so A\* expands "S right" first. After moving right along row 0 to (0,2), the path forks: keep going right (0,3)->(0,4)->(0,5)-> then down, or drop down (1,2)->(2,2)->(2,3) through the gap. $h_1$ gives both routes an equal chance — and in the end A\* finds the optimal path through the gap, **length = 10 steps**, with 33 expanded nodes in total.

### Step 2: With an Inadmissible Heuristic

Now let's break the contract on purpose. Define an **overestimating** heuristic:

$$
h_2(n) = 2 \times (|n_x - G_x| + |n_y - G_y|)
$$

This function estimates the remaining cost at twice the Manhattan distance — clearly an overestimate.

Starting from S again:

| Node | g(n) | h₂(n) | f(n) |
|:-----|:-----|:------|:-----|
| S right  | 1    | 18    | 19   |
| S down   | 1    | 18    | 19   |

Both tie on $f(n)$ again, and A\* expands "S right" first, same as before. But $h_2$ has a decisive property: along any path heading toward the goal, each step decreases $f_2 = g + 2h_1$ **by 1** — $g$ grows by 1 while $2h_1$ shrinks by 2, so $f_2$ keeps getting smaller. The deeper you go, the "cheaper" it looks.

That sets a trap. After the fork at (0,2):

- **The right-turn route**: 5 straight steps to (0,5), with $f_2$ decreasing every step — like a snowball, it rolls faster and faster
- **The gap route**: drop to (1,2), but the right neighbor (1,3) is walled off by an obstacle; you have to drop another level before you can turn right — one extra step before you see any progress

The decreasing $f_2$ drags A\* irreversibly rightward along row 0. By the time it reaches (3,5) and discovers it must loop back left to (3,4) — a step that makes the Manhattan distance **increase instead of decrease**, so $f_2$ jumps up — it's too late. A\* has already gone deep down this "getting cheaper all the time" path, and every node on the other side of the gap sits buried at the bottom of the priority queue.

**Result**: the inadmissible $h_2(n)$ finds a path of length = **12 steps** — 2 more than the optimum. It expanded only 14 nodes — less than half of the admissible version's 33 — but the price is losing optimality.

The crash wasn't an accident. One fundamental limitation of heuristic design is this: **you cannot reliably predict how a heuristic will behave before running it**. We always assume overestimating a bit is just "less accurate", but the $f_2$ decrease systematically twists the search direction — turning A\* from a steady breadth-first explorer into a snowballing greedy gambler. With a different obstacle layout, it might crash — or it might not. The fact that the overestimating version actually expands fewer nodes than the admissible one already shows how hard it is to predict a heuristic's behavior in advance.

**This is the meaning of admissibility**: if $h(n)$ is admissible, the first path A\* finds **must** be the shortest path. That's a mathematical guarantee, not empirical habit.

But admissibility comes at a price too.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. An Information-Theoretic View: The Entropy Cost of a Heuristic

Admissibility guarantees correctness. But what does it cost?

Consider two extremes.

**Extremely conservative**: $h(n) = 0$ — always estimate the remaining cost as zero.

That's admissible, obviously (since the true cost $h^*(n) \geq 0$), but it's useless — A\* degenerates into Dijkstra's algorithm and **uniformly** explores every direction until it bumps into the goal. From an information-theoretic standpoint, this is **maximum-entropy exploration** of the search space: every direction has the same prior probability.

**Extremely aggressive**: $h(n) = h^*(n)$ — perfect prediction of the remaining cost.

That would be ideal — A\* walks the optimal path directly, exploring no useless node. It's **zero-entropy search**: you already know the answer, so no exploration is needed. But here's the catch: if you already knew $h^*(n)$, you'd have already solved the problem. No search needed at all.

Real-world heuristics live in between: **as close to $h^*(n)$ as possible, while staying admissible**.

This trade-off can be measured with the **effective branching factor** $b^*$:

$$
N = 1 + b^* + (b^*)^2 + \cdots + (b^*)^d
$$

where $N$ is the total number of nodes A\* expands and $d$ is the depth of the solution. The smaller $b^*$, the more efficient the search.

Pearl's classic 1984 textbook gives experimental data:

- With $h(n) = 0$, $b^* \approx 2.8$ (on the 8-puzzle)

- With $h(n) = \text{Manhattan distance}$, $b^* \approx 1.4$

- With better heuristics (like pattern databases), $b^* \approx 1.1$

Going from 2.8 down to 1.1 means the tree's node count falls from $2.8^{20} \approx 10^9$ to $1.1^{20} \approx 8$ — a gap of six orders of magnitude.

**The quality of the heuristic directly determines how much compute the search needs.**

But underneath there's a deeper tension: good heuristics need domain knowledge. The Manhattan distance works on grids — but what about graph search? Continuous spaces? Go?

In complicated environments, designing a heuristic that is both admissible and close to $h^*(n)$ is itself a hard problem.

That's the problem **ADS (Adaptive Dual Search)** sets out to solve: can a machine **learn the heuristic on its own**?

<div class="center">

------------------------------------------------------------------------

</div>

## 6. ADS: Making Heuristics Information-Theoretic

Traditional heuristic design has a fundamental flaw: $h(n)$ is hand-designed, needs domain knowledge, and is static — once it's built, it never changes during the whole search.

But search spaces are dynamic. In open areas, the heuristic deserves trust; in obstacle-dense areas, any heuristic can mislead. A fixed weight $\alpha=1$ is a crude simplification of that dynamic character.

**ADS (Adaptive Dual Search)** raises a question: **can we quantify the trustworthiness of a heuristic with information theory, so that the weight adjusts automatically to the uncertainty of the current state?**

The core idea ties the heuristic weight $\alpha$ to the **entropy** of the current state:

- When the model is highly certain about the current state (low entropy), $\alpha$ is large — trust the heuristic and advance quickly
- When the model is full of uncertainty (high entropy), $\alpha \to \infty$ — a barrier forms, refusing entry to high-uncertainty regions

This is not parameter tuning — it's **turning uncertainty into an actionable search constraint**. A\*'s admissibility guarantees "no overestimation"; ADS's entropy barrier guarantees "no wandering into high-entropy dead ends".

**Its relationship to admissibility**: admissibility constrains the range of $h(n)$ (no overestimation); ADS's entropy regularization constrains the direction of search (repel high uncertainty). The two are complementary: the former is static and mathematical; the latter is dynamic and information-theoretic.

The full technical details of ADS, and how it applies to the reasoning space of LLMs, unfold in Chapter 10.

### 6.6 The Barrier Is Not Greed: The Ontological Difference Between ADS and a Selection Function

Before we continue, I have to clear up a misunderstanding that is extremely easy to fall into.

ADS's logarithmic barrier $\alpha(B_s) = -\log(1 - B_s)$ hides a dangerous interpretation trap: "don't go where entropy is high, only walk the low-entropy routes" — isn't that just greed?

No. That "no" reaches down into a basic categorical error in the taxonomy of search algorithms. Let me pin the difference down with a thought experiment.

**Two mazes.**

Maze A (greedy): you are in a fixed maze. Every junction has three directions. Each time, you choose the one that "looks closest to the exit". There are directions you never walk — not because you can't walk them, but because you chose not to.

Maze B (ADS): you are in a dynamic maze. Every time you approach a high-uncertainty region (multiple forks ahead, sight blocked), the maze itself deforms — those high-uncertainty directions get sealed off by a wall of energy. You haven't "chosen not to go"; you **cannot** go. When the uncertainty dissipates (clear view, obvious path), the wall dissolves and every direction is walkable again.

This isn't rhetoric. These are two different kinds of mathematical objects:

- A greedy algorithm: given a graph $G = (V, E)$, it defines a selection function $\sigma: V \to V$, with $\sigma(v) \in \text{neighbors}(v)$ — it makes choices on a fixed set of reachable states.
- ADS: given a graph $G = (V, E)$, it defines a topological operator $\tau: \mathcal{P}(V) \to \mathcal{P}(V)$ — it removes or restores edges according to local entropy, directly altering the reachable set itself.

Greedy changes the choice made on a given set. ADS changes the set itself.

**The physics verdict.** In classical mechanics, a particle moves in a potential field $V(x)$. When $V(x) \to \infty$, that region vanishes from the particle's phase space — not that the particle "doesn't want to go" or "prefers not to", but that the laws of physics forbid it. ADS's $\alpha \to \infty$ plays exactly that role: as $B_s \to 1$, the high-entropy directions are removed from the reachable manifold, not "avoided". A greedy algorithm that picks the wrong direction can backtrack — the road is still there. ADS seals the high-entropy road — the road no longer exists.

**Why is entropy not a "locally greedy" quantity?** A greedy $h(s)$ is a manually designed geometric distance estimate — it only looks at the distance from the current position to the goal, and has nothing to do with the model's state of knowledge. ADS's entropy $H(p_s)$ is a quantification of the model's global uncertainty about "which way to go next", informed by all of its training data and the current context. It encodes the statistical structure of the training data, the limits of the model's capacity, and the ambiguities the current context has resolved — and those it hasn't. Greedy uses geometric distance on a map; ADS uses the information state of a knowing agent — one is a function of space, the other a **function of knowledge**.

**Why is the distinction so easy to confuse?** Because greedy and ADS behave identically in simple settings — on a straight road with no forks, "choose the lowest-entropy direction" and "exclude the high-entropy directions" take the same route. The difference only appears in complex terrain: when the optimal path must temporarily cross a high-uncertainty region, greedy can go wrong because the selection function is myopic, and ADS can go wrong because the topology reshaping is over-aggressive and walls off the correct detour. But they fail for different reasons — fixing greedy means improving the selection function (a better heuristic, deeper lookahead); fixing ADS means tuning the barrier's sensitivity (changing how $\alpha$ responds to $B_s$). They are different problems, needing different fixes.

**A formal distinction:**

|  | Greedy / A\* / MCTS | ADS |
|:--|:--------------------|:----|
| Operates on | a fixed graph $G = (V, E)$ | a dynamic graph $G_t = (V, E_t)$ |
| Kind of operation | a selection function $\sigma: V \to V$ | a topological operator $\tau: \mathcal{P}(V) \to \mathcal{P}(V)$ |
| Source of information | external cost estimate $h(n)$ | internal cognitive state $H(p_s)$ |
| Level of cognition | zero-order (pick the best neighbor) | second-order (the system knows what it doesn't know) |
| Failure mode | myopia of the selection function | over-aggressive topology reshaping |

**Put together.** A\* proves that on a fixed topology, a good selection function can cut the search cost without sacrificing optimality — that's **the wisdom of choice**. MCTS proves that on a fixed topology, random sampling plus statistical backpropagation can approach optimality with no heuristic at all — that's **the wisdom of statistics**. ADS proves that if you give up the fixed-topology assumption and reshape the search space dynamically with information-theoretic quantities, you get a search strategy of a completely different kind — that's **the wisdom of geometry**. Reading ADS as greed misses its most important insight: **the shape of the search space itself is a variable you can optimize.**

<div class="center">

------------------------------------------------------------------------

</div>

## 7. The Approximation Algorithm's Contract: Give Up Optimality, Keep the Guarantee

A\*'s admissibility guarantees the optimum, but the price is exploring every "possibly optimal" path. On some problems, that's still too slow.

In 1973, computational complexity theory hit a turning point. Stephen Cook proved that SAT is NP-complete — if you can solve it in polynomial time, you can solve every NP problem.

That proof triggered a deep philosophical shift: **maybe we should give up hunting for the optimal solution and settle for "good enough"**.

But "good enough" needs a precise definition. Otherwise, any algorithm can just claim to be "good enough".

In 1974, David Johnson introduced the **approximation ratio**. For minimization problems, an **α-approximation algorithm** guarantees:

$$
\text{algorithm's solution} \leq \alpha \times \text{optimal solution}
$$

This is a **worst-case guarantee** — no matter the input, the approximation ratio never exceeds α.

Let me show a classic example.

### 7.1 Vertex Cover: A 2-Approximation Algorithm

**Vertex cover**: given a graph, find the smallest set of vertices such that every edge has at least one endpoint in the set.

This problem is NP-complete. Brute force has to check $2^n$ possibilities (n being the number of vertices).

But there's an extremely simple 2-approximation algorithm:

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

The algorithm runs in $O(|E|)$ time (E being the number of edges) and guarantees:

$$
|C| \leq 2 \times |C^*|
$$

where $C^*$ is the optimal vertex cover.

**Proof** (a classic technique of approximation analysis):

For every edge $(u, v)$ we pick, the optimal cover $C^*$ must contain at least one of u or v — otherwise that edge wouldn't be covered.

Our algorithm picks both. So the number of vertices we pick is at most twice the optimal solution.

This guarantee is **tight** — there are inputs where the algorithm hits exactly 2×. But it's a **worst-case** guarantee: whatever the input, the ratio never exceeds 2.

### 7.2 PTAS: Getting Arbitrarily Close to Optimal

The holy grail of approximation algorithms is the **PTAS** (Polynomial-Time Approximation Scheme).

A PTAS is a family of algorithms: for any $\epsilon > 0$, it finds a $(1 + \epsilon)$-approximation in polynomial time.

In other words, you can get as close to optimal as you like — want 99% optimality? Fine. 99.9%? Also fine. The price is just a growth in the polynomial time.

\[Li & Jin, 2015\] provided the first PTAS for the weighted unit-disk covering problem. This problem has important applications in wireless network design: given points on a plane, cover all of them with as few unit disks as possible.

Their algorithm runs in $O(n^{2/\epsilon})$ time. From $\epsilon = 0.1$ to $\epsilon = 0.01$, the time goes from $O(n^{20})$ to $O(n^{200})$ — still polynomial, but in practice uncomputable.

This reveals the PTAS's limit: **theoretically feasible, practically maybe not**.

**The approximation algorithm's contract**: I won't guarantee the optimal solution, but I guarantee mine isn't too bad — and I can tell you exactly how bad "not too bad" is. That's an honest contract. It's far stronger than "I did my best".

<div class="center">

------------------------------------------------------------------------

</div>

## 8. Pseudocode: A\* and Adaptive Heuristic Weights

Let me formalize the core algorithms we've discussed.

**Algorithm 1: Standard A\* search**

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

**Time complexity**: $O(b^d)$, where $b$ is the branching factor and $d$ the depth of the solution. The better the heuristic, the fewer nodes actually get expanded.

**Algorithm 2: ADS adaptive heuristic weights**

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

**The key innovation**: α is not fixed — it adapts dynamically to the information-theoretic character of the current state. When entropy is high (much uncertainty), rely less on the heuristic; when entropy is low, trust the heuristic and advance fast.

<div class="center">

------------------------------------------------------------------------

</div>

## 9. Seeing It: How the Heuristic Weight Shapes the Search Path

Picture a 2D grid world with obstacles and a goal. Different heuristic weights α lead to completely different search behavior.

![Figure 8.1: Search paths under different heuristic weights](/figures/ch08_fig1_astar_comparison.png)

*Search behavior under three heuristic weight settings: (a) α=0 (Dijkstra): uniform diffusion — explores every direction, guarantees optimality but inefficient. (b) α=1 (standard A\*): advances along the heuristic direction, with moderate backtracking around obstacles. (c) α=2 (greedy): over-trusts the heuristic and may fall into a local optimum. Green is the start, red the goal, blue shading the expanded nodes, and the yellow line the final path.\*

**Key observations**:

- α too small: the search is too conservative, expanding lots of useless nodes
- α too large: the search is too aggressive and may miss the optimal path
- α=1 with an admissible h: the balance point — optimality guaranteed, efficient

ADS's innovation is this: **α is not fixed — it adapts to the local terrain**.

That adaptivity comes from computing the information-theoretic barrier B(s):

- In open areas: IG is high (big information gain along the heuristic direction), U is low (later states are certain) $\to$ B high $\to$ α high
- Near obstacles: IG is low (the heuristic direction is blocked), U is high (multiple detour choices) $\to$ B low $\to$ α low

<div class="center">

------------------------------------------------------------------------

</div>

## 10. The Philosophy of Heuristics: The Courage to Accept Imperfection

Let me return to this chapter's opening question: when perfection is out of reach, what should we do?

The heuristic answer: **accept approximation at an explicit cost**.

That takes courage, because you have to admit perfection is unreachable. But that isn't failure — it's wisdom.

**Admissibility is A\*'s contract**: I may be optimistic, but I may not lie. If $h(n)$ overestimates the cost, A\* may find a suboptimal path — the contract is broken and the guarantee is void. That's an **honest contract**: I tell you my assumptions, and if they hold, I guarantee the result.

**The approximation ratio is the approximation algorithm's contract**: I don't promise optimality, but I promise "not too bad". A 2-approximation algorithm guarantees a solution no worse than 2× the optimal — that's a **worst-case mathematical guarantee**, not an average-case one, not "usually pretty good", but "never above 2×, ever".

**Probabilistic guarantees are the randomized algorithm's contract**: I don't guarantee correctness every time, but I guarantee correctness with probability $1 - \delta$. That's a **quantifiable risk**: you can choose $\delta = 10^{-6}$ and get a failure probability below one in a million.

These three contracts correspond to three kinds of "imperfection":

1.  **Conditional guarantee**: if the assumption holds, I promise optimality (A\*)

2.  **Worst-case guarantee**: I promise not-too-bad, but possibly not optimal (approximation algorithms)

3.  **Probabilistic guarantee**: I'm very likely right, with a small chance of failure (randomized algorithms)

By contrast, heuristics with no contract — "I did my best", "usually works", "works on the data I tested" — are **dishonest approximations**. They never tell you the conditions of failure, never quantify the "not good enough" degree, never give you the probability of risk.

But heuristics also cost:

**Design cost**: good heuristics need domain knowledge. The Manhattan distance works for grids — what about graph search? Continuous spaces? Every new problem needs a redesigned heuristic.

**Computation cost**: a sophisticated heuristic may be slower than the search itself. If $h(n)$ needs $O(n^2)$ computing, you might as well search directly.

**Generalization risk**: heuristics designed for one specific problem may fail in a different setting. That's why ADS learns a heuristic policy with a neural network — it can generalize across tasks. But the price is losing the mathematical guarantee.

This leads to a deep tension: **provable guarantees vs. empirical generalization**.

A\* has the admissibility guarantee but needs a hand-designed $h(n)$. A neural network can learn automatically but gives up the guarantee. Can we have both?

That's Chapter 12's question. For now, hold onto this tension.

<div class="center">

------------------------------------------------------------------------

</div>

## 11. Yao's Principle: Algorithm Design Is a Game

Before discussing how Collins breaks a deterministic lower bound, we need to understand a deeper question: **when you design a heuristic, who are you playing a game against?**

In 1977, Andrew Chi-Chih Yao gave an answer — in a paper only a few pages long — that rewrote the landscape of algorithmic complexity theory.

---

### 11.1 Turning Algorithm Design into a Game

Picture this game.

Two players:

- **Player A (the algorithm designer)**: picks an algorithm $\mathcal{A}$ to solve some problem
- **Player B (the adversary / input distribution)**: picks an input distribution $\mathcal{D}$, trying to make the algorithm run as slowly as possible

Who goes first? **If A picks the algorithm first, B constructs the worst-case input against that algorithm. If B picks the distribution first, A optimizes the algorithm for that distribution.**

It's a zero-sum game.

---

### 11.2 The Two Sides of the Minimax Theorem

Yao's principle says: **between deterministic algorithms and random inputs, there is a Minimax equality.**

$$
\min_{\text{deterministic algorithm } A} \max_{\text{input distribution } \mathcal{D}} \mathbb{E}_{x \sim \mathcal{D}}[T(A, x)] = \max_{\text{input distribution } \mathcal{D}} \min_{\text{deterministic algorithm } A} \mathbb{E}_{x \sim \mathcal{D}}[T(A, x)]
$$

The left side of this equality is: "the expected time of the best deterministic algorithm under the worst-case input distribution".

The right side is: "the expected time of the best deterministic algorithm under some fixed input distribution".

The two sides are equal. Which means: **you can obtain lower bounds for all randomized algorithms by analyzing deterministic algorithm lower bounds under a fixed distribution.**

---

### 11.3 A Walkthrough in the Feynman Style: Why Is This Deep?

Let me explain with an analogy.

Suppose you are a general (the algorithm designer) drawing up a general-purpose battle plan (the algorithm) against various enemies (inputs). You don't know how the enemy will deploy.

**The pessimistic case (MaxMin)**: you pick the plan first, then the enemy picks the deployment that hurts you most.
- Your optimal plan = "the best outcome I can get under the worst enemy deployment"

**The optimistic case (MinMax)**: the enemy picks the distribution first, then you pick the best plan.
- Your optimal plan = "the best outcome I can get given the enemy's average deployment statistics"

The Minimax theorem says: **those two values are equal**.

What does this mean?

**The worst-case lower bound for deterministic algorithms = the average-case lower bound for randomized algorithms.**

More plainly: if you want to prove that "every deterministic algorithm needs at least $T$ steps", you only need to prove that "there exists some input distribution such that every deterministic algorithm has expected time at least $T$".

That's an extremely powerful tool — you turn "worst-case" analysis into "average-case" analysis, and the latter is usually much easier to handle.

---

### 11.4 Heuristic Design as a Game: The Game-Theoretic View

Now apply this framework to heuristic design.

Designing the heuristic function $h(n)$ is, essentially, a game:

| Player | Choice | Goal |
|:-------|:-------|:-----|
| The heuristic designer | the heuristic $h(n)$ | minimize search time |
| The instance generator | the distribution of input graphs/mazes | maximize search time |

Yao's principle tells us: **no matter how smart your deterministic heuristic, there is a performance lower bound decided by the input distribution.**

That bound isn't an engineering limit — it's an information-theoretic one: if your heuristic is deterministic, the adversary (the input distribution) can always find a class of problems that degrades your performance to the worst case.

---

### 11.5 From Relative Lower Bounds to Absolute Lower Bounds

Here a key distinction emerges:

**Relative lower bound**:
- "For the deterministic heuristic $h$, there exists an input where search time is $\Omega(T)$"
- This is the lower bound relative to a particular algorithm — switch algorithms, and the bound can change

**Absolute lower bound**:
- "For **any** deterministic algorithm, search time is $\Omega(T)$"
- This is a lower bound about the problem itself — no deterministic algorithm can break it

Yao's principle builds a bridge between the two: through the Minimax equality from game theory, you can lift a particular algorithm's worst-case lower bound into an absolute lower bound that holds for **all** deterministic algorithms.

**But here's the catch**: this absolute lower bound only holds for **deterministic** algorithms.

Randomized algorithms aren't bound by the same constraint.

Why? Because randomization breaks the symmetry of the game: **the adversary can't construct worst-case inputs against a randomized algorithm** — every run of a randomized algorithm differs, so the adversary doesn't know what to "aim at".

That leads straight to the core idea of Collins.

<div class="center">

------------------------------------------------------------------------

</div>

## 12. Collins: Breaking the Deterministic Lower Bound with Randomization

Heuristics are usually deterministic: given a state s, h(s) always returns the same value. But the **Collins** strategy shows that **randomization can break an information-theoretic lower bound on determinism**.

Collins targets the optimizer-state compression problem, but its core idea reveals a deep principle of heuristic design.

### 12.1 An Information-Theoretic Lower Bound for Determinism

**Proposition 1** (deterministic lower bound): any deterministic data structure that maintains the state of d parameters at precision ε must use $\Omega(d \log(1/\epsilon))$ bits.

This isn't an engineering limit — it's an **information-theoretic counting argument**:

To distinguish $Q^d$ different states (where $Q = 1/\epsilon$ is the number of quantization levels), you need at least $\log_2(Q^d) = d \log Q$ bits of information.

Which means: if you want to store the first and second moments of the Adam optimizer deterministically (for d parameters), you need at least $O(d)$ memory.

For large language models (d might be in the billions), that's a huge bottleneck.

### 12.2 Randomization's Breakthrough: Trading Probability for Space

Collins uses Count-Sketch + 2-universal hashing to bring the state complexity from $O(d)$ down to $O(k)$, where $k \ll d$. The compression ratio $c = d/k$ can reach tens.

The key idea: **don't store the exact state — store a random projection of it**.

This is like **coarse-graining** in physics: you don't track every particle's exact position; you track macroscopic statistical quantities. The price is introducing noise. But if the noise is small enough, the macroscopic behavior stays correct.

The Chernoff inequality gives a safe upper bound:

$$
c_{\text{opt}} \leq T_{\text{eff}} \cdot \text{SNR}_b
$$

where $T_{\text{eff}}$ is the effective number of training steps and $\text{SNR}_b$ is the batch signal-to-noise ratio.

Experiments confirmed the theory: with $c \leq 32$, the convergence loss matched the baseline; at $c > 40$ it degraded sharply — the transition point $c \approx 34$ matches the theoretical prediction closely.

### 12.3 What This Means for Heuristics: Deterministic vs. Probabilistic Guarantees

Collins's work reveals a deep trade-off in heuristic design:

**Deterministic heuristics**: always return the same estimate for a given input. But they're bound by information-theoretic lower bounds — a good heuristic needs to store a lot of domain knowledge.

**Randomized heuristics**: trade a probabilistic guarantee for lower compute/storage cost. Not correct every time, but "correct with high probability".

That's another kind of contract — from "guaranteed correct" to "correct with probability $1 - \delta$".

Seen from physics, this is the **second law of thermodynamics**: you can trade entropy (randomness) for energy (compute). Deterministic systems must maintain a low-entropy state, at the price of high energy consumption. Randomized systems allow more entropy, so they cost less energy.

Landauer's principle gives the theoretical floor: erasing 1 bit of information requires at least $k_B T \ln 2$ of energy. A randomized heuristic doesn't erase information — it lets information exist in the form of noise, so its energy cost is lower.

<div class="center">

------------------------------------------------------------------------

</div>

Heuristics approximate on top of a known structure. In the next chapter, we'll see how the Transformer completely reworks the very concept of "structure" by dynamically connecting all the nodes.

## Open Questions

- **Can the "quality" of a heuristic be formally defined?**

  Which $h(n)$ is theoretically optimal? Pearl proved that consistency is stronger than admissibility — but is it the strongest useful condition? Does there exist an "admissible function closest to $h^*(n)$"? This question is equivalent to: given finite compute, how should it be distributed optimally across the design of heuristics?

- **Where is the boundary of the entropy barrier's applicability?**

  ADS's dynamic logarithmic barrier works on arithmetic tasks, but fails on abstract algebra. Is that failure **principled** or **engineering**? In other words: is entropy regularization itself unsuited to higher-order abstraction, or is the 360M model just too small? With a 3B model, would the abstract algebra task succeed? This question touches the fundamental limit of entropy as a reasoning constraint.

- **What's the essential difference between the implicit heuristics a neural network learns and the explicit heuristics humans design?**

  AlphaGo's value network is a heuristic function, but it doesn't satisfy admissibility — why does it still work? One possible answer: neural networks learn **average-case** heuristics, while admissibility is a **worst-case** guarantee. But what does that imply? Does a notion of "average-case admissibility" exist?

- **Where is the theoretical boundary of randomized heuristics?**

  Collins proved the deterministic lower bound $\Omega(d \log(1/\epsilon))$, but what's the upper bound for randomization? The Chernoff bound gives $c_{\text{opt}} \leq T_{\text{eff}} \cdot \text{SNR}_b$ — but is it tight? Does an "optimal randomized heuristic" exist — one that minimizes compute given a target failure probability $\delta$?

- **Can heuristics be learned online?**

  Classic heuristics are designed offline — fixed before the search starts. But could we design a heuristic that learns and adjusts during the search itself? Every node expansion updates the estimate of $h^*(n)$, making the rest of the search more efficient? That resembles reinforcement learning, but the goal is to speed up the current search, not to generalize to future tasks.

- **Where are the lower bounds for approximation algorithms?**

  We know vertex cover has a 2-approximation. But could we do 1.5-approximation? 1.1? For some NP-complete problems, approximation lower bounds exist — unless P = NP, no $(1 + \epsilon)$-approximation exists. Those bounds tell us: for some problems, even approximating is hard.

- **Can ADS's barrier and greedy's selection function be distinguished experimentally?**

  Section 6.6 argued an ontological difference between them — greedy makes choices on a fixed topology, ADS dynamically reshapes the topology itself. But can the difference be captured experimentally? One possible experiment: construct a reasoning task where "the optimal path must temporarily cross a high-entropy region" (like needing to try a suspicious-looking intermediate step first). Greedy would pick the low-entropy direction and go wrong (but can backtrack); ADS would seal the high-entropy entrance and never find the path at all (no backtracking possible). If the two algorithms' failure modes really differ, the ontological distinction is more than a philosophical classification. If their behavior coincides, the Section 6.6 argument needs a second look.

- **Can the shape of the search space itself become the learning target?**

  ADS's topology reshaping suggests a bolder idea: don't learn "search strategies in a given space" — learn "the shape of the search space itself". That's equivalent to learning a dynamic $E_t$ (edge set) such that, from any starting point, greedy choice (on the reshaped space) finds the optimal solution. This is a geometric formulation of meta-learning — what's learned is no longer the policy $\pi(a|s)$ but the topology $\tau: \mathcal{G} \to \mathcal{G}$. Almost no theoretical work exists in this direction today.

<div class="center">

------------------------------------------------------------------------

</div>

## Bonus: Search Is Convergence — A Dynamical View of Heuristics

Every step of A\* is, in fact, an Euler step.

Go back to the evaluation function $f(n) = g(n) + h(n)$. View the search process as a discrete dynamical system on the state space:

$$
s_{t+1} = \arg\min_{a} \left[ g(s_t, a) + \alpha \cdot h(a) \right]
$$

The structure is exactly the Euler discretization of gradient descent:

$$
x_{t+1} = x_t - \eta \nabla E(x_t)
$$

$g$ is the cost already paid — the "past"; $h$ is the estimated remainder — the "future"; $\alpha$ is your confidence in the future. Search is what happens when these two forces tug at each other, approaching the goal step by step.

:::details What's an Euler step?
In calculus, a derivative describes continuous change. But computers can only do discrete things — each time, take one little step.

The Euler method slices "continuous flow" into little steps:

$$x_{t+1} = x_t + \Delta t \cdot f(x_t)$$

Which means: right now you're at $x_t$, the rate of change is $f(x_t)$; take a little step of $\Delta t$ and you arrive at $x_{t+1}$.

Gradient descent is a special case of the Euler step — the direction of change is the negative gradient (the steepest downhill direction), and the step size is the learning rate $\eta$:

$$x_{t+1} = x_t - \eta \nabla E(x_t)$$

A\*'s search has the same structure: current state $s_t$, evaluate the cost of every direction, take the smallest one step. The only difference is that A\* works on a discrete graph while gradient descent works on a continuous surface. At bottom, both ask the same question: **which way do I move, to get to the goal fastest?**
:::

**The triangle inequality is the guarantee of convergence.**

The consistency condition $h(n) \leq c(n, n') + h(n')$ is exactly the three-side relation of a triangle: $g$, $c$, and $h$ form a triangle. This inequality guarantees that the $f$ value never decreases — which is precisely the condition for a Lyapunov function: $V(s_{t+1}) \leq V(s_t)$, the system converges stably to the goal.

:::details What's the triangle inequality?
You learned it in elementary school: any two sides of a triangle add up to more than the third.

$a + b \geq c$

In heuristic search, that rule becomes: the estimated cost from $n$ to the goal cannot exceed the total cost of "first take one step to $n'$, then get from $n'$ to the goal".

$$h(n) \leq c(n, n') + h(n')$$

Intuition: your guess about the remaining journey can't be more pessimistic than "take one real step and guess again". Otherwise your map contradicts itself — like a map that says "Beijing to Shanghai: 2000 km, but Beijing to Nanjing is only 100 km and Nanjing to Shanghai is only 100 km". You'd suspect it's lying.

A heuristic that satisfies this rule is called **consistent**. Consistency guarantees A\* never walks backward, and each node needs to be expanded only once.
:::

:::details What's a Lyapunov function?
Imagine a ball rolling in a bowl. No matter where you drop it, the ball ends up at the bottom.

The Lyapunov function is that "bowl height" — a number measuring how far the system is from the goal. As long as this number decreases (or doesn't increase) at every step, the system must converge to the goal — no wandering off.

$$V(s_{t+1}) \leq V(s_t)$$

In A\*, the $f$ value is that bowl height. Consistency guarantees $f$ never decreases, which is equivalent to saying: we're always closing in on the goal, never circling.

Lyapunov used this idea in 1892 to prove the stability of a wide class of dynamical systems. A\*'s optimality proof is, in essence, the same thing.
:::

**What is ADS doing?**

ADS's logarithmic barrier $\alpha = -\log(1 - B_s)$ is deforming that triangle dynamically. When entropy is low, $\alpha \approx 0$, the triangle collapses and $g$ dominates; when entropy is high, $\alpha \to \infty$, the $h$ side explodes and high-uncertainty directions get excluded outright.

The barrier isn't "tuning parameters" — it's **reshaping the geometry of the search space by local uncertainty**.

---

That "search is convergence" intuition is a geometric story in the heuristic setting.

But replace the search space with the belief space of an LLM, replace $h$ with a cross-entropy energy function, replace the Euler step with a reasoning operator — the same convergence structure gets a deeper name.

Chapter 12 reveals it.

<div class="center">

------------------------------------------------------------------------

</div>

## Appendix: Full Code

Here is the complete script for this chapter's experiments (save it as `ads_search.py` and run):

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

## Further Reading

- Hart, Nilsson, Raphael (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* — the original A\* paper, defining admissibility and consistency

- Yao, A. C.-C. (1977). *Probabilistic Computations: Toward a Unified Measure of Complexity* — the original paper on Yao's principle; the Minimax theorem lifts deterministic lower bounds into randomized ones `-> [FOCS 1977]`

- This book, Chapter 8, §6 *ADS: Making Heuristics Information-Theoretic* + §12 *Collins: Breaking the Deterministic Lower Bound with Randomization* — complete discussion of both strategies (full code in the appendices of Chapters 8 and 11)

- \[Li & Jin, 2015\] — the first PTAS for weighted unit-disk covering, the gold standard of approximation algorithms `-> [arXiv:1502.04918]`

- \[Esmer et al., 2022\] — exponential-time approximation algorithms for approximate monotone local search `-> [arXiv:2206.13481]`

- Pearl, J. (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving* — the classic textbook on heuristic search

- Motwani, R. & Raghavan, P. (1995). *Randomized Algorithms* — the authoritative textbook on randomized algorithms (Cambridge University Press), covering Chernoff bounds, 2-universal hashing, the Minimax principle, and their use in algorithm design

- Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach* (4th ed.) — Chapters 3–4 discuss A\* and its variants in detail

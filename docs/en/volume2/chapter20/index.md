# Chapter 20: The Formal Contract of Heuristics — A Precise Mathematical Definition of "Good Enough"

> Admissibility is not an engineering compromise. It is a mathematical promise.

---

Chapter 19's conclusion was a downer: many reasoning problems are fundamentally hard, in a computational sense. Not because the algorithms aren't good enough, but because the geometry of the problem itself says there's no shortcut.

Yet humans and machines keep solving these "hard" problems. Not exactly, not always optimally, but usually well enough, and fast enough.

There's a question hiding here: **what does "good enough" mean?** If the answer is arbitrary, then any algorithm that doesn't crash can call itself a "heuristic." But if "good enough" has a precise mathematical meaning, then a heuristic isn't an engineering compromise — it's a promise with a contract: what it promises, at what cost, and under what conditions it delivers.

That's what this chapter does: writes that contract out clearly.

---

## 20.0 The Optimistic Cartographer Game: Heuristics Must Sign a Contract

Chapter 19 told us that many mazes are far too big to count on brute force. So we hire a cartographer. The cartographer can't tell you the complete route; at every intersection it only gives you an estimate: roughly how much farther to the goal.

This is the heuristic function $h(n)$.

But this cartographer has to sign a contract. The first contract is called **admissibility**: you may be optimistic, but you may not exaggerate the difficulty. In other words, the estimated distance must not exceed the true shortest distance:

$$h(n) \leq h^*(n)$$

If the cartographer underestimates the distance, the search just takes a few extra steps. If the cartographer overestimates it, the search may abandon the truly optimal path entirely. The second contract is called **consistency**: the estimates at neighboring intersections must obey a triangle inequality — the cartographer can't say "we're almost there" today, then suddenly say "it's awfully far" after a single step.

:::details The Formal Skeleton of This Game
- **State space**: the nodes of the search graph, known costs $g(n)$, heuristic estimates $h(n)$, and a priority queue.
- **Legal moves**: expand the node with the smallest $f(n) = g(n) + h(n)$, then update its successors.
- **Transition rules**: take a node $n$ off the open list, and add or update each successor $n'$ with cost $g(n') = g(n) + c(n, n')$.
- **Winning condition**: find the optimal path under a provable guarantee, or get an approximation ratio / PAC-style quality bound.
- **Failure modes**: the heuristic has no contract — it just "looks fine", and ends up delivering the fastest wrong answer.
:::

This game makes "good enough" precise. A heuristic without a contract is just luck; a heuristic with a contract is a mathematical object. Admissibility, consistency, approximation ratio, PAC guarantees — these are all different clauses of different contracts.

The dignity of a heuristic isn't in always producing the optimal answer. It's in honestly stating: what I promise, what I don't; under which conditions I'm reliable, and under which conditions I'll fail.

::: info Professor Pallas's Cat's Commentary
Optimism can be a virtue, or it can be fraud. A* allows the heuristic function to be optimistic, because optimism keeps you exploring; it doesn't allow the heuristic to be arrogant, because arrogance makes you give up on a real path too early. The whole ethics of heuristics hides inside this one inequality.
:::

### 20.0.5 Playing the Game in Code

Section 20.2.5 traced one A* search by hand on a six-node graph. The program below runs the same graph through the same algorithm twice — once with the book's admissible heuristic, once with a single overestimate at node A. Same maze, same algorithm: one returns the optimal route, the other the fastest wrong answer.

```python
#!/usr/bin/env python3
"""
optimistic_cartographer.py -- the Optimistic Cartographer Game of Ch. 20.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 20, Section 20.0
"The Optimistic Cartographer Game: Heuristics Must Sign a Contract."

The maze is too big for brute force, so we hire a cartographer who, at
every intersection, estimates how far the goal still is: the heuristic
h(n). This program runs A* on the exact six-node graph the book traces
by hand in Section 20.2.5, with two cartographers:
  * the honest one, whose estimates never exceed the true distance
    (admissible: h <= h*) -- the contract of Section 20.0;
  * the arrogant one, who overestimates at node A -- no contract.
Same maze, same algorithm: one returns the optimal route, the other
delivers the fastest wrong answer. That is the whole ethics of
heuristics, hiding inside one inequality.
Run:  python3 optimistic_cartographer.py
"""
import heapq

# The book's graph, Section 20.2.5: S (start), A, B, C, D, G (goal).
EDGES = {
    "S": (("A", 1), ("B", 4)),
    "A": (("C", 3), ("D", 2)),
    "B": (("D", 2),),
    "C": (("G", 2),),
    "D": (("G", 4),),
    "G": (),
}

HONEST_H = {"S": 5, "A": 4, "B": 4, "C": 2, "D": 2, "G": 0}
ARROGANT_H = {"S": 5, "A": 10, "B": 1, "C": 2, "D": 2, "G": 0}

START, GOAL = "S", "G"

def astar(edges, h, start, goal):
    """A* on a graph; returns (path, cost, expanded_in_order)."""
    heap = [(h[start], 0, start, [start])]
    best_g = {start: 0}
    expanded = []
    while heap:
        _, g, node, path = heapq.heappop(heap)
        if g > best_g.get(node, float("inf")):
            continue
        expanded.append(node)
        if node == goal:
            return path, g, expanded
        for nxt, cost in edges.get(node, ()):
            ng = g + cost
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                heapq.heappush(heap, (ng + h[nxt], ng, nxt, path + [nxt]))
    return None, None, expanded

def trace(name, h, expected):
    path, cost, expanded = astar(EDGES, h, START, GOAL)
    print(f"\n{name}:")
    print(f"  route found:  {' -> '.join(path)}")
    print(f"  total cost:   {cost}")
    print(f"  nodes expanded in order: {' -> '.join(expanded)}")
    assert path == expected["path"] and cost == expected["cost"]

def demo():
    print("=" * 70)
    print(" THE OPTIMISTIC CARTOGRAPHER GAME -- two cartographers, one maze")
    print(" The Reasoning Kingdom, Volume 2, Chapter 20 (Sec. 20.0, 20.2.5)")
    print("=" * 70)
    print("\nThe maze (the same six nodes the book traces by hand):")
    print("  S->A:1  S->B:4  A->C:3  A->D:2  B->D:2  C->G:2  D->G:4")
    print("\nThe honest cartographer -- h never exceeds the true distance:")
    print("  h: S=5, A=4, B=4, C=2, D=2, G=0   (h(n) <= h*(n) everywhere)")
    trace("honest cartographer (admissible)", HONEST_H,
          {"path": ["S", "A", "C", "G"], "cost": 6})
    print("  optimal route S->A->C->G, cost 6 -- exactly what the hand")
    print("  trace in the book found.")
    print("\nThe arrogant cartographer -- one lie, at node A:")
    print("  h: S=5, A=10, B=1, C=2, D=2, G=0   (h(A)=10 > h*(A)=5: contract broken)")
    trace("arrogant cartographer (inadmissible)", ARROGANT_H,
          {"path": ["S", "B", "D", "G"], "cost": 10})
    print("  route S->B->D->G, cost 10. The overestimate at A made the")
    print("  search abandon the truly optimal path. Optimism keeps you")
    print("  exploring; arrogance makes you give up on a real path early.")
    print("  Same maze, same algorithm -- one inequality of difference.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

---

## 20.1 How the Two Kinds of Approximation Fail

Let's look at two approximation schemes first. Both sound intuitively reasonable. Both go wrong in some situations.

**Scheme one: greedy algorithms.** At each step, make the choice that looks best locally, and never look back. In the traveling salesman problem, always walk to the nearest unvisited city; in the graph coloring problem, always give the current vertex the smallest-numbered color that doesn't clash with its neighbors.

Greedy algorithms are usually fast (polynomial time), but the results can be awful — sometimes an exponential factor worse than the optimum. Greedy makes no promise; it just "looks fine".

**Scheme two: random search.** Sample the solution space at random, and report the best result found. Random enough, and run enough times, and maybe you'll find a good solution.

Random search makes no promise either — "maybe" is not a contract. In the worst case, good solutions are so sparse that random search may never find one.

::: info Professor Pallas's Cat's Commentary
A large chunk of real-world systems run on "maybe" — random restarts, repeated sampling, keep the best result. These methods sometimes work in practice, but you don't know when they stop working, and you don't know the probability of failure. Without a contract there are no failure conditions — you can't even tell whether the algorithm is working. "Sometimes it works" is not a guarantee; it's luck.
:::

The common problem with both schemes: they give no provable bound on "how far off". And that bound is exactly what a heuristic contract requires.

### A Concrete Failure Scene

Take an extremely simple traveling salesman problem — 4 cities, edge weights below — and you can see the difference between greedy and random with your own eyes:

```
Cities: A, B, C, D
Edge weights (distances):
  A-B: 10, A-C: 15, A-D: 20
  B-C: 35, B-D: 25
  C-D: 30
Optimal Hamiltonian cycle: A->B->D->C->A, total cost = 10+25+30+15 = 80
```

**Greedy (nearest neighbor) starting from A**: A to B (10, shortest) -> B to D (25, shortest from B aside from A, since B to C is 35) -> D to C (30) -> forced back to A (15). Route: A->B->D->C->A, cost = 10+25+30+15 = 80. This time it happened to be right, because the graph is tiny.

But change the starting point: **starting from B** -> B to A (10) -> A to C (15) -> C to D (30) -> forced back to B (35). Route: B->A->C->D->B, cost = 10+15+30+35 = 90. Suboptimal. And these are just 4 cities; as the cities grow, greedy's error can balloon exponentially.

**Random search**: sample 100 tours at random; the best might be 85, or it might be 110 — no promise. Run it again and you get a different answer each time.

Neither has a contract. Greedy is right on some graphs, random is never certain. What's missing is exactly that inequality: **$h(n) \leq h^*(n)$**. With it, you know where the limit of optimism is.

::: info Professor Pallas's Cat's Commentary
See? Greedy is right on this tiny graph from A, and wrong from B. You can't use "it was right on A" to prove an algorithm works — an algorithm without a contract isn't "roughly effective", it's "you don't know when it fails". That can be fatal.
:::

---

## 20.2 Admissibility: Never Overestimate

The most classic heuristic framework in search is the A* algorithm. Its core is a **heuristic function** $h(n)$, which estimates the cost from the current state $n$ to the goal.

A* maintains a priority queue and, at each step, expands the node with the smallest estimated total cost $f(n) = g(n) + h(n)$, where $g(n)$ is the known actual cost from the start to $n$, and $h(n)$ is the estimated cost from $n$ to the goal.

The condition under which this algorithm finds the optimal solution lands precisely on one property of $h$:

**Definition (admissibility)**: A heuristic function $h$ is **admissible** if, for every node $n$,

$$h(n) \leq h^*(n)$$

where $h^*(n)$ is the true optimal cost from $n$ to the goal.

What admissibility means: $h$ never overestimates. It may underestimate — underestimation is optimism, and optimism means this path will keep being explored. But if $h$ overestimates, the algorithm may abandon a path that is in fact optimal, far too early.

**Theorem (A* completeness)**: if the heuristic $h$ is admissible, then A* is guaranteed to find the optimal solution whenever one exists.

This is the first form of the heuristic contract: **give me an estimate that never overestimates, and I'll guarantee you the optimal solution**. The cost is that you must spend time constructing that estimate, and the algorithm may need to explore a larger search space.

::: info How Admissibility Gets Violated
A common mistake is using a crude upper bound on the true cost as the heuristic — that naturally overestimates, breaking admissibility. For example, in map search, if you use straight-line distance as $h$, that's admissible (the straight-line distance never exceeds the actual route distance). But if some estimate overestimates on curvy roads, it's inadmissible. Violations of admissibility are usually not obvious — they depend on the specific structure of the problem.
:::

### 20.2.5 Hands-On: Tracing One Complete A* Search

Reading the definition isn't enough. Below is a 6-node directed graph; let's trace every single step of A*. Get a sheet of paper and draw along.

```
Nodes: S (start), A, B, C, D, G (goal)
Edges (S->node, cost):
  S->A: 1,  S->B: 4
  A->C: 3,  A->D: 2
  B->D: 2
  C->G: 2
  D->G: 4

Heuristic function h (estimated cost to G):
  h(S)=5, h(A)=4, h(B)=4, h(C)=2, h(D)=2, h(G)=0

True optimal cost h* (verifiable via reverse Dijkstra):
  Actual: S->A->D->G: 1+2+4=7; S->B->D->G: 4+2+4=10; S->A->C->G: 1+3+2=6
  Shortest: S->A->C->G = 6, so h*(S)=6, h(S)=5 ≤ 6 ✓ admissible
  h*(A)=5 (A->C->G=3+2=5), h(A)=4 ≤ 5 ✓
  h*(B)=6 (B->D->G=2+4=6), h(B)=4 ≤ 6 ✓
  h*(C)=2, h(C)=2 ✓ (exactly precise)
  h*(D)=4, h(D)=2 ≤ 4 ✓
  h*(G)=0, h(G)=0 ✓

For every node, h(n) ≤ h*(n) holds, so h is admissible.
```

**Tracing A\***:

Initial: priority queue = [(S, f=0+5=5)]. g(S)=0.

**Step 1**: expand S (smallest f=5). Successors:
- A: g=1, f=1+4=5 -> add to queue
- B: g=4, f=4+4=8 -> add to queue
Queue: [(A,5), (B,8)]

**Step 2**: expand A (smallest f=5). Successors:
- C: g=1+3=4, f=4+2=6 -> add
- D: g=1+2=3, f=3+2=5 -> add
Queue: [(D,5), (C,6), (B,8)]

**Step 3**: expand D (smallest f=5). Successors:
- G: g=3+4=7, f=7+0=7 -> add
Queue: [(C,6), (G,7), (B,8)]

**Step 4**: expand C (smallest f=6). Successors:
- G: g=4+2=6, f=6+0=6 -> update G (a version with f=7 already exists; the new f=6 is better, so replace it)
Queue: [(G,6), (B,8)]

**Step 5**: expand G (smallest f=6). G is the goal; found the path S->A->C->G, total cost 6.

Note: at Step 3, A* did not go down S->A->D->G (total cost = 7), because after D was expanded, G's f was 7, while C's f=6 in the queue had higher priority. A* correctly skipped the suboptimal path.

Now, what if we change $h$ to be inadmissible? Set $h'(A) = 6$ (an overestimate, since the true $h^*(A) = 5$). Then:
- After expanding S, A's f=1+6=7, B's f=4+4=8
- Queue: [(A,7), (B,8)]
- Expand A (f=7), D's f=5, C's f=6
- Queue: [(C,6), (D,5), (B,8)]
- Expand C, G's f=6… it still finds the optimum in the end, but only because the graph is tiny.

On more complex graphs, overestimation can make A* decide early on that some path "isn't worth exploring" — and that path happens to be optimal. Admissibility doesn't guarantee efficiency; admissibility guarantees you won't miss the optimum. Drop it, and the optimum may slip away from you, and you'll never even know.

::: info Professor Pallas's Cat's Commentary
Look closely at step 3: when D was expanded, the queue had C(f=6) and D(f=5). A* chose D, because D was more optimistic. Optimism makes A* willing to take risks, but it also lets A* turn around promptly when it sees a better path — because optimism is revised downward, never upward. If $h$ is arrogant (overestimating), A* discards certain nodes too early and never looks back. Optimism is the fuel of exploration; arrogance is the reason for missing things.
:::

---

## 20.3 Consistency: The Triangle Inequality Constraint

Admissibility guarantees the quality of the final result, but A*'s efficiency depends on another property of $h$.

**Definition (consistency)**: A heuristic function $h$ is **consistent** if, for every node $n$ and each of its successors $n'$,

$$h(n) \leq c(n, n') + h(n')$$

where $c(n, n')$ is the actual edge cost from $n$ to $n'$.

This is a triangle inequality: the estimated cost from $n$ to the goal is no more than the cost of first walking to $n'$ and then from $n'$ to the goal. Intuitively, consistency says your estimate for every node is "coherent" — there's no situation where, one step later, the estimate suddenly jumps up.

Consistency implies admissibility (you can verify this), but the reverse doesn't hold. Consistency is the stronger condition.

**Under consistency, $f$ is non-decreasing on every node A* expands.** This guarantees: when A* first reaches the goal node, the path it found is the optimal one — no need to search further. Consistency pushes A* from "eventually finds the optimum" to "finds the optimum as early as possible".

These two properties — admissibility and consistency — are the precise mathematical content of the heuristic contract. With them, a "heuristic" isn't a random guess; it's an estimation mechanism that operates within a provable bound.

Back to the optimistic cartographer game: admissibility requires that the cartographer can't call a short road a long road, and consistency requires that the cartographer can't flip-flop between neighboring intersections. A*'s cleverness isn't "trusting intuition"; it's trusting only the intuition that has signed a contract. Without the contract, a heuristic is just a smooth-talking guide; with it, the heuristic becomes a legal move inside a formal system.

### 20.3.5 When Consistency Is Violated: A*'s Perplexity

Using the same graph as 20.2.5, let's build an admissible-but-inconsistent $h$:

Set $h''(S) = 5$, $h''(A) = 4$, $h''(B) = 5$, $h''(C) = 2$, $h''(D) = 1$, $h''(G) = 0$.

Check admissibility: is $h''(n) \leq h^*(n)$ for every node?
- $h''(B) = 5 \leq 6$ ✓, $h''(D) = 1 \leq 4$ ✓. Yes, admissible.

Check consistency: for every edge $(n, n')$, is $h''(n) \leq c(n, n') + h''(n')$?
- Edge B->D: $h''(B) = 5$, $c(B, D) + h''(D) = 2 + 1 = 3$. Is $5 \leq 3$? **No.**

What just happened? From B, the estimate to G is 5; but from D (one step after B), the estimate is suddenly 1. The cartographer said "still far" at B, then after one step to D said "almost there". This flip-flop perplexes A*:

Starting from S: A(f=5), B(f=4+5=9). Pick A. Then D's f=1+2+1=4. Pick D. Then G's f=3+4+0=7. Pick C(f=6). C->G(f=6). Found the optimal path S->A->C->G, cost 6.

Looks fine? But the problem hides in this detail: after D is expanded, the queue holds C(f=6) and G(f=7, f via D). If we later discover another shorter route to D (say S->B->D with a smaller cost? No — S->B->D is 4+2=6, worse than S->A->D's 3), and $h''$ is inconsistent, we may need to **re-expand** nodes that were already expanded. Consistency guarantees that $f$ is non-decreasing, so the first arrival at the goal is the optimum; with inconsistency, A* may have to keep searching at the goal node until $f$ exceeds the current best cost.

In larger searches, the cost of inconsistency is enormous: A* may need to re-expand nodes, and the algorithm degrades to nearly Dijkstra's efficiency. Consistency isn't decoration — it's the contract that underwrites A*'s efficiency.

::: info Professor Pallas's Cat's Commentary
An inconsistent heuristic is like a fickle tour guide. He tells you 5 km are left at B; you walk to D and he says it's actually 1 km. Added up, he only counted 2 km for the B->D stretch (5 -> 1), and the actual distance is 2 km too — "just happen to match". But if, at D, he suddenly says 4 km are left (to the next intersection), you'd start to wonder: does he even know what he's talking about? A*'s suspicion is mathematical: with inconsistency, $f$ can drop, and the algorithm no longer guarantees the first solution is optimal.
:::

---

## 20.4 Approximation Ratio: The Worst-Case Guarantee

A* works well when optimality makes sense as a goal. But for many NP-complete problems, the optimal solution itself is hard to find, even when approximation is allowed. That's where another contract comes in: the **approximation ratio**.

**Definition (approximation algorithm)**: For a minimization problem, if an algorithm $\mathcal{A}$ always returns, on any instance, a solution whose cost is at most $\rho$ times the cost of the optimal solution, then $\mathcal{A}$ is called a $\rho$-approximation algorithm, and $\rho$ is called its **approximation ratio**.

The approximation ratio is the worst-case version of the heuristic contract: no matter the input, the quality of the output stays within $\rho$ times the optimum. $\rho = 1$ is the exact solution; $\rho = 2$ means "at most twice the optimum".

Some famous results:

- **Vertex cover**: there is a 2-approximation algorithm (find a maximal matching, take both endpoints).
- **Traveling salesman problem (metric version)**: there is a 1.5-approximation algorithm (Christofides' algorithm, 1976). This result stood for nearly half a century, until it was slightly improved in 2020.
- **Traveling salesman problem (general version, no triangle inequality)**: if P ≠ NP, then for any constant $\rho$, there is no polynomial-time $\rho$-approximation algorithm.

That last result is the crux: for some problems, approximating is itself NP-hard. Allowing error doesn't always make a problem easier. **The approximation ratio has its own information-theoretic lower bound**, and this bound comes from the problem's intrinsic structure, independent of any specific algorithm.

::: info Why the Approximation Ratio Is Unbreakable
The impossibility proof for the general TSP uses a reduction: if a polynomial-time $\rho$-approximation algorithm existed, we could use it to decide whether a Hamiltonian cycle exists — and the Hamiltonian cycle problem is NP-complete. This reduction shows that any approximation good enough would imply exact-solving power. The problem's intrinsic structure leaves its fingerprints even at the approximation level.
:::

### 20.4.5 Hands-On: The 2-Approximation for Vertex Cover — The Power of a Maximal Matching

The vertex cover problem: given a graph $G = (V, E)$, find the smallest vertex set $C \subseteq V$ such that every edge has at least one endpoint in $C$. This is NP-complete. But a simple 2-approximation exists — remember, 2 means the set you find is at most 2 times as big as the optimal one.

**Algorithm (maximal matching method)**:
1. Find a **maximal matching** $M$ of the graph (a matching to which no more edges can be added)
2. Output both endpoints of every edge in $M$

**Why is this a 2-approximation?**

Let $M$ be the found maximal matching, with $k$ edges. The output set $C$ contains the $2k$ endpoints of these $k$ edges.

- **Lower bound**: the optimal vertex cover needs at least $k$ vertices. Since the $k$ edges in $M$ share no endpoints (that's what a matching is), covering these $k$ edges takes at least $k$ vertices — each edge needs at least one endpoint picked, and the edges picked can't share vertices. So $\text{OPT} \geq k$.
- **Upper bound**: we output $2k$ vertices, and $|C| = 2k \leq 2 \cdot \text{OPT}$. ✓

**Why is maximal (not maximum) enough?** A simple greedy construction of a maximal matching: walk through all the edges; if both endpoints of the current edge aren't yet covered by the matching, add this edge to it. The result isn't necessarily a maximum matching (the one with the most edges), but it's definitely maximal (no more edges can be added). For 2-approximation, maximal is enough.

**A concrete example**:

```
Graph: vertices {1,2,3,4,5}
Edges: (1,2), (2,3), (3,4), (4,5), (1,5)
```

Greedy maximal matching (walking edges in order):
- Encounter (1,2): 1 and 2 are both unmatched -> add to M. M={(1,2)}
- Encounter (2,3): 2 is matched -> skip
- Encounter (3,4): 3 and 4 are both unmatched -> add to M. M={(1,2), (3,4)}
- Encounter (4,5): 4 is matched -> skip
- Encounter (1,5): 1 is matched -> skip

Output C = {1,2,3,4}, size 4.

What's the optimum? This graph is a 5-cycle. Covering all 5 edges takes at least 3 vertices (take every other vertex; {2,4,5} covers every edge). So OPT=3. Our solution is 4, approximation ratio 4/3 ≈ 1.33 ≤ 2. ✓

What if you change the traversal order? If you encounter (2,3) first and then (1,2) and (3,4), $M$ may become {(2,3),(4,5)} or {(2,3),(1,5)}, and the output size is 4 either way. Different maximal matchings give different solutions, all within a factor of 2.

::: info Professor Pallas's Cat's Commentary
Notice the beauty of this algorithm: it doesn't try to choose vertices "cleverly". It relies on an extremely simple fact — the size of a matching is naturally a lower bound on OPT. That bound isn't computed; it's provided by the graph's structure itself. Good approximation algorithms often aren't "cleverly approaching the optimum", but rather "founded a simple structure that is necessarily close to the optimum". This resonates with Lyapunov functions — both find a quantity that naturally decreases (or is naturally bounded).
:::

---

## 20.5 PAC Learning: The Probabilistic Version of "Good Enough"

The approximation ratio is about the quality of a solution; the PAC framework applies the same spirit to learning problems.

**Definition (PAC learning)**: A concept class $\mathcal{C}$ is **PAC learnable** (Probably Approximately Correct) if there exists an algorithm $\mathcal{L}$ such that: for any target concept $c \in \mathcal{C}$, any data distribution $\mathcal{D}$, and any parameters $\varepsilon > 0$ (error tolerance) and $\delta > 0$ (failure probability), when the sample size $m$ is large enough (polynomial in $\varepsilon, \delta$), $\mathcal{L}$ outputs, with probability at least $1 - \delta$, a hypothesis $h$ such that

$$P_{x \sim \mathcal{D}}[h(x) \neq c(x)] \leq \varepsilon$$

In plain words: with high probability (at least $1 - \delta$), it outputs an approximately correct hypothesis (error rate at most $\varepsilon$). PAC is short for "Probably Approximately Correct" — that's not self-deprecation; it's a precise mathematical promise.

::: info Professor Pallas's Cat's Commentary
Many people read "approximately correct with high probability" as lowering the bar. It's the opposite — $\varepsilon$ and $\delta$ are not fuzzy words; they're numbers you can compute. This is more honest than many methods that claim to be "accurate": it knows exactly what it promises, and exactly where the promise breaks. Knowing your boundary is a lot more honest than pretending you have none.
:::

The core conclusion of the PAC framework: the sample size $m$ only needs to grow polynomially in $1/\varepsilon$ and $1/\delta$, and learning is done. Which means: **you don't need infinite data — finite, polynomially-many samples suffice to learn an approximately correct concept with high probability**.

::: info VC Dimension: The Complexity of the Hypothesis Space
In the PAC framework, the required sample size also depends on another quantity: the **VC dimension** (Vapnik-Chervonenkis dimension) of the hypothesis space $\mathcal{H}$. The VC dimension measures the size of the largest point set that $\mathcal{H}$ can "shatter" — being able to shatter a point set of size $d$ means that for any labeling of those points, some hypothesis in $\mathcal{H}$ fits it perfectly. The higher the VC dimension, the more complex the hypothesis space, and the more samples are needed to constrain it. The basic PAC sample complexity theorem: $m = O\!\left(\frac{1}{\varepsilon}\left(d \ln \frac{1}{\varepsilon} + \ln \frac{1}{\delta}\right)\right)$, where $d$ is the VC dimension. This turns the "overfitting" intuition of Chapter 5 of Volume 1 — a too-complex hypothesis space needs more data — into a computable bound.
:::

### 20.5.5 Hands-On: Three Exercises on VC Dimension

**Exercise 1: intervals on the real line.** Let the hypothesis space $\mathcal{H}_1$ be the closed intervals $[a, b]$ on the real line: for a point $x$, predict +1 if $x \in [a, b]$, and -1 otherwise. What's the largest point set it can shatter?

2 points: put them at $x_1 = 0$, $x_2 = 1$. Can all 4 labelings be produced?
- (+,+): the interval $[0, 1]$ ✓
- (-,-): the interval $[2, 3]$ (covers neither point) ✓
- (+,-): the interval $[0, 0]$ ✓
- (-,+): impossible. Because if an interval covers the right-hand point ($x_2 = 1$), it must also cover any point in between. To cover $x_2$ without covering $x_1$? If $x_1 = 0 < x_2 = 1$, any interval containing $x_2$ and all points between it and $x_1$ must… wait — an interval is continuous, but $x_1$ and $x_2$ are discrete points. The interval $[0.5, 1.5]$ contains $x_2$ but not $x_1 = 0$. So (-,+) IS possible! The interval $[0.5, 1.5]$ labels $x_1$ as -1 and $x_2$ as +1. ✓

All 4 labelings are producible. So VC dimension ≥ 2.

3 points: with $x_1 = 0$, $x_2 = 1$, $x_3 = 2$, can we produce (+,-,+)? We'd need to cover $x_1$ and $x_3$ but not $x_2$. But intervals are continuous — any interval covering 0 and 2 must also cover 1. So (+,-,+) is impossible. VC dimension = 2.

**Exercise 2: linear classifiers in the plane (lines through the origin).** $\mathcal{H}_2$: $\text{sign}(w \cdot x)$, $w \in \mathbb{R}^2$. Can it shatter 3 non-collinear points in the plane? Yes — all 8 labelings are achievable. 4 points? No — there exist sets of 4 points that no line through the origin can shatter. VC dimension = 2 (because the line must pass through the origin, one degree of freedom is lost; without the through-the-origin requirement, the VC dimension is 3).

**Exercise 3: the tension between neural networks and VC dimension.** A neural network with $W$ weights has VC dimension on the order of $O(W \log W)$. But modern deep networks routinely have billions of parameters — by PAC theory they would need an astronomical number of samples. In practice, these networks generalize with far fewer samples than the theory demands. This tension is exactly the starting point for Chapter 21's "generalization = compression" viewpoint: the VC dimension measures the complexity in the **worst case**, but an actual network's effective complexity is far below its parameter count — a large share of the parameters is squeezed by structural constraints (weight decay, architecture design, SGD's implicit regularization) into a much smaller effective space.

Put the three exercises side by side, and what do you see? The VC dimension isn't some fuzzy "complexity"; it's a concrete number you can compute by hand. 2, 2 (or 3), billions — these numbers tell you why some hypothesis classes are easy to learn, some need more data, and some leave theorists puzzled. Computability is the best sobriety check.

---

## 20.6 Unifying the Three Contracts

Put this chapter's contents together:

| Contract | Promise | Condition | Cost |
|---------|------|------|------|
| Admissibility + A* | Find the optimal solution | $h$ never overestimates | May explore a larger space |
| Approximation ratio $\rho$ | Solution no more than $\rho$ times the optimum | Worst-case guarantee | Give up optimality |
| PAC learning | Approximately correct with high probability | Enough samples | Allows failure probability $\delta$ |

Three contracts, three different brands of "good enough" — each with its own cost structure and guarantee type. What they share: each turns "approximately" from an intuition into a mathematical quantity. A contract can be violated, but violations are detectable. This is fully in the spirit of Chapter 14's formal systems: precise definitions allow precise criticism.

A heuristic isn't a second-rate compromise forced on us because exact reasoning is too hard. It is a rational commitment to inference quality under computational constraints — knowing what you promise, and where the promise's boundary lies.

---

## 20.7 After the Contract Is Signed: Who Writes the Contract?

This chapter built a reassuring framework: once a contract is signed, a heuristic's behavior is mathematically guaranteed. But there's a question we've been avoiding: **who writes the contract?**

An admissible heuristic $h$ must be designed by hand. PAC learning's hypothesis space $\mathcal{H}$ must be chosen by hand. The approximation ratio of an approximation algorithm must be proved by hand. Every one of these "contracts" comes from **a designer outside the system**.

Back to the language of Chapter 15: in a formal system, axioms must be fed in from outside; the system can't choose its own axioms. In a heuristic contract, the terms of the contract (the form of $h$, the structure of $\mathcal{H}$, the proof of the approximation ratio) also come from outside the system.

Which means: all of today's "guaranteed reasoning" rests on an external promise — you trust that the designer's chosen heuristic is reasonable, and that the chosen hypothesis space is appropriate. But what if that external promise is itself wrong? What if, in some scenario, your heuristic systematically overestimates, and you don't know it?

This question leads to a deeper one: **can the contract be written automatically, from data?** Can we let the system observe its own behavior and infer a suitable heuristic from its successes and failures? If designing a heuristic is itself an inference problem — inferring, from a history of problem instances, which estimation strategy works on future instances — then heuristic design is **meta-learning**.

And meta-learning is precisely the door Chapter 21 opens: can learning be understood as inferring the shortest description from data? If learning is inverse inference, then "learning a good heuristic" — can that itself be understood as a kind of inverse inference?

These three questions — who writes the contract, whether the contract can be written automatically, and whether writing the contract counts as learning — push us from "formalizing heuristics" toward "the inverse-inference structure of learning". Formalizing the contract was this chapter's job; where the contract comes from is the next chapter's job. The final answer is in Chapter 21.

---

## Open Questions

**Can the boundary of approximation be broken?** For some problems, the approximation ratio has an insurmountable lower bound (assuming P ≠ NP). But the theory of approximation algorithms still has plenty of open problems: what is the optimal approximation ratio for the traveling salesman problem? Does the Unique Games Conjecture hold — it pins down approximation lower bounds for a huge number of problems. Questions in this direction are often easier to state than P ≠ NP itself, and just as deep.

**The computational version of PAC learning**: the PAC framework guarantees sample complexity, not computational complexity. Some concept classes are learnable in the sample sense, but the learning algorithm itself may need exponential time. Distinguishing "information-theoretically learnable" from "computationally learnable" is the central tension of learning theory — the former asks how much data you need, the latter asks how much time.

**Can heuristics be discovered automatically?** A* needs a hand-designed $h$, and PAC learning needs a hand-chosen hypothesis space $\mathcal{H}$. Can good heuristic functions be discovered automatically from data, or can a suitable hypothesis space be inferred automatically from problem structure? This turns heuristic design itself into an inference problem — and inference problems are Chapter 21's theme: **can learning be understood as inferring the shortest description from data?**

---

## Exercises

**★ Warm-up**

In A* pathfinding on a map, straight-line distance is commonly used as the heuristic $h$. Say which of the following $h$'s are admissible and which are not, and give a one-sentence reason for each.

1. $h(n) = 0$ (the zero heuristic; A* degenerates into Dijkstra's algorithm)
2. $h(n) =$ straight-line (Euclidean) distance from $n$ to the goal
3. $h(n) =$ straight-line distance from $n$ to the goal $\times 2$
4. $h(n) =$ the actual shortest path length from $n$ to the goal (assuming you already know it)
5. $h(n) =$ Manhattan distance from $n$ to the goal (when only up/down/left/right moves are allowed)

---

**★★ Derivation**

1. **The cost of admissibility**: design a simple search problem (draw a 5-node graph with edge weights) such that an admissible heuristic $h_1$ must expand more nodes to find the optimal solution, while an inadmissible $h_2$ finds the same solution with fewer nodes. Give concrete values for both $h$'s, and trace A*'s expansion order in both cases. What does admissibility guarantee, and what doesn't it guarantee?

2. **PAC vs. Bayes**: Chapter 17's Bayesian inference and this chapter's PAC learning both deal with "generalizing from finite data". PAC's guarantee holds for the worst-case distribution; Bayes's guarantee depends on the prior. Which guarantee assumes less? In a scenario where the prior is completely wrong, which framework breaks down first?

---

**★★★ Challenge**

The general traveling salesman problem (TSP) (when the triangle inequality isn't satisfied) has been proven: if P ≠ NP, then for any constant $\rho$, no polynomial-time $\rho$-approximation algorithm exists.

The proof of this "inapproximability" uses a reduction: if a $\rho$-approximation algorithm existed, it would solve the Hamiltonian cycle problem (NP-complete). Try to sketch the general idea of this reduction in plain language — how would you turn an instance of the Hamiltonian cycle decision problem into an instance of the TSP, so that "finding an approximate solution" is equivalent to "deciding whether a Hamiltonian cycle exists"?

The goal of this exercise isn't a complete proof; it's to understand why allowing error doesn't always make the problem easier — in some cases, approximating is exactly as hard as solving exactly.

# Chapter 20: The Formal Contract of Heuristics — The Precise Mathematical Definition of "Approximately Right"

> Admissibility is not an engineering compromise. It is a mathematical promise.

---

The conclusion of Chapter 19 is disheartening: many reasoning problems are computationally fundamentally hard. Not because the algorithms are not good enough, but because the geometric shape of the problem itself dictates there are no shortcuts.

Yet humans and machines have been solving these "hard" problems all along. Imprecisely, not always optimally, but usually well enough, fast enough.

A question lurks here: **what does "good enough" mean?** If the answer is arbitrary, then any algorithm that doesn't crash can be called a "heuristic." But if "good enough" has a precise mathematical meaning, then a heuristic is not an engineering makeshift, but a contractual promise — promising what, at what cost, under what conditions it will be honored.

What this chapter sets out to do is write this contract clearly.

---

## 20.0 The Optimistic Mapmaker Game: Heuristics Must Sign a Contract

Chapter 19 told us that many mazes are too large to expect exhaustive search. So we hire a mapmaker. The mapmaker cannot tell you the complete path, but can only give an estimate at every intersection: roughly how much farther from here to the destination.

This is the heuristic function $h(n)$.

But this mapmaker must sign a contract. The first contract is called **admissibility**: you may be optimistic, but you must not exaggerate the difficulty. That is, the estimated distance must not exceed the true shortest distance:

$$h(n) \leq h^*(n)$$

If the mapmaker underestimates the distance, the search merely takes a few more steps; if the mapmaker overestimates the distance, the search may directly abandon the genuinely optimal path. The second contract is called **consistency**: estimates between adjacent intersections must satisfy the triangle inequality; one cannot today say "the destination is very close," and after one step suddenly say "the destination is absurdly far."

:::details The Formal Skeleton of This Game
- **State space**: nodes in the search graph, known cost $g(n)$, heuristic estimate $h(n)$, priority queue.
- **Legal moves**: expand the node with the smallest $f(n)=g(n)+h(n)$, and update its successors.
- **Transition rules**: take node $n$ from the open list, add or update successor $n'$, with cost $g(n')=g(n)+c(n,n')$.
- **Victory condition**: find the optimal path under provable guarantees, or obtain an approximation ratio / PAC-quality bound.
- **Failure mode**: the heuristic has no contract, is merely "nice-looking," and ultimately delivers the fastest wrong answer.
:::

This game makes "approximately right" precise. A heuristic without a contract is merely luck; a heuristic with a contract is a mathematical object. Admissibility, consistency, approximation ratio, PAC guarantees — all are different forms of contract clauses.

The dignity of a heuristic lies not in always producing the optimal answer, but in honestly stating: what I promise, what I do not promise; under what conditions I am reliable, under what conditions I will fail.

::: info Professor Pallas's Cat's Commentary
Optimism can be a virtue, or it can be fraud. A* allows the heuristic function to be optimistic, because optimism makes you continue exploring; it does not allow the heuristic function to be arrogant, because arrogance makes you prematurely abandon the true path. The entire ethic of heuristics is hidden inside this inequality.
:::

## 20.1 Two Failures of Approximation

First, let us look at two approximation schemes, both intuitively reasonable, both wrong under certain conditions.

**Scheme One: Greedy algorithm**. At each step, choose the locally best-looking option, no backtracking. In the traveling salesman problem, at each step go to the nearest unvisited city; in graph coloring, assign the current vertex the smallest-numbered color that doesn't conflict with neighbors.

Greedy algorithms are typically fast (polynomial time), but the results can be very poor — sometimes off from the optimal solution by an exponential factor. Greed makes no promise; it is merely "nice-looking."

**Scheme Two: Random search**. Randomly sample the solution space, report the best result found. Random enough, enough times, perhaps a good solution can be found.

Random search also makes no promise — "perhaps" is not a contract. In the worst case, good solutions in the solution space are extremely sparse, and random search may never find them.

::: info Professor Pallas's Cat's Commentary
A great many practical systems are using exactly "perhaps" — random restarts, multiple sampling, taking the best result. These methods sometimes work in practice, but you don't know under what circumstances they will fail, nor what the probability of failure is. Without a contract, there is no failure condition; you can't even tell whether the algorithm is working. "Sometimes works" is not a guarantee; it is luck.
:::

The common problem of these two schemes: **no provable bound on "how far off."** What heuristic contracts need is precisely this bound.

### A Concrete Failure Scene

Use a minimal Traveling Salesman Problem (4 cities, with edge weights as follows) to see clearly the difference between greedy and random:

```
Cities: A, B, C, D
Edge weights (distances):
  A-B: 10, A-C: 15, A-D: 20
  B-C: 35, B-D: 25
  C-D: 30
Optimal Hamiltonian cycle: A->B->D->C->A, total cost = 10+25+30+15 = 80
```

**Greedy (nearest-neighbor) starting from A**: A to B (10, shortest) -> B to D (25, shortest excluding A, since B to C is 35) -> D to C (30) -> forced back to A (15). Path: A->B->D->C->A, cost = 10+25+30+15 = 80. It happened to get it right this time, because the graph is tiny.

But try a different starting point: **starting from B** -> B to A (10) -> A to C (15) -> C to D (30) -> forced back to B (35). Path: B->A->C->D->B, cost = 10+15+30+35 = 90. Suboptimal. And this is only 4 cities; as the number of cities grows, the greedy error can scale exponentially.

**Random search**: sample 100 tours at random. The best might be 85, or 110—no commitment. Run it again, get a different answer each time.

Neither has a contract. Greedy is right only on some graphs; random is never certain. What's missing is precisely that inequality: **$h(n) \leq h^*(n)$**. With it, you know where the boundary of optimism lies.

::: info Professor Manul
See that? Greedy on this tiny graph is right starting from A, wrong starting from B. You can't use "it worked on A" to prove the algorithm is effective—an algorithm without a contract is not "roughly effective"; rather, "you don't know when it will fail." That can be fatal.
:::

---

## 20.2 Admissibility: Never Overestimate

The most classic heuristic framework in search problems is the A* algorithm. Its core is a **heuristic function** $h(n)$, estimating the cost from the current state $n$ to the goal.

The A* algorithm maintains a priority queue, expanding at each step the node with the smallest estimated total cost $f(n) = g(n) + h(n)$, where $g(n)$ is the known actual cost from the start to $n$, and $h(n)$ is the estimated cost from $n$ to the goal.

The condition for this algorithm to find the optimal solution rests precisely on one property of $h$:

**Definition (Admissibility)**: A heuristic function $h$ is **admissible** if, for all nodes $n$,

$$h(n) \leq h^*(n)$$

where $h^*(n)$ is the true optimal cost from $n$ to the goal.

The meaning of admissibility: $h$ never overestimates. It may underestimate — underestimation means optimism, and optimism means this path will continue to be explored. But if it overestimates, the algorithm may prematurely abandon a path that is actually optimal.

**Theorem (Optimality of A*)** : If the heuristic function $h$ is admissible, then the A* algorithm, when a solution exists, will certainly find the optimal solution.

This is the first form of the heuristic contract: **give me an estimate that never overestimates, and I guarantee you the optimal solution**. The cost is that you must spend time constructing this estimate, and the algorithm may need to explore a larger search space.

::: info How Admissibility Is Violated
A common mistake is to use a rough upper bound on the true cost as the heuristic function — this will of course overestimate, thereby destroying admissibility. For instance, in map search, if straight-line distance is used as $h$, this is admissible (straight-line distance never exceeds actual path distance); if an estimated value causes overestimation on winding roads, it is not admissible. Violations of admissibility are typically not obvious — they depend on the specific structure of the problem.
:::

### 20.2.5 Hands-On: Tracing a Complete A* Search

Reading the definition is not enough. Below, we trace every step of A* on a 6-node directed graph. I suggest you take out a piece of paper and follow along.

```
Nodes: S (start), A, B, C, D, G (goal)
Edges (from -> to, cost):
  S->A: 1,  S->B: 4
  A->C: 3,  A->D: 2
  B->D: 2
  C->G: 2
  D->G: 4

Heuristic function h (estimated cost to G):
  h(S)=5, h(A)=4, h(B)=4, h(C)=2, h(D)=2, h(G)=0

True optimal costs h* (verifiable via reverse Dijkstra):
  Shortest: S->A->C->G = 6, so h*(S)=6, h(S)=5 ≤ 6 ✓ admissible
  h*(A)=5 (A->C->G=3+2=5), h(A)=4 ≤ 5 ✓
  h*(B)=6 (B->D->G=2+4=6), h(B)=4 ≤ 6 ✓
  h*(C)=2, h(C)=2 ✓ (exact)
  h*(D)=4, h(D)=2 ≤ 4 ✓
  h*(G)=0, h(G)=0 ✓

All nodes satisfy h(n) ≤ h*(n); h is admissible.
```

**Tracing A***:

Initial: priority queue = [(S, f=0+5=5)]. g(S)=0.

**Step 1**: Expand S (f=5 is smallest). Successors:
- A: g=1, f=1+4=5 -> add to queue
- B: g=4, f=4+4=8 -> add to queue
Queue: [(A,5), (B,8)]

**Step 2**: Expand A (f=5 smallest). Successors:
- C: g=1+3=4, f=4+2=6 -> add
- D: g=1+2=3, f=3+2=5 -> add
Queue: [(D,5), (C,6), (B,8)]

**Step 3**: Expand D (f=5 smallest). Successors:
- G: g=3+4=7, f=7+0=7 -> add
Queue: [(C,6), (G,7), (B,8)]

**Step 4**: Expand C (f=6 smallest). Successors:
- G: g=4+2=6, f=6+0=6 -> update G (existing version has f=7; new f=6 is better, replace)
Queue: [(G,6), (B,8)]

**Step 5**: Expand G (f=6 smallest). G is the goal; path found: S->A->C->G, total cost 6.

Notice: at Step 3, A* did not take the path S->A->D->G (total cost = 7), because after D was expanded, G had f=7, while C in the queue had the better f=6. A* correctly skipped the suboptimal path.

Now, what if we change h to be non-admissible? Set $h'(A)=6$ (an overestimate, since true h*=5). Then:
- After expanding S: A's f=1+6=7, B's f=4+4=8
- Queue: [(A,7), (B,8)]
- Expand A (f=7): D has f=5, C has f=6
- Queue: [(C,6), (D,7), (B,8)]
- Expand C: G has f=6... We still found the optimum, but only because the graph is tiny.

On more complex graphs, overestimation causes A* to prematurely deem a certain path "not worth exploring"—and that path happens to be the optimal one. Admissibility does not guarantee efficiency; admissibility guarantees **the optimum will not be missed**. Lose it, and the optimum can slip through your fingers without you ever knowing.

::: info Professor Manul
Look carefully at Step 3: when expanding D, the queue contains C(f=6) and D(f=5). A* chooses D because D is more optimistic. Optimism makes A* willing to take risks, but also lets it turn back in time when it sees a better path—because optimism adjusts downward, not upward. If h is arrogant (overestimates), A* discards certain nodes prematurely and never looks back. Optimism is the engine of exploration; arrogance is the cause of missed opportunities.
:::

---

## 20.3 Consistency: The Constraint of the Triangle Inequality

Admissibility guarantees the quality of the final result, but the efficiency of A* also depends on another property of $h$.

**Definition (Consistency)**: A heuristic function $h$ is **consistent** if, for all nodes $n$ and its successor $n'$,

$$h(n) \leq c(n, n') + h(n')$$

where $c(n, n')$ is the actual edge cost from $n$ to $n'$.

This is a triangle inequality: the estimated cost from $n$ to the goal must not exceed the sum of first going to $n'$ and then the estimated cost from $n'$ to the goal. Intuitively, consistency says: your estimates for each node are "coherent" — there will be no situation where, after taking one step, the estimated value suddenly skyrockets.

Consistency implies admissibility (it can be verified), but the converse does not hold. Consistency is a stronger condition.

**Under consistency, the $f$-values of nodes expanded by A* are non-decreasing**. This guarantees: the first time A* reaches the goal node, it has already found the optimal path, requiring no further search. Consistency advances A* from "eventually finds the optimal" to "finds the optimal as early as possible."

These two properties — admissibility and consistency — constitute the precise mathematical content of the heuristic contract. With them, a "heuristic" is no longer a casual guess, but an estimation mechanism operating within provable bounds.

Returning to the Optimistic Mapmaker Game, admissibility stipulates that the mapmaker cannot call a short road a long road; consistency stipulates that he cannot arbitrarily change his story between adjacent intersections. The cleverness of A* lies not in "trusting intuition," but in trusting only intuition that has signed a contract. Without this contract, a heuristic is merely a guide who talks a good game; with this contract, it becomes a legitimate move within the formal system.

### 20.3.5 When Consistency Is Violated: A*'s Confusion

Using the same graph as in 20.2.5, construct an admissible but inconsistent h:

Set $h''(S)=5, h''(A)=4, h''(B)=5, h''(C)=2, h''(D)=1, h''(G)=0$.

Check admissibility: $h''(n) \leq h^*(n)$ for all?
- $h''(B)=5 \leq 6$ ✓, $h''(D)=1 \leq 4$ ✓. Yes, admissible.

Check consistency: for each edge $(n, n')$, $h''(n) \leq c(n,n') + h''(n')$?
- Edge B->D: $h''(B)=5$, $c(B,D)+h''(D)=2+1=3$. $5 \leq 3$? **Violation.**

What happened here? From B, the estimate to G appears to be 5; but from D (one step away from B), the estimate suddenly becomes 1. The mapmaker said "still far" at B, and after one step to D, changed his tune to "almost there." This flip confuses A*:

From S: A(f=5), B(f=4+5=9). Choose A. Then D's f=1+2+1=4. Choose D. Then G's f=3+4+0=7. Choose C(f=6). C->G(f=6). Optimal path S->A->C->G found, cost 6.

Looks fine? But the problem appears in this detail: after D is expanded, the queue has C(f=6) and G(f=7, via D). If we later discover a shorter path to D, with h'' inconsistent, we might need to **re-expand** already expanded nodes. Consistency guarantees f-values are non-decreasing, so the first arrival at the goal is optimal; without consistency, A* may need to keep searching past the goal node until the f-value exceeds the current best cost.

In larger-scale search, the cost of inconsistency is enormous: A* may need to re-expand nodes, causing the algorithm to degrade toward Dijkstra's efficiency. Consistency is not decoration—it is A*'s efficiency contract.

::: info Professor Manul
An inconsistent heuristic function is like a capricious tour guide. He tells you at B that there are still 5 km remaining; you walk to D, and he says actually only 1 km remains. Added together, the B->D stretch only "cost" 2 km (5->1), and the actual distance is also 2 km—it "happens" to match. But if at D he suddenly says there are still 4 km remaining (to the next intersection), you'd start to wonder: does he even know what he's talking about? A*'s doubt is mathematical: when consistency fails, f-values can decrease, and the algorithm no longer guarantees the first solution is optimal.
:::

---

## 20.4 Approximation Ratio: Worst-Case Guarantees

The A* framework works well on problems where optimality is meaningful. But for many NP-complete problems, the optimal solution itself is hard to find, even when approximation is allowed. Here another kind of contract is needed: **approximation ratio**.

**Definition (Approximation algorithm)**: For a minimization problem, if algorithm $\mathcal{A}$ always returns a solution whose cost does not exceed $\rho$ times the optimal solution cost on any instance, then $\mathcal{A}$ is called a $\rho$-approximation algorithm; $\rho$ is called the **approximation ratio**.

The approximation ratio is the worst-case version of the heuristic contract: regardless of the input, the quality of the output is guaranteed within a factor of $\rho$ of the optimal. $\rho = 1$ is the exact solution; $\rho = 2$ is "at most twice the optimal."

Some famous results:

- **Vertex cover problem**: there exists a 2-approximation algorithm (find a maximal matching, take both endpoints).
- **Traveling salesman problem (metric version)**: there exists a 1.5-approximation algorithm (Christofides' algorithm, 1976). This result stood for nearly half a century, only slightly improved in 2020.
- **Traveling salesman problem (general version, not satisfying the triangle inequality)**: if P ≠ NP, then for any constant $\rho$, no polynomial-time $\rho$-approximation algorithm exists.

This last result is crucial: for certain problems, approximation is equally NP-hard. Allowing error does not always make the problem easier. **The approximation ratio itself has an information-theoretic lower bound**, a bound arising from the intrinsic structure of the problem, independent of any specific algorithm.

::: info The Unbreakability of Approximation Ratios
The proof of the inapproximability of the general version of the traveling salesman problem uses reduction: if a polynomial-time $\rho$-approximation algorithm existed, it could be used to decide whether a Hamiltonian cycle exists — and the Hamiltonian cycle problem is NP-complete. This reduction shows that any sufficiently good approximation implies exact solving capability. The intrinsic structure of the problem leaves its traces at the approximation level.
:::

### 20.4.5 Hands-On: Vertex Cover 2-Approximation — The Power of Maximal Matching

The Vertex Cover problem: given a graph $G=(V,E)$, find the smallest vertex set $C \subseteq V$ such that every edge has at least one endpoint in $C$. This is an NP-complete problem. Yet there exists a simple 2-approximation algorithm—remember, 2 means the set you find is at most twice the size of the optimal solution.

**Algorithm (maximal matching method)**:
1. Find a **maximal matching** $M$ of the graph (a matching to which no further edge can be added)
2. Output both endpoints of every edge in $M$

**Why is this a 2-approximation?**

Let $M$ be the maximal matching found, containing $k$ edges. The output set $C$ contains the $2k$ endpoints of these $k$ edges.

- **Lower bound**: The optimal vertex cover needs at least $k$ vertices. Because the $k$ edges in $M$ share no endpoints (property of a matching), covering these $k$ edges requires at least $k$ vertices—at least one endpoint per edge, and edges from different pairs cannot share vertices. Thus $\text{OPT} \geq k$.
- **Upper bound**: We output $2k$ vertices, $|C| = 2k \leq 2\cdot\text{OPT}$. ✓

**Why is maximal (not maximum) sufficient?** A simple greedy construction of a maximal matching: iterate through all edges; if neither endpoint of the current edge is already covered by the matching, add this edge to the matching. The final result may not be a maximum matching (the matching with the most edges), but it is certainly maximal (no further edge can be added). For a 2-approximation, maximal is sufficient.

**A concrete example**:

```
Graph: vertices {1,2,3,4,5}
Edges: (1,2), (2,3), (3,4), (4,5), (1,5)
```

Greedy maximal matching (iterating edges in order):
- Encounter (1,2): both 1 and 2 unmatched -> add to M. M={(1,2)}
- Encounter (2,3): 2 already matched -> skip
- Encounter (3,4): both 3 and 4 unmatched -> add to M. M={(1,2), (3,4)}
- Encounter (4,5): 4 already matched -> skip
- Encounter (1,5): 1 already matched -> skip

Output C = {1,2,3,4}, size 4.

What is the optimal solution? This graph is a 5-vertex cycle. To cover all 5 edges, at least 3 vertices are needed (taking every other vertex, e.g., {2,4,5} covers all edges). So OPT=3. Our solution is 4, approximation ratio 4/3≈1.33 ≤ 2. ✓

What if we traverse edges in a different order? If we encounter (2,3) first, then (1,2) and (3,4), M may become {(2,3),(4,5)} or {(2,3),(1,5)}; the output size is still 4. Different maximal matchings yield different solutions, all within a factor of 2.

::: info Professor Manul
Notice the elegance of this algorithm: it makes no attempt to "cleverly" select vertices; it relies on an exceedingly simple fact—the size of the matching is a natural lower bound on OPT. This lower bound is not computed; it is provided by the structure of the graph itself. Good approximation algorithms are often not about "cleverly approximating the optimum," but about "finding a simple structure that is necessarily close to the optimum." This shares a deep resonance with Lyapunov functions—both find a quantity that naturally decreases (or naturally bounds).
:::

---

## 20.5 PAC Learning: The Probabilistic Version of "Approximately Right"

Approximation ratios concern the quality of solutions; the PAC learning framework applies the same spirit to learning problems.

**Definition (PAC Learning)**: A concept class $\mathcal{C}$ is **PAC-learnable** (Probably Approximately Correct) if there exists an algorithm $\mathcal{L}$ such that: for any target concept $c \in \mathcal{C}$, any data distribution $\mathcal{D}$, and any parameters $\varepsilon > 0$ (error tolerance) and $\delta > 0$ (failure probability), when the sample size $m$ is sufficiently large (polynomial in $\varepsilon, \delta$), $\mathcal{L}$ outputs, with probability at least $1 - \delta$, a hypothesis $h$ such that

$$P_{x \sim \mathcal{D}}[h(x) \neq c(x)] \leq \varepsilon$$

In plain language: with high probability (at least $1-\delta$), output a hypothesis that is approximately correct (error rate at most $\varepsilon$). PAC is the abbreviation of "Probably Approximately Correct" — this is not self-deprecation, but a precise mathematical promise.

::: info Professor Pallas's Cat's Commentary
Many people, upon seeing "probably approximately correct," think this is lowering the standard. Quite the opposite — $\varepsilon$ and $\delta$ are not fuzzy words, but numbers that can be computed. This is more honest than many methods that claim "accuracy": it knows what it is promising, and knows where the promise will break. Knowing one's boundaries is far more honest than pretending there are none.
:::

The core conclusion of the PAC framework: the sample size $m$ need only grow polynomially in $1/\varepsilon$ and $1/\delta$ for learning to succeed. This means: **infinite data are not required; a finite, polynomial quantity of data suffices to learn an approximately correct concept with high probability**.

::: info VC Dimension: The Complexity of Hypothesis Spaces
In the PAC framework, the required sample size also depends on another quantity: the **VC dimension** (Vapnik-Chervonenkis dimension) of the hypothesis space $\mathcal{H}$. The VC dimension measures the largest set of points that $\mathcal{H}$ can "shatter" — being able to shatter a set of $d$ points means that for any labeling of these points, $\mathcal{H}$ contains a hypothesis that can perfectly fit them. The higher the VC dimension, the more complex the hypothesis space, and the more samples are needed to constrain it. The basic PAC sample complexity theorem: $m = O\!\left(\frac{1}{\varepsilon}\left(d \ln \frac{1}{\varepsilon} + \ln \frac{1}{\delta}\right)\right)$, where $d$ is the VC dimension. This turns the "overfitting" intuition from Chapter 5 (Volume 1) — hypothesis space too complex, need more data — into a computable bound.
:::

### 20.5.5 Hands-On: Three Exercises on VC Dimension

**Example 1: Intervals on the real line**. Hypothesis space $\mathcal{H}_1$ is closed intervals $[a,b]$ on the real line: for a point $x$, predict +1 if $x \in [a,b]$, otherwise -1. How large a set of points can it shatter?

2 points: place them at $x_1=0, x_2=1$. Can all 4 labelings be produced?
- (+,+): interval $[0,1]$ ✓
- (-,-): interval $[2,3]$ (covers no points) ✓
- (+,-): interval $[0,0]$ ✓
- (-,+): interval $[0.5, 1.5]$ covers $x_2$ but not $x_1=0$ ✓

All 4 are possible. So VC dimension ≥ 2.

3 points: at $x_1=0, x_2=1, x_3=2$. Can we produce (+,-,+)? Need to cover $x_1$ and $x_3$ but not $x_2$. But intervals are continuous—any interval covering 0 and 2 must cover 1. So (+,-,+) is impossible. VC dimension = 2.

**Example 2: Linear classifiers in the plane** (lines through the origin). $\mathcal{H}_2$: $\text{sign}(w \cdot x)$, $w \in \mathbb{R}^2$. Can it shatter 3 non-collinear points in the plane? Yes—all 8 labelings can be realized. 4 points? No—there exists a set of 4 points that no line through the origin can shatter. VC dimension = 2 (because the line must pass through the origin, losing one degree of freedom; lines not required to pass through the origin have VC dimension = 3).

**Example 3: The tension between neural networks and VC dimension**. A neural network with $W$ weights has a VC dimension on the order of $O(W \log W)$. But modern deep networks have billions of parameters—by PAC theory, they would need astronomical numbers of samples. Yet in practice, these networks generalize with far fewer samples than theory demands. This tension is precisely the starting point for Chapter 21's "generalization = compression" perspective: the VC dimension measures complexity in the **worst case**, but the effective complexity of real networks is far lower than their parameter count—most parameters are compressed into a far smaller effective space by structural constraints (weight decay, architecture design, implicit regularization of SGD).

Lay these three exercises side by side—what do you see? The VC dimension is not a vague "complexity" but a concrete number you can compute by hand. 2, 2 (or 3), billions—these numbers tell you why some hypothesis classes are easy to learn, why some need more data, and why some baffle theorists. Computability is the best sobering agent.

---

## 20.6 The Unification of the Three Contracts

Let us lay out the content of this chapter side by side:

| Contract type | Promise | Condition | Cost |
|---------|------|------|------|
| Admissibility + A* | Find optimal solution | $h$ never overestimates | May explore larger space |
| Approximation ratio $\rho$ | Solution within $\rho$ times optimal | Worst-case guarantee | Sacrifice optimality |
| PAC learning | Probably approximately correct | Sufficiently many samples | Allow failure probability $\delta$ |

Three contracts, three different kinds of "approximately right" — corresponding to three different cost structures and guarantee types. Their commonality is: all turn "approximately" from an intuitive word into a mathematical quantity. Contracts can be violated, but violations are detectable. This is entirely consistent with the spirit of the formal system in Chapter 14: precise definitions enable precise criticism.

Heuristics are not defective products forced upon us because exact reasoning is too hard. They are rational promises about reasoning quality under computational constraints — knowing what they promise, knowing where the boundaries of their promises lie.

---

## 20.7 After the Contract Is Signed: Who Writes the Contract?

This chapter has established a reassuring framework: once a contract is signed, the behavior of the heuristic has mathematical guarantees. But there is a question we have been avoiding: **who writes the contract?**

The admissible heuristic function $h$ must be designed by a human. The hypothesis space $\mathcal{H}$ for PAC learning must be chosen by a human. The approximation ratio of an approximation algorithm must be proven by a human. These "contracts" all come from **a designer outside the system**.

Returning to the language of Chapter 15: in a formal system, axioms must be supplied from outside; the system itself cannot choose its axioms. In the heuristic contract, the contract terms (the form of $h$, the structure of $\mathcal{H}$, the proof of the approximation ratio) likewise come from outside the system.

This means: all current "guaranteed reasoning" depends on an external commitment—you trust that the designer chose a reasonable heuristic function, trust that the chosen hypothesis space is appropriate. But what if this external commitment itself is wrong? What if, in some scenario, your heuristic function systematically overestimates, and you don't know it?

This question leads to a deeper one: **can contracts be automatically written from data?** Can a system observe its own behavior and infer suitable heuristics from its successes and failures? If heuristic design is itself an inference problem—inferring, from a history of problem instances, what kind of estimation strategy will be effective on future instances—then heuristic design is **meta-learning**.

And meta-learning is precisely the door that Chapter 21 will open: can learning be understood as inferring the shortest description from data? If learning is inverse inference, then can "learning a good heuristic" itself be understood as a form of inverse inference?

These three questions—who writes the contract, can contracts be written automatically, does writing contracts itself count as learning—push us from "the formalization of heuristics" toward "the inverse-inferential structure of learning." The formalization of contracts is the work of this chapter; where contracts come from is the work of the next chapter. The final answer is in Chapter 25.

---

## Unresolved

**Can the boundaries of approximation be broken?** For some problems, the approximation ratio has an insurmountable lower bound (assuming P ≠ NP). But approximation algorithm theory still has vast unresolved questions: what is the optimal approximation ratio for the traveling salesman problem? Does the Unique Games Conjecture hold — it determines the approximation lower bounds for a great many problems. Questions in this direction are often easier to state than P ≠ NP itself, and equally deep.

**The computational version of PAC learning**: The PAC framework guarantees sample complexity, but not computational complexity. Some concept classes are learnable in the sample sense, but the learning algorithm itself may require exponential time. Distinguishing "information-theoretically learnable" from "computationally learnable" is the core tension of learning theory — the former asks how much data is needed, the latter asks how much time is needed.

**Can heuristics be automatically discovered?** A* requires human-designed $h$, PAC learning requires human-chosen hypothesis space $\mathcal{H}$. Can good heuristic functions be automatically discovered from data, or appropriate hypothesis spaces automatically inferred from problem structure? This question turns heuristic design itself into an inference problem — and inference problems are precisely the theme of Chapter 21: **can learning be understood as inferring the shortest description from data?**

---

## Exercises

**★ Warm-up**

The A* algorithm commonly uses straight-line distance as the heuristic function $h$ in map pathfinding. Determine which of the following $h$ are admissible and which are not, and give a one-sentence justification.

1. $h(n) = 0$ (zero heuristic, degenerates into Dijkstra's algorithm)
2. $h(n) =$ straight-line distance from $n$ to the goal (Euclidean distance)
3. $h(n) =$ straight-line distance from $n$ to the goal $\times 2$
4. $h(n) =$ the true shortest path length from $n$ to the goal (assuming you already know it)
5. $h(n) =$ Manhattan distance from $n$ to the goal (only allowing up/down/left/right movement)

---

**★★ Derivation**

1. **The cost of admissibility**: Design a simple search problem (draw a 5-node graph, label edge weights), such that: an admissible heuristic function $h_1$ requires expanding more nodes to find the optimal solution, while a non-admissible $h_2$ finds the same solution with fewer nodes. Write out the specific values of the two $h$'s, and trace the expansion order of A* in both cases. What does this illustrate about what admissibility guarantees, and what it does not guarantee?

2. **The relationship between PAC and Bayesian approaches**: Bayesian inference from Chapter 17 and PAC learning from this chapter both deal with "generalizing from finite data." PAC guarantees hold for the worst-case distribution; Bayesian guarantees depend on the prior. Which kind of guarantee makes fewer assumptions? In a scenario where the "prior is completely wrong," which framework will collapse first?

---

**★★★ Challenge**

The general version of the traveling salesman problem (TSP) (not satisfying the triangle inequality) has been proven: if P ≠ NP, then for any constant $\rho$, no polynomial-time $\rho$-approximation algorithm exists.

The proof of this "inapproximability" uses reduction: if a $\rho$-approximation algorithm existed, it could solve the Hamiltonian cycle problem (NP-complete). Try to describe, in natural language, the rough outline of this reduction — how would you transform an instance of the Hamiltonian cycle decision problem into an instance of the traveling salesman problem, such that "finding an approximate solution" is equivalent to "deciding whether a Hamiltonian cycle exists"?

The goal of this exercise is not to write out the complete proof, but to understand: **why allowing error does not always make the problem easier** — in some cases, approximation is just as hard as exact solution.

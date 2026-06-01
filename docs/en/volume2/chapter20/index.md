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

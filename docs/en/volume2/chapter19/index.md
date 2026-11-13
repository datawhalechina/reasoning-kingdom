# Chapter 19: Complexity as the Geometry of Reasoning — Why Some Reasoning Can't Be Sped Up at All

> P ≠ NP is not a claim about how fast machines run. It's a claim about the intrinsic structure of problems.

---

The end of Chapter 18 left us with an uncomfortable observation: even with a complete causal graph in hand, doing inference on the graph — identifying causal effects, computing identifiable expressions — becomes dramatically more expensive in complex cases. It's not that the algorithms are written badly; the problem itself is hard.

This "hard in itself" needs a precise mathematical meaning.

In the setting of formal systems, "hard" has a quantified version: how many steps does it take to get from the axioms to a theorem? Chapter 14 noted that verifying a proof can be done in polynomial time, but there is no universal efficient algorithm for finding one. This asymmetry has a formal name in computational theory: the **separation of P and NP** — if it holds.

What this chapter does is turn that intuition of "hard" into a precise geometric picture.

---

## 19.0 The Seeker and Ticket-Checker Game: Why Verifying Is Easier Than Finding

Complexity theory can start from a very simple game.

A golden ticket is hidden in a maze. The seeker's job is to find it; the ticket checker's job is to check whether a ticket someone hands over is genuine. The seeker may have to feel around in an exponential number of branching corridors. The ticket checker has it easy: the moment a ticket arrives, he just checks whether the stamp, the serial number, and the path record follow the rules.

That's the intuitive difference between P and NP.

If a problem is in P, the seeker can find the answer on his own in polynomial time. If a problem is in NP, then at least when the answer is "yes," someone can hand the ticket checker a certificate of polynomial length, and the checker can quickly confirm it.

NP-complete problems are something even more fearsome: they're like a universal maze that every NP maze can be efficiently translated into. If you can quickly find the golden ticket in this maze, you can quickly find the golden ticket in every NP maze.

:::details The Formal Skeleton of This Game
- **State space**: an input instance $x$, a candidate certificate $w$, and the verifier's state.
- **Legal moves**: the seeker generates a candidate $w$; the ticket checker examines $V(x,w)$ in polynomial time.
- **Transition rules**: the search process moves through candidate space; the verification process scans the certificate according to a deterministic algorithm.
- **Win condition**: the seeker finds a certificate such that $V(x,w)=1$, or proves that no such certificate exists.
- **Failure mode**: the certificate is easy to check, but the candidate space is too large to exhaust; verifying fast doesn't mean finding fast.
:::

This game spells out the cold part of complexity theory: what's hard isn't "confirming the answer once you know it," but "finding the answer when you don't know it." Proofs, planning, SAT, combinatorial search — all of them get stuck on this asymmetry.

The halting problem pushes the game deeper still: some mazes aren't too big — the problem is that you simply cannot design a ticket checker who can always decide whether the seeker will get out. That isn't NP-hardness; that's the boundary of computability.

::: info Professor Pallas's Cat's Commentary
The sharpest thing about the ticket-checker game is that it humiliates hindsight. Once the answer is laid out, everyone finds it obvious; the real difficulty hides before the answer appears. What P and NP ask is this: is there a principled chasm between creating and checking?
:::

## 19.0.5 Playing the Game in Code

The maze is small enough to exhaust. The program below runs subset-sum on 18 numbers: the ticket checker verifies in a handful of additions, while the seeker has to walk the candidate space — and every operation is counted.

```python
#!/usr/bin/env python3
"""
seeker_ticket_checker.py -- the Seeker and Ticket-Checker Game of Ch. 19.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 19, Section 19.0
"The Seeker and Ticket-Checker Game: Why Verifying Is Easier Than Finding."
A golden ticket hides in a maze. The seeker must search corridor after
corridor; the ticket checker only glances at a handed-in ticket. This
program runs the game on a real NP problem, subset-sum:
  * the TICKET CHECKER verifies a claimed subset in linear time,
  * the SEEKER must exhaust candidates -- and we count every operation,
  * a fake ticket gets refused on the spot.
The asymmetry you will see is the cold part of complexity theory: what is
hard is not confirming the answer once you know it, but finding it when
you do not. (NP-completeness: subset-sum is one of the universal mazes
every NP problem can be translated into -- one line here, a chapter in
the book.)
Run:  python3 seeker_ticket_checker.py
"""
from itertools import combinations

NUMBERS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
TARGET = 128  # = 2 + 7 + 19 + 41 + 59, a sum some subset actually reaches


def verify(numbers, target, ticket):
    """The ticket checker: is this subset a genuine ticket?
    ticket: a tuple of indices. Cost: one addition per element.
    Returns (ok, operations_used)."""
    ops = 0
    total = 0
    for i in ticket:
        total += numbers[i]
        ops += 1
    return total == target, ops


def seek(numbers, target):
    """The seeker: exhaust candidate subsets until one sums to target.
    Returns (ticket, candidates_tried, operations_used). Real, counted."""
    ops = 0
    tried = 0
    for size in range(len(numbers) + 1):
        for ticket in combinations(range(len(numbers)), size):
            tried += 1
            total = 0
            for i in ticket:
                total += numbers[i]
                ops += 1
            if total == target:
                return ticket, tried, ops
    return None, tried, ops


def demo():
    print("=" * 70)
    print(" THE SEEKER AND TICKET-CHECKER GAME -- subset-sum, run for real")
    print(" The Reasoning Kingdom, Volume 2, Chapter 19 (Sec. 19.0)")
    print("=" * 70)
    print(f"\nThe maze: {len(NUMBERS)} numbers, find a subset summing to {TARGET}.")
    print(f"  numbers: {NUMBERS}")

    # The checker's easy life: a real ticket, then a fake one.
    real_ticket = (0, 3, 7, 12, 16)          # 2 + 7 + 19 + 41 + 59
    ok, ops = verify(NUMBERS, TARGET, real_ticket)
    print(f"\nTicket checker, genuine ticket {real_ticket}:")
    print(f"  verdict: {ok}, operations used: {ops}")
    assert ok

    fake_ticket = (1, 4, 9)
    ok, ops = verify(NUMBERS, TARGET, fake_ticket)
    print(f"Ticket checker, fake ticket {fake_ticket}:")
    print(f"  verdict: {ok}, operations used: {ops} -- refused on the spot")
    assert not ok

    # The seeker's hard life.
    ticket, tried, ops = seek(NUMBERS, TARGET)
    print("\nSeeker, searching corridor by corridor:")
    print(f"  candidates tried: {tried}")
    print(f"  operations used:  {ops}")
    print(f"  ticket found:     {ticket}")
    print(f"  check the find:   {verify(NUMBERS, TARGET, ticket)}")
    assert ticket is not None

    print("\nThe asymmetry, in one glance:")
    print(f"  checking the answer cost {len(real_ticket)} additions;")
    print(f"  finding it cost {ops} additions across {tried} candidates.")
    print("  The candidate space has 2**18 = 262144 subsets; the checker")
    print("  never has to look at 262143 of them. That gap is what P and")
    print("  NP are about: creating is not the same job as checking.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
```

---

## 19.1 The Intrinsic Structure of Problems

Let's start with a distinction that may come as a surprise.

There are two kinds of "hard," superficially similar, fundamentally different.

**The first kind**: the problem is large, the data is vast, and the computer needs to run for a long time. This is difficulty in the engineering sense, and it can be eased with faster hardware, smarter algorithms, and more parallel computing. This kind of hardness has no principled boundary.

**The second kind**: no matter which algorithm you use, no matter how fast the hardware, some problems simply require exponential time. This is difficulty in the mathematical sense — not an engineering issue, but a consequence of the problem's intrinsic structure.

Complexity theory cares about the second kind. Its question isn't "is this algorithm fast?" — it's "can this problem be solved fast by any algorithm whatsoever?"

::: info Professor Pallas's Cat's Commentary
Many people hear P ≠ NP and think "computers aren't fast enough." That's completely wrong. P ≠ NP says: even if you're given a computer of unlimited speed, some problems still need exponential time. Speed can't buy structure. The geometry of a problem doesn't change when machines get faster.
:::

To give "fast" a precise meaning, you need a measuring scale.

**Time complexity**: for a problem with input size $n$, how many steps does the best algorithm need to produce the answer? If an algorithm exists whose step count is a polynomial in $n$ ($n^2, n^3, n^{100}$……), the problem is called **polynomial-time solvable**; if every known algorithm needs an exponential number of steps ($2^n, n!, \ldots$), the problem quickly becomes intractable in practice as $n$ grows.

The gap between polynomial and exponential is the central dividing line of complexity theory. At $n=100$, $n^{10} = 10^{20}$ is already enormous; $2^{100} \approx 10^{30}$, more than the number of atoms in the universe. As $n$ grows further, this gap only becomes more absurd.

### 19.1.5 Facing the Numbers: What Polynomial vs. Exponential Actually Means

Putting actual numbers on the table beats any abstract talk of "exponential explosion." Suppose each operation takes 1 nanosecond ($10^{-9}$ seconds).

| Input size $n$ | $n^2$ (polynomial) | $2^n$ (exponential) |
|:------------|:---------------|:------------|
| 10 | 100 ns | 1 μs |
| 20 | 400 ns | 1 ms |
| 50 | 2.5 μs | 13 days |
| 100 | 10 μs | $4 \times 10^{13}$ years |
| 200 | 40 μs | … |

At $n=100$, $2^{100}$ steps would take about 3,000 times longer than the age of the universe. This isn't "a bit slow" — this is **physically impossible**.

But wait — $n^{10}$ is also a polynomial. At $n=100$, $10^{20}$ nanoseconds ≈ 3,171 years. So polynomials aren't necessarily fast. The key difference is this: a polynomial's growth is bounded by a fixed exponent $k$ ($n^k$), while in an exponential function the exponent itself is what grows ($k^n$). As $n$ increases, a polynomial is always left behind by an exponential — that's an asymptotic dividing line, not a matter of absolute speed.

More importantly, most "naturally" occurring polynomial-time algorithms have small exponents ($n, n^2, n^3, n\log n$), not $n^{100}$. In the theoretical framework, the boundary between "polynomial" and "exponential" captures the practical divide between "solvable" and "unsolvable" — even though theory admits that $n^{100}$ is a polynomial, in practice nobody would ever use it.

---

## 19.2 P and NP: Two Kinds of Capability

Now we can define the two most important complexity classes.

**P (polynomial time)**: the set of all decision problems that can be **solved** in polynomial time. A decision problem is one that asks yes or no: does this graph have a Hamiltonian cycle? Is this number prime? Is this proposition provable in this logical system?

"In P" means: there is an algorithm that, for any input of size $n$, gives the correct "yes" or "no" within $O(n^k)$ steps, where $k$ is some fixed constant independent of the input.

**NP (nondeterministic polynomial time)**: the set of all decision problems whose answers can be **verified** in polynomial time. More precisely: for every "yes" instance, there exists a certificate of polynomial length that can be verified in polynomial time.

"In NP" means: if the answer is "yes," someone can show you a brief proof, and you can confirm within polynomial time that the proof is correct.

Note the asymmetry here — it's an echo of the asymmetry from Chapter 14: **verifying** a proof is efficient (polynomial time), but **finding** a proof need not be (it may take exponential time).

Every problem in P is in NP (if you can solve a problem in polynomial time, you can certainly verify an answer in polynomial time — just solve it again). The question is: **is every problem in NP also in P?**

That's the question $\mathsf{P} \stackrel{?}{=} \mathsf{NP}$.

::: info Why Almost Everyone Believes P ≠ NP
Strictly speaking, P ≠ NP is an unproven conjecture. But decades of experience among mathematicians and computer scientists have accumulated into a strong belief: between solving and verifying there lies a fundamental chasm. If P = NP, then any solution that can be quickly verified could also be quickly found — which would mean the collapse of cryptography (breaking a code as fast as checking a key), it would mean creativity is equivalent to checking, and it would mean there's no essential difference between finding a mathematical proof and verifying one. That picture is devastating to every intuition we have about reasoning and creation. The belief in P ≠ NP is the mathematical expression of the basic intuition that solving is harder than verifying.
:::

---

## 19.3 NP-Complete: The Hardest Problems

The problems in NP are not all alike. Some problems have the property that, if you could solve them quickly, you could solve every problem in NP quickly. These problems are called **NP-complete**.

This concept requires one tool: **polynomial-time reduction**. If problem $A$ can be reduced in polynomial time to problem $B$ (written $A \leq_p B$), it means: there is a polynomial-time transformation that turns any instance of $A$ into an equivalent instance of $B$. If you can solve $B$ quickly, then through this transformation you can also solve $A$ quickly.

:::details Polynomial-Time Reduction: Understanding "Just as Hard" Through "Translation"
A polynomial-time reduction ($A \leq_p B$) says this: problem $A$ is no harder than problem $B$ — you can use an efficient "translation" to turn any instance of $A$ into an equivalent instance of $B$, then solve $A$ with $B$'s solver.

Analogy: if you have an English–Chinese dictionary (a translation function) and you can read Chinese, then you can read English — English reading is no harder than Chinese reading (given the dictionary).

**What NP-completeness means**: if $C$ is NP-complete, then every problem in NP can be translated into $C$ in polynomial time. Therefore:
- If someone finds a polynomial-time algorithm for $C$, then through the reduction chain, all NP problems can be solved in polynomial time — i.e., P = NP
- If P ≠ NP, then $C$ has no polynomial-time algorithm

That's why "solving one NP-complete problem" is equivalent to "solving all NP problems" — the NP-complete problems are the "representatives" of NP difficulty.
:::

**The definition of NP-complete**: problem $C$ is NP-complete if and only if:
1. $C \in \mathsf{NP}$ (its answer can be quickly verified);
2. For every $A \in \mathsf{NP}$, $A \leq_p C$ (every problem in NP can be reduced to $C$).

The first problem proven NP-complete was **SAT** (the boolean satisfiability problem): given a boolean formula, is there an assignment of variables that makes the formula true? Stephen Cook's 1971 proof — the Cook–Levin theorem — is a founding work of complexity theory.

SAT being NP-complete means: if you can quickly decide whether a boolean formula is satisfiable, you can quickly solve every problem in NP — including the traveling salesperson problem, graph coloring, the knapsack problem … and, yes, some causal discovery problems.

::: info Why SAT Is at the Heart of Reasoning
In the language of formal logic, the boolean satisfiability problem is: given a set of propositional formulas, is there a truth assignment that makes them all true? That's exactly the problem of finding a model — the core operation of logical semantics. Chapter 14 said "verifying a proof can be done in polynomial time." The counterpart here: given a candidate assignment, checking whether it satisfies the formula merely requires substituting each value in — done in polynomial time. Whereas "finding" an assignment that makes the formula true is SAT — NP-complete. The asymmetry of reasoning gets a precise computational characterization here.
:::

### 19.3.5 Hands-on: Build a Reduction Yourself (3-SAT → Vertex Cover)

A reduction is not an abstract concept — it's a translation you can do with your own hands. Below we translate a 3-SAT instance into a vertex cover instance — every step traceable.

**The source problem**: a 3-SAT instance:
$$(x_1 \lor \neg x_2 \lor x_3) \land (\neg x_1 \lor x_2 \lor \neg x_3) \land (x_2 \lor x_3 \lor \neg x_3)$$

Three clauses, three variables.

**The translation rules**:
1. For each variable $x_i$, create a "variable edge" connecting $x_i$ and $\neg x_i$ (two vertices).
2. For each clause, create a triangular "clause gadget" — three vertices, one for each literal in the clause.

**The target value** $k = |\text{variables}| + 2 \times |\text{clauses}| = 3 + 2 \times 3 = 9$.

**The constructed graph (the node list)**:
- Variable gadgets: $(x_1, \neg x_1)$, $(x_2, \neg x_2)$, $(x_3, \neg x_3)$, each pair joined by one edge
- Clause 1 gadget: three nodes in a triangle, labeled $\{x_1, \neg x_2, x_3\}$ (the three literals of clause 1)
- Clause 2 gadget: a triangle of three nodes, labeled $\{\neg x_1, x_2, \neg x_3\}$
- Clause 3 gadget: a triangle of three nodes, labeled $\{x_2, x_3, \neg x_3\}$

**The wiring**: each literal node in a clause gadget is connected to the matching vertex in the variable gadget. For example, clause 1's $x_1$ node gets an edge to $x_1$ in the variable gadget; clause 1's $\neg x_2$ node gets an edge to $\neg x_2$ in the variable gadget.

**Why does this reduction work? (The intuition)**:
- To cover all edges: for each variable gadget's edge, you must pick at least one of $\{x_i, \neg x_i\}$ — which corresponds to assigning the variable true or false.
- Each clause gadget's triangle has 3 internal edges, needing at least 2 vertices to cover — but the variable vertices you pick may already cover one of the clause triangle's edges "for free" (via the wiring). If some literal of the clause is true, its corresponding variable vertex already covers that wiring edge, so the clause triangle only needs 2 more vertices.
- A vertex cover of size $k=9$ exists if and only if the 3-SAT formula is satisfiable.

**The correspondence between a specific assignment and a cover**: take the assignment $x_1 = 1, x_2 = 1, x_3 = 1$. In the variable gadgets, pick $\{x_1, x_2, x_3\}$ (3 vertices). Clause 1: $x_1 = 1$ already covers the wiring edge for $x_1$, so the triangle leaves 2 vertices $\{\neg x_2, x_3\}$ — pick them (2 vertices). The other clauses work the same way. Total vertices = $3 + 3 \times 2 = 9$. ✓

The reduction's construction takes polynomial time: draw a triangle for each clause, add two vertices for each variable edge, wire each clause's literal nodes to the matching variable nodes. All of it can be done in $O(n+m)$ time ($n$ = number of variables, $m$ = number of clauses).

**What did you just do?** You translated a logic problem ("does there exist an assignment making all these clauses true at once?") into a graph problem ("does this graph have a vertex cover with no more than 9 vertices?"). If you had a black-box vertex-cover solver, you could use it to solve 3-SAT — because you can translate any 3-SAT instance into an equivalent vertex cover instance. That's the power of reduction: it doesn't prove that one problem is hard; it proves that if one problem is easy, then the other is easy too.

::: info Professor Pallas's Cat's Commentary
Building a reduction with your own hands and reading a definition of reduction are completely different experiences. Once you've done one, you realize: a reduction isn't an abstract "if A then B" — it's a concrete, programmable translation function. Every reduction is a compiler — it translates the language of one problem into the language of another. The "equivalence" of NP-complete problems isn't a philosophical similarity; it's computation you can actually run.
:::

---

## 19.4 The Halting Problem: A Boundary Deeper Than NP

The discussion of P and NP presupposes that the problems can at least be answered by some algorithm — just fast or slow. But there's another kind of boundary: for some problems, no algorithm whatsoever can answer them, no matter how much time you allow.

**The halting problem**: given a program and its input, will the program eventually halt, or will it run forever?

Turing proved in 1936: no algorithm can correctly answer the halting problem for all programs.

The proof is the classic self-referential construction, essentially the same method as Gödel's.

Suppose a program $H(P,I)$ exists, which accepts a program $P$ and input $I$, and always correctly answers, in finitely many steps: does $P$ halt on input $I$? Now construct another program $D$:

$$D(P) = \begin{cases} \text{loop forever} & \text{if } H(P,P) = \text{halts} \\ \text{halt} & \text{if } H(P,P) = \text{does not halt} \end{cases}$$

$D$ takes the program $P$ as its own input, asks $H$: "Does $P$ halt on input $P$?" — and then does the opposite.

Now ask: does $D$ halt on input $D$?

- If $H(D,D) = \text{halts}$: then $D$ will loop forever — $H$ lied.
- If $H(D,D) = \text{does not halt}$: then $D$ will halt — $H$ lied again.

In every case $H$ gives the wrong answer. Contradiction. So $H$ does not exist.

The structure of this proof runs exactly parallel to the construction of Gödel's sentence $G$ in Chapter 15: $G$ says "I am not provable in this system," and $D$ says "I am the opposite of what $H$ judges." Both use self-reference to force any system that tries to fully characterize itself into contradiction. The halting problem and the incompleteness theorem are two sides of the same coin.

::: info Professor Pallas's Cat's Commentary
Once you see this, Gödel's and Turing's two "shocking results" become one result expressed in two languages. Mathematical self-reference and computational self-reference have exactly the same structure. If you've just memorized the two theorems separately, you haven't really understood them.
:::

::: info The Equivalence from Logic to Computation
More precisely: the Gödel incompleteness of Peano arithmetic can be proven through the halting problem of Turing machines, and vice versa. Computability theory establishes this exact correspondence: a proposition is undecidable in a system if and only if the corresponding computational problem is undecidable. Incompleteness is not a phenomenon peculiar to logic — it's an intrinsic limit shared by every sufficiently strong formal system, including any universal computer.
:::

The halting problem being undecidable means there's a boundary deeper than NP-complete: not "needs exponential time," but "no algorithm works within any finite time." The complexity landscape, from bottom to top, looks roughly like:

$$\mathsf{P} \subseteq \mathsf{NP} \subseteq \mathsf{PSPACE} \subseteq \cdots \subseteq \text{decidable} \subseteq \text{enumerable} \subsetneq \text{all problems}$$

The halting problem lives at the "enumerable but undecidable" level — you can verify in finitely many steps that a halting program really halts (just run it and wait for it to stop), but you cannot confirm in finitely many steps that a non-halting program will never stop (you wait 100 million steps, and it hasn't stopped — that doesn't mean it never will).

### 19.4.5 Space and Alternation: NP Isn't the Whole Story

Besides time, there's another constraint: **space (memory)**.

**PSPACE**: the set of all problems solvable in polynomial space. One can prove $\mathsf{NP} \subseteq \mathsf{PSPACE}$ (you can enumerate all possible certificates in polynomial space — reclaim the space once you're done rather than storing all of them at once). But whether NP equals PSPACE is unknown.

PSPACE contains problems harder than anything in NP. For example: **quantified boolean formulas (QBF)** are PSPACE-complete. QBF extends SAT: instead of just asking "does an assignment exist that makes the formula true," it asks about layers of interleaved quantifiers, such as "$\forall x_1 \exists x_2 \forall x_3 \ldots$ makes the formula true." This "alternating quantifier" structure, at the intersection of complexity and logic, corresponds to the **polynomial hierarchy** — layering the existential quantifiers of NP over the universal quantifiers of coNP, with each layer (conjecturally) harder than the one below.

**The polynomial hierarchy** (rough version):
- $\Sigma_1^p = \mathsf{NP}$: $\exists$ one certificate that can verify the problem
- $\Pi_1^p = \mathsf{coNP}$: $\forall$ certificates, some condition holds
- $\Sigma_2^p$: $\exists$ one certificate such that $\forall$ another certificate, some condition holds
- $\Pi_2^p$: $\forall$ certificates, $\exists$ another certificate …
- … and so on

Every level (conjecturally) strictly contains the one below. If any level collapses (equals the one below), the whole hierarchy collapses down to that level. For example, if NP = coNP, then $\Sigma_1^p = \Pi_1^p$, and from there one can prove the entire polynomial hierarchy collapses to the first level.

**What does this have to do with Chapter 15?** The alternating-quantifier structure of the polynomial hierarchy corresponds directly to proof length in logic: how many alternations of "for all" and "there exists" a proposition contains determines which level of the hierarchy its proof complexity sits at. Gödel incompleteness and complexity theory are linked precisely through this correspondence — a system's "sufficient strength," in the language of complexity, is equivalent to "being able to express enough layers of quantifier alternation."

---

## 19.5 Oracle Separation: The Boundary Is Real

A natural question arises: is the separation between P and NP merely a limitation of our current techniques, rather than a principled boundary?

Relativization gives us part of the answer.

Give an algorithm an **Oracle** — a black box that can answer one specific kind of question in a single step. Depending on which Oracle you choose, the relationship between P and NP can change. In 1975, Baker, Gill, and Solovay proved:

- There exists an Oracle $A$ such that, relative to $A$, $\mathsf{P}^A = \mathsf{NP}^A$;
- There exists an Oracle $B$ such that, relative to $B$, $\mathsf{P}^B \neq \mathsf{NP}^B$.

The meaning of this result is: **a proof of P ≠ NP cannot be carried out using relativization**. Any proof route based on counting or algebra that "relativizes" (i.e., still holds in Oracle models) cannot distinguish the world where P = NP from the world where P ≠ NP.

This doesn't say P ≠ NP is unprovable — it says the proof must be non-relativizing: it must reach into the essential structure of computation rather than stay at the level of algorithmic manipulation. So far, nobody has satisfied that requirement.

Every known technical barrier — relativization, natural proofs, algebraization — says the same thing: the road to P ≠ NP does not pass through any route we're familiar with. The depth of this problem is perhaps proportionate to its importance.

---

## 19.6 The Price of Reasoning: Back to Causality

The causal discovery problems of Chapter 18 can now be placed precisely within the complexity framework.

**Causal structure learning**: given observational data on a set of variables, find the causal graph consistent with the data. In general — with $n$ variables, the number of possible DAGs grows super-exponentially in $n$, so exhaustive search isn't even remotely feasible.

Known results:

- Under certain restrictions (assuming the causal Markov condition and the faithfulness condition), the PC algorithm can recover the skeleton of the causal graph (the undirected edges) in polynomial time;
- But determining the directions of edges — distinguishing different graphs within a Markov equivalence class — is NP-hard in general;
- With hidden variables, even the skeleton is hard to recover.

These results say: **exact causal inference is fundamentally hard computationally**. Not because the algorithms are written badly, but because the problem itself falls in the NP-hard regime. Even with perfect data, there is no systematic way to find the answer quickly.

This is a punch-back at Chapter 18: give me the graph and I can do causal inference — but obtaining the graph itself may cost more than one can pay.

The dilemma of causal inference is one instance of a general dilemma of reasoning: **exact inference is often computationally infeasible**. This is not an algorithmic problem — it's determined by the geometry of the problem space, and there are no shortcuts.

Back to the seeker and ticket-checker game: hand it a candidate causal graph, and often enough you can check whether it satisfies certain conditional-independence relations; but finding that graph among all possible DAGs is a different problem. The ticket checker can stamp quickly — that doesn't mean the seeker can get out of the maze quickly. Chapter 18 gave us the rules for cutting wires; Chapter 19 reminds us that finding which graph to cut may itself be an exponential maze.

---

## 19.7 The Equivalence of Proof and Computation

Chapter 14 said: proofs are syntactic objects; verification can be done in polynomial time; finding them need not be. This chapter turns that intuition into rigorous statements.

**The satisfiability of propositional logic (SAT) is NP-complete**: finding an assignment that makes a given formula true, and finding a proof, are computationally equivalent in complexity.

**A deeper correspondence: proof length and computation steps.** The length of the shortest proof of a proposition in some system corresponds to the minimum number of steps needed to solve the corresponding problem under some computational model. Proof complexity is precisely the field that studies this correspondence — it connects the expressive power of logical systems to computational complexity classes.

One known conclusion: if there existed a propositional proof system that could give polynomial-length unsatisfiability proofs for every unsatisfiable instance of SAT, then NP = coNP. In other words: **any sufficiently weak proof system must contain true propositions whose proofs require exponential length**. This is not Gödel incompleteness — it doesn't say the proposition is unprovable; it says even if it is provable, the proof may be so long as to be unusable.

This is the formal answer to the lingering question from Chapter 14 ("does the length of a proof matter?"): **yes, it matters enormously**. Proof length is the mirror image of complexity, and the expressive efficiency of a formal system directly translates to the difficulty of computational problems.

---

## Open Questions

**Is P ≠ NP true?** The overwhelming majority believe so, but nobody has proved it. It's one of the most important unsolved problems in mathematics. If it's false — if P = NP — then every problem that can be quickly verified could be quickly solved: cryptography collapses, creativity becomes equivalent to inspection, and the intrinsic cost of reasoning disappears. That world contradicts all our intuitions, yet is not logically self-contradictory. Our belief in P ≠ NP is stronger than our proof of it.

**Are the boundaries of complexity classes sharp?** Does a middle ground exist between P and NP — problems that are neither in P nor NP-complete? Ladner's theorem tells us: if P ≠ NP, then such problems exist, called NP-intermediate problems. The graph isomorphism problem is suspected to be one, but as yet that cannot be proved. The complexity landscape is far richer than the two classes P and NP.

**If exact inference is infeasible, what about approximate inference?** P ≠ NP says exact solutions are hard to find. But maybe finding a "good enough" approximate solution costs far less? For some problems, yes — there exist polynomial-time approximation algorithms that guarantee the solution quality within some ratio of the optimum. But for other problems, approximation is likewise NP-hard — even allowing error, you can't escape the exponential cost. This distinction is the topic of Chapter 20: heuristics are not arbitrary approximations; they are commitments backed by exact mathematical contracts.

---

## Exercises

**★ Warm-up**

Classify each of the following: is it in P, in NP (but not known to be in P), or undecidable? Just give the classification and a one-sentence reason.

1. Given a sorted array of integers and a target value, decide whether the target value is in the array (binary search)
2. Given a boolean formula, decide whether it is satisfiable (SAT)
3. Given a program, decide whether it never outputs anything
4. Given a graph, decide whether it has a Hamiltonian cycle
5. Given two programs, decide whether they produce the same output on every input

---

**★★ Derivation**

1. **The direction of reduction**: if $A \leq_p B$ ($A$ can be reduced to $B$ in polynomial time) and $B \in \mathsf{P}$, then $A \in \mathsf{P}$. Conversely, if $A \leq_p B$ and $A$ is NP-complete, what does that imply about $B$? Consider the two cases: $B \in \mathsf{NP}$ and $B \notin \mathsf{NP}$.

2. **Variants of the halting problem**: of the three halting-problem variants below, which is decidable and which is undecidable? State your reasoning clearly.
   - Given program $P$ and input $I$: does $P$ halt on $I$ within $10^{100}$ steps?
   - Given program $P$: does $P$ halt on every input?
   - Given program $P$: does $P$ halt on some input?

---

**★★★ Challenge**

Chapter 14 said: verifying a propositional-logic proof is polynomial-time doable. That means "propositional-logic theorem verification" — given a proposition $\varphi$ and a candidate proof sequence, decide whether the sequence is a legal proof of $\varphi$ — is in P.

Now consider a different problem: "Given a proposition $\varphi$, decide whether $\varphi$ is a theorem (i.e., whether a proof exists)."

1. Is this problem in NP? (Hint: what serves as the "certificate"?)
2. SAT is NP-complete, and SAT is equivalent to "given a propositional formula, decide whether it is satisfiable." In propositional logic, what is the relationship between "satisfiable" and "is a theorem"?
3. Link the two things: if P = NP, what does that mean for "finding proofs" in mathematics? And if P ≠ NP, does that mean some mathematical theorems have proofs that, even though they exist, can in principle never be found?

The third question touches a genuine problem in the philosophy of mathematics, and there's no consensus answer today. Try to distinguish the two cases — "a proof exists but cannot be found" and "no proof exists" — and state that distinction clearly in the language of this chapter.

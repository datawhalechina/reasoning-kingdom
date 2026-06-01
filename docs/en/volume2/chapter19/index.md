# Chapter 19: Complexity as the Geometry of Reasoning — Why Some Reasoning Cannot Be Accelerated at All

> P≠NP is not a proposition about machine speed. It is a proposition about the intrinsic structure of problems.

---

Chapter 18 left us with an uncomfortable observation: even with a complete causal graph, performing inference on the graph — identifying causal effects, computing identifiable expressions — sees the cost rise sharply in complex situations. This is not because the algorithm is poorly written; the problem itself is hard.

This "intrinsically hard" requires a precise mathematical meaning.

In the context of formal systems, "hard" has a quantified version: how many steps are needed from axioms to a theorem? Chapter 14 mentioned that verifying a proof can be done in polynomial time, but discovering a proof has no universally efficient algorithm. This asymmetry has a formal name in computation theory: the **separation of P and NP** — if it indeed holds.

What this chapter sets out to do is turn this intuition of "hard" into a precise geometric picture.

---

## 19.0 The Seeker and the Ticket-Checker Game: Why Verification Is Easier Than Discovery

Complexity theory can begin with a very simple game.

A golden ticket is hidden inside a maze. The seeker's task is to find it; the ticket-checker's task is to verify whether a ticket handed in by others is genuine. The seeker may have to grope through an exponential number of branching paths. The ticket-checker, however, has it easy: as soon as the ticket is handed over, he just checks the seal, the serial number, and whether the path record complies with the rules.

This is the intuitive difference between P and NP.

If a problem is in P, the seeker himself can find the answer in polynomial time. If a problem is in NP, then at least when the answer is "yes," someone can hand the ticket-checker a polynomial-length certificate, allowing the ticket-checker to confirm it quickly.

NP-complete problems are something even more terrifying: they are like a universal maze, into which all NP mazes can be efficiently translated. If you can quickly find the golden ticket in this maze, then you can quickly find the golden ticket in all NP mazes.

:::details The Formal Skeleton of This Game
- **State space**: input instance $x$, candidate certificate $w$, verifier state.
- **Legal moves**: the seeker generates candidate $w$; the ticket-checker checks $V(x,w)$ in polynomial time.
- **Transition rules**: the search process moves through candidate space; the verification process scans the certificate according to a deterministic algorithm.
- **Victory condition**: the seeker finds a certificate making $V(x,w)=1$, or proves no such certificate exists.
- **Failure mode**: certificates are easy to check, but the candidate space is too large to exhaustively enumerate; fast verification does not imply fast discovery.
:::

This game spells out the coldness of complexity theory: what is hard is not "confirming the answer once you know it," but "finding the answer when you don't know it." Proofs, planning, SAT, combinatorial search — all are stuck on this asymmetry.

The halting problem pushes the game even deeper: some mazes are not just too large, but you fundamentally cannot design any ticket-checker that can always determine "whether the seeker will ever come out." That is not NP-type difficulty, but the boundary of computability.

::: info Professor Pallas's Cat's Commentary
The most stinging part of the ticket-checker game is that it humiliates "hindsight wisdom." Once the answer is laid out, everyone finds it simple; the true difficulty hides before the answer appears. What P and NP ask is precisely: between creation and verification, is there a principled abyss?
:::

## 19.1 The Intrinsic Structure of Problems

Let us begin with a distinction that catches people by surprise.

There are two kinds of "difficulty," superficially similar, yet fundamentally different.

**The first kind of difficulty**: the problem is large, the data are abundant, the computer needs to run for a long time. This is difficulty in the engineering sense, which can be alleviated by faster hardware, smarter algorithms, or more parallel computation. This kind of difficulty has no principled boundary.

**The second kind of difficulty**: no matter what algorithm you use, no matter how fast the hardware, certain problems require exponential time. This is difficulty in the mathematical sense — not an engineering issue, but determined by the intrinsic structure of the problem itself.

Complexity theory concerns the second kind of difficulty. Its question is not "is this algorithm fast," but "can this problem be solved quickly by any algorithm at all."

::: info Professor Pallas's Cat's Commentary
Many people, upon hearing P≠NP, associate it with "computers are not fast enough." This is a completely mistaken understanding. What P≠NP says is: even if you were given an infinitely fast computer, certain problems would still require exponential time. Speed cannot buy structure. The geometric shape of a problem does not change because the machine becomes faster.
:::

To give "fast" a precise meaning, a measuring stick is needed.

**Time complexity**: for a problem with input size $n$, how many steps does the optimal algorithm require to produce the answer? If there exists an algorithm such that the number of steps is polynomial in $n$ ($n^2, n^3, n^{100}$…), the problem is called **polynomial-time solvable**; if all known algorithms require an exponential number of steps ($2^n, n!, \ldots$), the problem rapidly becomes unsolvable in practice as $n$ grows.

The gulf between polynomial and exponential is the core dividing line of complexity theory. When $n = 100$, $n^{10} = 10^{20}$, already enormous; $2^{100} \approx 10^{30}$, more than the number of atoms in the universe. As $n$ grows further, this gap only becomes more absurd.

---

## 19.2 P and NP: Two Capabilities

Now we can define the two most important complexity classes.

**P (Polynomial time)**: the set of all decision problems that can be **solved** in polynomial time. A decision problem is a "yes or no" question: does this graph contain a Hamiltonian cycle? Is this number prime? Is this proposition provable in this logical system?

"Being in P" means: there exists an algorithm that, for any input of size $n$, produces the correct "yes" or "no" in $O(n^k)$ steps ($k$ is some fixed constant, independent of the input).

**NP (Nondeterministic Polynomial time)**: the set of all decision problems whose answers can be **verified** in polynomial time. More precisely: for "yes" instances, there exists a polynomial-length **certificate** that can be verified in polynomial time.

"Being in NP" means: if the answer is "yes," then someone can show you a short proof, allowing you to confirm in polynomial time that this proof is correct.

Note the asymmetry here — precisely the echo of the asymmetry from Chapter 14: **verifying** a proof is efficient (polynomial time); **discovering** a proof is not necessarily efficient (may require exponential time).

All problems in P are also in NP (if you can solve a problem in polynomial time, you can certainly verify the answer in polynomial time — just compute it again). The question is: **are all problems in NP also in P?**

This is the $\mathsf{P} \stackrel{?}{=} \mathsf{NP}$ problem.

::: info Why Almost Everyone Believes P ≠ NP
Strictly speaking, P ≠ NP is an unproven conjecture. But over the decades, the accumulated experience of mathematicians and computer scientists has built a strong conviction: between solving and verifying, there exists a fundamental gulf. If P = NP, then any solution that can be quickly verified can also be quickly found — this would mean the collapse of cryptography (breaking encryption would be as fast as verifying a key), would mean creativity is equivalent to verification, would mean that the discovery and verification of mathematical proofs have no essential difference. This picture is destructive to all our intuitions about reasoning and creativity. The belief in P ≠ NP is the mathematical expression of the basic intuition that "solving is harder than verifying."
:::

---

## 19.3 NP-complete: The Hardest Problems

Problems in NP are not all equal. Some problems, if they could be solved quickly, would enable the quick solution of **all** problems in NP. These problems are called **NP-complete**.

This concept requires a tool: **polynomial-time reduction**. If problem $A$ can be polynomial-time reduced to problem $B$ (denoted $A \leq_p B$), it means: there exists a polynomial-time transformation that converts any instance of $A$ into an equivalent instance of $B$. If you can solve $B$ quickly, then through this transformation, you can also solve $A$ quickly.

:::details 
Polynomial-time Reduction: Using "Translation" to Understand "Equally Hard"
Polynomial-time reduction ($A \leq_p B$) says: problem $A$ is no harder than problem $B$ — you can use an efficient "translator" to turn any instance of $A$ into an equivalent instance of $B$, and then use a solver for $B$ to solve $A$.

Analogy: if you have an English-Chinese dictionary (the translation function), and you can read Chinese, then you can read English — reading English is no harder than reading Chinese (given the dictionary).

**The significance of NP-completeness**: if $C$ is NP-complete, it means every problem in NP can be polynomial-time "translated" into $C$. Therefore:
- If someone finds a polynomial-time algorithm for solving $C$, then through the reduction chain, all NP problems can be solved in polynomial time, i.e., P = NP
- If P ≠ NP, then $C$ has no polynomial-time algorithm

This is why "solving one NP-complete problem" is equivalent to "solving all NP problems" — NP-complete problems are the "representatives" of NP-hardness.
:::


The definition of NP-complete: problem $C$ is NP-complete if and only if:
1. $C \in \mathsf{NP}$ (answers can be quickly verified);
2. For all $A \in \mathsf{NP}$, $A \leq_p C$ (all problems in NP can be reduced to $C$).

The first problem proven to be NP-complete was **SAT** (Boolean satisfiability problem): given a Boolean formula, does there exist an assignment of variables making the formula true? Stephen Cook's 1971 proof — the Cook-Levin theorem — is the foundational work of complexity theory.

That SAT is NP-complete means: if you can quickly determine whether a Boolean formula is satisfiable, you can quickly solve all problems in NP, including the traveling salesman problem, graph coloring, knapsack... and, certain causal discovery problems.

::: info Why SAT Is the Core of Reasoning
The Boolean satisfiability problem, in the context of formal logic, is: given a set of propositional formulas, does there exist a truth assignment making them all true? This is precisely the problem of finding a model — the core operation of logical semantics. Chapter 14 said "verifying a proof can be done in polynomial time," corresponding to: given a candidate assignment, checking whether it satisfies the formula merely requires term-by-term substitution, doable in polynomial time. And "discovering" an assignment that makes the formula true is SAT — NP-complete. The asymmetry of reasoning here acquires a precise computational characterization.
:::

---

## 19.4 The Halting Problem: A Boundary Deeper Than NP

The discussion of P and NP presupposes that problems are at least answerable by algorithms — only differing in speed. But there exists another kind of boundary: some problems admit no algorithm to answer them at all, no matter how much time is given.

**The halting problem**: given a program and its input, will this program eventually halt, or run forever?

Turing proved in 1936: there exists no algorithm that can correctly answer the halting problem for all programs.

The proof is a classic self-referential construction, identical in essence to Gödel's method.

Assume there exists a program $H(P, I)$ that accepts program $P$ and input $I$, and always, in finitely many steps, correctly answers: whether $P$ on input $I$ will halt. Now construct another program $D$:

$$D(P) = \begin{cases} \text{loop forever} & \text{if } H(P, P) = \text{halts} \\ \text{halt} & \text{if } H(P, P) = \text{does not halt} \end{cases}$$

$D$ treats the program $P$ as its own input, asks $H$: "Does $P$ on input $P$ halt?" and then does the opposite.

Now ask: does $D$ on input $D$ halt?

- If $H(D, D) = $ halts: $D$ will loop forever — $H$ has lied.
- If $H(D, D) = $ does not halt: $D$ will halt — $H$ has lied again.

$H$ gives the wrong answer in every case. Contradiction. Hence $H$ does not exist.

The structure of this proof is entirely parallel to the construction of the Gödel sentence $G$ in Chapter 15: $G$ says "I am not provable in this system," $D$ says "I do the opposite of what $H$ judges." Both, through self-reference, drive any system that attempts to completely characterize itself into contradiction. The halting problem and the incompleteness theorems are two sides of the same coin.

::: info Professor Pallas's Cat's Commentary
Once you see this clearly, Gödel's and Turing's two "shocking results" become one result, expressed in two languages. Mathematical self-reference and computational self-reference have exactly the same structure. If you have only memorized the two theorems separately, you have not yet understood them.
:::

::: info From Logic to Computation: The Equivalence
More precisely: Gödelian incompleteness in Peano arithmetic can be proved via the halting problem for Turing machines, and vice versa. Recursive function theory (Computability Theory) establishes this precise correspondence: a proposition is undecidable in a system if and only if the corresponding computational problem is undecidable. Incompleteness is not a phenomenon peculiar to logic, but an intrinsic limitation shared by any sufficiently strong formal system — including universal computers.
:::

That the halting problem is undecidable means there exists a boundary deeper than NP-completeness: not "requires exponential time," but "no algorithm in any finite time." The landscape of complexity, from bottom to top, is roughly:

$$\mathsf{P} \subseteq \mathsf{NP} \subseteq \mathsf{PSPACE} \subseteq \cdots \subseteq \text{decidable} \subseteq \text{enumerable} \subsetneq \text{all problems}$$

The halting problem lives at the "enumerable but not decidable" level — you can verify in finitely many steps that a halting program indeed halts (just run it and wait for it to stop), but you cannot confirm in finitely many steps that a non-halting program will never halt (you've waited a hundred million steps and it still hasn't halted; this does not mean it will never halt).

---

## 19.5 Oracle Separation: The Boundary Is Real

There is a natural question: is the separation between P and NP merely a limitation of our current techniques, rather than a principled boundary?

Relativization gives us a partial answer.

Equip an algorithm with an **Oracle** — a black box that can answer a certain class of questions in one step. Depending on the choice of Oracle, the relationship between P and NP can change. Baker, Gill, and Solovay proved in 1975:

- There exists an Oracle $A$ such that, relative to $A$, $\mathsf{P}^A = \mathsf{NP}^A$;
- There exists an Oracle $B$ such that, relative to $B$, $\mathsf{P}^B \neq \mathsf{NP}^B$.

The implication of this result is: **a proof of P ≠ NP cannot be accomplished using relativization techniques**. Any proof approach based on counting or algebra that "relativizes" (i.e., remains valid for Oracle models) cannot distinguish the P = NP world from the P ≠ NP world.

This does not say P ≠ NP cannot be proved, but that the proof must be non-relativizing — must penetrate into the essential structure of computation, rather than staying at the level of algorithmic operations. This requirement, to this day, no one has satisfied.

The known technical barriers — relativization, natural proofs, algebrization — each indicate: the road to P ≠ NP does not pass through any route familiar to us. The depth of this problem is perhaps commensurate with its importance.

---

## 19.6 The Cost of Reasoning: Returning to Causation

The causal discovery problem of Chapter 18 can now be precisely located within the complexity framework.

**Causal structure learning**: given observational data on a set of variables, find a causal graph consistent with the data. In the general case — with $n$ variables, the number of possible DAGs grows super-exponentially with $n$; exhaustive search is utterly infeasible.

Known results:

- Under certain restrictions (assuming the causal Markov condition and faithfulness condition), the PC algorithm can recover the skeleton (undirected edges) of the causal graph in polynomial time;
- But determining edge directions — distinguishing different graphs within a Markov equivalence class — is NP-hard in the general case;
- In the presence of latent variables, even the skeleton is difficult to recover.

This result indicates: **precise causal inference is computationally fundamentally hard**. Not because the algorithm is poorly written, but because the problem itself falls into the NP-hard category. Even with perfect data, one cannot systematically find the answer quickly.

This is a riposte to Chapter 18: give me the graph, and I can do causal inference — but obtaining the graph itself may carry an unbearable cost.

The dilemma of causal reasoning is an instance of the universal dilemma of reasoning: **exact reasoning is often computationally infeasible**. This is not a problem of algorithms; it is determined by the geometric shape of the problem space; there are no shortcuts.

Returning to the Seeker and Ticket-Checker game: given a candidate causal graph, in many cases you can check whether it satisfies certain conditional independence relations; but finding that graph among all possible DAGs is another problem. The ticket-checker can stamp quickly, but that does not mean the seeker can quickly walk out of the maze. Chapter 18 gave us the rules for cutting wires; Chapter 19 reminds us: first finding which graph to cut from may itself be an exponential maze.

---

## 19.7 The Equivalence of Proof and Computation

Chapter 14 stated: proofs are syntactic objects, verification is polynomial-time doable, discovery is not necessarily so. Chapter 19 turns this intuition into a rigorous statement.

**Propositional logic satisfiability (SAT) is NP-complete**: discovering an assignment that makes a given formula true, and discovering a proof, are equivalent in computational complexity.

A deeper correspondence: **proof length and computation steps**. The length of the shortest proof of a proposition in a given system corresponds to the minimum number of steps required to solve the corresponding problem in a given computational model. Proof complexity is precisely the field studying this correspondence — it connects the expressive power of logical systems to computational complexity classes.

A known result: if there exists some propositional proof system that can produce polynomial-length unsatisfiability proofs for any unsatisfiable instance of SAT, then NP = coNP. In other words, **any sufficiently weak proof system inevitably has true propositions requiring exponentially long proofs**. This is not Gödelian incompleteness — it is not saying the proposition is unprovable, but that even if provable, the proof may be so long as to be unusable.

This is the formal answer to the suspense left by Chapter 14 ("does proof length matter?"): **yes, it matters greatly**. Proof length is a mirror of complexity; the expressive efficiency of a formal system is directly related to the difficulty of computational problems.

---

## Unresolved

**Is P ≠ NP true?** The vast majority believe it is, but no one has proved it. This is one of the most important unsolved problems in mathematics. If it is false — if P = NP — then all problems that can be quickly verified can be quickly solved, cryptography collapses, creativity is equivalent to verification, and the intrinsic cost of reasoning vanishes. This world contradicts all our intuitions, yet is not logically self-contradictory. Our belief in P ≠ NP is more firm than any proof of it.

**Are the boundaries between complexity classes sharp?** Does an intermediate zone exist between P and NP — problems that are neither in P nor NP-complete? Ladner's theorem tells us: if P ≠ NP, then such problems exist, called NP-intermediate problems. The graph isomorphism problem is suspected to be such a problem, but this cannot currently be proved. The landscape of complexity is far richer than the two classes P and NP.

**If exact reasoning is infeasible, what about approximate reasoning?** P ≠ NP says exact solutions are hard to find. But perhaps finding a "good enough" approximate solution is far cheaper? For some problems, this is true — polynomial-time approximation algorithms exist that can guarantee the solution quality is within some ratio of the optimal. But for other problems, approximation is equally NP-hard — even allowing error cannot escape the exponential cost. This distinction is the theme of Chapter 20: heuristics are not arbitrary approximations; they are promises with precise mathematical contracts.

---

## Exercises

**★ Warm-up**

Classify each of the following problems: is it in P, in NP (but not known whether in P), or undecidable? Only classification and a one-sentence justification are required.

1. Given a sorted integer array and a target value, determine whether the target value is in the array (binary search)
2. Given a Boolean formula, determine whether it is satisfiable (SAT)
3. Given a program, determine whether it never outputs anything
4. Given a graph, determine whether it contains a Hamiltonian cycle
5. Given two programs, determine whether they produce the same output for all inputs

---

**★★ Derivation**

1. **Direction of reduction**: If $A \leq_p B$ ($A$ can be polynomial-time reduced to $B$), and $B \in \mathsf{P}$, then $A \in \mathsf{P}$. Conversely, if $A \leq_p B$ and $A$ is NP-complete, what does this imply for $B$? Consider two cases: $B \in \mathsf{NP}$ and $B \notin \mathsf{NP}$.

2. **Variants of the halting problem**: Which of the following three variants of the halting problem are decidable, and which are undecidable? Explain clearly.
   - Given program $P$ and input $I$, does $P$ on $I$ halt within $10^{100}$ steps?
   - Given program $P$, does $P$ halt on all inputs?
   - Given program $P$, does $P$ halt on some input?

---

**★★★ Challenge**

Chapter 14 said: verifying a propositional logic proof can be done in polynomial time. This means "propositional logic theorem proving" — given a proposition $\varphi$ and a candidate proof sequence, determine whether this sequence is a valid proof of $\varphi$ — is in P.

Now consider a different problem: "given proposition $\varphi$, determine whether $\varphi$ is a theorem (i.e., whether a proof exists)."

1. Is this problem in NP? (Hint: what serves as the "certificate"?)
2. SAT is NP-complete, and SAT is equivalent to "given a propositional formula, determine whether it is satisfiable." What is the relationship between "satisfiable" and "is a theorem" in propositional logic?
3. Connect these two matters: if P = NP, what would this mean for the activity of "discovering proofs" in mathematics? If P ≠ NP, does this mean the proofs of certain mathematical theorems, even if they exist, are in principle unfindable?

The third question touches a real problem in the philosophy of mathematics, currently without consensual answer. Try to distinguish the two situations "the proof exists but cannot be found" and "the proof does not exist" — write this distinction clearly using the language of this chapter.

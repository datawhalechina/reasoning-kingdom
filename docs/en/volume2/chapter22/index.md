# Chapter 22: Self-Reference and Emergence — When a Reasoning System Begins to Reason About Itself

> The Curry-Howard correspondence tells us: proofs are programs. So, what are programs proving?

---

The final paragraph of Chapter 21 brought us here: a sufficiently strong inference system begins to contain propositions about itself. At this moment, the Gödelian structure of Chapter 15 reappears.

But Chapter 22 is not a repetition of Chapter 15. Gödel spoke of **boundaries** — those true propositions you cannot reach. Chapter 22 speaks of **openings** — near this boundary, new things are emerging.

Having walked this far through the second volume, nine chapters of logical deduction, the vantage point is high enough for a panoramic view. This is the final chapter, and also the most honest one: some questions we can only state, not resolve.

---

## 22.1 Curry-Howard: Proofs as Programs

Chapter 14 established formal systems: propositions, inference rules, proofs. Chapter 19 discovered that computation and logic are equivalent at the level of complexity: SAT is NP-complete, and proof discovery in propositional logic is, in the worst case, as hard as SAT.

But how deep does the equivalence go?

The **Curry-Howard correspondence** (also called the propositions-as-types correspondence) gives the answer: **as deep as it possibly can be**.

Precisely stated, it establishes the following correspondence:

| Logic | Type Theory / Programs |
|------|-------------|
| Proposition $A$ | Type $A$ |
| Proof of proposition $A$ | Term (program) of type $A$ |
| $A \to B$ (implication) | Function type $A \to B$ |
| $A \land B$ (conjunction) | Product type $A \times B$ |
| $A \lor B$ (disjunction) | Sum type $A + B$ |
| $\bot$ (false proposition) | Empty type (uninhabited type) |
| Modus ponens | Function application |
| Deduction theorem | Function abstraction ($\lambda$ abstraction) |

This is not a metaphor but an isomorphism: every step of logical deduction corresponds to every step of program computation; the normalization of a proof (simplifying to its simplest form) corresponds to the evaluation of a program (running to termination).

::: info Professor Pallas's Cat's Commentary
Many people nod here and then continue treating "logic" and "program" as two separate things. Isomorphism means: every step you prove in logic is a computational step in a programming language; every function you write "proves" some proposition. If you don't find this staggering, you haven't yet truly understood what this correspondence is saying.
:::

A proposition "has a proof" is equivalent to its corresponding type "having a program" — the corresponding type is "inhabited." **Proofs are programs, propositions are types, verification is execution.**

::: info Why This Correspondence Is Shocking
Before the Curry-Howard correspondence, mathematics and computer science were two adjacent but separate fields. The correspondence tells us: they are two languages for the same thing. Every mathematical theorem can be understood as a program; every program "proves" some proposition — proving that its output type can be constructed from its input type. This is not a philosophical similarity, but an equivalence that can be precisely verified by formal means. Theorem-proving assistants like Lean, Coq, and Agda are built precisely on this equivalence: writing a program is writing a proof, type-checking is proof verification.
:::

---

## 22.2 Fixed Points: The Algebra of Self-Reference

The Gödel sentence $G$ of Chapter 15 is essentially a **self-referential construction**: $G$ talks about $G$'s own provability. The mathematical heart of this construction is the diagonalization lemma — a mechanism for constructing "a proposition that talks about itself."

But self-reference does not only appear in Gödel's theorem. It has a more general algebraic structure: **fixed points**.

**Definition (Fixed Point)**: Given a function $f$, if there exists $x$ such that $f(x) = x$, then $x$ is called a fixed point of $f$.

In the $\lambda$-calculus (the minimal core language of programs), there is a famous fixed-point combinator $Y$:

$$Y = \lambda f.\, (\lambda x.\, f\,(x\,x))\,(\lambda x.\, f\,(x\,x))$$

For any function $f$, $Y\,f$ is the fixed point of $f$: $Y\,f = f\,(Y\,f)$.

Expanding this equality, $Y\,f$ "runs" itself — it passes itself as an argument to $f$. The $Y$ combinator is the algebraic root of recursion: any recursive definition is, in essence, finding the fixed point of some functional equation.

:::details 
**The $Y$ Combinator and the $\lambda$-Calculus: The "Minimal Core" of Programs**

The $\lambda$-calculus (Lambda Calculus) is a model of computation proposed by Alonzo Church in the same period (the 1930s) as Turing, simpler than the Turing machine: it has only three elements — variables, function abstraction ($\lambda x. \text{body}$), function application ($f\, a$).

**$\lambda x. \text{body}$** means "a function that takes input $x$ and returns $\text{body}$." For example, $\lambda x. x+1$ is the "add-one function."

**The $Y$ combinator** is a tool in the $\lambda$-calculus for implementing "recursion." In ordinary programming languages, one can directly write `def f(x): return f(x-1)`, but in pure $\lambda$-calculus, functions have no names and cannot directly self-reference. The $Y$ combinator bypasses this restriction: $Y\,g$ passes $g$ itself as an argument to $g$, thereby achieving recursion.

**What does this have to do with self-reference?** The $Y$ combinator is "a function that finds its own fixed point" — it encodes self-reference as pure functional computation. Gödel's diagonalization lemma is, in essence, the same mathematical structure manifested in logic.
:::


Gödel's diagonalization lemma, in this language, is the fixed point of the "provability predicate": there exists a proposition $G$ such that $G \equiv \neg\mathsf{Prov}(\ulcorner G \urcorner)$ — $G$ is the fixed point of the predicate "negation of own provability." The program $D$ in the halting problem is likewise a fixed-point construction: the behavior of $D$ on input $D$ depends on the behavior of $D$ on input $D$.

**Self-reference, fixed points, diagonalization — these are the same mathematical structure under different names in different contexts.**

---

## 22.2.5 The MP Game: Proof Sequences as Dynamical Systems

Modus Ponens (MP) is the most fundamental inference rule of propositional logic: given $p$, given $p \to q$, one obtains $q$.

Textbooks give a definition of a proof: a finite sequence of formulas $\{p_1, p_2, \ldots, p_n\}$ where each $p_i$ is either an axiom or obtained from earlier formulas via MP. This is a static definition — it describes an already completed object.

But what if we view it dynamically?

**Game Setup**: Define a function $f: \mathsf{Formula} \to \mathsf{Formula}$, and let the proof sequence satisfy $p_{n+1} = f(p_n)$. In this way, a proof becomes an **orbit**:

$$p_1 \xrightarrow{f} f(p_1) \xrightarrow{f} f^2(p_1) \xrightarrow{f} \cdots$$

Each application of $f$ corresponds to one MP inference: from the current formula, deduce the next formula. A proof sequence is a segment of trajectory of this dynamical system.

**Three Fates of an Orbit**:

1. **Apparent Halting**: The sequence reaches some $p_k$ such that $f(p_k) = p_k$ — formally a fixed point. But this only means the successor equals itself, not that $p_k$ is the true QED. $p_k$ could be a self-referential construction; "halting" is merely the special case of a cycle of period 1.
2. **Finite Cycle**: There exist $k > j$ such that $p_k = p_j$ — the orbit goes in circles; no new content emerges.
3. **Infinite Extension**: The sequence never terminates — corresponding to undecidability. No algorithm can determine, for arbitrary given $f$ and initial formula, whether the orbit will halt.

**The Cauchy Condition and the Subtlety of Halting**: Judging whether the first fate represents "true completion" is equivalent to asking: does the sequence become a Cauchy sequence?

On the state space $\{0,1\}$ (or any discrete space), the Cauchy condition

$$\forall \varepsilon > 0,\ \exists N,\ \forall m,n > N:\ d(p_m, p_n) < \varepsilon$$

degenerates into "the sequence is eventually constant" — i.e., eventually fixed. $f(p_k) = p_k$ satisfies this condition. But the Cauchy condition only guarantees that the limit **exists**, not that the limit point is the "true answer."

Real-world computers have finite stack space, so we manually impose a cutoff condition — not halting automatically upon reaching the limit, but halting imposed from outside. This external condition, in a sufficiently strong system, cannot be internalized by the system itself.

**A fixed point is the endpoint of a proof** — but only when the limit point is meaningful. When $f(\varphi) = \varphi$, $\varphi$ has "stabilized" under inference, but where it has stabilized requires external verification.

This is not merely a metaphor. The diagonalization lemma says: for any formula $A(x)$, there exists a sentence $\varphi$ such that $\varphi \leftrightarrow A(\ulcorner \varphi \urcorner)$. If we interpret $A$ as some "inference transform," $\varphi$ is the fixed point of that transform — it talks about itself and remains invariant under the transform. The Gödel sentence $G \equiv \neg\mathsf{Prov}(\ulcorner G \urcorner)$ is the fixed point of the "negation of provability" transform.

**Curry-Howard's Reflection Here**: Under the CH isomorphism, formulas correspond to types, proof sequences correspond to reduction sequences of programs, $f$ corresponds to a function on types $F: \mathsf{Type} \to \mathsf{Type}$. The fixed point $f(\varphi) = \varphi$ corresponds to recursive types $F(T) \cong T$ — for example, the naturals $\mathsf{Nat} \cong 1 + \mathsf{Nat}$ is the fixed point of $F(T) = 1 + T$.

The MP game unifies three things in a single framework: **the structure of proof, the orbits of dynamical systems, and the fixed points of recursive types**. They are the same mathematical object read in three different languages: logic, analysis, and type theory.

::: info Professor Pallas's Cat's Commentary
The true danger of this game lies in the third fate: the infinitely extending orbit. You cannot determine from outside whether an orbit will halt — this is not a matter of insufficient computational power, but undecidability in principle. Every time you write down a proof, you are in effect claiming: this orbit will halt, and it will halt precisely where you want it to. This claim, in a sufficiently strong system, cannot be verified by the system itself.
:::

---

## 22.3 Dependent Types: Propositions as Objects

The Curry-Howard correspondence holds in simple type theory. But what happens if we make the type system strong enough — allowing types to depend on values?

Dependent types allow expressions like: the type $\mathsf{Vec}(A, n)$ means "a vector of type $A$ of length $n$," where $n$ is a concrete natural number, not a type variable. The "inhabitants" (programs) of this type are all vectors whose length is exactly $n$.

:::details Dependent Types: Letting Types Carry "Proofs" (Prior Work: Martin-Löf, 1975)
In ordinary programming languages, types and values are separated: `int` is a type, `42` is a value, and the type does not know what the specific value is.

Dependent types break this separation: **types can depend on values**.

Examples:
- Ordinary type: `Vec<int>` (integer vector, regardless of length)
- Dependent type: `Vec<int, 5>` (a vector of exactly 5 integers, the length encoded in the type)

**Why is this useful?** Because more precise types = stronger compile-time checks. If the function signature is `concat : Vec<A, m> -> Vec<A, n> -> Vec<A, m+n>`, the compiler will **automatically verify** that the length after concatenation is correct — no runtime checking needed, bounds violations are impossible.

**Relation to Curry-Howard**: In dependent types, "$\forall x. P(x)$" (for all $x$, $P(x)$ holds) corresponds to the $\Pi$-type (for each $x$, there is a proof of type $P(x)$). Writing a program = constructing a proof. This is what makes systems like Lean/Coq/Agda possible: you are writing mathematical proofs in a programming language.
:::


Under dependent types, the Curry-Howard correspondence extends to first-order logic:

| First-Order Logic | Dependent Types |
|---------|---------|
| $\forall x.\, P(x)$ | $\Pi$-type: $\prod_{x : A} P(x)$ (function mapping from each $x$ to a proof of $P(x)$) |
| $\exists x.\, P(x)$ | $\Sigma$-type: $\sum_{x : A} P(x)$ (a pair of an $x$ with an inhabitant of $P(x)$) |

Dependent type systems (such as Martin-Löf type theory) realize an ancient dream of mathematics: **the construction of mathematical objects and the proof of mathematical propositions are handled in a unified way within the same language**. Proving a proposition is constructing a program with the corresponding type; constructing a mathematical object is writing an inhabitant of the corresponding type.

This is not a formal trick, but a profound unification: mathematics and computation, propositions and data, proofs and programs — in a sufficiently precise language, they are the same thing.

---

## 22.4 Self-Reasoning of a Reasoning System: The Boundary Reappears

Now we arrive at the core question of this chapter.

A reasoning system — whether a formal logical system, a learning algorithm, or a large language model — when it becomes sufficiently complex and begins to reason about itself, the pattern of Chapter 15 inevitably emerges.

**At the formal system level**: Gödel's theorem says that any sufficiently strong consistent system has true propositions that it cannot prove (including propositions about its own consistency). Adding $G$ as an axiom yields a stronger system, but that stronger system has its own $G'$. This hierarchy has no top.

**At the learning system level**: The conclusion of Chapter 21 is that the inductive bias of learning cannot be determined from within the data — it is an external input. A learning system cannot infer from its own training data whether its inductive bias is appropriate, just as a formal system cannot prove its own consistency from its own axioms.

::: info Professor Pallas's Cat's Commentary
Read this sentence twice. This is not saying that inductive bias is hard to choose, but that in principle, data cannot tell you whether your inductive bias is appropriate — just as a formal system cannot prove its own consistency from within. The true connection between Chapter 21 and Chapter 15 is here: the blind spot of learning and the incompleteness of logic are two languages for the same structural limitation.
:::

**At the large language model level**: This is the most cutting-edge and also most uncertain question. Through training, large language models acquire some internal representation of language structure. Does this representation contain some formal approximation of "inference rules"? When a model "reasons" about its own reasoning process (e.g., chain-of-thought), what is it doing?

::: info A Formal Conjecture about Transformers
An informal conjecture discussed in the research community: the attention mechanism of Transformers, at sufficient scale and training, implements a kind of **implicit causal inference** — by selecting relevant positions in context, it performs an approximate conditional independence test, which has structural similarity to the d-separation of Chapter 18. This is not a proven theorem but a conjecture requiring precise mathematical characterization. If it holds in some sense, then large language models are not merely pattern matchers but are performing some approximate causal inference — this conclusion would profoundly change our understanding of AI reasoning capabilities. But "in some sense" is a large escape hatch; existing theoretical tools cannot yet turn this conjecture into a theorem.
:::

---

## 22.5 Emergence: When Quantitative Change Triggers Qualitative Change

Self-reference and fixed points explain the algebraic structure of "a system reasoning about itself." But there is another phenomenon, commonly appearing in complex systems, that is even harder to formalize: **emergence**.

Emergence says: when there are enough components and sufficiently rich interactions, a system exhibits properties that no individual part possesses and that cannot be predicted from simple superposition of the parts.

In the training practice of large language models, emergent phenomena have already been observed: when model scale exceeds a certain threshold, certain capabilities (such as arithmetic reasoning, multi-step inference) suddenly appear, while being completely absent in smaller models. Such "suddenly appearing" capabilities are called emergent capabilities.

But emergence is extremely difficult to handle in formal theory. It is not a mathematical concept that can be precisely defined — at least, current mathematical tools are insufficient. We have descriptions, not explanations; observations, not predictions.

::: info Is Emergence a Real Phenomenon or a Measurement Illusion?
Some researchers have pointed out: so-called "emergence" may partly be a product of how we measure. If measured with continuous (rather than discrete) performance metrics, some capabilities thought to emerge suddenly actually grow smoothly — it is only that at some threshold, smooth growth crosses the minimum capability threshold required by the task and is noticed by the observer. This does not deny the existence of emergent phenomena, but demands more precise characterization: does the performance curve truly have a discontinuity, or do our measurement tools introduce a spurious discontinuity? This question currently has no settled answer.
:::

---

## 22.6 The Boundary of the Second Volume Is Also an Opening

From Chapter 14 to Chapter 22, we have walked this path:

From the foundations of formal systems, to the limits of consistency and completeness, to resource-sensitive linear logic, to the extension of truth values by probability, to the formalization of intervention by causal inference, to the geometric characterization of reasoning cost by complexity theory, to the formal contract of heuristics, to learning as inverse inference, and finally to self-reference and emergence.

Every chapter was forced by the questions left by the previous chapter. Every chapter's ending left a question that made the next chapter inevitable.

This path has arrived here, touching several boundaries:

- The Gödel boundary: a sufficiently strong system has true propositions that cannot be seen from within.
- The Turing boundary: certain questions are in principle unanswerable by any algorithm.
- The complexity boundary: certain problems are in principle unsolvable efficiently.
- The inductive bias boundary: a learning system cannot determine from within whether its inductive bias is appropriate.

These boundaries are not obstacles but a map. They tell us: in the matter of reasoning, which regions are beyond the reach of both humans and machines, which regions are reachable but at high cost, and which regions can be made feasible through approximation.

Finally, there is a question that this book has not answered and cannot answer:

**Where, on this map, do the reasoning capabilities currently exhibited by AI systems lie?**

Where have they arrived, where are their boundaries, are they approaching the boundaries of some formal theory, or are they working in an entirely different place? This is not a question answerable with existing tools — not because the question is unimportant, but because we do not yet have a sufficiently precise language to state it.

Building such a language is the work of the coming decades.

---

## 22.7 From Logic to Dynamics: Three Lingering Questions

The MP game has given us a new perspective: the proof sequence $\{p_1, p_2, \dots, p_n\}$ is an orbit, $f: \mathsf{Formula} \to \mathsf{Formula}$ is the evolution operator, $p_{n+1} = f(p_n)$.

The orbit has three fates: finite halting (fixed point), finite cycle, infinite extension.

But this game leaves three questions that logical tools cannot answer:

**Question One: Where does the orbit halt?**

The fixed point $f(\varphi) = \varphi$ exists, but starting from an arbitrary initial formula $p_0$, which fixed point will the system ultimately converge to? Logic only tells us that fixed points exist, not the direction of attraction. This is a question of dynamics.

**Question Two: What is the energy of convergence?**

If the orbit does converge, there must be some quantity that decreases monotonically — some kind of "energy." Logic has no concept of energy. But if we view the belief space $\mathcal{P}$ as the state space of a dynamical system, an energy function naturally emerges.

**Question Three: Where does the energy come from?**

The traditional approach is to manually guess an energy function and verify that it decreases. But if we have already observed the system converging to some prior anchor point $A$, can we do the reverse — derive the energy function from the convergence behavior rather than guessing it?

These three questions are the entrance to Chapter 23.

::: info Professor Pallas's Cat's Commentary
The MP game is the ferry crossing between logic and dynamics. On the logic shore, we ask "does a proof exist"; on the dynamics shore, we ask "where does the system go." The fixed point is a landmark shared by both shores — in logic, it is the endpoint of proof; in dynamics, it is the attractor. Cross over, and you will discover that the attractor has a name: the Yonglin Limit.
:::

---

## Unresolved

**How far can the Curry-Howard correspondence extend?** Currently, the correspondence is precise at the level of simple type theory and first-order logic. Higher-order logics (second-order logic, higher-order logic), modal logic, and linear logic (Chapter 16) all have different forms of Curry-Howard correspondence, but with varying degrees of precision and completeness. Is there a unified framework that brings all formal logical systems under the "propositions-as-types" perspective? This is the question that Homotopy Type Theory (HoTT) attempts to answer, but HoTT itself is still under development.

**Does emergence have a mathematical theory?** Complexity science, phase transition theory, and information theory each provide partial characterizations of emergence, but there is no unified, predictive theory of emergence. Can the emergent capabilities of large language models be predicted and explained within some formal framework, or is it an essentially empirical phenomenon that cannot be captured in advance by theory?

**The capability boundary of self-referential reasoning**: When an AI system reasons about its own reasoning process, what true propositions about itself can it discover, and which boundaries — invisible from within — must it necessarily encounter? This is not merely a philosophical question but a practical one concerning AI safety and interpretability — a system that cannot reason about its own limitations: under what circumstances will it substitute confident error for honest uncertainty?

**The dynamical gap left by the MP game**: If a proof sequence is an orbit, then a fixed point is merely the name for "halting," not the guarantee of "halting at the right place." What kind of energy function can constrain this orbit? Does this energy come from training data, prior anchor points, or the system's own structure? This question has already exceeded the logical language of this chapter and must be handed over to Chapter 23.

---

## Exercises

**★ Warm-up**

The core of the Curry-Howard correspondence is: propositions correspond to types, proofs correspond to programs. For the following logical propositions, write out their corresponding types in type theory (using function types $\to$, product types $\times$, sum types $+$).

1. $A \to A$ (identity law)
2. $A \to (B \to A)$ (weakening)
3. $A \land B \to A$ (conjunction elimination)
4. $A \to A \lor B$ (disjunction introduction)

For item 1, write a concrete program (function) that "inhabits" this type, and verify that it indeed corresponds to "having proved $A \to A$."

---

**★★ Derivation**

1. **Fixed-point expansion**: The $Y$ combinator satisfies $Y\,f = f\,(Y\,f)$. Manually expand the first three steps of $Y\,(\lambda x.\, \text{if } x=0 \text{ then } 1 \text{ else } x \times (x-1))$ (interpret this as the recursive definition of the factorial function). Trace the expansion process and explain clearly how $Y$ "manufactures" recursion.

2. **Comparison of the Gödel sentence and the Y combinator**: The Gödel sentence $G$ says "I am unprovable"; $Y\,f$ satisfies $Y\,f = f\,(Y\,f)$. Both are self-referential constructions, but with different outcomes: $G$ leads to undecidability, $Y\,f$ leads to infinite looping (or recursive unfolding). List their similarities and differences in the following aspects: ① the mechanism of self-reference; ② the consequence of self-reference; ③ the way self-reference is "resolved" (if at all).

---

**★★★ Challenge**

Dependent type systems (such as Martin-Löf type theory) relate $\forall x.\, P(x)$ to the $\Pi$-type: a function that, for each $x$, returns a proof of $P(x)$.

Consider the proposition: "For all natural numbers $n$, the successor of $n$ is not equal to $0$," i.e., $\forall n \in \mathbb{N}.\, S(n) \neq 0$.

1. In dependent type theory, what type does this proposition correspond to? Write it out in the language of $\Pi$-types.
2. What does a "proof" of this proposition (i.e., an inhabitant of this type) look like? What input does it accept, and what does it return?
3. Now consider: type-checking (verifying that a program has a given type) is equivalent to verifying a proof. Is this verification operation decidable? (Hint: dependent types make type-checking harder — compared to simple type systems, how does the complexity change?)

The third question touches upon the practical limitation of dependent type systems: the more powerful the functionality, the higher the cost of type-checking; the type-checking of some dependent type systems is even undecidable — this is yet another recurrence of the law that runs throughout this book: "the more powerful, the higher the cost."

The third question has no standard answer. It is the final open question posed by this book, and also the kind of question the second volume wishes to leave for the reader to ponder: precise formal tools, at the boundary, begin to touch problems that lie beyond the range of formalization. At that boundary, logic and intuition, proof and conjecture, form and meaning, intertwine in a way we have not yet fully understood.

# Chapter 22: Self-Reference and Emergence — When a Reasoning System Starts Reasoning About Itself

> The Curry-Howard correspondence tells us: a proof is a program. So what is a program proving?

---

The last paragraph of Chapter 21 brought us here: a reasoning system strong enough to start including propositions about itself. At this moment, the Gödel structure of Chapter 15 reappears.

But Chapter 22 is not a repeat of Chapter 15. Gödel was about the boundary — the true propositions you can never reach. Chapter 22 is about the openings — the new things emerging around that boundary.

Nine chapters of logical deduction into Volume 2, and we are finally high enough to look down. This is one of the most honest chapters: some questions, we can only state, not solve.

---

## 22.1 Curry-Howard: A Proof Is a Program

Chapter 14 built a formal system: propositions, inference rules, proofs. Chapter 19 found that computation and logic are equivalent at the level of complexity: SAT is NP-complete, and finding a proof in propositional logic is, in the worst case, exactly as hard as SAT.

But how deep is the equivalence?

The **Curry-Howard correspondence** (also called the propositions-as-types correspondence) gives the answer: as deep as it can possibly be.

Precisely, it sets up the following correspondence:

| Logic | Type Theory / Programs |
|------|-------------|
| Proposition $A$ | Type $A$ |
| Proof of proposition $A$ | Term of type $A$ (a program) |
| $A \to B$ (implication) | Function type $A \to B$ |
| $A \land B$ (conjunction) | Product type $A \times B$ |
| $A \lor B$ (disjunction) | Sum type $A + B$ |
| $\bot$ (false proposition) | Empty type (uninhabited type) |
| Modus ponens | Function application |
| Deduction theorem | Function abstraction ($\lambda$ abstraction) |

This is not a metaphor — it is an isomorphism: every step of a logical derivation corresponds to a step of computation in a program; normalization of a proof (reducing it to its simplest form) corresponds to evaluation of a program (running it until it terminates).

::: info Professor Pallas says
Many people nod at this point and then go right on treating "logic" and "program" as two different things. An isomorphism means: every step you prove in logic is one computation step in a programming language; every function you write is "proving" some proposition. If this doesn't blow your mind, you haven't really understood what this correspondence is saying.
:::

A proposition "has a proof" exactly when its corresponding type "has a program" — the corresponding type is "inhabited". **A proof is a program, a proposition is a type, and verification is running the program.**

::: info Why This Correspondence Is So Striking
Before the Curry-Howard correspondence, mathematics and computer science were two neighboring but separate fields. The correspondence tells us: they are two languages for the same thing. Every mathematical theorem can be understood as a program; every program is "proving" some proposition — proving that its output type can be constructed from its input type. This is not a philosophical similarity; it is an equivalence that can be verified precisely with formal machinery. The proof assistants Lean, Coq, and Agda are built on exactly this equivalence: writing a program is writing a proof, and type checking is proof verification.
:::

---

## 22.2 Fixed Points: The Algebra of Self-Reference

The Gödel sentence $G$ of Chapter 15 is, at heart, a self-referential construction: $G$ talks about its own provability. The mathematical heart of this construction is the diagonalization lemma — a mechanism for building "propositions that talk about themselves".

But self-reference does not only appear in Gödel's theorem. It has a more general algebraic structure: the **fixed point**.

**Definition (fixed point)**: given a function $f$, if there exists $x$ such that $f(x) = x$, then $x$ is called a fixed point of $f$.

In the lambda calculus (the simplest core language of programs), there is a famous fixed-point combinator $Y$:

$$Y = \lambda f.\, (\lambda x.\, f\,(x\,x))\,(\lambda x.\, f\,(x\,x))$$

For any function $f$, $Y\,f$ is a fixed point of $f$: $Y\,f = f\,(Y\,f)$.

Unrolled, this equation shows that $Y\,f$ "runs" itself — it passes itself as an argument to $f$. The $Y$ combinator is the algebraic root of recursion: every recursive definition, in its essence, is looking for the fixed point of some function equation.

:::details The Y Combinator and the Lambda Calculus: The "Simplest Core" of Programs
The lambda calculus is a model of computation proposed by Alonzo Church in the same era as Turing (the 1930s), and it is more compact than the Turing machine: only three kinds of elements — variables, function abstraction ($\lambda x. \text{body}$), and function application ($f\,a$).

$\lambda x. \text{body}$ means "a function that takes input $x$ and returns body". For example, $\lambda x. x+1$ is the "add-1 function".

The $Y$ combinator is the tool for doing "recursion" in the lambda calculus. Ordinary programming languages let you write `def f(x): return f(x-1)` directly, but in the pure lambda calculus functions have no names, so they cannot refer to themselves directly. The $Y$ combinator gets around this restriction: $Y\,g$ passes $g$ itself as an argument to $g$, which achieves recursion.

**Why does this relate to self-reference?** The $Y$ combinator is "a function that finds its own fixed point" — it encodes self-reference as pure functional computation. Gödel's diagonalization lemma is, in essence, the same mathematical structure appearing in logic.
:::

Gödel's diagonalization lemma, in this language, is a fixed point of the "provability predicate": there exists a proposition $G$ such that $G \equiv \neg\mathsf{Prov}(\ulcorner G \urcorner)$ — $G$ is a fixed point of the predicate "I am not provable". The program $D$ in the halting problem is likewise a fixed-point construction: the behavior of $D$ on input $D$ depends on the behavior of $D$ on input $D$.

**Self-reference, fixed points, and diagonalization are different names for the same mathematical structure in different contexts.**

### 22.2.3 Hands-on: Unrolling the Y Combinator in Three Steps

The definition of the $Y$ combinator: $Y = \lambda f.\, (\lambda x.\, f\,(x\,x))\,(\lambda x.\, f\,(x\,x))$.

Trace the unrolling with the recursive definition of the factorial function. Let $f_{\text{fact}}$ be a "factorial template" — it knows the logic of factorial but not how to call itself recursively:

$$f_{\text{fact}} = \lambda \text{self}.\, \lambda n.\, \text{if } n = 0 \text{ then } 1 \text{ else } n \times \text{self}(n-1)$$

Note that the `self` parameter of $f_{\text{fact}}$ is a function — give $f_{\text{fact}}$ a function that computes factorials, and $f_{\text{fact}}$ returns a new factorial function.

Now compute the behavior of $Y\,f_{\text{fact}}$ at $n = 3$.

**Step 0 (definition)**:
$$Y\,f_{\text{fact}} = (\lambda x.\, f_{\text{fact}}\,(x\,x))\,(\lambda x.\, f_{\text{fact}}\,(x\,x))$$

Let $w = \lambda x.\, f_{\text{fact}}\,(x\,x)$; then $Y\,f_{\text{fact}} = w\,w$.

**Step 1 (unroll one layer)**:
$$w\,w = f_{\text{fact}}\,(w\,w)$$

Pass $w\,w$ to the `self` parameter of $f_{\text{fact}}$. Unroll the definition of $f_{\text{fact}}$:
$$f_{\text{fact}}\,(w\,w) = \lambda n.\, \text{if } n = 0 \text{ then } 1 \text{ else } n \times (w\,w)(n-1)$$

Now $Y\,f_{\text{fact}}$ is a function that takes $n$ and, when $n > 0$, calls $(w\,w)(n-1)$.

**Step 2 (evaluate at $n = 3$)**:
$$(Y\,f_{\text{fact}})(3) = \text{if } 3 = 0 \text{ then } 1 \text{ else } 3 \times (w\,w)(2)$$

Compute $(w\,w)(2)$ — this is exactly the same as computing $(Y\,f_{\text{fact}})(2)$! Because $w\,w = Y\,f_{\text{fact}}$. So:
$$(w\,w)(2) = f_{\text{fact}}\,(w\,w)(2) = \text{if } 2 = 0 \text{ then } 1 \text{ else } 2 \times (w\,w)(1)$$

**Step 3 (keep unrolling)**:
$$(w\,w)(1) = f_{\text{fact}}\,(w\,w)(1) = \text{if } 1 = 0 \text{ then } 1 \text{ else } 1 \times (w\,w)(0)$$
$$(w\,w)(0) = f_{\text{fact}}\,(w\,w)(0) = \text{if } 0 = 0 \text{ then } 1 \text{ else } \ldots = 1$$

**Collecting results**:
$$(w\,w)(1) = 1 \times 1 = 1$$
$$(w\,w)(2) = 2 \times 1 = 2$$
$$(Y\,f_{\text{fact}})(3) = 3 \times 2 = 6 \checkmark$$

**The key observation**: at every step, $(w\,w)$ "feeds" itself to $f_{\text{fact}}$ when recursion is needed. $f_{\text{fact}}$ does not know that it is calling itself — it merely received a function argument `self` and calls it. But the construction of $w\,w$ guarantees that `self` is $w\,w$ itself. The $Y$ combinator has no magic — it just encodes "calling itself" as "applying a function to itself".

**Compare this with the Gödel sentence $G$**:
- $Y$: $Y\,f = f\,(Y\,f)$ — $Y\,f$ is a fixed point of $f$; every unrolling of $Y\,f$ passes itself back to $f$
- $G$: $G \leftrightarrow \neg\mathsf{Prov}(\ulcorner G \urcorner)$ — $G$ is a fixed point of the predicate "not provable", and $G$ asserts a proposition about its own provability

Both are the same mathematical structure: find an object such that the result of applying a certain transformation to it is, in an equivalence sense, the object itself. The $Y$ combinator uses function application as the transformation; the diagonalization lemma uses the provability predicate. Same mathematical structure, different instances.

::: info Professor Pallas says
Unrolling the Y combinator three steps by hand gives you a feeling you cannot get from the definition: self-reference is not black magic, but a mechanism you can trace step by step. $Y$ does not "see through" itself; it merely mechanically applies a function to itself. But this one mechanical act creates recursion, formal models of self-awareness, and the computational version of Gödel's theorem. Complexity comes from iterating simple rules under self-reference — one of the deepest themes of this whole book.
:::

### 22.2.5 The MP Game: Proof Sequences as Dynamical Systems

Modus ponens (MP) is the most basic inference rule of propositional logic: with $p$, and with $p \to q$, you get $q$.

Textbooks give a static definition of proof: a finite sequence of formulas $\{p_1, p_2, \ldots, p_n\}$, where each $p_i$ is either an axiom or obtained from earlier formulas by MP. This is a static definition — it describes an already-finished object.

But what if we look at it dynamically?

**Setting up the game**: define a function $f: \mathsf{Formula} \to \mathsf{Formula}$, and require the proof sequence to satisfy $p_{n+1} = f(p_n)$. Then a proof becomes an **orbit**:

$$p_1 \xrightarrow{f} f(p_1) \xrightarrow{f} f^2(p_1) \xrightarrow{f} \cdots$$

Each application of $f$ is one MP inference: from the current formula, derive the next one. The proof sequence is a trajectory of this dynamical system.

**Three fates of an orbit**:

1. **Apparent stopping**: the sequence reaches some $p_k$ such that $f(p_k) = p_k$ — formally a fixed point. But this only says the successor term equals itself; it does not guarantee that $p_k$ is a true QED. $p_k$ might be a self-referential construction; "stopping" is just the special case of a cycle of period 1.
2. **Finite cycle**: there exist $k > j$ such that $p_k = p_j$ — the orbit circles around, no new content emerges.
3. **Infinite extension**: the sequence never terminates — this corresponds to undecidability. No algorithm can decide, for an arbitrary given $f$ and initial formula, whether the orbit will stop.

**The Cauchy condition and the subtlety of halting**: determining whether the first fate is "truly complete" is equivalent to asking: does the sequence become Cauchy?

On a state space like $\{0, 1\}$ (or any discrete space), the Cauchy condition

$$\forall \varepsilon > 0,\ \exists N,\ \forall m,n > N:\ d(p_m, p_n) < \varepsilon$$

degenerates into "the sequence is eventually constant" — that is, eventually fixed. $f(p_k) = p_k$ satisfies this condition. But the Cauchy condition only guarantees that the limit exists, not that the limit point is the "true answer".

Real computers have finite stack space, so we impose a truncation condition by hand — not stopping automatically at the limit, but halting imposed from outside. This external condition, in a sufficiently strong system, cannot be internalized by the system itself.

**A fixed point is the end of a proof** — but only when the limit point is meaningful. When $f(\varphi) = \varphi$, $\varphi$ has "stabilized" under inference, but where it stabilized needs external verification.

This is not just a metaphor. The diagonalization lemma says: for any formula $A(x)$, there exists a sentence $\varphi$ such that $\varphi \leftrightarrow A(\ulcorner \varphi \urcorner)$. Reading $A$ as some kind of "inference transformation", $\varphi$ is a fixed point of that transformation — it talks about itself and is unchanged under the transformation. The Gödel sentence $G \equiv \neg\mathsf{Prov}(\ulcorner G \urcorner)$ is the fixed point of the transformation "not provable".

**Curry-Howard in view**: under the CH isomorphism, formulas correspond to types, proof sequences correspond to reduction sequences of programs, and $f$ corresponds to a function $F: \mathsf{Type} \to \mathsf{Type}$ on types. The fixed point $f(\varphi) = \varphi$ corresponds to a recursive type $F(T) \cong T$ — for instance, the natural numbers $\mathsf{Nat} \cong 1 + \mathsf{Nat}$ are the fixed point of $F(T) = 1 + T$.

The MP game brings three things together in one framework: **the structure of proofs, the orbits of a dynamical system, and the fixed points of recursive types**. They are different readings, in the languages of logic, analysis, and type theory, of the same mathematical object.

::: info Professor Pallas says
The real danger of this game is in the third fate: an orbit that extends forever. You cannot tell from outside whether an orbit will stop — this is not because your computing power is insufficient, but because it is undecidable in principle. Every time you write down a proof, you are actually claiming: this orbit will stop, and it will stop where I want it to. In a sufficiently strong system, this claim cannot be verified by the system itself.
:::

### 22.2.6 The MP Game in Code

The orbits are small enough to run. The program below shows all three fates of an orbit – apparent stopping, finite cycle, infinite extension – then the Cauchy trap, and finally a miniature of the mirror: the Goedel sentence as a fixed point of the transformation "not provable."

```python
#!/usr/bin/env python3
"""
mp_game.py -- the MP Game of Chapter 22: proof sequences as orbits.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 22, Section
22.2.5 "The MP Game: Proof Sequences as Dynamical Systems."

A proof is usually a finished object: a finite sequence of formulas. The
MP game reads it dynamically: p_{n+1} = f(p_n), one modus ponens per
step, and a proof becomes an orbit of a dynamical system. An orbit has
exactly three fates -- and this program runs all three, plus the two
subtleties the book warns about:

  1. apparent stopping:  f(p) = p, but is p a TRUE QED?
  2. finite cycle:       no new content ever appears;
  3. infinite extension: the orbit never ends (undecidable, in general);
  4. the Cauchy trap:    a converging sequence whose limit point is not
                         where the transformation rests;
  5. the mirror:         the Goedel sentence as a fixed point of the
                         transformation F(phi) = "not provable(phi)",
                         built here as a tiny self-referential program
                         (Chapter 15's companion script runs the full
                         numbering machinery; this is the miniature).

Run:  python3 mp_game.py
"""


def orbit(f, x0, n):
    """The first n+1 points of the orbit x0, f(x0), f(f(x0)), ..."""
    out = [x0]
    x = x0
    for _ in range(n):
        x = f(x)
        out.append(x)
    return out


def demo():
    print("=" * 70)
    print(" THE MP GAME -- proof sequences as dynamical systems")
    print(" The Reasoning Kingdom, Volume 2, Chapter 22 (Sec. 22.2.5)")
    print("=" * 70)

    # Fate 1: apparent stopping.
    f1 = lambda x: x if x <= 0 else x - 1
    print("\nFate 1 -- apparent stopping:  f(x) = max(x-1, 0), start at 5")
    print("  orbit:", orbit(f1, 5, 8))
    print("  f(0) = 0, a fixed point. But 'stopping' only says the")
    print("  successor equals itself -- it does not guarantee 0 is a TRUE")
    print("  QED. Apparent stopping is a cycle of period 1.")

    # Fate 2: finite cycle.
    f2 = lambda x: (x + 1) % 3
    print("\nFate 2 -- finite cycle:  f(x) = (x+1) mod 3, start at 0")
    print("  orbit:", orbit(f2, 0, 7))
    print("  round and round; no new content ever emerges.")

    # Fate 3: infinite extension.
    f3 = lambda x: x + 1
    print("\nFate 3 -- infinite extension:  f(x) = x+1, start at 0")
    print("  orbit (first 8 points):", orbit(f3, 0, 7))
    print("  it never terminates. And for an arbitrary f, no algorithm")
    print("  can decide whether the orbit will ever stop -- that is")
    print("  undecidability in principle, not slow hardware.")

    # The Cauchy trap.
    print("\nThe Cauchy trap:")
    print("  sequence p_n = 1/n is Cauchy and converges to 0,")
    print("  but under the transformation f(x) = x + 1, f(0) = 1.")
    print("  the limit exists; the transformation does not rest there.")
    print("  The Cauchy condition guarantees the limit EXISTS -- not that")
    print("  the limit point is the TRUE answer.")

    # The mirror: fixed point of "not provable".
    THEOREMS = {"T1", "T2"}   # a toy provability predicate over strings

    def provable(s):
        return s in THEOREMS

    q = "q = %r; content = not provable(q %% q)"
    G = q % q
    scope = {"provable": provable}
    exec(G, {"__builtins__": {}}, scope)
    print("\nThe mirror -- G as a fixed point of F = 'not provable':")
    print(f"  G = {G}")
    print(f"  what G says, computed: {scope['content']}")
    print(f"  provable(G) = {provable(G)}")
    print("  so G <-> not provable(G): G is a fixed point of the")
    print("  transformation F(phi) = not Prov(<<phi>>). The diagonalization")
    print("  lemma in Chapter 15 is the same fixed point, grown to full")
    print("  arithmetic; the miniature here shows its shape.")

    print("=" * 70)


if __name__ == "__main__":
    demo()
```

---

## 22.3 Dependent Types: Propositions as Objects

The Curry-Howard correspondence holds in simple type theory. But what happens if the type system is made strong enough — allowing types to depend on values?

Dependent types allow expressions like: the type $\mathsf{Vec}(A, n)$ means "vectors of length $n$ with elements of type $A$", where $n$ is a specific natural number, not a type variable. The "inhabitants" (programs) of this type are all the vectors of exactly length $n$.

:::details Dependent Types: Making Types Carry "Proofs" (Earlier Work: Martin-Löf, 1975)
In ordinary programming languages, types and values are separate: `int` is a type, `42` is a value, and a type does not know what the specific value is.

Dependent types break this separation: **types can depend on values**.

Example:
- Ordinary type: `Vec<int>` (a vector of integers, length unexplained)
- Dependent type: `Vec<int, 5>` (a vector of exactly 5 integers; the length is encoded in the type)

**Why is this useful?** Because a more precise type = stronger compile-time checking. If the function signature is `concat : Vec<A, m> -> Vec<A, n> -> Vec<A, m+n>`, the compiler automatically verifies that the length after concatenation is correct — no runtime check needed, out-of-bounds is impossible.

**Relation to Curry-Howard**: in dependent types, "$\forall x. P(x)$" (for all $x$, $P(x)$ holds) corresponds to a $\Pi$-type (for each $x$, a proof of type $P(x)$). Writing a program = constructing a proof. This makes systems like Lean/Coq/Agda possible: you are writing mathematical proofs in a programming language.
:::

Under dependent types, the Curry-Howard correspondence extends to first-order logic:

| First-Order Logic | Dependent Types |
|---------|---------|
| $\forall x.\, P(x)$ | $\Pi$-type: $\prod_{x : A} P(x)$ (a function mapping each $x$ to a proof of $P(x)$) |
| $\exists x.\, P(x)$ | $\Sigma$-type: $\sum_{x : A} P(x)$ (a pair of an $x$ such that $P(x)$ is inhabited) |

Dependent type systems (such as Martin-Löf type theory) realize an old dream of mathematics: constructing mathematical objects and proving mathematical propositions, handled together in the same language. Proving a proposition is constructing a program with the corresponding type; constructing a mathematical object is writing an inhabitant of the corresponding type.

This is not formal trickery, but a deep unification: mathematics and computation, propositions and data, proofs and programs — in a language precise enough, they are the same thing.

---

## 22.4 A Reasoning System Reasoning About Itself: The Boundary Returns

Now to the most central question of this chapter.

A reasoning system — whether a formal logic system, a learning algorithm, or a large language model — once it is complex enough and starts reasoning about itself, the pattern of Chapter 15 inevitably appears.

**At the level of formal systems**: Gödel's theorem says that any sufficiently strong consistent system has true propositions it cannot prove (including the proposition about its own consistency). Add $G$ to the axioms and you get a stronger system, but this stronger system has its own $G'$. This hierarchy has no top.

**At the level of learning systems**: Chapter 21 concluded that a learning system's inductive bias cannot be determined from inside the data — it is external input. A learning system cannot infer from its own training data whether its inductive bias is appropriate, just as a formal system cannot prove its own consistency from its own axioms.

::: info Professor Pallas says
Read that sentence twice. This is not saying that inductive bias is hard to choose; it is saying that, in principle, the data cannot tell you whether your inductive bias is appropriate — just as a formal system cannot prove its own consistency from inside. The real connection between Chapter 21 and Chapter 15 is here: the blind spot of learning and the incompleteness of logic are two languages for the same structural limit.
:::

**At the level of large language models**: this is the frontier, and the most unsettled question. Through training, a large language model obtains some internal representation of linguistic structure. Does this representation contain some form of approximation of "inference rules"? When the model "reasons" about its own reasoning process (for instance, chain-of-thought), what is it doing?

::: info A Formal Conjecture about Transformers
An informal conjecture discussed in the research community: the attention mechanism of a Transformer, at sufficient scale and with sufficient training, implements some kind of implicit causal inference — by selecting relevant positions in the context, it performs an approximate test of conditional independence, which has a structural similarity to the d-separation of Chapter 18. This is not a proven theorem, but a conjecture that needs precise mathematical characterization. If it holds in some sense, then large language models are not merely pattern matchers; they are performing some approximate causal inference — and this conclusion would deeply reshape our understanding of AI reasoning ability. But "in some sense" is a large escape hatch; existing theoretical tools cannot yet turn this conjecture into a theorem.
:::

---

## 22.5 Emergence: When Quantity Turns Into Quality

Self-reference and fixed points explain the algebraic structure of "a system reasoning about itself". But there is another phenomenon, widespread in complex systems and even harder to formalize: **emergence**.

Emergence says: when the parts are numerous enough and the interactions rich enough, the system exhibits properties that no single part has, and that cannot be predicted from a simple sum of the parts.

In the training practice of large language models, emergence has been observed: when the model scale crosses some threshold, certain abilities (such as arithmetic reasoning, multi-step inference) suddenly appear, and are completely absent in smaller models. These "suddenly appearing" abilities are called emergent capabilities.

### Three Levels of Emergence

Emergence is not a single phenomenon; it happens on at least three levels, each corresponding to different theoretical tools.

**Level 1: compositional emergence.** Simple atoms combine into new behavioral capacity, even though no single atom has it on its own. A water molecule consists of hydrogen and oxygen — hydrogen and oxygen are each not "wet", but combined they acquire the macroscopic property of "wet". This is the most basic form of emergence, and phase-transition theory can roughly describe it — near the critical point, the system's behavior changes drastically.

**Level 2: computational emergence.** The single rule of a Turing machine (the $\delta$ function) is extremely simple — based on the current state and the symbol read, decide the next state, the symbol to write, and the head's direction of movement. You cannot see from a single rule that it can do calculus, play chess, or write poetry. But the "universal Turing machine" is a simple $\delta$ plus a long tape. Computational power is emergent — it belongs to no single rule, but arises from the combination of rules + infinite tape + enough time.

**Level 3: representational emergence.** In large language models, the abilities that billions of parameters produce after training (such as analogical reasoning, emotion recognition, code generation) are not the function of any single parameter, and were not in the designer's plans. The training objective (next-token prediction) contains no explicit instruction to "understand emotions" or "write code" — but these abilities show up at sufficient scale.

### The Problem: Is There a Theory of Emergence?

But emergence is extremely hard to handle in formal theory. It is not a mathematical concept that can be precisely defined — at least, the current mathematical tools are not enough. We have descriptions, no explanations; observations, no predictions.

Information theory offers some clues. Emergence can be understood as: the system has a total description length shorter than that of its parts. A gas consists of $10^{23}$ molecules, but its state is completely described by a few macroscopic variables (temperature, pressure, volume). The description length drops from $10^{23}$ to 3 — this is the "information compression" face of emergence. But how to derive this compressive description from the interactions of the parts is, at present, without a general theory.

Dynamical systems theory offers another view: emergence corresponds to phase transitions — when a parameter crosses some critical value, the behavior changes drastically and qualitatively. This view is explanatory for physical systems, but for neural-network emergence (abilities suddenly appearing as scale grows) there is no precise predictive framework yet.

::: info Is Emergence a Real Phenomenon, or an Illusion of Measurement?
Some researchers point out: what is called "emergence" may be, in part, a product of the way it is measured. If measured with continuous (rather than discrete) performance metrics, some abilities taken to be suddenly emergent actually grow smoothly — just at some threshold, the smooth growth crosses the minimum ability bar required by the task and is noticed by the observer. This does not deny the existence of emergence, but demands a more precise characterization: does the performance curve really have a discontinuity, or do our measuring tools introduce a false discontinuity? There is currently no consensus.
:::

---

## 22.6 The Boundary of Volume 2, Which Is Also an Opening

From Chapter 14 to Chapter 22, the path has been this:

From the foundations of formal systems, to the limits of consistency and completeness, to resource-sensitive linear logic, to the expansion of probability over truth, to the formalization of intervention in causal inference, to complexity theory's geometric characterization of the cost of reasoning, to the formal contract of heuristics, to learning as inverse inference, and finally to self-reference and emergence.

Each chapter was forced out by the questions left by the previous one. The ending of each chapter left a question, and that question made the next chapter inevitable.

Along this path, we have touched several boundaries:

- The Gödel boundary: sufficiently strong systems have true propositions they cannot see from inside.
- The Turing boundary: certain problems cannot, in principle, be answered by any algorithm.
- The complexity boundary: certain problems cannot, in principle, be solved efficiently.
- The inductive bias boundary: a learning system cannot determine from inside whether its own inductive bias is appropriate.

These boundaries are not obstacles; they are maps. They tell us: in this business of reasoning, which places are unreachable by both humans and machines, which are reachable but costly, and which can be traded for feasibility with approximations.

Finally, there is one question this book does not answer, and cannot:

**Where does the reasoning ability AI systems currently display sit on this map?**

How far has it reached, where are its boundaries, is it approaching the boundaries of some formal theory, or is it working in a completely different place? This is not a question answerable with existing tools — not because the question is unimportant, but because we do not yet have a language precise enough to state it.

Building such a language is the work of the coming decades.

---

## 22.7 From Logic to Dynamics: Three Leftover Questions

The MP game gave us a new perspective: a proof sequence $\{p_1, p_2, \ldots, p_n\}$ is an orbit, $f: \mathsf{Formula} \to \mathsf{Formula}$ is the evolution operator, and $p_{n+1} = f(p_n)$.

An orbit has three fates: finite stopping (fixed point), finite cycle, infinite extension.

But the game leaves three questions that the tools of logic cannot answer:

**Question 1: where does the orbit stop?**

A fixed point $f(\varphi) = \varphi$ exists, but starting from an arbitrary initial formula $p_0$, which fixed point does the system eventually converge to? Logic only tells us that fixed points exist, not the direction of attraction. This is a question of dynamics.

**Question 2: what is the energy of convergence?**

If the orbit does converge, there must be some quantity decreasing monotonically — some "energy". Logic has no concept of energy. But if we view the belief space $\mathcal{P}$ as the state space of a dynamical system, the energy function emerges naturally.

**Question 3: where does the energy come from?**

The traditional approach is to hand-guess an energy function and verify that it decreases. But if we have already observed the system converging to some prior anchor $A$, can we go the other way — derive the energy function from the convergence behavior, rather than guessing it?

These three questions are the entrance to Chapter 23.

::: info Professor Pallas says
The MP game is the ferry between logic and dynamics. On the logic shore, we ask "does a proof exist?"; on the dynamics shore, we ask "where does the system go?". The fixed point is the landmark shared by both shores — in logic it is the end of a proof; in dynamics it is the attractor. Cross over, and you will find that the attractor has a name: the Yonglin limit.
:::

---

## Open Questions

**How far can the Curry-Howard correspondence extend?** At present, the correspondence is exact at the level of simple type theory and first-order logic. Higher-order logic (second-order logic, higher-order logic), modal logic, and linear logic (Chapter 16) all have Curry-Howard correspondences of different forms, but with differing degrees of exactness and completeness. Is there a unified framework that brings all formal logic systems into the "propositions as types" view? This is the question Homotopy Type Theory (HoTT) attempts to answer, but HoTT itself is still developing.

**Is there a mathematical theory of emergence?** Complexity science, phase-transition theory, and information theory each provide partial characterizations of emergence, but there is no unified, predictive theory of emergence. Can the emergent capabilities of large language models be predicted and explained within some formal framework, or is it in essence an empirical phenomenon that theory cannot capture in advance?

**The capability boundary of self-referential reasoning**: when an AI system reasons about its own reasoning process, what true propositions about itself can it discover, and which boundaries must it inevitably fail to see from inside? This is not just a philosophical question, but a practical question about AI safety and interpretability — under what circumstances will a system that cannot reason about its own limits substitute confident error for honest uncertainty?

**The dynamical gap left by the MP game**: if a proof sequence is an orbit, then a fixed point is only the name for "stopping", not a guarantee of "stopping in the right place". What kind of energy function can constrain this orbit? Does this energy come from the training data, from the prior anchor, or from the system's own structure? This question already goes beyond the logical language of this chapter and must be handed to Chapter 23.

---

## Exercises

**★ Warm-up**

The core of the Curry-Howard correspondence: propositions correspond to types, proofs correspond to programs. For each of the following logical propositions, write the corresponding type in type theory (using the function type $\to$, product type $\times$, and sum type $+$).

1. $A \to A$ (law of identity)
2. $A \to (B \to A)$ (weakening)
3. $A \land B \to A$ (conjunction elimination)
4. $A \to A \lor B$ (disjunction introduction)

For item 1, write out a concrete program (function) that "inhabits" this type, and verify that what it corresponds to is indeed "a proof of $A \to A$".

---

**★★ Derivation**

1. **Fixed-point unrolling**: the $Y$ combinator satisfies $Y\,f = f\,(Y\,f)$. Manually unroll the first three steps of $Y\,(\lambda x.\, \text{if } x = 0 \text{ then } 1 \text{ else } x \times (x-1))$ (read this as the recursive definition of the factorial function). Trace the unrolling and explain clearly how $Y$ "creates" recursion.

2. **Gödel sentence vs. the Y combinator**: the Gödel sentence $G$ says "I am unprovable"; $Y\,f$ satisfies $Y\,f = f\,(Y\,f)$. Both are self-referential constructions, but with different results: $G$ leads to undecidability, $Y\,f$ leads to infinite looping (or recursive unrolling). List their similarities and differences in: (1) the mechanism of self-reference; (2) the consequence of self-reference; (3) the way self-reference is "resolved" (if any).

---

**★★★ Challenge**

Dependent type systems (such as Martin-Löf type theory) correspond "$\forall x.\, P(x)$" to a $\Pi$-type: a function that returns, for each $x$, a proof of $P(x)$.

Consider the proposition: "for all natural numbers $n$, the successor of $n$ is not equal to $0$", i.e. $\forall n \in \mathbb{N}.\, S(n) \neq 0$.

1. In dependent type theory, what type does this proposition correspond to? Write it in the language of $\Pi$-types.
2. What does the "proof" of this proposition (that is, the program inhabiting this type) look like? What input does it take, and what does it return?
3. Now consider: type checking (verifying that a program has a given type) is equivalent to verifying a proof. Is this verification operation decidable? (Hint: dependent types make type checking harder — compared with simple type systems, how does the complexity change?)

Question 3 touches the practical limits of dependent type systems: the more powerful the functionality, the higher the cost of type checking; the type checking of some dependent type systems is even undecidable — yet another performance of the book-wide law "the more powerful, the more costly".

Question 3 has no standard answer. It is the last open question Volume 2 raises at the technical level — formal tools at the boundary begin to touch problems beyond the reach of formalization. On that boundary, logic and intuition, proof and conjecture, form and meaning, entangle each other in ways we have not yet fully understood. The book's final answer is in Chapter 23.

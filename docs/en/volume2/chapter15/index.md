# Chapter 15: Consistency and Completeness — The Two Walls of Formal Systems

> Can you build a system that never lies and yet knows everything?

---

Chapter 14 gave us a precision machine: a formal system accepts axioms, runs according to inference rules, and outputs theorems. This machine seems very reliable. But "seems" is not good enough. We need to know how reliable it **actually** is.

This problem has two dimensions, in opposite directions.

The first dimension: Does the system lie? That is, are the things it proves all truly valid? This is called **consistency**.

The second dimension: Does the system miss things? That is, can it prove everything that is true? This is called **completeness**.

Both requirements are very reasonable. In the early 20th century, the mathematician David Hilbert ambitiously believed that one could build a formal foundation for all of mathematics that simultaneously satisfied both conditions — rigorous, complete, and unassailable. This plan was called the Hilbert Program.

In 1931, a 26-year-old Austrian mathematician, with two theorems, completely destroyed this program.

---

## 15.0 The Mirror Proposition Game: When the System Begins to See Itself

The axiom stamp game of Chapter 14 was still relatively safe: you stamped, moved according to rules, and finally obtained theorems. Now we add a mirror to this game.

The rules change as follows: every formula, every proof, every string of symbols can be encoded as a natural number. Thus the system contains not only propositions that talk about numbers, like "$1+1=2$," but also propositions that talk about "whether a certain formula number is provable."

This is like a library suddenly acquiring its own catalog, and every book in the catalog can in turn reference the catalog itself. The game starts to become dangerous from here.

The player must take three steps:

1. **Numbering**: turn formulas into numbers, so that statements become objects operable inside the system.
2. **Referencing**: construct a predicate $\mathsf{Prov}(n)$, meaning "the formula numbered $n$ is provable."
3. **Looking in the mirror**: produce a proposition $G$ that speaks not about the external world, but about itself: $G$ is not provable.

If the system proves $G$, it proves a proposition claiming "I am not provable," and thus the system lies; if the system cannot prove $G$, then what $G$ says is exactly true, but the system cannot capture it.

:::details The formal skeleton of this game
- **State space**: formulas, proof sequences, and their Gödel numbers.
- **Legal actions**: encode symbol strings; express in the system "a certain number is a proof"; construct self-referential propositions.
- **Transition rules**: move from ordinary arithmetic propositions to arithmetic propositions about "provability."
- **Victory condition**: the Hilbertian dream — the system is both consistent and able to capture all arithmetic truths.
- **Failure mode**: appearance of $G \leftrightarrow \neg\mathsf{Prov}(\ulcorner G \urcorner)$; the system gets stuck by its own mirror image.
:::

What this game cuts into is not "some problems are too hard." Hard problems may still have answers that you simply haven't found yet. Gödel's danger runs deeper: when a system is strong enough to describe its own proof behavior, it generates certain true propositions that cannot be proved inside the system.

A formal system fails not because it is weak, but because it is strong enough to see itself — and thus cracks open.

::: info Professor Pallas's Cat comments
The cruelty of the mirror proposition game lies in this: the mirror is not an external enemy. It is part of the system's expressive power. You want the system to be strong enough to talk about arithmetic; it conveniently learns to talk about itself. Then, the crack appears. It is not that someone attacked the castle; the castle grew its own reflection.
:::

## 15.1 Consistency: The System Does Not Lie

First, let's say precisely what "lying" means.

If a formal system can simultaneously prove some proposition $A$ and its negation $\neg A$, the system is **inconsistent**. Once this happens, the disaster is total: from $A$ and $\neg A$, using standard inference rules, one can prove **any** proposition. A system that can prove all propositions has proved nothing at all.

Thus consistency is the bottom line for a formal system; a system below this line is useless.

::: info Professor Pallas's Cat comments
"Useless" is not an adjective; it is a diagnostic conclusion. Once inconsistent, the system can prove all propositions, including "1=2." This is not the system being "a little off"; it means the system has lost all meaning of proof — being able to prove everything is equivalent to having proved nothing. This is total failure, not a local fault.
:::

::: info "Any proposition is provable" — why is this a disaster?
This inference has a Latin name: *ex contradictione quodlibet* (from a contradiction, any conclusion follows). The proof process is very short: knowing $A$ is true, so $A \lor B$ is true (addition of disjunction); knowing $\neg A$ is true, from $\neg A$ and $A \lor B$ one obtains $B$ (disjunctive elimination). $B$ is any proposition. That is to say, once a contradiction exists inside the system, all propositions become "provable" — including "1 = 2" and its negation. The system does not become "partially wrong," but **totally fails**.
:::

The standard method for verifying consistency is to construct a **model** — an interpretation under which all axioms are true. Propositional logic is consistent because its axioms are all tautologies under truth tables, and inference rules preserve the property of being a tautology, so the system cannot derive a contradiction.

But for stronger systems — such as Peano Arithmetic — constructing a model becomes difficult, and relying on such methods to guarantee consistency becomes unreliable. The reason lies in the second theorem, to be revealed shortly.

---

## 15.2 Completeness: The System Does Not Miss Things

Completeness says: whatever holds semantically can be proved syntactically.

More precisely, if $\vDash A$ ($A$ is a tautology, true under all interpretations), then $\mathcal{F} \vdash A$ ($A$ has a formal proof in the system $\mathcal{F}$).

Gödel proved in 1930 that first-order predicate logic satisfies this property. This is an encouraging result: the inference rules of first-order logic are exactly "correct," missing no inferences that should semantically hold.

::: info The significance of the completeness theorem
What this theorem says is: the inference rules we wrote down — modus ponens, universal instantiation, existential introduction… — exactly capture all semantically valid inferences, not one more, not one less. This is not the result of design, but a non-trivial fact that requires proof. If completeness did not hold, it would mean that some "obviously valid" inference was missed by the rule system and required new inference rules to be added — and Gödel told you: not necessary.
:::

This was the last piece of good news Gödel brought.

::: info Professor Pallas's Cat comments
"The last piece of good news" — this sentence is worth pausing on. The completeness theorem of 1930 and the incompleteness theorems of 1931 were brought by the same person in adjacent years, in completely opposite directions. First-order logic is complete; sufficiently strong arithmetic systems are incomplete. The boundary is right here, very precise, not some vague pessimism.
:::

One year later, he brought something completely different. The problem lies in those three words "sufficiently strong." Once the system is powerful enough to express arithmetic — natural numbers, addition, multiplication — completeness is permanently lost.

---

## 15.3 Gödel's Two Theorems

In 1931, Gödel published "On Formally Undecidable Propositions of Principia Mathematica and Related Systems." The paper directly gave the conclusions at the beginning, without any warm-up:

**Theorem One (First Incompleteness Theorem)**: Any consistent formal system that is sufficiently strong contains propositions that can be neither proved nor refuted.

**Theorem Two (Second Incompleteness Theorem)**: Any consistent formal system that is sufficiently strong cannot prove its own consistency within itself.

The "sufficiently strong" in these two theorems is a technical condition: the system can express elementary arithmetic. Peano Arithmetic satisfies it, ZFC set theory satisfies it, and nearly all systems used in actual mathematics satisfy it.

Let us trace how Gödel did it.

---

## 15.4 Gödel Numbering: Letting the System See Itself

Gödel's key insight can be stated in one sentence: **syntax is also a mathematical object.**

The symbols, formulas, and proofs in a formal system are all finite strings. Finite strings can be encoded as natural numbers — assign a number to each symbol, encode formulas as some arithmetic combination of these numbers. This encoding is called **Gödel numbering**; the number of proposition $A$ is written $\ulcorner A \urcorner$.

:::details Gödel Numbering: turning "language" into "numbers" (prior work: Gödel, 1931)
**Gödel numbering** is the core technique of Gödel's 1931 paper: assign a natural number to each symbol of the formal language, then encode any formula or proof sequence as a large natural number.

Simplified version: suppose $\neg$ is numbered 1, $\land$ is numbered 2, $P$ is numbered 3, $Q$ is numbered 4. Then the proposition $\neg P$ can be encoded as the number 13 (by concatenation), or more precisely as $2^1 \cdot 3^3 = 2 \cdot 27 = 54$ (using products of prime powers).

**Why is this technique so powerful?** Because once syntax is encoded as numbers, "is this string of symbols a legal formula?" or "is this sequence a legal proof?" — these **questions about language** become **questions about numbers**. And Peano Arithmetic can talk about numbers — thus the system acquires the ability to talk about its own syntax.

This is not cheating; it is a precise mathematical construction. The numbering scheme can vary, but the fact that "provability can be expressed by an arithmetic predicate" does not change.
:::


Once syntax is encoded as natural numbers, "this proposition is provable in the system" becomes an arithmetic statement about natural numbers. And Peano Arithmetic exactly can express statements about natural numbers. Thus the system acquires a strange ability: **it can talk about itself**.

::: info Is encoding a form of cheating?
This is a reasonable doubt. Is Gödel numbering merely a technical trick, substantively meaningless? No. The key to the encoding is not the numbering itself, but this: **the question of "which symbol strings are legal proofs" can be expressed by an arithmetic predicate** — this is a verifiable mathematical fact that does not depend on the specific numbering scheme. In other words, any sufficiently strong arithmetic system can use its own language to talk about "provability," and not just about numbers. Gödel merely exploited this possibility to its limit.
:::

This is the entry point for self-reference.

Returning to the mirror proposition game, this step is equivalent to turning "formulas" into game pieces that the system can hold in its hands and manipulate. The axiom stamp game of Chapter 14 only permitted you to write formulas on paper; now, the formulas on paper have numbers, and numbers can return to the paper to be talked about by another formula. The state space of the game suddenly gains one more layer: it is no longer just "what a proposition is," but "how a proposition talks about propositions."

This is the source of danger. Once the mirror is installed, the system no longer only looks outward at the world, but begins to look inward at its own proof behavior. Gödel's sentence is not a monster descending from the sky; it is a move naturally permitted by the mirror rules.

---

## 15.5 Gödel's Sentence: The Proposition That Says "I Am Unprovable"

Using Gödel numbering, one can express inside the system a predicate $\mathsf{Prov}(n)$, meaning "the proposition numbered $n$ is provable in this system." This predicate is itself an arithmetic proposition, completely legal.

The task now is: construct a proposition $G$ that is logically equivalent to $\neg\,\mathsf{Prov}(\ulcorner G \urcorner)$, i.e., "I myself am unprovable in this system."

This looks like a circular definition — the content of $G$ involves $G$'s own number, and $G$'s number depends on what $G$ is. How can one first write $G$, and then talk about $G$'s number?

**The Diagonal Lemma** is precisely what resolves this knot.

:::details The Diagonal Lemma: How is self-reference legally achieved in mathematics?
Diagonalization is the core technique used by Gödel, Turing, and others to construct self-referential propositions; the name comes from Cantor's diagonal argument.

Intuitive version: suppose you have an infinitely large table, rows and columns both natural numbers. You want to construct an object that "is not in any row of the table" — the method: take the element at row 1, column 1 and modify it; take the element at row 2, column 2 and modify it; and so on, modifying along the diagonal. This new object differs from every row of the table in at least one position, and thus is not in the table.

**In Gödel's proof**: apply this idea to "a formula referencing itself" — construct a formula $D(y)$ such that when $y$ is the Gödel number of $D$ itself, $D(y)$ talks about $D(y)$ itself. Through the $\mathsf{sub}$ substitution function, this cycle can be "unfolded" into arithmetic operations without any circularity.

Result: $A \equiv P(\ulcorner A \urcorner)$ — proposition $A$ talks about whether it itself has property $P$, completely legal, no circularity.
:::


First consider a unary predicate $P(x)$, where $x$ is a natural number (i.e., the Gödel number of some proposition). We want to find a proposition $A$ such that $A$ is equivalent to $P(\ulcorner A \urcorner)$ — that is, $A$ talks about "the proposition $A$ itself satisfies property $P$."

The construction proceeds in two steps.

**Step one: introduce a substitution function.** Define an arithmetic function $\mathsf{sub}(m, n)$, meaning "the Gödel number of the formula obtained by substituting the free variable in the formula numbered $m$ with the numeral $n$." This is a purely arithmetic operation — input two numbers, output one number — that Peano Arithmetic can fully express.

**Step two: write the self-referential formula.** Take a predicate $P(x)$, and consider the following formula containing the free variable $y$:

$$D(y) \;\equiv\; P(\mathsf{sub}(y,\, y))$$

$D(y)$ says: "substitute the free variable in the formula numbered $y$ with $y$ itself; the resulting proposition satisfies $P$."

Now compute the Gödel number of $D$ itself, written $d = \ulcorner D \urcorner$. Substitute $d$ into $D(y)$, obtaining the proposition:

$$A \;\equiv\; D(d) \;\equiv\; P(\mathsf{sub}(d,\, d))$$

Key observation: $\mathsf{sub}(d, d)$ is "the number of the formula obtained by substituting the free variable in $D(y)$ with $d$" — and that formula is exactly $D(d)$, i.e., $A$ itself. So $\mathsf{sub}(d,d) = \ulcorner A \urcorner$, hence:

$$A \;\equiv\; P(\ulcorner A \urcorner)$$

$A$ talks about whether it itself satisfies property $P$, without circularity, without presupposing $A$'s number — the number is naturally produced through the $\mathsf{sub}$ function in the construction process.

Taking $P(x)$ to be $\neg\,\mathsf{Prov}(x)$, the resulting $A$ is Gödel's sentence $G$:

$$G \;\equiv\; \neg\,\mathsf{Prov}(\ulcorner G \urcorner)$$

$G$ means: "The proposition numbered $\ulcorner G \urcorner$ — that is, myself — is unprovable in this system."

$G$ is not a metaphor, not a philosophical posture, but a clearly written arithmetic proposition within Peano Arithmetic, whose content exactly describes its own provability status.

Now let us look at the fate of $G$. Consider two cases:

**If $G$ is provable**: then $\mathsf{Prov}(\ulcorner G \urcorner)$ is true, i.e., $\neg G$ is true, i.e., the system simultaneously proves $G$ and $\neg G$. The system is inconsistent.

**If $G$ is not provable**: then $\neg\,\mathsf{Prov}(\ulcorner G \urcorner)$ is true, i.e., $G$ itself is true — it is indeed unprovable in the system, just as it says. But the system cannot prove this.

Therefore, under the premise that the system is **consistent**, $G$ is unprovable. And $G$ is actually true — we can see this from outside the system, but the system internally can never reach it.

What about $\neg G$? If $\neg G$ is provable, then the system claims $G$ is provable, but we already know $G$ is unprovable — the system is lying about its own provability, which likewise destroys consistency.

Conclusion: in a consistent system, $G$ is neither provable nor refutable. It is an **undecidable** proposition — true, but beyond the inferential reach of the system.

::: info What does "seeing from outside the system" mean?
This is the most subtle step in the entire argument. When we say "$G$ is actually true," we are relying on reasoning at the **meta-level**: we have argued, **outside** the system, that the system is consistent, from which it follows that $G$ is unprovable, from which it follows that $G$ is true. But this meta-level argument is itself carried out within some stronger system. Gödel did not find "absolute truth"; he only proved: for any sufficiently strong system, there always exist truths that require climbing out of this system to be seen. The levels are real, and there is no top.
:::

---

## 15.6 The Second Theorem: The System Cannot Vouch for Itself

The second incompleteness theorem is a corollary of the first theorem, but its impact is no less than the first.

Take the statement "the system $\mathcal{F}$ is consistent" and encode it via Gödel numbering as an arithmetic proposition, written $\mathsf{Con}(\mathcal{F})$.

The proof process of the first theorem can itself be formalized inside the system — because the proof process is a finite symbol manipulation, and the system can express arithmetic. After formalization, one obtains:

$$\mathcal{F} \vdash \mathsf{Con}(\mathcal{F}) \to \neg\,\mathsf{Prov}(\ulcorner G \urcorner)$$

That is, the system can itself prove: "**If I am consistent, then $G$ is unprovable.**"

Now suppose the system can also prove its own consistency, i.e., $\mathcal{F} \vdash \mathsf{Con}(\mathcal{F})$. Combining these two via modus ponens, the system can prove that $G$ is unprovable — but "$G$ is unprovable" is equivalent to $G$ itself, so the system proves $G$, contradicting the first theorem.

Therefore, the system **cannot** prove its own consistency.

This is not modesty, nor a lack of ability. It is a structural fact of logic: **you cannot use a ruler to measure whether the ruler itself is accurate**. Any proof of a system's consistency must come from a stronger external system, and the consistency of that stronger system is again an open question — this hierarchy has no top.

::: info Professor Pallas's Cat comments
What is truly shocking is that the second theorem says not "very hard" but "in principle impossible" — not an engineering problem, but a logical structural problem. A system's proof of its own consistency forever requires a stronger external system to provide the guarantee. There is no top. This is the point, not the ruler metaphor.
:::

::: info Then how do we believe ZFC is consistent?
To be honest: we cannot prove it. The mathematical community accepts ZFC because it has been used for nearly a hundred years, producing a large body of mutually consistent results, with no contradictions appearing. This is inductive reasoning, not deductive proof. Gödel's second theorem tells us that this is also the best we can do. The foundation of mathematics is not a rock proven to be flawless, but a supporting structure that has been repeatedly stress-tested and has not yet cracked. This difference is more important than most math classes admit.
:::

---

## 15.7 The End of the Hilbert Program and Its Legacy

Hilbert's goal was to establish a complete, consistent, and decidable foundation for mathematics. The two incompleteness theorems show that:

- **Completeness**: in a sufficiently strong consistent system, impossible to achieve.
- **Self-guarantee of consistency**: impossible to complete inside the system.
- **Decidability**: Turing proved in 1936 that it is impossible; this is the story of Chapter 19.

Three goals, all wiped out.

::: info But Hilbert was not a failure
The failure of the Hilbert Program is one of the most valuable failures in the history of science. It is precisely because he set the goals precisely enough that Gödel and Turing could prove precisely that those goals are unreachable. If Hilbert had said "mathematics should have a good foundation," there would be nothing to refute, and nothing to learn. Precise questions can receive precise negative answers — and precise negative answers are more valuable than vague affirmative ones.
:::

But this is not the endpoint of tragedy, but the starting point of a more sober understanding. The failure of the Hilbert Program did not cause mathematics to collapse; rather, it let mathematicians know more precisely where the boundaries are. Inside the boundaries, formal systems remain the most reliable knowledge-construction tools in human history. Outside the boundaries, there are permanently open questions — the truth or falsity of some propositions forever depends on which axioms you choose to accept.

This sobriety is more valuable than blind optimism.

---

## 15.8 Hands-on: What It Feels Like When Arithmetic Is Formalized

This chapter has been talking about "sufficiently strong systems" — strong enough to express arithmetic. This is not an abstract description. Peano Arithmetic (PA) is such a system; it has five axioms about the natural numbers. Before entering "Unresolved," spend a few minutes actually encountering it.

**Peano Axioms (informal version):**

1. $0$ is a natural number.
2. Every natural number $n$ has a unique successor $S(n)$, which is also a natural number.
3. $0$ is not the successor of any natural number (i.e., there is no $n$ such that $S(n) = 0$).
4. Different natural numbers have different successors: if $S(m) = S(n)$, then $m = n$.
5. **Principle of mathematical induction**: if a property holds for $0$, and whenever it holds for an arbitrary $n$ it also holds for $S(n)$, then it holds for all natural numbers.

Within these five axioms, addition can be defined:

$$m + 0 = m \qquad m + S(n) = S(m + n)$$

This is not a convention; it is a definition — the entire behavior of addition is determined by these two recursive equations.

---

**Exercise: using this definition, derive $1 + 1 = 2$.**

First, establish notation: $1 = S(0)$, $2 = S(S(0))$.

$$1 + 1 = S(0) + S(0) = S(S(0) + 0) = S(S(0)) = 2$$

Which definition was used at each step? The first step is expanding notation, the second used the second definition of addition ($m + S(n) = S(m+n)$, with $m = S(0), n = 0$), the third used the first definition of addition ($m + 0 = m$), the fourth is expanding notation.

This is the formal proof of $1 + 1 = 2$. It depends on no intuition, relying only on two recursive equations and the Peano axioms.

---

Now feel what it means for this system to be "sufficiently strong": it can express all elementary statements about the natural numbers — "there exist infinitely many primes," "every even number is the sum of two primes (Goldbach's conjecture)," even "this system itself is consistent" — all can be written as formulas in the language of PA. It is precisely this expressive power that enables Gödel's diagonal lemma to function. A system can talk about itself because it can talk about all sufficiently complex number-theoretic statements — and statements about itself are a special case of number-theoretic statements.

---

## Unresolved

**$G$ is "true," but in what sense?** We see from outside the system that $G$ is true, but this "outside" depends on a stronger system, and that stronger system has its own $G$. Is "truth" an absolute concept, or is it always relative to some system?

**Are incompleteness and the halting problem the same thing?** Both Gödel and Turing established boundaries through self-referential constructions; the two are structurally highly similar. Recursive function theory gives a partial answer — they are indeed two sides of the same coin — but the full picture of this story will only unfold in Chapter 19.

**Can we escape incompleteness?** Adding $G$ as an axiom yields a stronger system, but the new system has its own Gödel sentence. Add that one too, and so on ad infinitum. Does this transfinite hierarchy have a limit? The answer is yes, but what the limit itself is, is a frontier problem of modern set theory.

**Which structural rule causes incompleteness?** Chapter 14 listed three structural rules running silently in the background: weakening, contraction, exchange. The existence of Gödel's theorem is rooted in the system being sufficiently strong — and "sufficiently strong" depends on assumptions being arbitrarily reusable. If we question this — if each assumption can be used only once — what does reasoning become? This is the starting point of Chapter 16.

---

## Exercises

**★ Warm-up**

The core of the diagonal lemma is the substitution function $\mathsf{sub}(m, n)$. Judge the following descriptions as true or false:

1. The inputs of $\mathsf{sub}(m, n)$ are two natural numbers, and the output is also a natural number. (True/False?)
2. $\mathsf{sub}(\ulcorner D \urcorner, \ulcorner D \urcorner)$ yields the Gödel number obtained by substituting the free variable in formula $D$ with formula $D$ itself. (True/False?)
3. The specific method of Gödel numbering (e.g., using products of prime powers or another encoding) affects the conclusion of the first incompleteness theorem. (True/False?)

(The purpose of these three questions is to confirm: Gödel numbering is an arithmetic operation, not dependent on the choice of encoding scheme.)

---

**★★ Derivation**

Take Gödel's sentence $G$ ("I am unprovable in the system $\mathcal{F}$") and add it as a new axiom to $\mathcal{F}$, obtaining $\mathcal{F}' = \mathcal{F} + \{G\}$.

1. Is $\mathcal{F}'$ still consistent? (Hint: what does it mean that $G$ is unprovable in $\mathcal{F}$? Will adding $G$ contradict other theorems of $\mathcal{F}$?)
2. Does $\mathcal{F}'$ have its own Gödel sentence $G'$? Is $G'$ the same proposition as $G$? Why not?
3. This process can be repeated indefinitely. If all such Gödel sentences are added, is the resulting limit system complete? Try to explain in natural language why the answer is negative — completeness cannot be escaped.

---

**★★★ Challenge**

The second incompleteness theorem says: $\mathcal{F}$ cannot prove $\mathsf{Con}(\mathcal{F})$ within itself. But we (in a stronger external system) can indeed argue that $\mathcal{F}$ is consistent — for instance, by constructing a model in which all axioms of $\mathcal{F}$ are true.

Try to write out, using the language of this chapter: what does this "external argument" itself depend on? Which stronger system's consistency does it use? And the consistency of that stronger system, in turn, depends on what?

This question has no endpoint, but try to write clearly the structure of infinite regress — expressed using a chain of symbols $\mathcal{F}$, $\mathcal{F}_1$, $\mathcal{F}_2$… What does the existence of this chain imply for the question "does mathematics have an ultimate foundation"?

No need to solve these questions — only need to think clearly about why each question is harder than it looks.

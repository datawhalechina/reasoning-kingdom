# Volume II Preface: Before Building on the Foundation

> Three things make intuition dangerous: it is too fast, too confident, too hard to question.

---

You don't need to read this chapter.

If you've already encountered formal proofs, read a bit of mathematical logic, and know the difference between propositions and predicates, you can start directly from Chapter 14.

But if you've never seriously thought about what a "proof" really is—not the kind of "from the problem statement we can derive…" reasoning from math class, but a sequence of symbols where every single step has an explicit source and can be mechanically checked—then spending twenty minutes here first will make every subsequent chapter clearer.

---

## Why Natural Language Reasoning Is Unreliable

Let's look at three examples first. They will make you uncomfortable. This is intentional.

---

**Example One: The Barber Paradox**

In a village there is a barber who shaves all villagers who do not shave themselves, and only shaves such villagers.

Question: Does the barber shave himself?

- If he shaves himself: according to the rule, he only shaves those who do not shave themselves, so he should not shave himself. Contradiction.
- If he does not shave himself: according to the rule, he shaves all those who do not shave themselves, so he should shave himself. Contradiction.

Both cases lead to contradiction. Such a barber cannot exist in this village.

But the problem is: this contradiction is derived from the assumption that "such a set exists." A rule described in natural language can, without your noticing, contain a self-contradictory assumption—and you won't discover it before you introduce it.

One purpose of formal systems is to make such contradictions visible at the syntactic level, rather than having them suddenly erupt after long chains of reasoning.

---

**Example Two: The Lottery Paradox**

The probability of a single lottery ticket winning is one in a million. So for any given ticket, rationally speaking, I believe it will not win.

Fine. Now for one million tickets, I believe of each one individually that it will not win.

But I know that one of them will win.

Thus I simultaneously believe "every ticket will not win" and "one ticket will win." Taken together, these two are contradictory.

What went wrong here?

"Reasonably believe" is not a predicate in formal logic; it has no mechanically checkable rules. The moment you use the word "reasonable," you have already left the formal system and entered a kind of inference that requires probability theory to handle. Chapter 17 is precisely about translating "reasonably believe" into something with precise rules.

---

**Example Three: The Intuition of the Axiom of Choice**

Given arbitrarily many non-empty sets, you can choose one element from each set and form a new set.

This sounds completely obvious—choose one, what's so hard about that?

But when there are infinitely many sets, and the elements within each set have no distinguishing features (no order, no labels, cannot be named), the act of "choosing" becomes fuzzy. What are you choosing? By what rule?

The Axiom of Choice accepts "such a choice always exists" as an axiom—not because it can be proved, but because without it, many important mathematical theorems cannot be established. Yet with it, you can derive conclusions that violently conflict with intuition, such as the Banach-Tarski theorem: a ball can be decomposed into finitely many pieces and reassembled into two balls each identical to the original.

The lesson here is: axioms are not "obvious truths," but the starting points you choose to accept. Different axiom sets give different mathematical universes, and the theorems in these universes may be entirely different, yet within each system they hold with flawless rigor.

---

These three examples point to the same thing: **intuitive reasoning in natural language is dangerous until it is made precise**. Not because our intuition is poor, but because natural language permits ambiguity, permits self-reference, permits hiding implicit assumptions in innocuous phrasing.

What Volume II aims to do is to build an unambiguous foundation for reasoning—not to constrain reasoning, but to make its boundaries visible.

---

## What Background You Need for This Book

Let me be honest: Volume II has certain mathematical logic prerequisites. What follows are specific requirements, not vague suggestions.

**What you need to know (without these, Chapters 14-15 will be very difficult):**

- **Truth tables**: Given assignments to propositional variables, be able to compute the truth values of $P \land Q$, $P \lor Q$, $P \to Q$, $\neg P$. In particular, the fact that $P \to Q$ is true when $P$ is false.
- **The concept of proof**: Have encountered at least once the process of "deriving a conclusion from assumptions," even if only the "Given… Prove…" format from high school geometry.
- **Functions and sets**: Know what a function is (a mapping from input to output), what a set is (a collection of elements), and be able to read notation like $f: A \to B$.
- **Natural numbers and induction**: Know the form of mathematical induction (holds for $n=0$, if it holds for $n$ then it holds for $n+1$, therefore holds for all $n$).

**If you don't have these yet, recommended reading (sorted by time cost):**

| Title | Time Required | Corresponding Chapters |
|------|---------|---------|
| Velleman, *How to Prove It*, Chapters 1-3 | ~10 hours | Language foundation for Chapters 14-15 |
| Smullyan, *What Is the Name of This Book?* (or any paradox introduction) | ~5 hours | Intuition for self-reference in Chapter 15 |
| Sipser, *Introduction to the Theory of Computation*, Chapter 1 | ~5 hours | Computational model language for Chapters 16, 19 |

The last three chapters (Chapters 17-18) require you to have encountered basic probability (conditional probability, law of total probability), but nothing beyond the high school level.

**What you don't need (don't be intimidated by these words):**

- No need to know Lean, Coq, or any theorem-proving assistant
- No need to have read any mathematical logic textbook
- No need to know how to program (Chapter 16 mentions Rust only as an analogy, not requiring Rust knowledge)
- No need to have read Volume I (but you'll have a richer feel if you have, especially Vol. I Ch. 7 and Ch. 13)

---

## The Structure of Volume II, and What Each Chapter Says

Volume II's eleven chapters have a logical main thread, but each chapter can also be read relatively independently.

```
ch14 Formal Systems        ← Foundation: rules of inference are made precise
ch15 Incompleteness Theorems ← Cracks in the foundation: the system cannot see its own entirety
ch16 Linear Logic          ← Questioning one rule: assumptions are not free
     ↓ Bridge: three-valued logic (Section 16.7)
ch17 Probabilistic Inference ← Truth-value expansion: from certainty to uncertainty
ch18 Causal Inference      ← A new verb of inference: intervention is not conditioning
ch19 Complexity Theory     ← The geometry of computation: reasoning has intrinsic cost
ch20 The Heuristic Contract ← Promises under cost: the precise meaning of "approximately correct"
ch21 Learning Theory       ← Inverse inference: deriving laws from observations
ch22 Self-Reference and Emergence ← What happens when a system reasons about itself
ch23 Reasoning System Stability ← Deriving Lyapunov functions from Yonglin convergence
ch24 Category Theory Perspective ← Explaining the structural necessity of reasoning convergence with category theory
```

If you can only read three chapters: read **ch14, ch15, ch19**. They form the most central thread of Volume II: reasoning is formalized, then encounters boundaries that cannot be crossed, and then those boundaries are precisely measured.

---

## How to Read This Book

Each chapter of Volume II has three kinds of material, used differently:

**Main text** is narrative, aimed at building intuition and motivation. You can read it quickly, and it's okay if some parts aren't immediately clear—keep going, and they'll become clear when you circle back.

**Thought exercises** are in three tiers (★/★★/★★★). ★ problems are hands-on exercises, and you must do them—they confirm that you truly understand the definitions, not just the words. ★★ problems take time; skip if you must. ★★★ problems have no standard answer; they just push you to the edge and let you take a look.

**Open Questions** are not unfinished—they are genuinely unresolved. These are open problems in current mathematics or theoretical computer science, or questions on which there is no philosophical consensus. The correct attitude toward them is: know they exist; do not pretend there are answers.

---

One last thing.

The first chapter of Volume II (Chapter 14) will spend a significant amount of space clarifying something you might feel "isn't that just obvious": what is a proof, what is a rule of inference, why do these need precise definitions.

Please don't skip it.

That feeling of "isn't that just obvious" is precisely what needs to be examined carefully.

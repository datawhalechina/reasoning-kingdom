# Volume 2 Introduction: Before Building on the Foundation

> Three things make intuition dangerous: it is too fast, too confident, and too hard to question.

---

You don't have to read this chapter.

If you have already met formal proofs, read a bit of mathematical logic, and know the difference between a proposition and a predicate, you can start straight at Chapter 14.

But if you have never seriously thought about what a "proof" actually is — not the "it follows from the conditions that…" kind of argument from math class, but a sequence of symbols where every step has a clear source and can be checked mechanically — then twenty minutes here will make every chapter that follows clearer.

---

## Why Reasoning in Natural Language Is Unreliable

Three examples first. They will make you uncomfortable. That is intentional.

---

**Example 1: The barber paradox**

In a village there lives a barber who shaves every villager who does not shave himself — and only those villagers.

Question: does the barber shave himself?

- If he shaves himself: by the rule, he shaves only people who don't shave themselves, so he should not shave himself. Contradiction.
- If he does not shave himself: by the rule, he shaves everyone who doesn't shave themselves, so he should shave himself. Contradiction.

Both cases lead to a contradiction. No such barber can exist in this village.

But here is the problem: this contradiction was derived from the assumption that "such a set exists". A rule stated in natural language can hide a self-contradictory assumption without anyone noticing — and you can't discover it until after you've let it in.

One of the purposes of a formal system is to make contradictions like this one visible at the syntactic level, instead of letting them explode suddenly after a long stretch of reasoning.

---

**Example 2: The lottery paradox**

A lottery ticket has a one-in-a-million chance of winning. So for any given ticket, rationally speaking, I believe it won't win.

Fine. Now take a million tickets — I believe, of each one, that it won't win.

But I know that one of them will win.

So I simultaneously believe "no ticket will win" and "some ticket will win". Together, these two are contradictory.

Where did it go wrong?

"Rationally believing" is not a predicate of formal logic; it has no rules that can be checked mechanically. The moment you use the word "reasonable", you have left the formal system and entered a kind of inference that only probability theory can handle. Chapter 17 is exactly about translating "rationally believe" into something with precise rules.

---

**Example 3: The intuition of the axiom of choice**

Given any collection of non-empty sets, you can pick one element from each set and assemble a new set.

This sounds completely obvious — just pick one, how hard can it be?

But when there are infinitely many sets, and the elements inside each set are utterly indistinguishable (no order, no labels, no way to name them), the act of "choosing" becomes blurry. What are you choosing? By what rule?

The axiom of choice accepts "such a choice always exists" as an axiom — not because it can be proved, but because without it, many important mathematical theorems fail. And with it, you get conclusions that clash violently with intuition, like the Banach–Tarski theorem: a ball can be cut into finitely many pieces and reassembled into two balls exactly the size of the original.

The lesson here: an axiom is not an "obvious truth"; it is a starting point you choose to accept. Different axiom sets give different mathematical universes — the theorems in these universes may differ completely, yet within each system they hold without a flaw.

---

These three examples point to one thing: **intuitive reasoning in natural language is dangerous until it has been made precise**. Not because our intuition is bad, but because natural language allows ambiguity, allows self-reference, and allows hidden assumptions to hide inside harmless wording.

What Volume 2 does is build an unambiguous foundation under reasoning — not to restrict reasoning, but to make its boundaries visible.

---

## What You Need to Read This Book

To be honest: Volume 2 has some demands on your background in mathematical logic. Here is a concrete list, not vague advice.

**What you need to know (without these, Chapters 14–15 will be rough going):**

- **Truth tables**: given an assignment to the propositional variables, work out the truth values of $P \land Q$, $P \lor Q$, $P \to Q$, $\neg P$. In particular, that $P \to Q$ is true when $P$ is false.
- **The idea of a proof**: you have gone through the process of "deriving a conclusion from assumptions" at least once, even if it was only the "Given… Prove…" of high-school geometry.
- **Functions and sets**: know what a function is (a mapping from inputs to outputs), what a set is (a collection of elements), and can read notation like $f: A \to B$.
- **Natural numbers and induction**: know the form of mathematical induction (it holds for $n=0$; if it holds for $n$ it holds for $n+1$; hence it holds for all $n$).

**If you don't have these yet, read these first (ordered by time cost):**

| Book | Time needed | Helps with |
|------|-------------|------------|
| Velleman, *How to Prove It*, Chapters 1–3 | about 10 hours | the language foundation for Chapters 14–15 |
| Smullyan, *What Is the Name of This Book?* (any intro book on paradoxes) | about 5 hours | the self-reference intuition for Chapter 15 |
| Sipser, *Introduction to the Theory of Computation*, Chapter 1 | about 5 hours | the computation-model language of Chapters 16 and 19 |

The next two chapters (Chapters 17–18) need basic probability (conditional probability, the law of total probability), but nothing beyond high-school level.

**What you don't need (don't let these words scare you):**

- No knowledge of Lean, Coq, or any theorem prover is needed
- No need to have read any textbook on mathematical logic
- No need to know programming (Rust is mentioned in Chapter 16 only as an analogy; no Rust required)
- No need to have read Volume 1 (it helps, though — especially Volume 1's Chapters 7 and 13)

---

## How Volume 2 Is Organized, and What Each Chapter Says

The eleven chapters of Volume 2 follow one logical line, but each can also be read fairly independently.

```
ch14 Formal systems                  ← the foundation: inference rules made precise
ch15 Incompleteness theorems         ← a crack in the foundation: a system cannot see all of itself
ch16 Linear logic                    ← questioning one rule: assumptions are not free
     ↓ bridge: three-valued logic (Section 16.7)
ch17 Probabilistic inference         ← truth expands: from certain to uncertain
ch18 Causal inference                ← a new verb: intervention is not conditioning
ch19 Complexity theory               ← the geometry of computation: reasoning has an intrinsic cost
ch20 Heuristic contracts             ← promises under cost: the exact meaning of "roughly right"
ch21 Learning theory                 ← inverse inference: inferring laws from observations
ch22 Self-reference and emergence    ← what happens when a system reasons about itself
ch23 The Yonglin Corollary           ← from the Hessian spectrum to the ordinary equation of intelligence and emergence
ch24 An Inquiry into the NP Problem: The LIT Theorem  ← −∞ carries back through the change of base as 0; 0 is not the resident 1_L
```

If you can read only three chapters: read Chapters 14, 15, 19. Together they form the most central line of Volume 2: reasoning gets formalized, then hits a boundary it cannot cross, and then that boundary is measured precisely.

---

## How to Read This Book

Every chapter of Volume 2 has three kinds of material, and each is used differently:

**The main text** is narrative. Its job is to build intuition and motivation. It can be read quickly; it's fine if some parts are not immediately clear — move forward, and they will become clear on the way back.

**The exercises** come in three levels (★/★★/★★★). ★ questions are hands-on; you must do them — they confirm that you really understand the definitions, not just read the words. ★★ questions take time; skipping them is fine. ★★★ questions have no standard answer; they merely push you to the boundary for one look.

**Open Questions** are not unfinished writing — they are genuinely open. These are open problems in current mathematics or theoretical computer science, or questions philosophy hasn't agreed on. The right attitude toward them: know that they exist, and don't pretend there are answers.

---

One last thing.

The first chapter of Volume 2 (Chapter 14) will spend a lot of pages getting one thing clear, a thing you might think "isn't it just like this?": what a proof is, what an inference rule is, and why these need precise definitions.

Please don't skip it.

That "isn't it just like this?" feeling is exactly the thing that needs careful inspection.

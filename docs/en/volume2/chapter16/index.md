# Chapter 16: Linear Logic and Resources — Each Assumption Used Only Once

> Classical logic assumes resources are unlimited. This is a lie, and an expensive one.

---

Chapter 15 left us with an unsettling conclusion: so long as a system is sufficiently strong, there must exist true propositions it cannot prove; the system itself also cannot guarantee its own consistency. This is Gödel's gift — or rather, his curse.

But what does "sufficiently strong" mean? Where does the strength lie?

A formal system is powerful partly because its assumptions can be arbitrarily reused. You know $P$, and you can use $P$ a hundred times — it does not wear out, does not get exhausted, forever sitting there waiting for you to summon it. This ability corresponds to a structural rule quietly listed in Chapter 14: **contraction**. What contraction says is: one copy of an assumption and two copies are equivalent in inference. The redundant can be folded.

This rule seems like common sense. But it is open to question.

In 1987, Jean-Yves Girard asked a simple question: what happens if you remove the contraction rule?

What happened went far beyond what he initially expected. What he obtained was not merely a weakened version of classical logic, but a completely new logic — **linear logic** — that precisely describes a world where resources are consumed.

---

## 16.0 The Disposable Kitchen Game: Each Resource Can Be Used Only Once

Classical logic is like a strange kitchen: you have one egg, yet you can fry a hundred omelets simultaneously. Because in classical logic, assumptions are not consumed. You use $A$ once, and $A$ is still there; use it again, and it is still there. This may be reasonable for "knowledge," but absurd for "resources."

So we switch kitchens.

On the table are several ingredient cards: egg $A$, flour $B$, butter $C$. Each recipe is a linear implication, for example

$$A \otimes B \multimap D$$

meaning: consume one copy of $A$ and one copy of $B$, and obtain one dish $D$. Once used, the ingredient card disappears from the table. You cannot duplicate eggs out of thin air, nor can you take back butter that has already been used. Unless a certain card has a special marker $!A$ in front of it, indicating it is not an ordinary resource but knowledge that can be repeatedly invoked.

This is the game-feel of linear logic: reasoning is no longer just "deriving truth from truth," but "completing transformations under resource conservation."

:::details The formal skeleton of this game
- **State space**: resource multisets, e.g., $\Gamma = \{A, A, B, !C\}$. The repeated appearance of $A$ means two real resources, not one symbol written twice.
- **Legal actions**: consume resources according to linear rules and generate new resources; only $!A$ can be copied or discarded.
- **Transition rules**: if the recipe $A \otimes B \multimap D$ exists, then the state $\Gamma, A, B$ can transition to $\Gamma, D$.
- **Victory condition**: construct the target resource $G$ using finite resources.
- **Failure mode**: the target seems derivable, but you secretly reused an already-consumed assumption along the way.
:::

This game forces the reader to see what classical logic hides: **reasoning sometimes has a physical cost**. In programs, memory is not infinite; in communication protocols, a message sent cannot simultaneously remain in place; in quantum systems, an unknown quantum state cannot be arbitrarily cloned.

Linear logic is not a bizarre variant of classical logic. It is reminding us: classical logic defaults to a ridiculously rich world, where resources can be freely copied. The real world is not this generous.

::: info Professor Pallas's Cat comments
The rules of the disposable kitchen game are simple: used means used. Many abstract logics, when they land in reality, collide with this sentence. You cannot use one five-dollar bill to buy two cups of coffee, nor can you use one pool of video memory to hold two mutually non-releasing models. Classical logic forgot this; linear logic handed the bill back.
:::

## 16.1 What Contraction Is Doing

Recall the form of the contraction rule:

$$\frac{\Gamma, A, A \vdash B}{\Gamma, A \vdash B}$$

Above the line: the context has two copies of $A$, and $B$ can be derived. Below the line: with only one copy of $A$, $B$ can also be derived.

What this rule says is: **you know $A$, and you can use it arbitrarily many times, without needing to "pay" one copy of $A$ for each use**.

In classical logic, this is perfectly reasonable. Knowledge is not material; you know "$1+1=2$," use it a hundred times, and it is still there, not diminished in the slightest.

But not all "premises" in the world are knowledge.

Consider this sentence: "I have one five-dollar bill, and I can buy one cup of coffee."

This is a completely legal propositional inference: having five dollars $\to$ can buy coffee.

But if you apply contraction to this proposition, you get: having one five-dollar bill, you can buy two cups of coffee. This is obviously wrong. That bill was used up.

Classical logic is helpless against this kind of error, because it does not distinguish at all between "knowledge that can be reused" and "resources that are consumed upon use." The starting point of linear logic is precisely: **these two kinds of things are fundamentally different, and require different symbols and different rules**.

::: info Professor Pallas's Cat comments
Classical logic handles knowledge; linear logic handles resources — this distinction is real, not fancy philosophy. When you write programs, memory runs out; quantum states cannot be cloned; messages in a protocol disappear once sent. Classical logic's modeling of these scenarios was wrong from the very beginning. Not approximately wrong, but fundamentally wrong.
:::

::: info Is logic describing the world, or regulating inference?

Here there is a philosophical divergence worth pausing to consider. Some would say: logic is not describing resource consumption; it is talking about truth values, and truth values are not "used up" or exhausted. This view is not wrong — classical logic is indeed about truth values.

But linear logic has chosen another stance: logic is regulating **the legality of inference**, and inference takes place in a world that has costs. Not every "known" can be freely reused. If you want to use a resource, you must pay for it. This is not a denial of classical logic, but its generalization — classical logic is the degenerate version of linear logic after adding the special assumption that "resources can be arbitrarily copied."
:::

---

## 16.2 The Symbol System of Linear Logic

Remove contraction, and the structure of logic changes. The most obvious change is: you can no longer casually "merge" two assumptions, because their counts begin to matter.

Linear logic introduces a set of connectives that run parallel to classical logic but have different meanings.

**Multiplicative connective $\otimes$ (tensor product)**: $A \otimes B$ means "simultaneously possess $A$ and $B$," and both of these resources genuinely exist, each occupying one copy. If you want to derive some conclusion from $A \otimes B$, you must simultaneously consume both resources $A$ and $B$. It corresponds to "parallel use" — two things happening at the same time.

**Additive connective $\&$ (with)**: $A \,\&\, B$ means "possess two options $A$ and $B$, and you can choose to use one of them." But you can only choose one — the resource's destiny is decided at the moment of selection. It corresponds to "choice" — only one of the two things happens.

**Multiplicative disjunction ⅋ (par)**: the dual connective of linear logic, corresponding to the negated form of multiplicative conjunction.

**Additive disjunction $\oplus$ (plus)**: $A \oplus B$ means "possess either $A$ or $B$, but don't know which one" — this is an expression of information scarcity, not a selectable abundance.

:::details The four connectives of linear logic: understood through "ordering food" (prior work: Girard, 1987)
The extra connectives in linear logic are clearest when analogized to ordering food at a restaurant:

| Symbol | Reading | Analogy | Meaning |
|------|------|------|------|
| $A \otimes B$ | tensor | Combo meal: burger **and** fries, both given | Simultaneously possess two resources; both must be consumed |
| $A \,\&\, B$ | with | Today's special: burger **or** salad, choose one, up to you | Possess the right to choose, but can only pick one |
| $A \oplus B$ | plus | Chef randomly gives you burger or salad; you don't know which | Possess one of them, but don't know which |
| $A \multimap B$ | linear implication | Use one $A$ to exchange for one $B$; $A$ is consumed | Resource transformation, not knowledge derivation |

**Why distinguish $\otimes$ and $\&$?** In classical logic, "having $A$ and $B$" has only one way to say it ($A \land B$), because resources can be copied, and "simultaneously possessing" and "having the option to choose" have no distinction. Linear logic removes copying; these two kinds of "possession" must be precisely distinguished.
:::


These four connectives form two sets of symmetric pairings; behind them lies a deep duality structure: multiplicative connectives concern the **concurrent existence** of resources, additive connectives concern the **selectability** of resources.

::: info Why so many connectives?

Classical logic needs only two basic connectives (e.g., $\neg$ and $\to$) to express everything. Linear logic needs four, which feels redundant. But the reason these four cannot be merged is that they are genuinely different in terms of resource meaning: simultaneously possessing vs. being able to choose — these two things are smoothed over in classical logic by the "contraction" rule — since knowledge can be infinitely copied, "simultaneously possessing" and "being able to choose" have no distinction. Once contraction is removed, this difference must be precisely distinguished, or else the logic will lose information.
:::

---

## 16.3 The Exclamation Mark: Resources That Can Be Reused

But if all assumptions can only be used once, we cannot talk about genuine knowledge — those facts that can indeed be repeatedly referenced. Linear logic solves this problem with a special symbol: the **exclamation mark** $!A$ (read as "of course $A$").

The meaning of $!A$ is: $A$ is an **infinitely available resource**. Possessing $!A$ means you can take out $A$ from it at any time, as many times as you like, without exhaustion.

Formally, $!A$ satisfies a special set of rules:

- **Weakening**: $!A \vdash \mathbf{1}$ ($!A$ can be discarded — you don't need to use it)
- **Contraction**: $!A \vdash !A \otimes !A$ ($!A$ can be copied — you can use it twice)
- **Promotion**: if $!A \vdash B$, then $!A \vdash !B$ (conclusions derived from infinite resources are also infinitely available)

With the exclamation mark, linear logic can restore the simulation of classical reasoning: if all assumptions carry $!$, the entire system degenerates into classical logic. The exclamation mark is the bridge between linear logic and classical logic — it precisely marks which assumptions can be arbitrarily used in the classical sense, and which are genuine disposable resources.

::: info Professor Pallas's Cat comments
Many people at this point will say "OK, $!$ just turns resources into knowledge." Hold on — have you realized: in linear logic, "can be reused" is a **privilege** that must be explicitly declared, not a default right? This subversion is more thorough than most people realize. In classical logic, every assumption carries an invisible $!$ by default, and you never knew it.
:::

---

## 16.4 The Meaning of Implication Has Changed

After removing contraction, implication $A \to B$ also splits into two meanings that need precise distinction.

Linear implication $A \multimap B$ (read as $A$ linearly implies $B$, or "A transforms into B") means: **consume exactly one copy of $A$, and produce one copy of $B$**.

This is not the transmission of knowledge, but the **transformation of resources**. Five dollars $\multimap$ one cup of coffee. This is an inference that linear logic can correctly express — after use, that five-dollar bill disappears, replaced by the coffee.

By contrast, $A \to B$ in classical logic does not consume $A$ — you know $A$, derive $B$, but $A$ is still there and can continue to be used. To express classical implication in linear logic, you need to write $!A \multimap B$: take one copy from the infinitely available $A$, produce one copy of $B$, while $!A$ itself suffers no harm.

This distinction reveals the essential difference between "knowledge" and "resource": knowledge is of the $!A$ type — not diminished after use; resources are of the bare $A$ type — consumed after use. Classical logic lacks this distinction because it assumes all premises are knowledge.

---

## 16.5 Three Real-World Correspondences

Linear logic is not a pure thought experiment of logicians. It precisely corresponds to three important realities.

**Memory management.** In program design, memory is a resource: allocate a block of memory, must release it after use, or else it leaks. Classical logic's intuition about memory is wrong — it assumes you "know" a block of memory and can reference it infinitely, but memory, after being freed, no longer exists. The core of linear type systems (such as Rust's ownership system) is precisely linear logic: each variable is owned exactly once; transferring ownership is consuming one copy of an assumption; borrowing is using a temporary view modified by $!$. Linear logic provides the formal semantics for Rust's ownership mechanism.

**Quantum computing.** Quantum mechanics has a famous prohibition: the **no-cloning theorem**. An unknown quantum state cannot be perfectly copied. In the language of logic, this is precisely "contraction is not allowed" — quantum bits are linear resources; you cannot produce two copies of $\psi$ from one $\psi$. The computational model of quantum computing naturally lives inside linear logic; quantum circuits are linear inference sequences.

**Concurrent protocols.** Communication protocols between two processes can be precisely encoded as linear logic propositions. The protocol's "one side sends, the other side receives" is exactly linear implication: resources transfer from one side to the other, uncopyable, undiscardable. Session Types are precisely the engineering realization of this correspondence; they allow compilers to verify protocol correctness at the type-checking stage — logic guarantees that the protocol will not deadlock and will not miss messages.

::: info An extension of the Curry-Howard correspondence

Chapter 14 mentioned that proofs and programs are structurally isomorphic (this correspondence will be formally unfolded in Chapter 22). Linear logic pushes this correspondence one step further: linear proofs correspond to **linear programs** — programs where each variable is used exactly once. This is not an accidental coincidence, but because linear logic was designed precisely to capture the intuition that "computation is a process of consuming resources." Girard initially discovered linear logic while analyzing the proof structure of System F (polymorphic λ-calculus) — he found traces of resources in the depths of logic.
:::

---

## 16.6 Two Kinds of Truth: Possession and Choice

Linear logic also prompts a more fundamental philosophical shift: **"truth" no longer has only one meaning**.

In classical logic, $A$ being true is $A$ being true — just this one situation.

In linear logic, "possessing $A$" can be two different things:

- **Tensor-style possession**: $A \otimes B$ — you simultaneously possess $A$ and $B$ as two independent resources.
- **Additive-style possession**: $A \,\&\, B$ — you possess two options and can decide which one to use.

These two kinds of "possession" have completely different values in resource economics. I have one apple and one pear — this is not equal to my having "the ability to choose apple or pear" — the former is two resources, the latter is one resource plus a right to choose.

Classical logic conflates these two situations, because at the level of truth values, "$A$ and $B$ are both true" is equivalent to "$A$ is true, or $B$ is true, and both are true anyway" — it is saying the same thing. But once resources have counts, these two things become sharply different.

This distinction will reappear in Chapter 17 in a completely unexpected way: when truth values themselves become a continuous interval $[0,1]$, the difference between "simultaneously possessing" and "being able to choose" becomes the difference between the multiplication rule of probability and two different modes of Bayesian updating.

---

## 16.7 Transition: Before Truth Values Become Continuous — Three-Valued Logic as Warm-up

Chapter 17 is going to do something radical: expand truth values from $\{0, 1\}$ to $[0, 1]$. Before that, let's do a small experiment: if truth values have three values instead of two, what happens?

The Polish logician Łukasiewicz proposed three-valued logic in 1920. The truth value set is $\{0, \frac{1}{2}, 1\}$, where $\frac{1}{2}$ means "uncertain." The rules for the connectives are as follows:

**Negation**: $\neg 0 = 1$, $\neg \frac{1}{2} = \frac{1}{2}$, $\neg 1 = 0$

**Conjunction**: $a \land b = \min(a, b)$

**Disjunction**: $a \lor b = \max(a, b)$

**Implication**: $a \to b = \min(1,\ 1 - a + b)$

(The rule for implication is the most counter-intuitive part of three-valued logic; look at it a few more times. When $a = 1, b = 0$, it yields $0$ — consistent with classical logic; when $a = \frac{1}{2}, b = 0$, it yields $\frac{1}{2}$ — "from uncertainty derive false; the conclusion is also uncertain.")

---

**Exercise:** Using the rules above, compute the values of the following propositions under all three truth-value assignment combinations. Let $P = \frac{1}{2}$ (uncertain), $Q = 0$ (false).

1. $P \to Q$
2. $\neg Q \to \neg P$ (the contrapositive)
3. $P \lor \neg P$ (the law of excluded middle)

The third question is the most interesting: in classical logic, $P \lor \neg P$ is a tautology, forever true. In three-valued logic, when $P = \frac{1}{2}$, $\neg P = \frac{1}{2}$, so $P \lor \neg P = \max(\frac{1}{2}, \frac{1}{2}) = \frac{1}{2}$.

**The law of excluded middle, in three-valued logic, is no longer a tautology.**

This is not a technical detail. It means: when you expand "truth values," you are implicitly modifying the inference rules you are willing to accept. Three-valued logic rejects the law of excluded middle; classical logic accepts it. This is not about which one is more "correct," but about two different choices corresponding to different understandings of the legality of inference.

Returning to the disposable kitchen game, what this chapter really changes is not the names of recipes, but the physical laws of the kitchen. The kitchen of classical logic defaults to ingredients being infinitely copyable; linear logic turns ingredients back into things that are consumed; three-valued logic further hints that even binary judgments like "cooked/not cooked" can be interrupted by a third state. Each change to the state space changes the legal actions.

What Chapter 17 is going to do is push this direction to the extreme: truth values are not three points, but a continuous interval over $[0, 1]$. At that point, what will the inference rules become?

---

## Unresolved

**What is "truth" in linear logic?** In classical logic, semantics is very clear: a proposition is either true or false; truth tables give everything. In linear logic, what is the "semantics" of resources? How does one define a model such that "$A$ is true" means some concrete resource possession? The answer to this question is phase semantics and coherence spaces, but they all point in the same direction: truth values are no longer $\{0,1\}$, but some richer structure.

**To what extent is "resource" a precise mathematical concept?** Linear logic gives us a formal framework, but "resource" itself remains an intuitive word. What is one copy of a resource? When is the copying of resources legal? These questions have different answers in different application domains, and linear logic provides a framework that can accommodate these different answers, not a single answer.

**What happens if we remove the exchange rule?** Chapter 14 listed three structural rules: weakening, contraction, exchange. Linear logic removes contraction (and sometimes also weakening). If we further remove the exchange rule — making the order of assumptions matter — we obtain **ordered logic**, which corresponds to resource use with temporal ordering, such as communication protocols where messages must be sent and received in order. This direction is still unfolding.

**If "truth" itself is a degree, how should inference rules change?** Linear logic, through resource counting, opens up one dimension of the $\{0,1\}$ truth values — assumptions have "copy count" distinctions, but each copy is still determinately true or false. But there is another direction: truth values are not "1 copy or 2 copies," but "0.3 or 0.7." This direction is no longer a problem of resources, but a problem of uncertainty. The generalization of inference rules under uncertainty is precisely the protagonist of Chapter 17.

---

## Exercises

**★ Warm-up**

Using the resource intuition of linear logic, judge whether the following inferences hold. No need to write formal proofs; just use resource examples like "five dollars buys coffee" to explain clearly why each is correct or wrong.

1. $A \otimes B \vdash A$ (simultaneously possessing $A$ and $B$, can one separately derive $A$?)
2. $A \,\&\, B \vdash A$ (possessing "the right to choose $A$ or $B$," can one derive $A$?)
3. $!A \vdash A$ (infinitely available $A$, can one take out one copy of $A$?)
4. $A \vdash !A$ (one copy of $A$, can it become infinitely available $A$?)

The contrast between 1 and 2 is the most important distinction in this chapter. 4 is the inference that directly fails once the contraction rule is removed.

---

**★★ Derivation**

Consider the following three inferences. Judge whether they hold in linear logic, and write out the reasons.

1. $A \multimap B,\ A \multimap B \vdash A \multimap B$ (two copies of the ability "$A$ becomes $B$," can they be merged into one?)
2. $!(A \multimap B),\ !A \vdash !B$ (infinitely available transformation rule, plus infinitely available input, can produce infinitely available output?)
3. $A \otimes B \multimap C \vdash A \multimap (B \multimap C)$ (consuming $A$ and $B$ to produce $C$, is it equivalent to first consuming $A$, then consuming $B$ to produce $C$?)

The first involves contraction. The second involves the promotion rule of the exclamation mark. The third is the linear logic version of currying — think about whether it genuinely holds in terms of resources.

---

**★★★ Challenge**

Rust's ownership system implements linear types: each value has exactly one owner; transferring ownership is consuming one copy of an assumption; borrowing corresponds to using a temporary view modified by $!$.

Try using the symbols of linear logic to write, for each of the following Rust operations, a corresponding linear logic inference formula:

- Moving ownership of variable `x` to function `f` (move semantics)
- Passing an immutable reference of variable `x` to function `g` (immutable borrow)
- Function returning a value (consuming input, producing output)

Complete precision is not required, but clearly state: which operations correspond to consumptive inference (linear implication $\multimap$), and which operations correspond to reusable views (exclamation mark $!$). The purpose of this exercise is to translate the intuitions of language design into the language of logic — in the process of translation, you will discover where correspondences hold and where they do not.

# Chapter 16: Linear Logic and Resources — Each Assumption Can Be Used Only Once

> Classical logic assumes, by default, that resources are unlimited. That's a lie — and an expensive one.

---

Chapter 15 left us with an unsettling conclusion: as long as a system is strong enough, there are always true propositions it can't prove, and the system itself can't guarantee its own consistency. That was Gödel's gift — or, if you prefer, his curse.

But what does "strong enough" actually mean? Where does the strength come from?

A formal system is powerful partly because its assumptions can be reused as much as you like. If you know $P$, you can use $P$ a hundred times — it doesn't wear out, it doesn't run out, it's always there waiting for you. This power corresponds to one of the structural rules quietly listed in Chapter 14: **contraction**. Contraction says: one copy of an assumption and two copies of an assumption are equivalent when it comes to inference. The extra copy can be folded away.

That rule looks like common sense. But it can be questioned.

In 1987, Jean-Yves Girard asked a simple question: what happens if you take away the contraction rule?

Far more happened than he expected at first. What he got wasn't just a weakened version of classical logic; it was a whole new logic — **linear logic** — one that describes a world in which resources get consumed.

---

## 16.0 The One-Shot Kitchen Game: Each Ingredient Can Be Used Only Once

Classical logic is like a strange kitchen: you have one egg, yet you can fry a hundred omelets at the same time. Because in classical logic, assumptions aren't consumed. You use $A$ once, and $A$ is still there; use it again, it's still there. That makes sense for "knowledge," but it's absurd for "resources."

So let's switch to a different kitchen.

On the table sit several ingredient cards: egg $A$, flour $B$, butter $C$. Each recipe is a linear implication, for example

$$A \otimes B \multimap D$$

which means: spend one $A$ and one $B$, and get a dish $D$. Once spent, the ingredient cards vanish from the table. You can't conjure an egg out of nothing, and you can't take back the butter you already used — unless one of the cards carries a special mark $!A$, meaning it isn't an ordinary resource but knowledge that can be called on again and again.

That's the game-like feel of linear logic: reasoning is no longer just "from true to true"; it's "completing a transformation within a resource budget."

:::details The formal skeleton of this game
- **State space**: a multiset of resources, e.g. $\Gamma = \{A, A, B, !C\}$. The repeated $A$ means two real resources, not one symbol written twice.
- **Legal moves**: consume resources and create new ones according to the linear rules; only $!A$ may be copied or dropped.
- **Transition rule**: if there is a recipe $A \otimes B \multimap D$, then the state $\Gamma, A, B$ can move to $\Gamma, D$.
- **Winning condition**: build the target resource $G$ out of finite resources.
- **Failure mode**: the target looks derivable, but somewhere along the way you silently reused an assumption that had already been spent.
:::

This game forces the reader to see what classical logic hides: **reasoning often carries a physical cost**. In a program, memory is not infinite; in a communication protocol, a message sent cannot also stay where it was; in a quantum system, an unknown quantum state cannot be arbitrarily cloned.

Linear logic isn't a bizarre variant of classical logic. It's a reminder: classical logic assumed a world obscenely rich in resources, where resources can be copied for free. The real world isn't that generous.

::: info Professor Manul comments
The rules of the one-shot kitchen game are simple: once used, it's used. Lots of abstract logic runs head-first into this sentence the moment it lands in reality. You can't buy two cups of coffee with one five-dollar bill, and you can't fit two models that never release their memory into one chunk of VRAM. Classical logic forgot about this; linear logic hands the bill back to you.
:::

## 16.0.5 Playing the Game in Code

The kitchen is small enough to run in a few lines. The program below keeps the table as an immutable tuple of cards, cooks what the recipes allow, and refuses — visibly — the moment you try to reuse a spent ingredient. That refusal is the whole game.

```python
#!/usr/bin/env python3
"""
one_shot_kitchen.py -- the One-Shot Kitchen Game of Chapter 16.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 16, Section 16.0
"The One-Shot Kitchen Game: Each Ingredient Can Be Used Only Once."
Classical logic is a strange kitchen: one egg, a hundred omelets at once,
because assumptions are never consumed. Linear logic is the kitchen where
an ingredient card vanishes the moment a recipe spends it. This program
runs that kitchen:
  * the table is a multiset of ingredient cards (an immutable sorted tuple),
  * a recipe consumes its inputs and produces one dish,
  * a card marked !C is knowledge, not a resource: it stays on the table,
  * the kitchen refuses to cook with a card that is already spent.
That refusal is the whole point. The failure mode of Section 16.0 is
"somewhere along the way you silently reused an assumption that had
already been spent." Here the reuse is not silent: it is a visible error.
Run:  python3 one_shot_kitchen.py
"""
# A recipe: (needed cards, dish produced, cards that are reusable knowledge)
# A x B -o D  : spend one egg and one flour, get one dish D.
# !C -o cup   : knowledge C can be called on again and again.
RECIPES = (
    (("A", "B"), "D", frozenset()),
    (("!C",), "cup", frozenset({"!C"})),
)

def cook(table, recipe):
    """Apply one recipe to the table if every needed card is present.
    Returns (new_table, dish), or None when a needed card is already
    spent. Pure: the old table is never touched.
    """
    needed, produced, reusable = recipe
    left = list(table)
    for card in needed:
        if card in reusable:
            continue                # knowledge is not consumed
        if card not in left:
            return None             # the card was spent earlier
        left.remove(card)
    left.append(produced)
    return tuple(sorted(left)), produced

def demo():
    print("=" * 70)
    print(" THE ONE-SHOT KITCHEN GAME -- resources that run out")
    print(" The Reasoning Kingdom, Volume 2, Chapter 16 (Sec. 16.0)")
    print("=" * 70)
    table = ("!C", "A", "B")
    print(f"\nTable at start: {table}")
    print("  recipes:  A x B -o D   (spend egg and flour, get dish D)")
    print("            !C -o cup   (!C is knowledge: never spent)")
    result = cook(table, RECIPES[0])
    assert result is not None
    table, dish = result
    print(f"\nCook the omelet:      {table}   (A and B are gone; {dish} appeared)")
    refused = cook(table, RECIPES[0])
    assert refused is None
    print("Cook it again:        refused.")
    print("  the egg card is spent. Classical logic would hand you the same")
    print("  egg back for free -- the contraction rule is a photocopier for")
    print("  assumptions. This kitchen does not own a photocopier.")
    table, cup = cook(table, RECIPES[1])
    assert cup == "cup"
    print(f"\nUse knowledge !C:     {table}")
    table, cup2 = cook(table, RECIPES[1])
    print(f"Use knowledge !C again: {table}")
    print("  !C never left the table. Knowledge can be used again and again;")
    print("  a resource cannot.")
    print("\nThe point of the game: reasoning often carries a physical cost.")
    print("  You cannot buy two cups of coffee with one five-dollar bill, and")
    print("  you cannot clone an unknown quantum state. Classical logic assumed")
    print("  a world obscenely rich in resources; the real world is not that")
    print("  generous. Linear logic hands the bill back to you.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

## 16.1 What Does the Contraction Rule Do?

Recall the form of the contraction rule:

$$\frac{\Gamma, A, A \vdash B}{\Gamma, A \vdash B}$$

Above the line: two copies of $A$ in the context, and you can derive $B$. Below the line: only one copy of $A$, and you can still derive $B$.

What this rule says is: **you know $A$, so you can use it as many times as you like, without "paying" one copy of $A$ for each use**.

In classical logic that's completely reasonable. Knowledge isn't matter — you know "$1 + 1 = 2$", use it a hundred times, and it's still there, not a bit smaller.

But not all "premises" in the world are knowledge.

Consider this sentence: "I have a five-dollar bill, and it buys a cup of coffee."

That's a perfectly legal propositional inference: have a five-dollar bill → can buy coffee.

But if you apply contraction to this proposition, you get: have a five-dollar bill, can buy two cups of coffee. Obviously wrong. The bill was used up.

Classical logic is helpless against this mistake, because it doesn't distinguish "knowledge that can be reused freely" from "resources that are consumed by use." The starting point of linear logic is exactly this: **the two are fundamentally different things and need different symbols and different rules**.

::: info Professor Manul comments
Classical logic handles knowledge; linear logic handles resources — and the distinction is real, not fancy philosophy. When you write a program, memory runs out; quantum states can't be cloned; in a protocol, a message sent is gone. Classical logic's model of these scenarios was wrong from the start. Not approximately wrong — fundamentally wrong.
:::

::: info Is logic describing the world, or governing inference?

There's a philosophical split here worth stopping over. Some people say: logic isn't describing resource consumption; it talks about truth values, and truth values aren't "used up" by being employed. That view isn't wrong — classical logic is indeed about truth values.

But linear logic takes another stance: logic governs **the legitimacy of inference**, and inference happens in a world with costs. Not every "known" fact can be reused for free. If you want to use a resource, you must pay for it. This isn't a denial of classical logic; it's a generalization — classical logic is the degenerate version of linear logic, arrived at by adding the special assumption "resources can be copied arbitrarily."
:::

---

## 16.2 The Symbol System of Linear Logic

Remove contraction, and the structure of logic changes. The most obvious change: you can no longer "merge" two assumptions at will, because their counts start to matter.

Linear logic introduces a set of connectives that run parallel to classical logic but mean something different.

**Multiplicative connective $\otimes$ (tensor)**: $A \otimes B$ means "you have both $A$ and $B$", and both resources are real, each occupying a slot. To draw any conclusion from $A \otimes B$, you must consume both $A$ and $B$. It corresponds to "using things in parallel" — the two things happen at the same time.

**Additive connective $\&$ (with)**: $A \,\&\, B$ means "you hold the option of $A$ and the option of $B$, and you may pick one to use." But you can pick only one — the resource's fate is decided at the moment you choose. It corresponds to "choosing" — of the two things, only one happens.

**Multiplicative disjunction ⅋ (par)**: the dual connective of linear logic, corresponding to the negated form of the multiplicative conjunction.

**Additive disjunction $\oplus$ (plus)**: $A \oplus B$ means "you have one of $A$ or $B$, but you have no idea which" — this expresses missing information, not an abundance of choices.

:::details The four connectives of linear logic, understood via ordering food (building on Girard, 1987)
The extra connectives of linear logic are easiest to grasp through a restaurant-ordering analogy:

| Symbol | Reads | Analogy | Meaning |
|------|------|------|------|
| $A \otimes B$ | tensor | Combo meal: burger **and** fries, you get both | Own both resources, must consume both |
| $A \,\&\, B$ | with | Today's special: burger **or** salad — your choice | You hold the choice, but can pick only one |
| $A \oplus B$ | plus | The chef hands you a burger or a salad at random; you don't know which | You get one of them, but don't know which |
| $A \multimap B$ | linear implication | Trade one $A$ for one $B$; the $A$ is consumed | Resource transformation, not knowledge derivation |

**Why distinguish $\otimes$ from $\&$?** In classical logic, "having $A$ and $B$" has only one phrasing ($A \land B$), because resources can be copied: "having both at once" and "being able to choose" are the same thing. Linear logic removed copying, so these two kinds of "having" must be told apart precisely.
:::

## 16.2.5 Hands-On: Writing a Proof in Linear Logic

Let's write a linear-logic proof in sequent calculus style. A sequent has the form $\Gamma \vdash \Delta$, read as: the resource set $\Gamma$ can be transformed into the resource set $\Delta$.

In linear sequent calculus, there is only one identity axiom:

$$\frac{}{A \vdash A}\ (\text{id})$$

But the key difference lies in **context management**: every inference rule states precisely what it consumes and what it produces. No weakening (you can't drop an assumption into thin air), no contraction (you can't copy an assumption).

Here is a complete proof example — proving $A \otimes B \multimap B \otimes A$ ("having both $A$ and $B$ at once" can transform into "having both $B$ and $A$ at once" — swapping the two isn't an axiom in linear logic; it's a theorem you can prove):

```
Proof tree:

                ————— (id)             ————— (id)
                A ⊢ A                 B ⊢ B
               ——————————— (⊗R)     ——————————— (⊗R)
               A, B ⊢ A ⊗ B          B, A ⊢ B ⊗ A
               ————————————— (⊗L, bundle A and B)
               A ⊗ B ⊢ A ⊗ B         ——————————————— (exchange? Not allowed!)
```

Wait — a problem just got exposed. In the standard linear sequent calculus, is exchange (reordering the context) an implicit structural rule, or an explicit one?

The answer: **it depends on which variant of linear logic you're using**. In classical linear logic (CLL), exchange is kept as a structural rule — you may freely reorder the resources in the context. In non-commutative linear logic (NLL), exchange is removed, and the left-right order of resources carries physical meaning (like temporal order).

In the version that keeps exchange, the proof is simple:

1. Start from $A \vdash A$ and $B \vdash B$.
2. Use $\otimes R$ to get $A, B \vdash A \otimes B$ and $B, A \vdash B \otimes A$.
3. Use exchange to reorder the second one: $A, B \vdash B \otimes A$.
4. Use $\multimap R$ to go from $A \otimes B \vdash A \otimes B$ and $A, B \vdash B \otimes A$… no wait — the more direct route is $\otimes L$:
5. To prove $A \otimes B \vdash B \otimes A$, suppose you have $A \otimes B$ (that is, $A$ and $B$ bundled together), use $\otimes L$ to split them open and recombine them. Finally use $\multimap R$.

In the version without exchange, $A \otimes B \vdash B \otimes A$ is **unprovable**! Because the positional information of resources has been baked into the logic — you can't "freely" rearrange the order of $A$ and $B$.

The point of this exercise isn't the numbering of axioms but the first-hand experience: **remove one structural rule, and the set of theorems of the whole logic changes**. That's not fine-tuning; that's a fundamental change.

::: info Professor Manul comments
In the version that keeps exchange, this proof takes only two steps; remove exchange, and it becomes unprovable. Most logic courses never let you see this contrast — they treat exchange like air; you breathe it without knowing the air exists. One of linear logic's jobs is to make the air visible.
:::

::: info Why so many connectives?

Classical logic needs only two basic connectives (say $\neg$ and $\to$) to express everything. Linear logic needs four, which feels redundant. But the four can't be merged, because they really are different in resource terms: having both at once versus being able to choose — classical logic flattened this difference with the contraction rule. Since knowledge can be copied without limit, "having both" and "being able to choose" don't differ. Once contraction is gone, the difference must be spelled out precisely, or the logic loses information.
:::

---

## 16.3 The Bang: Resources You Can Reuse

But if every assumption can be used only once, we can't talk about genuine knowledge — those facts you can legitimately cite again and again. Linear logic solves this with a special symbol: the bang $!A$ (read "of course $A$").

$!A$ means: $A$ is an **unlimited resource**. Owning a $!A$ means you can draw an $A$ out of it any time you like, as many times as you like, with nothing wrung out of it.

Formally, $!A$ obeys a special set of rules:

- **Weakening**: $!A \vdash \mathbf{1}$ ($!A$ can be discarded — you don't have to use it)
- **Contraction**: $!A \vdash !A \otimes !A$ ($!A$ can be copied — you can use it twice)
- **Promotion**: if $!A \vdash B$, then $!A \vdash !B$ (a conclusion drawn from an unlimited resource is itself unlimited)

With the bang in place, linear logic recovers its simulation of classical reasoning: if every assumption carries a $!$, the whole system degenerates into classical logic. The bang is the bridge between linear logic and classical logic — it marks out precisely which assumptions can be used freely in the classical sense, and which are truly one-shot resources.

::: info Professor Manul comments
Many people say at this point: "OK, so the bang turns resources into knowledge." Hold on — did you realize that in linear logic, "can be reused" is a **privilege** that must be declared explicitly, not a default right? This subversion is far more thorough than most people realize. In classical logic, every assumption carries an invisible $!$ by default — and you never knew it.
:::

---

## 16.4 Implication Now Means Something Different

Once contraction is gone, the implication $A \to B$ also splits into two meanings that must be told apart.

Linear implication $A \multimap B$ (read "$A$ linearly implies $B$", or "$A$ transforms into $B$") means: **consume exactly one $A$, produce one $B$**.

This isn't the passing along of knowledge — it's the **conversion of resources**. A five-dollar bill $\multimap$ a cup of coffee. This is exactly the kind of inference linear logic can express correctly — afterward, the five-dollar bill is gone, replaced by the coffee.

In contrast, classical logic's $A \to B$ doesn't consume $A$: you know $A$, you derive $B$, and $A$ is still there for later use. To express classical implication in linear logic, you'd write $!A \multimap B$: take one copy out of an unlimited $A$, produce a $B$, and the $!A$ itself emerges unharmed.

This distinction exposes the essential difference between "knowledge" and "resource": knowledge is of the $!A$ type — it doesn't shrink with use; resources are of the bare $A$ type — they're consumed by use. Classical logic makes no such distinction, because it assumes every premise is knowledge.

---

## 16.5 Three Real-World Correspondences

Linear logic isn't a pure thought experiment for logicians. It corresponds precisely to three important realities.

**Memory management.** In programming, memory is a resource: allocate a block, and you must free it when you're done, or it leaks. Classical logic's intuition about memory is wrong — it assumes you "know" a memory block and can reference it forever, but a freed block no longer exists. The core of linear type systems (like Rust's ownership system) is exactly linear logic: every variable is owned exactly once, handing over ownership means spending an assumption, and borrowing means using a $!$-decorated temporary view. Linear logic gives the formal semantics of Rust's ownership machinery.

**Quantum computing.** Quantum mechanics has a famous prohibition: the **no-cloning theorem**. An unknown quantum state cannot be perfectly copied. In the language of logic, this is precisely "contraction not allowed" — a qubit is a linear resource; you can't produce two copies of $\psi$ from one $\psi$. The computational model of quantum computing naturally lives inside linear logic; a quantum circuit is a sequence of linear inferences.

**Concurrent protocols.** A communication protocol between two processes can be encoded precisely as linear-logic propositions. The protocol's "one side sends, the other receives" is exactly a linear implication: the resource transfers from one side to the other — not copyable, not discardable. Session types are exactly the engineering realization of this correspondence, letting the compiler verify protocol correctness at type-check time — the logic guarantees the protocol won't deadlock, won't lose messages.

## 16.5.5 Hands-On: How Rust's Ownership System Mirrors Linear Logic

This is not a metaphor. The rules Rust's type checker enforces at compile time all have exact counterparts in linear logic. Here's a concrete Rust program to illustrate.

**Example 1: Move semantics = linear implication**

```rust
let x = String::from("Hello");  // x owns the string
let y = x;                      // ownership moves from x to y
// println!("{}", x);           // compile error! x has been "consumed"
// error[E0382]: use of moved value: `x`
```

The corresponding linear-logic proposition:

$$\text{String("Hello")} \multimap \text{String@y}$$

The resource is implicitly moved from x to y, and x is no longer usable. In linear logic, that's: consume one resource, produce one resource, and the original disappears.

**Example 2: Immutable borrow = a temporary view of $!A$**

```rust
let x = String::from("Hello");
let r1 = &x;  // immutable borrow — can read, cannot modify
let r2 = &x;  // multiple immutable borrows may coexist
println!("{} {}", r1, r2);  // ✓
println!("{}", x);          // ✓ x is still valid
```

The corresponding linear-logic proposition:

$$!A \vdash A \otimes A \otimes \cdots$$

$!A$ (the unlimited resource) allows multiple read-only views to be taken from it. When the borrow ends, x's ownership is unaffected.

**Example 3: Mutable borrow = exclusive access to a linear resource**

```rust
let mut x = String::from("Hello");
let r = &mut x;  // mutable borrow — exclusive access
r.push_str(", world");
// let r2 = &x;  // compile error! cannot hold a mutable and an immutable borrow at once
// error[E0502]: cannot borrow `x` as immutable because it is also borrowed as mutable
```

The corresponding linear logic: a mutable borrow temporarily "takes away" the exclusive modification right to the resource — unlike a move, the borrowed resource returns after the borrow ends; unlike an immutable borrow, a mutable borrow is exclusive, so two of them can't coexist. This is the borderland between linear logic and affine types (types where a resource is used at most once).

**Example 4: Function call = consuming an argument**

```rust
fn consume(s: String) {  // consumes the input String
    println!("{}", s);
    // s is dropped here
}
let x = String::from("Hello");
consume(x);               // x is moved into the function parameter
// println!("{}", x);     // compile error! x has been consumed
```

The corresponding linear-logic proposition:

$$\frac{\Gamma \vdash \text{String("Hello")} \multimap \mathbf{1}}{\Gamma, \text{String("Hello")} \vdash \mathbf{1}}$$

The function body consumes the resource and produces nothing ($\mathbf{1}$, which corresponds to Rust's `()` type); the resource is transferred at the call, and the caller no longer owns it.

**Summary of the four examples:**

| Rust operation | Linear logic | Explanation |
|-----------|---------|------|
| Move `let y = x` | $A \multimap B$ | Resource moves from one place to another |
| Immutable borrow `&x` | $!A \vdash A$ | Take a read-only view from an unlimited resource |
| Mutable borrow `&mut x` | Exclusive linear access | Temporarily transfer exclusive modification rights; return them when the borrow ends |
| Function call `f(x)` | $\multimap$ elimination | The resource is consumed by the function body |

This correspondence is no accident. Rust's core type system (ownership, borrowing, lifetimes) is built on linear types — the logic its type checker performs at compile time is exactly the inference rules of linear logic. The Rust compiler didn't "take inspiration from" linear logic; it **implements** linear logic.

::: info An extension of the Curry-Howard correspondence

Chapter 14 mentioned that proofs and programs are structurally isomorphic (that correspondence unfolds formally in Chapter 22). Linear logic carries that correspondence one step further: linear proofs correspond to **linear programs** — programs in which every variable is used exactly once. This is not an accidental coincidence; it's because linear logic was designed from the start to capture the intuition that "computation is a process that consumes resources." Girard originally discovered linear logic while analyzing the proof structure of System F (the polymorphic λ-calculus) — he found traces of resources at the depths of logic.
:::

---

## 16.6 Two Kinds of Truth: Possessing and Choosing

Linear logic also leads to a more fundamental philosophical shift: **"true" no longer has a single meaning**.

In classical logic, if $A$ is true, it's true — one case, done.

In linear logic, "having $A$" can be two different things:

- **Tensor-style possession**: $A \otimes B$ — you have both $A$ and $B$, two independent resources.
- **Additive-style possession**: $A \,\&\, B$ — you have two options and can pick which one to use.

These two kinds of "having" have completely different value in resource economics. Having an apple and a pear is not the same as having "the choice between apple and pear" — the former is two resources; the latter is one resource plus a right of choice.

Classical logic lumps them together, because at the level of truth values, "$A$ and $B$ are both true" is equivalent to "$A$ is true or $B$ is true, and both are true" — the same thing said differently. But once resources have a count, the two things are sharply different.

This distinction will reappear in Chapter 17 in a completely unexpected way: when truth value itself becomes a continuous interval $[0, 1]$, the difference between "having both" and "choosing one" turns into two different modes of probability — the multiplication rule and Bayesian updating.

---

## 16.7 Transition: Before Truth Values Go Continuous — Three-Valued Logic as a Warm-Up

Chapter 17 will do something radical: expand truth values from $\{0, 1\}$ to $[0, 1]$. Before that, let's run a small experiment: what if truth values took three values instead of two?

The Polish logician Łukasiewicz proposed three-valued logic in 1920. The truth-value set is $\{0, \frac{1}{2}, 1\}$, where $\frac{1}{2}$ means "undetermined." The connectives are:

**Negation**: $\neg 0 = 1$, $\neg \frac{1}{2} = \frac{1}{2}$, $\neg 1 = 0$

**Conjunction**: $a \land b = \min(a, b)$

**Disjunction**: $a \lor b = \max(a, b)$

**Implication**: $a \to b = \min(1, 1 - a + b)$

(The implication rule is the most counterintuitive piece of three-valued logic, so stare at it for a while. When $a = 1, b = 0$ you get $0$ — consistent with classical logic; when $a = \frac{1}{2}, b = 0$ you get $\frac{1}{2}$ — "deriving false from uncertainty leaves the conclusion uncertain.")

**Exercise:** Using the rules above, compute the value of each proposition below under this truth-value assignment: $P = \frac{1}{2}$ (undetermined), $Q = 0$ (false).

1. $P \to Q$
2. $\neg Q \to \neg P$ (contrapositive)
3. $P \lor \neg P$ (law of excluded middle)

The third one is the most interesting: in classical logic, $P \lor \neg P$ is a tautology, always true. In three-valued logic, when $P = \frac{1}{2}$, then $\neg P = \frac{1}{2}$, so $P \lor \neg P = \max(\frac{1}{2}, \frac{1}{2}) = \frac{1}{2}$.

**The law of excluded middle — in three-valued logic, no longer a tautology.**

This is not a technical detail. It means: when you expand "truth value," you are implicitly revising the inference rules you're willing to accept. Three-valued logic rejects excluded middle; classical logic accepts it. Neither is more "correct" — they're two different choices, two different notions of what makes an inference legitimate.

Back to the one-shot kitchen game: what really changed in this chapter isn't the recipe names — it's the kitchen's laws of physics. Classical logic's kitchen assumes ingredients can be copied endlessly; linear logic turns ingredients back into consumables; three-valued logic goes further, suggesting that even a binary judgment like "cooked or not cooked" can be interrupted by a third state. Every change to the state space changes the set of legal moves.

What Chapter 17 does is push this direction to the limit: truth values won't be three points, but a continuous interval $[0, 1]$. What will the inference rules become then?

---

## Open Questions

**What is "true" in linear logic?** In classical logic, the semantics is clear: a proposition is either true or false, and truth tables give you everything. In linear logic, what is the "semantics" of a resource? How do you define a model such that "$A$ is true" means some concrete resource possession? The answers are phase semantics and coherence spaces, but they point the same way: truth value is no longer $\{0, 1\}$ but some richer structure.

**To what extent is "resource" a precise mathematical concept?** Linear logic gives us a formal framework, but "resource" itself remains an intuitive word. What is one unit of a resource? When is copying a resource legal? These questions get different answers in different application domains; what linear logic provides is a framework that can hold those different answers — not one single answer.

**What happens if the exchange rule is removed too?** Chapter 14 listed three structural rules: weakening, contraction, exchange. Linear logic removes contraction (and sometimes weakening). Remove exchange as well — making the order of assumptions matter — and you get **ordered logic**, which corresponds to resource use with a temporal order, such as communication protocols where messages must be sent and received in sequence. This direction is still unfolding.

**If "true" itself is a matter of degree, how should inference rules change?** Linear logic stretched $\{0, 1\}$ truth values open along one dimension by counting resources — assumptions differ in "number of copies," but each copy is still a determinate truth or falsity. There's another direction, though: truth as "0.3 or 0.7" rather than "1 copy or 2." That direction is no longer about resources — it's about uncertainty. The generalization of inference rules under uncertainty is exactly what Chapter 17 is about.

---

## Exercises

**★ Warm-Up**

Using the resource intuition of linear logic, judge whether each of the following inferences holds. No formal proof required — just use a resource example like the "five-dollar bill buys coffee" one to say clearly why it's right or wrong.

1. $A \otimes B \vdash A$ (having both $A$ and $B$ — can you derive $A$ by itself?)
2. $A \,\&\, B \vdash A$ (holding "the choice between $A$ and $B$" — can you derive $A$?)
3. $!A \vdash A$ (unlimited $A$ — can you take one $A$ out of it?)
4. $A \vdash !A$ (one $A$ — can it become unlimited $A$?)

The contrast between items 1 and 2 is this chapter's most important distinction. Item 4 is an inference that fails outright once the contraction rule is removed.

---

**★★ Derivation**

Consider these three inferences. Judge whether each holds in linear logic, and write out your reason.

1. $A \multimap B,\ A \multimap B \vdash A \multimap B$ (two copies of the "A becomes B" ability — can they merge into one?)
2. $!(A \multimap B),\ !A \vdash !B$ (an unlimited transformation rule plus unlimited input — can it yield unlimited output?)
3. $A \otimes B \multimap C \vdash A \multimap (B \multimap C)$ (consuming $A$ and $B$ to produce $C$ — equivalent to consuming $A$ first, then consuming $B$ to produce $C$?)

Item 1 involves contraction. Item 2 involves the promotion rule for the bang. Item 3 is the linear-logic version of currying — think about whether it really holds in resource terms.

---

**★★★ Challenge**

Rust's ownership system implements linear types: every value has exactly one owner, transferring ownership means spending an assumption, and borrowing corresponds to using a $!$-decorated temporary view.

Try to express each of these Rust operations as a corresponding linear-logic inference:

- Moving the ownership of variable `x` to function `f` (move semantics)
- Passing an immutable reference to `x` to function `g` (immutable borrow)
- A function's return value (consumes input, produces output)

It doesn't have to be perfectly precise — but be clear about this: which operations correspond to consumptive inference (linear implication $\multimap$), and which correspond to reusable views (the bang $!$)? The purpose of this exercise is to translate the intuition of language design into the language of logic — and while translating, you'll discover which parts match up and which don't.

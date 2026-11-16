# Chapter 14: Formal Systems — A Foundation for Reasoning

> Before you can prove something, you have to say clearly what "proof" means.

---

The thirteen chapters of Volume 1 were one journey. We started from entropy, crossed the ruins of symbol systems, sidestepped the traps of vector spaces, and spent a long time at the boundary of complexity theory. All along, we kept talking about "reasoning" — but never actually defined it.

That was deliberate. Intuition has to come first; definitions follow.

But now it's time to pay the bill. Volume 2 has only one job: to say clearly what reasoning is. Not with metaphors, not with history, but with precise language — digging out the structure of reasoning and putting it under the light.

This needs a starting point. That starting point is called a **formal system**.

---

## 14.0 The Axiom-Stamp Game: What Does "Legally Derived" Mean?

Before we talk about "truth", let's play a colder game: **the axiom-stamp game**.

Three things sit on the table: a stack of blank paper, a box of stamps, and a sheet of rules. The stamps carve fixed formulas, like $P \to (Q \to P)$. The rules sheet lists a few allowed moves, for example: if the paper already has $P$ and $P \to Q$ on it, you may stamp out $Q$ on the next line.

When the game starts, there is nothing on the paper. You can do exactly two kinds of moves:

1. **Stamp an axiom**: instantiate an axiom template and write it down.
2. **Move one step by a rule**: from formulas already on the paper, write a new formula according to an inference rule.

Nothing else is allowed. You can't say "this is obviously true". You can't say "my gut says it should be like this". You can't bring the real world in to help. A "proof" in this game is just a string of legal stamp records; a "theorem" in this game is the formula appearing on the last line of some sheet.

:::details The formal skeleton of this game
- **State space**: all finite sequences of formulas $(A_1, A_2, \ldots, A_n)$.
- **Legal moves**: write down an axiom instance, or apply an inference rule to formulas already written.
- **Transition rule**: from sequence $s$ to $s' = s \mathbin{\|} A$, where $A$ must be legally produced by an axiom or a rule.
- **Win condition**: the target formula $G$ appears at the end of the sequence, i.e., $\mathcal{F} \vdash G$.
- **Fail mode**: you think $G$ is obviously true, but the rules don't let you write it. Formal systems don't care about your feelings.
:::

The point of this game: **legality comes before truth**. A formal system first decides which symbol strings can be written and which steps can be taken; whether those symbols actually describe the world is a question for "semantics" later.

This is the first cut of formalization. It cuts reasoning out of natural language and puts it in a game you can check. Winning the game doesn't mean you hold the truth; it means you proved one thing: under this set of rules, you did actually get there.

::: info Professor Pallas's Cat comments
A formal system is not a truth machine. It is a chessboard, a box of pieces, and a rulebook. The fact that you can reach a square only says the move was legal — it doesn't mean the universe owes you applause. Separating "legally derived" from "actually true" is the first lesson of Volume 2.
:::

## 14.0.5 Playing the Game in Code

The game is small enough to run for real. The program below checks proof lines against the rule sheet — stamping an axiom instance, or applying modus ponens to two earlier lines — and then tries a series of illegal moves so you can watch the checker refuse them. Every verdict printed is what the checker actually computed.

```python
#!/usr/bin/env python3
"""
axiom_stamp_game.py -- the Axiom-Stamp Game of Chapter 14, run for real.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 14, Section 14.0
"The Axiom-Stamp Game: What Does 'Legally Derived' Mean?"

The table holds blank paper, a box of stamps, and a rule sheet. A proof is
nothing but a string of legal stamp records. This program plays the game:
  * it CHECKS proof lines the way the rule sheet demands (stamp an axiom
    instance, or apply modus ponens to two earlier lines), and
  * it lets you try illegal moves, so you can watch them get rejected.

The system is Lukasiewicz's three axiom schemas plus modus ponens over the
atoms {P, Q} -- the same toy formal system Chapter 15's companion script
mirror_sentence_game.py runs its proof search on.

Honest scope: the checker only recognizes the three schemas by shape; it
does not prove anything about the system. That is the point of the game:
legality comes first, truth has no stamp.

Style: formulas are immutable tuples, every function is pure, no classes.
Run:  python3 axiom_stamp_game.py
"""
# ---------------------------------------------------------------------------
# Formulas: "P" / "Q", ("not", F), ("->", F, G)
# ---------------------------------------------------------------------------
def show(f):
    """Canonical parenthesized string of a formula."""
    if isinstance(f, str):
        return f
    if f[0] == "not":
        return "¬" + show(f[1])
    return "(" + show(f[1]) + "→" + show(f[2]) + ")"


def is_axiom(f):
    """Is f an instance of one of the three axiom schemas?

    A1:  X -> (Y -> X)
    A2:  (X -> (Y -> Z)) -> ((X -> Y) -> (X -> Z))
    A3:  (not X -> not Y) -> (Y -> X)
    """
    if not (isinstance(f, tuple) and f[0] == "→"):
        return False
    # A1:  X -> (Y -> X)
    if isinstance(f[2], tuple) and f[2][0] == "→" and f[2][2] == f[1]:
        return True
    # A3:  (not X -> not Y) -> (Y -> X)
    if isinstance(f[1], tuple) and f[1][0] == "→" \
            and isinstance(f[2], tuple) and f[2][0] == "→":
        nx, ny = f[1][1], f[1][2]
        if isinstance(nx, tuple) and nx[0] == "not" \
                and isinstance(ny, tuple) and ny[0] == "not" \
                and f[2][1] == ny[1] and f[2][2] == nx[1]:
            return True
        # A2:  (X -> (Y -> Z)) -> ((X -> Y) -> (X -> Z))
        x, yz = f[1][1], f[1][2]
        xy, xz = f[2][1], f[2][2]
        if isinstance(yz, tuple) and yz[0] == "→" \
                and isinstance(xy, tuple) and xy[0] == "→" and xy[1] == x \
                and isinstance(xz, tuple) and xz[0] == "→" and xz[1] == x \
                and xy[2] == yz[1] and xz[2] == yz[2]:
            return True
    return False


# ---------------------------------------------------------------------------
# The proof checker: the rule sheet, enforced line by line
# ---------------------------------------------------------------------------
def check_proof(lines):
    """Check a claimed proof.

    lines: list of (formula, justification), where justification is
      None        -> "I stamped an axiom"          (checked against the schemas)
      (i, j)      -> "modus ponens on lines i, j"  (1-based; line j must be
                                                    line_i -> formula)
      anything else -> no such move exists on the rule sheet.

    Returns a list of (formula, verdict, note) per line.
    """
    formulas = {}
    verdicts = []
    for k, (f, why) in enumerate(lines, start=1):
        if why is None:
            ok = is_axiom(f)
            note = "stamped axiom" if ok else "rejected: not an axiom instance"
        elif isinstance(why, tuple) and len(why) == 2:
            i, j = why
            a = formulas.get(i)
            b = formulas.get(j)
            ok = (
                b is not None
                and isinstance(b, tuple)
                and b[0] == "→"
                and b[1] == a
            )
            note = f"MP from lines {i},{j}" if ok else "rejected: illegal MP"
        else:
            ok = False
            note = "rejected: the rule sheet has no such move"
        formulas[k] = f
        verdicts.append((f, ok, note))
    return verdicts


# A legal five-line proof of P -> P (the stamp record, line by line).
P = "P"
LEGAL_PROOF = [
    (("→", P, ("→", P, P)), None),  # 1  A1
    (("→", P, ("→", ("→", P, P), P)), None),  # 2  A1
    (
        (
            "→",
            ("→", P, ("→", ("→", P, P), P)),
            ("→", ("→", P, ("→", P, P)), ("→", P, P)),
        ),
        None,  # 3  A2
    ),
    (("→", ("→", P, ("→", P, P)), ("→", P, P)), (2, 3)),  # 4  MP
    (("→", P, P), (1, 4)),  # 5  MP
]


def demo():
    print("=" * 70)
    print(" THE AXIOM-STAMP GAME -- the rule sheet, enforced for real")
    print(" The Reasoning Kingdom, Volume 2, Chapter 14 (Sec. 14.0)")
    print("=" * 70)
    print("\nA legal proof of P -> P, checked line by line:")
    print("  line  formula                                  verdict")
    for k, (f, ok, note) in enumerate(check_proof(LEGAL_PROOF), start=1):
        print(f"  {k:>4}  {show(f):<40} {note}")
    assert all(ok for _, ok, _ in check_proof(LEGAL_PROOF))
    print("\nNow the illegal moves -- watch the rule sheet refuse:")
    attempts = [
        (("→", "P", "Q"), None, "I stamp P -> Q"),
        (("→", "P", "P"), (1, 1), "MP of line 1 with itself"),
        (("→", "P", "P"), "it is obviously true", "the real world says so"),
        (("P",), None, "P is true, so stamp it"),
    ]
    for f, why, story in attempts:
        (_, ok, note) = check_proof([(f, why)])[0]
        verdict = "ACCEPTED" if ok else "REJECTED"
        print(f"  [{verdict:>8}]  {story:35} -> {note}")
        assert not ok
    print("\nThe point of the game: legality comes before truth.")
    print("  The checker never asked whether P -> Q is true. It asked only")
    print("  whether the rule sheet lets you write it. It does not.")
    print("  Formal systems do not care about your feelings -- they care")
    print("  about which moves are legal.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
```

## 14.1 The Problem of Ambiguity in Language

Consider this sentence: "If it rains, the ground gets wet. It rained. So the ground is wet."

This reasoning looks completely correct. But the question is: why is it correct?

You might say: because it agrees with common sense. But common sense is not mathematics. Or you might say: because logic forces it. But what does "logic forces it" mean? If we take this question seriously, we must find some account that doesn't rely on "common sense" and "obviously".

Natural language is ambiguous. "Or" can mean "at least one" in one context and "exactly one" in another. "If… then…" changes shape wildly in daily use — "If you pass the exam, I'll take you out to dinner" and "If this algorithm is correct, then its runtime is $O(n \log n)$" have completely different logical structures behind the same word "if".

The purpose of formalization is not to make language ugly; it is to **remove ambiguity**. We need a set of symbols where every symbol has exactly one meaning, given by a definition, independent of context and common sense.

::: info Professor Pallas's Cat comments
"Common sense is not mathematics" — well said. But don't imagine formalization means translating common sense into symbols. Formalization forces you to admit how much inside your "obviously"s you never actually thought through. Symbols are just the tool that strips you bare; they are not the answer itself. That's it.
:::

---

## 14.2 Propositions: The Raw Material of Reasoning

The smallest unit of reasoning is the **proposition** — a statement that can be asserted as true or false. "It is raining today" is a proposition. "Raining" is not, because it asserts nothing. "This sentence is false" looks like a proposition, but it triggers a self-reference paradox — let's set it aside; Chapter 15 will deal with it head-on.

Propositions are written with capital letters $P, Q, R, \ldots$. Complex propositions are built from simple ones with **logical connectives**:

| Symbol | Read as | Meaning |
|------|------|------|
| $\neg P$ | not $P$ | the negation of $P$ |
| $P \land Q$ | $P$ and $Q$ | both hold |
| $P \lor Q$ | $P$ or $Q$ | at least one holds |
| $P \to Q$ | if $P$ then $Q$ | when $P$ holds, $Q$ must hold |
| $P \leftrightarrow Q$ | $P$ iff $Q$ | both true together or both false together |

The meanings of these symbols come from precise rules, not from convention. $P \to Q$ does not mean "$P$ causes $Q$", and not "$P$ comes before $Q$ in time". It means only: **there is no case where $P$ is true and $Q$ is false**. This definition may be colder than your intuition — "if the moon is square, then 2 + 2 = 5" is a true proposition under it — but that coldness is exactly what makes mathematical reasoning possible.

::: info Why accept this counterintuitive definition?
Logical connectives are not describing causality; they are **constraining the legality of inference**. The question behind $P \to Q$ is: if you accept $P$, are you forced to accept $Q$? Only when $P$ is true and $Q$ is false is the answer no — so that is the single case to exclude. When the antecedent is false, the implication imposes no constraint at all, and is considered true. The price: "true proposition" no longer equals "meaningful proposition". Mathematicians find it worth paying.
:::

---

## 14.3 Formal Systems: Setting the Rules of the Game for Reasoning

With a language of propositions, the next step is to say clearly which inferences are **legal**. That is the job of a formal system.

A formal system $\mathcal{F}$ has four parts:

**Language** $\mathcal{L}$: specifies which symbol strings are legal propositions. Just as the grammar of Chinese specifies which strings of characters are legal sentences, the grammar of a formal language specifies which symbol strings are legal formulas.

**Axioms** $\mathcal{A}$: a batch of propositions accepted as true outright, needing no proof. For example, $P \to (Q \to P)$ in propositional logic — whatever $P$ and $Q$ are, this formula always holds. Axioms are the starting point of the entire system.

**Inference rules** $\mathcal{R}$: the legal steps that take existing propositions to new ones. The most basic is called **modus ponens**: if you know $P \to Q$ and you know $P$, you may conclude $Q$. Written in line form:

$$\frac{P \to Q \qquad P}{Q}$$

Premises sit above the line, conclusion below. This is not describing what is "true"; it is specifying which inference steps are **permitted**.

**Proof**: a finite sequence connecting axioms and inference rules. A proof is a list of propositions $A_1, A_2, \ldots, A_n$, where each $A_i$ is either an axiom or the result of applying some inference rule to earlier propositions. The last proposition $A_n$ is the **theorem** the proof establishes.

When we say $\mathcal{F} \vdash A$ — read as "$A$ is provable in $\mathcal{F}$" — we mean there exists such a proof sequence ending in $A$.

:::details ⊢ and ⊨: two look-alike symbols with completely different meanings
These two core symbols appear constantly in the rest of Volume 2; confusing them will make everything unreadable:

- **$\vdash$ (one line, the turnstile)**: **syntactic provability**. $\mathcal{F} \vdash A$ means "there exists a formal proof that derives $A$ step by step from the axioms by the rules". This is purely a result of symbol manipulation — nothing to do with "true or false".
- **$\vDash$ (two lines)**: **semantic truth**. $\vDash A$ means "$A$ is true under every possible interpretation" (a tautology). This is a statement about meaning.

Analogy: $\vdash$ is "the machine derived it by the rules"; $\vDash$ is "it actually holds in the real world".

**Soundness**: $\vdash A \Rightarrow \vDash A$ — whatever is provable is true.  
**Completeness**: $\vDash A \Rightarrow \vdash A$ — whatever is true is provable.

Neither direction holds automatically; both must be proved.
:::

---

## 14.4 Proofs Are Syntactic Objects

There is a key philosophical position here worth stopping to think about.

In a formal system, a proof is a **syntactic object** — a sequence of symbol strings that can be inspected mechanically. Verifying a proof does not require understanding what it "says". You only need to check each step legally applies some inference rule.

What does this mean? It means a machine — one that understands no mathematics, doesn't know what "true" means, and only matches symbols — can verify the correctness of any formal proof. Modern theorem provers (Lean, Coq, Isabelle) work exactly this way: you write the proof, the machine checks the symbols.

::: info Verification vs. Discovery
Notice the asymmetry. **Verifying** a proof is mechanical and can be done in polynomial time — just check, step by step, that every inference rule is legally applied. **Finding** a proof is another matter: no general algorithm guarantees finding one in every case (Chapter 15 will show this is impossible). This asymmetry runs through all of computability theory and is one of the roots of the puzzle behind "NP".
:::

This is not the end of mathematics; it is a deep feature of mathematics: **mathematical truth can be verified mechanically**. No reliance on authority, no trusting intuition — we only need to check symbol strings.

Of course, the process of thinking up a proof — finding the right inference path — is not mechanical, and perhaps never will be. But verification being mechanical is already deep enough.

::: info Professor Pallas's Cat comments
It's easy to get carried away here. "A machine can verify proofs" does not mean "a machine understands proofs". Verification is pattern-matching over symbol strings and has nothing to do with understanding. If you're going to celebrate mechanical verifiability as mathematics' glory, first get clear on what exactly you're celebrating. What gets verified is the form, not the meaning.
:::

---

## 14.5 Context: Bookkeeping for Assumptions

Real reasoning rarely starts from zero. Usually we say: "**Assume** $P$ holds, then we can derive…". Reasoning with assumptions needs a place to record them: the **context**, written $\Gamma$.

$\Gamma \vdash A$ reads: under the assumption set $\Gamma$, $A$ can be derived.

For instance, $\{P \to Q,\ P\} \vdash Q$ says: assuming "if $P$ then $Q$" and "$P$", you can derive $Q$.

The context looks like a bookkeeping device, but its behavior determines the character of the whole logical system. There are three basic rules about contexts, usually running silently in the background, so most people never notice they exist:

**Weakening**: if you can derive $A$ under assumptions $\Gamma$, you can still derive $A$ with more assumptions — extra assumptions do no harm. Very intuitive. But it says: the assumption set may contain things that are never used.

**Contraction**: the same assumption may be used any number of times. If you know $P$, you can use it once or a hundred times, with no wear and tear.

**Exchange**: the order of assumptions doesn't matter. $\{P, Q\}$ and $\{Q, P\}$ are the same.

In classical logic these three rules are taken for granted — infinite resources, order irrelevant. But Chapter 16 will take away contraction, and this one change will fundamentally alter the character of the whole logic: each assumption can be used exactly once, and reasoning becomes resource management.

:::details Weakening, contraction, exchange: they sound boring, but Chapter 16 will destroy them
These three **structural rules** are the hidden foundation of classical logic. Most logic textbooks never state them, because they are "obviously true":

- **Weakening**: extra premises do no harm. Knowing "it rains" and "my wallet is at home", you can derive "the ground is wet" — even though the wallet is useless.
- **Contraction**: the same premise may be reused endlessly. Knowing "I have five dollars", you can use that knowledge any number of times.
- **Exchange**: the order of premises makes no difference.

Why list them at all? Because Chapter 16's linear logic removes contraction — suddenly "five dollars can be used only once", and reasoning turns into resource management. These three rules exist precisely so that the contrast is sharp when Chapter 16 removes one of them.
:::

::: info Professor Pallas's Cat comments
Most people study logic for ten years and never realize they've been silently using these three rules all along. This is not extra knowledge; this is the ground under your reasoning. You don't even know what the ground is, and you build on it — and can't say why what you built stays standing. The absence of a name does not mean they don't exist.
:::

::: info Why do these three rules deserve names?
Because they are not "logic itself"; they are **meta-choices about logic** — choose them and you get classical logic; refuse one of them and you get a different logical system. Most math courses never mention these three rules, because in the classical context they are never violated. But it is exactly this "goes without saying" that hides logic's optionality. One of the things a formal system can do is make these hidden choices visible.
:::

---

## 14.6 Semantics: The Meaning Behind the Symbols

A formal system manipulates symbols at the syntactic level. But we usually want these symbols to "have meaning" — to correspond to some real world or mathematical structure.

Giving meaning to propositions is called **assignment**: mapping each propositional variable to truth or falsehood. Under a given assignment, the truth value of a compound proposition is decided recursively by the rules for the connectives — $P \land Q$ is true only when both $P$ and $Q$ are true, $P \to Q$ is true when $P$ is false or $Q$ is true, and so on.

If a proposition is true under every assignment, it is called a **tautology**, written $\vDash A$. This is a semantic notion — $A$ holds no matter what values the variables take.

Now there are two levels to keep apart:

- $\mathcal{F} \vdash A$: at the syntactic level, $A$ has a formal proof
- $\vDash A$: at the semantic level, $A$ is a tautology

The relationship between these two levels is the central question of formal logic:

**Soundness**: if $\mathcal{F} \vdash A$, then $\vDash A$. The system proves only true things; it doesn't lie.

**Completeness**: if $\vDash A$, then $\mathcal{F} \vdash A$. All true things are provable; nothing is missed.

Standard formal systems for propositional logic satisfy both. First-order predicate logic does too — Gödel proved it in 1930; that was the last good news he brought. One year later, he brought something completely different.

---

## 14.7 The Stronger the System, the Bigger the Price

Different formal systems have different expressive power. From weak to strong, roughly:

**Propositional logic**: handles only propositions and connectives — cannot talk about "all" and "exists". Simple enough to be mechanically decided — an algorithm exists that, for any proposition, decides in finitely many steps whether it is a tautology.

**First-order predicate logic**: adds the quantifiers $\forall$ (for all) and $\exists$ (there exists), and can express statements like "every natural number has a successor". Complete, but undecidable — no general algorithm decides whether a first-order proposition is provable.

**Peano arithmetic (PA)**: adds the axioms of arithmetic to first-order logic, strong enough to express nearly every statement of elementary number theory. This is the level where Gödel's bad news arrives.

**Set theory (ZFC)**: the common language of modern mathematics — almost every mathematical object can be encoded and studied within it.

There is a pattern to this hierarchy: the stronger the system, the greater its expressive power, and the weaker its ability to verify itself. Propositional logic is simple enough to exhaustively check itself; Peano arithmetic is strong enough that it cannot even prove its own consistency. Power has a price. That price is exactly the bill the next chapter will settle.

::: info Why does "stronger" bring a loss of self-verification?
Intuitively: a system strong enough can talk about itself, and once it can talk about itself it can construct paradoxical statements about itself. A weak system cannot express "my own propositions", so it never meets the problem. It's like language — the richer the expressive power, the easier self-referential paradoxes appear; a simple formal language is safe precisely because it cannot refer to itself. What Gödel discovered is not a defect of mathematics, but the intrinsic price of expressive power.
:::

---

## 14.8 Practice: Writing Your First Formal Proof

Before we go on, stop and do one thing: **write a real formal proof sequence**. Not "understand what it says" — actually write it out, marking where every line comes from.

It will make everything that follows different.

We'll take a minimal system for propositional logic with only two axioms (Łukasiewicz's simplified version, plenty for practice):

**Axiom schema L1**: $P \to (Q \to P)$

**Axiom schema L2**: $(P \to (Q \to R)) \to ((P \to Q) \to (P \to R))$

**The only inference rule**: modus ponens — from $P$ and $P \to Q$, derive $Q$.

An axiom schema means: substitute arbitrary propositional variables for $P, Q, R$, and whatever you get is an axiom. For instance $A \to (B \to A)$, $A \to ((A \to B) \to A)$, $(C \to C) \to (C \to C)$ — all of these are instances of L1.

**Goal**: prove $P \to P$ (a proposition implies itself).

It looks obvious — but in this system it is not an axiom; it must be derived from the axioms.

Here is a scaffold. Try filling in the source of each line (an axiom instance, or which two lines MP is applied to):

| Line | Proposition | Source |
|------|------|------|
| 1 | $P \to ((P \to P) \to P)$ | L1, with $Q$ set to $P \to P$ |
| 2 | $(P \to ((P \to P) \to P)) \to ((P \to (P \to P)) \to (P \to P))$ | ? |
| 3 | $(P \to (P \to P)) \to (P \to P)$ | ? (apply MP to lines 1 and 2) |
| 4 | $P \to (P \to P)$ | ? |
| 5 | $P \to P$ | ? (apply MP to lines 3 and 4) |

Line 2 is an instance of L2 — find what $P, Q, R$ must be to produce it. Line 4 is an instance of L1 — likewise find the corresponding $P, Q$.

Once you fill it in, you've completed a full formal proof: five lines, each with a clear source, depending on no intuition whatsoever — only symbol manipulation.

The meaning of this proof is not the conclusion $P \to P$; it's that you experienced firsthand what "a proof is a syntactic object" means — every step is a mechanical symbol check, with no "obviously" anywhere.

Back to the axiom-stamp game that opened this chapter: you've just stamped five seals with your own hands. What's truly worth remembering is not the last $P \to P$, but how the rules of the game rewrite "I think this is obvious" into "I can point out where every line comes from". And it is precisely here that the coldness of formal systems shows its democracy: anyone can check whether your stamps are legal; nobody has to believe in your genius, credentials, or intuition.

But for exactly the same reason, a sharper question surfaces: if every legal move is fixed in advance by the rule sheet, is the rule sheet itself reliable? Can it stamp out something false? Can it miss something true? These are the two walls the next chapter faces.

---

## Open Questions

**Can formalization capture all reasoning?** Mathematicians in daily work rely heavily on informal intuition — a construction "should" work, a proof "obviously" generalizes. Can these intuitions be formalized? Or does a formal system in essence miss some unspeakable dimension of reasoning?

**Is the choice of axioms objective?** Different axiom sets give different mathematical universes. The axiom of choice — the most controversial axiom in ZFC — is not "obviously" correct; mathematicians argued about it for a century. Formalization made the debate clearer but did not settle it.

**Is this machine reliable? Complete?** We built the formal system, described its language, axioms, and inference rules, and separated syntax from semantics. But one question we have not answered head-on: are the things it proves all true? Can it prove everything that is true? One answer will be reassuring; the other, shocking. Chapter 15 takes both head-on.

---

## Exercises

**★ Warm-up**

Which of the following symbol strings are legal propositional formulas, and which are not? No reasons needed — just judge.

1. $P \to Q \land R$
2. $\neg \neg P$
3. $\land P Q$
4. $(P \to Q) \to (\neg Q \to \neg P)$
5. $P \to$

(Hint: a legal propositional formula is assembled from propositional variables, connectives, and parentheses by the grammar rules. Connectives are binary or unary operators and cannot appear alone at the end.)

---

**★★ Derivation**

Which of the following three assertions hold? Explain clearly in natural language, and point out which concept from this chapter each assertion depends on (inference rules, weakening, contraction, or the deduction theorem).

1. If $\Gamma \vdash P \land Q$, then $\Gamma \vdash P$.
2. If $\Gamma \vdash P$ and $\Gamma \vdash Q$, then $\Gamma \vdash P \land Q$.
3. If $\Gamma, P \vdash Q$, then $\Gamma \vdash P \to Q$.

The third is called the **deduction theorem**. It says: deriving $Q$ under the assumption $P$ is equivalent to deriving "$P$ implies $Q$" without that assumption. The meaning of this rule: an assumption can be "absorbed" into the implication connective. Figure out whether this absorption happens at the syntactic level or the semantic level.

---

**★★★ Challenge**

The deduction theorem is a **meta-theorem** about formal systems, not an axiom or inference rule inside the system. Using the language introduced in this chapter ($\vdash$, the context $\Gamma$, inference rules), state precisely what distinguishes a "meta-theorem" from a "theorem inside the system".

Further: can the deduction theorem itself be proved within some formal system? Or must it be established outside the system, by induction on proof length using mathematical induction? How does the answer relate to "a system cannot prove its own consistency" in Chapter 15? No need to solve it — just state the question precisely in the language of this chapter.

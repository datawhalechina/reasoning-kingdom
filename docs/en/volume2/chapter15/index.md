# Chapter 15: Consistency and Completeness — The Two Walls of Formal Systems

> Can you build a system that never lies, and knows everything?

---

Chapter 14 gave us a precision machine: a formal system accepts axioms, runs according to inference rules, and outputs theorems. This machine seems very reliable. But "seems" is not good enough. We need to know how reliable it **actually** is.

This problem has two dimensions, in opposite directions.

The first dimension: Does the system lie? That is, are the things it proves all truly valid? This is called **consistency**.

The second dimension: Does the system miss things? That is, can it prove everything that is true? This is called **completeness**.

Both requirements are very reasonable. In the early 20th century, the mathematician David Hilbert ambitiously believed that one could build a formal foundation for all of mathematics that simultaneously satisfied both conditions — rigorous, complete, and unassailable. This plan was called the Hilbert Program.

In 1931, a 26-year-old Austrian mathematician, with two theorems, completely destroyed this program.

---

## 15.0 The Mirror-Sentence Game: When a System Starts Seeing Itself

The axiom stamp game of Chapter 14 was still relatively safe: you stamped, moved according to rules, and finally obtained theorems. Now we add a mirror to this game.

The rules change as follows: every formula, every proof, every string of symbols can be encoded as a natural number. Thus the system contains not only propositions that talk about numbers, like "$1+1=2$," but also propositions that talk about "whether a certain formula number is provable."

This is like a library suddenly acquiring its own catalog, and every book in the catalog can in turn reference the catalog itself. The game starts to become dangerous from here.

The player must take three steps:

1. **Numbering**: turn formulas into numbers, so that statements become objects operable inside the system.
2. **Referencing**: construct a predicate $\mathsf{Prov}(n)$, meaning "the formula numbered $n$ is provable."
3. **Mirroring**: produce a proposition $G$ that talks not about the outside world but about itself: $G$ is not provable.

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

::: info Professor Manul comments
The cruelty of the mirror-sentence game lies in this: the mirror is not an external enemy. It is part of the system's expressive power. You want the system to be strong enough to talk about arithmetic; it conveniently learns to talk about itself. Then, the crack appears. It is not that someone attacked the castle; the castle grew its own reflection.
:::

### 15.0.5 A Small Gödel Numbering: How Encoding Gives the System a Mirror

Gödel numbering is not the abstract "map symbols to numbers" — it is a mechanical encoding that can be demonstrated by hand. Let us perform it on a tiny system.

Consider the symbol set: $P$ (propositional variable), $\to$ (implication), $\neg$ (negation). Encode them as:
$$P \mapsto 1,\quad \to \ \mapsto 2,\quad \neg \mapsto 3$$

A formula (string of symbols) is encoded as the product of corresponding prime powers. The symbol sequence of the formula $\neg P \to \neg\neg P$ is $(\neg, P, \to, \neg, \neg, P)$:
$$\text{encoding} = 2^3 \cdot 3^1 \cdot 5^2 \cdot 7^3 \cdot 11^3 \cdot 13^1$$

**Key property**: prime factorization is unique — given this number, you can recover the original symbol string with 100% certainty. Each formula corresponds to a unique number, and each number corresponds to at most one formula.

A proof sequence (a list of formulas) is encoded one level further: $(F_1, F_2, \ldots, F_k)$ maps to $2^{\#F_1} \cdot 3^{\#F_2} \cdot \cdots \cdot p_k^{\#F_k}$.

Now the system can internally express $\mathsf{Prov}(n)$ — "there exists a proof sequence whose last formula has the number $n$." But $\mathsf{Prov}(n)$ is not some mysterious new predicate — it is just an arithmetic proposition about natural numbers, saying "there exists some natural number satisfying certain prime factorization conditions." It is entirely within the system's language — talking about multiplication, addition, primes — only through encoding, these arithmetic relations happen to correspond to the meta-mathematical concept of "a certain formula is provable."

**The power of this mirror**: the system originally could only talk about numeric propositions like $1+1=2$. With Gödel numbering, the system "conveniently" acquires the ability to talk about itself — because "proof" has been encoded as relations between numbers. It is not that the system was designed with this mirror; rather, when numbers become large enough to encode the system itself, the mirror emerges inevitably.

::: info Professor Manul comments
Gödel numbering is not an "encoding trick" — it demonstrates how self-reference arises as an inevitable byproduct of expressive power. Any system capable of talking about addition and multiplication automatically becomes capable of talking about certain facts about itself. Not because self-reference was "designed," but because the system's expressive power exceeds a certain threshold, and self-reference becomes an unavoidable mathematical fact.
:::

### 15.0.6 The Mirror-Sentence Game in Code

Section 15.0.5 numbered one formula by hand. The program below runs the whole game: it reproduces that numbering (and checks it against the exact number in the book), runs a real proof search in a small formal system, and then builds the mirror sentence G – "G is not provable" – and runs the fork on real computed values.

```python
#!/usr/bin/env python3
"""
mirror_sentence_game.py -- a runnable simulation of the Mirror-Sentence
Game (Volume 2, Chapter 15, Sec. 15.0 and 15.0.5).

The game has three steps, and this module runs all three:

  1. NUMBERING   -- encode symbol strings as natural numbers (Goedel
                    numbering), exactly the scheme of Sec. 15.0.5:
                    P->1, ->->2, not->3 (the machine adds Q->4 so the
                    toy system has two atoms).
  2. REFERENCING -- build a real toy formal system (Lukasiewicz's three
                    axiom schemas plus modus ponens), run an actual proof
                    search, and implement Prov(n).
  3. MIRRORING   -- diagonalize: build a sentence G whose content is
                    "G is not provable", then run the fork from Sec. 15.0
                    on real computed values.

Honest scope: the toy system is propositional logic -- sound, complete,
and decidable. No incompleteness fires inside it (its language has no
word for "provable"); the crack is derived in Part 3 at the level of the
machine's own code, where the fixed point genuinely exists. The full
theorem needs arithmetic; this shows the mechanism, not a faked theorem.
Proof searches are bounded; the caps below are generous for every formula
shown. Style: functional -- no classes, formulas are immutable tuples.

Run:  python3 mirror_sentence_game.py
"""
from __future__ import annotations

import functools
import itertools
import math
import string
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

# The book's scheme (P=1, ->=2, not=3) extended to Q and plain ASCII so
# Part 3 can quote sentences; real Goedel numbering covers every symbol
# of the formal language the same way.
CHARSET = "".join(
    dict.fromkeys(
        "P\u2192\u00acQ" + string.ascii_letters + string.digits
        + " (){}[]:;,'\"+-*/<>=!._%" + "\\" + "\n" + "\t"
    )
)
EXPONENT_OF: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(CHARSET)}

# ---------------------------------------------------------------------------
# Primes and unique factorization
# ---------------------------------------------------------------------------

@functools.cache
def prime(k: int) -> int:
    """The k-th prime, 1-indexed."""
    if k == 1:
        return 2
    n = prime(k - 1) + 1
    while not _is_prime(n):
        n += 1
    return n


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    limit = math.isqrt(n)
    i, p = 1, prime(1)
    while p <= limit:
        if n % p == 0:
            return False
        i += 1
        p = prime(i)
    return True


def factorize(n: int, i: int = 1) -> Tuple[Tuple[int, int], ...]:
    """Unique prime factorization of n >= 1, ascending primes."""
    if n <= 1:
        return ()
    p = prime(i)
    if n % p != 0:
        return factorize(n, i + 1)
    exponent, rest = 0, n
    while rest % p == 0:
        rest //= p
        exponent += 1
    return ((p, exponent),) + factorize(rest, i + 1)


# ---------------------------------------------------------------------------
# Part 1 -- NUMBERING
# ---------------------------------------------------------------------------

def encode_symbols(text: str) -> int:
    """Goedel-number a symbol string: product of prime(i)^EXPONENT_OF[sym_i].
    The empty string encodes to 1 (the empty product)."""
    return functools.reduce(
        lambda acc, pair: acc * prime(pair[0]) ** EXPONENT_OF[pair[1]],
        enumerate(text, start=1),
        1,
    )


def decode_number(n: int) -> Optional[str]:
    """Inverse of encode_symbols; None when n names no symbol string."""
    if n <= 0:
        return None
    try:
        symbols = [CHARSET[exponent - 1] for _, exponent in factorize(n)]
    except IndexError:
        return None
    return "".join(symbols)


def encode_proof(formulas: Tuple[str, ...]) -> int:
    """Second level: (F1,...,Fk) -> product of prime(i)^encode_symbols(Fi)."""
    return functools.reduce(
        lambda acc, pair: acc * prime(pair[0]) ** encode_symbols(pair[1]),
        enumerate(formulas, start=1),
        1,
    )


def decode_proof(n: int) -> Tuple[str, ...]:
    """Inverse of encode_proof."""
    return tuple(decode_number(exponent) for _, exponent in factorize(n))


# ---------------------------------------------------------------------------
# The toy formal language: formulas as immutable tuples
# Grammar: F := P | Q | not F | (F -> F);  not binds tightest, -> right-assoc.
# ---------------------------------------------------------------------------

def parse_formula(text: str):
    """Parse a symbol string into a formula tuple; ValueError if not well-formed."""
    formula, i = _parse_implication(text, 0)
    if i != len(text):
        raise ValueError(f"trailing symbols at position {i}: {text[i:]!r}")
    return formula


def _parse_implication(text: str, i: int):
    left, i = _parse_negation(text, i)
    if i < len(text) and text[i] == "\u2192":
        right, i = _parse_implication(text, i + 1)
        return ("\u2192", left, right), i
    return left, i


def _parse_negation(text: str, i: int):
    if i < len(text) and text[i] == "\u00ac":
        inner, i = _parse_negation(text, i + 1)
        return ("\u00ac", inner), i
    if i < len(text) and text[i] == "(":
        left, i = _parse_implication(text, i + 1)
        if i >= len(text) or text[i] != "\u2192":
            raise ValueError("expected '\u2192' inside parentheses")
        right, i = _parse_implication(text, i + 1)
        if i >= len(text) or text[i] != ")":
            raise ValueError("expected ')'")
        return ("\u2192", left, right), i + 1
    if i < len(text) and text[i] in ("P", "Q"):
        return text[i], i + 1
    raise ValueError(f"cannot parse at position {i}: {text[i:]!r}")


def show(formula) -> str:
    """Canonical, fully parenthesized symbol string of a formula tuple."""
    if isinstance(formula, str):
        return formula
    if formula[0] == "\u00ac":
        return "\u00ac" + show(formula[1])
    return "(" + show(formula[1]) + "\u2192" + show(formula[2]) + ")"


def complexity(formula) -> int:
    if isinstance(formula, str):
        return 0
    if formula[0] == "\u00ac":
        return 1 + complexity(formula[1])
    return 1 + complexity(formula[1]) + complexity(formula[2])


def variables(formula) -> FrozenSet[str]:
    if isinstance(formula, str):
        return frozenset({formula})
    if formula[0] == "\u00ac":
        return variables(formula[1])
    return variables(formula[1]) | variables(formula[2])


def subformulas(formula) -> FrozenSet:
    if isinstance(formula, str):
        return frozenset({formula})
    if formula[0] == "\u00ac":
        return subformulas(formula[1]) | {formula}
    return subformulas(formula[1]) | subformulas(formula[2]) | {formula}


def evaluate(formula, assignment: Dict[str, bool]) -> bool:
    if isinstance(formula, str):
        return assignment[formula]
    if formula[0] == "\u00ac":
        return not evaluate(formula[1], assignment)
    return (not evaluate(formula[1], assignment)) or evaluate(formula[2], assignment)


def is_tautology(formula) -> bool:
    vs = sorted(variables(formula))
    return all(
        evaluate(formula, dict(zip(vs, bits)))
        for bits in itertools.product((False, True), repeat=len(vs))
    )


# ---------------------------------------------------------------------------
# Part 2 -- REFERENCING: the toy formal system and a real Prov
# Lukasiewicz's three axiom schemas plus modus ponens; sound and complete
# for classical propositional logic.
# ---------------------------------------------------------------------------

def axiom_instances(subfs: FrozenSet, max_complexity: int = 8) -> List:
    """All axiom instances built from formulas in `subfs`, under the cap.
    (By the subformula property of Hilbert systems, a normal proof of F
    only needs instances built from parts of F -- that is what keeps the
    search fast without losing completeness.)"""
    subfs = tuple(sorted(subfs, key=lambda f: (complexity(f), show(f))))
    out: List = []
    for x in subfs:
        for y in subfs:
            a1 = ("\u2192", x, ("\u2192", y, x))
            if complexity(a1) <= max_complexity:
                out.append(a1)
            a3 = ("\u2192", ("\u2192", ("\u00ac", x), ("\u00ac", y)),
                  ("\u2192", y, x))
            if complexity(a3) <= max_complexity:
                out.append(a3)
            for z in subfs:
                a2 = ("\u2192",
                      ("\u2192", x, ("\u2192", y, z)),
                      ("\u2192", ("\u2192", x, y), ("\u2192", x, z)))
                if complexity(a2) <= max_complexity:
                    out.append(a2)
    return out


def _candidate_space(goal, max_complexity: int) -> FrozenSet:
    """Subformulas of the goal plus subformulas of all admissible axiom
    instances -- the only formulas a normal proof of the goal can need."""
    base = subformulas(goal)
    instances = axiom_instances(base, max_complexity)
    return functools.reduce(
        lambda acc, f: acc | subformulas(f), instances, frozenset(base)
    )


def _bfs_closure(candidates: FrozenSet, start: List, max_rounds: int = 24):
    """Forward proof search: the axiom-stamp game played honestly.
    Start from axioms, apply modus ponens among derived formulas, never
    leave the candidate space. Returns (parent, order); parent maps each
    formula to None (axiom) or (a, b) (MP on a and b)."""
    implications: Dict = {}
    for c in candidates:
        if isinstance(c, tuple) and c[0] == "\u2192":
            implications.setdefault(c[1], []).append(c)
    parent: Dict = {}
    order: List = []

    def add(f, src) -> None:
        if f not in parent:
            parent[f] = src
            order.append(f)

    for ax in start:
        add(ax, None)
    for _ in range(max_rounds):
        grown = False
        for a in list(order):
            for b in implications.get(a, ()):
                if b not in parent:
                    continue          # the major premise must be derived
                c = b[2]
                if c in candidates and c not in parent:
                    add(c, (a, b))
                    grown = True
        if not grown:
            break
    return parent, order


def _rebuild(parent: Dict, goal) -> List:
    """Recover a proof of `goal` from parent pointers, premises first."""
    steps: List = []
    seen = set()

    def visit(f) -> None:
        if f in seen:
            return
        src = parent[f]
        if src is not None:
            visit(src[0])
            visit(src[1])
        seen.add(f)
        steps.append(f)

    visit(goal)
    return steps


def search_proof(goal, max_complexity: int = 8, max_rounds: int = 24) -> Optional[List]:
    """A proof of `goal` in the toy system, or None if the bounded search
    closes without reaching it."""
    candidates = _candidate_space(goal, max_complexity)
    axioms = axiom_instances(subformulas(goal), max_complexity)
    parent, _ = _bfs_closure(candidates, axioms, max_rounds)
    if goal in parent:
        return _rebuild(parent, goal)
    return None


def provable(n: int, max_complexity: int = 8, max_rounds: int = 24) -> bool:
    """Prov(n): is the formula numbered n provable in the toy system?
    Numbers that decode to no symbol string, or to a string that is not a
    well-formed formula, are simply not provable."""
    text = decode_number(n)
    if text is None:
        return False
    try:
        formula = parse_formula(text)
    except ValueError:
        return False
    return search_proof(formula, max_complexity, max_rounds) is not None


# ---------------------------------------------------------------------------
# Part 3 -- MIRRORING: the diagonal sentence and the fork
# ---------------------------------------------------------------------------

QUINE_TEMPLATE = "q = %r; result = not provable(encode(q %% q))"


def make_mirror_sentence() -> str:
    """Fill the template with its own source, so the sentence's content
    becomes 'not provable(encode(G))' where G is the sentence itself.
    G is a fixed point of F(phi) = not Prov(<<phi>>) -- the same diagonal
    as the Goedel sentence (Ch. 22.2.5 reads it as the fixed point of an
    inference transformation)."""
    return QUINE_TEMPLATE % QUINE_TEMPLATE


def evaluate_mirror(sentence: str) -> bool:
    """Run the mirror sentence and return what it says (its truth value).
    Evaluating G computes: not provable(encode(G))."""
    scope: Dict[str, object] = {"provable": provable, "encode": encode_symbols}
    exec(compile(sentence, "<mirror-sentence>", "exec"), {"__builtins__": {}}, scope)
    return bool(scope["result"])


# ---------------------------------------------------------------------------
# The demonstration
# ---------------------------------------------------------------------------

@functools.cache
def _forms_at(c: int):
    """All formulas of exactly complexity c over {P, Q}."""
    if c == 0:
        return ("P", "Q")
    out = [("\u00ac", f) for f in _forms_at(c - 1)]
    for i in range(c):
        for f in _forms_at(i):
            for g in _forms_at(c - 1 - i):
                out.append(("\u2192", f, g))
    return tuple(out)


def _all_formulas(max_complexity: int) -> Iterator:
    for c in range(max_complexity + 1):
        yield from _forms_at(c)


def _completeness_sweep(max_complexity: int = 2, max_rounds: int = 24) -> Dict:
    """One shared proof search over {P, Q}, then compare 'tautology?'
    against 'derived?' for every formula up to the given complexity."""
    islands = list(_all_formulas(max_complexity))
    base = functools.reduce(
        lambda acc, f: acc | subformulas(f), islands, frozenset()
    )
    axioms = axiom_instances(base, max_complexity=8)
    candidates = functools.reduce(
        lambda acc, f: acc | subformulas(f), axioms, frozenset(base)
    )
    parent, _ = _bfs_closure(candidates, axioms, max_rounds)
    derived = frozenset(parent)
    taus = [f for f in islands if is_tautology(f)]
    return {
        "islands": islands,
        "derived": derived,
        "tautologies": taus,
        "derived_tautologies": [f for f in taus if f in derived],
        "derived_non_tautologies": [f for f in islands
                                    if f in derived and not is_tautology(f)],
    }


def demo() -> None:
    banner = "=" * 70
    print(banner)
    print(" THE MIRROR-SENTENCE GAME -- a runnable simulation")
    print(" The Reasoning Kingdom, Volume 2, Chapter 15 (Sec. 15.0, 15.0.5)")
    print(banner)

    # Part 1
    print("\nPART 1 -- NUMBERING (the book's scheme, run for real)")
    print("  scheme: P->1, ->->2, not->3  (this machine adds Q->4)")
    book_number = encode_symbols("\u00acP\u2192\u00ac\u00acP")
    expected = 2 ** 3 * 3 ** 1 * 5 ** 2 * 7 ** 3 * 11 ** 3 * 13 ** 1
    assert book_number == expected, "book example mismatch"
    print(f"  encode(\"\u00acP\u2192\u00ac\u00acP\") = {book_number}")
    print("    = 2^3 * 3^1 * 5^2 * 7^3 * 11^3 * 13^1   (matches Sec. 15.0.5)")
    print(f'  decode({book_number}) = "{decode_number(book_number)}"')
    print(f'  decode(32) = "{decode_number(32)}" -- not a formula: some')
    print("    numbers name no formula at all (as the book promises)")
    print('  encode("") = 1, decode(1) = ""  (the empty string lives at 1)')
    proof_number = encode_proof(("P", "Q", "\u00acP"))
    print(f"  proof level: encode_proof((P, Q, \u00acP)) = {proof_number}")
    print(f"    decode_proof -> {decode_proof(proof_number)}")
    print("  honest note: encoding the real 5-step proof below would need a")
    print("    number with ~10^160 digits. Nobody ever multiplies it out --")
    print("    the encoding exists to be expressible, not computed.")

    # Part 2
    print("\nPART 2 -- REFERENCING (a real toy formal system)")
    print("  system: Lukasiewicz's A1/A2/A3 schemas + modus ponens, over {P, Q}.")
    goal = parse_formula("P\u2192P")
    proof = search_proof(goal)
    assert proof is not None
    n_pp = encode_symbols("P\u2192P")
    print(f"  Prov(\u2308P\u2192P\u2309) = Prov({n_pp}) = True. The machine's actual proof:")
    parent, _ = _bfs_closure(
        _candidate_space(goal, 8), axiom_instances(subformulas(goal), 8), 24
    )
    index = {f: i + 1 for i, f in enumerate(proof)}
    for i, f in enumerate(proof, start=1):
        src = parent[f]
        how = "axiom" if src is None else f"MP: {index[src[0]]}, {index[src[1]]}"
        print(f"    {i}. {show(f):<42} [{how}]")
    n_p = encode_symbols("P")
    print(f"  Prov(\u2308P\u2309) = Prov({n_p}) = False: the search closed with no")
    print("    derivation (and P is not a tautology).")
    sweep = _completeness_sweep()
    islands, derived = sweep["islands"], sweep["derived"]
    print(f"  soundness: all {len(derived)} formulas the machine derived are")
    print("    tautologies (checked by truth tables). The system does not lie.")
    n_taut = len(sweep["tautologies"])
    n_ok = len(sweep["derived_tautologies"])
    n_bad = len(sweep["derived_non_tautologies"])
    print(f"  completeness sweep: {len(islands)} formulas of complexity <= 2:")
    print(f"    {n_ok}/{n_taut} tautologies derived; {n_bad} non-tautology derived")
    print("    (the second number must be 0).")
    assert n_ok == n_taut and n_bad == 0
    print("  sound and complete on this little island -- exactly why the")
    print("    mirror cannot crack it YET: its language has no word for")
    print('    "provable". (Ch. 15.1: a complete decidable system never lies.)')

    # Part 3
    print("\nPART 3 -- MIRRORING (the fork, run live)")
    sentence = make_mirror_sentence()
    n = encode_symbols(sentence)
    content = evaluate_mirror(sentence)
    proven = provable(n)
    print("  the machine builds a sentence G whose content is 'G is not provable':")
    print(f"    G = {sentence}")
    print(f"    n = encode(G) has {len(str(n))} digits, starting {str(n)[:40]}...")
    print(f"    what G says, computed: {content}")
    print(f"    can the toy system prove G: {proven}")
    print("  the fork from Sec. 15.0, with real values:")
    print("    G's claim is true, and the system cannot prove it. Here the")
    print("    failure is visible: the toy's proof search cannot even parse G")
    print("    -- a decidable system's language is too poor to mention")
    print('    "provable".')
    print("  why the crack must come once the system grows rich enough:")
    print("    assume the system can vouch for itself (Prov sound and complete):")
    print("      (1) provable*(G)  <=>  G is true")
    print("      (2) G is true     <=>  not provable*(G)   [by G's own content]")
    print("    (1) + (2)  =>  provable*(G) <=> not provable*(G)  -- contradiction.")
    print("  fixed-point reading (Ch. 22.2.5): G is a fixed point of")
    print("    F(phi) = not Prov(<<phi>>) -- the same diagonal as the Goedel")
    print("    sentence.")
    print("\n  Professor Pallas says:")
    print('    "The cruelty of the mirror-sentence game: the mirror is no external')
    print("    enemy. It is part of the system's expressive power. ... No one")
    print('    attacked the castle -- the castle grew its own reflection."')
    print("\n" + banner)
    print("  end of simulation -- every printed value came from a real computation")
    print(banner)


if __name__ == "__main__":
    demo()
```

## 15.1 Consistency: The System Does Not Lie

First, pin down exactly what "lying" means.

If a formal system can simultaneously prove some proposition $A$ and its negation $\neg A$, the system is **inconsistent**. Once this happens, the disaster is total: from $A$ and $\neg A$, using standard inference rules, one can prove **any** proposition. A system that can prove all propositions has proved nothing at all.

Thus consistency is the bottom line for a formal system; a system below this line is useless.

::: info Professor Manul comments
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

::: info What the completeness theorem means
The theorem says: those inference rules we wrote down — modus ponens, universal instantiation, existential introduction… — capture exactly all semantically valid inferences, not one more, not one less. This is not the result of design, but a non-trivial fact that requires proof. If completeness did not hold, it would mean that some "obviously valid" inference was missed by the rule system and required new inference rules to be added — and Gödel told you: not necessary.
:::

This was the last piece of good news Gödel brought.

::: info Professor Manul comments
"The last piece of good news" — this sentence is worth pausing on. The completeness theorem of 1930 and the incompleteness theorems of 1931 were brought by the same person in adjacent years, in completely opposite directions. First-order logic is complete; sufficiently strong arithmetic systems are incomplete. The boundary is right here, very precise, not some vague pessimism.
:::

One year later, he brought something completely different. The problem lies in those three words "sufficiently strong." Once the system is powerful enough to express arithmetic — natural numbers, addition, multiplication — completeness is permanently lost.

---

## 15.3 Gödel's Two Theorems

In 1931, Gödel published "On Formally Undecidable Propositions of Principia Mathematica and Related Systems." The paper directly gave the conclusions at the beginning, without any warm-up:

**Theorem One (First Incompleteness Theorem)**: any sufficiently strong consistent formal system contains propositions that can be neither proved nor refuted.

**Theorem Two (Second Incompleteness Theorem)**: any sufficiently strong consistent formal system cannot prove its own consistency inside itself.

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

Returning to the mirror-sentence game, this step is equivalent to turning "formulas" into game pieces that the system can hold in its hands and manipulate. The axiom stamp game of Chapter 14 only permitted you to write formulas on paper; now, the formulas on paper have numbers, and numbers can return to the paper to be talked about by another formula. The state space of the game suddenly gains one more layer: it is no longer just "what a proposition is," but "how a proposition talks about propositions."

This is the source of danger. Once the mirror is installed, the system no longer only looks outward at the world, but begins to look inward at its own proof behavior. Gödel's sentence is not a monster descending from the sky; it is a move naturally permitted by the mirror rules.

---

## 15.5 The Gödel Sentence: The Proposition That Says "I Am Not Provable"

Using Gödel numbering, one can express inside the system a predicate $\mathsf{Prov}(n)$, meaning "the proposition numbered $n$ is provable in this system." This predicate is itself an arithmetic proposition, completely legal.

The task now is: construct a proposition $G$ that is logically equivalent to $\neg\,\mathsf{Prov}(\ulcorner G \urcorner)$, i.e., "I myself am unprovable in this system."

This looks like a circular definition — the content of $G$ involves $G$'s own number, and $G$'s number depends on what $G$ is. How can one first write $G$, and then talk about $G$'s number?

The diagonalization lemma is what unties this knot.

:::details The diagonalization lemma: how self-reference is made legal in mathematics
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

The Second Incompleteness Theorem is a corollary of the first, but its impact is no less.

Take the statement "the system $\mathcal{F}$ is consistent" and encode it via Gödel numbering as an arithmetic proposition, written $\mathsf{Con}(\mathcal{F})$.

The proof process of the first theorem can itself be formalized inside the system — because the proof process is a finite symbol manipulation, and the system can express arithmetic. After formalization, one obtains:

$$\mathcal{F} \vdash \mathsf{Con}(\mathcal{F}) \to \neg\,\mathsf{Prov}(\ulcorner G \urcorner)$$

That is, the system can itself prove: "**If I am consistent, then $G$ is unprovable.**"

Now suppose the system can also prove its own consistency, i.e., $\mathcal{F} \vdash \mathsf{Con}(\mathcal{F})$. Combining these two via modus ponens, the system can prove that $G$ is unprovable — but "$G$ is unprovable" is equivalent to $G$ itself, so the system proves $G$, contradicting the first theorem.

Therefore, the system **cannot** prove its own consistency.

This is not modesty, nor a lack of ability. It is a structural fact of logic: **you cannot use a ruler to measure whether the ruler itself is accurate**. Any proof of a system's consistency must come from a stronger external system, and the consistency of that stronger system is again an open question — this hierarchy has no top.

::: info Professor Manul comments
What is truly shocking is that the second theorem says not "very hard" but "in principle impossible" — not an engineering problem, but a logical structural problem. A system's proof of its own consistency forever requires a stronger external system to provide the guarantee. There is no top. This is the point, not the ruler metaphor.
:::

::: info Then how do we believe ZFC is consistent?
To be honest: we cannot prove it. The mathematical community accepts ZFC because it has been used for nearly a hundred years, producing a large body of mutually consistent results, with no contradictions appearing. This is inductive reasoning, not deductive proof. Gödel's second theorem tells us that this is also the best we can do. The foundation of mathematics is not a rock proven to be flawless, but a supporting structure that has been repeatedly stress-tested and has not yet cracked. This difference is more important than most math classes admit.
:::

---

## 15.7 The End of Hilbert's Program and Its Legacy

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

## 15.8 Practice: What It Feels Like When Arithmetic Is Formalized

This chapter has been talking about "sufficiently strong systems" — strong enough to express arithmetic. This is not an abstract description. Peano Arithmetic (PA) is such a system; it has five axioms about the natural numbers. Before entering "Open Questions", spend a few minutes with the real thing.

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

Now feel what it means for this system to be "sufficiently strong": it can express all elementary statements about the natural numbers — "there exist infinitely many primes," "every even number is the sum of two primes (Goldbach's conjecture)," even "this system itself is consistent" — all can be written as formulas in the language of PA. It is exactly this expressive power that lets Gödel's diagonalization lemma do its work. A system can talk about itself because it can talk about all sufficiently complex number-theoretic statements — and statements about itself are a special case of number-theoretic statements.

---

## Open Questions

**$G$ is "true," but in what sense?** We see from outside the system that $G$ is true, but this "outside" depends on a stronger system, and that stronger system has its own $G$. Is "truth" an absolute concept, or is it always relative to some system?

**Are incompleteness and the halting problem the same thing?** Both Gödel and Turing established boundaries through self-referential constructions; the two are structurally highly similar. Recursive function theory gives a partial answer — they are indeed two sides of the same coin — but the full picture of this story will only unfold in Chapter 19.

**Can we escape incompleteness?** Adding $G$ as an axiom yields a stronger system, but the new system has its own Gödel sentence. Add that one too, and so on ad infinitum. Does this transfinite hierarchy have a limit? The answer is yes, but what the limit itself is, is a frontier problem of modern set theory.

**Which structural rule causes incompleteness?** Chapter 14 listed three structural rules running silently in the background: weakening, contraction, exchange. The existence of Gödel's theorem is rooted in the system being sufficiently strong — and "sufficiently strong" depends on assumptions being arbitrarily reusable. If we question this — if each assumption can be used only once — what does reasoning become? This is the starting point of Chapter 16.

---

## Exercises

**★ Warm-up**

The core of the diagonalization lemma is the substitution function $\mathsf{sub}(m, n)$. Judge the following descriptions as true or false:

1. The inputs of $\mathsf{sub}(m, n)$ are two natural numbers, and the output is also a natural number. (True/False?)
2. $\mathsf{sub}(\ulcorner D \urcorner, \ulcorner D \urcorner)$ yields the Gödel number obtained by substituting the free variable in formula $D$ with formula $D$ itself. (True/False?)
3. The specific method of Gödel numbering (e.g., using products of prime powers or another encoding) affects the conclusion of the First Incompleteness Theorem. (True/False?)

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

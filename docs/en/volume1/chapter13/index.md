# Chapter 13: The Boundaries of Reasoning — and Why We Must Accept Them

> Turing asked: Can machines think? The harder question is: Can we know that it is thinking?

<div class="center">

------------------------------------------------------------------------

</div>

## I. The Existence of Boundaries

This book began in Chapter 1 with entropy increase and prediction, traversed symbol systems, word vectors, neural networks, Transformers, search algorithms, complexity theory. We have seen the tremendous progress of AI reasoning, and we have also seen its fundamental limitations.

Now it is time to face a deeper question: **Does reasoning itself have boundaries?**

The answer is: yes. And this boundary is not a technical limitation, but a **logical necessity**.

The Halting Problem from Chapter 7 tells us: some problems are simply undecidable. Given a program and its input, no algorithm can determine whether it will halt.

The Yonglin Formula from Chapter 12 tells us: reasoning chains will ultimately converge back to prior anchor points, the object level closes, the meta-level fractures.

These two results point to the same fact: **any sufficiently powerful reasoning system contains problems it cannot solve**.

This is not a failure, but a mathematical truth.

<div class="center">

------------------------------------------------------------------------

</div>

## II. Gödel's Echo

In 1931, Gödel proved a theorem that shook the mathematical world: **any sufficiently strong formal system contains true propositions that it cannot prove**.

Let me restate this theorem in the simplest way.

Suppose you have a formal system $S$ that has:

- A set of axioms: the starting points the system acknowledges
- A set of inference rules: how to derive new propositions from axioms

As long as this system is sufficiently strong — strong enough to express basic arithmetic, strong enough to talk about "proof" itself — Gödel can construct a special proposition $G$ within the system:

$$
G = \text{"Proposition G is unprovable in system S"}
$$

Now ask: is $G$ true?

**Case 1**: If $G$ is provable, then what $G$ says — "G is unprovable" — is false. A sound formal system has proven a false proposition; the system collapses.

**Case 2**: If $G$ is unprovable, then what $G$ says — "G is unprovable" — is true. So $G$ is true, but the system cannot prove it.

Conclusion: **There exists a true proposition $G$ that system $S$ cannot prove**.

Stop, let this sentence turn around in your head a few times.

"Proposition $G$ is unprovable in system $S$" — if it's provable, it lied; if it's unprovable, it told the truth. This sentence is not describing the world; it is describing itself. And when you try to judge whether it's true or false, you've already fallen into the pit it dug.

This is not a failure of logic; it is the structural cost of self-reference. Any sufficiently strong system contains such a pit, because "sufficiently strong" itself means: it can not only talk about objects, but also talk about how it talks about objects.

:::details Why Does "Sufficiently Strong" Bring Self-Reference?
Gödel's technique is not simply writing a paradox, but encoding "proof" into arithmetic relations. Propositions, proof sequences, and inference rules can all be numbered, becoming relations between natural numbers. Thus, a system that seems only to talk about addition and multiplication suddenly acquires the ability to talk about "whether a certain proposition is provable."

This is where the danger lies: once a system can encode its own syntax, it can construct propositions that point to themselves. Self-reference is not an add-on feature, but a side effect of sufficient expressive power.
:::

This also explains why incompleteness is not "the system is not strong enough." On the contrary, the stronger the system, the more inevitably it encounters its own shadow.

<div class="center">

------------------------------------------------------------------------

</div>

## III. The Yonglin Formula and Gödel's Theorem: A Structural Isomorphism

Between the Yonglin Formula of [Zixi Li, 2025b] and Gödel's incompleteness theorem, there exists a structural isomorphism that deserves serious elaboration.

Recall the Yonglin Formula:

$$
\lim_{n \to \infty} \Pi^{(n)}(s) = A, \quad \text{but} \quad A \neq A^*
$$

This formula says: **the object level closes, the meta-level fractures**.

- **Object level**: the model can generate reasoning chains, maintaining internal self-consistency
- **Meta-level**: the model cannot truly jump out of its own parameters and training distribution to verify whether the reasoning chain reaches external truth
- **Prior anchor point $A$**: the internal attractor to which the model ultimately converges
- **True anchor point $A^*$**: the correct structure of the task in the external world

This and Gödel's theorem are not "somewhat similar," but share the same deep shape: a system can generate infinitely many legitimate steps internally, yet cannot, from within, guarantee that these steps arrive at truth outside the system.

| Gödel's Theorem | Yonglin Formula |
|:-----------|:---------|
| Formal system $S$ | AI reasoning system |
| Axioms + inference rules | Training data + network architecture |
| Proposition $G$: "G is unprovable" | Prior anchor point $A$: "A is correct" |
| $G$ is true but unprovable | $A \neq A^*$ but the system converges to $A$ |
| System cannot prove its own completeness from within | Model cannot verify its own reasoning from within |

First-row mapping: **Formal system $S$ ↔ AI reasoning system**.

A formal system is not a single sentence, but an entire machine for generating legitimate propositions: it has axioms, inference rules, and a criterion for "what counts as a proof." An AI reasoning system is also not a single answer, but an entire machine for generating reasoning trajectories: it has training data, network architecture, decoding strategies, context windows, attention distribution mechanisms.

So the correspondence here is not a metaphor, but a structural correspondence: both are "starting from internal rules, generating a sequence of seemingly legitimate symbols." Formal systems generate proofs; AI systems generate reasoning chains.

Second-row mapping: **Axioms + inference rules ↔ Training data + network architecture**.

The axioms of a formal system determine what it acknowledges as starting points; the inference rules determine how it is allowed to advance. The training data of a large model determines which patterns it treats as "natural"; the network architecture determines how these patterns are combined, compressed, retrieved, and unfolded.

If the axioms of a formal system are its worldview, then training data is the model's worldview. If inference rules are the kinematics of a formal system, then network architecture is the kinematics of the model. Where a system can go depends not only on where it "wants" to go, but even more on where it was allowed to start and in what way it is allowed to move.

Third-row mapping: **Gödel proposition $G$ ↔ Prior anchor point $A$**.

The uncanniness of the Gödel proposition $G$ lies in this: it stuffs the system's meta-level problem back into the object level. On the surface it is an ordinary proposition; more deeply, it asks "can the system prove this proposition."

The prior anchor point $A$ also has a similar dual identity. On the surface, it is merely the answer tendency to which the model converges at the end of a reasoning chain; looking deeper, it represents the training distribution structure that the model cannot escape. The model thinks it is reasoning about the external world, but in reality it is often sliding along internal attractors.

This is why $A$ is not an ordinary wrong answer. An ordinary wrong answer can be corrected by the next step of reasoning; the prior anchor point, however, is the attractor center of the reasoning dynamics itself. The more you let the model "keep thinking," the more likely it is to self-consistently orbit this center.

Fourth-row mapping: **True but unprovable ↔ Anchor point not equal to truth but still converged to**.

In Gödel's theorem, $G$ is true, but the system cannot prove $G$. In the Yonglin Formula, the external truth is $A^*$, but the model, in object-level reasoning, converges to $A$, and generally $A \neq A^*$.

This does not mean the model is necessarily wrong, nor that formal systems are necessarily useless. Quite the opposite: formal systems can prove vast numbers of theorems; models can complete vast amounts of reasoning. The problem is that, once you demand that the system provide, from within, a guarantee that "I have arrived at the complete truth," it will fail.

:::details Technical Detail: Why Can Training Data + Network Architecture Correspond to Axioms + Inference Rules?
A formal system can be written as a triple:

$$
S = (\mathcal{A}, \mathcal{R}, \vdash)
$$

Where $\mathcal{A}$ is the set of axioms, $\mathcal{R}$ is the inference rules, and $\vdash$ is the provability relation. An AI reasoning system can also be abstracted as:

$$
M = (D, \Theta, \Pi)
$$

Where $D$ is the training data distribution, $\Theta$ is the state transition mechanism determined by network architecture and parameters, and $\Pi$ is the distribution of reasoning trajectories generated by decoding.

Under this abstraction:

- $\mathcal{A} \leftrightarrow D$: the starting points the system acknowledges / the world statistics the model absorbs
- $\mathcal{R} \leftrightarrow \Theta$: legitimate derivation rules / implicit state transition rules
- $\vdash \leftrightarrow \Pi$: provability relation / generatable reasoning trajectories

The Yonglin Formula is not concerned with whether a particular output step is elegant, but with whether, when $\Pi$ is repeatedly iterated, it can leave the basin of attraction jointly shaped by $D$ and $\Theta$. The conclusion is: in the absence of external meta-level verification, reasoning trajectories will converge back to the prior anchor point $A$.
:::

:::details 
Technical Detail: Why Can $A$ Correspond to Gödel Proposition $G$?
The core of Gödel proposition $G$ is not "this sentence is convoluted," but that it turns the system's proof capability into an object within the system. It forces the system to answer: "Can I prove what I cannot prove?"

The prior anchor point $A$ also turns the model's meta-level limitation into object-level output. The model does not say "I cannot escape the training distribution"; it merely gives an answer that appears plausible, linguistically fluent, and internally self-consistent. This answer is the projection of the meta-level limitation onto the object level.

Therefore, the correspondence is not: $A$ and $G$ look alike.

The correspondence is: both compress the meta-level boundary that the system cannot cross into an object visible within the system.
:::

The bonus section of Chapter 9 revealed the physical intuition of this convergence: self-attention is mathematically equivalent to a one-step retrieval of Hopfield associative memory; the statistical bias $A$ of the training data is encoded as the global minimum of the energy function — an attractor. The longer the reasoning chain, the more the model is pulled toward this attractor. The convergence of the Yonglin Formula is the necessity of energy minimization, not merely the passive failure of Bayesian updating.

Thus, Gödel's logical incompleteness, the Yonglin Formula's meta-level fracture, and Hopfield's energy attraction describe the same thing projected in different languages: **any sufficiently strong reasoning system encodes its own limitations into its own structure**.

What is truly dangerous is not that the system makes mistakes. What is truly dangerous is that: the system can generate infinitely many seemingly correct reasons within its own boundaries, and mistake this internal self-consistency for external truth.

<div class="center">

------------------------------------------------------------------------

</div>

## IV. Hands-On: Construct an Undecidable Problem

Let us personally experience the undecidability of the Halting Problem.

**Step 1: Assume the existence of a halting decider**

**Step 2: Construct the diagonalization program**

The Python code below concretizes this thought experiment. We first define a "hypothetical" halting decider (in reality impossible to precisely implement), then construct the diagonalization program that makes it self-contradict, and finally verify the inevitability of the contradiction through exhaustive simulation.

```python
import sys
import threading

# ------------------------------------------------------------------
# Step 1: Simulate an "approximate halting decider" using timeout
#   A true HALT cannot exist; here we use a finite timeout as an approximation:
#   If func returns within timeout seconds, judge as "halts"; otherwise, "does not halt".
# ------------------------------------------------------------------
def approximate_halt(func, timeout=0.05):
    """
    Approximately judge whether func() will halt within timeout seconds.
    Returns True for "judged to halt", False for "judged not to halt".
    """
    result = [None]

    def runner():
        try:
            func()
            result[0] = True   # Function returned normally → halts
        except Exception:
            result[0] = True   # Throwing an exception also counts as halting

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    # Timed out without finishing → judged as "does not halt"
    return result[0] is True

# ------------------------------------------------------------------
# Step 2: Construct the diagonalization program DIAG
#   DIAG's logic:
#     If HALT(DIAG) returns True (judged that DIAG will halt)
#       → DIAG enters an infinite loop (actually does not halt)
#     Otherwise (judged that DIAG does not halt)
#       → DIAG immediately returns (actually halts)
# ------------------------------------------------------------------
def make_diag(halt_checker):
    """Factory function: construct a diagonalization program using the given halt_checker"""
    def diag():
        # First ask the "decider": will I (diag) myself halt?
        will_halt = halt_checker(diag)
        if will_halt:
            # Decider says I will halt → I'll just loop forever
            while True:
                pass
        else:
            # Decider says I won't halt → I'll just return immediately
            return

    return diag

# ------------------------------------------------------------------
# Step 3: Ask: will DIAG(DIAG) halt? Reveal the contradiction
# ------------------------------------------------------------------
print("=" * 55)
print("Halting Problem Undecidability: Diagonalization Argument")
print("=" * 55)

diag = make_diag(approximate_halt)

# First use the decider to predict the result
prediction = approximate_halt(diag)
print(f"\nDecider predicts: DIAG will {'halt' if prediction else 'not halt'}")

# Then use timeout to actually observe DIAG's behavior
actual_halted = approximate_halt(diag, timeout=0.1)
print(f"Actual observation: DIAG {'halted' if actual_halted else 'did not halt (timeout)'}")

# Determine whether a contradiction occurred
if prediction != actual_halted:
    print("\n→ Contradiction! The decider's prediction is opposite to DIAG's actual behavior.")
else:
    # Note: Due to timeout approximation, two observations may "coincidentally" agree,
    # but this is only an error of the approximate decider, not affecting the rigorous mathematical proof.
    print("\n→ Note: timeout approximation obscures the contradiction, but the rigorous mathematical proof")
    print("   guarantees that a precise HALT necessarily leads to contradiction (see analysis below).")

print("\n--- Rigorous Logical Analysis ---")
print("Case A: HALT(DIAG, DIAG) = True  → DIAG loops forever → Actually does not halt → Contradiction!")
print("Case B: HALT(DIAG, DIAG) = False → DIAG returns immediately → Actually halts   → Contradiction!")
print("\nConclusion: A precise halting decider HALT is logically impossible.")
print("      This is not because the algorithm is not smart enough, but a logical necessity.")
```

**Step 3: Ask DIAG(DIAG)**

Now ask: will DIAG(DIAG) halt?

- **If HALT(DIAG, DIAG) returns True**:

  - Meaning DIAG(DIAG) will halt

  - But according to DIAG's definition, it will enter an infinite loop

  - Contradiction!

- **If HALT(DIAG, DIAG) returns False**:

  - Meaning DIAG(DIAG) does not halt

  - But according to DIAG's definition, it will immediately return (halt)

  - Contradiction!

**Conclusion**: HALT cannot exist.

**What does this contradiction illustrate?**

It's not that we haven't yet found a sufficiently clever algorithm, but that **no algorithm can solve the Halting Problem**. This is logical necessity, not technical limitation.

<div class="center">

------------------------------------------------------------------------

</div>

## V. The Map of the Reasoning Kingdom

Now let us step back.

Spread out the previous thirteen chapters: Chapter 1's entropy increase tells us why the world needs prediction, Chapter 2's symbols tell us how humans wrote prediction into operable rules, Chapters 3 through 6 tell us how neural networks compress symbols into vectors, advance vectors into states, organize states into the attention flow of the Transformer. Chapter 7's complexity tells us: not all searches can be completed elegantly. Chapter 8's heuristics tell us: when exactness is unreachable, we must learn to compromise at a cost. Chapters 9 through 11 tell us: attention, search, and randomization can all push the boundary forward a bit, but cannot push away the boundary itself. Chapter 12's Yonglin Formula finally tells us: even the act of "continuing to reason" itself has its own convergence limit.

All of these ultimately fall onto this map.

**Central Zone: Decidable Problems.**

This is the brightest district of the Reasoning Kingdom. Sorting, shortest paths, dynamic programming, linear algebra, many engineering optimization problems — all live here. They are not necessarily simple, but at least there are clear roads: given an input, the algorithm will stop, the answer can be verified, the complexity can be estimated.

P-class problems are the main roads of the central district: solvable in polynomial time. NP-class problems are more complex neighborhoods: answers can be quickly verified, but finding answers may be extremely expensive. The $\mu(L,d)$ phase diagram from Chapter 7 tells us that even within NP, not every street is equally dark. Solvability is not a switch, but a continuous landscape.

**The Fog Belt: Phase Transition Regions.**

Moving further out, you will encounter fog.

OpenXOR's $\alpha \approx 4.26$ is a typical fog belt: instances abruptly change from "almost always solvable" to "almost always unsolvable." Collins' $c_{\text{opt}} \approx 34$ is also a fog belt: compression moves from safe to dangerous. The effective reasoning window of CoT is likewise a fog belt: reasoning slides from effective unfolding gradually toward repeating priors.

The most important feature of the fog belt is: it is not a wall. You cannot simply say "here it's solvable, there it's not." In the fog, tiny parameter changes can cause enormous consequences. One extra constraint, a bit of extra compression, a few too many steps in the reasoning chain — any of these can cause the system to slide from lucidity into hallucination.

**Outer Ring: Undecidability and Incompleteness.**

Further out lies the dark border of the Kingdom.

The Halting Problem tells us: some problems have no universal decision algorithm. Gödel's proposition tells us: some true propositions cannot be proven within the system. The Yonglin Formula tells us: some reasoning chains cannot complete meta-level self-verification within the model.

These three are not the same theorem, but they point in the same direction: when a system is sufficiently strong — strong enough to express, simulate, and reflect upon itself — it inevitably generates its own blind spots.

:::details Distinctions Among the Three Boundaries
- **Complexity boundary**: The problem is solvable, but may be too slow. A typical example is NP-hard search.
- **Phase transition boundary**: The problem does not suddenly become unsolvable, but becomes sharply harder in a certain parameter region. A typical example is 3-SAT's $\alpha \approx 4.26$.
- **Logical boundary**: The problem is not slow, but no universal algorithm exists. Typical examples are the Halting Problem and Gödel incompleteness.

Conflating these three leads to two kinds of errors: one is mistaking logical impossibility for insufficient engineering effort; the other is prematurely declaring engineering difficulty as philosophical destiny.
:::

This map tells us: **the boundary of reasoning is not a line, but a gradient of fog; at the end of the fog lies not a bigger model, but the horizon of logic itself**.

This is what Volume I truly wants to say: reasoning is not an infinitely ascending ladder, but a search for paths across the three terrains of decidability, phase transition, and incompleteness. Knowing where you stand is more important than blindly charging forward.

<div class="center">

------------------------------------------------------------------------

</div>

## VI. Pseudocode: Formalizing the Boundaries

**Algorithm 1: Diagonalization Proof of the Halting Problem**

    Proof of the Undecidability of the Halting Problem:

    Assumption: There exists a decider HALT(P, x)

    Construct the diagonal program:
    def DIAG(P):
      if HALT(P, P):
        loop_forever()  # Infinite loop
      else:
        return  # Halt

    Ask: HALT(DIAG, DIAG) = ?

    Case 1: HALT(DIAG, DIAG) = True
      → DIAG(DIAG) will halt
      → But DIAG's definition: if HALT returns True, then loop forever
      → Contradiction!

    Case 2: HALT(DIAG, DIAG) = False
      → DIAG(DIAG) does not halt
      → But DIAG's definition: if HALT returns False, then halt
      → Contradiction!

    Conclusion: HALT does not exist

![Diagonalization Argument: Visualization of the Undecidability of the Halting Problem](/figures/ch15_fig1_diagonalization.png)

*Figure 1: Matrix representation of the diagonalization argument. Rows are programs P, columns are inputs x, cells mark the result of HALT(P,x) (H=halt, L=loop). The cells on the diagonal, HALT(P,P), define DIAG: where the diagonal has H, make DIAG loop; where it has L, make DIAG halt. DIAG must appear in some row — but that row's value on the diagonal contradicts DIAG's definition. This contradiction is logical necessity, not technical limitation.*

**Algorithm 2: Meta-Level Fracture of the Yonglin Formula**

    Meta-Level Fracture Detection (AI system S, Reasoning task T):
    Input: AI system, Reasoning task
    Output: Object-level conclusion, Meta-level limitation

    1. Object-level reasoning:
       C = S.reason(T)  # System gives a conclusion

    2. Meta-level verification:
       Ask: "Can S prove that C is correct?"

    3. Incompleteness detection:
       if S cannot verify C at the meta-level:
         # There exists a true proposition C*, true at the object level but unprovable at the meta-level
         return (C, "Meta-level fracture")

    4. Analogy to Gödel:
       G = "Proposition G is unprovable"
       # G is true, but the system cannot prove G
       # Yonglin Formula: A is the prior, but A ≠ A*

    5. Return (C, "Object level closed, meta-level fractured")

<div class="center">

------------------------------------------------------------------------

</div>

<div class="center">

------------------------------------------------------------------------

</div>

## VII. The Future of Reasoning: How Far Can We Go Within the Boundaries

Acknowledging the existence of boundaries does not mean giving up. Boundaries give direction: **within the boundaries, how far can we go?**

---

### 7.1 From Impossibility to "Probabilistic Possibility"

The Yonglin Formula reveals meta-level fracture — when the model verifies its own reasoning, it uses the same set of parameters that produced the reasoning, unable to truly step outside.

But this is not binary. It is not "completely can" or "completely cannot," but a probabilistic spectrum:

- **Short-chain reasoning** (3-5 steps): meta-level fracture has small impact; the model can effectively self-correct
- **Medium-chain reasoning** (10-20 steps): error accumulation begins to manifest; external calibration points are needed
- **Long-chain reasoning** (50+ steps): prior anchor point dominates; self-correction is nearly ineffective

This has direct implications for system design: **do not let the model complete reasoning chains beyond the effective reasoning window alone**. Interrupting, calibrating, restarting — these are necessary engineering compensations.

---

### 7.2 External Verifiers: Breaking Meta-Level Closure

The fundamental problem of the Yonglin Formula is self-reference — using reasoning itself to verify reasoning.

The approach to break this cycle is to introduce **independent external verifiers**:

- **Formal verification**: Machine proofs of mathematical theorems (Lean, Coq). The model generates proof drafts; the formal system independently verifies each step. The verifier does not share the model's parameters; the meta-level fracture is repaired by external force.
- **Symbolic execution**: In code reasoning, let an interpreter actually run the code generated by the model, using execution results as verification signals.
- **Multi-model debate**: Use two independently trained models to challenge each other's reasoning, reducing the impact of systematic bias.

The common logic of these approaches is: **move the meta-level from inside the model to outside**. There is no parameter sharing between the external verifier and the internal reasoner; Gödelian self-reference is physically severed.

---

### 7.3 Extended Context: A Longer Effective Reasoning Window

The effective reasoning window $t_{\text{conv}}$ is not fixed; it is related to model capacity, task structure, and training method.

Research directions from the past two years:

- **Long-range CoT training**: DeepSeek-R1's GRPO discovered that intra-group competitive pressure spontaneously produces longer, more structurally complete reasoning chains — this amounts to extending the effective reasoning window during training.
- **Memory augmentation**: Externalizing intermediate reasoning steps to external storage (tool calls, vector databases), so the model doesn't have to hold all intermediate states in activations.
- **Recursive refinement**: Reasoning about the same problem multiple times, each time using the previous output as a new starting point, similar to iterative convergence in numerical methods.

---

### 7.4 Accepting Boundaries Is the Starting Point of Design

The Halting Problem tells us: there is no universal decidable algorithm. P≠NP (if it holds) tells us: there is no universal efficient search. The Yonglin Formula tells us: there is no complete self-verification.

But these three boundaries precisely outline a design space:

| Boundary | Engineering Compensation |
|:-----|:--------|
| Halting undecidability | Bounded computation, timeout termination |
| NP-hard search | Heuristics, approximation algorithms, randomization |
| Meta-level fracture | External verifiers, tool calls, multi-model debate |

Every tool in this book — A*, MCTS, ADS, Collins, Transformer — is an engineering compensation within some boundary. They do not break through boundaries, but **find the optimal way to survive within the boundaries**.

This is perhaps the most honest definition of reasoning: **finding the optimal path within constraints**.


## VIII. Accept Boundaries, Rather Than Deny Boundaries

The core argument of this book is: **reasoning has boundaries, and these boundaries are fundamental**.

But this is not a pessimistic conclusion; it is the starting point of liberation.

When we accept the existence of boundaries, we no longer pursue impossible perfection:

- No longer expect AI to solve all problems
- No longer believe that larger models can eliminate all limitations
- No longer view incompleteness as a defect

Instead, we begin to ask more meaningful questions:

- **Within the boundaries, how well can we do?**
- **How to identify whether a problem lies within or beyond the boundaries?**
- **How to make the best decisions under uncertainty?**

Chapter 7's μ(L,d) phase diagram tells us: solvability is not binary, but probabilistic. Near the phase transition point, problems exist in a superposition of solvable and unsolvable.

Chapter 8's heuristics tell us: when the optimal solution is unreachable, we need a "good enough" solution, and must be able to precisely quantify the degree of "good enough."

Chapter 10's MCTS tells us: reasoning does not require "understanding," only effective search.

Chapter 11's Collins tells us: randomization can break through deterministic lower bounds, trading probabilistic guarantees for efficiency gains.

Chapter 13's Yonglin Formula tells us: the value of CoT lies in the effective reasoning window before convergence, not in the total number of steps.

**These are all wisdoms within the boundaries.**

But there is a trap here.

Using heuristics, approximation algorithms, and external verifiers to optimize within the boundaries — this itself can also become another kind of "arrogance of artifacts": treating boundaries as engineering problems to be conquered, rather than logical facts to be respected. Technical systems are best at reformulating every problem as "can we optimize it a bit more." But in some places, the problem is not that you haven't optimized enough, but that you forgot to ask: optimization for what?

The true boundary of reasoning is not technical, but purposive. What is reasoning for? If only for accuracy, then continuing to optimize within the boundaries is enough. If it is for understanding — for humans to understand the world, understand themselves, understand others — then the boundary is not a wall, but a signpost. It tells you: up to here, beyond this is no longer the territory of reasoning.

What remains is humor, aesthetic judgment, moral courage, the capacity to love — those human qualities that algorithms cannot imitate, and should not be swallowed by algorithms.

<div class="center">

------------------------------------------------------------------------

</div>

## IX. Boundaries Are Possibility

Boundaries are not endpoints, but starting points.

When we know certain problems are undecidable, we don't waste time searching for perfect algorithms, but turn to approximations, heuristics, probabilistic guarantees.

When we know reasoning converges to prior anchor points, we don't blindly increase reasoning steps, but optimize the effective reasoning window.

When we know global receptive field and linear complexity cannot both be achieved, we don't pursue non-existent architectures, but search for the optimum in the trade-off.

**Boundaries define the space of possibility.**

Within the boundaries, we have infinite creative space:

- Better heuristic functions
- More efficient search algorithms
- More precise approximation guarantees
- Smarter resource allocation

Beyond the boundaries, we have deeper theoretical questions:

- Can boundaries be pushed forward?
- Do the boundaries of different problems share a unified mathematical structure?
- Is human reasoning also constrained by the same boundaries?

<div class="center">

------------------------------------------------------------------------

</div>

## Unresolved

This book began with unresolved questions and ends with unresolved questions.

- **Are the boundaries of reasoning fixed, or can they be pushed forward?** Can P vs NP be proven? Or is it itself an undecidable proposition?

- **Can we build a "sufficiently weak" system to avoid incompleteness?** If the system is not strong enough, it won't encounter the Gödel proposition. But is such a system still useful?

- **After accepting limitations, what should we do?** Optimize within the boundaries, or try to break through the boundaries?

- **Where are the boundaries of human reasoning?** Are we also constrained by the Yonglin Formula? If so, how do we overcome it?

- **Is uncertainty a defect or a feature?** Perhaps the essence of reasoning is precisely making the best guess under incomplete information, rather than seeking absolute truth.

**These questions have no answers. But asking the questions is itself the meaning of reasoning.**

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- [Zixi Li, 2025b] — The Yonglin Formula, theoretical proof of reasoning incompleteness, with structural isomorphism to Gödel's incompleteness theorem

- Gödel, K. (1931). *Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I* — Original paper on incompleteness theorems

- Turing, A. (1936). *On Computable Numbers, with an Application to the Entscheidungsproblem* — Halting Problem and computability

- Turing, A. (1950). *Computing Machinery and Intelligence* — The Turing Test, can machines think?

- Penrose, R. (1989). *The Emperor's New Mind* — Gödel's theorem and philosophical discussion of artificial intelligence

- [Hamkins & Nenu, 2024] — Historical Clarification of the Halting Problem `→ [arXiv:2310.07927]`

- Hofstadter, D. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid* — Classic work on self-reference, recursion, and consciousness

<div class="center">

------------------------------------------------------------------------

</div>

## Epilogue

The boundary of the Reasoning Kingdom is not a wall, but a horizon.

When you stand on the boundary, what you see is not an endpoint, but a vaster unknown.

This book has attempted to sketch the outline of this horizon — from entropy increase and prediction to symbol systems, from word vectors to neural networks, from Transformers to complexity theory, from search to reasoning, from efficiency to boundaries.

But the horizon is forever ahead.

Every answer leads to new questions. Every boundary hints at deeper structures.

**The journey of reasoning has no endpoint, only deeper understanding.**

And this is precisely the charm of reasoning.

Good luck.

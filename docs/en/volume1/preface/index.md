# A Reader's Guide

When GPT-4 defeats human contestants in mathematics competitions, when o3 surpasses experts on PhD-level scientific reasoning tests, we marvel at AI's "intelligence." But few ask the follow-up question: **do these systems truly "reason"?**

The deeper question is: **what is reasoning?**

Once you begin to think about it seriously, you discover: human reasoning may be nothing more than the illusion of pattern matching; logical reasoning may not even exist in the physical world; what we call "understanding" may be merely a byproduct of data compression.

This is not a book about how to use AI. It is a book about **why AI can reason, why it cannot, and what reasoning itself is**.

This book will not give you answers. But it will take you to the boundaries of reasoning — the questions that kept Turing, Gödel, and Shannon awake at night.

---

## Why Read This Book

If you're satisfied with "AI is amazing," this book is not for you.

If you want to know:
- Why do hundred-billion-parameter large models collapse on simple logical chains?
- Why does Chain-of-Thought (CoT) improve accuracy, yet ultimately converge back to the prior?
- Why is P≠NP not about "fast vs. slow," but about the asymmetry of the universe?
- Why does any sufficiently powerful reasoning system contain problems it cannot solve?

Then this book is written for you.

We won't stop at the surface of "what AI can do," but dive into the underlying mechanisms of **why AI can do it, and why it cannot**. The upper volume builds intuition through historical narrative and runnable experiments; the lower volume firms up that intuitive foundation with rigorous formal language.

---

## This Is a Construction, Not a Survey

The narrative of this book revolves around six original research projects. They are not summaries of prior work, but exploratory constructions the author undertook to understand the nature of reasoning:

**1. QMCB / OpenXOR: A Continuous Phase Diagram for NP Problems**

Traditional complexity theory tells us whether a problem "belongs to" P or NP, but cannot quantify how hard a specific instance is. The OpenXOR framework breaks through this limitation, transforming the solvability of NP problems from a binary verdict into a continuous phase diagram.

For an instance of size L and constraint density d, the solvability probability μ(L,d) satisfies:

$$
\mu(L,d) = \frac{1}{2}\left(1 - \text{erf}\left(\frac{d - d_c(L)}{0.1007}\right)\right)
$$

where the critical constraint density $d_c(L) = -0.0809 \ln(L) + 0.501$.

This formula reveals: **computability is not binary, but probabilistic**. NP is not a wall, but a gradient of fog. At the edge of the fog (μ≈0.5), the problem exists in a quantum superposition of solvable and unsolvable.

`→ [DOI: 10.13140/RG.2.2.22376.64006]`

---

**2. The Yonglin Formula: The Essential Incompleteness of AI Reasoning**

Why do large models fail on long-chain reasoning? Not because they lack parameters, but because **the object-level is closed while the meta-level is fractured**.

The Yonglin Formula proves: no matter how long the reasoning chain, it will ultimately converge back to the prior anchor:

$$
\lim_{n \to \infty} \Pi^{(n)}(s) = A, \quad \text{but} \quad A \neq A^*
$$

- $\Pi^{(n)}(s)$: the model's reasoning distribution at step n
- $A$: the prior anchor of the training data
- $A^*$: the true correct answer

The model can operate self-consistently at the object level (generating reasoning chains), but at the meta-level (verifying whether the reasoning is correct), it cannot escape the constraints of its own parameters. This shares a structural isomorphism with Gödel's incompleteness theorems — any sufficiently powerful formal system contains true propositions it cannot prove.

The value of CoT lies not in "longer is better," but in extending the **effective reasoning window** — those few steps before convergence are where true reasoning occurs.

`→ [Derived and explained in Chapter 12 of this book, original]`

---

**3. ADS (Adaptive Dual Search): Information-Theoretic Heuristic Weighting**

In search and reasoning, how do we dynamically balance "following the heuristic" and "autonomous exploration"? Traditional methods use a fixed weight α, but the optimal α varies dynamically with state uncertainty.

ADS renders α information-theoretic, adaptively adjusting the search strategy through the entropy of the current state-action distribution:

$$
\alpha_t = -\log\!\left(1 - \frac{H_t}{H_{\max}}\right)
$$

where $H_t$ is the entropy of the current output distribution and $H_{\max}$ is the maximum entropy of the uniform distribution. When entropy is high (high uncertainty), $\alpha_t$ increases, forming an information-theoretic barrier that repels high-entropy states, forcing the search to collapse onto a low-entropy manifold; when entropy is low, $\alpha_t$ approaches zero, trusting the heuristic to advance rapidly. This realizes Yao's Minimax Theorem in the context of adaptive search — finding the optimal exploration-exploitation mixed strategy in uncertain environments.

`→ [Chapter 8, §VI ADS: The Information-Theoretic Turn of Heuristics]` &nbsp; `→ [DOI: 10.13140/RG.2.2.17091.16164]`

---

**4. The Collins Optimizer: Touching the Physical Limits of Compression**

Optimizers like Adam and AdamW need to maintain momentum and second-order moments for every parameter, with memory overhead three times the parameter count. Can optimizer states be compressed?

Collins achieves O(1) state compression through randomization, with a safe compression ratio $c_{\text{opt}} \approx 34$:

$$
c_{\text{opt}} = \frac{d}{\log_2(d/\delta)}
$$

where d is the parameter dimension and δ is the error tolerance. This limit arises from information theory's rate-distortion theory — you cannot compress infinitely without losing information.

Experimental validation: on Yi-34B-Chat, performance loss after 34× compression was <2%, but collapsed after 64× compression. This is not an engineering issue, but a mathematical boundary.

`→ [DOI: 10.13140/RG.2.2.23802.04809]`

---

**5. A Causal-Topological Reinterpretation of Self-Attention: A Thought Experiment**

The standard interpretation of Self-Attention is the information retrieval analogy (Query-Key-Value). But starting from causal modeling, the same mathematical structure can be derived — and endowed with deeper semantics. This is not a proven theorem, but a conjecture searching for precise formulation.

Let the projection vectors of position i (effect) and position j (cause) be $q_i = W_Q x_i$ (row projection / effect modeling) and $k_j = W_K x_j$ (column projection / cause modeling). Their outer product encodes the complete structure of a causal hypothesis:

$$\mathcal{C}_{ij} = q_i \otimes k_j \in \mathbb{R}^{d_k \times d_k}$$

Performing Einstein summation (trace) over the shared dimension yields the causal hypothesis strength scalar:

$$A_{ij} = \mathrm{tr}(\mathcal{C}_{ij}) = q_i \cdot k_j$$

Applying softmax over candidate causes yields the causal posterior distribution — i.e., the standard attention matrix.

This derivation reveals three things: (1) $W_Q \neq W_K$ is not an engineering choice, but a necessary encoding of causal asymmetry; (2) softmax is a Bayesian posterior normalization over candidate causes, not an engineering trick for competitive attention; (3) GPT's unidirectional causal mask is equivalent to an explicit do-operation — $\text{do}(\text{future} \not\to \text{past})$ — enforcing a directed acyclic graph (DAG) constraint on the attention space.

This reinterprets the Transformer from a "powerful function approximator" to an **implicit causal inference machine**, and provides a causal language for the interpretability analysis of attention heads.

Experimental validation: extracting attention from GPT-2 on the causal sentence *"Because the storm intensified, the ship finally sank,"* the final-layer average DAG score = **0.810**, significantly above the random baseline of 0.5. This was not fitted — it grew out of the architecture's inductive bias.

This thought experiment currently remains open: can the attention matrix constitute a rigorous Structural Causal Model (SCM)? What is the causal division of labor across multi-head attention? Transformers are locked on Pearl's first and second rungs of the causal ladder — the third rung (counterfactuals) is forever closed to them. What does this mean?

`→ [Chapter 9 thought experiment, original]` &nbsp; `→ [Chapter 9 Bonus: Attention Is Causality](/chapter9/bonus)`

---

**6. CocDo: The Neural do-Operator — Implementing Pearl's Causal Calculus as λ-Calculus**

Chapter 18 defines $\mathsf{do}(X = v)$ as "delete incoming edges, propagate effects." CocDo turns this definition into runnable code: each causal edge is encoded as a COC-dependent Pi type (requiring strictly increasing hierarchy levels, making cycles inexpressible at the type level), the $\mathsf{do}$ operator is implemented as capture-avoiding substitution plus β-reduction, and gradient planning turns "find the optimal intervention value" into Adam optimization over an energy function.

$$v^* = \arg\min_v \sum_j \left(\|E_{\text{next}}[j]\| - y^*_j\right)^2$$

CausalSearch uses the Reasoning Kingdom's own chapters as a causal knowledge graph, employing Pearl's three-step method (abduction → action → prediction) for retrieval, continuously discovering cross-chapter causal chains that vector RAG misses.

`→ [github.com/lizixi-0x2F/CocDo](https://github.com/lizixi-0x2F/CocDo)` &nbsp; `→ [Causal Inference Bonus Chapter](/volume1/chapterbonous/)`

---

## What You Will See

This book is divided into two volumes, logically mirroring each other: the upper volume provides intuition; the lower volume provides foundations. They can be read independently, but only together do they reveal the full picture.

---

### Upper Volume: The Historical Evolution of Reasoning (Chapters 1–13)

The upper volume follows the thread of history, unfolding in a **problem-driven** way — each chapter begins with an unsettling question and follows humanity's historical footsteps in attempting to answer it.

**Part One: The Origins of Reasoning (Chapters 1–6)**

Starting from the second law of thermodynamics, understanding why reasoning is a necessity for survival. We will see how symbolic systems rose and collapsed, how vector spaces redefined "understanding," how the manifold hypothesis explains the hidden order of high-dimensional data, and why statistical correlation can never equal causal reasoning.

**Part Two: The Mechanisms of Reasoning (Chapters 7–11)**

A deep dive into the core mechanisms of AI reasoning. P vs NP reveals the computational asymmetry of the universe; heuristic algorithms sign a contract between "roughly right" and "exactly correct"; the Transformer reconstructs the infrastructure of reasoning through attention mechanisms; MCTS searches for optimal paths in uncertainty; the Collins optimizer touches the physical limits of efficient reasoning.

**Part Three: The Boundaries of Reasoning (Chapters 12–13)**

The Yonglin Formula reveals the essential incompleteness of AI reasoning. Gödel's theorems, the halting problem, and meta-level fracture together sketch the map of the Reasoning Kingdom. Boundaries are not endpoints — they are the starting points of design.

**Bonus Chapters**
- **Chapter 9 Bonus: Attention Is Causality** — deriving the mathematical structure of Self-Attention from causal modeling, revealing the Transformer's nature as an implicit causal inference machine
- **Chapter 13 Bonus: The Hidden Thread** — the concealed structure of the upper volume's thirteen chapters: a causal-logical deductive chain never explicitly stated

---

### Lower Volume: The Formal Deduction of Reasoning (Chapters 14–24)

The lower volume rebuilds from the foundations. It does not follow history, but follows **logical necessity** — each chapter's appearance is compelled by the questions left by the previous one; no chapter is merely "chatting along the way."

The style is rigorous: definitions are precise, arguments are complete, "roughly so" is not accepted. But the narrative is present: before each definition appears, you will know why we need it.

**The Deductive Chain:**

Chapter 14 establishes the foundations of formal systems — propositions, inference rules, axioms, proofs, and the fundamental separation of syntax and semantics. This is the common starting point for all lower-volume chapters.

Chapter 15 asks: is this machine reliable? Gödel's two incompleteness theorems answer precisely, and draw the hard boundary of formal systems' capabilities.

Chapter 16 removes "contraction" from the structural rules: each hypothesis is used exactly once; reasoning becomes resource management. This is linear logic, and the formal foundation of quantum computing and memory safety.

Chapter 17 expands truth values from $\{0, 1\}$ to $[0, 1]$; inference rules become probability propagation. Cox's axioms prove: the only consistent representation of rational belief under uncertainty is probability theory.

Chapter 18 introduces the intervention operator $\mathsf{do}$ into logic, distinguishing three levels: observation, intervention, and counterfactual. This is the formalization of Pearl's Causal Ladder — you cannot derive causality from data unless you are willing to explicitly state structural assumptions.

Chapter 19 equates the depth of derivation trees with computational complexity. P/NP is not about machine speed, but a theorem about the intrinsic structure of problems. The halting problem and Gödel's self-reference meet here again.

Chapter 20 gives "roughly right" a precise mathematical definition: admissibility, consistency, the PAC learning framework. Heuristics are not engineering compromises — they are contracts with formal guarantees.

Chapter 21 views learning as inverse inference: given observed theorems, reverse-engineer the most concise set of axioms. Generalization is compression by another name; Occam's razor is an information-theoretic theorem, not philosophical advice.

Chapter 22: when a reasoning system becomes sufficiently powerful, it begins to reason about propositions concerning itself. Curry-Howard correspondence, fixed-point theorems — this is where there are no answers yet, and where it is worth continuing to walk.

Chapter 23 adopts a different perspective: if reasoning is turned into a dynamical system, can its behavior be described and characterized?

Chapter 24 uses the core tool of modern mathematics — category theory — to explain the convergence properties of reasoning as a dynamical system and to deconstruct attention as implicit causal modeling through a categorical lens. In this chapter, we are not satisfied with formal propositional logic; we go further in applying category theory, this tool called "the mathematics of mathematics," to explain the dynamical-system properties of reasoning and the nature of causal modeling.

The Causal Inference Bonus Chapter is a landing: implementing Chapter 18's do-calculus as a runnable neural SCM. COC type theory makes cycles type errors; the $\mathsf{do}$ operator is implemented as term substitution in λ-calculus; NOTEARS turns DAG constraints into continuous optimization; gradient planning turns "find the optimal intervention" into Adam descent. This is the original work of Li Zixi (Mr. Pallas's Cat) — [CocDo](https://github.com/lizixi-0x2F/CocDo).

---

## How to Use This Book

**If you are a researcher**: each chapter's "Open Questions" section lists unresolved problems; the five original research projects provide directions for further exploration.

**If you are an engineer**: most upper-volume chapters have "Try It Yourself" sections with runnable code experiments. The lower volume's arguments are the theoretical foundation for understanding why certain engineering intuitions are correct and others are mistaken.

**If you are a student**: start from Chapter 1 and read the upper volume sequentially. Finish the upper volume before entering the lower volume — the upper volume gives you questions; the lower volume gives you tools.

**If you're just curious**: jump directly to the chapter that interests you. Upper-volume chapters aim to be as self-contained as possible; for the lower volume, starting from Chapter 14 is recommended, as it is the foundation for all subsequent chapters.

---

## Acknowledgments

Liang Yonglin: a child who pulled the author out of self-pity. The Yonglin Formula is named after him, not because of the formula itself, but because of this: being permitted.

Wang Leyi: my beloved, the weaver of fairy tales in daily life. In the hardest moments of writing this book, he was the one who pulled me back.

The Datawhale team: provided the platform for online publishing, distribution, and dissemination, allowing this book to meet readers in an open-source format.

All explorers of the Reasoning Kingdom: readers who filed Issues, submitted PRs, and sent messages telling me where things weren't clear — your reading allows this book to continue growing.

Thanks to the pioneers who explored at the boundaries of reasoning — Turing, Gödel, Shannon, Pearl — your work is the foundation of this book.

---

Let us enter the Reasoning Kingdom.

# Introduction

When GPT-4 beats human contestants at math competitions, and GPT-6 outdoes experts on PhD-level science reasoning tests, we marvel at the "intelligence" of AI. But almost nobody asks: are these systems really "reasoning"?

The deeper question is: what is reasoning?

Once you start thinking about it seriously, you find that human reasoning might just be an illusion of pattern matching, that logical reasoning might not really exist in the physical world, and that what we call "understanding" might be nothing more than a side product of data compression.

This is not a book about how to use AI. It is a book about **why AI can reason, why it cannot, and what reasoning itself is**.

This book will not hand you answers. But it will walk you to the edge of reasoning — the questions that kept Turing, Gödel, and Shannon up at night.

---

## Why Read This Book

If you are satisfied with "AI is amazing," this book is not for you.

If you want to know:

- Why do large language models (LLMs) — models trained on vast amounts of text to predict the next word — with hundreds of billions of parameters fall apart on simple logic chains?
- Why does chain of thought (CoT) — reasoning step by step and writing the steps down — improve accuracy for a while, then eventually collapse back to the prior?
- Why is P ≠ NP not about "fast vs. slow," but about an asymmetry built into the universe?
- Why does any reasoning system that is powerful enough contain problems it cannot solve?

Then this book was written for you.

We will not stop at the surface of "what AI can do." We'll dig into the underlying machinery of "why AI can, and why it cannot." Volume 1 builds your intuition with historical narratives and runnable experiments. Volume 2 hammers that intuition into solid ground with a strict formal language.

---

## This Is a Construction, Not a Survey

The book's story is built around six original research projects. They are not summaries of what others did. They are exploratory constructions the author made to understand the nature of reasoning:

**1. OpenXOR v2: A Continuous Phase Diagram for an NP Problem** (Chapter 7, with the code that ships with this book)

Classical complexity theory tells us whether a problem "belongs" to P or NP, but it cannot say how hard one specific instance is. The OpenXOR framework breaks through that limit: it turns the solvability of an NP problem from a binary yes/no verdict into a continuous phase diagram — one smooth map that shows where problems tip from solvable to unsolvable, like the way a phase transition is a sharp change in behavior as a parameter crosses a threshold. And none of this is borrowed from someone else: Chapter 7 gives you the complete generator and solver code (predicate checkpoints plus DPLL), so you can reproduce the whole experiment on your own machine.

Instances are random 3-CNF predicate checkpoints: $n$ operator positions, $m$ logical clauses ($\alpha = m/n$ controls the difficulty). The measured solvability probability $\mu(\alpha)$ follows:

$$
\mu(\alpha) = \frac{1}{2}\left(1 - \operatorname{erf}\left(\frac{\alpha - \alpha_c}{\sigma}\right)\right), \quad \alpha_c \approx 4.26,\ \sigma \approx 0.1
$$

where the critical density $\alpha_c \approx 4.26$ is the classic 3-SAT phase transition location — it does not drift with scale. This formula reveals: computability is not binary; it is probabilistic. NP is not a wall. It's a fog with a gradient. At the edge of the fog ($\mu \approx 0.5$), a problem sits in the critical state between solvable and unsolvable.

---

**2. Yonglin's Formula: The Essential Incompleteness of AI Reasoning**

Why do LLMs fail at long chains of reasoning? Not because they lack parameters, but because the object level closes, and the meta level breaks — the model cannot check its own reasoning from inside.

Yonglin's formula proves this: no matter how long the reasoning chain, it eventually collapses back to the prior anchor — the belief the model falls back to, the answer it would give with no evidence:

$$
\lim_{n \to \infty} \Pi^{(n)}(s) = A, \quad \text{but} \quad A \neq A^*
$$

- $\Pi^{(n)}(s)$: the model's reasoning distribution at step $n$
- $A$: the prior anchor of the training data
- $A^*$: the true correct answer

At the object level (generating the reasoning chain) the model can run self-consistently. At the meta level (checking whether the reasoning is right) it cannot step outside the limits of its own parameters. This aligns with the judgment behind the halting problem: a strong enough system inevitably produces its own blind spots — but the two mechanisms stand on their own, independently.

The value of CoT is not "the longer, the better." It's that CoT extends the effective reasoning window — the few steps before collapse, which is where the real reasoning happens.

`-> [Original derivation and explanation in Chapter 12]`

---

**3. ADS (Adaptive Dual Search): Making Heuristic Weights Information-Theoretic**

In search and reasoning, how do you balance "follow the heuristic" against "explore on your own" at each step? Traditional methods use a fixed weight $\alpha$. But the optimal $\alpha$ shifts dynamically with the uncertainty of the current state.

ADS makes $\alpha$ information-theoretic: it adapts the search strategy based on the entropy of the current state's action distribution (full definition in Chapter 8 §6 and Chapter 10 §6; complete code in each chapter's appendix):

$$
\alpha_t = -\log\left(1 - \frac{H_t}{H_{\max}}\right)
$$

where $H_t$ is the entropy of the current output distribution and $H_{\max}$ is the maximum entropy of a uniform distribution. When entropy is high (lots of uncertainty), $\alpha_t$ grows and forms an information-theoretic barrier that repels high-entropy states, forcing the search to collapse onto the low-entropy manifold. When entropy is low, $\alpha_t$ approaches zero, and the system trusts the heuristic and moves fast. This makes Yao's MiniMax theorem concrete in adaptive search: in an uncertain environment, you find the best mix of exploring and exploiting.

`-> [Chapter 8 §6 and Chapter 10 §6; ADS: heuristics made information-theoretic]`

---

**4. The Collins Optimizer: Touching the Physical Limit of Compression** (Chapter 11, with the script that ships with this book)

Optimizers like Adam and AdamW keep momentum and second moments for every parameter, so their memory cost is 3 times the parameter count. Can you compress the optimizer's state?

Collins reaches $O(1)$ state compression through randomization, at a safe compression ratio $c_{\mathrm{opt}} \approx 34$:

$$
c_{\mathrm{opt}} = \frac{d}{\log_2(d/\delta)}
$$

where $d$ is the parameter dimension and $\delta$ is the error tolerance. This limit comes from rate-distortion theory in information theory: you cannot compress without losing information, no matter how hard you try.

Experiment: on Yi-34B-Chat, a 34× compression loses less than 2% performance, but a 64× compression collapses. This is not an engineering problem. It's a mathematical boundary. The full derivation and experiments (code in Chapter 11's appendix, `collins_adam.py`) are in Chapter 11, §5–§9.

---

**5. A Causal-Topological Reinterpretation of Self-Attention: A Thought Experiment**

The standard reading of self-attention is an information-retrieval analogy (Query-Key-Value). But if you start from causal modeling instead, you can derive the same mathematical structure — and give it much deeper meaning. This is not a proven theorem. It is a conjecture still looking for a precise statement.

Let the projection vectors of position $i$ (the effect) and position $j$ (the cause) be $q_i = W_Q x_i$ (row projection / effect modeling) and $k_j = W_K x_j$ (column projection / cause modeling). Their outer product encodes the complete structure of the causal hypothesis:

$$
\mathcal{C}_{ij} = q_i \otimes k_j \in \mathbb{R}^{d_k \times d_k}
$$

Summing over the shared dimension (an Einstein summation, that is, taking the trace) gives a scalar for the strength of the causal hypothesis:

$$
A_{ij} = \mathrm{tr}(\mathcal{C}_{ij}) = q_i \cdot k_j
$$

Then a softmax over the candidate causes gives the causal posterior distribution — the standard attention matrix.

This derivation reveals three things: (1) $W_Q \neq W_K$ is not an engineering choice; it is the unavoidable encoding of causal asymmetry. (2) The softmax is a Bayesian posterior normalization over candidate causes, not an engineering trick for attention competition. (3) GPT's one-directional causal mask is equivalent to an explicit do operation — $\operatorname{do}(\text{future} \not\to \text{past})$ — forcing a directed acyclic graph (DAG) constraint onto attention space.

This reinterprets the Transformer from "powerful function fitter" into an implicit causal inference machine, and it gives the interpretability analysis of attention heads a causal language.

Experiment: with GPT-2, on the causal sentence "Because the storm intensified, the ship finally sank.", the last layer's average DAG score is 0.810, well above the random baseline of 0.5. This did not come from fitting. It grew out of the architecture's inductive bias.

This thought experiment is still open: can the attention matrix form a strict structural causal model (SCM)? What is the causal division of labor among multi-head attention? The Transformer is locked onto the first two rungs of Pearl's ladder of causation. What does it mean that the third rung — counterfactuals — is permanently closed to it?

---

**6. Gradient-Planning Tic-Tac-Toe: When Search Becomes Gradient Descent** (Chapter 10 §8, with the implementation that ships with this book)

Encode the board state as an embedding vector and the board rules as a causal graph, and search becomes gradient descent in embedding space. This chapter's script (complete code in Chapter 10's appendix) uses only NumPy: a "causal model" of gradient planning battles an MCTS tree search, and the result is: first player never loses, second player never wins. The result alone is a story: a continuous heuristic only does what it has learned.

---

## What You'll See

The book comes in two volumes that mirror each other logically. Volume 1 gives intuition. Volume 2 gives foundations. You can read them independently; together, you get the full picture.

---

### Volume 1: The Historical Evolution of Reasoning (Chapters 1–13)

Volume 1 moves along the thread of history and unfolds problem first — every chapter starts with an uncomfortable question and then walks the historical trail humans took to answer it.

**Part 1: The Origins of Reasoning (Chapters 1–6)**

We start from the second law of thermodynamics and understand why reasoning is a survival need. We will see how symbolic systems rose and collapsed, how vector spaces redefined "understanding," how the manifold hypothesis explains the hidden order in high-dimensional data, and why statistical correlation will never equal causal reasoning.

**Part 2: The Machinery of Reasoning (Chapters 7–11)**

Deep into the core machinery of AI reasoning. P vs NP exposes the computational asymmetry of the universe. Heuristics sign a contract between "roughly right" and "exactly right." The Transformer rebuilds the infrastructure of reasoning with attention. MCTS searches the best path under uncertainty. The Collins optimizer touches the physical limits of efficient reasoning.

**Part 3: The Boundaries of Reasoning (Chapters 12–13)**

Yonglin's formula reveals the essential incompleteness of AI reasoning. Gödel's theorem, the halting problem, and the meta-level break together sketch the map of the Reasoning Kingdom. The boundary is not the end of design. It is where design begins.

**Bonus Chapters**

- **Chapter 9 Bonus: Attention Is Causality** — deriving the mathematical structure of self-attention from causal modeling, exposing the Transformer as an implicit causal inference machine
- **Chapter 13 Bonus: The Hidden Thread** — the hidden structure of the thirteen chapters: a chain of causal logic deduction that was never stated out loud

---

### Volume 2: The Formal Deduction of Reasoning (Chapters 14–23)

Volume 2 rebuilds from the ground up. It does not follow history. It follows logical necessity — every chapter is forced into existence by the problem left by the previous one. No chapter is a "casual side talk."

The style is strict: definitions are exact, arguments are complete, and "roughly speaking" is not accepted. But the narrative is present: before each definition appears, you will know why we need it.

**The deduction chain:**

Chapter 14 lays the foundation of formal systems — propositions, inference rules, axioms, proofs, and the fundamental separation between syntax and semantics. This is the shared starting point for every chapter in Volume 2.

Chapter 15 asks: is this machine reliable? Gödel's two incompleteness theorems answer exactly that, and draw the hard boundary of what a formal system can do.

Chapter 16 takes "contraction" out of the structural rules: each hypothesis is used exactly once, and reasoning becomes resource management. This is linear logic, and it is also the formal foundation of quantum computing and memory safety.

Chapter 17 expands truth from $\{0, 1\}$ to $[0, 1]$, and inference rules become probability propagation. Cox's theorem proves: the only compatible representation of rational belief under uncertainty is probability theory.

Chapter 18 introduces the do operator $\mathsf{do}$ into logic, separating observation, intervention, and counterfactual. This is the formalization of Pearl's ladder of causation — you cannot derive causality from data unless you are willing to state structural assumptions out loud.

Chapter 19 equates the depth of a derivation tree with computational complexity. P/NP is not a theorem about machine speed; it is a theorem about the inner structure of problems. The halting problem and Gödel's self-reference meet again here.

Chapter 20 gives "roughly right" a precise mathematical definition: admissibility, consistency, and the PAC learning framework. Heuristics are not an engineering compromise. They are a contract with formal guarantees.

Chapter 21 treats learning as inverse inference: given the observed theorems, work backward to the simplest set of axioms. Generalization is another word for compression, and Occam's razor is an information-theoretic theorem, not philosophical advice.

Chapter 22: when a reasoning system is powerful enough, it starts reasoning about propositions about itself. Curry-Howard correspondence, fixed-point theorems — this is where no one has an answer yet, and where it is still worth walking on.

Chapter 23 returns to Yonglin convergence and computes how fast it happens. The engine is still Banach's fixed-point theorem and the energy surface is still the KL divergence — but now we read off the engine's power. The tangent Hessian has two endpoint eigenvalues, and from them comes the exact contraction rate $\rho^*=(\kappa_T-1)/(\kappa_T+1)$: the slope of the semi-log error curve, the number of steps needed to reach a given accuracy, and a criterion for telling real emergence from a change of readout. This is the [Yonglin Corollary](/en/volume2/chapter23/).

Chapter 9's bonus chapter made one attempt: reading the attention matrix as a soft causal adjacency matrix — the softmax is the posterior distribution over candidate causes, and attention is a causal edge. That reading translates the do operator of Chapter 18 from "a matrix operation" into "a causal intervention," and it puts out a prediction you can test numerically: intervene on an attention path, and the model's output should change along the causal path. This is one exploratory construction by Zixi Li (Professor Pallas).

---

## How to Use This Book

If you're a researcher: the "Open Questions" part of each chapter lists open problems, and the six original research projects point to directions worth digging deeper.

If you're an engineer: most Volume 1 chapters have a "Try It Yourself" section with runnable code experiments. Volume 2's arguments are the theoretical basis for understanding why some engineering instincts are right and some are wrong.

If you're a student: read Volume 1 in order, starting from Chapter 1. Then move into Volume 2 — Volume 1 gives you the questions, Volume 2 gives you the tools.

If you're just curious: jump straight to the chapters you're interested in. Volume 1 chapters try to be self-contained. In Volume 2, start from Chapter 14, because it is the foundation for every chapter after it.

---

## Acknowledgments

Liang Yonglin: a child who pulled the author out of self-pity. Yonglin's formula is named for him — not because of that formula, but because it was allowed.

Wang Leyi: my love, the fairy-tale weaver of everyday life. When this book hit its hardest moments, he was the one who pulled me back.

All the explorers of the Reasoning Kingdom: the readers who filed issues, sent pull requests, and wrote in to say where things were unclear — your reading keeps this book growing.

Thanks to the pioneers who explored the edge of reasoning — Turing, Gödel, Shannon, Pearl. Your work is the foundation of this book.

Online edition: thanks to Datawhale, the open-source community that hosts and maintains the website version of this book.

---

Let's enter the Reasoning Kingdom.

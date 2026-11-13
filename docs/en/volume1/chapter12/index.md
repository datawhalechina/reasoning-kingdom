# Chapter 12: Implicit Reasoning: The Internal Monologue of a Neural Network

> What happens before the model outputs its first token?

<div class="center">

------------------------------------------------------------------------

</div>

The previous four chapters formed a spectrum of search. Chapter 8: explicit search on a discrete graph (A\*, ADS). Chapter 9: soft search over a manifold of hidden states, performed with an $S \times S$ matrix (Transformer). Chapter 10: building a map by random simulation in the wilderness, with no map in hand (MCTS). Chapter 11: every one of those forms of search carries a physical bill — the $O(N^2)$ attention matrix, the exponential state space, storage requirements fixed by a deterministic lower bound.

But they all share one premise: search is explicit. You build a tree. You compute a matrix. You run N simulations. The model searches "outside".

Now a trickier question: what does the model do "inside"? Before it outputs any token, before it explicitly "searches" — what happens in its hidden layers? Is a neural network's forward pass itself — those $L$ Euler steps — already a kind of search? If so, what space does it search? What constrains it? Where does it stop?

This chapter answers those questions. The core of the answer is Yonglin's formula: $\lim_{n\to\infty}\Pi^{(n)}(s)=A\neq A^*$. Whether the search is explicit (the outer CoT loop) or implicit (the Euler steps of the forward pass), the limit of reasoning is the statistical bias of the training data, not the true answer. This is not an engineering limit; it is a structural necessity of any reasoning system that performs a contraction mapping on a belief space.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. Thinking Inside the Black Box

In 2022, Wei and colleagues found something surprising: if you ask a large language model (LLM — a model trained on vast text to predict the next word) to "say its reasoning out loud" before giving the answer, its accuracy jumps sharply.

That technique is chain of thought (CoT) — reasoning step by step, and writing the steps down.

:::details Chain of thought (CoT): why does saying the steps make the model smarter? (Previous work: Wei et al., 2022)
**Chain-of-Thought (CoT) prompting** is a trick discovered in 2022 by Wei and colleagues at Google Brain: add "let's think step by step" after the question, or show the reasoning steps in the few-shot examples, and a model's accuracy on complex reasoning jumps.

**Why does it work?** The mainstream explanation:

1. **A bigger compute budget**: generating each token costs another step of computation. Making the model write out intermediate steps is like letting it scribble on scratch paper instead of forcing it to calculate in its head.
2. **Each intermediate output becomes new input**: every step the model writes gets appended to the context and becomes the input to the next step. This splits a hard problem into simpler sub-problems.

**An important limit**: CoT does almost nothing for small models; it only emerges above roughly 100B parameters. That means it depends on some underlying ability that only large models have.

This phenomenon leaves a hook for the core question of this chapter: is the model actually "reasoning", or is it "matching a format"?
:::

For example, the question: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls, each can has 3 balls. How many balls does he have now?"

**Direct answer**:

    Model output: 11 balls
    Accuracy: 17%

**CoT prompt**: "Let's think step by step"

    Model output:
    Roger had 5 balls.
    He bought 2 cans with 3 balls each, so that's 2×3 = 6 balls.
    That's 5+6 = 11 balls in total.
    Answer: 11 balls
    Accuracy: 78%

Accuracy jumped from 17% to 78%. The gain did not come from more parameters or more training data. It came from making the model write its intermediate steps down.

That raises a deep question: what happens inside the model before it "says" the reasoning out loud?

:::details A Pause
17% to 78%. The act of stating the reasoning steps, by itself, changed the computation.

But wait — same model, same weights, same parameter count. The only thing that changed: it was asked to write the intermediate steps out.

What does that mean?

One explanation is: the model's ability was there all along, and CoT simply "unlocked" it — giving the model a chance to use more steps of computation.

The other explanation is: the model is not reasoning at all; it is matching the "format of reasoning steps". When it produces "Roger had 5 balls", that intermediate output becomes the input for the next step, turning the question into a shape that statistical pattern-matching handles much better.

For most cases, these two explanations make the same prediction. But they disagree completely about what the model is.

Which one do you think is right? More importantly: is there an experiment that tells them apart?

Put that question on the shelf for now.
:::

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Hidden Layers: The Darkroom of Reasoning

A network's forward pass is a layer-by-layer transformation:

$$
h_0 = x \quad \to \quad h_1 = f_1(h_0) \quad \to \quad h_2 = f_2(h_1) \quad \to \quad \cdots \quad \to \quad y = f_L(h_{L-1})
$$

Each layer $h_l$ is a hidden representation — the input's internal encoding at layer $l$.

What are these hidden layers doing?

One intuition: shallow layers extract low-level features (edges, textures), deep layers extract high-level features (objects, concepts). But in reasoning tasks, the hidden layers do something subtler.

In 2022, Olsson and colleagues studied the induction heads of GPT models — attention heads that specifically do "see a pattern, then predict what comes next".

For instance, take the input sequence: "A B C A B ?"

An induction head notices that the "A B" pattern repeats, and predicts the next token is "C".

This suggests: the hidden layers maintain something like a working memory — the model remembers earlier patterns and uses that memory to infer the future.

But how deep is that "working memory"? How complex a reasoning can it support?

<div class="center">

------------------------------------------------------------------------

</div>

## 3. What Really Happens in the Hidden Layers

By now we can state the question more precisely.

CoT works — not necessarily because the model suddenly learned new logical rules, but because it unrolls what used to be compressed inside a single forward pass into a string of intermediate states it can write down. In other words, the explicit reasoning chain is only the surface; what really happens first is the state evolution in the hidden layers.

That is the core question this chapter chases: how much work do the hidden layers do before the answer is spoken?

The most natural intuition: a forward pass is already an internal monologue unfolding layer by layer. The input does not turn into the answer in one step. It is first turned into some intermediate representation, then reorganized, compressed, amplified, and only at the end projected onto the output space. Written as a formula:

$$
h_0 = x \quad \to \quad h_1 = f_1(h_0) \quad \to \quad h_2 = f_2(h_1) \quad \to \quad \cdots \quad \to \quad y = f_L(h_{L-1})
$$

Each $h_l$ here is not the answer itself; it is an internal state that exists before the answer forms. It is neither the raw input nor the final output, but a "judgment still taking shape" somewhere in between.

On perception tasks, we often say shallow layers spot edges and deep layers spot objects. In reasoning tasks, the layering looks more like this: shallow layers first rewrite the problem into coordinates the model can handle, and deep layers integrate relations inside those coordinates.

That is also why Olsson and colleagues' 2022 study of GPT matters so much. They found that some attention heads perform an operation close to a "working memory": after seeing a pattern, they keep tracking its continuation. For example, with the input sequence "A B C A B ?", the induction head aligns the earlier "A B" with the later "A B" and so predicts the next token is "C".

The point is not that "the model can complete a sequence" — that is too shallow. What matters is this: the hidden layers are not a static warehouse of features; they maintain an operational internal state. They keep local relations, align distant pieces, and carry previously seen structures into the present moment to keep computing. That is already close to what we ordinarily call "intermediate reasoning".

So the real question is not "does the model have thoughts?" but: how far can this internal state evolution actually go?

Can it support genuine multi-step reasoning? Can it stay consistent when the chain grows long? Can it carry out stable logical propagation inside itself, without ever writing a step down?

Yonglin's formula, later in this chapter, gives a less cheerful answer: it can go a while, but not infinitely far; it can advance at the object level, but cannot certify itself at the meta level; it can form a temporary reasoning window, but in the end it gets pulled back to the prior anchor.

So it is not enough to say CoT "taught the model to say what it thinks". The more accurate statement is: CoT temporarily extends how long the internal monologue in the hidden layers stays useful.

That leads us to the next step: if the hidden layers really are unrolling an internal monologue, then when the reasoning chain gets longer, where does that monologue start to distort?

<div class="center">

------------------------------------------------------------------------

</div>

## 4. CoT: Making Implicit Reasoning Explicit

What does CoT do? One explanation: CoT forces the model to make implicit reasoning explicit.

Usually, a model's reasoning happens in the hidden layers — a single forward pass from input to output. But that process is "compressed": all the reasoning steps are squeezed into a limited stack of layers.

CoT buys the model more "compute time" by generating intermediate tokens:

- Each generated token costs one more full forward pass
- A reasoning chain of 10 tokens costs 10 forward passes
- That gives the model 10× the compute depth

This is like a human's "slow thinking": when a problem is hard, we don't answer right away — we write the intermediate steps on paper and move forward step by step.

But CoT has a subtle problem: is it really reasoning, or is it performing reasoning?

<div class="center">

------------------------------------------------------------------------

</div>

## 5. Yonglin's Formula: Where Does the Prior Come From?

Yonglin's formula exposes a fundamental limit of CoT: no matter how long the reasoning chain, in the end it converges back to the prior anchor.

Here is Yonglin's formula:

Notation first: this section gives a phenomenological statement of Yonglin's formula; the strict derivation and the full theorem are in Section 7 of this chapter, "The Strict Derivation of the Yonglin Limit" (contraction mapping + chain of Banach fixed points). Both notations refer to the same object: the $\Pi^{(n)}(s)$ here is the belief distribution $p_t$ of Section 7, and $n$ is $t$.

$$
\lim_{n \to \infty} \Pi^{(n)}(s) = A, \quad \text{but} \quad A \neq A^*
$$

Before we can read this formula, we need to know three things: what $\Pi^{(n)}(s)$ is, what $A$ is, and what $A^*$ is.

### Where Does the Prior Anchor $A$ Come From?

The prior anchor is not a mysterious thing. It is the statistical bias of the training data.

Imagine training a model on a dataset with 50% positive examples (the chain is complete, the answer is "yes") and 50% negative examples (the chain is broken, the answer is "no").

Given that statistical structure in the training set, the model learns a "default leaning": if I don't know how to answer, I guess 50/50.

That default leaning is the prior anchor $A$ — the belief the model falls back to: the answer it would give with no evidence at all.

More precisely:

$$
A = P_{\text{train}}(Y)
$$

— the label marginal of the training set.

If the training set is 60% positive, $A$ leans toward "yes" — about 0.6. If the training set is balanced, $A$ is 0.5.

The prior anchor is a statistical bias the model "soaks up" from the data. It is not an error; it is the model's best guess about the "average state of the world".

### What Is the Inference Distribution $\Pi^{(n)}(s)$?

$\Pi^{(n)}(s)$ is the model's probability distribution over the answer after $n$ steps of reasoning.

- $n=0$: the model has not started reasoning yet; the distribution is just the prior $A$
- $n=1$: the model has seen the first piece of evidence and updated its distribution
- $n=10$: the model has processed 10 reasoning steps
- $n\to\infty$: the limit after infinitely much reasoning

Yonglin's formula says this: even if you let the model reason infinitely many steps, its distribution ends up converging back to $A$ — not to the true answer $A^*$.

### What Is $A^*$?

$A^*$ is the distribution of the true correct answer.

If the chain is complete (A>B>C>…>Z all hold), then the answer to "A>Z" is "yes" with 100% certainty. So $A^*=1.0$ — a distribution with certainty 1.

But the model's limit is $A=0.5$ (the prior bias), not $A^*=1.0$.

That is what $A\neq A^*$ means: the limit of the model's reasoning is not the correct answer.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. The Philosophical Meaning of Implicit Reasoning

Yonglin's formula exposes a deep fact: reasoning incompleteness is not a bug; it is a feature.

Any sufficiently powerful reasoning system meets a meta-level break:

- it can produce reasoning chains at the object level
- but it cannot verify the chains' correctness at the meta level
- and in the end it can only fall back to the prior anchor

This is not a problem of model scale. Even GPT-4o with 200B parameters still converges — it's just that the convergence point may sit closer to the real distribution (the prior anchor $A$ is closer to $A^*$), never exactly on it.

The value of CoT is that it pushes $t_{\text{eff}}$ forward along the direction of $\rho$: the smaller $\rho$, the faster the compression and the longer the window — but it cannot change the endpoint $A$.

Seen this way, implicit reasoning (the internal computation in the hidden layers) and explicit reasoning (CoT's token generation) are the same thing in essence: both try to get as close as possible to the right answer within a finite depth of computation.

The difference:

- Implicit reasoning: compressed into one forward pass — fast but shallow
- Explicit reasoning: unrolled into many forward passes — slow but deep

Both are bound by Yonglin's formula. Both converge.

Yonglin's formula gives the inner boundary of reasoning. The next chapter is the last one: together, these boundaries tell us what the map of the Reasoning Kingdom looks like.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. The Strict Derivation of the Yonglin Limit: From Fixed Points to Contraction Mappings

### 1. Start with the Simplest Idea of a Fixed Point

Fixed point: the simplest definition. For a function $f: X \to X$, if there exists $x^* \in X$ such that $f(x^*) = x^*$, then $x^*$ is called a fixed point of $f$.

The definition looks trivial, but it hides a deep piece of philosophy: self-reference. The system's output equals its input; the cause equals the effect.

### 2. Iterating a Sequence: Discrete Dynamical Systems

Consider a sequence $\{x_t\}_{t=0}^\infty$ defined by the recursion:

$$
x_{t+1} = F(x_t), \quad t = 0, 1, 2, \dots
$$

This is a discrete dynamical system. We want to know: what happens to $x_t$ as $t\to\infty$?

### 2.1 A Concrete Example

The simplest linear case: $F(x) = ax + b$.

- Fixed point equation: $x^* = ax^* + b \Rightarrow x^* = \frac{b}{1-a}$ (if $a \neq 1$)
- Iteration: $x_{t+1} = ax_t + b$
- Solution: $x_t = a^t x_0 + \frac{b(1-a^t)}{1-a}$
- Convergence: when $|a| < 1$, $\lim_{t\to\infty} x_t = x^*$

That $|a| < 1$ is the first shape of contractiveness.

### 3. Setting Up the Belief Space

In a reasoning system, $x_t$ is not a real number; it is a belief distribution — a probability vector over the answer space.

Let $\mathcal{P} = \{ p \in \mathbb{R}^n : p_i \geq 0,\ \sum_i p_i = 1 \}$ be the probability simplex.

:::details From {0,1} to [0,1]: why should beliefs be quantitative?
In classical logic, a proposition is either true or false. "It will rain tomorrow" is 1 or 0. The state space is a discrete set of vertices.

But you don't know which one is the truth. All you can know is your degree of confidence.

The Bayesian move: replace "the state of the world" with "your cognitive state about the world".

- Classical logic: $s \in \{0, 1\}^n$; each proposition is true or false
- Bayesian belief: $p \in [0, 1]^n$; each proposition carries a confidence

That 0.7 does not mean "the world has a 70% chance of rain"; it means you, the cognitive subject, have confidence 0.7 in that proposition. The world itself is still 0 or 1 — what is fuzzy is your knowledge, not the world.

Add the normalization constraint $\sum_i p_i = 1$ (your confidence must sum to 100%), and you get the probability simplex.
:::

:::details Why is the simplex's dimension n-1 and not n?
There are $n$ answer options, and the belief vector $p = (p_1, p_2, \ldots, p_n)$ looks $n$-dimensional.

But there is a constraint: $p_1 + p_2 + \cdots + p_n = 1$.

This constraint removes one degree of freedom. You only need to freely choose the first $n-1$ components; the last one is then fully determined:

$$p_n = 1 - p_1 - p_2 - \cdots - p_{n-1}$$

Analogy: points on a line are 1-dimensional, but if we constrain $x + y = 1$, the original 2-dimensional plane gets sliced into a 1-dimensional segment. $n$ variables plus one equality constraint: the degrees of freedom are $n-1$.

Concretely:

- $n=2$ (pick one of two): the simplex is a line segment in $[0, 1]$ — 1 dimension
- $n=3$ (pick one of three): the simplex is an equilateral triangle in the plane — 2 dimensions
- $n=4$ (pick one of four): the simplex is a regular tetrahedron in space — 3 dimensions

Each vertex corresponds to "being completely sure about some answer" (the states of classical logic). Points in the interior are Bayesian territory — a continuous spectrum of uncertainty.
:::

:::details The geometric essence of the simplex: what hides in the definition of a convex set?
The simplex is not just a geometric shape; it is the smallest unit of convexity.

**Definition of a convex set**: a set $C$ is convex iff for any $x_1, \ldots, x_k \in C$, as long as $\theta_i \geq 0$ and $\sum_i \theta_i = 1$, we have $\sum_i \theta_i x_i \in C$.

Look at that condition: $\theta_i \geq 0$, $\sum_i \theta_i = 1$. Those are exactly the coordinate definition of the simplex $\Delta_{k-1}$.

So the definition of a convex set can be reread as: $C$ is convex iff the affine image of a simplex on any finite point set of $C$ still lies inside $C$. The essence of convexity is closure under linear interpolation of simplex coordinates.

The simplex is the "just-enough convexity container": $n+1$ affinely independent points span an $n$-dimensional space, with none to spare.

---

The simplex method of linear programming shares the same intuition. Linear programming minimizes a linear objective over a convex polytope (the feasible region). The key fact: on a convex polytope, a linear function must attain its optimum at a vertex.

The simplex method's strategy: leave from a vertex, walk along an edge to a neighboring vertex, as long as the objective is decreasing, until no neighbor improves it. Never enter the interior; only walk the boundary, because the optimum is hiding in a corner.

Two simplexes, the same stage, opposite strategies. That is not a coincidence; it is decided by the nature of the objective function:

| Scenario | Objective function | Where the optimum lies | Strategy |
|---|---|---|---|
| Simplex method (linear programming) | Linear | Vertex | Greedy walk along the edges, never entering the interior |
| Belief update (Bayesian) | Strictly convex (KL divergence, cross-entropy) | Interior | Gradient descent, going deep into the interior |
| Marginal analysis (economics) | Strictly concave (diminishing marginal returns) | Interior | The equilibrium is inside, not in a corner |

A linear function has no curvature, so its extremes can only appear on the boundary — the simplex method exploits exactly that, squeezing the search onto the finite set of vertices. A strictly convex function has a unique interior optimum, and gradient descent pulls you toward the interior.

The vertices of the probability simplex $\mathcal{P}$ are certain beliefs (the 0/1 world of classical logic); the interior is the continuous spectrum of uncertainty (Bayesian territory). Reasoning is finding where you should stand on this simplex — and that spot is almost always in the interior.
:::

The inference operator $F: \mathcal{P} \to \mathcal{P}$ maps the current belief to the updated belief.

### 4. KL Divergence: A Natural Measure of Distance Between Beliefs

For two distributions $p, q \in \mathcal{P}$, the KL divergence:

$$
D_{\text{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

Properties:

1. $D_{\text{KL}}(p \| q) \geq 0$ (Gibbs' inequality)
2. $D_{\text{KL}}(p \| q) = 0 \iff p = q$
3. Convexity: $D_{\text{KL}}(p \| q)$ is jointly convex in $(p, q)$

But the KL divergence is not a metric: it is not symmetric and does not satisfy the triangle inequality. So KL cannot carry the distance Banach's theorem needs. Its role in the proof below is the energy function and its curvature; the "distance" the Banach step needs is carried by the Euclidean norm on the tangent space (Steps 7-8).

### 5. The Key Insight: The Concrete Form of the Inference Operator

The contraction property is not assumed; it follows from the concrete construction of the inference operator.

### 5.1 The Gradient Descent View

For a trained neural network, a reasoning step can be seen as gradient descent on an energy surface.

Let the energy be $E: \mathcal{P} \to \mathbb{R}$; the inference operator:

$$
F(p) = \operatorname{proj}_{\mathcal{P}}\left( p - \eta \nabla E(p) \right)
$$

where $\operatorname{proj}_{\mathcal{P}}$ is the projection onto the probability simplex and $\eta > 0$ is the step size.

### 5.2 Discretizing the Euler Iteration

The continuous gradient flow: $\frac{dp}{dt} = -\nabla E(p)$

Euler discretization (step $\Delta t = \eta$):

$$
p_{t+1} = p_t - \eta \nabla E(p_t)
$$

Add the projection that keeps probability:

$$
p_{t+1} = \operatorname{proj}_{\mathcal{P}}\left( p_t - \eta \nabla E(p_t) \right)
$$

This is exactly the form of $F$.

:::details What is the Euler step here?
The continuous gradient flow $\frac{dp}{dt} = -\nabla E(p)$ describes a belief distribution flowing continuously downhill on the energy surface. But a computer can only take one small step at a time; it cannot be truly "continuous".

Euler's method cuts that continuous flow into discrete small steps:

$$p_{t+1} = p_t - \eta \nabla E(p_t)$$

Meaning: the current belief is $p_t$; the downhill direction of the energy there is $-\nabla E(p_t)$; walk one step $\eta$ along that direction, and you get the new belief $p_{t+1}$.

This has exactly the same structure as A\* search: you are at the current node $s_t$, you measure the cost of each direction, and you take one step in the best one. The only difference is:

- A\*'s stage is a discrete graph of nodes; it walks the direction of minimum cost
- This stage is the probability simplex; it walks the direction of the steepest energy descent

The $\operatorname{proj}_{\mathcal{P}}$ is added because after one step, the result may wander out of the probability simplex (some component turns negative), and it must be projected back to keep the constraint "all components nonnegative and sum to 1".

The error of an Euler step: the bigger the step size $\eta$, the farther each step goes, but the more it deviates from the true continuous trajectory. Too big a step and it diverges; too small and it converges slowly. This is the essence of learning-rate tuning.
:::

### 6. KL Divergence as a Bregman Divergence

The KL divergence can be written as a Bregman divergence. Let $\phi(p) = \sum_i p_i \log p_i$ (negative entropy); then:

$$
D_{\text{KL}}(p \| q) = \phi(p) - \phi(q) - \langle \nabla\phi(q), p-q \rangle
$$

The Bregman divergence obeys a three-point identity:

$$
D_{\text{KL}}(p \| q) = D_{\text{KL}}(p \| r) + D_{\text{KL}}(r \| q) - \langle \nabla\phi(q) - \nabla\phi(r), p-r \rangle
$$

### 7. The Proof of Contractiveness (The Core)

Now we prove: the gradient-descent-form inference operator $F$ is a contraction, under the right conditions.

**Setup.** Take a region $\mathcal{X} \subseteq \mathcal{P}$ that is closed, convex, and contains the anchor $A$ inside the simplex (for example $\mathcal{X}_\varepsilon = \{p: p_i \geq \varepsilon\}$, where $\varepsilon \leq \min_i A_i$); let $\Pi_{\mathcal{X}}$ be the Euclidean orthogonal projection onto $\mathcal{X}$; let the tangent space be $T = \{v: \mathbf{1}^\top v = 0\}$ and $P_T$ the orthogonal projection onto $T$. The inference operator is the Euclidean projected gradient

$$
F(p) = \Pi_{\mathcal{X}}\left( p - \eta P_T \nabla E(p) \right).
$$

Three choices behind this setup, and their reasons:

1. **The metric is the Euclidean norm on the tangent space, not KL.** The KL divergence is asymmetric and fails the triangle inequality, so the simplex is not a complete metric space under KL, and Banach's theorem cannot stand on it. KL is used where it belongs: as the energy function, whose Hessian decides where $F$ goes and how large a step may be (Steps 4 and 6).
2. **The operator is the Euclidean projected gradient.** The theorem below is about this exact operator — gradient descent with Euclidean projection plus the tangent-space correction. It is not automatically transferable to other projections (such as the softmax/Bregman projection used by real models); each projection defines its own system, and this theorem covers this one.
3. **The projection is non-expansive.** Euclidean projection onto a closed convex set never increases distance — this is the single inequality the proof needs (Step 2).

(Why $\mathcal{X}$ is needed: the KL Hessian $\operatorname{diag}(q_i/p_i^2)$ blows up at the boundary, and a spectral bound only exists away from the boundary. Why $P_T$ is needed: the normal component of $\nabla E$ is a gauge freedom — moving along the normal does not change the belief itself.)

**Theorem**: suppose on $\mathcal{X}$ the tangent-space Hessian has a uniform spectral bound $\mu I_T \preccurlyeq P_T \nabla^2 E(p) P_T \preccurlyeq L I_T$ ($\mu, L > 0$). Take a step size $\eta \in (0, 2\mu/L^2)$. Then $F$ is a contraction mapping on $(\mathcal{X}, \|\cdot\|_2)$, with contraction factor

$$
\rho = \sqrt{1 - 2\eta\mu + \eta^2 L^2} < 1.
$$

**Proof**:

**Step 1: translate the spectral bound into two inequalities.** For any $p_1, p_2 \in \mathcal{X}$ (note $p_1 - p_2 \in T$), by the integral mean-value theorem and the spectral bound:

$$
\langle P_T(\nabla E(p_1) - \nabla E(p_2)),\ p_1 - p_2 \rangle \geq \mu \|p_1 - p_2\|_2^2,
$$

$$
\|P_T(\nabla E(p_1) - \nabla E(p_2))\|_2 \leq L \|p_1 - p_2\|_2.
$$

(The lower bound: the spectral lower bound is strong convexity along tangent directions; the upper bound: the spectral upper bound is smoothness along tangent directions. The first inequality uses $p_1 - p_2 \in T$ and the symmetry of $P_T$.)

**Step 2: the projection does not amplify distance.** Euclidean projection onto a closed convex set is non-expansive: $\|\Pi_{\mathcal{X}}(x) - \Pi_{\mathcal{X}}(y)\|_2 \leq \|x-y\|_2$. Therefore

$$
\|F(p_1) - F(p_2)\|_2^2 \leq \left\|(p_1-p_2) - \eta P_T(\nabla E(p_1) - \nabla E(p_2))\right\|_2^2.
$$

**Step 3: expand.**

$$
\begin{aligned}
\|F(p_1) - F(p_2)\|_2^2 &\leq \|p_1-p_2\|_2^2 - 2\eta\langle P_T(\nabla E(p_1) - \nabla E(p_2)),\ p_1-p_2\rangle \\
&\quad + \eta^2\|P_T(\nabla E(p_1) - \nabla E(p_2))\|_2^2 \\
&\leq (1 - 2\eta\mu + \eta^2L^2)\|p_1-p_2\|_2^2 = \rho^2\|p_1-p_2\|_2^2.
\end{aligned}
$$

That is where $2\mu - \eta L^2$ comes from: strong convexity contributes the positive $2\mu$, smoothness contributes the negative $\eta L^2$, and their difference decides how much each step contracts. The step-size upper bound $\eta < 2\mu/L^2$ is exactly the condition that makes $\rho < 1$ ($\rho^2 \geq 1 - \mu^2/L^2 \geq 0$; nonnegativity follows from $\mu \leq L$).

**QED.**

What we get here is a uniform Lipschitz contraction over the whole region: it holds for any two points $p_1, p_2 \in \mathcal{X}$, not just "getting closer and closer to $A$". That is what a contraction mapping means in Banach's sense.

### 8. Applying the Banach Fixed-Point Theorem

Now apply the Banach fixed-point theorem:

Since $F$ is a contraction mapping ($\rho < 1$), and $(\mathcal{X}, \|\cdot\|_2)$ is a closed subset of $\mathbb{R}^n$ that is complete under the Euclidean metric (the setup of Step 7), then:

1. There exists a unique fixed point $A \in \mathcal{X}$ with $F(A) = A$
2. For any initial $p_0 \in \mathcal{X}$, the iteration $p_{t+1} = F(p_t)$ converges to $A$
3. Rate of convergence: $\|p_t - A\|_2 \leq \rho^t \|p_0 - A\|_2$

Note that the metric is the Euclidean norm, not KL. The role of KL divergence here is the energy function — it decides where $F$ goes, and (through the Hessian) how big a step it may take.

### 9. Identifying the Prior Anchor

Training minimizes empirical risk:

$$
\min_{p \in \mathcal{P}} \mathbb{E}_{(x,y) \sim D}[\mathcal{L}(p, y)]
$$

For the cross-entropy loss $\mathcal{L}(p, y) = -\log p_y$, the optimal solution is:

$$
A(y) = P_D(y) = \frac{\text{count of label } y \text{ in the training set}}{\text{total samples}}
$$

The key observation: this $A$ is exactly the fixed point of the inference operator. Because for $E(p) = \mathbb{E}_D[-\log p_y]$,

$$
\nabla_i E(p) = -\frac{q_i}{p_i}, \qquad \nabla E(A) = (-1, \ldots, -1)
$$

$\nabla E(A)$ lies entirely along the normal of the simplex — its projection on the tangent space is zero: $P_T \nabla E(A) = 0$. So

$$
F(A) = \Pi_{\mathcal{X}}(A - \eta \cdot 0) = A,
$$

that is, $A$ is a fixed point of $F$. This is exactly why $P_T$ sits inside the operator: with the normal component cut away, the fixed point is exact. At $A$ the gradient is $(-1, \ldots, -1)$, not the zero vector; the statement to keep is not "the gradient is zero" but "the gradient lies along the normal, and its tangent-space component is zero".

### 10. The Full Statement of the Yonglin Theorem

The Yonglin theorem (strict version): suppose a reasoning system satisfies:

1. **Region and spectral assumptions**: there exists a closed convex region $\mathcal{X} \subseteq \mathcal{P}$ containing the prior anchor $A$ (e.g. $\mathcal{X}_\varepsilon$), and the tangent-space Hessian of the energy $E$ has a uniform spectral bound on $\mathcal{X}$: $\mu I_T \preccurlyeq P_T \nabla^2 E(p) P_T \preccurlyeq L I_T$ ($\mu, L > 0$)
2. **Gradient descent form**: the inference operator $F(p) = \Pi_{\mathcal{X}}(p - \eta P_T \nabla E(p))$, with step size $\eta \in (0, 2\mu/L^2)$
3. **Training consistency**: the minimizer $A = \arg\min_{p \in \mathcal{P}} E(p)$ of $E$ corresponds to the empirical distribution of the training data

Then:

1. $F$ is a contraction mapping on $(\mathcal{X}, \|\cdot\|_2)$, with contraction factor $\rho = \sqrt{1 - 2\eta\mu + \eta^2L^2} < 1$
2. $A$ is the unique fixed point of $F$
3. For any initial belief $p_0 \in \mathcal{X}$, $\lim_{t\to\infty} p_t = A$
4. Convergence exponent: $\|p_t - A\|_2 \leq \rho^t \|p_0 - A\|_2$, and $D_{\text{KL}}(A\|p_t) \leq \kappa\rho^{2t} D_{\text{KL}}(A\|p_0)$, with $\kappa = L/\mu$

If the true answer $A^* \neq A$, then no matter how many reasoning steps the system takes, it can never reach $A^*$.

**Remark (a route that does not even need Banach)**: if for some $\gamma \in (0, 1)$ we have $E(F(p)) \leq \gamma E(p)$ for every $p \in \mathcal{X}$, then the fixed point is unique and $p_t \to A$ without Banach at all: the iteration gives $E(p_t) \leq \gamma^t E(p_0)$; a fixed point $q$ satisfies $E(q) = E(F(q)) \leq \gamma E(q)$, and $\gamma < 1$ forces $E(q) = 0$, i.e. $q = A$; Pinsker's inequality $D_{\text{KL}}(A\|p_t) \geq \frac{1}{2}\|p_t - A\|_1^2$ gives $\|p_t - A\|_1 \leq \sqrt{2\gamma^t E(p_0)} \to 0$. When $\kappa\rho^2 < 1$, this theorem satisfies the assumption directly (take $\gamma = \kappa\rho^2$).

### 10.7 Forward Pass = Reasoning: Euler Steps + Simplex Projection

In the earlier forward pass, Professor Pallas drew a steep hillside on the whiteboard and told Little Pig and Little Seal: "update the parameters along the negative gradient direction, with a suitable step size."

That was how he described training — walking gradient descent in the parameter space $\Theta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

That description is completely correct, but it tells only half the story. The other half: the forward pass is itself reasoning.

#### Tearing a Neural Network's Forward Pass Apart

A forward pass of any Transformer layer:

```
h_0 = x + positional_encoding          # initial state
h_1 = h_0 + Attention(h_0)             # step 1
h_2 = h_1 + FFN(h_1)                   # step 2
h_3 = h_2 + Attention(h_2)             # step 3
h_4 = h_3 + FFN(h_3)                   # step 4
...
h_L = ...                               # step L
y = softmax(h_L)                        # projection
```

Stare at the residual connection — $h_k = h_{k-1} + F(h_{k-1})$. That is Euler iteration:

$$
x_{t+1} = x_t + \eta \cdot v(x_t)
$$

Step size $\eta = 1$, velocity $v = \text{Attention}$ or $\text{FFN}$. All three ingredients are in place.

What about a plain fully connected layer without a residual connection? $h_k = \sigma(W_k h_{k-1} + b_k)$. Split it into "current state + increment":

$$
h_k = h_{k-1} + \underbrace{\left(\sigma(W_k h_{k-1} + b_k) - h_{k-1}\right)}_{\text{velocity } v(h_{k-1})}
$$

Same. Any hidden-state pass that can be written as $h_k = h_{k-1} + \Delta_k$: then $\Delta_k$ is the velocity and the layer is an Euler step. The residual connection just exposes the structure openly — writing $\eta = 1$ and the velocity function $v$ separately.

#### The Three Ingredients of an Euler Step

Whatever the network, as long as hidden states are being passed along, it can be modeled as Euler iteration. Three ingredients:

| Ingredient | General form | Residual connection (explicit) | Dense layer (implicit) |
|---|---|---|---|
| Current state | $x_t$ | $h_{k-1}$ | $h_{k-1}$ |
| Velocity function | $v(x_t)$ | $\text{Attention}(h_{k-1})$ | $\sigma(W_k h_{k-1} + b_k) - h_{k-1}$ |
| Step size | $\eta$ | $1$ (hard-coded) | $1$ (implicit) |
| Next state | $x_{t+1} = x_t + \eta \cdot v(x_t)$ | $h_k = h_{k-1} + \text{Attention}(h_{k-1})$ | $h_k = h_{k-1} + (\sigma(W_k h_{k-1} + b_k) - h_{k-1})$ |

An RNN's hidden-state recursion $h_t = \tanh(W h_{t-1} + U x_t)$ is an Euler step too — the current state is $h_{t-1}$ and the velocity is $\tanh(W h_{t-1} + U x_t) - h_{t-1}$. Mamba's discretized SSM $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ is also an Euler step — velocity $(\bar{A} - I)h_{t-1} + \bar{B} x_t$.

Any neural network's hidden-state pass is a discrete dynamical system. Layers = Euler steps.

#### Forward Pass = Reasoning

So the complete shape of the forward pass:

$$
\underbrace{h_0 \xrightarrow{\text{Euler step}} h_1 \xrightarrow{\text{Euler step}} \cdots \xrightarrow{\text{Euler step}} h_L}_{\text{discrete dynamics on the hidden-state space: } L \text{ Euler steps}} \xrightarrow{\text{softmax}} \underbrace{p}_{\mathcal{P}}
$$

That is reasoning. The hidden state evolves layer by layer over $L$ Euler steps. Each step nudges it slightly; after $L$ steps it lands somewhere at $h_L$. Finally softmax projects that point onto the probability simplex $\mathcal{P}$.

softmax is one concrete implementation of $\operatorname{proj}_{\mathcal{P}}$:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$z \in \mathbb{R}^n$ can be anything; the output $p \in \mathcal{P}$ is legal. This is not a coincidence — softmax is the maximum-likelihood projection of the exponential family, and it naturally maps the logit space onto the probability simplex.

CoT is just doing more rounds: for each generated token, the hidden state walks the $L$ Euler steps again and projects once more.

$$
\text{CoT} = (\text{Euler step}^L \circ \text{softmax}) \circ (\text{Euler step}^L \circ \text{softmax}) \circ \cdots
$$

#### Backpropagation Is a Different Paradigm

Training does not happen in the hidden-state space. Training walks gradient descent in the parameter space $\Theta$:

$$
\theta_{t+1} = \theta_t - \hat{\eta} \nabla_\theta \mathcal{L}(\theta_t)
$$

Backpropagation is just the way to compute $\nabla_\theta \mathcal{L}$ — because $\mathcal{L}$ is an $L$-layer composition:

$$
\mathcal{L}(\theta) = \ell \circ f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
$$

The chain rule runs through the $L$ layers:

$$
\nabla_{\theta_i} \mathcal{L} = J_{f_L} \cdot J_{f_{L-1}} \cdot \cdots \cdot J_{f_{i+1}} \cdot \frac{\partial f_i}{\partial \theta_i}
$$

Each $J_{f_k}$ is the Jacobian of layer $k$. Multiplying back-to-front (matrix-vector, $O(d^2)$) instead of front-to-back (matrix-matrix, $O(d^3)$) — that is all backpropagation is. A computational trick; it creates no new mathematics.

#### Two Paradigms, Not One

The forward pass and the backward pass are not "the same set of equations in different spaces". They are two completely different kinds of dynamics:

| | Forward pass = reasoning | Backward pass = training |
|---|---|---|
| Space | Hidden-state space $\mathcal{H}$ | Parameter space $\Theta$ |
| Type of dynamics | Discrete dynamical system (Euler steps) | Continuous optimization (gradient descent) |
| Iteration formula | $h_k = h_{k-1} + v_k(h_{k-1})$ | $\theta_{t+1} = \theta_t - \hat{\eta} \nabla_\theta \mathcal{L}$ |
| Where velocity comes from | Layer transform $F_k$ (attention, FFN, etc.) | Loss gradient $\nabla_\theta \mathcal{L}$ |
| Number of steps | Fixed ($L$ layers) | Variable (training steps) |
| Endpoint | softmax -> $\mathcal{P}$ | $\theta^*$ (fixed by the training data) |
| What it optimizes? | Nothing — it executes a fixed track | Minimizes the loss function |

The forward pass is not "optimizing" anything. Each layer's velocity function $v_k$ is decided by the trained weights. The forward pass merely executes the track that is already laid — $L$ Euler steps, softmax projection, output the answer.

The backward pass optimizes. It changes the weights $\theta$, and thereby changes the shape of the next forward pass's track. Training makes it steeper ($\mu$ grows) and smoother ($L$ shrinks), so the forward pass's $L$ Euler steps reach the right place faster.

#### The Only Meeting Point: $\operatorname{proj}_{\mathcal{P}}$

The two paradigms meet at softmax.

Training never touches $\mathcal{P}$ — it walks gradient descent in $\Theta$, where $\Theta = \mathbb{R}^d$ is flat and needs no projection. Partial derivatives are subtracted directly; the parameters can take any value.

The forward pass touches $\mathcal{P}$ at its end — the hidden state $h_L$ is projected into a probability distribution by softmax. softmax is part of the neural network's structure, not something bolted on.

Why must softmax = $\operatorname{proj}_{\mathcal{P}}$ be a projection in information geometry, not a simple clipping? Because $\mathcal{P}$ is not a Euclidean space. Consider:

$$
p = (0.99, 0.01)
\quad \text{vs} \quad
q = (0.01, 0.99)
$$

Euclidean distance: $\|p-q\|_2 \approx 1.39$. But the Euclidean distance tells you that $(0.5, 0.5)$ is "about equally far" from both points — which completely fails to reflect the huge cognitive gap between "sure of A" and "sure of B".

KL divergence: $D_{\text{KL}}(p\|q) \approx 4.60$ — the KL distance between two extreme beliefs is far bigger than the one starting from uniform. KL captures the directionality of beliefs: walking from "nearly certain A" to "nearly certain B" is a huge cognitive distance.

So $\operatorname{proj}_{\mathcal{P}}$ is not "clipping into the legal range". It is a projection done in the geometric sense of KL divergence (a Bregman divergence) — keeping beliefs on the simplex manifold, not on the Euclidean plane.

#### The Yonglin Limit Constrains Both Paradigms

Training: gradient descent in $\Theta$ converges to $\theta^*$. $\theta^*$ is fixed by the loss minimizer on the training data — that is, the prior anchor $A = P_D(Y)$.

Reasoning: the Euler steps walk $L$ steps in $\mathcal{H}$, and softmax projects onto $\mathcal{P}$. But if the reasoning chain is stretched (multiple CoT rounds), the hidden state is projected by softmax and sets off again, re-anchoring itself on the simplex at every step. After enough rounds, the hidden-state trajectory is captured by the attractor $A$ — that is the Yonglin limit.

Two paradigms, the same endpoint: $A$. Training encodes $A$ into the weight matrices; reasoning is pulled toward $A$ by the energy surface carved by $A$. $A$ is the statistical bias of the training data, not the true answer $A^*$. However many Euler steps the forward pass takes, however long CoT stretches — the attractor of the hidden-state dynamics is $A$.

#### Two Sentences

Earlier, Professor Pallas said: "update the parameters along the negative gradient direction, with a suitable step size."

That is training. Completely correct.

Add one more sentence: every layer of the forward pass is an Euler step. After the $L$ Euler steps, softmax projects the hidden state onto the simplex. That is reasoning. Training changes the velocity field of the Euler steps; reasoning executes the hidden-state trajectory on that velocity field.

Two paradigms: one optimizes (walking gradient descent in $\Theta$), one executes (walking Euler steps plus $\operatorname{proj}_{\mathcal{P}}$ in $\mathcal{H}$). They meet at softmax. Both end at $A$.

<div class="center">

------------------------------------------------------------------------

</div>

## 12. A Mathematical Statement of the Meta-Level Break

Let $V: \mathcal{P} \to \{0, 1\}$ be a verification function ($V(p) = 1$ means the belief $p$ is correct), whose truth value is defined by the external true answer $A^*$. A meta-layer verifier, if it exists, should be a computable mapping $M$ built from the inference operator $F$ (equivalently, from the energy function $E$) such that for every $p \in \mathcal{P}$:

$$
V(F(p)) = M(F, p) \qquad \forall p \in \mathcal{P}
$$

**Proposition (open problem)**: when $A \neq A^*$, no such $M$ exists.

Why this is an open problem and not a theorem: an early version tried to prove it by arguing "if $V$ is not a convex function of $E$, then $M$ does not exist" — but that argument does not hold: for a fixed $F$, $V \circ F$ is itself a well-defined function; convexity and monotonicity have nothing to do with the existence of $M$. The old proof line mistook "not yet proven" for "already proven".

The real problem is the semantic domain: the definition of $V$ involves $A^*$ (external truth), while the construction of $F$ only involves $A$ (training statistics). In category language: $V$ is a predicate of the external category $\mathcal{R}$, and $F$ is an endofunctor of the internal category $\mathcal{P}$, with no adjoint functor connecting the two. So the rigorous formulation of "is there a computable bridge from $F$ to $V \circ F$" is equivalent to "can the model deduce its own deviation from external truth from its own parameters alone" — and that is currently open.

The intuitive answer is: no. Here the meta-level break shows up as: the model can produce reasoning chains at the object level ($F$ is computable), but their correctness (the truth value of $V$) is defined by the out-of-domain object $A^*$, and $A^*$ never enters the construction of $F$.

<div class="center">

------------------------------------------------------------------------

</div>

## 13. Summary: The Path from Naive to Rigorous

1. **Starting point**: the naive concept of a fixed point $F(x) = x$
2. **Concretization**: the belief space $\mathcal{P}$ and the inference operator $F$
3. **Structural assumption**: $F$ is a discretization of gradient descent (Euler iteration)
4. **Division of labor**: KL divergence handles energy and curvature (Bregman geometry); the metric for Banach is the Euclidean norm on the tangent space
5. **Proof of contractiveness**: tangent-space Hessian spectral bound + non-expansive Euclidean projection $\Rightarrow$ $\|F(p_1) - F(p_2)\|_2 \leq \rho\|p_1 - p_2\|_2$; the KL energy decays exponentially as $\kappa\rho^{2t}$
6. **Applying the theorem**: the Banach fixed-point theorem guarantees existence, uniqueness, and convergence
7. **Physical meaning**: the fixed point $A$ is the statistics of the training data; the convergence speed is decided by the problem's curvature

This derivation avoids "tweaking formulas"; every step carries a clear mathematical reason. What it showcases:

- the bridge between discrete and continuous (Euler iteration)
- the unity of optimization and dynamical systems (gradient descent)
- the fusion of statistics and geometry (KL divergence as a Bregman divergence)

<div class="center">

------------------------------------------------------------------------

</div>

## 14. Direct Evidence: Numerical Verification of the Energy Terrain and the Closed-Form Solution

Step 7's contractiveness proof gave us the Yonglin limit; this section supplies direct evidence for it: implement the inference dynamics numerically on the belief simplex (the gradient flow of the KL energy + Euler steps projected onto the simplex) and check the closed form's four predictions one by one. The whole experiment is reproducible: the full source is `ch12_energy_terrain.py` (the complete code is right below), and the figure is `ch12_energy_terrain.png`.

**Experiment setup.** Three classes; the belief simplex $\Delta^2$. Energy function $E(p) = D_{\text{KL}}(p^* \| p)$, with $p^*$ the target belief / attractor; inference operator $p_{t+1} = \operatorname{proj}_{\Delta}(p_t - \eta \nabla E(p_t))$; the projection is the Euclidean projection onto the probability simplex (standard algorithm — the two implementation details are below). All closed-form quantities are the family defined in Step 7 (contractiveness proof) and Step 10 (Yonglin theorem): $\mu = 1/\max_i p_i$, $L = 1/\min_i p_i$, $\kappa = L/\mu$, $\eta_{\max} = 2\mu/L^2$.

![Figure 1: Numerical verification of the energy terrain and the closed-form solution](/figures/ch12_energy_terrain.png)

**How to read the four panels of Figure 1**:

**(a) The terrain exists.** The contour lines of $E$ form an energy basin on the simplex with $p^*$ at its bottom; inference trajectories started from three different initial beliefs and iterated with step $0.8\eta_{\max}$ all converge to $p^*$ (the star). Reasoning is motion on an energy terrain, not a black box — this is the most direct picture of this chapter's argument.

**(b) Curvature spectrum: closed form = numerical.** Along one reasoning trajectory, the analytic spectrum ($\lambda_{\min} = 1/\max_i p_i$, $\lambda_{\max} = 1/\min_i p_i$, at the fixed point) matches the eigenvalues of the numerically differentiated Hessian point for point, with $\kappa = L/\mu = 7.00 = 0.70/0.10$. Step 7's "the Hessian is $\operatorname{diag}(1/p_i)$" agrees with finite differences — visible directly from this panel, with no "approximately" needed.

**(c) The closed-form landscape of $\eta_{\max}$.** A logarithmic color map of $\eta_{\max}(p) = 2\min_i p_i^2/\max_i p_i$ on the simplex (log color scale from $10^{-6}$ to $10^{-1}$): in the center you may walk with big steps; the closer to the edge, the smaller the steps must be. "The terrain dictates the pace" is not rhetoric — it is this picture.

**(d) Stability boundary: the closed form predicts the numerics exactly.** The numerical instability threshold $\eta^*$ of the local perturbation dynamics (asymptotic criterion, below) versus the closed form $2/\theta_{\max}$ (tangent-space spectrum, below): across 30 random target distributions, $R^2 = 0.9978$ with median error 2.16% — the blue points all cling to the diagonal. The pink triangles are the numerical thresholds of a global scan starting from the uniform prior, sitting always above the conservative bound $2/(L\kappa)$.

Two improvements this experiment made along the way:

**First: the exact local stability bound is the tangent-space spectrum, not the full-space one.** The projection confines the dynamics to the tangent space $\{v: \sum v_i = 0\}$, so the exact instability condition is $\eta < 2/\theta_{\max}$, where $\theta_{\max}$ is the largest eigenvalue of $\operatorname{diag}(1/q_i)$ restricted to the tangent space. For $K = 3$, $\theta_{\max}$ has an analytic expression: it is the largest root of

$$
3q_0q_1q_2\theta^2 - 2(q_0q_1 + q_0q_2 + q_1q_2)\theta + 1 = 0
$$

Step 10's $\eta_{\max} = 2/(L\kappa)$ is still correct, but it is a conservative bound ($\theta_{\max} \leq L$, so $2/\theta_{\max} \geq 2/L = \kappa \cdot \eta_{\max}$). Both families of bounds are visible in one figure: the exact bound (points on the line) and the conservative bound (points always above it).

**Second: the stability criterion must use the asymptotic endpoint, not the maximum over the whole run.** Whether the dynamics are stable is judged by $\|p_T - p^*\| / \|p_0 - p^*\| < 1$ (the asymptotic endpoint distance ratio), not the maximum over the whole process — at moderate step sizes, the nonlinear transient grows first and contracts later, so the full-trajectory maximum misjudges a linearly stable region as unstable (the measured value is a spurious ratio of 0.79; with the asymptotic criterion it is 1.000, matching the closed form).

**Source code** (complete and runnable; when run, it saves the figure as `ch12_energy_terrain.png` in the same directory):

```python
"""
Energy Landscape of Reasoning: Closed-Form Curvature and Safe Step Size
========================================================================

Demonstration accompanying the Yonglin Limit derivation.

Setup: belief simplex Delta^2 (3 classes). The reasoning energy is
    E(p) = D_KL(p* || p),  p* = target/attractor belief.
Inference = Euler steps of the gradient flow on E with projection to the
simplex:  p_{t+1} = proj_Delta(p_t - eta * grad E(p_t)).

Closed-form facts under study (true at the fixed point p = p*):
    Hess E(p)  = diag(q_i / p_i^2)   [general position; q_i = p*_i]
               -> diag(1/p_i)        [at p = p*]
    mu = lambda_min = 1/max_i p_i,  L = lambda_max = 1/min_i p_i
    kappa = L/mu = max_i p_i / min_i p_i          (spectral ratio / condition number)
    eta_max = 2 mu / L^2 = 2 / (L * kappa) = 2 min_i p_i^2 / max_i p_i

On the tangent space {v: sum v_i = 0} (the subspace preserved by the
projection) the exact local stability boundary is
    eta_crit = 2 / theta_max,  theta_max = largest eigenvalue of
    diag(1/q_i) restricted to the tangent space;
    for K = 3, theta_max solves  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke

rng = np.random.default_rng(42)


# ----------------------------- core helpers -----------------------------
def proj_simplex(v):
    """Euclidean projection onto probability simplex (vectorized over last axis).
    Standard algorithm: find the last index j with u_j - (css_j - 1)/j > 0."""
    v = np.asarray(v, float)
    shape = v.shape
    v = v.reshape(-1, shape[-1])
    K = shape[-1]
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)
    avg = (css - 1.0) / np.arange(1, K + 1)
    cond = u - avg > 0
    j = np.where(cond)[1]    # column indices of True entries (rows repeat)
    # last True index per row:
    last = np.array([np.where(cond[i])[0][-1] for i in range(len(cond))])
    theta = (css[np.arange(len(v)), last] - 1.0) / (last + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out.reshape(shape)


def energy(p, pstar):
    """E(p) = D_KL(p* || p), elementwise-safe (p clipped away from 0)."""
    pc = np.clip(p, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sum(pstar * np.log(pstar / pc), axis=-1)
    return np.where(np.all(p > 0, axis=-1), e, np.inf)


def grad_energy(p, pstar):
    return -pstar / np.clip(p, 1e-12, 1.0)


def euler_step(p, pstar, eta):
    return proj_simplex(p - eta * grad_energy(p, pstar))


def eta_max_closed(pstar):
    """Closed-form safe step size at the fixed point: 2 min^2 / max."""
    return 2.0 * np.min(pstar) ** 2 / np.max(pstar)


def hessian_numeric(p, pstar, eps=1e-6):
    """Central-difference Hessian of E at p (K=3)."""
    K = len(p)
    H = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ei = np.zeros(K); ej = np.zeros(K)
            ei[i] = eps; ej[j] = eps
            H[i, j] = (energy(p + ei + ej, pstar) - energy(p + ei - ej, pstar)
                       - energy(p - ei + ej, pstar) + energy(p - ei - ej, pstar)) / (4 * eps * eps)
    return H


def to_xy(p):
    """Barycentric -> Cartesian (A=e1 at (0,0), B=e2 at (1,0), C=e3 at (0.5,sqrt3/2))."""
    return p[..., 1] * 1.0 + p[..., 2] * 0.5, p[..., 2] * (np.sqrt(3) / 2)


def from_xy(x, y):
    s = np.sqrt(3.0) / 2.0
    p3 = np.clip(y / s, 0.0, 1.0)
    p2 = np.clip(x - y / (np.sqrt(3.0)), 0.0, 1.0)
    p1 = 1.0 - p2 - p3
    mask = (p1 >= -1e-9) & (p2 >= -1e-9) & (p3 >= -1e-9)
    p1 = np.clip(p1, 0.0, 1.0)
    return np.stack([p1, p2, p3], axis=-1), mask


# ----------------------------- figure panels -----------------------------
PSTAR = np.array([0.70, 0.20, 0.10])
EMAX = eta_max_closed(PSTAR)          # ~ 2*0.01/0.7 = 0.02857
ETA_ILLUS = 0.8 * EMAX

fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
ps = withStroke(linewidth=3, foreground="white")

# ---------- (a) energy terrain + trajectories ----------
ax = axes[0, 0]
xgrid = np.linspace(-0.05, 1.05, 300)
ygrid = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, 260)
X, Y = np.meshgrid(xgrid, ygrid)
P, mask = from_xy(X, Y)
E = energy(P, PSTAR)
E = np.where(mask, E, np.nan)
lv = np.geomspace(1e-5, 50, 40)
cf = ax.contourf(X, Y, E, levels=lv, cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=50))
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label(r"$E(p)=D_{KL}(p^{*}\|p)$  [log scale]", fontsize=9)

starts = [np.array([1/3, 1/3, 1/3]), np.array([0.15, 0.60, 0.25]), np.array([0.45, 0.25, 0.30])]
for p0 in starts:
    p = p0.copy()
    xs, ys = [], []
    for t in range(120):
        p = euler_step(p, PSTAR, ETA_ILLUS)
        xs.append(to_xy(p)[0]); ys.append(to_xy(p)[1])
        if energy(p, PSTAR) < 1e-6:
            break
    ax.plot(xs, ys, lw=1.8, color="#D55E00", alpha=0.9)
    ax.plot(xs[0], ys[0], "o", ms=6, color="#D55E00", mfc="white")
    ax.annotate("$p_0$", (xs[0], ys[0]), textcoords="offset points", xytext=(-2, 8),
                fontsize=9, color="#D55E00")

px, py = to_xy(PSTAR)
ax.plot(px, py, "*", ms=22, color="#0072B2", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$  (fixed point)", (px, py), textcoords="offset points",
            xytext=(8, -16), fontsize=11, color="#0072B2", fontweight="bold")
ax.text(0.98, 0.94, f"$\\eta=0.8\\,\\eta_{{max}}$ = {ETA_ILLUS:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
ax.set_title("(a) The energy terrain of reasoning", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "belief simplex $\\Delta^2$: $p_1$-vertex (left), $p_2$-vertex (right), $p_3$-vertex (top)",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (b) curvature spectrum along one trajectory ----------
ax = axes[0, 1]
p = np.array([1/3, 1/3, 1/3])
lmax_a, lmin_a, kap_a = [], [], []
lmax_n, lmin_n = [], []
p_rec = []
for t in range(80):
    p = euler_step(p, PSTAR, ETA_ILLUS)
    p_rec.append(p.copy())
for p in p_rec:
    lmax_a.append(np.max(PSTAR / p**2)); lmin_a.append(np.min(PSTAR / p**2))
    H = hessian_numeric(p, PSTAR)
    w = np.linalg.eigvalsh(H)
    lmax_n.append(w[-1]); lmin_n.append(w[0])
ts = np.arange(len(p_rec))
ax.plot(ts, lmax_a, color="#0072B2", lw=1.8, label=r"closed form  $\lambda_{max}$")
ax.plot(ts, lmin_a, color="#009E73", lw=1.8, label=r"closed form  $\lambda_{min}$")
ax.plot(ts, lmax_n, "o", ms=4.2, color="#0072B2", mfc="none", label=r"numerical Hessian  $\lambda_{max}$")
ax.plot(ts, lmin_n, "s", ms=4.2, color="#009E73", mfc="none", label=r"numerical Hessian  $\lambda_{min}$")
ax.set_yscale("log")
ax.set_xlabel("reasoning step  $t$", fontsize=10)
ax.set_ylabel("curvature eigenvalues  $\\lambda$", fontsize=10)
ax.set_title("(b) Curvature spectrum: closed form = numerical", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
ax.grid(alpha=0.3)
kap = PSTAR.max() / PSTAR.min()
ax.text(0.98, 0.08, r"$\kappa = L/\mu = %.2f = 0.70/0.10$ (at $p^{*}$)" % kap,
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# ---------- (c) analytic eta_max surface ----------
ax = axes[1, 0]
XS, YS = np.meshgrid(np.linspace(0, 1, 260), np.linspace(0, np.sqrt(3) / 2, 230))
PS, maskC = from_xy(XS, YS)
eta = np.where(maskC & np.all(PS > 1e-4, axis=-1),
               2.0 * np.min(PS, axis=-1) ** 2 / np.max(PS, axis=-1), np.nan)
eta = np.where(maskC, eta, np.nan)
lv2 = np.geomspace(1e-6, 0.25, 35)
cf2 = ax.contourf(XS, YS, eta, levels=lv2, cmap="magma", norm=LogNorm(vmin=1e-6, vmax=0.25))
cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.85)
cbar2.set_label(r"$\eta_{max}(p)=2\min_i p_i^2/\max_i p_i$  [log]", fontsize=9)
ax.plot(px, py, "*", ms=22, color="#56B4E9", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$", (px, py), textcoords="offset points", xytext=(6, 10),
            fontsize=11, color="#56B4E9", fontweight="bold")
ax.set_title("(c) Analytic safe step size on the simplex", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "$\\eta_{max} \\to 0$ near the edges: the terrain dictates the pace",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]


def theta_max_tangent(q):
    """Largest eigenvalue of diag(q_i^{-1}) restricted to the tangent space {v: sum v_i = 0}.
    K = 3: analytic root of  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0."""
    q0, q1, q2 = q
    a = 3 * q0 * q1 * q2
    b = -2 * (q0 * q1 + q0 * q2 + q1 * q2)
    c = 1.0
    disc = b * b - 4 * a * c
    th = (-b + np.sqrt(disc)) / (2 * a)
    return th


def eta_crit_tangent_closed(q):
    """Exact local stability boundary: eta_safe < 2 / lambda_max of Hessian on the tangent space."""
    return 2.0 / theta_max_tangent(q)


# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

etas_scan = np.logspace(-4.0, 0.6, 140)

# --- LOCAL analysis: perturbations of p*; exact prediction = tangent-space spectrum ---
local_ana, local_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    pert = [proj_simplex(ps_ + 0.02 * rng.normal(0, 1, 3) * np.array([1, 0.5, 0.25])) for _ in range(8)]
    # asymptotic stability: final distance does not grow
    worst = np.zeros(len(etas_scan))
    for p0 in pert:
        P = np.broadcast_to(p0.copy(), (len(etas_scan), 3)).copy()
        d0 = np.linalg.norm(p0 - ps_)
        for _ in range(500):
            P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
        worst = np.maximum(worst, np.linalg.norm(P - ps_, axis=1) / d0)
    idx = np.where(worst < 1.0)[0]
    local_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    local_ana.append(eta_crit_tangent_closed(ps_))

# --- GLOBAL analysis: from uniform prior; conservative closed form = full-space spectrum ---
glob_ana, glob_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    P = np.broadcast_to(np.full(3, 1 / 3), (len(etas_scan), 3)).copy()
    for _ in range(400):
        P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
    fk = np.sum(ps_ * np.log(np.clip(ps_, 1e-12, 1) / np.clip(P, 1e-12, 1)), axis=1)
    idx = np.where(fk < 1e-3)[0]
    glob_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    glob_ana.append(eta_max_closed(ps_))

local_ana, local_num = np.array(local_ana), np.array(local_num)
glob_ana, glob_num = np.array(glob_ana), np.array(glob_num)

ax.loglog(local_ana, local_num, "o", ms=7, color="#0072B2", alpha=0.85, zorder=6,
          label=r"local: $2/\lambda_{\max}$ on tangent space (exact)")
ax.loglog(glob_ana, glob_num, "^", ms=7, color="#CC79A7", alpha=0.85, zorder=5,
          label=r"global: conservative bound $2/(L\kappa)$")
lims = [min(local_ana.min(), local_num.min()) * 0.6, max(local_ana.max(), local_num.max()) * 1.6]
ax.loglog(lims, lims, "k--", lw=1.5, label="y = x (exact prediction)")
logR = np.corrcoef(np.log10(local_ana), np.log10(local_num))[0, 1] ** 2
med = np.median(np.abs(local_num - local_ana) / local_ana)
ax.set_xlabel(r"closed-form  $\eta_{crit}$", fontsize=10)
ax.set_ylabel("numerical stability threshold  $\\eta_{crit}$", fontsize=10)
ax.set_title("(d) Closed form = the exact local stability boundary", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.05, f"exact (tangent spectrum): $R^2$ = {logR:.4f}, median err = {med:.2%}\nconservative (full spectrum): $\\eta_{{crit}} \\geq 2/(L\\kappa)$ always",
        transform=ax.transAxes, fontsize=9, color="k", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

fig.suptitle("The Energy Landscape of Reasoning: closed-form geometry and its verification\n"
             r"$E(p)=D_{KL}(p^{*}\|p)$,  inference $=$ Euler steps of $-\nabla E$ projected to the simplex",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "ch12_energy_terrain.png"   # saves next to the script
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
print(f"eta_max closed form (p* = {PSTAR}): {EMAX:.6f}")
print(f"kappa at fixed point: {PSTAR.max()/PSTAR.min():.2f}")
print(f"panel d: local R^2(log-log) = {logR:.4f}, n = {len(local_ana)}, median err = {med:.2%}")
print(f"panel d: global: numerically stable region >= eta_max (projection adds stability)")
```

*Figure 1 and source code: Zixi Li (Professor Pallas), 2026. Experiment script `ch12_energy_terrain.py` (complete code in the Step 14 code block).*

### 14.7 Direct Evidence on a Real LLM: Belief Anchors, Evidence Strength, and Memory Decay

The numerical experiment of Step 14 verified the formula's internal self-consistency: the dynamics were constructed from Euler steps of the closed-form solution, so of course they track it. This section gives direct evidence on a real LLM: instantiate the inference operator $F: \mathcal{P} \to \mathcal{P}$ of Section 7 with a real model (DeepSeek-V4-flash, reasoning type), inject evidence step by step, read the belief step by step, and check, item by item, the real-reasoning counterparts of Yonglin's formula and the Yonglin limit. The figure is `ch12_llm_evidence.png`; the experiment's full data design is as follows (the experiment itself and its numbers all live in the figure — the script involving the real model API is not repeated here).

**Experiment setup.** A three-color bag world, still on the belief simplex $\Delta^2$. The bag's ball proportions $p^*$ take three settings, including the book-wide baseline $(0.70, 0.20, 0.10)$ ($\kappa = 7.00$); a detector of known reliability $d \in \{0.90, 0.70, 0.55\}$ gives one noisy color report per step on the drawn ball (the likelihood $P(r|c)$ is exactly known, and the true posterior $p^B_t$ is computed in closed form step by step); the evidence sequence is the sequence of detector reports, $T = 6$. At each step, the model's input is "bag parameters + the previous $t$ reports + a forced one-word answer", and the belief is read off from the normalized top-k logprobs of the final answer token ($K = 3$). Grid: $3 \times 3 \times 4$ seeds — one ordinary trajectory plus two prompted trajectories per cell; anchor measurement: 3 option arrangements $\times$ 8 rephrasings.

![Figure 12.2: Belief-anchor and evidence-dynamics verification on a real LLM](/figures/ch12_llm_evidence.png)

**How to read the three panels of Figure 12.2** — (a) anchor: zero-information beliefs under the three option arrangements are all close to a vertex (position bias), global average $\kappa_A = 1.285$; (b) evidence absorption: two lines per curve — the solid line is the KL between the LLM and the true posterior $p^B_t$, the dashed line is the KL to a "$d' = 0.9$ fake posterior" — the dashed line is lower almost everywhere; (c) phase-difference memory decay: the median distance $d_t$ hits zero at the third step.

**Result 1: the no-information anchor $A$ is not the uniform point — it is a prior you can measure.** Beliefs under the three arrangements: Red/Blue/Black $(0.999, 0.001, 0)$; Black/Red/Blue $(0.125, 0.0001, 0.875)$; Blue/Black/Red $(0.0001, 1.000, 0)$. Position bias around 87.5%–100% (the single "defector": under the Black/Red/Blue arrangement, Red was chosen 1 time in 8). Global $A = (0.375, 0.334, 0.292)$, $\kappa_A = 1.285$. $A$ is only 1.285 away from uniform in condition-number terms, but its composition is not uniform: it is a mix of "option-position regularity × mild word preference". Step 9, "Identifying the Prior Anchor": the empirical anchor is right here — it is the same object as the $A$ in the $A \neq A^*$ of Yonglin's formula, now with numbers.

**Result 2: output-layer beliefs are spiky — reasoning hugs the simplex boundary.** At step 0 (the prompt already states the bag proportions 7/2/1, no reports yet), the median $\text{KL}(p^B_0 \| p^{\text{LLM}}_0)$ is about 6.8–7.4 nats: even though the prompt states the asymmetric proportions, the model's belief is still near a one-hot vertex. The model's faith in a probabilistic world is decisive — it does not roam the simplex interior (the continuous spectrum of uncertainty) but jumps straight to some near-vertex. For Yonglin's formula: the real reasoning system's attractor $\approx$ a vertex; $A$ is not only off $A^*$, it lands in vertex country.

**Result 3: evidence-strength absorption is asymmetric — weak evidence is amplified, strong evidence is followed faithfully.** The table below (median KL: KL between the LLM belief and the true posterior / KL to the $d' = 0.9$ fake posterior):

| d | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---|---|---|---|---|
| 0.55 (true posterior / $d' = 0.9$ posterior) | 5.82 / 0.96 | 3.30 / 1.34 | 3.74 / 0.40 | 4.06 / 0.40 | — |
| 0.70 | 3.32 / 0.92 | 0.79 / 0.05 | 1.72 / 0.38 | 0.40 / 0.02 | 0.28 / 0.02 |
| 0.90 | 0.60 | 0.03 | 0.31 | 0.02 | — |

For $d = 0.90$ it decreases monotonically (final median 0.02 — almost exactly Bayesian); for $d = 0.70$ it sits closer to the $d' = 0.9$ fake posterior from step 2 onward; for $d = 0.55$, the distance to the $d' = 0.9$ fake posterior is about 8× better than to the true posterior. The model treats the prompt's evidence reliability as an approximately fixed $d' \approx 0.9$: weak evidence is systematically amplified, strong evidence is followed precisely. This runs opposite to human conservatism (Edwards 1968) — an LLM's evidence-strength ceiling is about 0.9: weak evidence always means "don't trust the prompt, use it at maximum strength".

**Result 4: compression and memory — the phase-difference half-life is about 2 steps.** Same world, same tail of reports (reports 2–6 identical); the only difference between the two trajectories is whether the first report is red or blue. The initial branching distance is $d_1 \equiv 1.414 = \sqrt{2}$ (between two vertices; all 36 cells without exception); from $d_2$ onward the median distance is 0; single-step compression median 0.07–0.20; half-life 2.0–2.5 steps. The window formula of Step 10.7, $-\ln \varepsilon / \ln k$ ($\varepsilon = 0.1$), predicts about 2.3 steps — consistent with the measurement. Notes: (a) occasional re-branching: 1 of 12 cells still held a distance >0.1 at $t = 6$, and the maximum over the whole run can be $\sqrt{2}$ — a mirror image of Step 14's panel (d) "transient grows first, contracts later" (here it contracts first, then re-grows): even a deterministic model contains chaotic re-splitting; (b) at temperature 0, the part of the $d_t = 0$ "coincidence" is a deterministic artifact; statistically, only this reading counts: "one unit of phase difference no longer influences the next step's choices after two steps".

Four-row cross-check against the theory: anchor = the measurement of Step 9's prior anchor $A$ ($A \neq$ uniform, $\kappa_A = 1.285$); spikiness = the energy basin near the vertex (the reasoning trajectory moves in vertex country); weak-evidence overreaction = asymmetric evidence absorption (the $d' \approx 0.9$ ceiling); phase-difference half-life $\approx 2$ steps = the measured constant of Section 6's effective reasoning window, the same order of magnitude as the closed-form compression bound. One honest caveat: closed-form quantities such as $\eta_{\max}$ and $\theta_{\max}$ describe safe step sizes of continuous dynamics; a real LLM's "step size" is a function of evidence strength (weak evidence also carries an effective $d' \approx 0.9$), and its motion looks more like "big steps, short window, jumps between vertices".

*Figure 12.2 and source code: Zixi Li (Professor Pallas), 2026. The experiment is presented in Step 14.7 (the script's API version is not expanded in the book).*

### 14.8 The Same Limit on a Laptop: a Cobweb You Can Draw Yourself

Section 14.7 measured the belief dynamics of a large closed-parameter model. The price of that experiment: the reader gets the numbers, not the experiment. This subsection gives the same limit in a form you can run on your own laptop. The model is GPT-2, about 124M parameters, open and small; the belief is one-dimensional; and the resulting picture is the classic cobweb plot of a discrete dynamical system (Section 7, iterate a sequence $x_{t+1} = F(x_t)$) — drawn from a real model instead of a formula.

**Experiment setup.** Twenty-four transitive-chain questions (e.g. "A is taller than B. B is taller than C. Is A taller than C?"). All of them are TRUE, so the true answer distribution is exactly $A^* = 1.0$. The belief is $p = P(\text{YES})$: the model proposes one answer word, and we read the two answer logits and take the sigmoid of their difference ($P = \sigma(\text{logit}_{YES} - \text{logit}_{NO})$). The model runs at temperature 0, so the experiment is deterministic: the same numbers every time. One step of the dynamics: write the current belief into the prompt as a line of text ("Confidence so far: P(YES) = 0.437."), ask the model to answer again, and read the next belief. The map $F$ is measured on 31 grid points $x = 0.02 \ldots 0.98$ and averaged over the 24 questions; then the recursion is run for 128 steps from each question's own natural start belief (no confidence line at all).

**Result 1: the anchor is measured, not assumed.** The mean initial belief over the 24 questions is 0.4045; per-question values range from 0.1993 to 0.7105. The prior anchor of a text is a function of the text: for the road/path/lane chain the model starts at 0.199, for the X/Y/Z chain at 0.710. There is no single "training-data marginal" behind a question: the prior anchor is a mixture of per-question anchors.

**Result 2: the measured map is flat – one step, not a staircase.** Across the whole grid, $F(x)$ stays between 0.47 and 0.52 ($\sigma \approx 0.067$). Write down a belief of 0.03 or 0.97: the model answers back with about 0.49 either way. The cobweb trajectory is not a slow staircase; it is a cliff. All seven starts (0.03, 0.10, 0.25, 0.40, 0.70, 0.85, 0.97) are absorbed into the anchor band in a single step. The fixed point of the measured map is $A = 0.4849$; the gap to $A^* = 1.0$ is 0.5151. The map is flat and its fixed point is the prior: no read-back step brings the belief any closer to the truth.

**Result 3: the effective reasoning window is one step.** In the 128-step rollouts, the median $P(\text{YES})$ moves from 0.3944 ($t = 0$) to 0.5002 ($t = 1$), and then never again: from $t = 16$ through $t = 128$ the median stays inside $[0.490, 0.505]$, and the per-question min/max do not move at all (0.380 / 0.6885, constant). Each question freezes at its own local anchor (the 0.6885 maximum is the X/Y/Z chain's anchor). The measured window of usefulness is one step. "Think longer" is not merely slow on this model – it is already finished.

![Figure 12.3: Cobweb plot of one-step belief absorption on a real GPT-2: the measured map, seven iterated beliefs, the anchor fixed point A = 0.485 and the unreachable true answer A* = 1.0](/figures/yonglin_cobweb_measured.png)

![Figure 12.4: 128 steps of belief re-circulation on a real GPT-2: median and spread over 24 questions, frozen at the anchor after one step](/figures/yonglin_belief_evolution.png)

**How to read Figure 12.3 (the cobweb).** The grey dashed diagonal is "belief unchanged": any point on it is a candidate fixed point. The blue line is the measured map $F$: belief before one re-reading ($x$) against belief after one re-reading ($F(x)$). It is nearly flat at 0.49 – that flatness is the whole point. The colored staircases are the iterated beliefs from seven different starts: each one jumps vertically to the map, then lands horizontally on the diagonal, and everything after that happens inside the zoomed frame. The green star marks the fixed point $A = 0.4849$; the purple cross marks $A^* = 1.0$, the true answer. The zoomed frame shows the zig-zag dying at $A$: after the first step, the belief just vibrates in place. (The caption of the figure may use $A = 0.485$.)

**How to read Figure 12.4 (the time series).** Each line is one question's belief over 128 steps; the shaded bands are the interquartile and 10-90 percentiles over the 24 questions. Read the cliff at $t = 1$: the median jumps from 0.394 to 0.500 in one step, and no question ever gets meaningfully closer to $A^* = 1.0$. The dotted line is the measured anchor $A \approx 0.485$; the purple line at 1.0 is the truth that stays unreachable. The important thing is not that the lines are flat – it is where they freeze.

One honest caveat. The belief here is a one-dimensional projection of a 124M-parameter state, and writing the number back into the prompt is a stylized form of writing a belief down. The experiment does not exhibit GPT-2's psychology; it exhibits the dynamics of this projection: flat map, anchor at the prior, window of one step. What it does show, without any modeling, is that on a real model the Yonglin limit is not a slow fade – it is a cliff.

**Play with it.** Change the question bank, change the injected-confidence template, run `--model gpt2-medium`, change `--grid` and `--steps`. The anchor moves (it always moves); the map stays flat; $A \neq A^*$ is the pattern you will keep rediscovering.

**Source code** (complete and runnable; save it as `yonglin_gpt2.py` and run. It saves two PNGs next to itself: `yonglin_cobweb_measured.png` = Figure 12.3, `yonglin_belief_evolution.png` = Figure 12.4):

```python
#!/usr/bin/env python3
"""
Yonglin Cobweb: Belief-Recirculation Dynamics on a Real GPT-2
==============================================================
Companion experiment for The Reasoning Kingdom, Chapter 12
(Sections 5, 7, 14.7 and 14.8).

THE IDEA IN ONE PARAGRAPH
-------------------------
Yonglin's formula says: belief dynamics p_{t+1} = F(p_t) converges to the
prior anchor A, not to the true answer A* (A != A*). Section 14.7 verifies anchors and evidence
absorption on a large closed API model, but the reader cannot run it.
Section 14.8 gives the version you can RUN and PLAY WITH: a small open
model (GPT-2, ~124M params, runs on a laptop), a one-dimensional belief
state P(YES), and the belief recirculated through the model itself.
Each step you read the model's belief from its next-token logits, write
that belief back into the prompt as text, and measure the next belief.
That defines a one-dimensional dynamical system p_{t+1} = F(p_t) that you
can measure, plot, and look at as a COBWEB picture: the curve y = F(x),
the diagonal y = x, and the zig-zag staircase of the iterated belief.

WHAT THE SCRIPT DOES
--------------------
  1. loads the model (default: gpt2; check the HuggingFace cache first).
  2. read_belief(prompt): P(YES) = sigmoid(logit_YES - logit_NO).
  3. measure_F(): 31 grid points x in [0, 1], the belief written back
     into the prompt; F measured over 24 questions.
  4. rollouts(): 128 steps of belief re-circulation from each question's
     natural start; prints summaries and the distribution over questions.
  5. cobweb_points(): the classic zig-zag construction.
  6. two figures:
       (a) yonglin_cobweb_measured.png       -- the cobweb itself
       (b) yonglin_belief_evolution.png      -- p_t vs t, all 128 steps

HONESTY CAVEATS
---------------
  * "Belief" is a one-dimensional projection (next-token softmax of
    YES/NO). The real state is 124M-dimensional; this is the deliberate
    cut that makes the cobweb picture possible.
  * Writing the number p back into the prompt is a stylized way of
    writing the belief down. The model re-reads a text that contains
    its own last belief; it does not literally iterate a hidden state.
  * Every number printed and every point plotted comes from a live run
    of the model on this machine. Nothing is scripted.

Run:  python3 yonglin_gpt2.py
      python3 yonglin_gpt2.py --model gpt2-medium --steps 128 --grid 31
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------------------------
# 0. The question bank (24 short transitive-chain questions, all TRUE)
# ---------------------------------------------------------------------------
QUESTIONS = [
    "A is taller than B. B is taller than C. Is A taller than C?",
    "X is larger than Y. Y is larger than Z. Is X larger than Z?",
    "P is heavier than Q. Q is heavier than R. Is P heavier than R?",
    "The oak is older than the birch. The birch is older than the elm. Is the oak older than the elm?",
    "K is faster than L. L is faster than M. Is K faster than M?",
    "The river is longer than the stream. The stream is longer than the brook. Is the river longer than the brook?",
    "Red is darker than pink. Pink is darker than cream. Is red darker than cream?",
    "The mountain is steeper than the hill. The hill is steeper than the slope. Is the mountain steeper than the slope?",
    "Steel is harder than iron. Iron is harder than tin. Is steel harder than tin?",
    "The tiger is stronger than the wolf. The wolf is stronger than the fox. Is the tiger stronger than the fox?",
    "The poem is longer than the song. The song is longer than the haiku. Is the poem longer than the haiku?",
    "Bob is faster than Anne. Anne is faster than Carl. Is Bob faster than Carl?",
    "Winter is colder than autumn. Autumn is colder than spring. Is winter colder than spring?",
    "Silver is shinier than copper. Copper is shinier than iron. Is silver shinier than iron?",
    "The soup is hotter than the stew. The stew is hotter than the porridge. Is the soup hotter than the porridge?",
    "The whale is larger than the elephant. The elephant is larger than the horse. Is the whale larger than the horse?",
    "The road is longer than the path. The path is longer than the lane. Is the road longer than the lane?",
    "The clock is older than the watch. The watch is older than the phone. Is the clock older than the phone?",
    "Sarah is taller than Kate. Kate is taller than Anna. Is Sarah taller than Anna?",
    "The ocean is deeper than the lake. The lake is deeper than the pond. Is the ocean deeper than the pond?",
    "The horse is faster than the dog. The dog is faster than the cat. Is the horse faster than the cat?",
    "The house is taller than the garage. The garage is taller than the shed. Is the house taller than the shed?",
    "The sun is hotter than the flame. The flame is hotter than the candle. Is the sun hotter than the candle?",
    "Matthew is slower than Helen. Helen is slower than George. Is Matthew slower than George?",
]
ANSWER = "YES"                      # every question above is TRUE (A* = 1.0)
ID_YES, ID_NO = None, None          # filled in main()
TEMPLATE_ANSWER = "\nAnswer with exactly one word: YES or NO."


def confidence_template(p: float) -> str:
    """Write the current belief back into the prompt as one line."""
    return f"\nConfidence so far: P(YES) = {p:.3f}."


# ---------------------------------------------------------------------------
# 1. Belief reader
# ---------------------------------------------------------------------------
@torch.no_grad()
def read_belief(model, tokenizer, prompt: str, device) -> float:
    """P(YES) = sigmoid(logit_YES - logit_NO) at the last position."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = model(input_ids).logits[0, -1]
    return float(torch.sigmoid(logits[ID_YES] - logits[ID_NO]).item())


def ask(model, tokenizer, question: str, p: float, device) -> float:
    """One application of the map: belief p in, belief F(p) out."""
    prompt = question + confidence_template(p) + TEMPLATE_ANSWER
    return read_belief(model, tokenizer, prompt, device)


# ---------------------------------------------------------------------------
# 2. Measuring F on a grid
# ---------------------------------------------------------------------------
def measure_F(model, tokenizer, questions, grid_n: int, device):
    """F(x) on a grid, averaged over questions: (xs, mean, std)."""
    xs = np.linspace(0.02, 0.98, grid_n)
    Fs = np.zeros((len(questions), grid_n))
    for qi, q in enumerate(questions):
        for xi, x in enumerate(xs):
            Fs[qi, xi] = ask(model, tokenizer, q, float(x), device)
    return xs, Fs.mean(axis=0), Fs.std(axis=0)


# ---------------------------------------------------------------------------
# 3. Rollouts
# ---------------------------------------------------------------------------
def rollouts(model, tokenizer, questions, steps: int, device):
    """p_0 = natural belief; p_{t+1} = F(p_t). Returns array (Q, steps+1)."""
    traj = np.zeros((len(questions), steps + 1))
    for qi, q in enumerate(questions):
        p0 = read_belief(model, tokenizer, q + TEMPLATE_ANSWER, device)
        traj[qi, 0] = p0
        for t in range(steps):
            traj[qi, t + 1] = ask(model, tokenizer, q, float(traj[qi, t]), device)
    return traj


# ---------------------------------------------------------------------------
# 5. Cobweb construction
# ---------------------------------------------------------------------------
def cobweb_points(fn, p0, steps: int):
    """The zig-zag: vertical to the curve, horizontal to the diagonal."""
    pts_x, pts_y = [p0], [p0]
    p = p0
    for _ in range(steps):
        f = float(np.asarray(fn(np.array([p])))[0])
        pts_x.append(p)
        pts_y.append(f)
        p = f
        pts_x.append(p)
        pts_y.append(p)
    return np.array(pts_x), np.array(pts_y)


def interpolate_F(xs, Fm):
    def fn(x):
        return np.interp(np.asarray(x, dtype=float), xs, Fm)
    return fn


def fixed_point(xs, Fm):
    """Root of F(x) - x by linear interpolation on the grid."""
    g = Fm - xs
    idx = np.where(np.diff(np.sign(g)) != 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(xs[i] - g[i] * (xs[i + 1] - xs[i]) / (g[i + 1] - g[i]))


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def plot_measured_cobweb(xs, Fm, Fs, p0_mean, fixed, out_dir):
    """(a) The cobweb itself: multi-start, the flat anchor map."""
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    ax.plot(xs, xs, color="0.55", lw=1.6, ls="--", label="y = x  (belief unchanged)")
    ax.fill_between(xs, Fm - Fs, Fm + Fs, color="#9ecae1", alpha=0.45,
                    label="measured F(x) ± std over 24 questions")
    ax.plot(xs, Fm, color="#1f77b4", lw=3.2, label="F(x): GPT-2 re-reads its belief")
    starts = [0.03, 0.10, 0.25, float(p0_mean), 0.70, 0.85, 0.97]
    fn = interpolate_F(xs, Fm)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(starts)))
    for c, p0 in zip(colors, starts):
        px, py = cobweb_points(fn, float(p0), 20)
        ax.plot(px, py, color=c, lw=1.7, alpha=0.9)
        ax.plot([p0], [p0], "o", ms=8, color=c, zorder=6, label=f"start p0 = {p0:.3f}")
    ax.plot(fixed, fixed, "*", color="#2ca02c", ms=20, zorder=7,
            label=f"fixed point A = {fixed:.3f}")
    ax.plot([1.0], [1.0], "P", color="#9467bd", ms=15, zorder=7,
            label="A* = 1.0  (true answer)")
    ax.set_xlim(-0.02, 1.07); ax.set_ylim(-0.02, 1.07)
    ax.set_xlabel("belief before re-reading (x)")
    ax.set_ylabel("belief after re-reading (F(x))")
    ax.set_title("Yonglin cobweb on a real GPT-2: one-step absorption into the anchor\n"
                 f"every start falls into the flat map F(x) ~ {np.mean(Fm):.2f} "
                 f"-> A = {fixed:.3f}, never A* = 1.0",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax.grid(alpha=0.25)
    axin = ax.inset_axes([0.50, 0.04, 0.44, 0.44])
    axin.plot(xs, xs, color="0.55", lw=1.2, ls="--")
    axin.plot(xs, Fm, color="#1f77b4", lw=2.0)
    axin.fill_between(xs, Fm - Fs, Fm + Fs, color="#9ecae1", alpha=0.4)
    for c, p0 in zip(colors, starts):
        px, py = cobweb_points(fn, float(p0), 20)
        axin.plot(px, py, color=c, lw=1.0, alpha=0.9)
    axin.plot([fixed], [fixed], "*", color="#2ca02c", ms=11)
    axin.set_xlim(0.47, 0.66); axin.set_ylim(0.47, 0.66)
    axin.set_xticks([0.50, 0.55, 0.60, 0.65]); axin.set_yticks([0.50, 0.55, 0.60, 0.65])
    axin.tick_params(labelsize=7)
    axin.set_title("zoom: the zig-zag dies at A", fontsize=8.5)
    axin.grid(alpha=0.3)
    # the frame that tells the reader WHICH region of the main plot is zoomed:
    ax.indicate_inset_zoom(axin, edgecolor="0.30", lw=1.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "yonglin_cobweb_measured.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_evolution(traj, fixed, out_dir):
    """(b) p_t vs t over 128 steps: median, quartile band, samples."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    t = np.arange(traj.shape[1])
    med = np.median(traj, axis=0)
    q1, q3 = np.percentile(traj, [25, 75], axis=0)
    lo, hi = np.percentile(traj, [10, 90], axis=0)
    ax.fill_between(t, q1, q3, color="#9ecae1", alpha=0.55,
                    label="interquartile band (24 questions)")
    ax.fill_between(t, lo, hi, color="#c6dbef", alpha=0.35,
                    label="10-90% band")
    ax.plot(t, med, color="#1f77b4", lw=2.8, label="median P(YES)")
    for qi in range(min(4, traj.shape[0])):
        ax.plot(t, traj[qi], color=plt.cm.plasma(0.15 + 0.2 * qi),
                lw=1.0, alpha=0.65, label=f"question {qi + 1}")
    ax.axhline(fixed, color="#2ca02c", ls=":", lw=2.0,
               label=f"anchor A ≈ {fixed:.3f} (fixed point)")
    ax.axhline(1.0, color="#9467bd", ls="-", lw=2.0, label="A* = 1.0 (true answer)")
    ax.set_xlim(0, traj.shape[1] - 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("re-circulation step t")
    ax.set_ylabel("belief P(YES)")
    ax.set_title("128 steps of belief re-circulation: the window of usefulness",
                 fontsize=11.5)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "yonglin_belief_evolution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Yonglin cobweb experiment on GPT-2")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--replot", action="store_true",
                    help="skip measurement, redraw figures from the cached data")
    args = ap.parse_args()
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(out_dir, "yonglin_cache.npz")
    if args.replot and os.path.exists(cache_path):
        data = np.load(cache_path)
        xs, Fm, Fs, traj, p0s = (data["xs"], data["Fm"], data["Fs"],
                                 data["traj"], data["p0s"])
        p0_mean = float(np.mean(p0s))
        print(f"[replot] loaded cached data from {cache_path}")
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[setup] model={args.model} device={device}")
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
        global ID_YES, ID_NO
        ID_YES = tokenizer.encode(" YES")[0]
        ID_NO = tokenizer.encode(" NO")[0]
        print(f"[setup] YES token={ID_YES}  NO token={ID_NO}")
        print(f"\n[1] Initial belief per question (no confidence injected):")
        p0s = []
        for q in QUESTIONS:
            p0 = read_belief(model, tokenizer, q + TEMPLATE_ANSWER, device)
            p0s.append(p0)
            print(f"    P(YES) = {p0:.4f}   | {q[:62]}")
        p0_mean = float(np.mean(p0s))
        print(f"    mean p0 = {p0_mean:.4f}   (the measured prior anchor A)")
        print(f"\n[2] Measuring F on {args.grid} grid points x {len(QUESTIONS)} questions ...")
        xs, Fm, Fs = measure_F(model, tokenizer, QUESTIONS, args.grid, device)
        for x, fm, fs in zip(xs, Fm, Fs):
            print(f"    x={x:.3f}   F_mean={fm:.4f}   F_std={fs:.4f}")
        print(f"\n[3] Rollouts ({args.steps} steps of belief re-circulation):")
        traj = rollouts(model, tokenizer, QUESTIONS, args.steps, device)
        for t in range(0, args.steps + 1, 16):
            row = traj[:, t]
            print(f"    t={t:3d}  median={np.median(row):.4f}  "
                  f"q1={np.percentile(row, 25):.4f} q3={np.percentile(row, 75):.4f}  "
                  f"min={row.min():.4f}  max={row.max():.4f}")
        np.savez_compressed(cache_path, xs=xs, Fm=Fm, Fs=Fs, traj=traj,
                            p0s=np.asarray(p0s, dtype=float))
        print(f"\n[cache] saved measured data to {cache_path}")
    fixed = fixed_point(xs, Fm)
    print(f"\n[4] Fixed point of the measured map: A = {fixed:.4f}")
    print(f"    True answer A* = 1.0.  Gap = {abs(fixed - 1.0):.4f}")
    print(f"\n[5] Final belief after {args.steps} steps: "
          f"median={np.median(traj[:, -1]):.4f}")
    print(f"    One-step view: p0 mean={p0_mean:.4f}, p1 median={np.median(traj[:, 1]):.4f}")
    p1 = plot_measured_cobweb(xs, Fm, Fs, p0_mean, fixed, out_dir)
    p2 = plot_evolution(traj, fixed, out_dir)
    print(f"\n[6] Figures written:")
    for p in (p1, p2):
        print(f"    {p}")
    print("\nDone. The zig-zag in the first figure is Yonglin's formula,\n"
          "drawn from a real model instead of a formula.")


if __name__ == "__main__":
    main()
```

Run the script and you will see (excerpts from a real run – the numbers below were produced by exactly this script, gpt2 on a macOS MPS device; the trailing part of the run is representative):

```text
[setup] model=gpt2 device=mps
[setup] YES token=21560  NO token=8005
[1] Initial belief per question (no confidence injected):
    P(YES) = 0.5230   | A is taller than B. B is taller than C. Is A taller than C?
    P(YES) = 0.7105   | X is larger than Y. Y is larger than Z. Is X larger than Z?
    P(YES) = 0.4635   | P is heavier than Q. Q is heavier than R. Is P heavier than R?
    P(YES) = 0.1993   | The road is longer than the path. The path is longer than the lane. Is the road longer than the lane?
    P(YES) = 0.3056   | The tiger is stronger than the wolf. The wolf is stronger than the fox. Is the tiger stronger than the fox?
    P(YES) = 0.4031   | Winter is colder than autumn. Autumn is colder than spring. Is winter colder than spring?
    ...
    mean p0 = 0.4045   (the measured prior anchor A)
[2] Measuring F on 31 grid points x 24 questions ...
    x=0.020   F_mean=0.4834   F_std=0.0668
    x=0.212   F_mean=0.5067   F_std=0.0647
    x=0.500   F_mean=0.4700   F_std=0.0693
    x=0.788   F_mean=0.5002   F_std=0.0666
    x=0.980   F_mean=0.4867   F_std=0.0681
[3] Rollouts (128 steps of belief re-circulation):
    t=  0  median=0.3944  q1=0.3338  q3=0.4611  min=0.1993  max=0.7105
    t= 16  median=0.4930  q1=0.4615  q3=0.5241  min=0.3800  max=0.6885
    t= 32  median=0.5009  q1=0.4601  q3=0.5316  min=0.3800  max=0.6885
    t= 64  median=0.4937  q1=0.4681  q3=0.5241  min=0.3800  max=0.6885
    t= 96  median=0.4932  q1=0.4632  q3=0.5361  min=0.3769  max=0.6885
    t=128  median=0.4955  q1=0.4596  q3=0.5296  min=0.3800  max=0.6885
[4] Fixed point of the measured map: A = 0.4849
    True answer A* = 1.0.  Gap = 0.5151
[5] Final belief after 128 steps: median=0.4955
    One-step view: p0 mean=0.4045, p1 median=0.5002
```

*Figures 12.3-12.4 and source code: Zixi Li (Professor Pallas), 2026. The experiment was run on the default laptop build: GPT-2, transformers, numpy, matplotlib; the fixed-point and all percentages come straight from the model run, and the script is deterministic.*

<div class="center">

------------------------------------------------------------------------

</div>

## Physical Intuition: Bowls, Slopes, and Step Sizes

The four key parameters of the formal derivation, $(\mu, L, \eta, \rho)$, have a straightforward physical meaning:

### 1. Convexity (the bowl's shape)

- A bowl (convex function): only one lowest point; gradient descent is guaranteed to find it
- Potato chips (non-convex function): several dips; you can get stuck in a local minimum
- A well-trained model has an energy function shaped like a bowl; the diversity of the data decides the bowl's depth

### 2. Strong convexity coefficient $\mu$ (how steep the bowl is)

- Shallow bowl ($\mu$ small): gentle slope, slow convergence
- Deep bowl ($\mu$ large): steep slope, fast convergence
- Physical meaning: training-data diversity -> large $\mu$ -> the model easily learns clear patterns

### 3. Smoothness constant $L$ (the maximum rate at which the slope changes)

- Gentle terrain ($L$ small): the slope changes little; you can walk in big steps
- Rugged terrain ($L$ large): the slope swings violently; you must walk in small steps
- Physical meaning: architectural smoothness -> small $L$ -> stable reasoning

### 4. Step size $\eta$ (the span of one inference)

- Too large: may skip right past the correct path
- Too small: dawdling; convergence is slow
- Just right: $\eta < \frac{2\mu}{L^2}$ (the safe-downhill formula)

### 5. Contraction factor $\rho = \sqrt{1 - 2\eta\mu + \eta^2L^2}$

- $\mu$ large (steep bowl) -> $\rho$ small -> fast convergence
- $L$ large (rugged) -> $\rho$ large -> slow convergence
- $\eta$ just right -> $\rho$ minimal -> optimal convergence

### Analogy: Learning to Ride a Bicycle

- Convexity: a clear destination (home)
- $\mu$: the pull of home (distance and gravity)
- $L$: how even the road surface is
- $\eta$: how hard each pedal stroke is
- $\rho$: how fast you get there

### What This Looks Like in AI Reasoning

1. Data quality -> $\mu$: diverse data creates a "deep bowl"
2. Model architecture -> $L$: smooth design creates "gentle terrain"
3. Reasoning design -> $\eta$: the right step size balances exploration and exploitation
4. All three together -> $\rho$: it decides the length of the effective reasoning window

Good system: $\mu$ large + $L$ small -> big $\eta$ allowed -> $\rho$ small -> fast convergence
Bad system: $\mu$ small + $L$ large -> must be small $\eta$ -> $\rho$ large -> slow convergence

That is why different models and different tasks have different "effective reasoning windows" — their $(\mu, L, \eta)$ combos differ, so their convergence speeds differ.

The Yonglin limit is not a mystical phenomenon; it is the inevitable consequence of convex optimization on a belief space.

<div class="center">

------------------------------------------------------------------------

</div>

## Afterword: The Pallas's Cat Comes Down the Mountain

> Reasoning is like climbing a mountain, and the truth is at the top. But mountains have gravity, and training is how you build the road.

### A Little Story: Professor Pallas's Reasoning Expedition

Let me take you on a climb of "Truth Mountain" and use the metaphor to explain the core concepts of Yonglin's formula.

**Act 1: Choosing the terrain**

At the foot of the mountain there are three roads:

1. The deep-bowl road ($\mu$ large): steep but straight to the top
2. The shallow-plate road ($\mu$ small): gentle but long way round
3. The potato-chip road (non-convex): several dips; you can get lost

We would choose the deep-bowl road, because data diversity carved the steep bowl — the best reasoning terrain.

**Act 2: Checking the road surface**

The deep-bowl road itself comes in two kinds:

- The smooth road ($L$ small): the slope changes evenly
- The rugged road ($L$ large): steep one moment, gentle the next; easy to stumble

Luckily, our model architecture is smooth, $L$ is small, so we can walk with confidence.

**Act 3: Deciding the stride**

Here we need the "safe-downhill formula": $\eta < \frac{2\mu}{L^2}$

When $\mu$ is large (steep bowl) and $L$ small (smooth road), we can use a big step size $\eta$!

**Act 4: The contraction factor $\rho$**

With each step, the distance to the summit gets multiplied by $\rho = \sqrt{1 - 2\eta\mu + \eta^2L^2}$

- If $\rho = 0.9$: after 10 steps, the distance is $(0.9)^{10} \approx 0.35$
- If $\rho = 0.5$: after 10 steps, the distance is $(0.5)^{10} \approx 0.001$

When the $(\mu, L, \eta)$ combo is good, $\rho$ is small, and you get close to the summit quickly!

**Act 5: The Yonglin observation**

But here is the key catch — no matter how you walk, in the end you always return to the training camp, not the summit of truth.

That is because the training data carried us here — this is the prior anchor $A = P_D(y)$!

The stable point of the gradient descent: $P_T \nabla E(A) = 0$ (the gradient lies along the normal)

The fixed point of the inference operator: $F(A) = A$

**Act 7: The meta-level break**

You might think: "Since we know it will return to camp, why not verify the direction directly?"

But the trouble is: the mechanism that produces the reasoning (gradient descent) and the mechanism that verifies the reasoning (judging right from wrong) are broken apart.

$$
\nexists M \text{ such that } V(F(p)) = M(F, p)
$$

Between the object layer and the meta layer, there is a chasm.

### The Moral of the Story

1. The terrain is built by data ($\mu$): diverse data -> deep bowl -> good reasoning
2. The road is paved by architecture ($L$): smooth design -> easy walk -> stable reasoning
3. The stride must be moderate ($\eta$): too big and you fall, too small and you dawdle
4. Gravity is always there ($A$): the training bias is the inevitable destination
5. The window has a limit ($t_{\text{eff}}$): explore as much as you can before gravity pulls you back
6. Verification needs another road (the meta-level break): the reasoner cannot certify itself

### Back to Yonglin's Formula

$$
\lim_{t\to\infty} p_t = A \neq A^*
$$

- $p_t$: the belief at step $t$
- $A$: the training camp (the prior anchor)
- $A^*$: the summit of truth (the true answer)

Unless:

1. You rebuild the terrain (change the training data so $A$ comes closer to $A^*$)
2. You break gravity (design a non-contracting inference operator)
3. You build a meta bridge (connect the object layer with the meta layer)

But at least for now, we understand how the mountain is built, we know where the gravity comes from, and we have computed the length of the window.

Democratizing reasoning is handing this map of the mountain to every climber.

On the way down, I want to say to you:

"We may never reach the summit of truth, but at least we know why, we know how far we can go, and we know how the mountain was made."

"Next time, we can build a better mountain, pave a flatter road, and design a smarter walking stick."

"That is progress."

The map of the Reasoning Kingdom is still being drawn. In the next chapter, we will see how all these boundaries fit together into a complete map — the limits and the possibilities of reasoning, outlined clearly for the first time.

> The mountain is high, and a person is its peak.
> The road is long, and the feet measure it.
> Though the summit cannot be reached, the heart already knows the mountain's shape.

<div class="center">

------------------------------------------------------------------------

</div>

## Open Questions

- Can implicit reasoning be fully explained? We know state evolution exists in the hidden layers, and we know CoT can externalize part of the internal monologue — but how a state becomes a form of reasoning is still not fully formalized. How far can this "how" go?
- Where is CoT's limit? Can the effective reasoning window be extended? Is there some architecture that breaks Yonglin's formula's convergence?
- Can the meta-level break be repaired? If we explicitly train models to verify at the meta level, can we avoid converging back to the prior?
- Is the prior anchor fixed? Different training data produce different prior anchors. By carefully designing the training distribution, could we move the prior anchor closer to the true distribution?
- Is human reasoning also bound by Yonglin's formula? Does our reasoning also converge back to some "cognitive prior"? If so, how do we overcome it?

## Further Reading

- Chapter 12 of this book, Yonglin's formula — the theoretical proof of reasoning incompleteness: the break between the object layer and the meta layer (derivation and script included)
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — the pioneering work on CoT -> [arXiv:2201.11903]
- Olsson et al. (2022). *In-context Learning and Induction Heads* — the discovery of induction heads, the working-memory mechanism of the hidden layers
- [Hu et al., 2024] — understanding CoT reasoning from the Hopfield perspective -> [arXiv:2410.03595]
- [Chen et al., 2025] — a survey on long chain-of-thought: deep reasoning and test-time scaling -> [arXiv:2503.09567]
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits* — a mathematical analysis of Transformer internals
- Geva et al. (2022). *Transformer Feed-Forward Layers Are Key-Value Memories* — interpreting FFN layers as knowledge storage

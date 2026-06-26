# Bonus Chapter: From Theory to Code — CocDo Neural Causal Operators

> Theory tells you what causal inference is. Code tells you how it gets done.

---

:::details A Pallas's Cat Mini-Reflection from the Reasoning Kingdom: From Theory to Code — What Exactly Are We Translating?

The first volume's 13 chapters covered the theory: causal boundaries, attention mechanisms, the art of search, the limits of reasoning. Now the question is: how do these theories become runnable code?

This seems like an engineering problem, but it has a fundamental obstacle before the very first line of code: **what gets lost in the translation between mathematical structure and computational structure?**

Linear attention is a perfect example. Its algebraic core is a Monoid — the associativity of prefix sums allows $O(\log T)$ parallel scanning, effortlessly shedding the $O(T^2)$ complexity curse. This is mathematical structure directly delivering computational dividends.

But a Monoid has no inverse. Once $k_t \otimes v_t$ is added to the state matrix $S$, it is permanently aliased with all other historical contributions and cannot be individually extracted. And Pearl's do operator requires: **sever all incoming edges of variable $X$, leaving other edges untouched.** Translated into the language of state matrices, this means: subtract the outer product contribution of a specific token from $S$. This cannot be expressed in a Monoid at all—it's not an implementation difficulty; the algebraic structure lacks the "undo" operation.

This is the fundamental misalignment between mathematical structure and causal semantics: the Monoid's elegance comes from associativity; causal inference's precision comes from invertibility. The two are algebraically incompatible.

So you face a choice: **preserve the Monoid's complexity advantage and give up the full semantics of the do operator; or abandon the Monoid, preserve the do operator, and bear the $O(n^2)$ storage cost.**

The Spartacus model chose the first path, pushing the memory mechanism to its limit within the Monoid framework: vectorized decay gates give different dimensions independent memory lifespans—fast-decaying dimensions handle local syntax, slow-decaying dimensions handle long-range entity memory. It proves that even without upgrading to a Group, the vectorized decay structure can already approximate the behavior of "addressable memory."

CocDo chose the second path: abandon compressed history, preserve the complete causal adjacency matrix $A$, giving every edge a name, an address, and the ability to be individually severed. The cost is $O(n^2)$ graph storage; the payoff is the full semantics of the do operator.

The fork between these two paths is precisely whether the Monoid has an inverse element.

But this is not the end of the story. The translation from theory to code exposes other misalignments: how does the type system exclude cycles? How does β-reduction correspond to structural equation propagation? How does gradient descent become the inverse of the do operator? What is the relationship between attention mechanisms and causal discovery?

The answers to these questions are not in the theory—they are in the code. This chapter showcases CocDo—a neural causal model library that fuses Pearl's causal calculus with the Calculus of Constructions (CoC) type theory. It is not a toy implementation, but a complete system that can train, intervene, and plan on real data.

The translation process will lose some elegance, introduce some compromises, but will also expose structures invisible in pure theory—like "the attention matrix is the soft causal adjacency matrix," like "gradient planning is the calculus version of search," like "the system can use itself as a knowledge graph."

Theory tells you what causal inference is. Code tells you how it gets done—and why some things, within existing mathematical structures, simply cannot be done.

:::

---

The first volume traversed 13 chapters: from entropy increase and prediction (Chapter 1), to the ceiling of symbolic systems (Chapter 2), to the boundaries of causality (Chapter 6), attention mechanisms (Chapter 9), the art of search (Chapter 10), and finally stopping before the boundaries of reasoning (Chapter 13).

How do these theories become runnable code?

This chapter showcases [CocDo](https://github.com/lizixi-0x2F/CocDo)—a neural causal model library that fuses Pearl's causal calculus with CoC type theory. It is not a toy implementation, but a complete system that can train, intervene, and plan on real data.

The translation process from theory to code exposes problems invisible in pure mathematics: how does the type system exclude cycles at the structural level? What is the relationship between β-reduction and matrix propagation? How does gradient descent become the inverse of the do operator? What is the relationship between attention mechanisms and causal discovery?

---

## B.1 The Type System Excludes Cycles

Chapter 6 said that causal graphs must be acyclic—causality cannot form loops, otherwise it creates a paradox in time. But how is "acyclic" guaranteed in code?

The traditional approach is a runtime check: after constructing the graph, run a topological sort or DFS, and report an error if a cycle is found. The problem is that this error occurs after you have already spent a lot of time constructing the graph and training the model.

CocDo uses the type system to exclude cycles at **compile time**. The core idea is a single sentence:

> Each causal edge $X \to Y$ is encoded as a dependent Pi type $\Pi(X : \text{Type}_i).\, \text{Type}_j$, requiring $i < j$.

:::details What is Dependent Type Theory (CoC)?
**Calculus of Constructions (CoC)** is a type theory where types themselves can depend on values.

Key concepts:
- **Sort (universe)**: `Sort(i)` represents the type universe at level $i$, `Sort(0)` is the most basic type layer
- **Pi type**: $\Pi(x : A).\, B$ is the dependent function type—"for any $x$ of type $A$, return type $B$"
- **Level constraint**: CoC requires strict stratification of type universes; the type of $\text{Sort}(i)$ is $\text{Sort}(i+1)$, preventing self-referential paradoxes

CocDo borrows this structure: each causal node is assigned a level $i$, and the Pi type's level constraint $i < j$ is automatically equivalent to "the causal graph is acyclic."
:::

$i$ and $j$ are levels in the topological order—root nodes are $\text{Type}_0$, their children are $\text{Type}_1$, and so on. Requiring $i < j$ means: **an edge can only point from a lower level to a higher level**.

What does a cyclic graph mean? It means there exists a path $X \to \cdots \to X$, such that $X$'s level must be both less than some intermediate node and greater than it. This is contradictory in the type system—not a runtime error, but a type check failure:

```python
from cocdo.kernel.terms import Sort, Pi, Var
from cocdo.kernel.typing import type_of, Context

# Legal edge: X(level 0) -> Y(level 1)
ctx: Context = {"X": Sort(0), "Y": Sort(1)}
edge = Pi("X", Sort(0), Sort(1))   # ✓ 0 < 1

# Illegal edge: Y(level 1) -> X(level 0), forms a cycle
bad_edge = Pi("Y", Sort(1), Sort(0))  # type_of will reject this
```

The significance of this design is not just engineering defense. It says: **the acyclicity of a causal graph is not a runtime check, but a type invariant**. A causal model with cycles simply cannot be constructed in CocDo—just as in a strongly-typed language, you cannot assign a string to an integer variable.

::: info Professor Pallas's Cat comments
The type system elevates constraints from "runtime crash" to "compile-time rejection." This is not a technical detail; it is an epistemological step: you are not checking whether a model is legal; you are making it so that illegal models cannot be expressed. Chapter 14 says the value of formal systems is eliminating ambiguity; the value here is eliminating the possibility of an entire class of errors.
:::

---

## B.2 The Implementation of do(): Term Substitution in λ-Calculus

Chapter 6 defined the do operator—"replace $X$'s structural equation with the constant $X = v$, delete all incoming edges, and propagate the effect along descendants." This definition is mathematical, but how is it implemented in code?

In λ-calculus, this operation has a precise name: **capture-avoiding substitution**, denoted $[v/X]M$—replace all free occurrences of variable $X$ in term $M$ with $v$.

:::details 
β-reduction and capture-avoiding substitution: the two core operations of **$\lambda$**-calculus

**β-reduction** is the core computation rule of $\lambda$-calculus:

$$(\lambda x.\, M)\, N \;\to\; [N/x]M$$

Read as: applying the function $(\lambda x. M)$ to the argument $N$ is equal to replacing all $x$ in $M$ with $N$. This is the most basic step of "running a function."

**Capture-avoiding substitution $[N/x]M$**: replace free variable $x$ in $M$ with $N$, but be careful: if $M$ contains $\lambda x. \ldots$ (another function that binds a variable with the same name $x$), you must not enter that inner scope to perform the substitution—otherwise you would confuse the outer $x$ with the inner $x$, producing a semantic error. "Capture-avoiding" refers to avoiding this accidental binding.

**Significance in CocDo**: the semantics of `do(X=v)` is precisely "replace $X$'s structural equation in the causal graph with the constant $v$"—this is exactly capture-avoiding substitution. After substitution, running `beta_reduce` is "propagating the effect to descendant nodes along the topological order." Pearl's mathematical definition and the computational semantics of $\lambda$-calculus correspond perfectly here.
:::


CocDo's implementation:

```python
def subst(term, var: str, replacement):
    """Replace all free occurrences of var in term with replacement."""
    if isinstance(term, Var):
        return replacement if term.name == var else term
    elif isinstance(term, Lam):
        if term.var == var:          # bound variable shadows, stop substitution
            return term
        return Lam(term.var, term.domain, subst(term.body, var, replacement))
    elif isinstance(term, App):
        return App(subst(term.func, var, replacement),
                   subst(term.arg,  var, replacement))
    return term  # Const, Sort contain no free variables
```

"Capture-avoiding" handles a subtle situation: if $M$ contains a λ that binds a variable with the same name as $X$, the substitution should not enter that binding's scope—otherwise it would incorrectly replace an $X$ that actually refers to the outside. The `if term.var == var: return term` line in the code is this check.

After substitution, **β-reduction** is needed to simplify the result to normal form:

```python
def beta_reduce(term, steps=100):
    """Call-by-value reduction to a fixed point."""
    for _ in range(steps):
        reduced = _step(term)
        if reduced is term:   # fixed point: cannot reduce further
            break
        term = reduced
    return term
```

The core rule of `_step` is β-reduction: $(\lambda x. M)\, N \to [N/x]M$—simplifying function application to substitution. When both operands of `Add`/`Mul` are `Const` with values, the reducer directly computes:

```
App(App(Mul, Const(w=0.9)), Const(v=3.0))  ->  Const(2.7)
```

This means the propagation of the structural equation $E_j = \sum_i A_{ij} \cdot E_i + U_j$ **happens inside the term language**, not as a separate matrix multiplication. The semantics of the do operator and the computational semantics are the same thing.

::: info Correspondence between do() and β-reduction
| Pearl's operation | λ-calculus operation |
|-------------|-----------|
| Replace $X$'s equation with $X = v$ | `subst(mechanism, "X", Const(v))` |
| Delete all incoming edges of $X$ | After substitution, parent node terms vanish, no longer appear on the reduction path |
| Propagate effect along descendants | `beta_reduce` reduces to a fixed point along topological order |
| Cyclic graph illegal | Pi type requires strictly increasing levels; a cycle is a `TypeError` |
:::

---

## B.3 NOTEARS: Learning Causal Structure from Data

Chapter 6 said "observational data is never enough"—because correlation does not equal causation. But we still need to learn causal structure from data; this is the **causal discovery** problem.

Traditional methods (PC algorithm, FCI) are combinatorial search—finding the DAG that best fits the data among all possible DAGs. With $n$ nodes, the number of DAGs is super-exponential, making the search cost extremely high.

In 2018, Zheng et al. proposed NOTEARS, which transforms the discrete constraint "is it a DAG" into a **continuous differentiable equality constraint**:

$$h(A) = \mathrm{tr}(e^{A \circ A}) - n = 0 \iff A \text{ is a DAG}$$

where $A \circ A$ is element-wise squaring, and $e^M$ is the matrix exponential. This equality holds if and only if $A$ is a directed acyclic graph.

:::details NOTEARS: Using the matrix exponential to turn "is it acyclic" into continuous optimization (prior work: Zheng et al., 2018)
The traditional difficulty of **causal discovery**: one must search among all possible DAGs (directed acyclic graphs) for the one that best fits the data. The number of DAGs for $n$ nodes is super-exponential—brute-force search is completely infeasible.

**NOTEARS' key insight**: transform the discrete constraint "is a DAG" into a smooth continuous equality:

$$h(A) = \text{tr}(e^{A \circ A}) - n = 0$$

- $A$ is the edge weight matrix, $A[i,j]$ represents the weight of edge $i \to j$
- $A \circ A$ is element-wise squaring (making negative values positive)
- The trace of $e^M$ satisfies $\text{tr}(e^M) \geq n$, with equality if and only if the matrix is acyclic

With this continuous constraint, causal discovery becomes a **gradient optimization** problem—it can be directly solved using standard optimizers like Adam, without combinatorial search. This is the key step that makes causal structure learning feasible within a neural network framework.

CocDo uses $\|A\|_F^2$ as an approximation for sparse graphs, avoiding numerical overflow from the matrix exponential.
:::


With this constraint, causal discovery becomes a continuous optimization with an equality constraint:

$$\min_A \mathcal{L}_{\text{recon}}(A) + \lambda h(A) + \frac{\rho}{2} h(A)^2$$

:::details 
What is the reconstruction loss $\mathcal{L}_{\text{recon}}$?
**Reconstruction loss** measures how well the learned causal weight matrix $A$ explains the data.

Under the linear structural equation model (SEM) assumption, each variable is a linear combination of its parent nodes plus noise: $X = XA + \epsilon$. Therefore, if $A$ is the weight matrix of the true causal graph, $XA$ should approximately reconstruct $X$ itself:

$$\mathcal{L}_{\text{recon}}(A) = \frac{1}{N}\|X - XA\|_F^2$$

This is a least-squares objective—maximizing the degree to which the causal graph "explains" the data. Minimizing it alone would cause $A$ to degenerate into a fully connected graph (perfect explanation but not a DAG), so it must be jointly optimized with the acyclicity constraint $h(A) = 0$.
:::

:::details Augmented Lagrangian Method

The standard Lagrangian method handles the equality constraint $h(A) = 0$:

$$\mathcal{L}(A, \lambda) = f(A) + \lambda h(A)$$

But the pure Lagrangian method is numerically unstable; the multiplier $\lambda$ is difficult to converge. The **augmented Lagrangian method** adds a quadratic penalty term:

$$\mathcal{L}(A, \lambda, \rho) = f(A) + \lambda h(A) + \frac{\rho}{2} h(A)^2$$

The algorithm alternates two steps:
1. Fix $\lambda, \rho$, perform gradient descent on $A$ (inner optimization)
2. Update the multiplier: $\lambda \leftarrow \lambda + \rho \cdot h(A)$; if the constraint converges too slowly, increase $\rho$

The larger $\rho$, the stronger the penalty on the acyclicity constraint, forcing $A$ toward being a DAG. In the code, every 50 steps, it checks whether $h(A)$ has decreased sufficiently; if not, $\rho$ is multiplied by 10.
:::

Solved using the augmented Lagrangian method, tightening the multiplier $\lambda$ and penalty coefficient $\rho$ every few steps. Below is CocDo's complete training loop on synthetic data (from `examples/demo_gcastle.py`):

```python
# Use gCastle to generate a 5-node linear Gaussian DAG, 1000 samples
from castle.datasets import DAG, IIDSimulation
true_dag = DAG.erdos_renyi(n_nodes=5, n_edges=6, seed=42)
dataset  = IIDSimulation(W=true_dag, n=1000, method="linear", sem_type="gauss")
X = dataset.X  # (1000, 5)

# Train CausalFFNN to learn the causal weight matrix A
ffnn = CausalFFNN(d_embed=32, hidden=128)
rho, lam = 1.0, 0.0

for epoch in range(500):
    A, _ = ffnn(E_raw)
    recon = ((X @ A - X) ** 2).mean()
    h = acyclicity_loss(A)  # ≈ tr(e^{A∘A}) - n
    loss = recon + lam * h + (rho / 2) * h ** 2
    loss.backward(); optim.step()
    
    # Tighten constraint every 50 steps
    if (epoch + 1) % 50 == 0:
        lam += rho * h.item()
        if h > 0.25 * h_prev:
            rho = min(rho * 10, 1e6)
```

After training, `A` is the learned causal weight matrix. CocDo automatically extracts the topological order from `A` (Kahn's algorithm) and constructs `NeuralSCM`.

**Attention is causal discovery.** `CausalFFNN` uses low-rank bilinear scoring:

$$\text{score}(i \to j) = \frac{(W_q h_i) \cdot (W_k h_j)^\top}{\sqrt{r}}$$

This is mathematically identical to Transformer attention. The only differences are: using sigmoid instead of softmax (edges compete independently), and forcing the diagonal to zero (a variable cannot be its own cause).

::: info Attention is causal discovery
A Transformer's attention heads compute "the influence weight of token $i$ on token $j$"—this is precisely the definition of the causal weight matrix $A[i,j]$. The difference is that Transformers impose no acyclicity constraint and do not use do-calculus to interpret these weights. CocDo fills in both of these gaps.
:::

:::details Abstract Algebra Interlude: Why Linear Attention Is Inherently Unable to Do the do Operator (Yes, It's Pallas's Cat's Research Again)

Standard attention requires accessing the full KV cache at each step, at a cost of $O(T)$ storage. Linear attention's starting point is to compress the entire causal history into a fixed-size state matrix $S_t$:

$$S_t = \text{diag}(\alpha_t) \cdot S_{t-1} + k_t \otimes v_t, \qquad o_t = q_t \cdot S_t$$

Behind this recurrence lies an algebraic structure. Define the binary operator:

$$(\alpha,\, S) \oplus (\beta,\, X) = (\alpha \cdot \beta,\; \text{diag}(\beta) \cdot S + X)$$

This operator satisfies **associativity**, with identity element $(1, 0)$. Associativity + identity — this is a **Monoid**.

The engineering dividend from associativity is direct: during training, one can do $O(\log T)$-depth parallel prefix scanning; during inference, only $O(1)$ state updates per step. That linear attention can run at all—the Monoid is the algebraic root cause.

**But Monoid has no inverse.**

A Group = Monoid + inverse. In the matrix addition group, $S + (-S) = 0$; you can "undo" an update. There is no such operation in a Monoid—once $k_t \otimes v_t$ is added into $S_t$, it is permanently mixed with the contributions of all previous tokens, impossible to extract individually, impossible to sever individually.

Pearl's do operator requires: **sever all incoming edges of variable $X$, leaving other edges untouched.** Translated into the language of state matrices, this means: from $S_t$, subtract the outer product contribution $k_j \otimes v_j$ of a specific token. This cannot be expressed in a Monoid at all—it's not an implementation difficulty; it's a missing algebraic structure.

Spartacus goes further within this framework: it uses **vectorized decay** instead of scalar decay, $\alpha_t \in (0,1)^d$ giving each feature dimension an independent memory lifespan—fast-decaying dimensions handle local syntax, slow-decaying dimensions handle long-range entity memory. The decay gate uses sigmoid activation, with bias initialized to $3.0$ (so $\sigma(3) \approx 0.95$), and the model starts training from "remembering almost everything." Padding positions (PAD) are designed as the Monoid's identity element: $\alpha = 1, k = v = 0$, and the state matrix passes through unchanged, making variable-length sequences in batch processing completely transparent to the recurrence.

This is an attempt to push the memory mechanism to its limit within the Monoid structure. It does not solve the do operator problem—because the Monoid itself does not allow it. But it proves: even without upgrading to a Group, the vectorized decay structure can already let different dimensions take on different time scales, approximating the behavior of "addressable memory" without truly addressing it.

CocDo chose the other path: abandon compressed history, preserve the complete causal adjacency matrix $A$, giving every edge a name, an address, and the ability to be individually severed. The cost is $O(n^2)$ graph storage; the payoff is the full semantics of the do operator.

The fork between these two paths is precisely whether the Monoid has an inverse element.

> For Spartacus' full implementation and model weights, see [NoesisLab/Spartacus-1B-Instruct](https://huggingface.co/NoesisLab/Spartacus-1B-Instruct).
:::

---

## B.4 Gradient Planning: The Calculus Version of Search

Chapter 10 discussed the art of search—navigating vast reasoning spaces. Traditional search is sampling: try many times, remember which ones were good. Gradient planning is calculus: walk along the gradient direction.

The do operator is forward: given an intervention value $v$, compute the result $Y$. The more common problem in practice is the reverse: **given a target $Y = y^*$, find the optimal intervention value $v^*$**.

$$v^* = \arg\min_v \sum_j \left(\|E_{\text{next}}[j]\| - y^*_j\right)^2$$

where $E_{\text{next}}$ is the embedding state after do operator propagation, using L2 norm rather than full vector comparison to eliminate direction alignment issues.

The core of `CausalPlanner` is a differentiable single-step propagation:

```python
# do-calculus in embedding space:
A_do  = A * (1 - col_mask)      # zero out the incoming edge columns of intervened nodes
E_do  = (1 - row_mask) * E + row_mask * interv_E   # inject intervention values
E_next = A_do.T @ E_do + U      # structural equation propagation
```

Then compute the gradient with respect to the intervention value $a$, optimizing with Adam:

```python
a = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.Adam([a], lr=0.05)
for _ in range(200):
    E_next = planner._step(a, interv_nodes, E_init)
    energy = ((E_next[target_idx].norm(dim=-1) - scalar_targets) ** 2).sum()
    energy.backward(); opt.step()
```

The entire computation graph backpropagates from the target all the way to the intervention value—no sampling needed, no reinforcement learning.

::: info Professor Pallas's Cat comments
Reinforcement learning turns planning into a sampling problem: try many times, remember which ones were good. Gradient planning turns planning into a calculus problem: walk along the gradient direction. The latter requires a differentiable world model—this is precisely what neural SCMs provide. The cost is that the world must be differentiable, or at least approximable by a differentiable model. This assumption does not always hold, but in embedding space it is usually good enough.
:::

---

## B.5 CausalSearch: The System Uses Itself as a Knowledge Graph

Chapter 13 discussed the boundaries of reasoning, mentioning that systems begin to reason about themselves—this is the beginning of self-reference. CausalSearch is a concrete self-referential application: **using the chapters of the Reasoning Kingdom as a causal knowledge graph**.

The previous four sections dealt with "variables"—scalars or vectors, with explicit numerical meanings. But knowledge can also be nodes in a causal structure: one paragraph depends on another, one concept is built upon another.

`demo_causal_search.py` embeds all chapters of the Reasoning Kingdom (22 chapters, ~1800 paragraphs) using BGE, trains `CausalFFNN` to learn the causal weight matrix $A$ between paragraphs, and then uses Pearl's three-step method for retrieval:

**Step 1: Abduction**—find the paragraph $j^*$ nearest to the query.

**Step 2: Action**—$\text{do}(j^* = \text{query\_emb})$: inject the query vector, zero out $j^*$'s incoming edges.

**Step 3: Prediction**—$E_{\text{next}} = A_{\text{do}}^\top E_{\text{do}} + U$; sort all nodes by $\Delta\|E_{\text{next}}\|$.

Positive $\Delta$ = downstream activation (knowledge chain triggered by the query); Negative $\Delta$ = upstream prerequisites (concepts needed to understand the query).

```
Query: "The relationship between Transformer attention and Bayesian inference"

[Vector Retrieval RAG]
  1. ch9·The success of Transformers triggered...  cos=0.763
  2. ch9·Attention as causality...                cos=0.682

[CausalSearch · Pearl's Three Steps]
  Abduction anchor -> ch9·The success of Transformers...

  + Downstream activation:
    ch17·Comparison of Bayesian updating and ch14...  Δ=+2.69e-02
    ch20·PAC and Bayesian: continuation of ch17...    Δ=+2.34e-02

  * Unique to CausalSearch (missed by RAG):
    ch17 Bayesian inference ×4, ch1 generative model layer, ch19 proofs...
```

Vector retrieval finds "surface similarity"; CausalSearch finds "causal relatedness"—propagating along learned causal edges rather than measuring distance in embedding space.

::: info Professor Pallas's Cat comments
RAG is the first rung of the ladder: association. CausalSearch is the second rung: intervention. What you're asking is not "which paragraphs are similar to this query," but "if I inject this query into the knowledge graph, which nodes will be activated." These are two completely different questions; they just happen to both be called "retrieval."
:::

---

## B.6 Review: The Mapping from Theory to Code

Having reached this point in the first volume, we can draw a complete line:

| Volume 1 Chapter | Core Problem | Corresponding in CocDo |
|---------|---------|-----------------|
| Chapter 6 Causal Boundaries | Observational data cannot deduce causality | NOTEARS: learning DAG structure from data |
| Chapter 6 do operator | Intervention vs. observation | `subst` + `beta_reduce` = implementation of do |
| Chapter 9 Transformer | Mathematical structure of attention | `CausalFFNN`: low-rank bilinear scoring |
| Chapter 9 Bonus Attention as Causality | Attention matrix is the soft causal adjacency matrix | sigmoid edge weights, diagonal forced to zero |
| Chapter 10 The Art of Search | Navigating reasoning spaces | `CausalPlanner`: gradient planning, not sampling |
| Chapter 13 Boundaries of Reasoning | System reasoning about itself | `CausalSearch`: using its own chapters as a knowledge graph |

This is not a coincidence. Every chapter of the first volume asks: **what is reasoning, where are its boundaries, how does it work under real constraints?** CocDo is a runnable answer to these questions—incomplete, but honest.

The value of formalization is not only mathematical rigor, but also that it can be implemented. The CoC type system makes cycles into type errors; NOTEARS turns discrete search into gradient optimization—these are the engineering dividends of formalization.

---

## Thought Exercises

**★ Warm-up**

1. In CocDo, why is `sigmoid` used instead of `softmax` to compute edge weights $A[i,j]$? What impact does this choice have on the sparsity of the causal graph?
2. What situation does the "capture-avoiding" handling in the `subst` function address? Give an example of what would go wrong without this check.

---

**★★ Derivation**

Consider a three-node graph $X \to Y \to Z$, with structural equations:

$$E_Y = w_{XY} \cdot E_X + U_Y, \quad E_Z = w_{YZ} \cdot E_Y + U_Z$$

1. Manually compute the value of $E_Z$ after $\mathsf{do}(X = v)$ (express in terms of $w_{XY}, w_{YZ}, v, U_Y, U_Z$).
2. Verify your calculation using CocDo's `step` method, setting `rollout_steps=2`. Why is `rollout_steps=2` needed instead of 1?
3. If only `rollout_steps=1` were used, what would the value of $E_Z$ be? Which rung of Pearl's ladder does this correspond to?

---

**★★★ Challenge**

NOTEARS' acyclicity constraint $h(A) = \mathrm{tr}(e^{A \circ A}) - n = 0$ is a necessary and sufficient condition. CocDo uses $\|A\|_F^2$ as an approximation.

1. Prove: $\|A\|_F^2 = 0 \iff A = 0$ (the trivial DAG). In what situations is this approximation exact?
2. Construct a non-zero DAG (with at least one edge) such that $h(A) = 0$ but $\|A\|_F^2 > 0$. In what situations does this approximation fail?
3. During actual training, the reconstruction loss $\mathcal{L}_{\text{recon}}$ pushes $A$ to explain the data, while $\|A\|_F^2$ pushes $A$ toward zero. What is the equilibrium point of these two forces? How does it relate to L1 regularization?

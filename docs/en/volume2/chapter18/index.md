# Chapter 18: Formalizing Causal Structure — The Three-Rung Ladder and do-calculus

> From data alone, one can never deduce causation. Unless you are willing to admit that certain structures are assumptions, not discoveries.

---

Chapter 17 left us with an uncomfortable fact: probability theory, however refined, cannot distinguish "$X$ causes $Y$" from "$Y$ causes $X$" — as long as both produce the same joint distribution, Bayesian updating treats them identically.

This is not a problem of computational power, nor of data quantity, but a structural limitation of mathematics: **observation describes how the world looks when it is still, not how the world looks after it is prodded**.

Yet we prod the world constantly. Doctors prescribe medicine, policymakers adjust tax rates, engineers modify parameters. Every intervention asks a question that probability theory cannot answer: "If I change $X$, what happens to $Y$?"

Answering this question requires a new kind of inference rule. The task of this chapter is to formalize the very act of "changing."

---

## 18.0 The Causal Wire-Cutting Game: Observation Is Not Intervention

In the probability game, you can only see how the chips move. In the causal game, you finally get to cut the wires.

Imagine a causal circuit board. Each variable is a node, each directed edge is a wire: $X \to Y$ means that changes in $X$ travel along this wire to affect $Y$. When observing, you merely see that a certain node lights up; when intervening, you pick up scissors, cut all incoming edges to this node, and forcibly fix it at a certain value.

This is the game meaning of $do(X=x)$.

"Seeing $X=x$" does not alter the circuit. It merely tells you: right now this light is glowing as $x$. But "executing $do(X=x)$" changes the circuit structure: all wires that originally influenced $X$ are cut; $X$ is no longer determined by its parent nodes, but by your hand.

:::details The Formal Skeleton of This Game
- **State space**: a directed acyclic graph $G$, together with a set of structural equations $X_i = f_i(\mathrm{Pa}(X_i), U_i)$.
- **Legal moves**: observe certain nodes; or execute intervention $do(X=x)$, cutting all incoming edges to $X$ and fixing its value.
- **Transition rules**: observation updates the distribution; intervention replaces structural equations, yielding the mutilated graph (the graph after wire-cutting).
- **Victory condition**: identify $P(Y \mid do(X=x))$ or counterfactual quantities from the observable distribution and causal graph.
- **Failure mode**: treating $P(Y \mid X=x)$ as $P(Y \mid do(X=x))$, misreading correlation as causation.
:::

The enjoyable part of this game is that it turns "causation" from a metaphysical debate into a very concrete action: which wire to cut? Which wire to keep? After cutting, how does the signal propagate?

A backdoor path is a wire that sneaks around and comes in from behind; a confounding variable is a hand hidden behind the scenes pulling both nodes simultaneously; d-separation is the judgment of whether, after certain wires are blocked, information can still travel from one side to the other.

::: info Professor Pallas's Cat's Commentary
The first lesson of causal inference is brutal: seeing is not changing. Seeing the sprinkler on, versus turning the sprinkler on with your own hand, are not the same world. The most common arrogance in statistics is treating observation as intervention. The scissors of causal graphs are made precisely to cut away this arrogance.
:::

## 18.1 The Three-Rung Ladder

Judea Pearl uses a metaphor to describe the hierarchical structure of causal inference; he calls it the **Ladder of Causation**. The ladder has three rungs, from low to high, each requiring capabilities that the previous rung lacks.

**First Rung: Association**

This is the territory of probability theory. The form of the question is:

> "Seeing $X$, what is $Y$?"

In mathematical language: $P(Y \mid X)$. This rung requires only data — sufficiently many observations suffice to estimate conditional probabilities. Animals, infants, and nearly all statistical models live on this rung.

**Second Rung: Intervention**

The form of the question becomes:

> "If I set $X$ to a certain value, what would $Y$ be?"

The difference from the first rung is fundamental. "Seeing $X = x$" and "setting $X$ to $x$" are two completely different things. The former is passive observation, the latter is active manipulation.

In passive observation, $X = x$ may arise because some common cause $Z$ simultaneously affects both $X$ and $Y$. In active intervention, you sever the connection between $X$ and all its causes, forcibly fixing $X$ at $x$ — the variation in $Y$ at this point truly originates from the direct influence of $X$ on $Y$.

This rung requires not only data but also **action** — or, when action is infeasible, some tool that allows you to simulate action mathematically.

**Third Rung: Counterfactual**

The question becomes even harder:

> "If back then $X$ had not been that value, what would $Y$ have been?"

This is already an inquiry about a single individual in another possible world. "This patient recovered after taking the medicine — if he had not taken the medicine back then, would he still have recovered?" This question cannot be answered directly by any observation or experiment, because the parallel world of "not taking the medicine" is one we can never enter.

::: info The Three-Rung Ladder Is a Real Stratification, Not Philosophical Rhetoric
These three rungs correspond to genuine boundaries of capability. Pure observational data can only answer first-rung questions. Randomized controlled trials (RCTs) can answer second-rung questions, but at the cost of actually executing interventions. Counterfactual inference requires a complete causal model, plus additional assumptions about "individual mechanisms" — this exceeds the capability of any experiment. Most statisticians spend the better part of their careers working at the first rung, mistakenly believing they are answering second-rung questions. This confusion has produced erroneous inferences across vast swaths of scientific literature.
:::

::: info Professor Pallas's Cat's Commentary
"Mistakenly believing" is too polite. This is not a cognitive error; it is a tool error. When the only hammer you hold is correlation, every problem looks like a correlation nail. The issue is not the intellect of statisticians, but that standard training has never clearly drawn the boundary between the first and second rungs.
:::

---

## 18.2 Graphs: The Geometry of Causation

The first step is to give "causal structure" a precise mathematical representation.

**Directed acyclic graphs (DAGs)** are the most natural tool. Nodes represent variables, directed edges represent direct causal influence: $X \to Y$ means $X$ directly influences $Y$. "Acyclic" is an important constraint — causation cannot form loops (if $X$ causes $Y$ causes $X$, then a temporal paradox arises).

A simple example. Consider three variables: season ($S$), whether the sprinkler is on ($W$), whether the grass is wet ($G$). The intuitive causal structure is:

$$S \to W, \quad S \to G, \quad W \to G$$

Season affects the sprinkler (more likely on in summer), season also directly affects the grass (rain), and the sprinkler also directly affects the grass.

In this graph, $W$ and $G$ are correlated — when the sprinkler is on, the grass is more likely wet. But this correlation has two paths: one direct causal path $W \to G$, and another "backdoor path" through the common cause $S$: $W \leftarrow S \to G$.

Probability theory sees the superposition of both, unable to distinguish them. The graph explicitly draws out this structure.

::: info Graphs Are Not Read Out of Data
This is the most common beginner's mistake. A causal graph represents **domain knowledge**, not a product of statistical inference. You cannot derive this graph from $P(S, W, G)$ — data only tells you the degree of correlation among variables, not the direction of the arrows. The causal graph is where you write down the operating mechanisms of the world; it is an assumption, not a discovery. Accepting this requires an uncomfortable cognitive leap: scientific inference does not proceed purely from data; it always enters the field carrying structural assumptions.
:::

::: info Professor Pallas's Cat's Commentary
The direction of the arrows is brought in by you, not given by the data — this sentence will make many data scientists uncomfortable, because they are accustomed to "letting the data speak." But the data cannot speak on this matter. Admitting this requires courage; many papers evade this step. The cost of evasion is burying the assumption inside the method, pretending it is the conclusion.
:::

---

## 18.3 Structural Causal Models

A graph is only a skeleton. To turn it into a machine capable of inference, content must be attached to every edge. This is the **Structural Causal Model (SCM)**.

An SCM consists of three parts:

**Exogenous variables** $U$: noise or background factors coming from outside the system, not determined by any other variable within the model. They capture all randomness that we cannot observe or do not intend to model.

**Endogenous variables** $V$: variables determined by other variables within the model, including all observables of interest.

**Structural equations**: for each endogenous variable $V_i$, there is an equation

$$V_i = f_i(\mathsf{Pa}(V_i),\, U_i)$$

where $\mathsf{Pa}(V_i)$ are the **parent nodes** (direct causes) of $V_i$ in the causal graph, and $U_i$ is the noise term for that variable.

This equation is not a statistical regression equation — it is a **mechanism**, describing "given the parent nodes and noise, what value does this variable take." This mechanism is stable and local: changing the equations of other variables does not affect this equation. This local stability is the core feature that distinguishes causal models from statistical models.

Returning to the sprinkler example, the structural equations could be:

$$W = f_W(S, U_W), \quad G = f_G(S, W, U_G)$$

$W$ is determined by season and its own noise, $G$ by season, sprinkler state, and its own noise. Each equation describes a local mechanism, independent of the others.

---

## 18.4 The do Operator: Formalizing Intervention

Now we can precisely define "intervention."

The operational definition of **intervention** $\mathsf{do}(X = x)$ is: **replace the structural equation of $X$ with the constant equation $X = x$**, while keeping all other equations unchanged.

The effect of this operation on the graph is intuitive: **delete all edges pointing into $X$**. Because $X$ is forcibly fixed, the influence of its parent nodes upon it is severed. The descendants of $X$ are still affected by $X$ (outgoing edges of $X$ are preserved), but $X$ no longer responds to its causes (incoming edges disappear).

This "surgically altered graph" is called the **intervention graph**, denoted $G_{\overline{X}}$ (the graph with incoming edges to $X$ deleted).

:::details 
The do Operator vs. Conditioning: 
Why are **$P(Y \mid X=x)$** and **$P(Y \mid \mathsf{do}(X=x))$** different?
This is the most central distinction in causal inference, and also the easiest to confuse:

**Conditioning $P(Y \mid X=x)$**: Among the samples where $X=x$ is observed, what is the distribution of $Y$? This is **passive observation** — you are asking "in the world where $X$ happens to equal $x$, what does $Y$ look like?" The problem is that the causes of $X=x$ may share common causes with $Y$ (confounding variables), distorting the correlation.

**Intervention $P(Y \mid \mathsf{do}(X=x))$**: If I **forcibly** fix $X$ at $x$ (deleting all causes of $X$), what would $Y$ be? This is **active manipulation** — you have severed the confounding paths, leaving only the direct causal effect of $X$ on $Y$.

Concrete example:
- $P(\text{recovery} \mid \text{medication}=1)$: among those who chose to take medication, what is the recovery rate (possibly inflated, because patients with milder symptoms are more likely to choose medication)
- $P(\text{recovery} \mid \mathsf{do}(\text{medication}=1))$: if I randomly force a group of people to take medication (randomized controlled trial), what is the recovery rate (true drug efficacy)

**The do operator = the mathematical expression of a randomized controlled experiment**. When RCTs are infeasible, do-calculus provides rules for estimating intervention effects from observational data.
:::


The probability distribution after intervention, denoted $P(Y \mid \mathsf{do}(X = x))$, sometimes also written $P_x(Y)$, is defined in the intervention graph $G_{\overline{X}}$.

Compare it with conditional probability:

$$P(Y \mid X = x) \quad \text{vs.} \quad P(Y \mid \mathsf{do}(X = x))$$

The former is "the distribution of $Y$ among samples where $X = x$ is observed." The latter is "if $X$ in the entire world were forcibly set to $x$, what would the distribution of $Y$ be."

The two can differ dramatically. In the sprinkler example, $P(G = \text{wet} \mid W = \text{on})$ includes the confounding effect of "it is summer so the sprinkler is on, it is summer so it may rain"; whereas $P(G = \text{wet} \mid \mathsf{do}(W = \text{on}))$ severs the path $W \leftarrow S$, preserving only the direct effect $W \to G$. This is the true "causal effect of the sprinkler itself on grass wetness."

::: info What Randomized Controlled Trials Actually Do
The core operation of a randomized controlled trial (RCT) is, mathematically, precisely $\mathsf{do}$: randomly assigning subjects to treatment and control groups severs the connection between the treatment variable (taking/not taking medicine) and all possible confounding factors. Randomization is equivalent to deleting all edges pointing to the treatment variable in the causal graph. This is why RCTs are the "gold standard" for estimating causal effects — they physically implement the do operator. The value of do-calculus lies in: it tells you when, without running an experiment, you can compute $P(Y \mid \mathsf{do}(X))$ from observational data.
:::

::: details Runnable do Operator: CocDo Implementation

The do operator is not merely a mathematical symbol — it can be precisely implemented as **term substitution plus β-reduction** in λ-calculus.

[CocDo](https://github.com/lizixi-0x2F/CocDo) encodes each causal variable as a node in COC type theory, and each edge $X \to Y$ as a dependent Pi type $\Pi(X : \text{Type}_i).\, \text{Type}_j$ (requiring $i < j$, making cycles inexpressible at the type level).

`do(X = v)` is implemented in only two steps:

```python
# 1. Replace variable X with constant v (severing all incoming edges)
intervened = subst(mechanism, var="X", replacement=Const("X", v))

# 2. β-reduction: propagate effects along topological order
result = beta_reduce(intervened)
```

`subst` is capture-avoiding substitution; `beta_reduce` is call-by-value reduction to a fixed point. When both operands of `Add`/`Mul` are `Const` with values, the reducer directly computes tensor operations:

```
App(App(Mul, Const(w)), Const(v))  →  Const(w · v)
```

This means the entire propagation process of the structural equation $E_j = \sum_i A_{ij} \cdot E_i + U_j$ occurs inside the COC term language, not as a separate matrix multiplication.

**Correspondence with Pearl's definition:**

| Pearl's do operator | CocDo implementation |
|----------------|-----------|
| Replace the structural equation of $X$ with $X = v$ | `subst(mechanism, "X", Const("X", v))` |
| Delete all edges pointing into $X$ | After substitution, the parent node terms of $X$ disappear |
| Propagate effects along descendants | `beta_reduce` reduces along topological order |
| Cyclic graphs are illegal | Pi types require $i < j$, cycles are `TypeError` |

```python
from cocdo import NeuralSCM
import numpy as np

# Three-node graph: ad_spend → clicks → revenue
A = np.array([[0, 0.9, 0.8],
              [0,   0, 0.7],
              [0,   0,   0]])
E = np.random.randn(3, 16)
scm = NeuralSCM.from_embeddings(["ad_spend", "clicks", "revenue"], A, E)

# do(ad_spend = 3.0): sever ad_spend's incoming edges, propagate effects
state, E_next = scm.step({"ad_spend": 3.0})
print(state)  # {"ad_spend": 3.0, "clicks": ..., "revenue": ...}
```

:::

---

## 18.5 The Backdoor Criterion: The Geometry of Confounding

The core question of do-calculus is: **under what conditions can causal effects be estimated from observational data** — that is, when can $P(Y \mid \mathsf{do}(X = x))$ be computed using $P$ and the graph structure, without actually performing an intervention?

Answering this question requires understanding the geometric structure of "confounding."

In a causal graph, the total effect of $X$ on $Y$ is "confounded" if and only if there exists a non-causal path originating from $X$, not passing through descendants of $X$, but capable of reaching $Y$. Such paths are called **backdoor paths** — they bypass the direct effect $X \to Y$ and sneak in spurious associations from behind.

The **Backdoor Criterion** gives a precise condition: a set of variables $Z$ satisfies the backdoor criterion (relative to $X \to Y$) if and only if:

1. No node in $Z$ is a descendant of $X$;
2. $Z$ blocks all backdoor paths from $X$ to $Y$.

If such a $Z$ exists, the causal effect can be computed using the following formula:

$$P(Y \mid \mathsf{do}(X = x)) = \sum_z P(Y \mid X = x, Z = z)\, P(Z = z)$$

This formula is called the **adjustment formula**. Its meaning is: for each value of $Z$, separately compute the conditional distribution of $Y$ given $X = x$, then take a weighted average according to the marginal distribution of $Z$. This operation is called **adjusting for $Z$**, also known in epidemiology as "controlling for confounding variables."

In the sprinkler example, $X = W$, $Y = G$, the backdoor path is $W \leftarrow S \to G$. Choose $Z = \{S\}$: $S$ is not a descendant of $W$ (satisfies condition 1), $S$ blocks the backdoor path $W \leftarrow S \to G$ (satisfies condition 2). Hence:

$$P(G \mid \mathsf{do}(W = w)) = \sum_s P(G \mid W = w, S = s)\, P(S = s)$$

This quantity is entirely determined by observational data; no actual manipulation of the sprinkler is needed.

---

## 18.6 The Three Rules of do-calculus

The backdoor criterion covers a large number of practical situations, but not all. In some causal graphs, backdoor paths cannot be fully blocked by any set of observable variables — for instance, when unobservable confounding factors exist.

To handle more general cases, Pearl proposed **do-calculus**: three inference rules concerning the $\mathsf{do}$ operator, which together constitute a complete calculus for causal inference.

Let $X, Y, Z, W$ be sets of variables in causal graph $G$, let $G_{\overline{X}}$ denote the graph with incoming edges to $X$ deleted, and $G_{\underline{X}}$ the graph with outgoing edges from $X$ deleted.

**Rule 1 (Insertion/deletion of observations)**:

$$P(Y \mid \mathsf{do}(X),\, Z,\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

if and only if in $G_{\overline{X}}$, $Y \perp\!\!\!\perp Z \mid X, W$.

Meaning: if in the intervention graph, $Z$ provides no additional information about $Y$ (blocked by $X$ and $W$), then whether $Z$ is observed does not affect the inference about $Y$.

**Rule 2 (Action/observation exchange)**:

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, Z,\, W)$$

if and only if in $G_{\overline{X}\,\underline{Z}}$, $Y \perp\!\!\!\perp Z \mid X, W$.

Meaning: under certain conditions, intervening on $Z$ and observing $Z$ have equivalent effects on $Y$ — the intervention loses its special "forcible severing" character and degrades to ordinary conditioning.

**Rule 3 (Insertion/deletion of interventions)**:

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

if and only if in $G_{\overline{X}\,\overline{Z(W)}}$, $Y \perp\!\!\!\perp Z \mid X, W$ (where $Z(W)$ are the nodes in $Z$ that are not ancestors of $W$).

Meaning: under certain conditions, $\mathsf{do}(Z)$ has no effect on $Y$ and can be deleted from the formula.

::: info The Completeness of the Three Rules
Pearl and Shpitser proved: for any causal graph, if $P(Y \mid \mathsf{do}(X))$ can be computed from the observational distribution $P$ (i.e., is "identifiable"), then by repeatedly applying these three rules one can always obtain an expression involving only observational probabilities. This is the **completeness theorem** of do-calculus — it is not merely a toolkit, but a complete calculus, omitting no causal effect that can be identified.
:::

The form of these three rules is highly similar to the inference rules of Chapter 14: above the line is the condition, below the line is the inference that may be drawn. The difference is that the "language" here is not merely propositions, but probability expressions carrying the $\mathsf{do}$ operator; the graph structure — conditional independence — plays the role of axioms.

Returning to the causal wire-cutting game, the three rules of do-calculus are the move table after wire-cutting. When can an observation be deleted? When can an action be exchanged for an observation? When can an intervention be entirely eliminated? The answer lies not on the surface of the formula, but in the graph after wire-cutting. Causal inference is stronger than probabilistic inference because it has incorporated "action" into the grammar; it is also more dangerous, because every action depends on the correctness of the graph you have drawn.

---

## 18.7 d-separation: The Independence Language of Graphs

Each of the three rules of do-calculus depends on a core judgment: in a given graph, are two sets of variables **conditionally independent** given a third set of variables? How does one directly read conditional independence from graph structure? This is precisely the job of **d-separation** (directional separation).

Given a directed acyclic graph $G$ and three sets of nodes $X, Y, Z$. $X$ and $Y$ are **d-separated** by $Z$, denoted $X \perp_G Y \mid Z$, if and only if $Z$ blocks all paths between $X$ and $Y$.

The definition of "blocking" depends on the types of nodes along the path:

- **Chain** $A \to B \to C$: when $B$ is in $Z$, the path is blocked ($B$ transmits information; controlling $B$ stops the flow of information).
- **Fork** $A \leftarrow B \to C$: when $B$ is in $Z$, the path is blocked ($B$ is a common cause; controlling $B$ makes the association vanish).
- **Collider** $A \to B \leftarrow C$: when $B$ is **not** in $Z$, the path is blocked (collider nodes block information by default; but controlling them or their descendants instead opens the path — this is the source of "collider bias").

The behavior of collider nodes is counterintuitive and worth pausing to think through clearly. Consider "height" ($H$) and "basketball skill" ($T$) both affecting "whether selected for the basketball team" ($S$): $H \to S \leftarrow T$. In the entire population, $H$ and $T$ may be independent. But if one looks only at the sample of those "already selected for the team" — equivalent to controlling $S$ — shorter players tend to have especially good skills, while taller players sometimes have mediocre skills. Controlling a collider node makes originally independent variables become correlated. This is **selection bias** in statistics, a natural consequence of the collider node being controlled and the path being opened, in the language of graphs.

---

## 18.8 Counterfactuals: Another World for a Single Individual

The highest rung of the three-rung ladder, counterfactuals, requires a precise definition within structural causal models.

Consider a specific individual, in a specific situation, who experienced $X = x$, with outcome $Y = y$. The counterfactual question is: if this individual had instead experienced $X = x'$, what would the outcome $Y$ have been?

In the SCM, this question has a three-step answer:

**Step 1: Abduction**. Using the observed $X = x, Y = y$, and all other observed values, update the distribution of the individual's exogenous noise variables $U$ — from the prior distribution $P(U)$ to the posterior distribution $P(U \mid \text{observations})$. This step is completed using Bayesian updating, encoding the individual's "particularity" into the posterior of $U$.

**Step 2: Action**. In the structural equations, replace the equation for $X$ with $X = x'$ (i.e., apply intervention $\mathsf{do}(X = x')$), leaving other equations unchanged.

**Step 3: Prediction**. Under the modified model and the posterior distribution of $U$ obtained in Step 1, compute the distribution of $Y$. This yields the counterfactual conclusion: if this individual had faced $X = x'$, what would the distribution of $Y$ have been.

These three steps together give an operational definition of counterfactual inference — not philosophical speculation, but a computable procedure, provided you have a sufficiently complete SCM.

::: info Average Treatment Effect and Individual Treatment Effect
The "Average Treatment Effect" (ATE) commonly used in epidemiology is a second-rung concept: $\mathbb{E}[Y \mid \mathsf{do}(X=1)] - \mathbb{E}[Y \mid \mathsf{do}(X=0)]$, which answers "in the entire population, on average, how much did the treatment change the outcome." The "Individual Treatment Effect" (ITE) is a third-rung concept: for this specific person, if the treatment were changed, how much would his outcome change. ITE is in principle unidentifiable from observational data (because each person can only be in one treatment state); this is precisely the fundamental difficulty of counterfactual inference — and the mathematical root of the challenge in precision medicine.
:::

---

## 18.9 Causation and Inference: A Larger Picture

Let us review the path traveled in this chapter.

Starting from Chapter 14, inference rules concerned propositions: from a set of assumptions in $\Gamma$, derive new propositions. Chapter 15 discovered that this mechanism has intrinsic limitations — certain true propositions cannot be proved. Chapter 16 questioned the default that "assumptions can be reused," discovering that removing it yields a new kind of logic. Chapter 17 expanded truth values from $\{0,1\}$ to $[0,1]$, turning inference into belief updating.

Now, what Chapter 18 does is: add a new verb to the language of inference — **intervention**. The $\mathsf{do}$ operator is not conditioning; it is a new kind of inference rule, requiring separate treatment in the grammar. The graph provides the context for this rule: when can $\mathsf{do}$ be simplified away, and when can it not.

This structure is entirely parallel to the structure of formal systems:

| Formal system | Causal inference |
|---------|---------|
| Propositional variables | Variables |
| Inference rules | The three rules of do-calculus |
| Axioms | Graph (local independence assumptions) |
| Provable ($\vdash$) | Identifiable |
| Incompleteness theorems | Unidentifiability theorems |

The identifiability theorem plays the role of the incompleteness theorem: there exist certain causal effects that, even with complete observational data and a complete graph, can never be computed from observation alone — an actual experiment must be performed. This is not a defect of the method, but a structural boundary.

---

## Unresolved

**Where do causal graphs come from?** The entire chapter assumes you already possess a correct causal graph. But in practice, graphs are products of subjective knowledge and domain assumptions. If the graph is drawn incorrectly, the answers from do-calculus are also incorrect — the reliability of the system depends on the correctness of the graph, and the correctness of the graph cannot be verified from data. This is the deepest dilemma of causal inference: the more precise the tool, the heavier the weight of the assumptions.

**Can causal graphs be automatically discovered from data?** Causal discovery algorithms — PC algorithm, FCI algorithm, LiNGAM — attempt to infer certain aspects of causal structure from observational data. But in the general case, observable data can only determine a "Markov equivalence class" — a set of graphs that produce the same conditional independence relations, with arrow directions indistinguishable within the class. Selecting one graph from an equivalence class still requires domain knowledge, or additional assumptions about the error distribution.

**What is the cost of performing inference on causal graphs?** Even with a graph, the identification and computation of causal effects are, in the general case, computationally expensive. Determining whether a causal effect is identifiable can be done in polynomial time; but in complex graphs with latent variables, computing the specific expression for an identifiable effect sees the cost rise sharply. The full picture of this problem is the theme of Chapter 19: the cost of reasoning is not only a limitation of logic, but also a limitation of computation.

---

## Exercises

**★ Warm-up**

Classify each of the following questions as belonging to which rung of the causal ladder (association/intervention/counterfactual), and give reasons.

1. "How much higher is the lung cancer incidence among smokers compared to non-smokers?"
2. "If a smoking ban were forcibly implemented, by how much would lung cancer incidence decrease?"
3. "If this lung cancer patient had not smoked back then, would he now have cancer?"
4. "Among the population detected with hypertension, what proportion uses antihypertensive medication?"
5. "If this hypertensive patient is prescribed antihypertensive medication, by how much would his 10-year risk of cardiovascular events decrease?"

---

**★★ Derivation**

Consider the causal graph: $Z \to X \to Y$, and $Z \to Y$ ($Z$ simultaneously affects $X$ and $Y$, $X$ also affects $Y$).

1. List all paths from $X$ to $Y$. Which are causal paths (following arrow directions), and which are backdoor paths?
2. Using the backdoor criterion, write the adjustment formula for $P(Y \mid \mathsf{do}(X = x))$.
3. If $Z$ is an unobservable latent variable, can the adjustment formula still be used? Is $P(Y \mid \mathsf{do}(X = x))$ identifiable in this case?

---

**★★★ Challenge**

In the causal graph $Z \to X \to Y$ (with $Z$ observable), add a mediator variable $M$, satisfying $X \to M \to Y$, while retaining the direct path $X \to Y$.

You want to separately estimate:
- The **direct effect** of $X$ on $Y$ (the part not passing through $M$)
- The **indirect effect** of $X$ on $Y$ (the part passing through $M$)

To which rung of the causal ladder do direct effects and indirect effects belong? Try to write the definitions of these two quantities using the language of the $\mathsf{do}$ operator — you will discover that the definition of "direct effect" requires performing two interventions simultaneously. Further reflect: why is the definition of indirect effects (natural indirect effects) harder than direct effects, necessarily appealing to counterfactuals?

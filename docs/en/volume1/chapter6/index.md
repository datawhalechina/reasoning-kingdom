# Chapter 6: The Boundaries of Causality — Observational Data Is Never Enough

> You have watched a thousand sunrises. Does that prove the sun will rise tomorrow? Hume says: no.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. The Ghost of Induction

In 1748, David Hume posed an unsettling question in *An Enquiry Concerning Human Understanding*:

Why should we believe the future will be like the past?

The sun rises every day — that's a fact. We have watched a thousand sunrises, ten thousand sunrises. But does that **prove** the sun will rise tomorrow?

Hume's answer: no.

Logically, there is no necessary derivation from "the sun rose every day in the past" to "the sun will rise tomorrow." You can imagine a world in which the sun rose ten thousand times and then stopped on the ten-thousand-and-first — and that imagination contains no logical contradiction.

This is the **problem of induction**: from a finite set of observations, we cannot logically derive universal laws.

Hume's argument shook the history of philosophy, because it touches the foundation of the scientific method. Science depends on induction — deriving laws of nature from experimental data. If induction is logically unreliable, what is the status of science?

Three hundred years later, the problem returned in a more technical form: **from observational data, can we derive causal relations?**

The answer is still: no.

This chapter is about the precise meaning of that "no," and about how we can still make useful inferences within that limit.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Trap of Correlation

Let me start with a classic example.

In the 1950s, epidemiologists observed a strong statistical correlation: smokers got lung cancer at much higher rates than non-smokers.

Does this prove smoking causes lung cancer?

The tobacco companies' lawyers said: no. They proposed an alternative explanation: maybe there is a hidden variable — for instance, a genotype — that causes both "liking smoking" and "being prone to lung cancer." Under that explanation, smoking and lung cancer have no causal relationship; they are just two effects of a common cause.

In causal graphs:

**Hypothesis 1 (causal):**

    Smoking -> Lung Cancer

**Hypothesis 2 (confounding):**

    Genotype -> Smoking
    Genotype -> Lung Cancer

In Hypothesis 2, the correlation between smoking and lung cancer is **spurious** — they are statistically related, but there is no direct causal link.

The key question: **purely on the basis of observational data, can you distinguish the two hypotheses?**

The answer: no.

If the genotype is unobservable (as it truly was in the 1950s), then the two hypotheses make exactly identical predictions on all observational data. They are **observationally equivalent**.

This is not a matter of data size. You could observe a million samples; the correlation would become more significant, but you still couldn't pin down the causal direction — because observational data only tells you the joint distribution $P(X, Y)$, and causality is about **intervention**: if I force $X$ to change, how does $Y$ change?

Observation and intervention are two different questions.

![Simpson's paradox: aggregate correlation and stratified causal effect point in opposite directions](/figures/ch06_fig2_simpsons_paradox.png)

*Figure 2: The geometry of Simpson's paradox. The overall regression line (black dashed) has a negative slope — the treatment group actually recovers worse. But after stratifying by gender, the treatment effect within women (blue line) and within men (red line) is positive. The confounder (gender) affected both the probability of treatment and the baseline recovery rate, twisting the direction of the aggregate correlation. After removing confounding by backdoor adjustment, the causal effect gets the correct sign.*

<div class="center">

------------------------------------------------------------------------

</div>

## 3. Bayesian Networks: The Causal Coat of Probability

Before we get to Pearl's causal theory, we need to understand Bayesian networks — because they are the predecessor of the structural causal model, and also its trap.

A **Bayesian network** is a directed acyclic graph (DAG): the nodes are random variables, and the edges represent conditional dependence relations.

:::details Bayesian networks and DAGs: what do probabilities on a graph mean?
**Directed acyclic graph (DAG):** a graph is a set of nodes and edges connecting them. "Directed" means every edge has a direction (A->B is different from B->A). "Acyclic" means that by following the edges, you can never return to where you started.

**Bayesian network** is a DAG with probabilities attached:
- Each node is a random variable (e.g., "is it raining," "did you bring an umbrella")
- A directed edge A->B means "A conditionally influences B" — knowing A's value changes our probability estimate for B

Key concept — **conditional dependence**: if, once A is given, B and C become independent (they no longer influence each other), then A is the "separator" of B and C. Bayesian networks use these conditional independences to factor the complex joint probability into a product of smaller local probabilities, cutting down the computation enormously.

**Important warning**: an edge in a Bayesian network represents statistical dependence, **not necessarily causality** — and that is exactly the trap this section is about.
:::

Given a Bayesian network $G = (V, E)$, the joint probability distribution factors as:

$$
P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
$$

This factorization has a beautiful property: it splits a high-dimensional joint distribution into a product of low-dimensional conditional distributions, drastically reducing the number of parameters to estimate.

But there is a subtlety here: **the edges of a Bayesian network are not necessarily causal.**

A Bayesian network is just one way to factor a joint distribution. The same joint distribution can correspond to several different Bayesian networks — as long as they encode the same conditional independences.

For example, consider three variables: $A$ (the alarm goes off), $B$ (a burglary), $E$ (an earthquake).

The true causal structure is:

    B -> A ← E

A burglary or an earthquake can both trip the alarm.

But from a purely probabilistic point of view, the following structure also encodes the same conditional independences:

    A -> B, A -> E

The alarm went off, which increases the probability of both burglary and earthquake.

The two networks are equivalent on **observational data** — they correspond to the same joint distribution $P(A, B, E)$. But their **causal meanings** are completely different.

The first network says: intervening on $B$ (forcing a burglary) affects $A$ (the alarm rings).

The second network says: intervening on $A$ (forcing the alarm to ring) affects $B$ (a burglary happens) — which is plainly absurd; you can't produce burglaries by pressing an alarm button.

**A Bayesian network is a probabilistic model, not a causal model.** Its edges represent conditional dependence, not causal flow.

Pearl's contribution was to add **causal semantics** on top of Bayesian networks — to interpret the edges as causal relationships, and then define what intervention and counterfactuals mean under that interpretation.

<div class="center">

------------------------------------------------------------------------

</div>

## A Pause

The edges of a Bayesian network are not causal relations — that's unsettling enough.

But there's a deeper problem: even once you have Pearl's causal graph, **how do you know the graph is right?**

The causal graph is drawn by you. It's your prior assumption about the structure of the world. The data can tell you which variables are conditionally independent, but that only narrows the range of possible causal graphs — it can never pin down a single graph.

That means: every conclusion in causal inference rests on an assumption that **cannot be verified from the data** — the graph you drew.

If the graph is wrong, the intervention effects computed with the do operator are wrong, too. You get an internally consistent wrong answer.

So why do we trust our own causal graph? Experience? Intuition? Domain knowledge?

These are all answers — but none of them come from the data.

Let's leave this question on the table.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. Pearl's Ladder of Causation

In *Causality*, published in 2000, Judea Pearl presented a three-level framework for causal reasoning, which later became known as the **Ladder of Causation**.

:::details The Ladder of Causation: observation, intervention, counterfactuals — three different kinds of questions
Pearl's ladder divides "causal questions" into three levels, and each level requires stronger assumptions to answer:

**Level 1 (Observation):** *Given that I see X, what is the probability of Y?*
A purely statistical question; it just needs data. Example: "Among smokers, what share get lung cancer?"

**Level 2 (Intervention):** *If I force X to happen, what is the probability of Y?*
Needs a causal graph. Example: "If I made everyone smoke (randomly assigned), how much would the lung cancer rate change?" This is exactly what a randomized controlled trial (RCT) does.

**Level 3 (Counterfactual):** *If X had been different in the past, what would Y be now?*
Needs a full structural causal model. Example: "This patient smokes and got lung cancer — if he had never smoked, would he still have lung cancer?" This is a counterfactual about one particular person, and it can't be answered directly by experiments.

**Key point**: you cannot answer a higher-level question with lower-level information. No amount of observational data can answer an interventional or counterfactual question directly — that is the precise meaning of "observational data is never enough."
:::

**Level 1: Association (Observation)**

Question form: $P(Y \mid X)$

"Given that I've seen $X$, what is the probability of $Y$?"

This is a purely statistical question. Given the data, you can estimate conditional probabilities. Most of machine learning lives on this level: given features $X$, predict label $Y$.

**Level 2: Intervention**

Question form: $P(Y \mid do(X))$

"If I force $X$ to take a certain value, what is the probability of $Y$?"

Here $do(X)$ is the symbol Pearl introduced for an external intervention on $X$ — not observing $X$ happen naturally, but forcing $X$ to take a value through some external action.

$P(Y \mid do(X))$ and $P(Y \mid X)$ are usually different.

An example:

- $P(\text{recovery} \mid \text{drug})$ may be high — because the mildly ill are more likely to take the drug, and the mildly ill are more likely to recover anyway.

- $P(\text{recovery} \mid do(\text{drug}))$ is the true causal effect of the drug — if we randomly assigned who takes the drug, what's the recovery rate in the drug group.

The first is observational; the second is interventional. Confusing the two is one of the most common mistakes in medical research.

**Level 3: Counterfactual**

Question form: $P(Y_x \mid X', Y')$

"If $X$'s value in the past had been $x$ (rather than the actually observed $X'$), then what would $Y$ be (given that we observed $Y'$)?"

This is the hardest level, because it involves reasoning about **something that did not happen**.

Example:

- Observation: this patient took the drug, and recovered.

- Counterfactual: if the patient hadn't taken the drug, would he have recovered?

Counterfactual reasoning needs more than data; it needs a **structural causal model** — a mechanistic hypothesis about how the world works.

Pearl's core claim: **these three levels cannot replace each other.** You can't use Level 1 information to answer a Level 2 question, and you can't use Level 2 information to answer a Level 3 question.

Each level needs stronger assumptions and more structure.

![Pearl's ladder of causation: three levels of reasoning — observation, intervention, counterfactuals](/figures/ch06_fig1_causal_ladder.png)

*Figure 1: Pearl's ladder of causation. Level 1 (observation) needs only data and answers "probability of Y after seeing X." Level 2 (intervention) needs the causal graph and answers "probability of Y after forcing X." Level 3 (counterfactual) needs the full SCM parameters and answers "how Y would differ if X had been different." Each step up the ladder requires stronger structural assumptions.*

<div class="center">

------------------------------------------------------------------------

</div>

## 5. Structural Causal Models: A Blueprint of the World's Mechanisms

To answer intervention and counterfactual questions, we need a **structural causal model (SCM)**.

An SCM has three parts:

1. **Endogenous variables** $V = \{V_1, \ldots, V_n\}$: the variables we care about

2. **Exogenous variables** $U = \{U_1, \ldots, U_m\}$: external, unobservable random disturbances

3. **Structural equations** $\mathcal{F}$: each endogenous variable is determined by a function

$$
V_i = f_i(\text{Parents}(V_i), U_i)
$$

Here $\text{Parents}(V_i)$ are the direct causes of $V_i$ (the nodes pointing into $V_i$ in the causal graph), and $U_i$ is the exogenous noise affecting $V_i$.

**Example: Smoking and lung cancer**

Variables:

- $S$: whether the person smokes
- $C$: whether they get lung cancer
- $G$: genotype (unobservable)

Structural equations:

    G = U_G                    (the genotype is set by exogenous factors)
    S = f_S(G, U_S)            (smoking tendency is influenced by genes and other factors)
    C = f_C(S, G, U_C)         (lung cancer is influenced by smoking, genes, and other factors)

Causal graph:

    G -> S -> C
    G -> C

This model encodes an assumption about the world: the genotype $G$ affects both smoking tendency and lung cancer risk, and smoking $S$ also directly affects lung cancer $C$.

With an SCM, we can define **intervention**:

$$
do(S = s) \text{ means: replace the structural equation } S = f_S(G, U_S) \text{ with the constant } s
$$

After the intervention, the model becomes:

    G = U_G
    S = s                      (forced)
    C = f_C(s, G, U_C)

Now $S$ no longer depends on $G$ — we cut the edge $G \to S$. This is the graphical representation of intervention: **delete all edges pointing into the intervened variable**.

The post-intervention joint distribution $P(C \mid do(S=s))$ can be computed from the modified model.

Key point: $P(C \mid do(S=s)) \neq P(C \mid S=s)$, because the latter includes $G$'s confound of $S$ and $C$, while the former cuts that confounding off.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. The do-Calculus: A Bridge from Observation to Intervention

Now we come to a core question: **given observational data, can we compute the interventional distribution?**

That is: can we derive $P(Y \mid do(X))$ (interventional) from $P(V)$ (observational)?

Pearl's **do-calculus** gives the answer: yes, under certain conditions.

The do-calculus has three rules, which let you manipulate probability expressions that contain $do(\cdot)$ under the constraints of the causal graph $G$.

**Rule 1: Insertion / deletion of observation**

$$
P(Y \mid do(X), Z, W) = P(Y \mid do(X), W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}
$$

where $G_{\overline{X}}$ is the graph after deleting all edges pointing into $X$.

Intuition: if, after intervening on $X$, $Z$ and $Y$ are conditionally independent (given $W$), then observing $Z$ adds no extra information.

**Rule 2: Action / observation exchange**

$$
P(Y \mid do(X), do(Z), W) = P(Y \mid do(X), Z, W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\underline{Z}}}
$$

where $G_{\overline{X}\underline{Z}}$ is the graph after deleting edges pointing into $X$ and edges pointing out of $Z$.

Intuition: if $Z$ doesn't affect $Y$ (after intervening on $X$), then intervening on $Z$ is equivalent to observing $Z$.

**Rule 3: Insertion / deletion of action**

$$
P(Y \mid do(X), do(Z), W) = P(Y \mid do(X), W) \quad \text{if } (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\overline{Z(W)}}}
$$

where $G_{\overline{X}\overline{Z(W)}}$ is the graph after deleting edges pointing into $X$ and edges pointing into the nodes of $Z$ that are not ancestors of $W$.

Intuition: if $Z$ doesn't affect $Y$ (under a particular graph transformation), then intervening on $Z$ can be ignored.

These three rules look technical, but their power lies in this: **they are complete**.

Pearl and colleagues proved that if a causal effect $P(Y \mid do(X))$ can be identified from the observational distribution $P(V)$ and the causal graph $G$, then the three do-calculus rules suffice to derive it.

Conversely, if the do-calculus cannot derive $P(Y \mid do(X))$, then it is unidentifiable: with observational data and the graph structure alone, the causal effect cannot be determined.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. The Backdoor Criterion and the Frontdoor Criterion

The do-calculus is a complete tool in theory, but in practice we usually use simpler, more direct criteria to judge whether a causal effect is identifiable.

**Backdoor criterion**

Given a causal graph $G$, if there exists a set of variables $Z$ such that:

1. $Z$ blocks all **backdoor paths** from $X$ to $Y$ (paths that contain an edge pointing into $X$)

2. No descendant of $X$ is in $Z$

Then the causal effect can be computed with the adjustment formula:

$$
P(Y \mid do(X)) = \sum_z P(Y \mid X, Z=z) P(Z=z)
$$

This is **adjustment**: by stratifying on the confounders $Z$, you eliminate spurious correlation.

Back to the smoking and lung cancer example:

Causal graph:

    G -> S -> C
    G -> C

$G$ is a confounder lying on the backdoor path $S \leftarrow G \to C$. If we can observe $G$, then:

$$
P(C \mid do(S)) = \sum_g P(C \mid S, G=g) P(G=g)
$$

This is also why randomized controlled trials (RCTs) work: randomly assigning $S$ is equivalent to cutting $G \to S$, making $P(C \mid do(S)) = P(C \mid S)$.

**Frontdoor criterion**

But what if the confounder is unobservable?

Pearl found a clever case: even when the confounder is unobservable, if there exists a **mediator** $M$ such that:

1. $M$ fully mediates the effect of $X$ on $Y$ ($X$ only affects $Y$ through $M$)

2. $X$ blocks all backdoor paths from $M$ to $Y$

3. The empty set blocks all backdoor paths from $X$ to $M$

then the causal effect can be computed with the frontdoor formula:

$$
P(Y \mid do(X)) = \sum_m P(M=m \mid X) \sum_{x'} P(Y \mid M=m, X=x') P(X=x')
$$

This formula doesn't require observing the confounder — it sidesteps the confounding through the mediator $M$.

**Example: Smoking, tar, and lung cancer**

    U -> S -> T -> C
    U -> C

$U$ is the unobservable confounder (say, genotype), and $T$ is tar deposits (an observable mediator).

Even though we can't observe $U$, we can still compute $P(C \mid do(S))$ via the frontdoor criterion — because $T$ fully mediates the effect of $S$ on $C$.

This is a deep result: **under certain structures, observational data is enough to identify a causal effect, even in the presence of unobservable confounding.**

But "certain structures" is the key — not every causal graph satisfies the backdoor or frontdoor criteria.

## 8. Observational Equivalence Classes: When Causal Graphs Are Indistinguishable

Now we're at the most important part of this chapter. Even with unlimited observational data, you still can't pin down a unique causal graph — because several different causal graphs can produce the same observational distribution.

These graphs are said to form **Markov equivalence classes**.

**Definition:** two DAGs $G_1$ and $G_2$ are Markov equivalent if they encode the same conditional independence relations.

Equivalent graphs share the same **skeleton** (the undirected graph you get when you ignore arrow directions) and the same **v-structures** (structures of the form $A \to C \leftarrow B$, where $A$ and $B$ are not adjacent).

**Example: an equivalence class with three variables**

Consider three variables $X, Y, Z$. The following three graphs are Markov equivalent:

    X -> Y -> Z
    X ← Y -> Z
    X ← Y ← Z

They all encode the same conditional independence: $X \perp\!\!\!\perp Z \mid Y$.

From observational data, you cannot distinguish these three graphs — because they correspond to the same joint distribution.

But their causal meanings are completely different:

- The first graph: $X$ affects $Y$, $Y$ affects $Z$

- The second graph: $Y$ affects both $X$ and $Z$

- The third graph: $Z$ affects $Y$, $Y$ affects $X$

If you want $P(Z \mid do(X))$, these three graphs give different answers.

**Where the unidentifiability comes from**

This isn't a data-size problem, and not an algorithm problem. It is a **structural impossibility**.

Observational data can only tell you conditional independence — which variables are independent once other variables are given. But causality is not just independence; it is about **behavior under intervention**.

Two graphs can have the identical observational distribution yet behave completely differently under intervention.

That means: **from observational data alone, causal discovery can recover only the Markov equivalence class — never a single unique causal graph.**

To break out of the equivalence class, you need extra information:

1. **Interventional data**: if you can run experiments — forcibly change some variables and observe how others respond — you can determine the causal direction

2. **Temporal order**: if you know $X$ happens before $Y$, then $Y \to X$ is impossible

3. **Functional form assumptions**: if you assume the causal mechanisms are linear, or belong to additive noise models, some equivalence classes can be broken

4. **Prior knowledge**: if you know some edges cannot exist (for example, "age cannot be influenced by income"), you can rule out certain graphs

But if all you have is observational data and no extra assumptions, the causal graph is unidentifiable.

![Markov equivalence classes: three causal graphs, one observational distribution](/figures/ch06_fig3_markov_equivalence.png)

*Figure 3: A Markov equivalence class example. Three different causal graphs (X->Y->Z, X←Y->Z, X←Y←Z) encode the same conditional independence X ⊥ Z | Y, so they are exactly equivalent on observational data. But their causal meanings differ: in the first graph, do(X) affects Z; in the second, it does not. Observational data alone cannot tell them apart.*

## 9. The Faithfulness Assumption: A Fragile Bridge

Causal discovery algorithms (like the PC algorithm and the GES algorithm) usually depend on two assumptions:

**Assumption 1: Causal Markov condition**

Given its parents, every variable is independent of its non-descendants.

This is the standard assumption of Bayesian networks, and usually reasonable.

**Assumption 2: Faithfulness**

Every conditional independence in the observational distribution is entailed by d-separation in the causal graph.

In other words: if $X \perp\!\!\!\perp Y \mid Z$ holds in the data, then $X$ and $Y$ are d-separated by $Z$ in the graph.

The assumption looks harmless, but it is actually strong.

**A counterexample: coincidental cancellation of parameters**

Consider the following causal graph and structural equations:

    X -> Y -> Z
    X -> Z

    Y = a·X + U_Y
    Z = b·Y + c·X + U_Z

If the parameters happen to satisfy $b \cdot a + c = 0$, then the total effect of $X$ on $Z$ is zero — the direct effect $c$ and the indirect effect $b \cdot a$ through $Y$ cancel each other out.

In that case, the observational data would show $X \perp\!\!\!\perp Z \mid Y$ — not because of the graph structure, but because of a coincidence in the parameters.

The faithfulness assumption rules out such coincidences: it assumes the parameters are in "general position" and won't cancel exactly.

But in the real world, such coincidences may not be rare. Biological systems and economic systems are full of feedback and balancing mechanisms, and their parameters may happen to make certain effects cancel.

**The consequence of faithfulness failure**

If faithfulness does not hold, causal discovery algorithms will infer the wrong graph structure — they will believe certain edges don't exist, because the corresponding variables look independent in the data, when in fact they do have a causal relationship — the effect was just canceled out.

This is a deep fragility: causal discovery depends on an assumption about parameters, and that assumption cannot be tested in the data — because you cannot distinguish "true independence" from "coincidental cancellation."

<div class="center">

------------------------------------------------------------------------

</div>

## 10. Counterfactuals: The Hardest Level

The third level of Pearl's ladder — counterfactuals — is the hardest, because it involves reasoning about **things that did not happen**.

**Defining counterfactuals**

Given an SCM and observed evidence $X=x', Y=y'$, the counterfactual question is:

$$
P(Y_x \mid X=x', Y=y')
$$

"If $X$'s value were $x$ (instead of the observed $x'$), what would $Y$ be (given that we observed $Y=y'$)?"

Computing a counterfactual requires three steps:

**Step 1: Abduction**

Update your belief about the exogenous variables $U$ according to the observed evidence:

$$
P(U \mid X=x', Y=y')
$$

**Step 2: Action**

Modify the SCM, replacing the structural equation for $X$ with $X=x$ (an intervention).

**Step 3: Prediction**

Under the modified model and the updated distribution of $U$, compute the distribution of $Y$:

$$
P(Y_x \mid X=x', Y=y') = \sum_u P(Y_x \mid U=u) P(U=u \mid X=x', Y=y')
$$

**Example: the individual causal effect of a drug**

Suppose we have the following SCM:

    X = U_X                    (whether the drug was taken; set by exogenous factors)
    Y = a·X + U_Y              (recovery outcome)

Observation: a patient took the drug ($X=1$) and recovered ($Y=1$).

Counterfactual question: if the patient hadn't taken the drug ($X=0$), would he have recovered?

Step 1: infer $U_Y$ from $X=1, Y=1$:

$$
1 = a \cdot 1 + U_Y \implies U_Y = 1 - a
$$

Step 2: intervene with $X=0$.

Step 3: compute the counterfactual outcome:

$$
Y_0 = a \cdot 0 + U_Y = 1 - a
$$

If $a > 0$ (the drug has a positive effect), then $Y_0 < 1$ — if this patient hadn't taken the drug, his recovery would have been worse.

**Unidentifiability of counterfactuals**

The key problem: counterfactuals are usually **unidentifiable**.

Even if you know the causal graph, even with unlimited observational data, you still cannot uniquely determine the counterfactual distribution from data — because counterfactuals depend on the distribution of the exogenous variables $U$, and $U$ is unobservable.

In the example above, if we don't know the value of $a$ (the size of the drug effect), we cannot compute $Y_0$.

Counterfactuals need more than the causal graph; they need the **parameterized structural equations** — a stronger assumption than the graph structure alone.

That's why counterfactual reasoning is hard in practice: it requires a complete, parameterized world model, and such a model usually cannot be fully learned from the data.

## 11. Pseudocode: The Core Algorithms of Causal Inference

Let me formalize the core algorithm from the discussion above.

**Algorithm 1: Backdoor adjustment**

```python
import itertools
import numpy as np

def backdoor_adjustment(data, x_col, y_col, z_cols, x_val):
    """
    Estimate the causal effect P(Y | do(X=x_val)) via backdoor adjustment.
    data:    pandas DataFrame containing observational data
    x_col:   column name of the intervention variable
    y_col:   column name of the outcome variable
    z_cols:  list of adjustment-set column names (satisfying the backdoor criterion)
    x_val:   intervention value
    Returns: estimate of P(Y=1 | do(X=x_val))
    """
    if not z_cols:
        raise ValueError("Empty adjustment set; the causal effect is not identifiable via backdoor adjustment")

    # Enumerate all value combinations of Z
    z_values = [data[z].unique() for z in z_cols]
    result = 0.0

    for z_combo in itertools.product(*z_values):
        # Conditional probability P(Y | X=x_val, Z=z_combo)
        mask_xz = (data[x_col] == x_val)
        for z_col, z_val in zip(z_cols, z_combo):
            mask_xz &= (data[z_col] == z_val)
        if mask_xz.sum() == 0:
            continue
        p_y_given_xz = data.loc[mask_xz, y_col].mean()

        # Marginal probability P(Z=z_combo)
        mask_z = np.ones(len(data), dtype=bool)
        for z_col, z_val in zip(z_cols, z_combo):
            mask_z &= (data[z_col] == z_val)
        p_z = mask_z.mean()

        result += p_y_given_xz * p_z

    return result
```

## 12. A Small Pause

Let me take stock of what this chapter did.

Hume's problem of induction returned three hundred years later in the form of causal inference: from observational data, can we derive causal relations? The answer: no — at least, not uniquely.

Observational data can only tell us the joint distribution $P(X, Y)$ between variables, while causality is about intervention $P(Y \mid do(X))$. These are usually different, because observation includes confounding.

Pearl's ladder divides reasoning into three levels: observation, intervention, counterfactuals. Each level needs stronger assumptions. Most of machine learning stays on Level 1, and true causal reasoning needs Level 2 or Level 3.

Structural causal models (SCMs) provide a framework that encodes the world's mechanisms with structural equations and causal graphs. With an SCM, we can define intervention (delete edges pointing into the intervened variable) and counterfactuals (the abduction–action–prediction three-step method).

The do-calculus gives complete rules for deriving the interventional distribution from the observational distribution. The backdoor and frontdoor criteria are the more direct practical tools — under certain graph structures, observational data suffices to identify a causal effect.

But the core limits remain:

1. **Markov equivalence classes**: several causal graphs can produce the same observational distribution, and observational data alone cannot distinguish them

2. **The faithfulness assumption**: causal discovery relies on an assumption about parameters being in "general position," an assumption that cannot be tested in the data

3. **Unidentifiability of counterfactuals**: counterfactual reasoning requires parameterized structural equations, not just a causal graph

This leads to the question of Chapter 7: if observational data isn't enough, what do we need? The answer is intervention — actively changing the world instead of passively watching it. The randomized controlled trial (RCT) is the gold standard of causal inference, not because it is more precise, but because it breaks the cage of observation.

<div class="center">

------------------------------------------------------------------------

</div>

> Causal inference reveals the limits of observation. In the next chapter, we'll turn to another frontier — the asymmetry inside computation itself: why is finding an answer so much harder than verifying one?

## Open Questions

- How often does the faithfulness assumption fail in the real world? Is there any way to detect faithfulness failure?

- If two causal graphs are in the same Markov equivalence class but give different interventional predictions, which one should we trust? Is that a scientific choice, or a philosophical one?

- Counterfactual reasoning requires a fully parameterized SCM. Under what conditions can we learn those parameters from data? When is it impossible?

- Can large language models do causal reasoning? What they learn from text — is it $P(Y \mid X)$ or $P(Y \mid do(X))$? The answer to this question determines the limits of an LLM's abilities.

- If you train a causal discovery algorithm on purely observational data and then test it on interventional data, where will it fail systematically? What does that failure pattern teach us?

## Try It Yourself: See Through the Lies of Observation with Simpson's Paradox

The core claim of this chapter: observational data lies, because it contains confounding. You're going to build a Simpson's paradox example with your own hands, and watch the same data give completely opposite conclusions under different stratifications.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Generate Data with Confounding

Construct the following causal structure:

    Causal graph:
    G -> T -> Y
    G -> Y

    where:
    G = gender (0=female, 1=male)
    T = whether the patient received treatment (0=no, 1=yes)
    Y = recovery outcome (continuous, higher is better)

The true causal mechanism:

- Men tend to receive treatment more often (because they are sicker)
- Treatment has a positive effect (increases recovery)
- But men's baseline recovery is lower (because they are sicker)

```python
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000  # sample size

# Generate gender (confounder): 0=female, 1=male, 50% each
G = np.random.binomial(1, 0.5, N)

# Generate the treatment decision (affected by gender)
# Men are sicker and more likely to be treated
p_treatment = np.where(G == 1, 0.7, 0.3)  # men 70%, women 30%
T = np.random.binomial(1, p_treatment, N)

# Generate the recovery outcome (affected by both gender and treatment)
# Female baseline recovery score 80; male baseline 60 (men are sicker)
baseline = 80 - 20 * G        # vectorized: baseline score per sample
treatment_effect = 15 * T     # treatment adds 15 points (true causal effect)
noise = np.random.normal(0, 5, N)  # random noise
Y = baseline + treatment_effect + noise

# Arrange into a DataFrame for the following analysis
df = pd.DataFrame({'G': G, 'T': T, 'Y': Y})
print(f"Data generation complete: {N} samples")
print(df.describe().round(2))
print(f"\nTrue causal effect (treatment parameter) = +15 points")
```

**Your first question (answer it before generating any data):** Under this setup, what is the true causal effect of treatment? If we randomly assigned treatment (cutting G -> T), how much higher would the treatment group's recovery score be than the control group's?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Compute the Observed Correlation (No Confounder Adjustment)

After generating the data, directly compute:

```python
import pandas as pd

# Compare treatment vs. control directly, ignoring the gender confounder
treated_mean   = df[df['T'] == 1]['Y'].mean()   # mean of the treatment group
untreated_mean = df[df['T'] == 0]['Y'].mean()   # mean of the control group
observed_effect = treated_mean - untreated_mean  # observed effect (no confounder adjustment)

print(f"Treatment group mean: {treated_mean:.2f}")
print(f"Control group mean: {untreated_mean:.2f}")
print(f"Observed effect (unadjusted): {observed_effect:.2f}")
print(f"True causal effect: +15.00")
```

**Your second question:** What is the sign of the observed effect? Is it positive (the treatment group is better) or negative (the control group is better)? Does it match the true causal effect?

If it doesn't match, you have a Simpson's paradox — the overall correlation and the causal effect point in opposite directions.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Stratified Analysis (Adjusting for Confounding)

Now stratify by gender, and compute within each stratum:

```python
import pandas as pd

# --- Treatment effect within the female group (G=0) ---------------------------
female_treated   = df[(df['G'] == 0) & (df['T'] == 1)]['Y'].mean()
female_untreated = df[(df['G'] == 0) & (df['T'] == 0)]['Y'].mean()
effect_female    = female_treated - female_untreated

# --- Treatment effect within the male group (G=1) -----------------------------
male_treated   = df[(df['G'] == 1) & (df['T'] == 1)]['Y'].mean()
male_untreated = df[(df['G'] == 1) & (df['T'] == 0)]['Y'].mean()
effect_male    = male_treated - male_untreated

# --- Backdoor adjustment: weighted average of the stratified effects by gender's marginal distribution ---
P_G0 = (df['G'] == 0).mean()  # proportion of female samples
P_G1 = (df['G'] == 1).mean()  # proportion of male samples
adjusted_effect = P_G0 * effect_female + P_G1 * effect_male  # weighted average

print(f"Effect within females (G=0): {effect_female:.2f}")
print(f"Effect within males (G=1): {effect_male:.2f}")
print(f"Adjusted causal effect (backdoor adjustment): {adjusted_effect:.2f}")
print(f"True causal effect: +15.00")
print(f"Observed effect (unadjusted): {observed_effect:.2f}")
```

**Your third question (the core one):** After stratification, what is the sign of the treatment effect within each gender? Is it the same as the observed effect from Step 2?

If Step 2 was negative but both within-gender effects are positive, you just saw Simpson's paradox with your own eyes: the overall correlation and the stratified causal effect point in opposite directions.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Visualize the Geometry of Confounding

Plot a scatter with treatment (0 or 1) on the x-axis and recovery score on the y-axis:

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

fig, ax = plt.subplots(figsize=(9, 6))

# Scatter of the four subgroups (small jitter to avoid overlaps)
jitter = 0.04
groups = [
    (0, 0, 'royalblue',  'o', 'Females, control (G=0, T=0)'),
    (0, 1, 'royalblue',  '^', 'Females, treatment (G=0, T=1)'),
    (1, 0, 'tomato',     'o', 'Males, control (G=1, T=0)'),
    (1, 1, 'tomato',     '^', 'Males, treatment (G=1, T=1)'),
]
for g_val, t_val, color, marker, label in groups:
    mask = (df['G'] == g_val) & (df['T'] == t_val)
    t_jittered = df.loc[mask, 'T'] + np.random.uniform(-jitter, jitter, mask.sum())
    ax.scatter(t_jittered, df.loc[mask, 'Y'],
               c=color, marker=marker, alpha=0.5, s=20, label=label)

# Helper: draw an OLS regression line
def plot_regression(x, y, color, linestyle, label):
    coeffs = np.polyfit(x, y, 1)   # linear regression; coeffs=[slope, intercept]
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color=color, linestyle=linestyle, linewidth=2, label=label)
    return coeffs[0]  # return the slope

# 1. Overall regression line (ignoring gender) -- black dashed line
slope_overall = plot_regression(df['T'].values, df['Y'].values,
                                'black', '--', 'Overall regression line (ignoring gender)')

# 2. Within-female regression line -- blue solid line
female_mask = df['G'] == 0
slope_female = plot_regression(df.loc[female_mask, 'T'].values,
                               df.loc[female_mask, 'Y'].values,
                               'royalblue', '-', 'Within-female regression line')

# 3. Within-male regression line -- red solid line
male_mask = df['G'] == 1
slope_male = plot_regression(df.loc[male_mask, 'T'].values,
                             df.loc[male_mask, 'Y'].values,
                             'tomato', '-', 'Within-male regression line')

ax.set_xlabel('Received treatment (0=no, 1=yes)')
ax.set_ylabel('Recovery score Y')
ax.set_title("The geometry of Simpson's paradox\nOverall line slopes down, but within-group lines slope up")
ax.set_xticks([0, 1])
ax.set_xticklabels(['Control (T=0)', 'Treatment (T=1)'])
ax.legend(fontsize=8, loc='upper left')

print(f"Overall regression slope: {slope_overall:.2f} ({'positive' if slope_overall > 0 else 'negative'} -- treatment {'beneficial' if slope_overall > 0 else 'harmful'}?)")
print(f"Within-female regression slope: {slope_female:.2f}")
print(f"Within-male regression slope: {slope_male:.2f}")
plt.tight_layout()
plt.show()
```

**Your fourth question:** What is the sign of the overall regression slope? What about the within-female and within-male slopes?

If the overall line goes down (negative slope) while both within-group lines go up (positive slope), that's the geometry of Simpson's paradox: the confounder has reversed the direction of the correlation.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 5: Reverse the Causal Graph and See What Happens

Now suppose you mistakenly believe the causal graph is:

    Wrong causal graph:
    T -> G -> Y
    (treatment affects gender? obviously absurd, but let's pretend we don't know)

Under this wrong graph, the backdoor criterion tells you: no adjustment is needed, because there is no backdoor path.

```python
# Under the wrong causal graph T -> G -> Y:
# the graph says T is a cause of G, so G is not a confounder of T,
# the backdoor path T<-G->Y does not exist (the arrow direction is wrong),
# so we simply use the observed correlation as the "causal effect"

# "Causal effect" under the wrong graph = the direct observed effect (no adjustment)
wrong_causal_effect = observed_effect  # the unadjusted value computed in step 2

print(f"'Causal effect' under the wrong graph: {wrong_causal_effect:.2f}")
print(f"Correctly adjusted causal effect:       {adjusted_effect:.2f}")
print(f"True causal effect (generation setting): +15.00")
print(f"Deviation of the wrong inference from the truth: {abs(wrong_causal_effect - 15):.2f} points")
print()
print("Conclusion: a wrong causal graph assumption -> the inference deviates entirely from the true effect")
print("Observational data alone cannot tell you which graph is correct.")
```

**Your fifth question:** If you used the wrong causal graph, what would you conclude? How far would that conclusion be from the true causal effect?

This is the core point of Section 8: **the structural assumption of the causal graph is unavoidable. If the graph is wrong, the inference is wrong — and observational data cannot tell you whether the graph is correct.**

<div class="center">

------------------------------------------------------------------------

</div>

### Checklist

After finishing this exercise, you should be able to answer:

1. How does Simpson's paradox happen? How does a confounder make the observed correlation and the causal effect point in opposite directions?

2. How does backdoor adjustment (stratified analysis) remove confounding? Why is the stratified effect closer to the true causal effect?

3. If the structural assumption of the causal graph is wrong, what does backdoor adjustment deliver? How large is the error?

If you only do one thing, do Step 4. That plot will let you see in one glance how confounding twists correlations.

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Hand-Write the do Operation — Does Coffee Really Make You Smarter?

In the Simpson's paradox exercise, you saw **why** observational data lies. This exercise goes further: you'll compute $P(Y \mid do(X))$ yourself, turning "cutting the arrows in the causal graph" from an abstract operation into runnable code.

**Scenario**: you work on a data team at a tech company. Your boss sees data showing that coffee-drinking employees produce more code. He's about to push a policy of "three cups of coffee per day for everyone." You need to tell him whether the observed correlation and the real effect of forcing coffee are the same thing.

**Causal graph**:

```
Work Stress (S) ─────> Coffee (C) ─────> Code Output (Y)
     │                              ↑
     └──────────────────────────────┘
```

- **S** (Stress, work stress): 0 = low stress, 1 = high stress
- **C** (Coffee, drank coffee today): 0 = no, 1 = yes
- **Y** (code output, lines/hour, continuous)

The true mechanism: **high-stress employees want coffee more** (they use it to stay awake), **but high stress itself lowers output**. Coffee itself has only a mild positive effect on output.

---

### Step 1: Generate Data, Feel the Confounding

```python
import numpy as np
import pandas as pd
np.random.seed(42)

N = 2000

# Work stress (confounder)
S = np.random.binomial(1, 0.5, N)          # 50% high stress

# Coffee intake (driven by stress)
p_coffee = np.where(S == 1, 0.80, 0.25)    # high stress: 80% drink coffee; low stress: 25%
C = np.random.binomial(1, p_coffee, N)

# Code output (dragged down by stress, slightly boosted by coffee)
Y = (50                    # baseline output
     - 20 * S              # high stress reduces output by 20 lines
     + 5  * C              # coffee raises output by 5 lines (true causal effect)
     + np.random.normal(0, 8, N))

df = pd.DataFrame({'S': S, 'C': C, 'Y': Y})
```

**Question 1**: What is the true causal effect of coffee on output? (Read it straight off the generation process.)

---

### Step 2: Compute the Observed Correlation

```python
# Directly compare coffee drinkers vs. non-drinkers
obs_effect = df[df.C == 1]['Y'].mean() - df[df.C == 0]['Y'].mean()
print(f"Observed effect P(Y|C=1) - P(Y|C=0) = {obs_effect:.2f}")
```

**Question 2**: Is the observed effect positive or negative? Is it bigger or smaller than the true causal effect? Why?

(Hint: high-stress employees both drink more coffee and produce less — which binds coffee to lower output.)

---

### Step 3: Hand-Write the do Operation — the Backdoor Adjustment Formula

Now you'll implement $P(Y \mid do(C=1))$ yourself.

The backdoor criterion tells us: **S lies on a backdoor path between C and Y** ($C \leftarrow S \rightarrow Y$). Block S, and the causal effect becomes identifiable.

The formula is:

$$
P(Y \mid do(C=c)) = \sum_{s} P(Y \mid C=c, S=s) \cdot P(S=s)
$$

Translated into code:

```python
def do_calculus(df, intervention_value):
    """
    Compute P(Y | do(C = intervention_value))

    Steps:
    1. Stratify by S; within each stratum, compute the mean of Y at C=intervention_value
    2. Weight-average using the marginal distribution of S

    This is the code implementation of "cutting the arrow C<-S."
    """
    result = 0.0

    for s_val in [0, 1]:
        # Step 1: expectation of Y when S=s and C=intervention_value
        # (in this subset, the value of C is what we "observed", not what we intervened on)
        subset = df[(df['S'] == s_val) & (df['C'] == intervention_value)]
        E_Y_given_C_S = subset['Y'].mean()

        # Step 2: marginal probability of S=s (estimated from the original data)
        P_S = (df['S'] == s_val).mean()

        result += E_Y_given_C_S * P_S

    return result

# Compute the intervention effects
E_Y_do_C1 = do_calculus(df, intervention_value=1)  # do(C=1)
E_Y_do_C0 = do_calculus(df, intervention_value=0)  # do(C=0)

causal_effect = E_Y_do_C1 - E_Y_do_C0
print(f"Expectation of Y given do(C=1): {E_Y_do_C1:.2f}")
print(f"Expectation of Y given do(C=0): {E_Y_do_C0:.2f}")
print(f"Causal effect P(Y|do(C=1)) - P(Y|do(C=0)) = {causal_effect:.2f}")
print(f"True causal effect (set at generation time) = 5.00")
```

**Question 3**: The `do_calculus` function and "stratify then weighted-average" do the same thing — compared to the direct comparison in Step 2, which one gets closer to the true causal effect?

---

### Step 4: Understand What "Cutting the Arrow" Means

The essence of the do operation is constructing an **interventional dataset**: force C to some value while leaving S unchanged (S is a cause of C, but C is not a cause of S, so forcing coffee doesn't change work stress).

```python
# Build the intervention dataset: do(C=1) -- everyone drinks coffee
df_do_C1 = df.copy()
df_do_C1['C'] = 1   # force everyone to drink coffee
# Note: do not modify S! S is an external variable, unaffected by C

# On the intervention dataset, recompute Y (using the true generation formula)
df_do_C1['Y'] = (50
                 - 20 * df_do_C1['S']
                 + 5  * df_do_C1['C']   # C is now all 1s
                 + np.random.normal(0, 8, N))

# Build the intervention dataset: do(C=0) -- nobody drinks coffee
df_do_C0 = df.copy()
df_do_C0['C'] = 0
df_do_C0['Y'] = (50
                 - 20 * df_do_C0['S']
                 + 5  * df_do_C0['C']   # C is now all 0s
                 + np.random.normal(0, 8, N))

effect_intervention = df_do_C1['Y'].mean() - df_do_C0['Y'].mean()
print(f"Effect estimate from the direct intervention experiment: {effect_intervention:.2f}")
```

**Question 4**: The do_calculus result (Step 3) and the "direct intervention experiment" result (Step 4) should be close. If they are close, what does that tell you? If there is a gap, where does the gap come from?

---

### Step 5: Tell the Boss the Truth

```python
print("=" * 50)
print(f"Observed correlation (wrong basis): {obs_effect:.1f} lines/hour")
print(f"do-calculus causal effect (right basis): {causal_effect:.1f} lines/hour")
print(f"True causal effect (generation setting): 5.0 lines/hour")
print()
print("Conclusion:")
print(f"Observational data shows coffee-drinking employees output {'higher' if obs_effect > 0 else 'lower'} by {abs(obs_effect):.1f} lines")
print("But most of this gap comes from the confounder (work stress)")
print(f"The true effect of a mandatory coffee policy is only about 5 lines/hour")
```

**The final question**: if the boss pushes the "three cups of coffee" policy, what effect does he expect? What would actually happen?

This is the real value of the do operation: **estimating intervention effects from observational data, without a randomized controlled trial.**

---

### Extension Challenge (Optional)

If you want to go deeper, try modifying the causal graph by adding a collider:

```
S ───-> C ───> Y
           ↗
    Overtime (O)──-> Y
         ↑
         S
```

Overtime (O) is also affected by work stress, and it also affects output. Now the backdoor path has changed — you need to adjust for S, but you must not adjust for O (O is downstream of C). Try modifying `do_calculus` to handle this more complex graph.

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.) — the foundational work of causal inference; a complete treatment of SCMs, the do-calculus, and the counterfactual framework

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect* — the popular version of the ladder of causation; the history and philosophy of the causal revolution

- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* — the classic textbook on causal discovery algorithms; the original source of the PC and FCI algorithms

- Peters, J., Janzing, D. & Schölkopf, B. (2017). *Elements of Causal Inference* — causal inference from a machine learning angle, including independent causal mechanisms and additive noise models `-> [MIT Press]`

- Hume, D. (1748). *An Enquiry Concerning Human Understanding* — the philosophical starting point of the problem of induction; "the past cannot prove the future"

- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If* — causal inference in epidemiology and medicine, stressing the difference between intervention and observation `-> [Free online]`

- Schölkopf, B. et al. (2021). *Toward Causal Representation Learning* — the intersection of causal inference and deep learning; how to learn causal structure from data `-> [arXiv:2102.11107]`

- Bareinboim, E. & Pearl, J. (2016). *Causal inference and the data-fusion problem* — how to identify causal effects from multiple heterogeneous data sources `-> [PNAS]`

- Zhang, K. & Hyvärinen, A. (2009). *On the Identifiability of the Post-Nonlinear Causal Model* — identifiability of nonlinear causal models; a way to break Markov equivalence classes `-> [UAI 2009]`

- Uhler, C., Raskutti, G., Bühlmann, P. & Yu, B. (2013). *Geometry of the faithfulness assumption in causal inference* — geometric analysis of the faithfulness assumption and when it fails `-> [arXiv:1207.0547]`

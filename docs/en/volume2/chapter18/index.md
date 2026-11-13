# Chapter 18: Formalizing Causal Structure — The Three-Level Ladder and do-calculus

> You can never derive causality from data. Unless you're willing to admit that some structures are assumptions, not discoveries.

---

The end of Chapter 17 left us with an unpleasant fact: probability theory, however refined, cannot tell "$X$ causes $Y$" apart from "$Y$ causes $X$" — as long as the two produce the same joint distribution, Bayesian updating treats them alike.

This isn't a problem of computing power, and it isn't a problem of data volume. It's a structural limit of mathematics itself: **observation describes what the world looks like at rest, not what the world looks like after someone has pulled a lever**.

But we keep pulling levers. Doctors prescribe drugs. Policymakers adjust tax rates. Engineers tweak parameters. Every intervention asks a question probability theory cannot answer: "If I change $X$, what happens to $Y$?"

Answering that question requires a new kind of inference rule. The job of this chapter is to formalize the act of "changing" itself.

---

## 18.0 The Causal Wire-Cutting Game: Observing Is Not Intervening

In the probability game, you can only watch how the chips move. In the causal game, you finally get to cut wires.

Picture a causal circuit board. Each variable is a node, and each directed edge is a wire: $X \to Y$ means a change in $X$ travels down this wire and affects $Y$. When you observe, you just see which node is lit. When you intervene, you pick up a pair of scissors, cut every wire pointing into that node, and then force the node to sit at some fixed value.

That's what $\mathsf{do}(X=x)$ means in the game.

"Seeing $X=x$" doesn't change the circuit at all. It only tells you: right now this light happens to be lit at $x$. But "performing $\mathsf{do}(X=x)$" changes the circuit's structure: every wire that used to influence $X$ is cut, and $X$ is no longer decided by its parents — it's decided by your hand.

:::details The Formal Skeleton of This Game
- **State space**: a directed acyclic graph $G$, plus a set of structural equations $X_i = f_i(\mathrm{Pa}(X_i), U_i)$.
- **Legal moves**: observe some nodes; or perform an intervention $\mathsf{do}(X=x)$, cutting all incoming edges of $X$ and fixing its value.
- **Transition rules**: observation updates the distribution; intervention replaces the structural equations, yielding the mutilated graph (the graph after the wire cuts).
- **Win condition**: identify $P(Y \mid \mathsf{do}(X=x))$ or a counterfactual quantity, from the observable distribution and the causal graph.
- **Failure mode**: treating $P(Y \mid X=x)$ as $P(Y \mid \mathsf{do}(X=x))$ — misreading correlation as causation.
:::

What's fun about this game is that it turns "causality" from a metaphysical debate into a very concrete action: which wire do you cut? Which do you keep? Once the cuts are made, how does the signal travel?

A back-door path is a wire that sneaks around the back. A confounder is a hand hiding behind the board, pulling on two nodes at once. d-separation asks whether, once certain wires are blocked, information can still get from one side to the other.

::: info Professor Pallas's Cat's Commentary
The first lesson of causal inference is brutal: seeing is not changing. Your watching the sprinkler run, and your turning it on yourself, are not the same world. The most common arrogance in statistics is passing off observation as intervention. The scissors of a causal graph exist precisely to cut that arrogance away.
:::

## 18.0.5 Playing the Game in Code

The board is small enough to enumerate exactly — no sampling noise. The program below runs the two computations side by side: conditioning on "ice cream is high" and intervening $\mathsf{do}(\text{ice cream}=\text{high})$. The two numbers come out different, and the difference is exactly the confounder.

```python
#!/usr/bin/env python3
"""
causal_wire_cutting.py -- the Causal Wire-Cutting Game of Chapter 18.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 18, Section 18.0
"The Causal Wire-Cutting Game: Observing Is Not Intervening."

Each variable is a node, each edge a wire. Observing watches the lights;
intervening picks up scissors, cuts every wire into a node, and pins the
node to a value. This program runs a small causal circuit board:

      season (Z) ----> ice cream sales (X)
          |
          +----------> drownings (Y)

  * P(Y | X=x)      -- conditioning: watch the board, no scissors.
  * P(Y | do(X=x))  -- intervention: cut Z -> X, pin X, let the rest run.

The two numbers come out different. That difference IS the game -- and
it is why "seeing is not changing."

Everything below is exact enumeration over a small finite board: no
randomness, no sampling noise.

Run:  python3 causal_wire_cutting.py
"""
from itertools import product

# Worlds: (season, ux, uy). Noise ux/uy in 1..10 decides X and Y through
# the structural equations. Summer pushes both variables up -- that shared
# push is the confounder (the hand behind the board).
SEASONS = ("summer", "winter")
NOISE = range(1, 11)

def x_value(season, ux):
    """Structural equation of X: ice cream sales."""
    if season == "summer":
        return "high" if ux <= 9 else "low"
    return "high" if ux <= 2 else "low"

def y_value(season, uy):
    """Structural equation of Y: drownings."""
    if season == "summer":
        return "high" if uy <= 8 else "low"
    return "high" if uy <= 1 else "low"

def worlds():
    return product(SEASONS, NOISE, NOISE)

def prob_condition():
    """P(Y=high | X=high): watch the board, count the worlds."""
    both = total = 0
    for season, ux, uy in worlds():
        if x_value(season, ux) == "high":
            total += 1
            if y_value(season, uy) == "high":
                both += 1
    return both / total

def prob_intervene():
    """P(Y=high | do(X=high)): cut Z -> X, pin X=high, count the worlds.
    The scissors remove X's structural equation; X is decided by the hand.
    Y keeps its own equation, so only the seasonal push into Y remains.
    """
    high = total = 0
    for season, _, uy in worlds():
        total += 1
        if y_value(season, uy) == "high":
            high += 1
    return high / total

def demo():
    print("=" * 70)
    print(" THE CAUSAL WIRE-CUTTING GAME -- seeing is not changing")
    print(" The Reasoning Kingdom, Volume 2, Chapter 18 (Sec. 18.0)")
    print("=" * 70)
    print("\nCircuit:  season -> ice cream -> (nothing)")
    print("          season -> drownings")
    print("  the season pushes both ice cream sales and drownings up at")
    print("  once: a confounder, the hand behind the board.")
    p_cond = prob_condition()
    p_do = prob_intervene()
    print(f"\nP(drownings=high | ice cream=high)     = {p_cond:.4f}")
    print(f"P(drownings=high | do(ice cream=high)) = {p_do:.4f}")
    print("\nThe same 'ice cream is high' observation means different things:")
    print("  conditioning: the board is untouched; a high ice cream reading")
    print("    suggests summer, so drownings look likely too.")
    print("  intervention: scissors cut season -> ice cream, the hand pins")
    print("    X=high. Summer is no longer implied, and the drowning rate")
    print("    drops back to its unconditional level.")
    print(f"\nDifference: {p_cond - p_do:.4f} -- pure confounding, no data")
    print("  collection can remove it; only a wire cut can.")
    print("\nThe point of the game: correlation is what the board shows you;")
    print("  causation is what you are allowed to do to the board. The most")
    print("  common arrogance in statistics is passing off observation as")
    print("  intervention. The scissors exist to cut that arrogance away.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

## 18.1 The Three-Level Ladder

Judea Pearl described the layers of causal inference with a metaphor he calls the **Ladder of Causation**. The ladder has three rungs, from low to high, and each rung requires a capability the rung below it does not have.

**Level 1: Association**

This is probability theory's home turf. The questions look like:

> "Seeing $X$, what is $Y$?"

In math: $P(Y \mid X)$. This level needs nothing but data — enough observations, and you can estimate a conditional probability. Animals, babies, and almost every statistical model live on this rung.

**Level 2: Intervention**

The questions change shape:

> "If I set $X$ to some value, what would $Y$ be?"

The difference from Level 1 is fundamental. "Seeing $X=x$" and "setting $X$ to $x$" are two completely different acts. The first is passive observation; the second is active manipulation.

When you observe passively, $X=x$ might hold because some common cause $Z$ influenced both $X$ and $Y$. When you intervene actively, you cut all the links between $X$ and everything that causes it, and force $X$ to stay fixed at $x$ — only then does a change in $Y$ really come from $X$'s direct effect on $Y$.

This level needs more than data. It needs action — or, when action isn't possible, some tool that lets you simulate the action mathematically.

**Level 3: Counterfactual**

The questions get harder still:

> "If $X$ hadn't been that value, what would $Y$ have been?"

This is now a question about one individual in another possible world. "This patient took the drug and recovered — if he hadn't taken the drug, would he still have recovered?" No amount of observation or experimentation can answer that directly, because that parallel world in which he didn't take the drug is one we can never enter.

::: info The Three Levels Are a Real Hierarchy, Not Philosophical Rhetoric
These three levels correspond to real boundaries of capability. Pure observational data can only answer Level 1 questions. A randomized controlled trial (RCT — an experiment that randomly assigns people to a treatment group or a control group) can answer Level 2, but the price is actually carrying out the intervention. Counterfactual inference needs a complete causal model plus extra assumptions about "individual mechanisms" — beyond anything an experiment can reach. Most statisticians have spent the better part of their careers working on Level 1 while believing they were answering Level 2 questions. That confusion has produced a great many faulty conclusions in the scientific literature.
:::

::: info Professor Pallas's Cat's Commentary
"Mistakenly believed" is far too kind. This isn't a cognitive error; it's a tool error. When the only hammer you hold is correlation, every problem looks like a correlation nail. The problem isn't statisticians' intelligence — it's that standard training has never drawn the line between Level 1 and Level 2 cleanly.
:::

---

## 18.2 Graphs: The Geometry of Causality

The first step is to give "causal structure" a precise mathematical representation.

The **directed acyclic graph (DAG)** is the most natural tool. Nodes represent variables, and directed edges represent direct causal influence: $X \to Y$ means $X$ directly affects $Y$. "Acyclic" is an important constraint — causality can't loop around (if $X$ causes $Y$ and $Y$ causes $X$, then time itself becomes paradoxical).

A simple example. Consider three variables: season ($S$), whether the sprinkler is on ($W$), and whether the lawn is wet ($G$). The intuitive causal structure is:

$$S \to W, \quad S \to G, \quad W \to G$$

Season affects the sprinkler (it runs more in summer), season also directly affects the lawn (rain), and the sprinkler also directly affects the lawn.

In this graph, $W$ and $G$ are correlated — when the sprinkler is on, the lawn is more likely to be wet. But this correlation travels along two routes: a direct causal route $W \to G$, and a "back-door path" through the common cause $S$: $W \leftarrow S \to G$.

Probability theory sees the combined result of both routes and can't separate them. The graph draws the structure out explicitly.

::: info Graphs Aren't Read Off from Data
This is the most common beginner mistake. A causal graph represents **domain knowledge**, not a product of statistical inference. You can't derive this graph from $P(S, W, G)$ — the data only tells you how strongly variables are correlated, not which way the arrows point. The causal graph is where you write down how the world works. It is an assumption, not a discovery. Accepting this takes an uncomfortable cognitive leap: scientific inference never starts from data alone — it always enters the room carrying structural assumptions.
:::

::: info Professor Pallas's Cat's Commentary
The arrows' directions are brought in by you, not given by the data — this sentence will make a lot of data scientists uncomfortable, because they're used to "letting the data speak." But data can't say this. Admitting it takes courage, and many papers dodge this step. The price of dodging is hiding the assumption inside the method and pretending it's a conclusion.
:::

---

## 18.3 Structural Causal Models

A graph is just a skeleton. To turn it into a machine that can do inference, you need to attach content to every edge. That's the **structural causal model (SCM)**.

An SCM consists of three parts:

**Exogenous variables $U$**: noise or background factors from outside the system, not determined by any other variable in the model. They capture all the randomness we cannot observe, or choose not to model.

**Endogenous variables $V$**: variables determined by other variables in the model, including every observable quantity we care about.

**Structural equations**: for each endogenous variable $V_i$, there is an equation

$$V_i = f_i(\mathsf{Pa}(V_i),\, U_i)$$

where $\mathsf{Pa}(V_i)$ is the set of **parents** of $V_i$ in the causal graph (its direct causes), and $U_i$ is the noise term for that variable.

This equation is not a statistical regression equation — it's a **mechanism**, describing "what value this variable takes, given its parents and its noise." The mechanism is stable and local: changing the equation of some other variable doesn't affect this one. That local stability is the core feature that separates causal models from statistical models.

Back to the sprinkler example. The structural equations might be:

$$W = f_W(S, U_W), \quad G = f_G(S, W, U_G)$$

$W$ is decided by season and its own noise; $G$ is decided by season, the sprinkler's state, and its own noise. Each equation describes one local mechanism, and they are independent of each other.

---

## 18.4 The do Operator: Formalizing Intervention

Now we can define "intervention" precisely.

The operational definition of the intervention $\mathsf{do}(X=x)$ is: **replace the structural equation for $X$ with the constant equation $X=x$**, while keeping every other equation unchanged.

The effect on the graph is easy to see: **delete every edge pointing into $X$**. Because $X$ is now forced to a fixed value, its parents' influence on it is cut off. $X$'s descendants still feel $X$ (the outgoing edges remain), but $X$ no longer responds to its causes (the incoming edges are gone).

This "operated-on" graph is called the **interventional graph**, written $G_{\overline{X}}$ — the graph with $X$'s incoming edges removed.

:::details The do Operator vs. Conditioning
Why do $P(Y \mid X=x)$ and $P(Y \mid \mathsf{do}(X=x))$ differ? This is the most central distinction in causal inference, and the easiest thing to confuse:

**Conditioning $P(Y \mid X=x)$**: among the samples where you observed $X=x$, what is the distribution of $Y$? This is **passive observation** — you're asking "in the worlds where $X$ happens to equal $x$, what does $Y$ look like?" The trouble is that what made $X=x$ might share a common cause with $Y$ (a confounder), which twists the correlation.

**Intervention $P(Y \mid \mathsf{do}(X=x))$**: if I **force** $X$ to stay fixed at $x$ (deleting all of $X$'s causes), what would $Y$ be? This is **active manipulation** — you've cut off the confounding routes and left only $X$'s direct causal effect on $Y$.

A concrete example:
- $P(\text{recovered} \mid \text{treatment}=1)$: among the people who chose to take the drug, what is the recovery rate? (Probably too high — milder patients are more likely to choose the drug.)
- $P(\text{recovered} \mid \mathsf{do}(\text{treatment}=1))$: if I forcibly assign a random set of people to take the drug (a randomized controlled trial), what is the recovery rate? (The true drug effect.)

The **do operator** is the mathematical expression of a randomized controlled trial. When an RCT isn't feasible, do-calculus provides the rules for estimating an interventional effect from observational data.
:::

The post-intervention probability distribution is written $P(Y \mid \mathsf{do}(X=x))$, sometimes also written $P_x(Y)$, and it is defined in the interventional graph $G_{\overline{X}}$.

Compare it with the conditional probability:

$$P(Y \mid X=x) \quad \text{vs.} \quad P(Y \mid \mathsf{do}(X=x))$$

The first is "the distribution of $Y$ among the samples where $X=x$ was observed." The second is "what the distribution of $Y$ would be if every $X$ in the whole world were forced to $x$."

The two can differ enormously. In the sprinkler example, $P(G=\text{wet} \mid W=\text{on})$ includes the confounding effect of "summer makes the sprinkler run, and summer also makes it rain." But $P(G=\text{wet} \mid \mathsf{do}(W=\text{on}))$ cuts off the route $W \leftarrow S$ and keeps only the direct effect $W \to G$. That is the "causal effect of the sprinkler on the lawn's wetness."

::: info What a Randomized Controlled Trial Actually Does
The core operation of a randomized controlled trial — randomly assigning subjects to a treatment group and a control group — is mathematically exactly $\mathsf{do}$: it cuts the links between the treatment variable (does take the drug / does not) and every possible confounding factor. Randomization is equivalent to deleting all the edges pointing into the treatment variable in the causal graph. That's why the RCT is the "gold standard" for estimating causal effects — it implements the $\mathsf{do}$ operator in physical reality. The value of do-calculus is this: it tells you when you can compute $P(Y \mid \mathsf{do}(X))$ from observational data without running an experiment at all.
:::

### What the do operator looks like in a matrix and in code

The $\mathsf{do}$ operator isn't just a mathematical symbol — it can be implemented precisely in two lines of code: cut the incoming edges, then propagate the effect.

Write the unit structural equation $E_j = \sum_i A_{ij} \cdot E_i + U_j$ in matrix form: $E = A^{\top} E + U$. Then $\mathsf{do}(X=v)$ is:

```python
import numpy as np
A = np.array([[0, 0.9, 0.8],      # structure: ad_spend -> clicks -> revenue
              [0,   0, 0.7],
              [0,   0,   0]])
E = np.random.randn(3, 16)         # one embedding per variable
U = np.zeros_like(E)
# step 1: cut all incoming edges of X (fill in column X with zeros)
def do_cut(A, var):
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do
# step 2: propagate the effect along the remaining edges
def propagate(A_do, E, U):
    return A_do.T @ E + U
# do(ad_spend = 3.0): the value enters as a constant, then effects flow
A_do = do_cut(A, 0)
E_next = propagate(A_do, E, U)
```

**The correspondence with Pearl's definition:**

| Pearl's do operator | Matrix/code implementation |
|---|---|
| Replace $X$'s structural equation with $X=v$ | Zero out the column for $X$; the value $v$ enters as a constant |
| Delete all edges pointing into $X$ | Set column $A_{\cdot,X}$ to zero |
| Propagate the effect through descendants | One matrix multiplication via `propagate` |
| Cyclic graphs are illegal | $A$ must respect a topological order (upper triangular) |

That's all there is to it — the most profound operator in causal inference, at the implementation level, is "zero out a column, multiply once." You don't need any extra framework or type system.

---

## 18.5 The Back-Door Criterion: The Geometry of Confounding

The core question of do-calculus is: **under what conditions can you estimate a causal effect from observational data** — that is, when can $P(Y \mid \mathsf{do}(X=x))$ be computed from $P$ and the structure of the graph, without actually intervening?

Answering that requires understanding the geometry of "confounding."

In a causal graph, $X$'s total effect on $Y$ is "confounded" exactly when there exists a non-causal path that starts at $X$, does not pass through any descendant of $X$, and still reaches $Y$. Such paths are called **back-door paths** — they route around the direct effect $X \to Y$ and smuggle spurious correlation in through the back door.

The **back-door criterion** gives an exact condition: a variable set $Z$ satisfies the back-door criterion (with respect to $X \to Y$) if and only if:

1. $Z$ contains no descendants of $X$;
2. $Z$ blocks every back-door path from $X$ to $Y$.

If such a $Z$ exists, the causal effect can be computed with this formula:

$$P(Y \mid \mathsf{do}(X=x)) = \sum_z P(Y \mid X=x, Z=z)\, P(Z=z)$$

This formula is called the **adjustment formula**. It means: for each value of $Z$, compute the conditional distribution of $Y$ at $X=x$, then average them, weighted by the marginal distribution of $Z$. This operation is called **adjusting for $Z$**, and in epidemiology it's known as "controlling for the confounder."

In the sprinkler example, $X=W$, $Y=G$, and the back-door path is $W \leftarrow S \to G$. Pick $Z=\{S\}$: $S$ is not a descendant of $W$ (condition 1 holds), and $S$ blocks the back-door path $W \leftarrow S \to G$ (condition 2 holds). So:

$$P(G \mid \mathsf{do}(W=w)) = \sum_s P(G \mid W=w, S=s)\, P(S=s)$$

This quantity is fully determined by observational data — no need to actually touch the sprinkler.

### 18.5.5 Hands-on: Front-Door Adjustment — When the Back Door Is Blocked

The back-door criterion requires an observable confounder $Z$. But what if the confounder is not observable? Say you don't know about a hidden variable, "desire to smoke," which affects both the amount smoked and the risk of lung cancer.

That's when the **front-door criterion** can save you — as long as there is a **mediator variable** $M$ satisfying:

1. $M$ blocks all causal paths from $X$ to $Y$;
2. There is no back door from $X$ to $M$ (no unobservable confounding between $X$ and $M$);
3. The back door from $M$ to $Y$ is blocked by $X$ (after adjusting for $X$, there is no confounding from $M$ to $Y$).

The classic example: **smoking $\to$ tar accumulation $\to$ lung cancer** (from Pearl's original paper).

- $X = \text{smoking}$, $M = \text{tar accumulation}$, $Y = \text{lung cancer}$
- The hidden variable $U = \text{desire to smoke / genes}$ affects both $X$ and $Y$ (a back-door path), but does **not** affect $M$ (no hidden confounding between $X$ and $M$ — the amount smoked directly determines the tar; genes don't directly change the tar)
- Back-door criterion: you'd need to observe $U$, but $U$ is unobservable $\to$ the back-door criterion fails
- Front-door criterion: identify the effect through $M$ (the tar)

The front-door adjustment formula:

$$P(Y \mid \mathsf{do}(X=x)) = \sum_m P(M=m \mid X=x) \cdot \sum_{x'} P(Y \mid X=x', M=m) \cdot P(X=x')$$

Unpacked in plain language:

- The **first term** $P(M=m \mid X=x)$: when someone smokes $x$, how much tar $m$ results? This is $X$'s causal effect on $M$ (no back door, directly estimable).
- The **second term** $\sum_{x'} P(Y \mid X=x', M=m) \cdot P(X=x')$: given tar $m$, the weighted average of lung-cancer probability over the smoking population — here you adjust for $X$ to block the back door from $M$ to $Y$.
- **Multiply, then sum over $m$**: combining $X$'s effect on $M$ with $M$'s effect on $Y$ gives the total causal effect of $X$ on $Y$.

**A numerical example** (fictional data, simplified):

| Smoking $X$ | Tar $M$ | Lung cancer $Y$ | Count |
|---|---|---|---|
| 1 | high | 1 | 300 |
| 1 | high | 0 | 100 |
| 1 | low | 1 | 50 |
| 1 | low | 0 | 150 |
| 0 | high | 1 | 30 |
| 0 | high | 0 | 70 |
| 0 | low | 1 | 20 |
| 0 | low | 0 | 280 |

Compute $P(Y=1 \mid \mathsf{do}(X=1))$ — if everyone were forced to smoke, what would the lung-cancer probability be?

1. $P(M=\text{high} \mid X=1) = \frac{400}{600} = 0.667$; $P(M=\text{low} \mid X=1) = \frac{200}{600} = 0.333$

2. Given $M=\text{high}$:

   $$\sum_{x'} P(Y=1 \mid X=x', M=\text{high}) \cdot P(X=x')$$
   $$= P(Y=1 \mid X=1, M=\text{high}) \cdot P(X=1) + P(Y=1 \mid X=0, M=\text{high}) \cdot P(X=0)$$
   $$= \frac{300}{400} \cdot 0.6 + \frac{30}{100} \cdot 0.4 = 0.75 \cdot 0.6 + 0.30 \cdot 0.4 = 0.45 + 0.12 = 0.57$$

   Given $M=\text{low}$:

   $$\sum_{x'} P(Y=1 \mid X=x', M=\text{low}) \cdot P(X=x')$$
   $$= \frac{50}{200} \cdot 0.6 + \frac{20}{300} \cdot 0.4 = 0.25 \cdot 0.6 + 0.067 \cdot 0.4 = 0.15 + 0.027 = 0.177$$

3. $P(Y=1 \mid \mathsf{do}(X=1)) = 0.667 \times 0.57 + 0.333 \times 0.177 = 0.380 + 0.059 = 0.439$

If everyone were forced to smoke, the lung-cancer probability would be 43.9%. The plain conditional probability $P(Y=1 \mid X=1) = \frac{350}{600} = 0.583$ is higher — because it includes the confounding effect of $U$. do-calculus correctly separates the causal effect from the confounding effect.

::: info Professor Pallas's Cat's Commentary
Front-door adjustment is an underrated gem. Most textbooks teach only the back-door criterion, because the back door is more intuitive. But the back door requires observing the confounder — which is often just not possible. The front-door criterion does not require observing the confounder at all; it uses a mediator you can observe. In practice that makes it far more usable than the back door, and far more counterintuitive — you bypass an unseen confounder by going through a mediator. Sometimes a detour is the right way.
:::

---

## 18.6 The Three Rules of do-calculus

The back-door criterion covers a lot of real cases, but not all. In some causal graphs, no set of observable variables can completely block the back-door paths — for instance, when unobservable confounders are present.

To handle the more general cases, Pearl introduced **do-calculus**: three inference rules about the $\mathsf{do}$ operator that together form a complete calculus for causal inference.

Let $X, Y, Z, W$ be variable sets in a causal graph $G$. Write $G_{\overline{X}}$ for the graph with $X$'s incoming edges removed, and $G_{\underline{X}}$ for the graph with $X$'s outgoing edges removed.

**Rule 1 (Insertion and Deletion of Observations)**:

$$P(Y \mid \mathsf{do}(X),\, Z,\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

holds if and only if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}}$.

In words: if in the interventional graph $Z$ carries no extra information about $Y$ (it is blocked by $X$ and $W$), then observing $Z$ or not does not affect your inference about $Y$.

**Rule 2 (Interchange Between Action and Observation)**:

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, Z,\, W)$$

holds if and only if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}\,\underline{Z}}$.

In words: under certain conditions, intervening on $Z$ and observing $Z$ have the same effect on $Y$ — the intervention loses its "forcefully cutting" specialness and degenerates into ordinary conditioning.

**Rule 3 (Elimination of Interventions)**:

$$P(Y \mid \mathsf{do}(X),\, \mathsf{do}(Z),\, W) = P(Y \mid \mathsf{do}(X),\, W)$$

holds if and only if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}\,\overline{Z(W)}}$ (where $Z(W)$ consists of the nodes in $Z$ that are not ancestors of $W$).

In words: under certain conditions, $\mathsf{do}(Z)$ has no effect on $Y$ and can be dropped from the formula.

::: info Completeness of the Three Rules
Pearl and Shpitser proved this: for any causal graph, if $P(Y \mid \mathsf{do}(X))$ can be computed from the observational distribution $P$ (that is, it is "identifiable"), then repeatedly applying these three rules will always produce an expression containing only observational probabilities. This is the **completeness theorem** of do-calculus — it is not merely a toolbox but a genuine calculus, one that never misses any causal effect that could be identified.
:::

The form of these three rules closely resembles the inference rules of Chapter 14: what goes on top of the line is a condition; what goes below it is the inference you may draw. The difference is that the "language" here is not just propositions — it's probability expressions carrying the $\mathsf{do}$ operator. And the graph's structure, the conditional independencies, plays the role of the axioms.

Back to the wire-cutting game: the three rules of do-calculus are the move tables you consult after the cuts. When can you drop an observation? When can you trade an action for an observation? When can an intervention be eliminated entirely? The answers lie not on the surface of the formulas, but in the graph after the cutting. Causal inference is stronger than probabilistic inference because it admits "actions" into the syntax; it is more dangerous for the same reason — every action depends on your having drawn that graph correctly.

---

## 18.7 d-Separation: The Independence Language of Graphs

Each of the three rules of do-calculus relies on one core judgment: whether two sets of variables are **conditionally independent** given a third set, in some graph. How do you read conditional independence directly off a graph's structure? That's exactly the job of **d-separation** (directional separation).

Given a directed acyclic graph $G$ and three groups of nodes $X, Y, Z$, we say $X$ and $Y$ are **d-separated** by $Z$, written $X \perp_G Y \mid Z$, if and only if $Z$ blocks every path between $X$ and $Y$.

Whether a path is "blocked" depends on the types of nodes along the path:

- **Chain** $A \to B \to C$: the path is blocked when $B$ is in $Z$ ($B$ passes the information on; once you control $B$, the flow stops).
- **Fork** $A \leftarrow B \to C$: the path is blocked when $B$ is in $Z$ ($B$ is a common cause; once you control $B$, the correlation disappears).
- **Collider** $A \to B \leftarrow C$: the path is blocked when $B$ is **not** in $Z$ (a collider stops information by default; but controlling it, or any of its descendants, actually opens the path — that's where "collider bias" comes from).

The behavior of colliders is counterintuitive, and it's worth stopping to think it through. Suppose "height" ($H$) and "basketball skill" ($T$) both affect "made the basketball team" ($S$): $H \to S \leftarrow T$. Across the whole population, $H$ and $T$ may be independent. But look only at the people who made the team — which is equivalent to controlling for $S$ — and short players are often unusually skilled while tall players are sometimes merely average. Control a collider and variables that were independent become correlated. That's **selection bias** in statistics, which in the language of graphs is the natural consequence of a path opening once its collider is controlled.

---

## 18.8 Counterfactuals: Another World for a Single Individual

The top rung of the ladder, the counterfactual, needs a precise definition inside the structural causal model.

Consider one particular individual who, in a particular situation, experienced $X=x$ and got the outcome $Y=y$. The counterfactual question is: if this individual had experienced $X=x'$ instead, what would $Y$ have been?

In an SCM, this question has a three-step answer:

**Step 1: Abduction.** Using the observed $X=x$, $Y=y$, along with all other observations, update the distribution of this individual's exogenous noise variables $U$ — pushing the prior $P(U)$ to a posterior $P(U \mid \text{observations})$. This step is done by Bayesian updating, and it encodes the individual's "particularity" into the posterior of $U$.

**Step 2: Action.** In the structural equations, replace the equation for $X$ with $X=x'$ (that is, apply the intervention $\mathsf{do}(X=x')$), leaving every other equation unchanged.

**Step 3: Prediction.** Under the modified model and the posterior distribution of $U$ from Step 1, compute the distribution of $Y$. This gives the counterfactual conclusion: if this individual had faced $X=x'$, what would the distribution of $Y$ have been.

These three steps together give an operational definition of counterfactual inference — not philosophical speculation, but a computable procedure, provided you have a sufficiently complete SCM.

::: info Average Treatment Effect vs. Individual Treatment Effect
The "average treatment effect" (ATE) familiar from epidemiology is a Level 2 concept: $\mathbb{E}[Y \mid \mathsf{do}(X=1)] - \mathbb{E}[Y \mid \mathsf{do}(X=0)]$. It answers "across the whole population, on average, how much does the treatment change the outcome?" The "individual treatment effect" (ITE) is a Level 3 concept: for this particular person, if the treatment changed, how much would their outcome change? The ITE is in principle unidentifiable from observational data (every person can only ever be in one treatment state) — that is the fundamental difficulty of counterfactual inference, and the mathematical root of the precision-medicine problem.
:::

---

## 18.9 Causality and Inference: A Bigger Picture

Review the road this chapter has traveled.

Back in Chapter 14, inference rules were about propositions: from a batch of assumptions $\Gamma$, derive new propositions. Chapter 15 found that this mechanism has internal limits — some true propositions simply cannot be proven. Chapter 16 questioned the default that "assumptions can be reused" and found that dropping it yields a new kind of logic. Chapter 17 stretched truth values from $\{0,1\}$ to $[0,1]$, turning inference into the updating of beliefs.

Now this chapter does something new: it adds a new verb to the language of inference — **intervention**. The $\mathsf{do}$ operator is not conditioning; it is a new inference rule that needs to be handled separately in the grammar. The graph supplies the context for the rule: when $\mathsf{do}$ can be simplified away, and when it cannot.

This structure runs exactly parallel to the structure of a formal system:

| Formal system | Causal inference |
|---|---|
| Propositional variables | Variables |
| Inference rules | The three rules of do-calculus |
| Axioms | The graph (local independence assumptions) |
| Provable ($\vdash$) | Identifiable |
| Incompleteness theorem | Non-identifiability theorem |

The non-identifiability theorem plays the role of the incompleteness theorem: there are causal effects that, even given complete observational data and a complete graph, can never be computed from observation alone — you must actually run the experiment. That is not a flaw in the method; it is a structural boundary.

---

## Open Questions

**Where do causal graphs come from?** The whole chapter assumes you already have a correct causal graph. But in practice, a graph is the product of subjective knowledge and domain assumptions. If the graph is wrong, the answers do-calculus gives are wrong too — the reliability of the system rests on the correctness of the graph, and the graph's correctness cannot be verified from data. This is the deepest dilemma of causal inference: the more precise the tool, the heavier the assumptions it carries.

**Can causal graphs be discovered automatically from data?** Causal discovery algorithms — the PC algorithm, the FCI algorithm, LiNGAM — try to infer aspects of causal structure from observational data. But in general, observable data can only pin down a "Markov equivalence class" — a set of graphs that yield the same conditional-independence relations, with arrow directions indistinguishable within the class. Picking one graph out of the class still requires domain knowledge, or additional assumptions about the error distributions.

**How expensive is inference on a causal graph?** Even with the graph in hand, identifying and computing causal effects is computationally expensive in general. Deciding whether a causal effect is identifiable can be done in polynomial time; but in complex graphs with hidden variables, computing the actual expression for an identifiable effect becomes dramatically more expensive. The full picture of this problem is the topic of Chapter 19: the price of reasoning is not only a logical limit, but a computational one.

---

## Exercises

**★ Warm-up**

For each of the following questions, determine which rung of the ladder of causation it belongs to (association / intervention / counterfactual), and explain why.

1. "How much higher is the lung-cancer rate among smokers than among non-smokers?"
2. "If a smoking ban were strictly enforced, how much would the lung-cancer rate fall?"
3. "If this lung-cancer patient had never smoked back then, would he have cancer now?"
4. "Among people diagnosed with high blood pressure, what fraction are on blood-pressure medication?"
5. "If this high-blood-pressure patient is prescribed medication, how much will his risk of a cardiovascular event drop over the next 10 years?"

---

**★★ Derivation**

Consider the causal graph: $Z \to X \to Y$, and also $Z \to Y$ ($Z$ affects both $X$ and $Y$, and $X$ affects $Y$ as well).

1. List every path from $X$ to $Y$. Which are causal paths (following the arrows), and which are back-door paths?
2. Using the back-door criterion, write out the adjustment formula for $P(Y \mid \mathsf{do}(X=x))$.
3. If $Z$ is an unobservable hidden variable, can the adjustment formula still be used? Is $P(Y \mid \mathsf{do}(X=x))$ identifiable in that case?

---

**★★★ Challenge**

In the causal graph $Z \to X \to Y$ (with $Z$ observable), add a mediator variable $M$ satisfying $X \to M \to Y$, while keeping the direct path $X \to Y$ intact.

You want to separately estimate:

- the **direct effect** of $X$ on $Y$ (the part not going through $M$)
- the **indirect effect** of $X$ on $Y$ (the part going through $M$)

Which rung of the ladder do the direct and indirect effects belong to? Try writing the definitions of these two quantities in the language of the $\mathsf{do}$ operator — you'll find that defining the "direct effect" requires two interventions at the same time. Then think further: why is the indirect effect (the natural indirect effect) harder to define than the direct effect, forcing you to appeal to counterfactuals?

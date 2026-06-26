# Chapter 23: Stability and Convergence Boundaries of Reasoning Systems

> What is the biggest pain point of Lyapunov functions? Exactly — you have to manually specify an energy function. If we use Yonglin's convergence hypothesis to derive the Lyapunov function for a reasoning system, that would be much better.

---

At the end of Chapter 22, we stood at the boundary of self-reference and emergence, witnessing the strange moment when a reasoning system begins to reason about itself. That boundary, mathematically, manifests as fixed points, diagonalization, undecidability.

But the boundary is not only logical. The boundary is also **dynamical** — how the system evolves over time, whether it is stable, where it converges. The Yonglin formula gives a specific convergence pattern: the inference chain ultimately returns to the prior anchor point. This convergence, in the language of dynamical systems, is precisely an **attractor**.

This chapter will do one thing: translate the convergence of the Yonglin formula into the language of Lyapunov stability. And then the reverse: use the convergence hypothesis to derive the Lyapunov function, rather than manually specifying it.

**Objective**: Using the Yonglin convergence hypothesis + the information distance of KL distribution bias + Euler-step iterative updates, derive the Lyapunov function of the reasoning system. It is known that Lyapunov functions traditionally require manual construction, dependent on human priors. But by observing the system's dynamical behavior (Yonglin convergence), we reverse-engineer the energy function — this is deriving structure from behavior, another recurrence of the **inverse problem**, and also an embodiment of the art of dynamical systems.

---

## 23.0 Prologue: The Dynamical Story of a Stack

To understand Lyapunov functions, let us first play a little game.

Imagine a stack data structure. It has two pointers:
- **$\beta$ pointer**: always points to the bottom of the stack, fixed in place
- **$\alpha$ pointer**: points to the top of the stack, can move up and down

The operations on the stack are simple:
- **`push(x)`**: push element `x` onto the top of the stack, the $\alpha$ pointer moves up one cell
- **`pop()`**: pop the top element, the $\alpha$ pointer moves down one cell

**But there is an important constraint**: negative stack is illegal. That is, the $\alpha$ pointer must always be above (or equal to) the $\beta$ pointer. When $\alpha = \beta$, the stack is empty; if one attempts to execute `pop()` on an empty stack, the operation is invalid — the $\alpha$ pointer does not move, or the program reports an error. In our dynamical story, we assume invalid `pop` operations are ignored.

Students of the Olympiad in Informatics (OI) are all too familiar with this structure. But Professor Pallas's Cat is not asking an algorithmic question today, but a **dynamical question**:

> Given a very, very long — so long you cannot imagine — sequence of `push` and `pop` operations, will the position of the $\alpha$ pointer (its height relative to the bottom of the stack) converge? Will it settle at a specific height, no matter how the operation sequence is arranged?

**Wait**, we also have the "negative stack illegal" constraint. Under this constraint, things become even more interesting.

::: info Professor Pallas's Cat's Question
Now the $\alpha$ pointer cannot go below the $\beta$ pointer. Attempting to `pop` when the stack is empty ($\alpha = \beta$) is ignored. Where do you think the $\alpha$ pointer will converge, under such constraints? Regardless of how complex or long the operation sequence is, will the $\alpha$ pointer eventually stabilize at some height? Or will it oscillate within some range?

Take three seconds to think.
:::

**What intuition might tell you**: Since `pop` is invalid when the stack is empty, the stack may become empty more easily — because an empty stack cannot be further popped, while `push` can make it non-empty. But conversely, if there are many `push` operations, the stack will become very tall.

**Key observation**: The stack height $h$ (the distance between $\alpha$ and $\beta$) now has a **hard lower bound** of $0$. This lower bound changes the system's dynamics.

### Stack Height as Energy

Define stack height $h$ = the distance between the $\alpha$ pointer and the $\beta$ pointer (measured in number of elements). Each `push` increases $h$ by 1; each `pop` decreases $h$ by 1.

Now define a function $V(h) = h^2$. What is special about this function?

**Observation**:
- $V(h) \geq 0$, and $V(h) = 0$ if and only if $h = 0$ (stack empty)
- If the operation sequence is random (`push` and `pop` equally probable), then the expected change in $h$ is zero — a random walk

But Professor Pallas's Cat is asking not about expectation but about **deterministic** behavior: if we know whether each step is `push` or `pop` (not random), how will $V(h)$ change?

### Something Magical Happens

Consider the continuous-time approximation: assume operations occur very fast, treating discrete operations as a continuous flow. Define the rate of change of stack height $\dot{h} = u(t)$, where $u(t) = +1$ indicates `push`, $u(t) = -1$ indicates `pop`.

Now compute the rate of change of $V(h) = h^2$:

$$\dot{V} = \frac{d}{dt}(h^2) = 2h\dot{h} = 2h \cdot u(t)$$

This seems patternless — $u(t)$ can be positive or negative, and $\dot{V}$ can also be positive or negative. But what happens if we integrate $\dot{V}$ over a long time?

**If the operation sequence is ultimately balanced** — `push` and `pop` are roughly equal in number — then $h$ will oscillate around some average value. $V(h) = h^2$ will not decrease monotonically, but its **long-term average** may converge.

### A Smarter Energy Function

Let us try a different definition: $V(h) = |h - h_0|$, where $h_0$ is some target height. If the operation sequence is **balanced** (`push` and `pop` ultimately equal in number), then $h$ will converge to the initial height $h_0$ (assuming the stack is initially non-empty). At this point, does $V(h)$ decrease over time? Not necessarily — $h$ may oscillate around $h_0$.

But if we consider the **variance** $V(h) = (h - h_0)^2$, and the operation sequence is **random**, then $\mathbb{E}[V(h)]$ may grow over time (the variance of a random walk grows linearly).

### The Magic of Negative Stack Illegal: Convergence to Empty Stack

Now add the constraint that negative stack is illegal. The $\alpha$ pointer cannot go below the $\beta$ pointer; `pop` on an empty stack is ignored.

**Professor Pallas's Cat's answer**: Under such constraints, if the operation sequence is sufficiently long and contains enough `pop` operations, the $\alpha$ pointer will ultimately converge to the **empty stack** state — $h = 0$.

Why?

Consider $V(h) = h$, the simplest possible energy function. When $h > 0$:
- Execute `push`: $h$ increases by 1, $V$ increases
- Execute `pop`: $h$ decreases by 1, $V$ decreases

But the key is: when $h = 0$, `pop` is ignored, $h$ remains 0, $V$ remains 0. So $h = 0$ is an **absorbing state** — once entered, one cannot leave (unless there is a `push`).

If the operation sequence is infinitely long and the number of `pop` operations is sufficiently large (not necessarily more than `push`, but as long as `pop` occurs), the system has a probability of entering the $h = 0$ state. Once entered, subsequent `pop` operations are invalid; only `push` can pull it out. But if the sequence is **non-deterministic** (e.g., random), then in the long run, the system will frequently visit the $h = 0$ state.

More rigorously: define $V(h) = h$. Then the change in $V$, $\Delta V = \Delta h$:
- `push`: $\Delta V = +1$
- `pop` (when $h>0$): $\Delta V = -1$
- `pop` (when $h=0$): $\Delta V = 0$

$V$ does not decrease monotonically, but $V$ has a lower bound of 0, and the system repeatedly visits the $V=0$ state. On an infinitely long time scale, the proportion of time the system spends in the $V=0$ state may be very high.

**But this is not Lyapunov stability**, because $V$ does not decrease monotonically. Lyapunov requires $\dot{V} \leq 0$ to hold for all time, which is not satisfied here.

### Key Insight

What does the stack story tell us?

1. **Dynamical system**: The stack height $h$ is a dynamical system; its evolution is driven by the operation sequence $u(t)$.
2. **Energy function**: $V(h)$ is an "energy" metric for the system. We want to use $V$ to judge whether the system converges.
3. **Convergence condition**: If there exists $V$ such that $\dot{V} \leq 0$ (energy does not increase), then the system is stable.
4. **Problem**: For the stack, simple $V(h) = h$ or $V(h) = h^2$ do not satisfy $\dot{V} \leq 0$, because $u(t)$ can be positive or negative.
5. **Effect of the lower bound**: The hard constraint $h \geq 0$ gives the system an absorbing state $h=0$, but the existence of an absorbing state does not guarantee that the stability theorem holds.

So we need a more ingenious $V$, or we need to impose constraints on the operation sequence $u(t)$.

### Lessons from the Stack Model

The stack is the simplest possible dynamical system, but we **still need to think hard to find a suitable $V$**. For more complex systems (such as neural networks, reasoning systems), finding $V$ is even harder.

This is the **pain point** of Lyapunov functions: you have to guess a $V$, verify the conditions, and guess again if it doesn't work. Guess correctly, and system stability is proven; fail to guess, and the proof gets stuck.

But wait — if we **observe** the system's behavior and find that it does converge, can we **reverse-engineer** $V$ from the convergence behavior? This is the core idea of the Yonglin-Lyapunov combination.

### Stack and the MP Game: A Comparison

In the MP game of Chapter 22, the proof sequence $\{p_1, p_2, \dots\}$ is an orbit, $p_{n+1} = f(p_n)$, and the orbit has three fates: finite halting, finite cycle, infinite extension.

The stack story is a concrete version of the same thing:

- The height $h$ of the $\alpha$ pointer is the state
- `push`/`pop` is the evolution operator $f$
- The empty stack $h=0$ is the absorbing state — the **fixed point** $f(\varphi)=\varphi$ of the MP game

The question left by the MP game is: fixed points exist, but starting from an arbitrary $p_0$, which fixed point will the system be attracted to? The stack story gives an intuition: in a system with a hard constraint ($h \geq 0$), the absorbing state is precisely that constraint boundary. The "hard constraint" of a reasoning system is the training data — it defines the prior anchor point $A$, i.e., the Yonglin Limit.

---

## 23.2 The Pain Point of Lyapunov Functions

**Definition (Lyapunov Function)**: For a dynamical system $\dot{x} = f(x)$, if there exists a continuously differentiable function $V(x)$ satisfying

1. $V(x) \geq 0$, and $V(x) = 0$ if and only if $x = x^*$ (equilibrium point)
2. $\dot{V}(x) = \frac{dV}{dt} \leq 0$ holds for all $x$

then $x^*$ is **stable**. If $\dot{V}(x) < 0$ (strictly negative except at $x^*$), then $x^*$ is **asymptotically stable**.

$V$ is called a Lyapunov function, intuitively the "energy" of the system — the energy does not increase as the system evolves, and ultimately the system settles at the point of lowest energy.

**The pain point**: $V$ must be **manually constructed**. There is no universal algorithm that can automatically find a suitable $V$ for an arbitrary system. This is like the heuristic function $h$ of Chapter 20 — admissibility requires that $h$ never overestimate, but how do you find such an $h$? There is no universal answer.

::: info Professor Pallas's Cat's Commentary
The construction of Lyapunov functions is an art, not a science. You guess a $V$, verify the conditions, and if it doesn't work, guess again. Behind this "guessing" lies the intuition, experience, and luck of the engineer. This is a fundamental gap in dynamical systems theory: stability can be verified, but the **proof** of stability (finding $V$) has no universal method. This gap bears a profound similarity to the undecidability of the halting problem — verification vs. construction, that theme again.
:::

If a reasoning system is a dynamical system, do we also have to manually guess a $V$? Or does the reasoning system's special structure — particularly its convergence behavior — allow us to **derive** $V$?

---

## 23.3 The Reasoning System as a Dynamical System

We formalize the reasoning process as a discrete-time dynamical system.

Let $x_t \in \mathcal{P}$ be the model's belief distribution (probability vector) over answers after the $t$-th step of reasoning. $\mathcal{P}$ is the belief space (e.g., $\Delta^{k-1}$, the simplex over $k$ possible answers).

A reasoning step is a mapping $F: \mathcal{P} \to \mathcal{P}$, taking the current belief $x_t$ as input and outputting the next-step belief $x_{t+1} = F(x_t)$. This mapping $F$ encodes the model's inference rules — which could be attention mechanisms, Bayesian updates, or any internal computation.

**The Yonglin formula** in this language is:

$$\lim_{t \to \infty} x_t = A, \quad \text{but} \quad A \neq A^*$$

where $A$ is the prior anchor point (the statistical bias of the training data), and $A^*$ is the distribution of the true answer. Convergence to $A$ means that $A$ is a fixed point of the dynamical system: $F(A) = A$.

**Key observation**: $A$ is a **global attractor** — starting from any initial belief $x_0$, iterating $F$ ultimately converges to $A$. This convergence is **structural**, not accidental. It comes from the constraints that training data impose on model parameters, encoded in the weights of $F$.

Returning to the MP game of Chapter 22: $F(A) = A$ is precisely the fixed-point condition $f(\varphi) = \varphi$. But Chapter 22 left an unresolved case — the existence of a fixed point does not mean that halting there is the "true QED." The Cauchy condition only guarantees that the sequence is eventually constant, not that the limit point is meaningful. The Yonglin formula makes this unresolved case concrete: the system does converge (Cauchy satisfied), but it converges to $A$, not $A^*$. **Convergence $\neq$ correctness** — this is the dynamical version of the first fate in the MP game.

---

## 23.4 Dynamical Construction: From Euler-Step Iteration to Energy Function

Now we do something bolder: rather than starting from a Lyapunov function to derive Yonglin, we do the reverse — **using the Yonglin convergence hypothesis + the information distance of KL distribution bias + Euler-step iterative updates, we construct the Lyapunov function**.

This is an art of dynamical systems: observe how the system evolves step by step, and "read off" the energy function from its behavior.

### Euler Steps: Discrete-Time Dynamics

**What is an Euler step?** It is the key that turns mathematics from static description into dynamical evolution.

A continuous-time dynamical system is described by a differential equation $\dot{x} = f(x)$. This equation says: the rate of change $\dot{x}$ of the state $x$ equals some function $f(x)$. For example, $\dot{x} = -x$ means the decay speed of $x$ is proportional to $x$ itself.

But differential equations are **continuous** — time $t$ is a real number, and changes occur in infinitesimal instants. Computers cannot handle "infinitesimal"; they can only handle discrete steps. Euler's method is the simplest discretization:

$$x_{t+1} = x_t + \Delta t \cdot f(x_t)$$

$\Delta t$ is the time step size, small but not zero. This formula says: starting from the current state $x_t$, compute the rate of change $f(x_t)$, multiply by the step size $\Delta t$, and obtain the next state $x_{t+1}$.

**Euler steps make mathematics "move."** Without them, a differential equation is merely a static relational formula; with them, we can simulate the system's evolution step by step and see how it develops from the initial state into the future.

Discrete-time systems are more direct: $x_{t+1} = F(x_t)$, where $F$ is the system's **evolution operator**. This can be seen as a special case of the Euler step (when $\Delta t = 1$ and $f$ is appropriately defined).

For a reasoning system, $F$ is the model's inference rule. We do not know the precise form of $F$, but we can observe its **behavior**: given input $x_t$, output $x_{t+1}$. This is the Euler step — the system advances one small step in time. Each step updates the belief; countless steps linked together form an inference trajectory.

::: info Professor Pallas's Cat's Science Corner: The Philosophy of Solving Ordinary Differential Equations
Ordinary differential equations (ODEs) describe the relationship between rates of change and states. Solving an ODE means finding the complete trajectory of the state over time. Analytic solutions (solutions expressed as formulas) are often hard to find, or may not even exist. Numerical solutions (such as Euler's method) abandon the "perfect formula" and accept the "approximate trajectory."

This abandonment is not a compromise but an **epistemological shift**: from pursuing "knowing the exact value at every moment" to "being able to simulate approximate values at arbitrary moments." In AI reasoning, we can rarely write down an analytic formula for the evolution of beliefs, but we can observe how the model updates step by step — this is the idea of numerical solutions.

The error of the Euler step is $O(\Delta t)$, not precise enough, but conceptually extremely important: **it decomposes continuous dynamics into discrete decisions**. At each step, the system decides the next step based on the current state; countless steps linked together form macroscopic behavior. The "chain of thought" of a reasoning system is, in essence, the iteration of Euler steps.
:::

### The Yonglin Hypothesis: Existence of an Attractor $A$

The core hypothesis of the Yonglin formula is: the system converges to the prior anchor point $A$. In dynamical language: $A$ is a fixed point of the system ($F(A) = A$) and is a **global attractor** — starting from any initial point, iterating $F$ converges to $A$.

This hypothesis is not a mathematical theorem but an empirical observation (though Chapter 12 provided theoretical support). We accept it as a **dynamical fact**.

### KL Divergence: The Information Distance of Distribution Bias

Now we introduce the KL divergence $D_{\text{KL}}(x \| A)$. In information theory, it measures the **extra number of bits** required to encode samples from distribution $x$ using distribution $A$. This is the "information distance" of $x$ relative to $A$.

Intuitively: if the current belief $x_t$ is close to the prior $A$, then $D_{\text{KL}}(x_t \| A)$ is small; if $x_t$ is far from $A$, then $D_{\text{KL}}(x_t \| A)$ is large.

### From Observation to Construction

Here comes the crucial step. We **observe** the system's evolution: starting from $x_0$, iterating $x_{t+1} = F(x_t)$, we observe the trajectory $\{x_0, x_1, x_2, \dots\}$ converging to $A$.

**If convergence occurs, then $D_{\text{KL}}(x_t \| A)$ must decrease over time.** Why?

Because convergence means $x_t \to A$, and the KL divergence is zero when $x = A$ and is a continuous function of $x$. Therefore $D_{\text{KL}}(x_t \| A) \to 0$. More strictly, convergence usually means that each step is closer to $A$: $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$.

This inequality is not something we prove, but something we **infer from observation**. We observe the system converging, and we infer that KL divergence decreases.

### Constructing the Lyapunov Function

Define $V(x) = D_{\text{KL}}(x \| A)$. Now verify the Lyapunov conditions:

1. **Non-negativity**: $V(x) \geq 0$, and $V(x) = 0$ if and only if $x = A$ (a property of KL divergence)
2. **Decreasing property**: $V(F(x)) \leq V(x)$, because we observe $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$

Thus $V$ is a Lyapunov function of the system.

**Note**: We did not guess $V$, nor did we derive $V$ from first principles. We **constructed $V$ from the observed system behavior**. The Yonglin convergence hypothesis provided the observation, KL divergence provided the natural distance metric, and Euler-step iteration exhibited the decreasing property.

::: info Professor Pallas's Cat's Commentary
This is the essence of the art of dynamical systems: not sitting in a chair guessing an energy function, but standing up and observing how the system moves, "listening" from its trajectory for the energy decreasing. The Yonglin hypothesis tells you where the system ultimately stops; KL divergence tells you how to measure "how far you still are from there"; Euler steps show how each step shortens that distance. Combine the three, and the energy function emerges naturally.
:::

**Comparison with the traditional approach**:
- Traditional: guess $V$ -> verify $\dot{V} \leq 0$
- Here: observe convergence -> define $V$ using KL divergence -> verify that $V$ decreases (guaranteed by convergence)

**Why does this solve the pain point?** Because we no longer need to manually guess $V$. $V$ is derived from the system's behavior. The cost is: we need the observation of Yonglin convergence as a premise.

**Concrete example**: Suppose the system's update rule is $x_{t+1} = (1-\alpha)x_t + \alpha A$ (linear interpolation, $\alpha \in (0,1)$). This is a simple Euler step: each step moves a small distance toward $A$. In this case, one can strictly prove $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$ (see Exercise 1). This example shows how the Euler-step update guarantees the decrease of KL divergence, so that $V(x) = D_{\text{KL}}(x \| A)$ is a Lyapunov function. The actual reasoning system's $F$ is more complex, but the Yonglin convergence observation suggests a similar decreasing property.

---

## 23.5 Reverse Derivation: How the Lyapunov Function Explains the Yonglin Formula

Now we look at the other direction: if we already have $V(x) = D_{\text{KL}}(x \| A)$ as a Lyapunov function (however obtained), what can it tell us about the Yonglin formula?

The Lyapunov stability theorem says: if $V$ decreases, the system converges to the minimum point of $V$. The minimum point of $V$ is $A$ (because KL divergence is zero at $x = A$ and that is the unique minimum point). Hence the system converges to $A$.

**The Yonglin inference formula** $\lim_{t \to \infty} x_t = A$ is the statement of this conclusion.

But the Yonglin formula has a second part: $A \neq A^*$. How can this be derived from the Lyapunov function?

$A \neq A^*$ means that **the system's attractor is not the true answer**. In the language of dynamical systems, this is equivalent to: the true answer $A^*$ is not an equilibrium point of the system, or even if it is an equilibrium point, it is not attractive (it may be unstable).

From the perspective of $V$, the minimum point of $V$ is $A$, not $A^*$. So the construction of $V$ itself encodes the system's "bias" — it regards $A$ as the state of lowest "energy," not $A^*$. This bias comes from the training data, encoded in $F$, and ultimately reflected in the definition of $V$.

**Key insight**: The $A$ in $V(x) = D_{\text{KL}}(x \| A)$ is precisely the statistical bias of the training data. So the Lyapunov function $V$ is not a neutral metric; it **embeds the system's prior**. The decrease of $V$ is the system's regression toward the prior anchor point.

::: info Information-Theoretic Perspective: KL Divergence as a Measure of Distribution Bias
The KL divergence $D_{\text{KL}}(x \| A)$ has a clear meaning in information theory: the **extra number of bits** required to encode samples from distribution $x$ using distribution $A$. This "extra" is relative to the optimal case of using $x$ itself for encoding.

When $x$ is close to $A$, $D_{\text{KL}}(x \| A)$ is small — encoding the current belief $x$ using the prior $A$ requires almost no extra cost. When $x$ is far from $A$, $D_{\text{KL}}(x \| A)$ is large — the current belief differs greatly from the prior, requiring more bits to describe this deviation.

Thus $V(x) = D_{\text{KL}}(x \| A)$ measures the **"information distance" of the current belief relative to the prior**. The system converging to $A$ is the reduction of information distance, ultimately reaching zero — belief perfectly consistent with the prior, requiring no extra information to describe the deviation.

This interpretation transforms the stability problem of reasoning systems into an **information efficiency** problem: the system is optimizing information encoding, evolving toward the most economical state (the one requiring the fewest extra bits). That state happens to be the prior anchor point $A$, not the true answer $A^*$. The system's "bias," in the language of information theory, is the **presupposition of the encoding scheme**.
:::

---

## 23.6 Combined: The Yonglin-Lyapunov Correspondence

Now we put both directions together.

**Yonglin -> Lyapunov**: Observing convergence to $A$, define $V(x) = D_{\text{KL}}(x \| A)$, verify that $V$ decreases. In this way, the Lyapunov function **is derived from convergence behavior**, no longer requiring manual guessing.

**Lyapunov -> Yonglin**: Given $V(x) = D_{\text{KL}}(x \| A)$, the Lyapunov stability theorem yields convergence to $A$. If $A \neq A^*$, then the system's limit is not the true answer.

These two directions form a closed loop: convergence behavior defines the energy function, and the energy function guarantees the convergence behavior. The **core parameter** of this closed loop is the prior anchor point $A$. $A$ is the statistical bias of the training data, the "world model" that the model absorbs from data.

::: info Professor Pallas's Cat's Commentary
The structure of this combination is the same pattern as learning as inverse inference in Chapter 21: from data (observed convergence), reverse-engineer the law (Lyapunov function). Another inverse problem. But here there is an extra layer: the law ($V$) itself predicts the observation (convergence). This is a self-referential structure — the system's behavior defines its energy, and the energy explains its behavior. This self-reference is not a paradox but a harmony: observation and theory mutually lock each other in place.
:::

---

## 23.7 Connection to Gödelian Incompleteness

Gödel's theorem of Chapter 15 says: any sufficiently strong formal system has true propositions that it cannot prove. The core of the theorem's proof is **self-reference** — constructing a proposition that talks about its own provability.

Within the Yonglin-Lyapunov combination, there is also a self-referential structure: the system's convergence behavior defines its energy function, and the energy function describes its convergence behavior. This self-reference is not the self-reference of logical propositions, but **dynamical self-reference**.

More profoundly, Gödel's theorem reveals the rupture between the **internal perspective** and the **external perspective** of a formal system: the system cannot internally prove certain true propositions about itself. The Yonglin formula reveals the rupture between the **object level** and the **meta-level** of a reasoning system: the system can generate inference chains (object level), but cannot verify the correctness of inference chains (meta-level).

The Lyapunov function, in this analogy, is a tool of the **external perspective**: it describes the system's stability from outside. But through the Yonglin-Lyapunov combination, we **internalize** this external tool — deriving it from the system's own convergence behavior. This is somewhat like attempting to construct, inside the system, a proof about its own stability. Does this attempt encounter Gödelian limitations?

---

## 23.8 Significance: Interpretability and Stability Guarantees

What is the practical significance of this combination?

**Significance One: Interpretability**. The Lyapunov function $V(x) = D_{\text{KL}}(x \| A)$ provides a clear explanation: the reasoning system is "reducing the divergence between its own beliefs and the prior anchor point." Every step of reasoning brings the belief closer to the statistical bias implicit in the training data. This explanation is more transparent than the internal computations of a black-box neural network.

**Significance Two: Stability guarantee**. Once we have $V$ and have verified that $V$ decreases, we have a rigorous guarantee of convergence. This is important for safety-critical applications: knowing that the system will eventually stabilize somewhere (even if that somewhere is not the correct answer) is better than not knowing where the system might drift.

**Significance Three: No need for manual design**. Traditional Lyapunov methods require the intuition and trial-and-error of engineers. Here, $V$ is directly derived from observed data (statistics of training data) and observed behavior (convergence). This lowers the barrier to application.

**But the cost**: This $V$ depends on the hypothesis of Yonglin convergence. If convergence does not hold (e.g., the system is chaotic, or has multiple attractors), the construction of $V$ fails. The Yonglin formula itself is an empirical observation, not a mathematical theorem (though it has theoretical support). So this combination is **conditional**: if the system converges to the prior anchor point, then we can construct $V$ in this way.

---

## 23.9 Unresolved

**Is Yonglin convergence universal?** The Yonglin formula has been observed experimentally, but what is the scope of its theoretical validity? Do all reasoning systems based on statistical learning satisfy this convergence? Or is it applicable only to specific architectures (such as Transformers)? This question requires more rigorous mathematical characterization.

**The multi-attractor case**: If the system has multiple attractors (multiple prior anchor points, corresponding to different tasks or contexts), how should the Lyapunov function be defined? $V$ might become a complex non-convex function with multiple local minima. This reflects the system's **multi-stability** — reasoning may converge to different conclusions depending on initial conditions. This is more similar to human cognition: the same question has different interpretations in different contexts.

**Uniqueness of the Lyapunov function**: Given convergence behavior, is $V$ unique? Obviously not. $V$ can be monotonically transformed. But $D_{\text{KL}}(x \| A)$ has a special status in information theory — it is a measure of "surprise" of $x$ relative to $A$. Is there a more fundamental reason for choosing this $V$?

**Connection to learning theory**: Chapter 21's learning as inverse inference used the MDL principle to interpret generalization as compression. $V(x) = D_{\text{KL}}(x \| A)$ can also be understood as a kind of "description length": the extra number of bits required to encode the current belief $x$ using the prior $A$. The decrease of $V$ is the shortening of description length — the system is "compressing" its own beliefs, moving toward a more economical representation. This perspective unifies stability, compression, and generalization.

---

## Exercises

**★ Warm-up**

1. Does the Lyapunov function $V(x) = x^2$ satisfy the conditions for the system $\dot{x} = -x$? Compute $\dot{V}$ and determine whether the system is stable.
2. In the Yonglin formula, if the training data is perfectly balanced (50% positive, 50% negative examples), what is the prior anchor point $A$? What form does $V(x) = D_{\text{KL}}(x \| A)$ take in this case?

---

**★★ Derivation**

1. **Discrete-time Lyapunov**: For a discrete system $x_{t+1} = F(x_t)$, the Lyapunov condition is $V(F(x)) \leq V(x)$. Assume $V(x) = D_{\text{KL}}(x \| A)$ and $F$ is the following update: $x_{t+1} = (1-\alpha)x_t + \alpha A$, where $0 < \alpha < 1$. Prove that $V(x_{t+1}) \leq V(x_t)$.
   
2. **$V$ for multiple attractors**: Suppose the system has two attractors $A_1$ and $A_2$, with convergence depending on initial conditions. Design a function $V$ such that $V$ is zero at both attractors, positive elsewhere, and decreases along system trajectories. Hint: consider $V(x) = \min(D_{\text{KL}}(x \| A_1), D_{\text{KL}}(x \| A_2))$. What is the problem with this $V$? (Non-differentiable, hard to verify decrease)

---

**★★★ Challenge**

In the proof of Gödel's theorem, the key step is constructing the self-referential proposition $G$: "$G$ is unprovable." In dynamical systems, is there an analogous self-referential construction? Consider a function $F$ whose definition depends on its own attractor. For example: define $F$ such that its attractor is the solution to some equation, and that equation in turn involves $F$ itself. Could such self-reference lead to a Gödel-like incompleteness — certain properties being undeterminable from within the system?

The answer to this question may point toward **incompleteness in dynamical systems**: the stability of certain systems cannot be determined from their own dynamics, requiring an external perspective. This conjecture extends Gödelian incompleteness from the logical domain into the dynamical domain.

---


> The Yonglin-Lyapunov combination tells us: the system's limit is encoded in its energy function. And the energy function can be read off from the limit. This cycle is not a logical paradox but a dynamical harmony — the observer and the observed system mutually define each other within this cycle. This definition ultimately settles at the prior anchor point. Not because we want to stop there, but because the system's energy is lowest there.

---

**References**

- \[Zixi Li, 2025b\] — Yonglin Formula, a theoretical proof of inference incompleteness
- Lyapunov, A. M. (1892) — The General Problem of the Stability of Motion
- Cover, T. M., & Thomas, J. A. (2006) — Elements of Information Theory (KL divergence)
- Chapter 15 — Consistency and Completeness (Gödelian incompleteness)
- Chapter 21 — Learning as Inverse Inference (MDL principle)
- Chapter 25 — The Unification of Boundaries (the book's conclusion and Impossible Triangle)

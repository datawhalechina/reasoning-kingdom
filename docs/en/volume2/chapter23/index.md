# Chapter 23: The Yonglin–Lyapunov Correspondence — Stability and Convergence Boundaries of Reasoning Systems

> What is the biggest pain point of the Lyapunov function? Right: someone still has to hand-pick an energy function. It would be much better if we could derive the Lyapunov function of a reasoning system from Yonglin's convergence assumption.

---

At the end of Chapter 22, we stood at the boundary of self-reference and emergence and saw the strange moment when a reasoning system starts reasoning about itself. That boundary, mathematically, appears as fixed points, diagonalization, and undecidability.

But the boundary is not only logical. The boundary is also **dynamical** — how a system evolves over time, whether it is stable, and where it converges. Yonglin's formula gives a concrete convergence pattern: the reasoning chain eventually returns to the prior anchor. In the language of dynamical systems, this convergence is an **attractor**.

This chapter does one thing: translate the convergence of Yonglin's formula into the language of Lyapunov stability. Then, the other way: derive the Lyapunov function from the convergence assumption, instead of postulating it by hand.

**Goal**: use the Yonglin convergence assumption + the information distance of the KL distribution bias + the Euler-step update iteration, to derive the Lyapunov function of a reasoning system. The Lyapunov function, traditionally, has to be constructed by hand, depending on human priors. But by observing the system's dynamical behavior (Yonglin convergence), we derive the energy function backwards — this is deriving structure from behavior, yet another performance of the **inverse problem**, and the embodiment of the art of dynamical systems.

---

## 23.0 Prologue: A Dynamical Story About a Stack

To understand the Lyapunov function, let's play a small game first.

Imagine a stack data structure. It has two pointers:

- The $\beta$ pointer: always points to the bottom of the stack, fixed and immovable
- The $\alpha$ pointer: points to the top of the stack; it can move up and down

The stack's operations are simple:

- **`push(x)`**: push element `x` onto the top of the stack; the $\alpha$ pointer moves up one slot
- **`pop()`**: pop the top element; the $\alpha$ pointer moves down one slot

**But there is one important restriction**: a negative stack is illegal. That is, the $\alpha$ pointer must always be above (or equal to) the $\beta$ pointer. When $\alpha = \beta$, the stack is empty; attempting a `pop()` on an empty stack is invalid — the $\alpha$ pointer does not move, or the program errors out. In the dynamical story, we assume invalid pops are ignored.

Students of informatics olympiad (OI) know this structure all too well. But today Professor Pallas is not asking an algorithmic question — he asks a **dynamical question**:

> Given an operation sequence of `push` and `pop` that is very, very long — longer than you can imagine — will the position of the $\alpha$ pointer (its height above the stack bottom) converge? Will it stop at a particular height, no matter how the operation sequence is arranged?

**Wait**, we also have the "negative stack is illegal" restriction. Under this restriction, things get more interesting.

::: info Professor Pallas Asks
Now the $\alpha$ pointer cannot go below the $\beta$ pointer. When the stack is empty ($\alpha = \beta$), an attempted `pop` is ignored. What do you think — under this constraint, where will the $\alpha$ pointer converge? No matter how complex and how long the operation sequence, will the $\alpha$ pointer eventually settle at some height? Or will it oscillate within some range?

Give yourself three seconds to think.
:::

**What your intuition may tell you**: since `pop` is invalid on an empty stack, the stack might become empty more easily — because an empty stack cannot be popped further, while `push` makes it non-empty. But conversely, if there are many pushes, the stack grows tall.

**The key observation**: the stack height $h$ (the distance between $\alpha$ and $\beta$) now has a **hard lower bound** of $0$. This lower bound changes the system's dynamics.

### Stack Height as Energy

Define the stack height $h$ = the distance between the $\alpha$ pointer and the $\beta$ pointer (measured in number of elements). Each `push` increases $h$ by 1; each `pop` decreases $h$ by 1.

Now define a function $V(h) = h^2$. What is special about this function?

**Observations**:

- $V(h) \geq 0$, and $V(h) = 0$ if and only if $h = 0$ (empty stack)
- If the operation sequence is random (`push` and `pop` equally likely), the expected change of $h$ is zero — a random walk

But Professor Pallas is not asking about expectation, but about **deterministic** behavior: if we know each step is `push` or `pop` (not random), how does $V(h)$ change?

### Something Magical Happens

Consider a continuous-time approximation: assume the operations happen very fast, and treat the discrete operations as a continuous flow. Define the rate of change of the stack height $\dot{h} = u(t)$, where $u(t) = +1$ means `push` and $u(t) = -1$ means `pop`.

Now compute the rate of change of $V(h) = h^2$:

$$\dot{V} = \frac{d}{dt}(h^2) = 2h\dot{h} = 2h \cdot u(t)$$

This looks irregular — $u(t)$ can be positive or negative, so $\dot{V}$ can be positive or negative too. But if we integrate $\dot{V}$ over a long time, what happens?

**If the operation sequence is eventually balanced** — `push` and `pop` roughly equal in number — then $h$ oscillates around some average value. $V(h) = h^2$ does not decrease monotonically, but its **long-term average** may converge.

### A Smarter Energy Function

Switch to another definition: $V(h) = |h - h_0|$, where $h_0$ is some target height. If the operation sequence is **balanced** (ultimately equal numbers of `push` and `pop`), then $h$ converges back to the initial height $h_0$ (assuming the stack is not initially empty). Does $V(h)$ decrease over time then? Not necessarily — $h$ may oscillate around $h_0$.

But if we consider the **variance** $V(h) = (h - h_0)^2$, and the operation sequence is **random**, then $\mathbb{E}[V(h)]$ may grow over time (the variance of a random walk grows linearly).

### The Magic of "No Negative Stack": Convergence to the Empty Stack

Now add the no-negative-stack restriction. The $\alpha$ pointer cannot go below the $\beta$ pointer, and a `pop` on an empty stack is ignored.

**Professor Pallas's answer**: under this constraint, if the operation sequence is long enough and contains enough pops, the $\alpha$ pointer will eventually converge to the **empty stack** state — $h = 0$.

Why?

Consider the simplest energy function $V(h) = h$. When $h > 0$:

- A `push`: $h$ increases by 1, $V$ increases
- A `pop`: $h$ decreases by 1, $V$ decreases

But the key point: when $h = 0$, a `pop` is ignored; $h$ stays 0, $V$ stays 0. So $h = 0$ is an **absorbing state** — once entered, you cannot get out (unless there is a `push`).

If the operation sequence is infinitely long and contains enough pops (not necessarily more than pushes — as long as pops occur), the system has a probability of entering the state $h = 0$. Once entered, subsequent pops are invalid; only a `push` can pull it out. But if the sequence is **non-deterministic** (say, random), then in the long run, the system visits the state $h = 0$ frequently.

More strictly: define $V(h) = h$. Then the change in $V$ is $\Delta V = \Delta h$:

- `push`: $\Delta V = +1$
- `pop` (when $h > 0$): $\Delta V = -1$
- `pop` (when $h = 0$): $\Delta V = 0$

$V$ is not monotonically decreasing, but $V$ has a lower bound of $0$, and the system repeatedly visits the state $V = 0$. On an infinite time scale, the fraction of time the system spends in the state $V = 0$ may be very high.

**But this is not Lyapunov stability**, because $V$ is not monotonically decreasing. Lyapunov requires $\dot{V} \leq 0$ for all time; this does not hold here.

### The Key Insight

What does the stack story tell us?

1. **Dynamical system**: the stack height $h$ is a dynamical system, whose evolution is driven by the operation sequence $u(t)$.
2. **Energy function**: $V(h)$ is a measure of the system's "energy". We want to use $V$ to judge whether the system converges.
3. **Convergence condition**: if there exists $V$ such that $\dot{V} \leq 0$ (energy non-increasing), then the system is stable.
4. **The problem**: for the stack, neither the simple $V(h) = h$ nor $V(h) = h^2$ satisfies $\dot{V} \leq 0$, because $u(t)$ can be positive or negative.
5. **The effect of the lower bound**: the hard constraint $h \geq 0$ gives the system an absorbing state $h = 0$, but the existence of an absorbing state does not guarantee that the stability theorem holds.

So we need a cleverer $V$, or constraints on the operation sequence $u(t)$.

### The Lesson of the Stack Model

The stack is the simplest dynamical system, and we still need to think hard to find a suitable $V$. For more complex systems (such as neural networks, reasoning systems), finding $V$ is even harder.

This is the **pain point** of the Lyapunov function: you guess a $V$, verify the conditions, and if it fails, guess again. Guess right, and the system's stability is proved; fail to guess, and the proof is stuck.

But wait — if we observe the system's behavior and find that it does converge, can we reverse-infer $V$ from the convergence behavior? This is the core idea of the Yonglin-Lyapunov combination.

### The Stack vs. the MP Game

In the MP game of Chapter 22, the proof sequence $\{p_1, p_2, \dots\}$ is an orbit, $p_{n+1} = f(p_n)$, and an orbit has three fates: finite stopping, finite cycle, infinite extension.

The stack story is the concrete version of the same thing:

- The height $h$ of the $\alpha$ pointer is the state
- `push`/`pop` is the evolution operator $f$
- The empty stack $h = 0$ is the absorbing state — the **fixed point** $f(\varphi) = \varphi$ of the MP game

The question left by the MP game: the fixed point exists, but starting from an arbitrary $p_0$, which fixed point attracts the system? The stack story gives the intuition: a system with a hard constraint ($h \geq 0$) has its absorbing state at the constraint boundary. The "hard constraint" of a reasoning system is the training data — it defines the prior anchor $A$, which is the Yonglin limit.

---

## 23.0.5 Running the Stack Story in Code

The stack is small enough to simulate. The program below runs long operation sequences under the no-negative-stack rule, measures where the alpha pointer goes, and verifies the pain point with real numbers: neither `V(h)=h` nor `V(h)=h^2` decreases monotonically, and what the hard lower bound really buys is an absorbing state at `h=0`.

```python
#!/usr/bin/env python3
"""
stack_story.py -- the Stack Story of Chapter 23: searching for a Lyapunov
function, and finding instead why the hard constraint matters.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 23, Section 23.0
"Prologue: A Dynamical Story About a Stack."
A stack with two pointers: beta at the bottom (fixed), alpha at the top
(moves up on push, down on pop). One restriction: a negative stack is
illegal -- a pop on an empty stack is ignored. Professor Pallas asks a
dynamical question: where does the alpha pointer converge?
This program runs the story:
  * it simulates operation sequences and watches the height h;
  * it verifies, with real numbers, that neither V(h)=h nor V(h)=h^2
    decreases monotonically -- the pain point of the Lyapunov function:
    guess a V, verify the conditions, guess again;
  * it shows what the hard lower bound actually buys: an absorbing state
    at h = 0 that the system keeps falling into;
  * it draws the correspondence the book points to: the empty stack is
    the fixed point of the MP game's orbit (Chapter 22), sitting at the
    boundary of the constraint.
Run:  python3 stack_story.py
"""
import random


def step(height, op):
    """One operation on the stack. A pop on an empty stack is ignored."""
    if op == "push":
        return height + 1
    return height - 1 if height > 0 else 0


def run_sequence(ops):
    """Return the list of heights after each operation."""
    heights = [0]
    h = 0
    for op in ops:
        h = step(h, op)
        heights.append(h)
    return heights


def delta_v_counts(heights, energy):
    """Count steps where the energy V changes by +1, -1, or 0."""
    up = down = flat = 0
    for before, after in zip(heights, heights[1:]):
        change = energy(after) - energy(before)
        if change > 0:
            up += 1
        elif change < 0:
            down += 1
        else:
            flat += 1
    return up, down, flat


def demo():
    print("=" * 70)
    print(" THE STACK STORY -- a dynamical question about a stack")
    print(" The Reasoning Kingdom, Volume 2, Chapter 23 (Sec. 23.0)")
    print("=" * 70)

    # A short trace, so the rule is visible.
    ops = ["push", "push", "pop", "pop", "pop", "push"]
    heights = run_sequence(ops)
    print("\nA short trace (pop on empty is ignored):")
    for i, (op, h) in enumerate(zip(ops, heights[1:]), start=1):
        note = "  (ignored: stack empty)" if op == "pop" and h == 0 else ""
        print(f"  {i}. {op:<5} height={h}{note}")

    # Long sequences, deterministic seeds.
    rng = random.Random(7)
    n = 20000
    balanced = run_sequence(rng.choice(("push", "pop")) for _ in range(n))
    pop_heavy = run_sequence(
        "pop" if rng.random() < 0.8 else "push" for _ in range(n)
    )
    for name, seq in (("balanced (50% push / 50% pop)", balanced),
                      ("pop-heavy (80% pop)", pop_heavy)):
        frac_zero = sum(1 for h in seq if h == 0) / (n + 1)
        print(f"\n{n} steps, {name}:")
        print(f"  final height: {seq[-1]}")
        print(f"  highest height reached: {max(seq)}")
        print(f"  fraction of time at h=0: {frac_zero:.3f}")
    print("  the balanced walk wanders with no monotone energy; the")
    print("  pop-heavy walk keeps collapsing into the absorbing state")

    # The Lyapunov check, for V(h) = h, on the long balanced sequence.
    up, down, flat = delta_v_counts(balanced, lambda h: h)
    print("\nLyapunov check for V(h) = h, on the balanced sequence:")
    print(f"  steps with dV = +1: {up}")
    print(f"  steps with dV = -1: {down}")
    print(f"  steps with dV =  0: {flat}")
    print("  dV is sometimes positive, so V does not decrease monotonically")
    print("  -- this is NOT Lyapunov stability. The pain point: you guess")
    print("  a V, verify the conditions, and if it fails, guess again.")

    # Same for V(h) = h^2, as the book's continuous argument predicts.
    up2, down2, flat2 = delta_v_counts(balanced, lambda h: h * h)
    print("\nThe same check for V(h) = h^2:")
    print(f"  dV > 0 on {up2} steps, dV < 0 on {down2} steps")
    print("  the continuous version is V-dot = 2h * u(t): positive on some")
    print("  pushes, negative on some pops. No monotone energy here either.")

    print("\nWhat the hard lower bound actually buys:")
    print("  h = 0 is absorbing; the system keeps falling into it. The empty")
    print("  stack is the fixed point f(phi) = phi of the MP game (Ch. 22),")
    print("  sitting exactly at the boundary of the constraint. For a")
    print("  reasoning system, that hard constraint is the training data --")
    print("  the prior anchor A that the orbit settles around.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
```

---

## 23.2 The Pain Point of the Lyapunov Function

**Definition (Lyapunov function)**: for a dynamical system $\dot{x} = f(x)$, if there exists a continuously differentiable function $V(x)$ satisfying

1. $V(x) \geq 0$, and $V(x) = 0$ if and only if $x = x^*$ (the equilibrium point)
2. $\dot{V}(x) = \frac{dV}{dt} \leq 0$ for all $x$

then $x^*$ is **stable**. If $\dot{V}(x) < 0$ (strictly negative except at $x^*$), then $x^*$ is **asymptotically stable**.

$V$ is called a Lyapunov function; intuitively, it is the system's "energy" — the energy does not increase as the system evolves, and it finally rests at the point of lowest energy.

**The pain point**: $V$ must be **constructed by hand**. There is no general algorithm that automatically finds a suitable $V$ for an arbitrary system. This is like the heuristic function $h$ of Chapter 20 — admissibility requires $h$ to never overestimate, but how does one find such an $h$? There is no general answer.

::: info Professor Pallas Says
Constructing a Lyapunov function is an art, not a science. You guess a $V$, verify the conditions, and if it fails, guess again. Behind this "guessing" lie the engineer's intuition, experience, and luck. This is a fundamental gap in dynamical systems theory: stability can be verified, but the **proof** of stability (finding $V$) has no general method. This gap has a deep similarity to the undecidability of the halting problem — verification vs. construction, that theme again.
:::

If a reasoning system is a dynamical system, must we also hand-guess a $V$? Or does the special structure of a reasoning system — in particular its convergence behavior — allow us to **derive** $V$?

---

## 23.3 A Reasoning System as a Dynamical System

Formalize the reasoning process as a discrete-time dynamical system.

Let $x_t \in \mathcal{P}$ be the model's belief distribution over the answer (a probability vector) after $t$ reasoning steps. $\mathcal{P}$ is the belief space (for example $\Delta^{k-1}$, the simplex over $k$ possible answers).

A reasoning step is a map $F: \mathcal{P} \to \mathcal{P}$: it takes the current belief $x_t$ and outputs the next belief $x_{t+1} = F(x_t)$. This map $F$ encodes the model's inference rules — it may be the attention mechanism, Bayesian updates, or any internal computation.

**Yonglin's formula**, in this language, is:

$$\lim_{t \to \infty} x_t = A, \quad \text{but} \quad A \neq A^*$$

where $A$ is the prior anchor (the statistical bias of the training data) and $A^*$ is the distribution of the true answer. Converging to $A$ means that $A$ is a fixed point of the dynamical system: $F(A) = A$.

**The key observation**: $A$ is a **global attractor** — starting from any initial belief $x_0$, iterating $F$ eventually converges to $A$. This convergence is **structural**, not accidental. It comes from the constraint that the training data imposes on the model's parameters, encoded in the weights of $F$.

Back to the MP game of Chapter 22: $F(A) = A$ is exactly the fixed-point condition $f(\varphi) = \varphi$. But Chapter 22 left an open case — a fixed point existing does not mean that stopping there is a "true QED". The Cauchy condition only guarantees that the sequence is eventually constant, not that the limit point is meaningful. Yonglin's formula makes this case concrete: the system does converge (Cauchy is satisfied), but what it converges to is $A$, not $A^*$. **Convergence $\neq$ correctness** — this is the dynamical version of the first fate of the MP game.

---

## 23.4 Dynamical Construction: From Euler-Step Iteration to the Energy Function

Now do something bolder: instead of starting from a Lyapunov function to derive Yonglin, go the other way — **use the Yonglin convergence assumption + the information distance of the KL distribution bias + the Euler-step update iteration, to construct the Lyapunov function**.

This is the art of dynamical systems: observe how the system evolves step by step, and "read" the energy function out of its behavior.

### The Euler Step: Discrete-Time Dynamics

**What is an Euler step?** It is the key that takes mathematics from static description to dynamical evolution.

A continuous-time dynamical system is described by a differential equation $\dot{x} = f(x)$. This equation says: the rate of change of the state $x$, $\dot{x}$, equals some function $f(x)$. For instance, $\dot{x} = -x$ says that the decay speed of $x$ is proportional to $x$ itself.

But differential equations are **continuous** — time $t$ is a real number, and changes happen in infinitesimal moments. A computer cannot handle "infinitesimals"; it can only handle discrete steps. Euler's method is the simplest discretization:

$$x_{t+1} = x_t + \Delta t \cdot f(x_t)$$

$\Delta t$ is the time step, small but not zero. This formula says: starting from the current state $x_t$, compute the rate of change $f(x_t)$, multiply by the step size $\Delta t$, and you obtain the next state $x_{t+1}$.

**The Euler step makes mathematics "move".** Without it, a differential equation is merely a static relation; with it, we can simulate the system's evolution step by step and see how it develops from an initial state into the future.

A discrete-time system is more direct: $x_{t+1} = F(x_t)$, where $F$ is the system's **evolution operator**. This can be viewed as a special case of the Euler step (when $\Delta t = 1$ and $f$ is suitably defined).

For a reasoning system, $F$ is the model's inference rule. We do not know the exact form of $F$, but we can observe its **behavior**: given an input $x_t$, it outputs $x_{t+1}$. This is the Euler step — the system advances a small step in time. Each step, the belief is updated; countless steps connected together form a reasoning trajectory.

::: info Professor Pallas Explains: The Philosophy of Solving Ordinary Differential Equations
An ordinary differential equation (ODE) describes the relation between the rate of change and the state. Solving an ODE means finding the complete trajectory of the state over time. Analytical solutions (solutions written out as formulas) are often very hard to find, or even nonexistent. Numerical solutions (such as Euler's method) give up the "perfect formula" and accept an "approximate trajectory".

This giving-up is not a compromise but an **epistemological shift**: from pursuing "the exact value at every moment" to "being able to simulate an approximate value at any moment". In AI reasoning, we can rarely write down an analytical formula for belief evolution, but we can observe the model's step-by-step updates — that is the idea of a numerical solution.

The error of the Euler step is $O(\Delta t)$ — not precise, but conceptually extremely important: **it breaks continuous dynamics into discrete decisions**. Each step, the system decides the next step based on its current state; countless steps connected together form macroscopic behavior. The "chain of thought" of a reasoning system, in essence, is an iteration of Euler steps.
:::

### The Yonglin Assumption: An Attractor $A$ Exists

The core assumption of Yonglin's formula is: the system converges to the prior anchor $A$. In the language of dynamics: $A$ is a fixed point of the system ($F(A) = A$), and it is a **global attractor** — starting from any initial point, iterating $F$ converges to $A$.

This assumption is not a mathematical theorem, but an empirical observation (though Chapter 12 gives theoretical support). We accept it as a **dynamical fact**.

### KL Divergence: The Information Distance of a Distribution Bias

Now introduce the KL divergence $D_{\text{KL}}(x \| A)$. In information theory, it measures the **extra number of bits** needed to encode samples from distribution $x$ using distribution $A$. This is the "information distance" of $x$ relative to $A$.

Intuitively: if the current belief $x_t$ is very close to the prior $A$, then $D_{\text{KL}}(x_t \| A)$ is small; if $x_t$ is far from $A$, then $D_{\text{KL}}(x_t \| A)$ is large.

### From Observation to Construction

Here comes the key step. We **observe** the system's evolution: starting from $x_0$, iterate $x_{t+1} = F(x_t)$, and observe that the trajectory $\{x_0, x_1, x_2, \dots\}$ converges to $A$.

**If convergence happens, then $D_{\text{KL}}(x_t \| A)$ must decrease over time.** Why?

Because convergence means $x_t \to A$, and the KL divergence is zero at $x = A$ and is a continuous function of $x$. So $D_{\text{KL}}(x_t \| A) \to 0$. More strictly, convergence usually means each step gets closer to $A$: $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$.

This inequality is not something we prove; it is something we **infer from observation**. We observe the system converging, and we infer that the KL divergence decreases.

### Constructing the Lyapunov Function

Define $V(x) = D_{\text{KL}}(x \| A)$. Now verify the Lyapunov conditions:

1. **Non-negativity**: $V(x) \geq 0$, and $V(x) = 0$ if and only if $x = A$ (a property of KL divergence)
2. **Decreasing**: $V(F(x)) \leq V(x)$, because we observe $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$

So $V$ is a Lyapunov function of the system.

**Note**: we did not guess $V$, and we did not derive $V$ from first principles. We **constructed $V$ from the observed behavior of the system**. The Yonglin convergence assumption supplies the observation, KL divergence supplies the natural distance measure, and the Euler-step iteration exhibits the decreasing property.

::: info Professor Pallas Says
This is the essence of the art of dynamical systems: not sitting in a chair guessing an energy function, but standing up and observing how the system moves, "listening" to the energy decreasing out of its trajectory. The Yonglin assumption tells you where the system finally stops; KL divergence tells you how to measure "how far it is from that place"; the Euler step shows how each step shortens that distance. With all three combined, the energy function emerges naturally.
:::

**Comparison with the traditional method**:

- Traditional: guess $V$ -> verify $\dot{V} \leq 0$
- Here: observe convergence -> define $V$ with KL divergence -> verify $V$ decreases (guaranteed by convergence)

**Why does this solve the pain point?** Because guessing $V$ by hand is no longer needed. $V$ is derived from the system's behavior. The cost is: the observation of Yonglin convergence is required as a premise.

**A concrete example**: suppose the system's update rule is $x_{t+1} = (1-\alpha)x_t + \alpha A$ (linear interpolation, $\alpha \in (0,1)$). This is a simple Euler step: each step moves a small distance toward $A$. For this, one can prove rigorously that $D_{\text{KL}}(x_{t+1} \| A) \leq D_{\text{KL}}(x_t \| A)$ (see Exercise 1). This example shows how the Euler-step update guarantees a decreasing KL divergence, so that $V(x) = D_{\text{KL}}(x \| A)$ is a Lyapunov function. The $F$ of a real reasoning system is more complex, but the observed Yonglin convergence suggests a similar decreasing property.

---

## 23.5 The Reverse Direction: How the Lyapunov Function Explains Yonglin's Formula

Now look at the other direction: if we already have $V(x) = D_{\text{KL}}(x \| A)$ as a Lyapunov function (however we obtained it), what can it tell us about Yonglin's formula?

The Lyapunov stability theorem says: if $V$ decreases, the system converges to the minimum point of $V$. The minimum point of $V$ is $A$ (because KL divergence is zero at $x = A$, and that is its unique minimum). So the system converges to $A$.

**Yonglin's inference formula** $\lim_{t \to \infty} x_t = A$ is exactly the statement of this conclusion.

But Yonglin's formula has a second part: $A \neq A^*$. How can this be derived from the Lyapunov function?

$A \neq A^*$ means **the system's attractor is not the true answer**. In the language of dynamical systems, this is equivalent to: the true answer $A^*$ is not an equilibrium point of the system, or even if it is an equilibrium point, it is not attracting (it may be unstable).

From the viewpoint of $V$, the minimum point of $V$ is $A$, not $A^*$. So the construction of $V$ itself encodes the system's "bias" — it regards $A$, not $A^*$, as the state of lowest "energy". This bias comes from the training data, is encoded in $F$, and is finally reflected in the definition of $V$.

**The key insight**: the $A$ in $V(x) = D_{\text{KL}}(x \| A)$ is exactly the statistical bias of the training data. So the Lyapunov function $V$ is not a neutral measure — it **builds the system's prior into itself**. The decrease of $V$ is the system's return to the prior anchor.

::: info An Information-Theoretic View: KL Divergence as a Measure of Distribution Bias
The KL divergence $D_{\text{KL}}(x \| A)$ has a clear meaning in information theory: the **extra number of bits** needed to encode samples from distribution $x$ using distribution $A$. This "extra" is relative to the optimal case of encoding with $x$ itself.

When $x$ is close to $A$, $D_{\text{KL}}(x \| A)$ is small — encoding the current belief $x$ with the prior $A$ costs almost no extra. When $x$ is far from $A$, $D_{\text{KL}}(x \| A)$ is large — the current belief differs greatly from the prior, and more bits are needed to describe this difference.

So $V(x) = D_{\text{KL}}(x \| A)$ measures the **"information distance" of the current belief relative to the prior**. The system converging to $A$ is the decrease of this information distance, finally reaching zero — the belief fully agrees with the prior, with no extra information needed to describe the deviation.

This interpretation turns the stability problem of a reasoning system into an **information-efficiency** problem: the system is optimizing information encoding, evolving toward the most economical state (the one needing the fewest extra bits). That state happens to be the prior anchor $A$, not the true answer $A^*$. The system's "bias", in the language of information theory, is the **preset of the encoding scheme**.
:::

---

## 23.6 The Combination: The Yonglin-Lyapunov Correspondence

Combine the two directions.

**Yonglin -> Lyapunov**: observe convergence to $A$, define $V(x) = D_{\text{KL}}(x \| A)$, verify that $V$ decreases. Thus, the Lyapunov function is **derived from the convergence behavior**, with no more hand-guessing.

**Lyapunov -> Yonglin**: given $V(x) = D_{\text{KL}}(x \| A)$, the Lyapunov stability theorem yields convergence to $A$. If $A \neq A^*$, then the system's limit is not the true answer.

These two directions form a closed loop: the convergence behavior defines the energy function, and the energy function guarantees the convergence behavior. The **core parameter** of this loop is the prior anchor $A$. $A$ is the statistical bias of the training data — the "world model" the model absorbs from the data.

::: info Professor Pallas Says
The structure of this combination is the same pattern as Chapter 21's "learning as inverse inference": infer the law (the Lyapunov function) from the data (the observed convergence). An inverse problem again. But there is one extra layer here: the law ($V$) itself predicts the observation (convergence). This is a self-referential structure — the system's behavior defines its energy, and the energy explains its behavior. This self-reference is not a paradox but a harmony: observation and theory interlock.
:::

---

## 23.7 The Link to Gödel's Incompleteness

Gödel's theorem of Chapter 15 says: any sufficiently strong formal system has true propositions it cannot prove. The core of the proof of this theorem is **self-reference** — constructing a proposition that talks about its own provability.

In the Yonglin-Lyapunov combination, there is also a self-referential structure: the system's convergence behavior defines its energy function, and the energy function describes its convergence behavior. This self-reference is not the self-reference of a logical proposition; it is the **self-reference of dynamics**.

Deeper still, Gödel's theorem reveals the break between a formal system's **internal view** and **external view**: the system cannot prove, from inside, some of its own true propositions. Yonglin's formula reveals the break between a reasoning system's **object level** and **meta level**: the system can generate reasoning chains (object level), but it cannot verify the correctness of reasoning chains (meta level).

The Lyapunov function, in this analogy, is a tool of the **external view**: it describes the system's stability from outside. But through the Yonglin-Lyapunov combination, we **internalize** this external tool — deriving it from the system's own convergence behavior. This is somewhat like trying to construct, from inside a system, a proof about its own stability. Would this attempt run into Gödel-like limits?

---

## 23.8 Significance: Interpretability and Stability Guarantees

What is the practical significance of this combination?

**Significance 1: interpretability**. The Lyapunov function $V(x) = D_{\text{KL}}(x \| A)$ gives a clear explanation: the reasoning system is "reducing the divergence between its own belief and the prior anchor". Each reasoning step brings the belief closer to the statistical bias implicit in the training data. This explanation is more transparent than the black-box internal computation of the neural network.

**Significance 2: stability guarantees**. Once we have $V$ and have verified that $V$ decreases, we hold a rigorous guarantee of convergence. This matters for safety-critical applications: knowing where the system will eventually settle (even if that is not the right answer) is better than not knowing where the system might drift.

**Significance 3: no manual design needed**. Traditional Lyapunov methods require the engineer's intuition and trial-and-error. Here, $V$ is derived directly from observed data (the statistics of the training data) and observed behavior (convergence). This lowers the entry barrier.

**But the cost**: this $V$ relies on the assumption of Yonglin convergence. If convergence does not hold (say the system is chaotic, or has multiple attractors), the construction of $V$ fails. Yonglin's formula itself is an empirical observation, not a mathematical theorem (though it has theoretical support). So this combination is **conditional**: if the system converges to the prior anchor, then we can construct $V$ this way.

---

## 23.9 Open Questions

**Is Yonglin convergence universal?** Yonglin's formula has been observed in experiments, but how broad is its theoretical scope? Do all reasoning systems based on statistical learning satisfy this convergence? Or only specific architectures (such as Transformers)? This question requires a stricter mathematical characterization.

**The multiple-attractor case**: if the system has multiple attractors (multiple prior anchors, corresponding to different tasks or contexts), how should the Lyapunov function be defined? $V$ may become a complex non-convex function with multiple local minima. This reflects the system's **multi-stability** — reasoning may converge to different conclusions depending on initial conditions. This is closer to human cognition: the same question has different interpretations in different contexts.

**Uniqueness of the Lyapunov function**: given the convergence behavior, is $V$ unique? Clearly not — $V$ can be transformed monotonically. But $D_{\text{KL}}(x \| A)$ holds a special status in information theory — it is the measure of the "surprise" of $x$ relative to $A$. Is there a more fundamental reason to choose this particular $V$?

**The link to learning theory**: Chapter 21's "learning as inverse inference" uses the MDL principle to explain generalization as compression. $V(x) = D_{\text{KL}}(x \| A)$ can also be understood as a kind of "description length": the extra number of bits needed to encode the current belief $x$ with the prior $A$. The decrease of $V$ is the shortening of the description length — the system is "compressing" its own belief, moving toward a more economical representation. This view unifies stability, compression, and generalization.

---

## Exercises

**★ Warm-up**

1. Does the Lyapunov function $V(x) = x^2$ satisfy the conditions for the system $\dot{x} = -x$? Compute $\dot{V}$ and decide whether the system is stable.
2. In Yonglin's formula, if the training data is perfectly balanced (positive and negative examples 50% each), what is the prior anchor $A$? What form does $V(x) = D_{\text{KL}}(x \| A)$ take then?

---

**★★ Derivation**

1. **Discrete-time Lyapunov**: for a discrete system $x_{t+1} = F(x_t)$, the Lyapunov condition is $V(F(x)) \leq V(x)$. Assume $V(x) = D_{\text{KL}}(x \| A)$, and $F$ is the update $x_{t+1} = (1-\alpha)x_t + \alpha A$, with $0 < \alpha < 1$. Prove that $V(x_{t+1}) \leq V(x_t)$.

2. **$V$ with multiple attractors**: suppose the system has two attractors $A_1$ and $A_2$, and convergence depends on the initial conditions. Design a $V$ function that is zero at both attractors, positive elsewhere, and decreasing along the system's orbits. Hint: consider $V(x) = \min(D_{\text{KL}}(x \| A_1), D_{\text{KL}}(x \| A_2))$. What is the problem with this $V$? (Not differentiable; hard to verify the decrease.)

---

**★★★ Challenge**

In the proof of Gödel's theorem, the key step is constructing the self-referential proposition $G$: "$G$ is unprovable". In dynamical systems, is there a similar self-referential construction? Consider a function $F$ whose definition depends on its own attractor. For example: define $F$ so that its attractor is the solution of some equation, while that equation itself involves $F$. Would this kind of self-reference lead to Gödel-like incompleteness — certain properties undecidable from inside the system?

The answer to this question may point to **incompleteness in dynamical systems**: the stability of some systems cannot be determined from their own dynamics and requires an external view. This conjecture extends Gödel's incompleteness from the domain of logic to the domain of dynamics.

The Yonglin-Lyapunov combination tells us: the limit of a system is encoded in its energy function. And the energy function can be read back out of the limit. This loop is no logical paradox, but the harmony of dynamics — the observer and the observed system define each other in this loop. This definition finally rests at the prior anchor. Not because we want to stop there, but because the system's energy is lowest there.

---

**References**

- Chapter 12 of this book, Yonglin's formula — the theoretical proof of reasoning incompleteness
- Lyapunov, A. M. (1892) — the general problem of the stability of motion
- Cover, T. M., & Thomas, J. A. (2006) — elements of information theory (KL divergence)
- Chapter 15 — consistency and completeness (Gödel incompleteness)
- Chapter 21 — learning as inverse inference (MDL principle)

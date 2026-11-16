# Chapter 23: The Yonglin Corollary — Intelligence and the Log Curves of Emergence

> How fast a system converges is decided by one number.

---

Suppose a model faces only three answers. Its current belief is

$$
p_0=(0.20,\ 0.30,\ 0.50),
$$

and the prior anchor left by the training data is

$$
A=(0.70,\ 0.20,\ 0.10).
$$

[Chapter 12](/en/volume1/chapter12/) already proved that, as long as the step size lies in the right window, this belief walks step by step toward $A$, not toward the true answer $A^*$. Convergence is settled.

But “it converges” and “how fast it converges” are two different things. What you actually want to know is:

- How many steps does it take?
- Do all directions shrink at the same speed?
- If the prior, the model, or the scale changes, how does the speed change?

Here is what it actually looks like. The table below iterates the inference operator from Chapter 12 with step size $\eta=0.183$ (how this number is chosen is 23.3's job; for now, just watch the phenomenon):

| $t$ | $p_t$ | $\lVert p_t-A\rVert_2$ | error ratio |
|---|---|---|---|
| 0 | $(0.200,\ 0.300,\ 0.500)$ | 0.648 | — |
| 1 | $(0.573,\ 0.156,\ 0.271)$ | 0.217 | 0.335 |
| 2 | $(0.621,\ 0.215,\ 0.163)$ | 0.102 | 0.470 |
| 3 | $(0.665,\ 0.223,\ 0.113)$ | 0.0438 | 0.429 |
| 4 | $(0.684,\ 0.214,\ 0.102)$ | 0.0211 | 0.482 |
| 5 | $(0.692,\ 0.206,\ 0.102)$ | 0.00997 | 0.473 |

After two or three steps, the error is multiplied by about 0.45 each time: $0.648\to0.217\to0.102\to0.0438\to0.0211\to0.00997$. The ratio wobbles around 0.45.

That 0.45 is the protagonist of this chapter. It is called the **contraction rate**, written $\rho^*$.

This chapter does one thing: **compute this number, and see what follows from it.**

## The Chapter in Five Steps

1. **23.1–23.3:** compress “how fast” into one number, $\rho^*$;
2. **23.4:** show that this number depends only on two endpoints, and where those endpoints come from;
3. **23.5:** draw $\rho^*$ as a semi-log line and compute the window;
4. **23.6:** use $\rho^*$ to diagnose “emergence”;
5. **23.7:** close.

If you are short on time, read 23.1, 23.3, 23.5, and 23.6. Sections 23.2 and 23.4 answer “why.”

The same object will appear again and again:

$$
\boxed{\ \rho^*=\frac{\kappa_T-1}{\kappa_T+1}\ }
$$

where $\kappa_T=\theta_{\max}/\mu_T$ is the condition number of the tangent Hessian. Don't worry yet about where it comes from; you only need to know that it measures how many times steeper the steepest direction of the energy surface is than the flattest. This number determines the slope of the error curve, the number of steps needed to reach accuracy $\varepsilon$, and whether an “emergent” jump is real. No second scalar is needed.

Why “corollary”? Because the engine of convergence is still Banach's fixed-point theorem, and the energy surface is still the KL divergence from information theory. This chapter does not replace the engine. It reads off the engine's power.

This is not all of intelligence. But any intelligent system whose output space is the probability simplex cannot avoid this piece of geometry.

> **The Yonglin Corollary = an information-theoretic energy surface + Banach completeness + one rate.**

:::info A pointer to the Polish school
The completeness used here comes from Banach's fixed-point theorem, which belongs to the Polish school of mathematics between the two world wars. The story of that school — and the methodological question of why convergence should be placed in a complete space — is told in the Polish-school note in [Chapter 12](/en/volume1/chapter12/). This chapter uses the result and does not retell the history.
:::

## Prologue: Inference and Training Are the Same Road

Put forward inference and backward training on the same sheet of paper. The output of one forward pass is a point $p$ on the simplex; each step of CoT pushes that point one step further. Chapter 12 already writes this “push” as

$$
F(p)=\Pi_{\mathcal X}\bigl(p-\eta P_T\nabla E(p)\bigr),
\qquad
E(p)=D_{\mathrm{KL}}(A\|p).
$$

The anchor $A$ here is not a prior inserted from outside; it is the label distribution left by the training data. For a fixed prompt, $A$ is the conditional label distribution $P_{\text{train}}(Y\mid X)$; change the prompt and $A$ becomes a different point. This is why the GPT-2 experiment found a different anchor for each question.

The objective of backward training is the cross-entropy

$$
L(p)=-\sum_y A_y\log p_y.
$$

Split it open:

$$
L(p)=H(A)+D_{\mathrm{KL}}(A\|p).
$$

The data entropy $H(A)$ is constant, so minimizing cross-entropy is minimizing KL. What is loosely called “minimum-entropy training” is, strictly, “minimum relative-entropy training”: making the model distribution $p$ as close as possible to the empirical distribution $A$. Differentiating with respect to $p$,

$$
\nabla_p L=-\frac{A}{p},
$$

which is exactly the gradient of $E(p)=D_{\mathrm{KL}}(A\|p)$. So at the level of output distributions, training and inference are not two energy surfaces; they are one.

Then why does the code show the logit gradient $p-\mathbf{1}_y$? Because softmax maps logits to the simplex, and the gradient must pass through the softmax Jacobian. For a single sample,

$$
\frac{\partial L}{\partial z}=p-\mathbf{1}_y;
$$

taking the expectation over the data distribution gives $p-A$. This is not a different direction. The softmax Jacobian is $\operatorname{diag}(p)-pp^\top$, which on the tangent space is exactly the inverse of the Fisher metric; applied to $-A/p$, it gives $p-A$. In other words, backpropagation takes the natural gradient, while forward inference takes the Euclidean projected gradient; the two share one energy surface and differ only in the ruler underfoot.

The “empirical simplex” is the set of all possible output distributions: the training data contributes one point $A$, the model output another point $p$. Training drags the point toward $A$; inference, when evidence arrives, pushes it away from $A$ and then lets it fall back. Both move on the same simplex. The constraint $\sum_i p_i=1$ makes the direction “add the same number to every component” meaningless — adding a constant to logits does not change the softmax, and $P_T$ deletes that normal component. So the directions that actually exist are the tangent space

$$
T=\left\{v:\ \sum_i v_i=0\right\}.
$$

The backward gradient lands on $T$ automatically after passing through the softmax Jacobian; the inference gradient is projected onto $T$ by $P_T$. Both are forced by the same constraint into the same tangent space, and therefore see the same Hessian:

$$
H_T=P_T\,\operatorname{diag}\!\left(\frac1{A_i}\right)P_T.
$$

One boundary of language matters here: a parameter-space update does not itself live on the simplex; it is pushed forward through the output map before it lands in the tangent space. “Tangent-space training” refers to the tangential motion of the output distribution, not to the parameters walking inside the simplex.

Linearize the two dynamics near $A$: both multiply the error by $I-\eta H_T$. $H_T$ has $K-1$ eigenvalues, but the long-run rate is set by only the slowest and the fastest endpoints:

$$
\kappa_T=\frac{\theta_{\max}}{\mu_T},
\qquad
\rho^*=\frac{\kappa_T-1}{\kappa_T+1}.
$$

This is the Yonglin Corollary. It is not only “how fast inference returns to the prior”; it is also “how fast the last stretch of training sticks to the label distribution.” Training builds $A$; inference returns to $A$; the last stretch of both approaches is priced by the same pair of spectral endpoints.

One boundary must be stated plainly: the exact derivation of $\rho^*$ uses Euclidean projected gradient on the output simplex; real training is closer to KL mirror descent, whose ruler is the Fisher metric. Whether a Bregman version exists that still depends only on two endpoints is open — it is the first open question at the end of this chapter. What can be said rigorously now is that the energy surface is the same, the tangent space is the same, and the two-endpoint mechanism is the same; the exact scalar holds in the Euclidean version and needs a separate proof in the mirror version. Real training also involves non-convexity, stochasticity, finite samples, and parameterization; the local quadratic picture is a theorem about the last stretch, not a global theory of training.

But the last stretch is enough for what this chapter explains. Training builds the anchor; inference returns to it; one condition number prices both roads.

---

## 23.1 Turning “How Fast” into a Number

Start with the error

$$
e_t=p_t-A.
$$

Convergence means $e_t\to0$. “How fast” means: by what factor does $e_t$ shrink each step?

### First, one dimension

If the error is a single number and the update is $e_{t+1}=j e_t$, the answer is simple: $j$ is the per-step multiplier. $j=0.5$ means halving each step; $j=0.9$ means losing only a tenth each step.

In a world of $K$ answers, the error is a vector and the update is approximately a matrix:

$$
e_{t+1}\approx J e_t.
$$

A matrix compresses different directions by different amounts. Its eigenvalues $\lambda_i$ are the multipliers: along the $i$-th eigendirection, the error is multiplied by $\lambda_i$ each step.

In the long run, **the slowest direction decides everything.** If one direction multiplies by $0.9$ and another by $0.1$, after ten steps the first still has $0.35$ while the second is down to $10^{-10}$. The fast direction left long ago; the shape and speed of the error are now controlled by the slow direction.

So “how fast” is not an average; it is the slowest multiplier:

$$
\rho=\max_i|\lambda_i|.
$$

### Writing down the multiplier matrix

The inference operator is still the one from Chapter 12:

$$
F(p)=\Pi_{\mathcal X}\bigl(p-\eta P_T\nabla E(p)\bigr),
\qquad
E(p)=D_{\mathrm{KL}}(q\|p).
$$

Each symbol does exactly one job:

- $E$ is the energy surface, with anchor $q=A$;
- $\nabla E$ says which way is downhill;
- $P_T$ projects the gradient onto the tangent space, removing the normal component that adds the same amount to every answer;
- $\Pi_{\mathcal X}$ pulls the point back into the probability simplex.

At the anchor, the curvature of the energy surface is

$$
\nabla^2E(A)=\operatorname{diag}\!\left(\frac1{q_i}\right),
\qquad
H_T=P_T\,\nabla^2E(A)\,P_T.
$$

$H_T$ is the tangent Hessian. It describes how curved the energy surface is along each direction, under the constraint that the probabilities sum to one.

The derivative of one update — the multiplier matrix for the error — is

$$
J_T=I_T-\eta H_T.
$$

Its eigenvalues are $1-\eta\lambda_i$, where $\lambda_i$ are the eigenvalues of $H_T$. Hence

$$
\rho(\eta)=\max_i|1-\eta\lambda_i|.
$$

> Don't mix the two sets of eigenvalues: $\lambda_i$ is the curvature of the energy surface $H_T$; $1-\eta\lambda_i$ is the multiplier of one update. The first says how steep the slope is; the second says how much error is left after one step.

**Why $\max$ and not an average?** Because if one direction shrinks slowly, it will eventually fill the whole error. Fast directions just leave early; the slow direction decides the ending.

**This is not an upper bound. It is the exact value.**

:::details Why this is the exact rate, not an upper bound
$J_T$ is symmetric on the tangent space, so its operator norm equals its spectral radius:

$$
\|J_T\|_{T\to T}=\rho(\eta).
$$

For initial points near $A$, the asymptotic rate of the linearized error is exactly $\rho(\eta)$:

$$
\lim_{t\to\infty}\left(\frac{\|p_t-A\|_2}{\|p_0-A\|_2}\right)^{1/t}=\rho(\eta).
$$

(If the initial error has a nonzero component in the dominant eigendirection; otherwise it is only faster.)
:::

![Figure 23.1: Geometric compression inside the simplex — after entering the anchor's neighborhood, every step multiplies the error by the same factor $\rho^*$](/figures/ch23_simplex_orbit.png)

Figure 23.1 draws compression on the probability simplex. The orbit starts at $p_0$ and moves toward the anchor $A=q$ at every iteration. Just after entering the anchor's neighborhood the contraction ratio fluctuates briefly; once inside the range where linearization is valid, every step multiplies the error by the same number $\rho^*\approx0.454$. This is what “geometric convergence” means geometrically: not the same distance removed each time, but the same proportion removed each time.

> **One sentence:** the contraction rate is the retention ratio of the slowest direction; after each step the error is at most $\rho$ times what it was, and this “at most” is attainable.

---

## 23.2 Why Only Two Endpoints Matter

$H_T$ has $K-1$ eigenvalues. A natural worry: do we have to compute all of them to get the rate?

No.

Start with a simpler question. Fix the step size $\eta$ and plot

$$
g(\lambda)=|1-\eta\lambda|.
$$

It is a V-shaped curve, with its minimum at $\lambda=1/\eta$ and both arms rising.

We want the maximum of this V over the interval $[\mu_T,\theta_{\max}]$. A V-shape is convex, and a convex function on an interval attains its maximum only at an endpoint. So how many eigenvalues lie in between, and where, does not affect the worst case.

**Lemma 23.1.** Suppose $\mu_T\le\lambda_i\le\theta_{\max}$. For every $\eta>0$,

$$
\max_i|1-\eta\lambda_i|
=\max\left(|1-\eta\mu_T|,\ |1-\eta\theta_{\max}|\right).
$$

**Proof.** Fix $\eta$. The function $g(\lambda)=|1-\eta\lambda|$ is convex in $\lambda$ (an absolute value composed with an affine function). A convex function on an interval attains its maximum at an endpoint. Thus $g(\lambda_i)\le\max(g(\mu_T),g(\theta_{\max}))$ for every $\lambda_i$, and the endpoints are themselves eigenvalues, so equality holds. $\square$

This lemma compresses a $(K-1)$-dimensional spectrum into two numbers: **the minimum $\mu_T$ and the maximum $\theta_{\max}$.** How many eigenvalues lie in between, and where, does not affect the contraction rate.

It also gives the exact stability window:

$$
\rho(\eta)<1
\iff
0<\eta<\frac{2}{\theta_{\max}}.
$$

Why this boundary? When the step is too large, the fastest mode $1-\eta\theta_{\max}$ is the first to cross $-1$: the error starts flipping sign and growing. The boundary is exact, not “approximately here.”

> **One sentence:** watch only the flattest and the steepest directions; the eigenvalues in between change the shape of the spectrum, not the worst case.

---

## 23.3 Optimal Step Size: Let the Slowest and the Fastest Arrive Together

Now choose the step size.

Too small: the slow direction barely moves and the error stalls early. Too large: the fast direction overshoots, oscillates, and can even diverge. The optimum lies in between.

Draw the two endpoint errors as two lines:

- slow mode: $|1-\eta\mu_T|$, falling as $\eta$ grows;
- fast mode: $|1-\eta\theta_{\max}|$, rising as $\eta$ grows.

One falls, one rises; their intersection is the optimum:

$$
\rho^*=\inf_{\eta>0}\ \max\left(|1-\eta\mu_T|,\ |1-\eta\theta_{\max}|\right).
$$

At the optimum, the two branches are equal:

$$
1-\eta\mu_T=\eta\theta_{\max}-1.
$$

Solving,

$$
\eta^*=\frac{2}{\mu_T+\theta_{\max}},
\qquad
\rho^*=\frac{\theta_{\max}-\mu_T}{\theta_{\max}+\mu_T}
=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T}.
$$

The name of $\kappa_T$ is the **condition number**. Here it is just a ratio: how many times steeper the steepest direction is than the flattest.

**Theorem 23.2 (the rate form of the Yonglin Corollary).** Inside the stability window, the minimum of $\rho(\eta)$ is $\rho^*$, attained uniquely at $\eta^*$. Moreover,

$$
|1-\eta^*\mu_T|=|1-\eta^*\theta_{\max}|=\rho^*.
$$

That is, the optimal step size makes the slowest and the fastest modes contract at exactly the same speed.

This rate has four exact properties, each a one-line calculation:

1. **Range.** $0\le\rho^*<1$. When $\kappa_T=1$ (all tangent eigenvalues equal), $\rho^*=0$ and one step suffices; as $\kappa_T\to\infty$, $\rho^*\to1$ and convergence becomes arbitrarily slow.
2. **Monotonicity.** $\dfrac{d\rho^*}{d\kappa_T}=\dfrac{2}{(\kappa_T+1)^2}>0$. A larger condition number means a worse rate, without exception.
3. **Invertibility.** $\kappa_T=\dfrac{1+\rho^*}{1-\rho^*}$. Rate and condition number correspond one-to-one: knowing the rate is knowing the condition number, and vice versa. In this sense, $\rho^*$ is a complete invariant of the condition number.
4. **Endpoint dependence.** By Lemma 23.1, $\rho^*$ is determined only by $\mu_T,\theta_{\max}$. The shape of the spectrum in between never enters the formula.

Property 3 is worth a second look. It says that “how fast it converges” and “how anisotropic the energy surface is” are not two independent questions: they are two ways of writing the same number.

![Figure 23.2: Three readings of one scalar — the exact V, the rate-condition-number inverse, and the exact linear sandwich for the window](/figures/ch23_rate_properties.png)

Panel (a) is the V-shape of $\rho(\eta)$. The left branch is controlled by the slow mode $|1-\eta\mu_T|$, the right branch by the fast mode $|1-\eta\theta_{\max}|$, and the intersection is $(\eta^*,\rho^*)$. The dashed line is the stability boundary $2/\theta_{\max}$.

Panel (b) is $\rho^*=(\kappa_T-1)/(\kappa_T+1)$. It is monotone, bounded, and invertible.

Panel (c) is the window. Section 23.5 turns it into an exact inequality.

![Figure 23.3: The rate–condition-number hyperbola — two directions, one relation](/figures/ch23_rate_hyperbola.png)

Figure 23.3 isolates the relation. The left panel plots $\rho^*=(\kappa_T-1)/(\kappa_T+1)$ as a function of $\kappa_T$: a hyperbola branch with horizontal asymptote $\rho^*=1$. The right panel plots its inverse $\kappa_T=(1+\rho^*)/(1-\rho^*)$: as $\rho^*\to1$, the condition number shoots up vertically. The two panels say the same thing — the better the rate, the closer the condition number is to 1; as the rate approaches 1, the condition number explodes.

Back to the opening example. $q=(0.70,0.20,0.10)$ gives

$$
\mu_T=2.9904,\quad \theta_{\max}=7.9620,\quad \kappa_T=2.6625,
$$

$$
\eta^*=0.18261,\quad \rho^*=0.45393.
$$

These are the 0.183 and 0.45 in the table. The first ratio, 0.335, is the residue of the fast direction; after a few steps the slow direction takes over and the ratio settles near 0.454.

> **One sentence:** the optimal step size is not the fastest step; it is the step that lets the slowest and the fastest arrive together.

---

## 23.4 Where the Two Endpoints Come From

Only one thing remains: where do $\mu_T$ and $\theta_{\max}$ come from?

Take an example with $K=6$ and anchor

$$
q=(0.35,\ 0.25,\ 0.15,\ 0.12,\ 0.08,\ 0.05).
$$

![Figure 23.4: Three readings of the Hessian — the raw matrix, the tangent projection, and the tangent spectrum](/figures/ch12_special_hessian_matrix.png)

Read this figure as three panels: the raw Hessian, the projected Hessian, and its spectrum.

### Panel (a): The raw Hessian — the “local stiffness” of each answer

$$
\nabla^2E(q)=\operatorname{diag}\!\left(\frac1{q_1},\ldots,\frac1{q_K}\right)
=\operatorname{diag}(2.86,\ 4.00,\ 6.67,\ 8.33,\ 12.50,\ 20.00).
$$

The $i$-th diagonal entry is $1/q_i$. The rarer the answer, the steeper that direction. Off the diagonal everything is zero: if probabilities could vary independently, the answers would not be coupled.

But probabilities cannot vary independently. They must sum to one. This constraint plants off-diagonal entries.

### Panel (b): The tangent Hessian — the constraint couples the answers

After projection, each entry of $H_T$ is

$$
(H_T)_{ij}
=\frac{\delta_{ij}}{q_i}
-\frac{1}{K}\left(\frac{1}{q_i}+\frac{1}{q_j}\right)
+\frac{1}{K^2}\sum_{k=1}^{K}\frac{1}{q_k}.
$$

Three readings:

1. **Every row sums to zero.** The all-ones direction is normal, $H_T\mathbf{1}=0$. Adding the same number to every answer does not change the belief.
2. **Diagonal entries are positive, and the total trace shrinks.** $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i=45.30$, smaller than the raw trace $54.36$. The constraint eats part of the total curvature.
3. **Most off-diagonal entries are negative, but not all.** The two rarest answers are strongly negatively coupled ($-3.91$ between $q_5$ and $q_6$): raise one and you must lower the other. The two most common answers have a small positive coupling ($+0.37$ between $q_1$ and $q_2$). The only strict statements are: row sums vanish, and the matrix is positive semidefinite on the tangent space.

An analogy: six balls threaded on a rope of fixed total length. Pull one left and the others must move right. The constraint is not background; it is the source of the coupling.

A beautiful special case: when $q$ is uniform, $1/q_i=K$, so

$$
H_T=K\,P_T=K\left(I-\frac1K\mathbf{1}\mathbf{1}^\top\right).
$$

This is the graph Laplacian of the complete graph $K_K$, with eigenvalues $0$ (normal) and $K$ (tangent, multiplicity $K-1$). The tangent geometry of a uniform belief is the graph in which every answer has an equal-weight edge to every other answer.

### Panel (c): The tangent spectrum — only the two ends matter

For this anchor, the tangent spectrum is

$$
\lambda=(3.32,\ 5.24,\ 7.57,\ 11.13,\ 18.03).
$$

Hence

$$
\mu_T=3.32,\qquad \theta_{\max}=18.03,\qquad \kappa_T=5.43.
$$

- The smallest eigenvalue is the slow mode: “redistribute mass between the two most likely answers”;
- The largest eigenvalue is the fast mode: “correct the rarest answer”;
- The three eigenvalues in between contribute nothing to the optimal rate. This is Lemma 23.1 in action.

![Figure 23.5: Interlacing of the tangent spectrum with the raw diagonal; only the two endpoints enter the rate](/figures/ch23_hessian_spectrum.png)

Panel (a) of Figure 23.5 draws the interlacing of the secular equation: gray levels are the raw diagonal $d_i=1/q_i$, blue marks are the tangent eigenvalues. Every blue root lies between two adjacent gray levels, $d_{(1)}<\lambda_1<d_{(2)}<\cdots$. Panel (b) switches to three anchors with $K=6$: the uniform anchor has five equal eigenvalues ($\kappa_T=1$); the baseline anchor gives $\kappa_T=5.43$; an anchor near a vertex pushes all eigenvalues up together, but the ratio only reaches $5.70$ — the tangent condition number is pinned near $K=6$ by the constraint. The eigenvalues in between change the shape of the spectrum, not $\rho^*$.

:::details If you want to compute it: the tangent spectrum is determined by one scalar equation

Let $D=\operatorname{diag}(d_1,\ldots,d_K)$ with $d_i=1/q_i$, and $H_T=P_TDP_T$. For $x\in T$, $H_Tx=\lambda x$ is equivalent to $P_TDx=\lambda x$. Since $x\in T$, $P_Tx=x$, so $Dx-\lambda x=c\mathbf{1}$, i.e. $x_i=c/(d_i-\lambda)$. Using $\sum_ix_i=0$ gives the **secular equation**

$$
\sum_{i=1}^{K}\frac{1}{d_i-\lambda}=0.
$$

Its $K-1$ roots are the tangent spectrum, interlacing with $d_i$:

$$
d_{(1)}<\lambda_1<d_{(2)}<\cdots<\lambda_{K-1}<d_{(K)}.
$$

To get $\mu_T$ and $\theta_{\max}$, bisect once between the two smallest $d_i$ and once between the two largest. For $K=3$, the secular equation becomes

$$
3s_3\lambda^2-2s_2\lambda+1=0,
\qquad
s_2=q_1q_2+q_1q_3+q_2q_3,\quad s_3=q_1q_2q_3,
$$

which is exactly the quadratic that appeared in the “direct evidence” section of [Chapter 12](/en/volume1/chapter12/).

:::

:::details A beautiful limit: the tangent condition number saturates at $K$

The raw Hessian explodes at the boundary: as $q_{\min}\to0$, $1/q_{\min}\to\infty$. But the tangent spectrum is pinned by the constraint. Take the symmetric vertex limit

$$
q(t)=\frac{1-t}{K}\mathbf{1}+t\,e_1,\qquad t\to1.
$$

The full-space condition number $q_{\max}/q_{\min}\sim1/(1-t)$ diverges, while the tangent condition number satisfies

$$
\kappa_T\to K.
$$

Numerically, $K=5,10,50,200,1000$ give $\kappa_T=5.0000,10.0000,50.0000,199.9998,999.9990$. The reason: the $K-1$ rare answers have equal $d_i$, the secular equation pulls only one root $d_{\text{rare}}/K$ out of that block, and the remaining $K-2$ roots coincide. The constraint smears the explosion of “one tiny probability” into a coupling among $K$ answers.

![Figure 23.6: The full-space condition number diverges; the tangent condition number saturates at $K$](/figures/ch23_kappa_saturation.png)

Panel (a) of Figure 23.6 is the symmetric vertex limit for $K=5$: the red line is the full-space condition number, diverging logarithmically as $t\to1$; the blue line is the tangent condition number, sitting firmly at $K=5$. Panel (b) normalizes the spectrum by $\mu_T$: the minimum stays at 1, while the other $K-1$ eigenvalues rise together to $K$. The divergence has not disappeared; the constraint has redistributed it into a coupling among $K$ answers.

:::

> **One sentence:** the endpoints are not guessed; they are handed over by the matrix. The constraint smears the explosion of “one tiny probability” into a coupling among $K$ answers.

---

## 23.5 The Rate Is the Slope of the Log Curve

Back to the table. The errors are

$$
0.648,\ 0.217,\ 0.102,\ 0.0438,\ 0.0211,\ 0.00997.
$$

From the third step on, each is about $0.45$ times the previous one. If the decay were exactly $\rho^t$, then

$$
\ln\|p_t-A\|_2=\ln\|p_0-A\|_2+t\ln\rho.
$$

This is a linear function of $t$. **On a semi-log plot it is a straight line, and the slope is exactly $\ln\rho$.**

This is also what “geometric convergence” means: not the same distance removed each time, but the same proportion removed each time.

Two kinds of coordinates must be kept apart:

- **Geometric convergence** is a straight line on a semi-log plot;
- **A power law** is a straight line on a log-log plot.

They are two different laws, not two ways of drawing the same curve. Figure 23.7 plots the same data twice.

![Figure 23.7: A straight line on semi-log axes, a curve on log-log axes](/figures/ch23_semilog_loglog.png)

Panel (a) of Figure 23.7 uses semi-log axes: three geometric orbits are exact straight lines, with slopes equal to $\ln\rho$. With too small a step (light blue) the slope is shallow; with the optimal step (dark blue) it is steepest; with too large a step (red), although the slow mode is crushed quickly, the fastest mode drags it down and the overall rate is worse. Panel (b) switches to log-log axes: the same geometric orbit bends into a curve and is no longer straight; the gray dashed line is a genuine power law $t^{-1.5}$, which is straight on log-log axes. So “a log-log straight line” belongs to power laws; “a semi-log straight line” belongs to geometric convergence.

At rate $\rho^*$, the continuous number of steps needed to reach accuracy $\varepsilon$ is

$$
N_{\mathrm{real}}(\varepsilon)=\frac{\ln(\|p_0-A\|_2/\varepsilon)}{\ln(1/\rho^*)},
\qquad
N(\varepsilon)=\left\lceil N_{\mathrm{real}}(\varepsilon)\right\rceil.
$$

For the baseline $q=(0.70,0.20,0.10)$, substituting $\rho^*=0.45393$ with $\|p_0-A\|_2=1$ and $\varepsilon=0.1$:

$$
N_{\mathrm{real}}(0.1)=2.915,
\qquad
N(0.1)=3.
$$

That is: about three steps. This matches the opening table — at step 2 the error is still $0.102$; at step 3 it is $0.0438$.

Here is a harder statement: without first computing $\rho^*$, the two endpoints alone sandwich the window.

**Theorem 23.3 (exact linear bounds on the window).** For $\kappa_T>1$,

$$
\frac{\kappa_T-1}{2}\ln\frac1\varepsilon
\ \le\ N_{\mathrm{real}}(\varepsilon)\ \le\
\frac{\kappa_T}{2}\ln\frac1\varepsilon.
$$

:::details Proof

Let $x=2/(\kappa_T-1)>0$. Since $(\kappa_T+1)/(\kappa_T-1)=1+x$,

$$
\ln\frac1{\rho^*}=\ln\frac{\kappa_T+1}{\kappa_T-1}=\ln(1+x).
$$

Two standard inequalities give

$$
\frac{2x}{2+x}\le\ln(1+x)\le x.
$$

Substituting $x=2/(\kappa_T-1)$:

$$
\frac{2}{\kappa_T}\le\ln\frac1{\rho^*}\le\frac{2}{\kappa_T-1}.
$$

Taking reciprocals and multiplying by $\ln(1/\varepsilon)$ gives the claim. $\square$

:::

Both bounds are exact, not asymptotic approximations. They sandwich the window between two lines of slopes $(\kappa_T-1)/2$ and $\kappa_T/2$. Panel (c) of Figure 23.2 draws this sandwich: the blue line is the exact $N$, and the shaded region is bounded by the two exact lines.

For the baseline $q=(0.70,0.20,0.10)$, the exact bounds give

$$
\frac{2.6625-1}{2}\ln10=1.914
\ \le\ 2.915\ \le\
\frac{2.6625}{2}\ln10=3.065.
$$

The window falls inside the bounds, and the two bounds differ by only $1.15$ steps. This is not “about three steps”; it is “exactly between 1.914 and 3.065.”

> **One sentence:** the rate is the slope; the window is computed, not felt.

---

## 23.6 Emergence: The Rate Changed, or Only the Ruler Did

Now bring “emergence” into the picture.

An error curve is $\rho^t$. If $\rho$ does not change, the curve is a smooth geometric curve with no kink.

So a kink that looks like “it suddenly learned” has only two possible explanations:

1. **The ruler changed, the rate did not.** The metric is nonlinear. For example, accuracy is $\mathbf{1}[\text{error}<\varepsilon]$; when a smooth error curve crosses a threshold, accuracy looks like a step. Schaeffer et al. (2023) argue systematically that this kind of “emergence” can be manufactured entirely by the choice of metric.
2. **The rate really changed.** This can happen only through a change in $\mu_T$ or $\theta_{\max}$: a new slow mode enters the low end of the spectrum, or a new fast mode enters the high end. Then $\rho^*$ itself changes, and the error curve really bends.

With this chapter's scalar, the diagnostic rule is one line:

> First measure how $\rho^*$ changes with scale. If $\rho^*$ is unchanged while a benchmark jumps, it is a readout; if $\rho^*$ jumps, it is emergence in the dynamical sense.

![Figure 23.8: Two kinds of “emergence” — a readout jump, or a real change of slope](/figures/ch23_emergence.png)

Figure 23.8 draws the criterion. In panel (a), loss is a straight line on log-log axes and the rate has not changed; but accuracy is a threshold metric, and when the curve crosses the threshold it looks like a step. In panel (b), $\rho^*$ itself drops from $0.82$ to $0.45$, and the slope of the semi-log error curve changes — over the same number of steps, the error differs by orders of magnitude. The first is a readout; the second is dynamics.

This is harder than “does the curve look emergent?” It turns “emergence” from an adjective into a measurable spectral event. The emergent phenomena described by Wei et al. (2022) are genuine empirical facts; this chapter supplies a criterion for separating the part that comes from the rate from the part that comes from the readout.

:::details A boundary that must be admitted: this chapter does not define “intelligence”
The title “Intelligence and the Log Curves of Emergence” is easy to read as “this chapter explains intelligence.” It does not.

This chapter explains a class of learning dynamics: when a system minimizes KL and outputs over the probability simplex, what mathematics governs its error curve. It does not answer “what is understanding,” “what is consciousness,” or “why do some abilities appear only in sufficiently large models.” Those questions need other tools, and perhaps tools that do not yet exist.

Treating the Yonglin Corollary as a definition of intelligence is a mistake; treating it as a geometric bill that any intelligent system must pay is appropriate.
:::

> **One sentence:** measure the slope first, then talk about emergence.

---

## 23.7 Conclusion: One Number

Back to the opening: three answers, initial error $0.648$, three steps to $0.1$, each step multiplying by about $0.45$.

That number is

$$
\rho^*=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T},
\qquad
\kappa_T=\frac{1+\rho^*}{1-\rho^*}.
$$

It determines exactly four things:

1. **Slope.** The semi-log error curve has slope $\ln\rho^*$.
2. **Steps.** The number of steps to reach $\varepsilon$ is exactly sandwiched between $\frac{\kappa_T-1}{2}\ln\frac1\varepsilon$ and $\frac{\kappa_T}{2}\ln\frac1\varepsilon$.
3. **Stability.** The stability window is $0<\eta<2/\theta_{\max}$, and the optimal step is $\eta^*=2/(\mu_T+\theta_{\max})$.
4. **The emergence criterion.** If $\rho^*$ is unchanged while the metric jumps, it is a readout; if $\rho^*$ jumps, it is dynamics.

The whole chain has only three links:

> one number, determined by two endpoints; two endpoints, determined by the tangent Hessian; the tangent Hessian, determined by the anchor and the simplex constraint.

[Chapter 12](/en/volume1/chapter12/) says: inference converges to the prior anchor $A$, not to the true answer $A^*$. This chapter adds the “how fast”:

> How fast is determined by one condition number; the condition number is determined by two endpoints of the tangent Hessian; the two endpoints are determined by the matrix. Beyond this, there is no second scalar.

“Scaling laws are not mysticism” can now stand, and it does not need a private model to testify. GPT can push a loss curve down to 2.5, while a small FFN may only reach 9: their numerical floors differ, but they share the same log-log curve. The model is only the tool that probes downward; it is not the inventor of the curve. As long as a system performs simplex projection, moves inside the simplex, and trains with a minimum-entropy (KL) objective, it exhibits this geometric criterion. In other words, **any architecture satisfying these three conditions can present the same “geometric certificate”; the architecture sets the numerical floor on the certificate, not its format.**

---

## Open Questions

**What is the exact rate for mirror descent?** The contraction rate here is built on Euclidean projected gradient descent. Real models are closer to KL mirror descent, and KL is not a metric, so Banach does not apply directly. Is there a Bregman version of the spectral-radius formula? If so, does it still depend only on two endpoints? This is open.

**Can $\mu_T$ and $\theta_{\max}$ be measured directly on large models?** The spectra here are computed on explicit distributions with $K=3$ to $2048$. A real LLM has a huge vocabulary, and the Hessian cannot be stored explicitly. Can stochastic trace estimation, Lanczos, or implicit differentiation give reliable endpoint estimates? That would turn “emergence diagnosis” from theory into experiment.

**When does $\rho^*$ change with scale?** This chapter gives the criterion but not a model for $\rho^*(N)$. If the appearance of a slow mode has a critical scale, what determines that critical point? This is the question that “emergence” really needs to answer.

---

## Exercises

**★ Warm-up**

1. Using the opening table, verify at which step the error ratio settles near $\rho^*$. Why is the first ratio noticeably smaller?
2. For $K=3$ and $q=(1/3,1/3,1/3)$, verify $H_T=3P_T$, write down all eigenvalues, and compute $\rho^*$.
3. For $q=(0.70,0.20,0.10)$, use the secular equation to compute $\mu_T,\theta_{\max},\kappa_T$, then $\eta^*,\rho^*$, and $N(0.1)$.

**★★ Derivations**

1. Prove Lemma 23.1: the convex function $g(\lambda)=|1-\eta\lambda|$ attains its maximum on an interval at an endpoint.
2. Prove $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i$.
3. Prove Theorem 23.2: $\eta^*=2/(\mu_T+\theta_{\max})$ is the unique minimizer, $\rho^*=(\kappa_T-1)/(\kappa_T+1)$, and the two endpoint branches are equal at the optimum.
4. Prove the two bounds in Theorem 23.3: $\frac{2x}{2+x}\le\ln(1+x)\le x$.

**★★★ Challenge**

1. For $K=3$, write the secular equation as $3s_3\lambda^2-2s_2\lambda+1=0$, and verify that it matches the quadratic in [Chapter 12](/en/volume1/chapter12/).
2. Design an experiment that uses only the change of $\rho^*$ with scale to distinguish a “readout effect” from a “spectral bifurcation.” State what you would see in each case and how to rule out confounders.
3. Try to construct a counterexample showing that under KL mirror descent the Euclidean contraction rate $\rho^*$ is no longer the exact rate; then state what extra condition would restore an exact rate formula.

---

## Appendix: Core Computation Script

The code below does one thing: given an anchor $q$, compute the two endpoints of the tangent spectrum via the secular equation, then the exact rate, the optimal step, and the window. It trains no model.

```python
import numpy as np

def tangent_spectrum(q, iters=90):
    """Eigenvalues of P_T diag(1/q) P_T on T={sum=0}, via secular equation."""
    d = np.sort(1.0 / np.asarray(q, float))
    K = len(d)
    roots = np.empty(K - 1)
    for j in range(K - 1):
        lo, hi = d[j], d[j + 1]          # one root in each interval
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = np.sum(1.0 / (d - mid))  # sum 1/(d_i - lambda)
            if f > 0: hi = mid
            else:     lo = mid
        roots[j] = 0.5 * (lo + hi)
    return roots


def rate(q, eps=0.1, e0=1.0):
    lam = tangent_spectrum(q)
    mu, theta = lam.min(), lam.max()
    kappa = theta / mu
    eta_star = 2.0 / (mu + theta)
    rho_star = (kappa - 1.0) / (kappa + 1.0)
    n_real = np.log(e0 / eps) / np.log(1.0 / rho_star)
    n_steps = np.ceil(n_real)
    lo = (kappa - 1.0) / 2.0 * np.log(e0 / eps)
    hi = kappa / 2.0 * np.log(e0 / eps)
    return dict(mu_T=mu, theta_max=theta, kappa_T=kappa,
                eta_star=eta_star, rho_star=rho_star,
                n_real=n_real, n_steps=n_steps, n_lo=lo, n_hi=hi)


print(rate([0.70, 0.20, 0.10]))
# mu_T=2.9904, theta_max=7.9620, kappa_T=2.6625,
# eta_star=0.18261, rho_star=0.45393, n_real=2.915, n_steps=3,
# n_lo=1.914, n_hi=3.065

print(rate([0.35, 0.25, 0.15, 0.12, 0.08, 0.05]))
# kappa_T=5.4268
```

---

**References**

- Banach, Stefan (1922). “Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.” *Fundamenta Mathematicae* 3(1), 133–181. DOI: 10.4064/fm-3-1-133-181. — the original paper on the contraction mapping principle
- Schaeffer, Rylan; Miranda, Brando; and Koyejo, Sanmi (2023). “Are Emergent Abilities of Large Language Models a Mirage?” arXiv:2304.15004. — emergent abilities as a metric artifact
- Wei, Jason; Tay, Yi; Bommasani, Rishi; Raffel, Colin; Zoph, Barret; Borgeaud, Sebastian; Yogatama, Dani; Bosma, Maarten; Zhou, Denny; Metzler, Donald; Chi, Ed H.; Hashimoto, Tatsunori; Vinyals, Oriol; Liang, Percy; Dean, Jeff; and Fedus, William (2022). “Emergent Abilities of Large Language Models.” arXiv:2206.07682. — a systematic description of emergent phenomena
- Amari, Shun-ichi (2016). *Information Geometry and Its Applications*. Applied Mathematical Sciences 194. Springer Japan. DOI: 10.1007/978-4-431-55978-8. — Bregman geometry, Fisher information, and natural gradient
- Nesterov, Yurii (2018). *Lectures on Convex Optimization*, 2nd edition. Springer Optimization and Its Applications 137. Springer. DOI: 10.1007/978-3-319-91578-4. — strong convexity, smoothness, step size, and condition number
- Boyd, Stephen; and Vandenberghe, Lieven (2004). *Convex Optimization*. Cambridge University Press. DOI: 10.1017/CBO9780511804441. — nonexpansiveness of projection and convergence rates
- [Chapter 12](/en/volume1/chapter12/), Yonglin's Formula — the global version of the Yonglin theorem, tangent spectral bounds, and direct evidence

---

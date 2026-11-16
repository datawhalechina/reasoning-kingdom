# Chapter 23: The Yonglin Corollary — Intelligence, Emergence, and the Ordinary Equation

> Intelligence is no exception, and neither is emergence; they are names that one ordinary equation takes at different scales.

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

This chapter does one thing: **compute this number, then use it to reproduce the most famous curve in modern AI.**

That curve is called scaling law. Put loss on log-log axes with scale on the horizontal axis, and it is a straight line. In recent years it has been put under a spotlight; the spotlight falls on engineering scale, while the formula itself is plain: loss minus its floor is proportional to a negative power of scale. At the end of this chapter we will reproduce that straight line from the simplex with our own hands. Not by repeating any one company's experiment, but by reproducing its “straightness” — you will see that this line was not invented by anyone; it is a necessary consequence of simplex geometry.

## The Chapter in Five Steps

1. **23.1–23.3:** compress “how fast” into one number, $\rho^*$;
2. **23.4:** show that this number depends only on two endpoints, and where those endpoints come from;
3. **23.5:** draw $\rho^*$ as a semi-log line and compute the window;
4. **23.6:** reproduce the scaling-law line from the simplex;
5. **23.7:** close.

If you are short on time, read 23.1, 23.3, 23.5, and 23.6. Sections 23.2 and 23.4 answer “why.”

The same object will appear again and again:

$$
\boxed{\ \rho^*=\frac{\kappa_T-1}{\kappa_T+1}\ }
$$

where $\kappa_T=\theta_{\max}/\mu_T$ is the condition number of the tangent Hessian. Don't worry yet about where it comes from; you only need to know that it measures how many times steeper the steepest direction of the energy surface is than the flattest. This number determines the slope of the error curve, the number of steps needed to reach accuracy $\varepsilon$, and whether an “emergent” jump has a dynamical basis. No second scalar is needed.

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

The anchor $A$ here is not a prior inserted from outside; it is the label distribution left by the training data. For a fixed prompt, $A$ is the conditional label distribution $P_{\text{train}}(Y\mid X)$; change the prompt and $A$ becomes a different point. This is why per-question experiments found a different anchor for each question.

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

taking the expectation over the data distribution gives $p-A$. This is not a different direction; it is the same gradient in another set of coordinates. Softmax maps logits to the simplex, and the chain rule turns $-A/p$ into $p-A$:

$$
\bigl(\operatorname{diag}(p)-pp^\top\bigr)\left(-\frac{A}{p}\right)=p-A.
$$

Backpropagation and forward inference share one energy surface, and both land in the same tangent space. The chapter uses one ruler throughout: the Euclidean norm on the tangent space.

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

What is described here is the local dynamics on the output distribution: the tangential motion of the last stretch of training on the simplex. Real training also involves non-convexity, stochasticity, finite samples, and parameterization; the local quadratic picture is a theorem about the last stretch, not a global theory of training. This boundary needs no second theory to state: the chapter uses only the Euclidean norm on the tangent space, and KL serves only as the energy surface, never as a distance.

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

The next section applies this distinction to scaling law: the industrial curve is a log-log straight line, while the orbit in this chapter is a semi-log straight line. They are not the same curve; but the “straightness” of both comes from the same multiplicative bookkeeping.

---

## 23.6 Reproducing That Curve

The previous section separated two coordinate systems: geometric convergence is a straight line on semi-log axes, while a power law is a straight line on log-log axes. The most famous curve in industry belongs to the latter. Its name is scaling law.

First, flatten the formula. Let $s$ denote scale — samples, parameters, or compute; choose one axis — and write a scaling law as

$$
L(s)=L_\infty+c\,s^{-\alpha}.
$$

$L_\infty$ is the floor, $c$ is a constant, and $\alpha$ is the exponent. Plot $L(s)-L_\infty$ on log-log axes and it is a straight line of slope $-\alpha$. On first seeing this line, one may assume it hides a secret; in fact, any multiplicative relation is a straight line on log-log paper. What needs explaining is not the “straightness” but three numbers: where the floor is, how large the constant is, and what determines the exponent.

This section does not explain the exponent of any particular laboratory. The exponent is decided jointly by data, architecture, optimizer, and training recipe; it is a question for empirical science. Here we do something smaller and more basic: in a toy system, compute this straight line from the simplex. The slope we get is $-1$. It does not impersonate an industrial exponent; it only shows how ordinary the “straightness” of this line can be.

### The data axis: one ordinary count

Let the true conditional distribution be

$$
q=(q_1,\dots,q_K).
$$

A system can estimate it only from $s$ samples, obtaining the empirical distribution $\hat q$. The energy surface from Chapter 12 gives

$$
D_{\mathrm{KL}}(q\|\hat q)\approx \frac12(\hat q-q)^\top H_T(\hat q-q),
$$

where $H_T$ is exactly the tangent Hessian from 23.1. The covariance of empirical frequencies is

$$
\mathbb E\bigl[(\hat q-q)(\hat q-q)^\top\bigr]
=\frac1s\bigl(\operatorname{diag}(q)-qq^\top\bigr).
$$

In the second-order approximation, taking the expectation of the quadratic form gives

$$
\mathbb E\,D_{\mathrm{KL}}(q\|\hat q)=\frac{K-1}{2s}+O(s^{-2}).
$$

:::details A one-line verification
Write $C=\operatorname{diag}(q)-qq^\top$. Every row of $C$ sums to zero, so it already lies in the tangent space and $P_TCP_T=C$. Hence

$$
\operatorname{tr}(H_TC)
=\operatorname{tr}\bigl(\operatorname{diag}(1/q)C\bigr)
=\sum_i(1-q_i)=K-1.
$$
:::

This formula uses no model structure. It says only: with $K$ answers and $s$ samples, the error falls as $1/s$. Plot $L(s)-H(q)$ on log-log axes and you get a straight line of slope $-1$; different $K$ shift the intercept but not the slope. In this toy the floor is exactly the data entropy $H(q)$; a real system's floor also contains irreducible noise, which is why we write $L_\infty$.

This is the second-order term. The exact KL is inflated by zero-frequency events at finite sample size; that is a separate boundary, and it changes the intercept at small $s$, not the slope.

Note that $H_T$ is not new. It is the matrix from 23.1, and 23.4 already showed that its tangent condition number is pinned near $K$ by the constraint. The same $H_T$ does two jobs: its two endpoints decide how fast convergence is ($\rho^*$), and its trace decides how high the finite-sample floor is ($(K-1)/(2s)$). One curve governs speed, the other governs the floor; they are not two theories, but the two sides of one sheet of paper.

### The dimension axis: the other half of the same Hessian

The floor formula still contains $K$, and $K$ is not free. The window in 23.5 says that the number of steps to reach accuracy $\varepsilon$ is approximately

$$
N_{\mathrm{steps}}\approx \frac{\kappa_T}{2}\ln\frac1\varepsilon.
$$

And the saturation limit in 23.4 says $\kappa_T\to K$. Thus, for symmetric anchors near a vertex, the number of steps is proportional to $K$: the more answers, the longer the road.

Put the two statements together:

> **Data buys the floor; dimension buys the road.** One more sample lowers the floor as $1/s$; one more answer lengthens the road as $K$. Scaling law usually reports the first sentence; the Yonglin Corollary measures the second. Both use the same $H_T$.

This is why the scalar in this chapter is not a replacement for scaling law, but its other half. The industrial curve measures “how low the floor can be pushed”; $\rho^*$ measures “how far you must walk to reach it.” Reporting only the floor invites the belief that scale has no cost; reporting only the road invites the belief that convergence speed is everything. Together, the two ledgers make one complete geometric bill.

![Figure 23.8: Reproducing the scaling-law line from the simplex — the data axis gives slope $-1$, the dimension axis gives different intercepts, and the semi-log orbits give the rate of the same Hessian](/figures/ch23_scaling_reproduction.png)

Panel (a) of Figure 23.8 is the data-axis scaling law: for $K=3,6,12$ answers, the second-order term of the excess KL is plotted on log-log axes, and all three lines land on slope $-1$, with intercepts shifted by $K$. Panel (b) is the other half of the same Hessian: for symmetric anchors near a vertex, $\kappa_T\approx K$, so the semi-log error curve has slope $\ln\rho^*$; the larger $K$ is, the closer $\rho^*$ is to 1, and the more steps are needed to reach the same accuracy. Neither curve is copied from an existing experiment; both are computed directly from $q\mapsto H_T$.

:::details A boundary that must be stated: the toy's $-1$ is not the industrial $\alpha$
The exponent of an industrial scaling law is usually not $1$; it varies with the data distribution, architecture, and training recipe. The derivation above shows only that a log-log straight line can be produced by an ordinary count; it does not predict the $\alpha$ of any specific system. Treating the toy's $-1$ as an industrial exponent pushes a local model too far.

What this chapter computes rigorously is: given $H_T$, the finite-sample floor term is $(K-1)/(2s)$ and the rate is $\rho^*$. Why a real system's $\alpha$ takes a particular value is a question for experiment. This chapter does not answer it on its behalf.
:::

:::info An incidental readout question: emergence
A word about “emergence.” An error curve that goes as $\rho^t$ is smooth; a sudden jump on a benchmark is usually a problem with the ruler — accuracy is a threshold metric, so a smooth curve crossing the threshold looks like a step. To separate a “readout jump” from a “dynamical change,” simply measure whether $\rho^*$ changes with scale: if $\rho^*$ is unchanged while the metric jumps, it is a readout; if $\rho^*$ jumps, the slope itself has changed. This criterion belongs here as a footnote to the scaling curve, and no more.
:::

:::details A boundary that must be admitted: this chapter does not define “intelligence”
Treating scaling law as a definition of intelligence is a mistake; treating it as a geometric bill that any intelligent system must pay is appropriate.

This chapter explains a class of learning dynamics: when a system minimizes KL and outputs over the probability simplex, what mathematics governs its error curve. It does not answer “what is understanding,” “what is consciousness,” or “why do some abilities appear only in sufficiently large models.” Those questions need other tools, and perhaps tools that do not yet exist.
:::

> **One sentence:** data decides the floor, dimension decides the road; geometry decides the necessity, engineering decides the numbers.

---

## 23.7 Conclusion: Intelligence, Emergence, and the Ordinary Equation

Back to the opening: three answers, initial error $0.648$, three steps to $0.1$, each step multiplying by about $0.45$.

That number is

$$
\rho^*=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T},
\qquad
\kappa_T=\frac{1+\rho^*}{1-\rho^*}.
$$

Together with the $H_T$ behind it, it determines exactly five things:

1. **Slope.** The semi-log error curve has slope $\ln\rho^*$.
2. **Steps.** The number of steps to reach $\varepsilon$ is exactly sandwiched between $\frac{\kappa_T-1}{2}\ln\frac1\varepsilon$ and $\frac{\kappa_T}{2}\ln\frac1\varepsilon$.
3. **Stability.** The stability window is $0<\eta<2/\theta_{\max}$, and the optimal step is $\eta^*=2/(\mu_T+\theta_{\max})$.
4. **The data-axis floor.** The finite-sample excess KL is $(K-1)/(2s)$; on log-log axes it is a straight line of slope $-1$.
5. **The emergence criterion.** If $\rho^*$ is unchanged while the metric jumps, it is a readout; if $\rho^*$ jumps, it is dynamics. This one is a footnote, and need not steal the stage.

The whole chain has only three links:

> one number, determined by two endpoints; two endpoints, determined by the tangent Hessian; the tangent Hessian, determined by the anchor and the simplex constraint.

[Chapter 12](/en/volume1/chapter12/) says: inference converges to the prior anchor $A$, not to the true answer $A^*$. This chapter adds the “how fast”:

> How fast is determined by one condition number; the condition number is determined by two endpoints of the tangent Hessian; the two endpoints are determined by the matrix. Beyond this, there is no second scalar.

Now we can say which part of scaling law is ordinary and which part is not. What is ordinary is its format: as long as the output lies on the simplex, the loss is recorded as KL, and samples are counted, $L(s)-L_\infty$ will be a straight line on log-log paper. The format comes from geometry; the numerical values from engineering. What is not ordinary is why it keeps going down: a real system's floor falls with parameters, data, and compute together; what the exponent is, and how long it can keep falling, is a question for empirical science, requiring experiments and time. Keeping these two things apart, we neither need to enshrine the curve nor to tear it down.

What this chapter reproduces is not any one company's curve, but the “straightness” of that curve. A straight line needs no extra assumption; it needs only one ordinary count. **Engineering decides the numbers; geometry decides the necessity.** This holds for any system that puts its output on the simplex, and for a three-answer toy as well — which is exactly what it means to call the result universal.

Intelligence and emergence look like miracles mainly when we stare at a single readout; put the readout back on the curve, and they are simply names that one ordinary equation takes at different scales.

> **One sentence:** the curve does not belong to any one company; it belongs to everyone who counts on the simplex.

---

## Open Questions

**How should the rate be estimated away from the anchor's neighborhood?** The $\rho^*$ here is the exact rate of the linearization near $A$. A real trajectory carries higher-order terms before it enters that small neighborhood; the first ratio in the opening table being noticeably smaller than $\rho^*$ is a shadow of this “head start.” Can the tangent spectrum alone give a finite-step, non-asymptotic error bound? If the large components of the initial error happen to lie in fast directions, the actual window will be shorter than $\rho^*$ predicts — can that gap be computed exactly? This is open.

**Can $\mu_T$ and $\theta_{\max}$ be measured directly on large models?** The spectra here are computed on explicit distributions with $K=3$ to $2048$. A real LLM has a huge vocabulary, and the Hessian cannot be stored explicitly. Can stochastic trace estimation, Lanczos, or implicit differentiation give reliable endpoint estimates? That would turn the diagnosis of whether the slope changes with scale from theory into experiment.

**When does $\rho^*$ change with scale?** This chapter gives the criterion but not a model for $\rho^*(N)$. If the appearance of a slow mode has a critical scale, what determines that critical point? This is what the other half of the scaling curve really needs to answer.

**What lies between $(K-1)/(2s)$ and an industrial exponent?** The toy here gives only the second-order term on the data axis. A real system's scaling exponent is also shaped by architecture, data distribution, and optimizer. Which deviations can be explained by the spectrum of $H_T$, and which must be left to experiment? This is where the “ordinary formula” meets the “empirical curve.”

---

## Exercises

**★ Warm-up**

1. Using the opening table, verify at which step the error ratio settles near $\rho^*$. Why is the first ratio noticeably smaller?
2. For $K=3$ and $q=(1/3,1/3,1/3)$, verify $H_T=3P_T$, write down all eigenvalues, and compute $\rho^*$.
3. For $q=(0.70,0.20,0.10)$, use the secular equation to compute $\mu_T,\theta_{\max},\kappa_T$, then $\eta^*,\rho^*$, and $N(0.1)$.
4. For $K=3$ and $q=(0.70,0.20,0.10)$, estimate the excess KL at $s=10^4$ using $(K-1)/(2s)$; then sample from a multinomial distribution and see how far the second-order expansion is from the actual value.

**★★ Derivations**

1. Prove Lemma 23.1: the convex function $g(\lambda)=|1-\eta\lambda|$ attains its maximum on an interval at an endpoint.
2. Prove $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i$.
3. Prove Theorem 23.2: $\eta^*=2/(\mu_T+\theta_{\max})$ is the unique minimizer, $\rho^*=(\kappa_T-1)/(\kappa_T+1)$, and the two endpoint branches are equal at the optimum.
4. Prove the two bounds in Theorem 23.3: $\frac{2x}{2+x}\le\ln(1+x)\le x$.
5. Prove $\operatorname{tr}\bigl(H_T(\operatorname{diag}(q)-qq^\top)\bigr)=K-1$, and use it to obtain $\mathbb E\,D_{\mathrm{KL}}(q\|\hat q)=(K-1)/(2s)+O(s^{-2})$.

**★★★ Challenge**

1. For $K=3$, write the secular equation as $3s_3\lambda^2-2s_2\lambda+1=0$, and verify that it matches the quadratic in [Chapter 12](/en/volume1/chapter12/).
2. Design an experiment that uses only the change of $\rho^*$ with scale to distinguish a “readout effect” from a “spectral bifurcation.” State what you would see in each case and how to rule out confounders.
3. Try to construct a counterexample showing that away from the anchor's neighborhood, where second-order terms cannot be ignored, the Euclidean linearized rate $\rho^*$ is no longer the exact rate; then state what extra condition would restore an exact rate formula.

---

## Appendix: Core Computation Script

The code below does two things: given an anchor $q$, it computes the two endpoints of the tangent spectrum via the secular equation, then the exact rate, the optimal step, and the window; it also computes the second-order term of the data-axis scaling line from 23.6. It trains no model.

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


def scaling_floor(K, s):
    """Leading finite-sample excess KL: (K-1)/(2s)."""
    return (K - 1.0) / (2.0 * s)


print([(K, scaling_floor(K, 10_000)) for K in (3, 6, 12)])
# [(3, 0.0001), (6, 0.00025), (12, 0.00055)]
```

---

**References**

- Banach, Stefan (1922). “Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.” *Fundamenta Mathematicae* 3(1), 133–181. DOI: 10.4064/fm-3-1-133-181. — the original paper on the contraction mapping principle
- Schaeffer, Rylan; Miranda, Brando; and Koyejo, Sanmi (2023). “Are Emergent Abilities of Large Language Models a Mirage?” arXiv:2304.15004. — emergent abilities as a metric artifact
- Wei, Jason; Tay, Yi; Bommasani, Rishi; Raffel, Colin; Zoph, Barret; Borgeaud, Sebastian; Yogatama, Dani; Bosma, Maarten; Zhou, Denny; Metzler, Donald; Chi, Ed H.; Hashimoto, Tatsunori; Vinyals, Oriol; Liang, Percy; Dean, Jeff; and Fedus, William (2022). “Emergent Abilities of Large Language Models.” arXiv:2206.07682. — a systematic description of emergent phenomena
- Nesterov, Yurii (2018). *Lectures on Convex Optimization*, 2nd edition. Springer Optimization and Its Applications 137. Springer. DOI: 10.1007/978-3-319-91578-4. — strong convexity, smoothness, step size, and condition number
- Boyd, Stephen; and Vandenberghe, Lieven (2004). *Convex Optimization*. Cambridge University Press. DOI: 10.1017/CBO9780511804441. — nonexpansiveness of projection and convergence rates
- Cover, Thomas M.; and Thomas, Joy A. (2006). *Elements of Information Theory*, 2nd edition. Wiley. — the local quadratic expansion of KL and the covariance of empirical distributions
- [Chapter 12](/en/volume1/chapter12/), Yonglin's Formula — the global version of the Yonglin theorem, tangent spectral bounds, and direct evidence

---

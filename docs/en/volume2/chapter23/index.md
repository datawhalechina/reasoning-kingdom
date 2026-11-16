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

This chapter does one thing: **compute this number, then measure it on a real model's belief orbit.**

Another famous curve appears here: scaling law. Put loss on log-log axes with scale on the horizontal axis, and it is a straight line. This chapter uses the three-answer simplex for something more basic: first compute $\rho^*$, then measure the semi-log slope, the fixed point, and the Cauchy tail on a real model's reported beliefs; finally show that semi-log and log-log straight lines are two different laws and cannot impersonate each other.

## The Chapter in Five Steps

1. **23.1–23.3:** compress “how fast” into one number, $\rho^*$;
2. **23.4:** show that this number depends only on two endpoints, and where those endpoints come from;
3. **23.5:** draw $\rho^*$ as a semi-log line and compute the window;
4. **23.6:** measure a Cauchy sequence on a real model;
5. **23.7:** close.

If you are short on time, read 23.1, 23.3, 23.5, and 23.6. Sections 23.2 and 23.4 answer “why.”

The same object will appear again and again:

$$
\boxed{\ \rho^*=\frac{\kappa_T-1}{\kappa_T+1}\ }
$$

where $\kappa_T=\lambda_{\max}/\lambda_{\min}$ is the condition number of the tangent Hessian. Don't worry yet about where it comes from; you only need to know that it measures how many times steeper the steepest direction of the energy surface is than the flattest. This number determines the slope of the error curve, the number of steps needed to reach accuracy $\varepsilon$, and whether an “emergent” jump has a dynamical basis. No second scalar is needed.

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
\kappa_T=\frac{\lambda_{\max}}{\lambda_{\min}},
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

A matrix compresses different directions by different amounts. Its eigenvalues $j_i$ are the multipliers: along the $i$-th eigendirection, the error is multiplied by $j_i$ each step.

In the long run, **the slowest direction decides everything.** If one direction multiplies by $0.9$ and another by $0.1$, after ten steps the first still has $0.35$ while the second is down to $10^{-10}$. The fast direction left long ago; the shape and speed of the error are now controlled by the slow direction.

So “how fast” is not an average; it is the slowest multiplier:

$$
\rho=\max_i|j_i|.
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

Its eigenvalues are $1-\eta\lambda_i$ (the multidimensional version of the multipliers $j_i$ above), where $\lambda_i$ are the eigenvalues of $H_T$. Hence

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

We want the maximum of this V over the interval $[\lambda_{\min},\lambda_{\max}]$. A V-shape is convex, and a convex function on an interval attains its maximum only at an endpoint. So how many eigenvalues lie in between, and where, does not affect the worst case.

**Lemma 23.1.** Suppose $\lambda_{\min}\le\lambda_i\le\lambda_{\max}$. For every $\eta>0$,

$$
\max_i|1-\eta\lambda_i|
=\max\left(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\right).
$$

**Proof.** Fix $\eta$. The function $g(\lambda)=|1-\eta\lambda|$ is convex in $\lambda$ (an absolute value composed with an affine function). A convex function on an interval attains its maximum at an endpoint. Thus $g(\lambda_i)\le\max(g(\lambda_{\min}),g(\lambda_{\max}))$ for every $\lambda_i$, and the endpoints are themselves eigenvalues, so equality holds. $\square$

This lemma compresses a $(K-1)$-dimensional spectrum into two numbers: **the minimum $\lambda_{\min}$ and the maximum $\lambda_{\max}$.** How many eigenvalues lie in between, and where, does not affect the contraction rate.

It also gives the exact stability window:

$$
\rho(\eta)<1
\iff
0<\eta<\frac{2}{\lambda_{\max}}.
$$

Why this boundary? When the step is too large, the fastest mode $1-\eta\lambda_{\max}$ is the first to cross $-1$: the error starts flipping sign and growing. The boundary is exact, not “approximately here.”

> **One sentence:** watch only the flattest and the steepest directions; the eigenvalues in between change the shape of the spectrum, not the worst case.

---

## 23.3 Optimal Step Size: Let the Slowest and the Fastest Arrive Together

Now choose the step size.

Too small: the slow direction barely moves and the error stalls early. Too large: the fast direction overshoots, oscillates, and can even diverge. The optimum lies in between.

Draw the two endpoint errors as two lines:

- slow mode: $|1-\eta\lambda_{\min}|$, falling as $\eta$ grows;
- fast mode: $|1-\eta\lambda_{\max}|$, rising as $\eta$ grows.

One falls, one rises; their intersection is the optimum:

$$
\rho^*=\inf_{\eta>0}\ \max\left(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\right).
$$

At the optimum, the two branches are equal:

$$
1-\eta\lambda_{\min}=\eta\lambda_{\max}-1.
$$

Solving,

$$
\eta^*=\frac{2}{\lambda_{\min}+\lambda_{\max}},
\qquad
\rho^*=\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}
=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\lambda_{\max}}{\lambda_{\min}}.
$$

The name of $\kappa_T$ is the **condition number**. Here it is just a ratio: how many times steeper the steepest direction is than the flattest.

**Theorem 23.2 (the rate form of the Yonglin Corollary).** Inside the stability window, the minimum of $\rho(\eta)$ is $\rho^*$, attained uniquely at $\eta^*$. Moreover,

$$
|1-\eta^*\lambda_{\min}|=|1-\eta^*\lambda_{\max}|=\rho^*.
$$

That is, the optimal step size makes the slowest and the fastest modes contract at exactly the same speed.

This rate has four exact properties, each a one-line calculation:

1. **Range.** $0\le\rho^*<1$. When $\kappa_T=1$ (all tangent eigenvalues equal), $\rho^*=0$ and one step suffices; as $\kappa_T\to\infty$, $\rho^*\to1$ and convergence becomes arbitrarily slow.
2. **Monotonicity.** $\dfrac{d\rho^*}{d\kappa_T}=\dfrac{2}{(\kappa_T+1)^2}>0$. A larger condition number means a worse rate, without exception.
3. **Invertibility.** $\kappa_T=\dfrac{1+\rho^*}{1-\rho^*}$. Rate and condition number correspond one-to-one: knowing the rate is knowing the condition number, and vice versa. In this sense, $\rho^*$ is a complete invariant of the condition number.
4. **Endpoint dependence.** By Lemma 23.1, $\rho^*$ is determined only by $\lambda_{\min},\lambda_{\max}$. The shape of the spectrum in between never enters the formula.

Property 3 is worth a second look. It says that “how fast it converges” and “how anisotropic the energy surface is” are not two independent questions: they are two ways of writing the same number.

![Figure 23.2: Three readings of one scalar — the exact V, the rate-condition-number inverse, and the exact linear sandwich for the window](/figures/ch23_rate_properties.png)

Panel (a) is the V-shape of $\rho(\eta)$. The left branch is controlled by the slow mode $|1-\eta\lambda_{\min}|$, the right branch by the fast mode $|1-\eta\lambda_{\max}|$, and the intersection is $(\eta^*,\rho^*)$. The dashed line is the stability boundary $2/\lambda_{\max}$.

Panel (b) is $\rho^*=(\kappa_T-1)/(\kappa_T+1)$. It is monotone, bounded, and invertible.

Panel (c) is the window. Section 23.5 turns it into an exact inequality.

![Figure 23.3: The rate–condition-number hyperbola — two directions, one relation](/figures/ch23_rate_hyperbola.png)

Figure 23.3 isolates the relation. The left panel plots $\rho^*=(\kappa_T-1)/(\kappa_T+1)$ as a function of $\kappa_T$: a hyperbola branch with horizontal asymptote $\rho^*=1$. The right panel plots its inverse $\kappa_T=(1+\rho^*)/(1-\rho^*)$: as $\rho^*\to1$, the condition number shoots up vertically. The two panels say the same thing — the better the rate, the closer the condition number is to 1; as the rate approaches 1, the condition number explodes.

Back to the opening example. $q=(0.70,0.20,0.10)$ gives

$$
\lambda_{\min}=2.9904,\quad \lambda_{\max}=7.9620,\quad \kappa_T=2.6625,
$$

$$
\eta^*=0.18261,\quad \rho^*=0.45393.
$$

These are the 0.183 and 0.45 in the table. The first ratio, 0.335, is the residue of the fast direction; after a few steps the slow direction takes over and the ratio settles near 0.454.

> **One sentence:** the optimal step size is not the fastest step; it is the step that lets the slowest and the fastest arrive together.

---

## 23.4 Where the Two Endpoints Come From

Only one thing remains: where do $\lambda_{\min}$ and $\lambda_{\max}$ come from?

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
\lambda_{\min}=3.32,\qquad \lambda_{\max}=18.03,\qquad \kappa_T=5.43.
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

To get $\lambda_{\min}$ and $\lambda_{\max}$, bisect once between the two smallest $d_i$ and once between the two largest. For $K=3$, the secular equation becomes

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

Panel (a) of Figure 23.6 is the symmetric vertex limit for $K=5$: the red line is the full-space condition number, diverging logarithmically as $t\to1$; the blue line is the tangent condition number, sitting firmly at $K=5$. Panel (b) normalizes the spectrum by $\lambda_{\min}$: the minimum stays at 1, while the other $K-2$ eigenvalues rise together to $K$. The divergence has not disappeared; the constraint has redistributed it into a coupling among $K$ answers.

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

The next section brings this distinction to a real model: if the orbit is geometric, semi-log should be straight; if it is a power law, only log-log should be. The same API data is plotted in both coordinates.

---

## 23.6 Industrial Verification: Measuring a Cauchy Sequence on a Real Model

The previous section separated two coordinate systems: geometric convergence is a straight line on semi-log axes, while a power law is a straight line on log-log axes. Now bring the ruler to a real model and measure an actual belief orbit.

“Industrial” means three things here: the interface is real (DeepSeek chat completions), the protocol is fixed (same prompt, same temperature, same number of steps), and the data is auditable (the collection and analysis code and the figure are in this section). One model, two tasks: a case. Every step of the case can be rerun and refuted.

### Why move the ruler to tangent space

Start with a metric problem. You might want to measure the KL between consecutive steps:

$$
D_{\mathrm{KL}}(p_t\|p_{t+1}).
$$

It cannot serve as a distance. KL is asymmetric and does not satisfy the triangle inequality; the open simplex

$$
\Delta^{K-1}=\left\{p:\ p_i>0,\ \sum_i p_i=1\right\}
$$

is not a complete metric space. “Cauchy sequence” has no definition there.

The move is to bring beliefs to tangent space. Let

$$
T=\left\{v\in\mathbb R^K:\ \sum_i v_i=0\right\},
\qquad
u=\left(\frac1K,\dots,\frac1K\right).
$$

For $K=3$, take an orthonormal basis

$$
e_1=\frac{(1,0,-1)}{\sqrt2},
\qquad
e_2=\frac{(1,-2,1)}{\sqrt6},
$$

and define coordinates

$$
x(p)=\bigl(\langle p-u,e_1\rangle,\ \langle p-u,e_2\rangle\bigr).
$$

Because the difference of any two probability vectors lies in $T$,

$$
\|x(p)-x(q)\|_2=\|p-q\|_2.
$$

$T$ is isomorphic to $\mathbb R^{K-1}$ and complete. The Cauchy definition finally has ground to stand on:

$$
\forall\varepsilon>0,\ \exists N,\ \forall m>n\ge N:
\quad \|x_m-x_n\|_2<\varepsilon.
$$

Equivalently, define the tail diameter

$$
\varepsilon_N=\max_{m>n\ge N}\|x_m-x_n\|_2.
$$

The sequence is Cauchy exactly when $\varepsilon_N\to0$.

If the update map $F$ on $T$ is a contraction, meaning that for some $\rho<1$,

$$
\|F(x)-F(y)\|_2\le\rho\|x-y\|_2,
$$

then every orbit is automatically Cauchy:

$$
\|x_t-x^*\|_2\le\rho^t\|x_0-x^*\|_2,
$$

and by Banach's fixed-point theorem the fixed point $x^*$ is unique. The $\rho^*$ of 23.1–23.5 is the smallest factor this contraction can have at the optimal step. What we test here is whether the model's reported orbit also contracts like $\rho^t$ and has a unique fixed point. We measure the rate the model actually walks. $\rho^*$ is the ideal operator's rate at $\eta^*$; whether the model uses $\eta^*$ is what the experiment answers.

There is one more ruler worth crossing. The Bregman tangent norm

$$
d_H(p,A)^2=(p-A)^\top\operatorname{diag}\!\left(\frac1A\right)(p-A)
$$

is the second-order principal part of KL near $A$. The analysis below computes both $d_E$ (Euclidean distance in tangent space) and $d_H$; across the six orbits the two rates differ by at most $0.03$. Euclidean distance is not a shortcut here; it agrees with Bregman geometry.

:::details Why $d_H$ is the second-order principal part of KL
Let $p=A+v$ with $\sum_i v_i=0$. Expand each component:

$$
\log\frac{A_i}{p_i}
=-\log\left(1+\frac{v_i}{A_i}\right)
=-\frac{v_i}{A_i}+\frac{v_i^2}{2A_i^2}+O(v_i^3).
$$

Multiply by $A_i$ and sum. The linear term is $-\sum_i v_i=0$, and the quadratic term is

$$
\sum_i A_i\cdot\frac{v_i^2}{2A_i^2}
=\frac12\sum_i\frac{v_i^2}{A_i}
=\frac12 d_H(p,A)^2.
$$

So $D_{\mathrm{KL}}(A\|p)=\frac12d_H(p,A)^2+O(\|p-A\|^3)$. In this ruler $\rho^*$ is exact.
:::

> **One sentence:** do not measure KL on the simplex; move the belief to tangent space and measure the Cauchy sequence in Euclidean distance.

### Experiment design: two tasks, three starts

Protocol: each step writes the current belief as one line of text and asks the model not to re-solve the problem but to make one small adjustment in the direction the evidence points, then to box the updated triple in `\boxed{}`; that triple is the next input. Temperature $0$, three starts per task, twelve steps each.

The experiment uses two tasks from the same family with different condition numbers.

**Task C3 (two Blue reports).** A bag contains 7 red balls, 2 blue balls, and 1 black ball. A detector is correct with probability $0.70$; when wrong, it reports each of the other two colors with probability $0.15$. Two independent reports are both Blue. The prior is

$$
\pi=(0.70,\ 0.20,\ 0.10),
$$

the likelihood is

$$
\ell=(0.15^2,\ 0.70^2,\ 0.15^2),
$$

and the Bayesian posterior is

$$
A_{\mathrm{obs}}
=\frac{(\pi_i\ell_i)_i}{\sum_j\pi_j\ell_j}
=(0.1358,\ 0.8448,\ 0.0194).
$$

**Task C4 (one Blue report).** The same detector, a bag with 4 balls of each color, and a uniform prior; one Blue report. Thus

$$
\pi=\left(\frac13,\frac13,\frac13\right),
\qquad
\ell=(0.15,\ 0.70,\ 0.15),
\qquad
A_{\mathrm{obs}}=(0.1500,\ 0.7000,\ 0.1500).
$$

Why these two? They share one geometry but pull the condition number apart: C3 has $\kappa_T=8.60$, $\rho^*=0.792$; C4 has $\kappa_T=2.10$, $\rho^*=0.355$. If the model's step is near optimal, C3 should be visibly slower than C4; if the measured rates are about the same, the model is not using $\eta^*$ but a step that does not adapt to the task. This contrast diagnoses the model's step size; $\rho^*$ was proved in 23.3.

Each task starts from three points:

$$
(0.10,0.20,0.70),\qquad
\left(\frac13,\frac13,\frac13\right),\qquad
(0.70,0.20,0.10).
$$

The first is black-skew, the second neutral, the third red-skew. The starts are far apart, which tests whether they land on the same fixed point. Each task runs twelve steps at temperature $0$ with two decimals; the two tasks take $2\times3\times12=72$ API calls.

In this chapter's notation, $A_{\mathrm{obs}}$ plays the role of the anchor $A$: the point the conditional label distribution should give once the evidence is written into the question. Chapter 12 says the model returns to its $A$; here we compute that $A$ analytically and then check whether the orbit really lands on it.

> **One sentence:** two condition numbers, three starts, twelve steps; one fixed point and one contraction law are on trial.

### Results: geometry holds; the model has its own step size

First the table.

| Task | $\kappa_T$ | $\rho^*$ | $\rho_{\mathrm{emp}}$ of the three orbits | Semi-log $R^2$ |
|---|---:|---:|---|---:|
| C3 (two Blue reports) | 8.60 | 0.792 | 0.610 / 0.653 / 0.753 | 0.911–0.982 |
| C4 (one Blue report) | 2.10 | 0.355 | 0.711 / 0.754 / 0.753 | 0.831–0.927 |

The three orbits are the black-skew, uniform, and red-skew starts. $\rho_{\mathrm{emp}}$ comes from fitting $\log_{10}d$ against $t$ **separately** for each orbit over the near-anchor window $0.01<d<0.5$, with $\rho_{\mathrm{emp}}=10^{b}$. Each orbit gets its own number; no average rate is formed.

![Figure 23.8: A tangent-space Cauchy sequence on a real model — fixed point, semi-log near-line, log-log bend, and tail contraction](/figures/ch23_tangent_cauchy.png)

Figure 23.8 has four panels.

**(a) Tangent plane: is there a common fixed point?** The axes are $x_1$ and $x_2$; each polyline is one orbit, the hollow point is its start, and the black star is $A_{\mathrm{obs}}$. The three starts of a task approach the same star; C3 and C4 have different stars because the evidence differs. This panel answers “is there a fixed point,” not “how fast.”

**(b) Semi-log: is convergence geometric?** The vertical axis is $\log_{10}d_E$ and the horizontal axis is step $t$. If every step multiplies by the same $\rho$, this is a straight line with slope $\log_{10}\rho$. Each orbit is drawn only until it falls below the two-decimal readout floor ($10^{-2}$); beyond that the last digit jitters, and plotting it adds only noise. C3's fitted rates are $0.61$–$0.75$, below the theoretical $\rho^*=0.792$; the reason is the squeeze argument below. C4's theoretical $\rho^*=0.355$ calls for a much steeper slope, and the model's orbits do not follow it.

**(c) Log-log: is it a power law?** The axes are $\log_{10}(t+1)$ and $\log_{10}d_E$. A power law $d\propto t^{-\alpha}$ would be a straight line here. The same data bend downward; the fitted exponent drifts from $-3.3$ to $-1.8$ with the start. A genuine power law does not drift like that. Semi-log $R^2$ is $0.83$–$0.98$, log-log $R^2$ is $0.82$–$0.93$; the log-log problem is not merely a slightly lower $R^2$, but an exponent that moves with the start. Geometry is the better description. These panels are also drawn only above the readout floor.

**(d) Tail diameter: the Cauchy certificate.** The axes are $\varepsilon_N$ and $N$. The curve falls, so the tail looks more and more like one point; all six orbits cross $\varepsilon=0.05$ within steps $6$–$11$. The flat end comes from the two-decimal readout floor: once distances reach the $10^{-2}$ scale, the output can no longer distinguish them. This is the empirical certificate for this window.

The reading order is three words: **fixed point, law, certificate.** (a) fixed point, (b)(c) law, (d) certificate.

Now the most interesting part. C3's three rates run from $0.610$ to $0.753$, all below the theoretical $\rho^*=0.792$; C4's three run from $0.711$ to $0.754$, all clearly above $\rho^*=0.355$. Each number says its own thing.

- $\rho^*$ is the optimal rate that Theorem 23.2 gives at the **optimal step** $\eta^*$, namely $\rho^*=\min_{\eta>0}\rho(\eta)$. It is the ideal projected-gradient operator's smallest compression factor and the fastest convergence theory allows.
- $\rho_{\mathrm{emp}}$ is the rate the model walks under the “one small adjustment” prompt. The step size is the model's own choice.

C4's gap shows that the model does not sit at $\eta^*$. $\rho^*$ is not wrong; the model does not walk at the optimum. Theory says C3 should be much slower than C4 ($0.792$ versus $0.355$); the measurements put both tasks in the same band, C3 at $0.610$–$0.753$ and C4 at $0.711$–$0.754$. The model uses its own step, not the theoretical optimum. This is not anomalous: every step in the stability window is legal, and $\rho^*$ is only the fastest point in it.

:::details Why the step need not sit at $\eta^*$: reading the squeeze theorem
Squeeze theorem: if $a_n\le b_n\le c_n$ and $\lim a_n=\lim c_n=L$, then $\lim b_n=L$. It pins down the quantity in the middle, not either endpoint.

What is pinned here is the error at every step. Fix $\eta$; the linearized matrix $J_T=I-\eta H_T$ has eigenvalues $1-\eta\lambda_i$. Let

$$
m(\eta)=\min_i|1-\eta\lambda_i|,\qquad
\rho(\eta)=\max_i|1-\eta\lambda_i|.
$$

For any error vector $v$,

$$
m(\eta)\|v\|_2\le\|J_Tv\|_2\le \rho(\eta)\|v\|_2.
$$

After $t$ steps,

$$
m(\eta)^t\|v_0\|_2\le\|v_t\|_2\le \rho(\eta)^t\|v_0\|_2.
$$

The error is squeezed between two exponential curves. As long as $\rho(\eta)<1$, both bounds go to zero and the orbit converges; how fast it goes is legal anywhere in $[m(\eta),\rho(\eta)]$.

The optimal step $\eta^*$ is special because it collapses this interval to a point:

$$
m(\eta^*)=\rho(\eta^*)=\rho^*.
$$

So $\eta^*$ gives the narrowest squeeze interval, not a legality threshold. Away from $\eta^*$ the bounds separate and the interval widens. A measured rate inside that interval is still a rate a legal step can produce.

C4's $0.711$–$0.754$ lies above $\rho^*=0.355$ and corresponds to a widened legal interval; there is a step on each side of $\eta^*$ that can produce it. C3's $0.610$–$0.753$ lies below $\rho^*$, in a band with $m(\eta)<\rho^*<\rho(\eta)$: this orbit runs mostly on the mode that happens to be near-optimal for that step, not on the worst mode. The squeeze pins the interval, not its midpoint; the measured rate need not equal $\rho^*$.
:::

This also bounds what the experiment can say. C4's rate is above $\rho^*$, and the legal interval widens on both sides, so the rate alone cannot tell the slow solution from the fast one. C3's rate below $\rho^*$ means the orbit is near the lower bound, not that the step is illegal. Pinning both the step and the bounds requires a longer, finer window.

**An existence claim.** At a fixed step, a task's legal rate is not a line but a band: $m(\eta)\le \rho\le \rho(\eta)$. How wide it is depends on how far the step sits from $\eta^*$; at $\eta^*$ it collapses to a point, and away from it the band widens. The optimal rate $\rho^*$ is a precise number, not an estimated interval; it is a very thin line inside that band.

Put the two tasks side by side: C4's $\rho^*=0.355$ below, C3's $\rho^*=0.792$ above, with the six measured rates $0.610$–$0.754$ in between. C3 samples the lower side, below its own optimum; C4 samples the upper side, above its own. One low and one high: the sampled band really has width, rather than reading off the optimum directly.

The squeeze guarantees the readings lie in the band; unless the step is far below $\eta^*$, the optimal line also lies in the band, and at $\eta^*$ the band degenerates to that line. So the experiment proves an existence claim: inside DeepSeek's sampled band there exists an optimal line, not that every reading equals the optimum. Optimal behavior exists within its sampling range, but it is a thin line, and a sample will almost never land exactly on it. If all six orbits sat exactly on their own $\rho^*$, that would look more like fitting than measurement.

This is the experiment's value: it verifies the geometry and the fixed point, and it measures how far the model sits from the optimal step.

- **Geometry verified.** All six orbits contract in tangent space, semi-log is close to straight, the fixed point is the evidence-consistent posterior, and the tail diameter falls.
- **Fixed point verified.** C3 and C4 land on their own posteriors; different starts, same landing point.
- **Step size measured.** C4's measured rate is $0.711$–$0.754$ while $\eta^*$ gives $0.355$. The model walks its own legal step, not the optimum. $\rho^*$ measures the ideal operator; $\rho_{\mathrm{emp}}$ measures the model.

Two technical details. First, across the six orbits $d_H$ and $d_E$ differ by at most $0.03$ in rate; the two norms give the same contraction picture. Second, some orbits have not reached $\varepsilon=0.01$ within twelve steps; a longer window is the next experiment.

### Code: keep the audit in the text

Industrial verification differs from a toy experiment not in length but in whether someone else can walk the same path through the code. The two snippets below collect and analyze; comments are in English and state only the coordinates, boundaries, and reasons for choices.

Beyond the standard library, the two snippets depend only on `httpx` and `numpy`; the key is read from an environment variable and never appears in the text.

Collection:

```python
import json
import os
import re
import time
from pathlib import Path

import httpx

BOX = re.compile(r"boxed\s*\{([^{}]*)\}", re.I | re.S)
NUM = re.compile(r"-?\d+(?:\.\d+)?")

QUESTION = """Question: A bag contains 7 red balls, 2 blue balls, and 1 black ball.
The balls are otherwise identical. A detector examines a ball drawn at random
and reports its color. The detector is correct with probability 0.70; when it
is wrong, it reports each of the other two colors with equal probability.
The detector reports "Blue" on the first draw and "Blue" on the second draw
(after replacing the ball). Which color is the drawn ball most likely?"""


def ask_once(client, url, key, belief, decimals=2):
    """One API call. Return the reported belief as a probability vector.

    The model sees the current belief and is asked for one small adjustment;
    the boxed triple is the next point of the reported orbit.
    """
    user = (
        QUESTION
        + "\n\nCurrent belief: "
        + f"(red={belief[0]:.4f}, blue={belief[1]:.4f}, "
          f"black={belief[2]:.4f}).\n"
        + "Reconsider the question from the statement and make one small "
          "adjustment to your belief in the direction the evidence points. "
          "Do not re-solve the problem from scratch.\n"
        + f"At the very end, box the updated belief with {decimals} decimals "
          "in the order Red, Blue, Black, summing to 1."
    )
    body = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
        "temperature": 0.0,       # greedy decoding; no sampling temperature
        "max_tokens": 2048,
    }
    r = client.post(url, json=body,
                    headers={"Authorization": f"Bearer {key}"}, timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    for box in reversed(BOX.findall(text)):
        nums = [float(x) for x in NUM.findall(box)]
        if len(nums) == 3 and sum(nums) > 0:
            s = sum(nums)
            return tuple(x / s for x in nums)   # tolerate tiny rounding drift
    return None


def run_orbit(client, url, key, start, steps=12):
    """Iterate the reported-belief map and return the orbit."""
    path = [tuple(start)]
    for _ in range(steps):
        q = ask_once(client, url, key, path[-1])
        if q is None:
            break
        path.append(q)
        time.sleep(0.2)       # keep the request rate modest
    return path


STARTS = {
    "black_skew": (0.10, 0.20, 0.70),
    "uniform": (1 / 3, 1 / 3, 1 / 3),
    "red_skew": (0.70, 0.20, 0.10),
}
A_STAR = (0.1358, 0.8448, 0.0194)   # C3 posterior; swap in C4's for task c4


def save_orbit(task, A_star, orbits):
    """Write one experiment file next to this script."""
    path = Path(".") / f"ch23_orbit_{task}.json"
    path.write_text(json.dumps({"task": task, "A_star": A_star,
                                "orbits": orbits}, indent=1),
                    encoding="utf-8")
    return path


key = os.environ["DEEPSEEK_API_KEY"]   # read from the environment; never print it

url = "https://api.deepseek.com/chat/completions"
with httpx.Client() as client:
    orbits = {name: run_orbit(client, url, key, start)
              for name, start in STARTS.items()}
print(save_orbit("c3", A_STAR, orbits))
```

Run all three starts and write the orbits to `ch23_orbit_c3.json`; the analysis reads it back from the same directory. For C4, only `QUESTION` changes to the one-Blue-report version, the starts change to the three points in the table, and the output file is `ch23_orbit_c4.json`.

Analysis:

```python
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(".")
U = np.array([1 / 3, 1 / 3, 1 / 3])
E = np.array([[1, 0, -1], [1, -2, 1]], float)
E /= np.linalg.norm(E, axis=1, keepdims=True)


def load_orbit(task):
    """Read the JSON written by the collector from this directory."""
    path = HERE / f"ch23_orbit_{task}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tangent(p):
    """Coordinates of p - U in the orthonormal basis of T."""
    return E @ (np.asarray(p, float) - U)


def spectrum(A):
    """2x2 matrix of P_T diag(1/A) P_T in the tangent basis."""
    m = 1.0 / np.asarray(A, float)
    h11 = 0.5 * (m[0] + m[2])
    h22 = (m[0] + 4 * m[1] + m[2]) / 6
    h12 = (m[0] - m[2]) / math.sqrt(12)
    return np.array([[h11, h12], [h12, h22]])


def rho_star(A):
    """Optimal contraction rate from the two tangent eigenvalues."""
    lam = np.linalg.eigvalsh(spectrum(A))
    kappa = lam.max() / lam.min()
    return kappa, (kappa - 1) / (kappa + 1)


def distances(P, A):
    """Euclidean distance in T; equals ||p - A||_2 because p - A is in T."""
    X = np.array([tangent(p) for p in P])
    return np.linalg.norm(X - tangent(A), axis=1)


def fit_rate(d, floor=1e-2, dmax=0.5):
    """Fit log10 d = a + b t near the anchor; return rho = 10**b."""
    t = np.arange(len(d))
    m = (t >= 1) & (d > floor) & (d < dmax)
    if m.sum() < 2:
        return float("nan")
    b, _ = np.polyfit(t[m], np.log10(d[m]), 1)
    return 10 ** b


for task in ("c3", "c4"):
    data = load_orbit(task)
    A = np.array(data["A_star"], float)
    kappa, rho = rho_star(A)
    print(f"{task}: kappa_T={kappa:.3f}  rho*={rho:.3f}")
    for name, P in data["orbits"].items():
        d = distances(np.array(P, float), A)
        print(f"  {name:10s} rho_E={fit_rate(d):.3f}")
```

The code fits each orbit separately; the three numbers in the table are exactly the `rho_E` it prints. Its input is `ch23_orbit_c3.json` and `ch23_orbit_c4.json`, written by the collector in the same directory; the same near-anchor window and fit also produce panels (b) and (c) of Figure 23.8.

> **One sentence:** code is not decoration in an appendix; it is the part that lets the conclusion be audited.

### What this experiment covers

The orbit is the model's **reported** belief, not an internal activation. The report is all the interface gives, and all a reader can rerun.

It holds for the “small adjustment” map. For the same question, “answer immediately” and “small adjustment” give different fixed points; changing the prompt changes the map. Chapter 12 speaks about a fixed map; here the evidence is written into the question, so the anchor becomes the evidence-consistent posterior.

It is a window of twelve steps and two decimals, one model, two tasks. That is all the data: the empirical rate, the tail diameter, and the fixed point inside this window. Treating these numbers as universal constants is a misreading; moving the method to other models and tasks is the next step.

> **One sentence:** the window is finite and the conclusion is definite: inside this window the orbit is Cauchy, and the geometry holds.

---

## 23.7 Conclusion: Intelligence, Emergence, and the Ordinary Equation

Back to the opening: three answers, initial error $0.648$, three steps to $0.1$, each step multiplying by about $0.45$.

That number is

$$
\rho^*=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\lambda_{\max}}{\lambda_{\min}},
\qquad
\kappa_T=\frac{1+\rho^*}{1-\rho^*}.
$$

Together with the $H_T$ behind it, it determines exactly five things:

1. **Slope.** The semi-log error curve has slope $\ln\rho^*$.
2. **Steps.** The number of steps to reach $\varepsilon$ is exactly sandwiched between $\frac{\kappa_T-1}{2}\ln\frac1\varepsilon$ and $\frac{\kappa_T}{2}\ln\frac1\varepsilon$.
3. **Stability.** The stability window is $0<\eta<2/\lambda_{\max}$, and the optimal step is $\eta^*=2/(\lambda_{\min}+\lambda_{\max})$.
4. **Empirical readout.** A real model's reported beliefs form a Cauchy sequence in tangent space; semi-log is close to straight and log-log bends; the fixed point is the evidence-consistent posterior; the measured $\rho_{\mathrm{emp}}$ runs over C3 $0.610$–$0.753$ and C4 $0.711$–$0.754$ on three orbits. This is the rate the model walks under “one small adjustment.” The geometry holds on the real orbit; the step size is the model's own.
5. **The emergence criterion.** If $\rho^*$ is unchanged while the metric jumps, it is a readout; if $\rho^*$ jumps, it is dynamics.

The whole chain has only three links:

> one number, determined by two endpoints; two endpoints, determined by the tangent Hessian; the tangent Hessian, determined by the anchor and the simplex constraint.

[Chapter 12](/en/volume1/chapter12/) says: inference converges to the prior anchor $A$, not to the true answer $A^*$. This chapter adds the “how fast”:

> How fast is determined by one condition number; the condition number is determined by two endpoints of the tangent Hessian; the two endpoints are determined by the matrix. Beyond this, there is no second scalar.

Now we can say which part of scaling law is ordinary and which part is not. What is ordinary is its format: as long as the output lies on the simplex, the loss is recorded as KL, and samples are counted, $L(s)-L_\infty$ will be a straight line on log-log paper. The format comes from geometry; the numerical values from engineering. What is not ordinary is why it keeps going down: a real system's floor falls with parameters, data, and compute together; what the exponent is, and how long it can keep falling, is a question for empirical science, requiring experiments and time. Separate the two: format belongs to geometry, exponent to experiment.

This chapter does not reproduce any company's curve, nor use a toy to impersonate it. It measures the same geometry behind that curve: a system whose output lies on the simplex converges as $\rho^t$, and a real model's reported beliefs do the same. **Engineering decides the numbers; geometry decides the necessity.** This holds for any system whose output lies on the simplex, and for a real model's belief orbit — which is exactly what it means to call the result universal.

Intelligence and emergence look like miracles mainly when we stare at a single readout; put the readout back on the curve, and they are simply names that one ordinary equation takes at different scales.

> **One sentence:** the curve does not belong to any one company; it belongs to everyone who counts on the simplex.

---

## Open Questions

**How should the rate be estimated away from the anchor's neighborhood?** The $\rho^*$ here is the exact rate of the linearization near $A$. A real trajectory carries higher-order terms before it enters that small neighborhood; the first ratio in the opening table being noticeably smaller than $\rho^*$ is a shadow of this “head start.” Can the tangent spectrum alone give a finite-step, non-asymptotic error bound? If the large components of the initial error happen to lie in fast directions, the actual window will be shorter than $\rho^*$ predicts — can that gap be computed exactly? This is open.

**Can $\lambda_{\min}$ and $\lambda_{\max}$ be measured directly on large models?** The spectra here are computed on explicit distributions with $K=3$ to $2048$. A real LLM has a huge vocabulary, and the Hessian cannot be stored explicitly. Can stochastic trace estimation, Lanczos, or implicit differentiation give reliable endpoint estimates? That would turn the diagnosis of whether the slope changes with scale from theory into experiment.

**When does $\rho^*$ change with scale?** This chapter gives the criterion but not a model for $\rho^*(N)$. If the appearance of a slow mode has a critical scale, what determines that critical point? This is what the other half of the scaling curve really needs to answer.

**Are twelve steps and two decimals enough?** The Cauchy certificate in 23.6 covers one finite window; the two-decimal readout floor is around $10^{-2}$. Can longer orbits and a finer readout estimate $\rho_{\mathrm{emp}}$ more stably? Does its gap from $\rho^*$ come from the model's step not being $\eta^*$, the readout precision, or a map that is not projected gradient? These are experimental questions, not algebraic ones.

**Why does the fixed point move when the interface changes?** The “small adjustment” map in 23.6 converges to the evidence-consistent posterior; the earlier “answer immediately” map converged to a prior-dominated point. Which map is the model's real inference dynamics? This requires treating the interface, the prompt, and the readout as variables, not mistaking one readout for the whole model.

---

## Exercises

**★ Warm-up**

1. Using the opening table, verify at which step the error ratio settles near $\rho^*$. Why is the first ratio noticeably smaller?
2. For $K=3$ and $q=(1/3,1/3,1/3)$, verify $H_T=3P_T$, write down all eigenvalues, and compute $\rho^*$.
3. For $q=(0.70,0.20,0.10)$, use the secular equation to compute $\lambda_{\min},\lambda_{\max},\kappa_T$, then $\eta^*,\rho^*$, and $N(0.1)$.
4. Using the table in 23.6, compare each C3 orbit's $\rho_{\mathrm{emp}}$ with $\rho^*$, then do the same for C4 and compare the two ranges.

**★★ Derivations**

1. Prove Lemma 23.1: the convex function $g(\lambda)=|1-\eta\lambda|$ attains its maximum on an interval at an endpoint.
2. Prove $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i$.
3. Prove Theorem 23.2: $\eta^*=2/(\lambda_{\min}+\lambda_{\max})$ is the unique minimizer, $\rho^*=(\kappa_T-1)/(\kappa_T+1)$, and the two endpoint branches are equal at the optimum.
4. Prove the two bounds in Theorem 23.3: $\frac{2x}{2+x}\le\ln(1+x)\le x$.
5. Prove that for any two probability vectors $p,q$, $p-q\in T$ and $\|x(p)-x(q)\|_2=\|p-q\|_2$. Then prove that the Euclidean norm on $T$ and the Bregman tangent norm $\|\cdot\|_{H_T}$ are equivalent, and give the equivalence constants.
6. Prove that if $F:T\to T$ satisfies $\|F(x)-F(y)\|_2\le\rho\|x-y\|_2$ with $\rho<1$, then every orbit is Cauchy, the limit is unique, and $\|x_t-x^*\|_2\le\rho^t\|x_0-x^*\|_2$.

**★★★ Challenge**

1. For $K=3$, write the secular equation as $3s_3\lambda^2-2s_2\lambda+1=0$, and verify that it matches the quadratic in [Chapter 12](/en/volume1/chapter12/).
2. Design an experiment that uses only the change of $\rho^*$ with scale to distinguish a “readout effect” from a “spectral bifurcation.” State what you would see in each case and how to rule out confounders.
3. Try to construct a counterexample showing that away from the anchor's neighborhood, where second-order terms cannot be ignored, the Euclidean linearized rate $\rho^*$ is no longer the exact rate; then state what extra condition would restore an exact rate formula.
4. Extend the twelve steps of 23.6 to thirty, or raise the output precision to six decimals, and re-estimate $\rho_{\mathrm{emp}}$. If it is stable but far from $\rho^*$, the model's step is not $\eta^*$; if it drifts with the window, the window is still too short. Design a control experiment that separates the two explanations.

---

## Appendix: Core Computation Script

The code below does one thing: given an anchor $q$, it computes the two endpoints of the tangent spectrum via the secular equation, then the exact rate, the optimal step, and the window. It trains no model.

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
    lam_min, lam_max = lam.min(), lam.max()
    kappa = lam_max / lam_min
    eta_star = 2.0 / (lam_min + lam_max)
    rho_star = (kappa - 1.0) / (kappa + 1.0)
    n_real = np.log(e0 / eps) / np.log(1.0 / rho_star)
    n_steps = np.ceil(n_real)
    lo = (kappa - 1.0) / 2.0 * np.log(e0 / eps)
    hi = kappa / 2.0 * np.log(e0 / eps)
    return dict(lam_min=lam_min, lam_max=lam_max, kappa_T=kappa,
                eta_star=eta_star, rho_star=rho_star,
                n_real=n_real, n_steps=n_steps, n_lo=lo, n_hi=hi)


print(rate([0.70, 0.20, 0.10]))
# lam_min=2.9904, lam_max=7.9620, kappa_T=2.6625,
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
- Nesterov, Yurii (2018). *Lectures on Convex Optimization*, 2nd edition. Springer Optimization and Its Applications 137. Springer. DOI: 10.1007/978-3-319-91578-4. — strong convexity, smoothness, step size, and condition number
- Boyd, Stephen; and Vandenberghe, Lieven (2004). *Convex Optimization*. Cambridge University Press. DOI: 10.1017/CBO9780511804441. — nonexpansiveness of projection and convergence rates
- Cover, Thomas M.; and Thomas, Joy A. (2006). *Elements of Information Theory*, 2nd edition. Wiley. — the local quadratic expansion of KL and the covariance of empirical distributions
- [Chapter 12](/en/volume1/chapter12/), Yonglin's Formula — the global version of the Yonglin theorem, tangent spectral bounds, and direct evidence

---

# Chapter 23: The Yonglin Corollary — Intelligence and the Log Curves of Emergence

> How fast a system converges is decided by one number.

---

[Chapter 12](/en/volume1/chapter12/) already proved convergence: under the tangent spectral bounds and the step-size window, the inference operator $F$ is a contraction, and Banach's fixed-point theorem guarantees that the iteration converges to the prior anchor $A$. The energy surface there is

$$
E(p)=D_{\mathrm{KL}}(A\|p).
$$

Convergence is established, and so is the “direction.” But “eventually converges” and “converges in $N$ steps” are two different statements.

This chapter does one thing: **compute the number that controls the speed, and see what follows from it.**

The same object will appear again and again:

$$
\boxed{\ \rho^*=\frac{\kappa_T-1}{\kappa_T+1}\ }
$$

where $\kappa_T=\theta_{\max}/\mu_T$ is the condition number of the tangent Hessian. This number determines the slope of the error curve, the number of steps needed to reach accuracy $\varepsilon$, and whether an “emergent” jump is real. No second scalar is needed.

Why “corollary”? Because the engine of convergence is still Banach's fixed-point theorem, and the energy surface is still the KL divergence from information theory. This chapter does not replace the engine. It reads off the engine's power.

> **The Yonglin Corollary = an information-theoretic energy surface + Banach completeness + one rate.**

:::info A pointer to the Polish school
The completeness used here comes from Banach's fixed-point theorem, which belongs to the Polish school of mathematics between the two world wars. The story of that school — and the methodological question of why convergence should be placed in a complete space — is told in the Polish-school note in [Chapter 12](/en/volume1/chapter12/). This chapter uses the result and does not retell the history.
:::

---

## 23.1 One Number: The Contraction Rate

Write the inference operator as

$$
F(p)=\Pi_{\mathcal{X}}\left(p-\eta P_T\nabla E(p)\right),
$$

where $E(p)=D_{\mathrm{KL}}(q\|p)$ and $P_T=I-\frac1K\mathbf{1}\mathbf{1}^\top$ is the orthogonal projection onto the tangent space

$$
T=\left\{v\in\mathbb{R}^K:\ \mathbf{1}^\top v=0\right\}.
$$

At the fixed point $A=q$,

$$
\nabla^2E(A)=\operatorname{diag}\!\left(\frac1{q_i}\right),
\qquad
H_T=P_T\,\nabla^2E(A)\,P_T.
$$

$H_T$ is the tangent Hessian. It is symmetric positive semidefinite on $T$ and has $K-1$ eigenvalues. Let

$$
\mu_T=\lambda_{\min}(H_T),\qquad \theta_{\max}=\lambda_{\max}(H_T).
$$

The tangent derivative of $F$ at $A$ is

$$
J_T=I_T-\eta H_T.
$$

Its eigenvalues are $1-\eta\lambda_i$. Define the **contraction rate**

$$
\rho(\eta)=\max_i|1-\eta\lambda_i|.
$$

This is the only protagonist of the chapter.

**First mathematical property: $\rho(\eta)$ is not an upper bound. It is the exact spectral radius.**

Because $J_T$ is symmetric on $T$, its operator norm equals its spectral radius:

$$
\|J_T\|_{T\to T}=\rho(\eta).
$$

Hence, for initial points near $A$, the asymptotic rate of the linearized error is exactly $\rho(\eta)$:

$$
\lim_{t\to\infty}\left(\frac{\|p_t-A\|_2}{\|p_0-A\|_2}\right)^{1/t}=\rho(\eta).
$$

(If the initial error has a nonzero component in the dominant eigenspace; otherwise it is only faster.) There is no room here for “approximately contractive” or “roughly converges.” The contraction rate is the spectral radius, and the spectral radius is the slope of the log curve.

![Figure 23.1: Geometric compression inside the simplex — after entering the anchor's neighborhood, every step multiplies the error by the same factor $\rho^*$](/figures/ch23_simplex_orbit.png)

Figure 23.1 draws compression on the probability simplex. The orbit starts at $p_0$ and moves toward the anchor $A=q$ at every iteration. Just after entering the anchor's neighborhood the contraction ratio fluctuates briefly; once inside the range where linearization is valid, every step multiplies the error by the same number $\rho^*\approx0.454$. This is what “geometric convergence” means geometrically: not the same distance removed each time, but the same proportion removed each time.

---

## 23.2 Only Two Endpoints Matter

$H_T$ has $K-1$ eigenvalues. A natural worry: do we need the whole spectrum to compute the rate?

No.

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

The left side holds because $1-\eta\lambda_i>-1$ for all $i$; the right side holds because the fastest mode $1-\eta\theta_{\max}$ is the first to cross $-1$. The boundary is exact, not “approximately here.”

---

## 23.3 The Optimal Rate: Exact Formula

Now choose a step size that minimizes $\rho(\eta)$. This is a minimax problem between the two endpoint branches:

$$
\rho^*=\inf_{\eta>0}\ \max\left(|1-\eta\mu_T|,\ |1-\eta\theta_{\max}|\right).
$$

The optimum is where the two branches meet:

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

**Theorem 23.2 (the rate form of the Yonglin Corollary).** Inside the stable window, the minimum of $\rho(\eta)$ is $\rho^*$, attained uniquely at $\eta^*$. Moreover,

$$
|1-\eta^*\mu_T|=|1-\eta^*\theta_{\max}|=\rho^*.
$$

That is, the optimal step size makes the slowest and fastest modes contract at exactly the same speed.

This rate has four exact properties, each requiring one line of algebra:

1. **Range.** $0\le\rho^*<1$. When $\kappa_T=1$ (all tangent eigenvalues equal), $\rho^*=0$ and the linearized map converges in one step; as $\kappa_T\to\infty$, $\rho^*\to1$ and convergence becomes arbitrarily slow.
2. **Monotonicity.** $\dfrac{d\rho^*}{d\kappa_T}=\dfrac{2}{(\kappa_T+1)^2}>0$. A larger condition number always means a worse rate.
3. **Invertibility.** $\kappa_T=\dfrac{1+\rho^*}{1-\rho^*}$. Rate and condition number are in one-to-one correspondence: knowing one is knowing the other. In this sense, $\rho^*$ is a complete invariant of the condition number.
4. **Endpoint dependence only.** By Lemma 23.1, $\rho^*$ depends only on $\mu_T$ and $\theta_{\max}$. The shape of the intermediate spectrum never enters the formula.

Property 3 deserves a second look. It says that “how fast convergence is” and “how anisotropic the energy surface is” are not independent questions: they are two ways of writing the same number.

![Figure 23.2: Three readings of one scalar — the exact V-shape, the one-to-one rate–condition-number map, and the exact linear bounds on the window](/figures/ch23_rate_properties.png)

Panel (a) is the V-shape of $\rho(\eta)$. The left branch is controlled by the slow mode $|1-\eta\mu_T|$, the right branch by the fast mode $|1-\eta\theta_{\max}|$, and their intersection is $(\eta^*,\rho^*)$. The dashed line is the stability edge $2/\theta_{\max}$.

Panel (b) plots $\rho^*=(\kappa_T-1)/(\kappa_T+1)$. It is monotone, bounded, and invertible.

Panel (c) is the window. The next section turns it into exact inequalities.

![Figure 23.3: The rate–condition-number hyperbola — one relation read in two directions](/figures/ch23_rate_hyperbola.png)

Figure 23.3 isolates the relation. The left panel plots $\rho^*=(\kappa_T-1)/(\kappa_T+1)$ as a function of $\kappa_T$: one branch of a hyperbola, with horizontal asymptote $\rho^*=1$. The right panel is the inverse, $\kappa_T=(1+\rho^*)/(1-\rho^*)$: as $\rho^*\to1$, the condition number shoots vertically to infinity. Both panels say the same thing — the better the rate, the closer the condition number is to 1; the closer the rate is to 1, the more the condition number explodes. The baseline $q=(0.70,0.20,0.10)$ is the same point $(2.663,0.454)$ in both panels.

---

## 23.4 The Hessian: Where $\mu_T$ and $\theta_{\max}$ Come From

The rate needs only two endpoints, but those endpoints still have to be computed from the matrix. Start with a concrete picture. Take $K=6$ and the anchor

$$
q=(0.35,\ 0.25,\ 0.15,\ 0.12,\ 0.08,\ 0.05).
$$

![Figure 23.4: Three readings of the Hessian — the raw matrix, the tangent projection, and the tangent spectrum](/figures/ch12_special_hessian_matrix.png)

### Panel (a): The raw Hessian — the “local stiffness” of each answer

$$
\nabla^2E(q)=\operatorname{diag}\!\left(\frac1{q_1},\ldots,\frac1{q_K}\right)
=\operatorname{diag}(2.86,\ 4.00,\ 6.67,\ 8.33,\ 12.50,\ 20.00).
$$

The $i$-th diagonal entry is $1/q_i$. The rarer the answer, the steeper that direction. Everything off the diagonal is zero: if probabilities could vary independently, the answers would not interact.

But probabilities cannot vary independently. They must sum to 1. The constraint plants off-diagonal entries.

### Panel (b): The tangent Hessian — the constraint couples the answers

After projection, the entries of $H_T$ are

$$
(H_T)_{ij}
=\frac{\delta_{ij}}{q_i}
-\frac{1}{K}\left(\frac{1}{q_i}+\frac{1}{q_j}\right)
+\frac{1}{K^2}\sum_{k=1}^{K}\frac{1}{q_k}.
$$

Three readings:

1. **Every row sums to zero.** The all-ones direction is normal, $H_T\mathbf{1}=0$. Adding the same number to every answer leaves the belief unchanged.
2. **The diagonal is positive, and the total trace shrinks.** $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i=45.30$, less than the raw trace $54.36$. The constraint consumes part of the total curvature.
3. **Most off-diagonal entries are negative, but not all.** Between the two rarest answers the coupling is strongly negative ($q_5$ and $q_6$: $-3.91$): raising one forces the other down. Between the two most common answers there is a small positive coupling ($q_1$ and $q_2$: $+0.37$). The only exact statements are: rows sum to zero, and the matrix is positive semidefinite on the tangent space.

A beautiful special case: when $q$ is uniform, $1/q_i=K$, so

$$
H_T=K\,P_T=K\left(I-\frac1K\mathbf{1}\mathbf{1}^\top\right).
$$

This is the graph Laplacian of the complete graph $K_K$, with eigenvalues $0$ (normal, multiplicity one) and $K$ (tangent, multiplicity $K-1$). The tangent geometry of a uniform belief is the graph in which every answer is connected to every other answer by an edge of equal weight.

### Panel (c): The tangent spectrum — only the two ends matter

For this anchor, the tangent spectrum is

$$
\lambda=(3.32,\ 5.24,\ 7.57,\ 11.13,\ 18.03).
$$

Hence

$$
\mu_T=3.32,\qquad \theta_{\max}=18.03,\qquad \kappa_T=5.43.
$$

- The smallest eigenvalue is the slow mode that redistributes mass between the two most likely answers.
- The largest eigenvalue is the fast mode that corrects the rarest answer.
- The three eigenvalues in between contribute nothing to the optimal rate. This is Lemma 23.1 in action.

![Figure 23.5: The tangent spectrum interlaces the raw diagonal; only the two endpoints enter the rate](/figures/ch23_hessian_spectrum.png)

Panel (a) of Figure 23.5 draws the “interlacing” of the secular equation: gray stems are the raw diagonal $d_i=1/q_i$, blue stems are the tangent eigenvalues. Every blue root lies between two adjacent gray levels, $d_{(1)}<\lambda_1<d_{(2)}<\cdots$. Panel (b) switches to three anchors with $K=6$: the uniform anchor has five equal eigenvalues ($\kappa_T=1$); the baseline anchor gives $\kappa_T=5.43$; the near-vertex anchor pushes all eigenvalues up together, but their ratio only reaches $5.70$ — the tangent condition number is pinned near $K=6$ by the constraint. The interior eigenvalues change the shape of the spectrum, not $\rho^*$.

:::details Lemma: the tangent spectrum is determined by one scalar equation

Let $D=\operatorname{diag}(d_1,\ldots,d_K)$ with $d_i=1/q_i$, and $H_T=P_TDP_T$. For $x\in T$, $H_Tx=\lambda x$ is equivalent to $P_TDx=\lambda x$. Since $x\in T$, $P_Tx=x$, so $Dx-\lambda x=c\mathbf{1}$ for some scalar $c$, i.e. $x_i=c/(d_i-\lambda)$. The condition $x\in T$ gives the **secular equation**

$$
\sum_{i=1}^{K}\frac{1}{d_i-\lambda}=0.
$$

Its $K-1$ roots are the tangent spectrum, interlacing with the $d_i$:

$$
d_{(1)}<\lambda_1<d_{(2)}<\cdots<\lambda_{K-1}<d_{(K)}.
$$

To find $\mu_T$ and $\theta_{\max}$, bisect once between the two smallest $d_i$ and once between the two largest. For $K=3$, the secular equation becomes

$$
3s_3\lambda^2-2s_2\lambda+1=0,
\qquad
s_2=q_1q_2+q_1q_3+q_2q_3,\quad s_3=q_1q_2q_3,
$$

which is exactly the quadratic that appears in the “direct evidence” section of [Chapter 12](/en/volume1/chapter12/).

:::

:::details Boundary: the tangent condition number saturates at $K$

The raw Hessian blows up at the boundary: as $q_{\min}\to0$, $1/q_{\min}\to\infty$. The tangent spectrum is held back by the constraint. Take the symmetric vertex limit

$$
q(t)=\frac{1-t}{K}\mathbf{1}+t\,e_1,\qquad t\to1.
$$

The full-space condition number $q_{\max}/q_{\min}\sim1/(1-t)$ diverges, but the tangent condition number satisfies

$$
\kappa_T\to K.
$$

Numerically, $K=5,10,50,200,1000$ give $\kappa_T=5.0000,10.0000,50.0000,199.9998,999.9990$. The reason is that the $K-1$ rare answers have equal $d_i$; the secular equation pulls out one root $d_{\text{rare}}/K$ from that block, while the other $K-2$ roots coincide. The constraint spreads the blow-up of “one tiny probability” into the coupling among $K$ answers.

![Figure 23.6: The full-space condition number diverges; the tangent condition number saturates at $K$](/figures/ch23_kappa_saturation.png)

Panel (a) of Figure 23.6 is the symmetric vertex limit for $K=5$: the red curve is the full-space condition number, diverging logarithmically as $t\to1$; the blue curve is the tangent condition number, sitting flat at $K=5$. Panel (b) normalizes the spectrum by $\mu_T$: the minimum stays at 1, and the other $K-1$ eigenvalues rise together to $K$. The divergence has not disappeared; the constraint has redistributed it into the coupling among $K$ answers.

:::

---

## 23.5 The Rate Is the Slope of the Log Curve

Start with the linearized model. The error recursion is exact:

$$
\|p_t-A\|_2=\rho^t\|p_0-A\|_2.
$$

Taking natural logarithms,

$$
\ln\|p_t-A\|_2=\ln\|p_0-A\|_2+t\ln\rho.
$$

This is an affine function of $t$. **In semi-log coordinates, it is a straight line whose slope is exactly $\ln\rho$.** In log-log coordinates it is no longer straight: setting $t=e^s$ turns an affine function into an exponential one. So the statement “geometric convergence is a straight line in log-log coordinates” is false; the correct statement is “it is a straight line in semi-log coordinates.” Nonlinear orbits carry higher-order terms, but the asymptotic slope determined by the dominant eigenvalue is still $\ln\rho$.

![Figure 23.7: Straight in semi-log coordinates, curved in log-log coordinates](/figures/ch23_semilog_loglog.png)

Figure 23.7 plots the same data twice. Panel (a) is semi-log: all three geometric orbits are exactly straight lines, with slopes equal to $\ln\rho$. A step that is too small (light blue) gives a shallow slope; the optimal step (dark blue) gives the steepest; a step that is too large (red) compresses the slow mode quickly but is held back by the fastest mode, so it is slower overall. Panel (b) switches to log-log: the same geometric orbit bends into a curve and is no longer a line; the gray dashed curve is a genuine power law $t^{-1.5}$, which is straight in log-log. So “a straight line in log-log” belongs to power laws, while geometric convergence is a straight line in semi-log.

At rate $\rho^*$, the continuous number of steps needed to reach accuracy $\varepsilon$ is

$$
N_{\mathrm{real}}(\varepsilon)=\frac{\ln(\|p_0-A\|_2/\varepsilon)}{\ln(1/\rho^*)},
\qquad
N(\varepsilon)=\left\lceil N_{\mathrm{real}}(\varepsilon)\right\rceil.
$$

Set $\|p_0-A\|_2=1$ and $L=\ln(1/\rho^*)$; then $N_{\mathrm{real}}=\ln(1/\varepsilon)/L$. The following theorem gives an exact sandwich for $L$.

**Theorem 23.3 (exact linear bounds on the window).** For $\kappa_T>1$,

$$
\frac{\kappa_T-1}{2}\ln\frac1\varepsilon
\ \le\ N_{\mathrm{real}}(\varepsilon)\ \le\
\frac{\kappa_T}{2}\ln\frac1\varepsilon.
$$

**Proof.** Let $x=2/(\kappa_T-1)>0$. Since $(\kappa_T+1)/(\kappa_T-1)=1+x$,

$$
L=\ln\frac{\kappa_T+1}{\kappa_T-1}=\ln(1+x).
$$

Two standard inequalities give

$$
\frac{2x}{2+x}\le\ln(1+x)\le x.
$$

Substituting $x=2/(\kappa_T-1)$,

$$
\frac{2}{\kappa_T}\le L\le\frac{2}{\kappa_T-1}.
$$

Taking reciprocals and multiplying by $\ln(1/\varepsilon)$ proves the claim. $\square$

Both bounds are exact, not asymptotic approximations. They pin the window between two lines of slopes $(\kappa_T-1)/2$ and $\kappa_T/2$. Panel (c) of Figure 23.2 shows the sandwich: the blue curve is the exact $N$, and the shaded region is bounded by the two exact lines.

For the baseline $q=(0.70,0.20,0.10)$:

$$
\mu_T=2.9904,\quad \theta_{\max}=7.9620,\quad \kappa_T=2.6625,
$$

$$
\eta^*=0.18261,\quad \rho^*=0.45393,\quad N_{\mathrm{real}}(0.1)=2.915,
$$

so $N=\lceil2.915\rceil=3$ steps. The exact bounds give

$$
\frac{2.6625-1}{2}\ln10=1.914
\ \le\ 2.915\ \le\
\frac{2.6625}{2}\ln10=3.065.
$$

The window lies inside the bounds, and the two bounds are only $1.15$ steps apart. This is not “about 3 steps”; it is “exactly between 1.914 and 3.065.”

---

## 23.6 Emergence: The Rate Changed, or Only the Readout Did

The error curve is $\rho^t$. If $\rho$ does not change, the curve is exactly geometric, with no kink.

So a “suddenly it can do it” kink has only two possible sources:

1. **The readout changed; the rate did not.** The metric is nonlinear. For example, accuracy is $\mathbf{1}[\text{error}<\varepsilon]$; when a smooth error curve crosses the threshold, accuracy looks like a step function. Schaeffer, Miranda, and Koyejo (2023) argued systematically that this kind of “emergence” can be manufactured entirely by the choice of metric.
2. **The rate really changed.** This can happen only through $\mu_T$ or $\theta_{\max}$: a new slow mode enters the low end of the spectrum, or a new fast mode enters the high end. Then $\rho^*$ itself changes, and the error curve really bends.

With the scalar of this chapter, the diagnostic rule is one line:

> Measure how $\rho^*$ changes with scale. If $\rho^*$ is constant and a benchmark jumps, it is a readout effect. If $\rho^*$ jumps, it is emergence in the dynamical sense.

![Figure 23.8: Two kinds of “emergence” — a readout jump, or a real change of slope](/figures/ch23_emergence.png)

Figure 23.8 draws the criterion. In panel (a), loss is a straight line in log-log and the rate does not change; but accuracy is a threshold metric, so when the curve crosses the threshold it looks like a step. In panel (b), $\rho^*$ itself drops from $0.82$ to $0.45$, the slope of the semi-log error curve changes, and the same number of steps leaves errors several orders of magnitude apart. The first is readout; the second is dynamics.

This is harder than “looks like emergence.” It turns emergence from an adjective into a measurable spectral event. Wei et al. (2022) documented emergence as a real empirical phenomenon; this chapter supplies a criterion for separating the part that comes from the rate from the part that comes from the readout.

:::details A boundary we must acknowledge: this chapter does not define “intelligence”
The title “Intelligence and the Log Curves of Emergence” can easily be read as “this chapter explains intelligence.” It does not.

This chapter explains one class of learning dynamics: when a system minimizes KL with the probability simplex as its output space, what mathematics governs its error curve. It does not answer “what is understanding,” “what is consciousness,” or “why do some abilities appear only in sufficiently large models.” Those questions need other tools, and perhaps tools that do not yet exist.

Treating the Yonglin Corollary as a definition of intelligence is a mistake. Treating it as a geometric bill that any intelligence system must pay is appropriate.
:::

---

## 23.7 Conclusion: One Number

The whole chapter is one number:

$$
\rho^*=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T},
\qquad
\kappa_T=\frac{1+\rho^*}{1-\rho^*}.
$$

It determines four things exactly:

1. **Slope.** The slope of the semi-log error curve is $\ln\rho^*$.
2. **Steps.** The number of steps to reach $\varepsilon$ is exactly sandwiched between $\frac{\kappa_T-1}{2}\ln\frac1\varepsilon$ and $\frac{\kappa_T}{2}\ln\frac1\varepsilon$.
3. **Stability.** The stable window is $0<\eta<2/\theta_{\max}$, and the optimal step size is $\eta^*=2/(\mu_T+\theta_{\max})$.
4. **A criterion for emergence.** If $\rho^*$ is constant and the metric jumps, it is a readout effect; if $\rho^*$ jumps, it is dynamics.

[Chapter 12](/en/volume1/chapter12/) says that reasoning converges to the prior anchor $A$, not to the true answer $A^*$. This chapter adds the “how fast”:

> How fast is decided by one condition number; the condition number is decided by the two endpoints of the tangent Hessian; the endpoints are decided by the matrix. There is no second scalar.

The sentence “Scaling Law is not mysticism” now stands, and it does not need some proprietary model to testify on its behalf. GPT can push a loss curve down to 2.5; a small FFN may only reach 9. Their concrete floors differ, but they share the same log-log curve. A model is a tool for probing downward, not the inventor of the curve. Any system that projects onto the probability simplex, moves in the simplex, and trains with a minimum-entropy (KL) objective will exhibit this geometric criterion. In other words, **any architecture satisfying those three conditions can present the same “geometric-criterion certificate”; the architecture determines the numerical floor written on the certificate, not the format of the certificate.**

---

## Open Questions

**What is the exact rate of mirror descent?** The contraction rate here is built on Euclidean projected gradient. Real models are closer to KL mirror descent, and KL is not a metric, so Banach does not apply directly. Is there a Bregman version of the spectral-radius formula? If so, does it still depend only on two endpoints? This is open.

**Can we measure $\mu_T$ and $\theta_{\max}$ at the scale of large models?** The spectra here are computed on explicit distributions with $K=3$ to $2048$. A real LLM has a huge vocabulary, and its Hessian cannot be stored explicitly. Can stochastic trace estimation, Lanczos, or implicit differentiation give reliable endpoint estimates? That would turn the emergence diagnostic from theory into experiment.

**When does $\rho^*$ change with scale?** This chapter gives the criterion, but not a model for $\rho^*(N)$. If the appearance of a slow mode has a critical scale, what determines that critical point? This is the question that emergence actually needs to answer.

---

## Exercises

**★ Warm-up**

1. For $K=3$ and $q=(1/3,1/3,1/3)$, verify that $H_T=3P_T$, write down all eigenvalues, and compute $\rho^*$.
2. For $q=(0.70,0.20,0.10)$, use the secular equation to compute $\mu_T,\theta_{\max},\kappa_T$, then $\eta^*,\rho^*$, and $N(0.1)$.

**★★ Derivations**

1. Prove Lemma 23.1: the convex function $g(\lambda)=|1-\eta\lambda|$ attains its maximum on an interval at an endpoint.
2. Prove $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i$.
3. Prove Theorem 23.2: $\eta^*=2/(\mu_T+\theta_{\max})$ is the unique minimizer, $\rho^*=(\kappa_T-1)/(\kappa_T+1)$, and the two endpoint branches take equal values at the optimum.
4. Prove the two bounds in Theorem 23.3: $\frac{2x}{2+x}\le\ln(1+x)\le x$.

**★★★ Challenges**

1. For $K=3$, write the secular equation as $3s_3\lambda^2-2s_2\lambda+1=0$ and verify that it agrees with the quadratic in [Chapter 12](/en/volume1/chapter12/).
2. Design an experiment that uses only the change of $\rho^*$ with scale to distinguish a readout effect from a spectral bifurcation. Say what you would see in each case, and how you would rule out confounders.
3. Try to construct a counterexample showing that under KL mirror descent, the Euclidean $\rho^*$ is no longer the exact rate; then state what extra condition would restore an exact rate formula.

---

## Appendix: Core Computation Script

The code does one thing: given an anchor $q$, compute the two endpoints of the tangent spectrum from the secular equation, then the exact rate, optimal step size, and window. It trains no model.

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

- Banach, Stefan (1922). “Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.” *Fundamenta Mathematicae* 3(1), 133–181. DOI: 10.4064/fm-3-1-133-181.
- Schaeffer, Rylan; Miranda, Brando; and Koyejo, Sanmi (2023). “Are Emergent Abilities of Large Language Models a Mirage?” arXiv:2304.15004.
- Wei, Jason; Tay, Yi; Bommasani, Rishi; Raffel, Colin; Zoph, Barret; Borgeaud, Sebastian; Yogatama, Dani; Bosma, Maarten; Zhou, Denny; Metzler, Donald; Chi, Ed H.; Hashimoto, Tatsunori; Vinyals, Oriol; Liang, Percy; Dean, Jeff; and Fedus, William (2022). “Emergent Abilities of Large Language Models.” arXiv:2206.07682.
- Amari, Shun-ichi (2016). *Information Geometry and Its Applications*. Applied Mathematical Sciences 194. Springer Japan. DOI: 10.1007/978-4-431-55978-8.
- Nesterov, Yurii (2018). *Lectures on Convex Optimization*, 2nd edition. Springer Optimization and Its Applications 137. Springer. DOI: 10.1007/978-3-319-91578-4.
- Boyd, Stephen; and Vandenberghe, Lieven (2004). *Convex Optimization*. Cambridge University Press. DOI: 10.1017/CBO9780511804441.
- This book, [Chapter 12](/en/volume1/chapter12/), Yonglin's formula — the global version of the Yonglin theorem, tangent spectral bounds, and direct evidence.

# Chapter 17: Probability as the Expansion of Logic — Truth Values from {0,1} to [0,1]

> Probability is not frequency. It is the unique consistent representation of rational belief under uncertainty.

---

The end of Chapter 16 left a hanging thread: the semantics of linear logic hint that "truth values" are no longer a simple $\{0,1\}$, but some richer structure. But that direction — phase semantics, coherence spaces — is a technically difficult road, left to researchers.

There is a broader road.

If "truth value" is not true-or-false, but a real number between 0 and 1, what happens? If this real number represents some agent's **degree of belief** in the proposition being true, what shape should the inference rules take?

This is the question of this chapter. The answer is called **Bayesian probability theory** — but not the frequentist version you may have learned; rather, its logical-foundations version: probability as the expression of rational belief, obeying laws that can be deductively derived.

---

## 17.0 The Belief Chip Game: Staying Self-Consistent Under Uncertainty

Now we replace truth values — from black and white chess pieces to chips.

On the table are several propositions: $H_1, H_2, \ldots, H_k$. In your hand are one hundred belief chips, which you must allocate among these propositions. The more chips a proposition receives, the more you believe it. The requirement of the game is not "always bet correctly," but something more basic: **your bets must not contradict themselves**.

Then evidence arrives. You see $E$. The rules require you to reallocate chips: hypotheses that support $E$ gain more chips; hypotheses that struggle to explain $E$ lose chips. This movement is not arbitrary but must obey Bayesian updating:

$$P(H \mid E) = \frac{P(E \mid H)P(H)}{P(E)}$$

You can think of Bayes' formula as the anti-cheating rule of a casino: if you don't update in this way, a sufficiently clever bookie can design a set of bets such that you lose money no matter how the world is. This is the intuition behind the Dutch book argument.

:::details The formal skeleton of this game
- **State space**: probability distribution $p_t \in \Delta^{k-1}$, i.e., all legal allocations of belief chips.
- **Legal actions**: after observing evidence $E_t$, update the prior $p_t$ to the posterior $p_{t+1}$.
- **Transition rule**: $p_{t+1}(H) \propto P(E_t \mid H)p_t(H)$.
- **Victory condition**: the belief distribution both absorbs evidence and maintains the consistency of probability axioms.
- **Failure mode**: the chip allocation violates probability rules, leading to internal contradictions, or being stably harvested by a Dutch book.
:::

The key of this game is not "probability can guarantee you are correct." Probability has no such grand power. It only guarantees one more modest but more necessary thing: when evidence is incomplete, your belief updates do not fight each other.

Formal logic handles legal derivation in a deterministic world; probabilistic logic handles legal updating in an uncertain world. The former asks "can this conclusion be derived"; the latter asks "after seeing this evidence, where should I move my beliefs."

::: info Professor Manul comments
Probability is not fuzzy logic prepared for cowards. Probability is the posture that rationality can still hold steady when information is insufficient. You don't know the answer, so you allocate chips; evidence arrives, so you move chips. What is truly irrational is not uncertainty, but pretending that you still possess certainty amid uncertainty.
:::

## 17.1 The Debate Between Two Kinds of Probability

What does "probability $\frac{1}{2}$" mean?

The frequentist answer: toss this coin infinitely many times; the proportion of heads approaches $\frac{1}{2}$. Probability is long-run frequency, meaningful only for repeatable experiments.

The Bayesian answer: I believe the probability that the next toss of this coin lands heads is $\frac{1}{2}$. Probability is a measure of belief, equally meaningful for single events.

This debate has lasted a century and is still not fully settled. But there is one question that frequentism cannot answer, while Bayesianism can:

"What is the probability of rain tomorrow?"

Tomorrow only happens once. There are no infinite repetitions. You cannot wait for infinitely many "tomorrows" to measure the frequency. Yet the weather forecast says 70% chance of rain, and this 70% is meaningful — it describes the forecaster's strength of belief in the proposition "it will rain tomorrow," based on existing meteorological data.

The core claim of Bayesian probability theory is: **probability is the logic of belief, not the statistics of frequency**. Moreover, the beliefs of a rational agent must obey the probability axioms — not because natural laws dictate it, but because beliefs that violate the probability axioms are **incoherent** and will generate contradictions during inference.

---

## 17.2 Cox's Theorem: The Necessity of the Axioms

Richard Cox, in 1946, asked a question: if you want to use real numbers to express strength of belief, such that this system of expression is **internally consistent**, what constraints must these real numbers satisfy?

His starting point was three requirements, each a minimal standard for rational belief:

**Requirement One (Ordinality)**: Beliefs are comparable. For any two propositions $A$ and $B$, your degree of belief in $A$ is either greater than, equal to, or less than your degree of belief in $B$.

**Requirement Two (Consistency)**: Belief in a compound proposition is fully determined by belief in its constituent propositions. Your degree of belief in "$A$ and $B$" is some function of your belief in $A$ and your belief in $B$ given that $A$ holds.

**Requirement Three (Duality)**: Belief in $A$ and belief in $\neg A$ are complementary — complete confidence in $A$ means complete disbelief in $\neg A$.

Cox proved: under these three requirements, any internally consistent measure of belief must, up to some monotonic transformation, be equivalent to **standard probability**. That is, you can choose a different scale (using $[0,100]$ instead of $[0,1]$), but the structure of the inference rules is completely determined:

$$P(A \land B) = P(A) \cdot P(B \mid A)$$

$$P(A) + P(\neg A) = 1$$

This is not a law discovered through experiment, but an inevitable consequence of rational self-consistency. If you use numbers to express belief, and your beliefs are internally consistent, you are using probability — you just may not have realized it yet.

::: info Professor Manul comments
The conclusion of Cox's theorem makes many people mistakenly think "Bayesianism is the only rational choice." Slow down — the premise of the theorem is that belief can be represented by a real-number linear order. If you question this premise, the entire theorem does not apply. The power of the theorem comes from its premises; think clearly about the premises first, then discuss the necessity of the conclusion. Don't treat a conditional conclusion as an unconditional truth.
:::

::: info The philosophical implications of Cox's theorem

The profundity of Cox's theorem lies in its "uniqueness": a measure of belief that satisfies the requirements of rationality is structurally unique. This means that probability theory is not a set of tools invented by humans — it is the **necessary form** of rational belief. If you reject probability theory, you must either reject the comparability of beliefs (abandon ordinality), or accept internal contradictions among your beliefs.

This is completely consistent with the spirit of Chapter 14: the axioms of a formal system are not arbitrary conventions, but the minimal commitments that must be accepted to avoid contradictions. Cox's theorem applies the same logic to belief: to avoid incoherence, beliefs must obey the probability axioms.
:::

---

## 17.3 Bayesian Updating: The Probabilistic Version of Inference Rules

With probability as a measure of belief, what does "inference" become?

In formal logic, inference produces new true propositions from known true propositions. In probability theory, inference **updates** degrees of belief in propositions from known observations.

The rule for this update is the most important theorem of probability theory:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

Spelling out these four quantities in plain language:

- $P(H)$: degree of belief in hypothesis $H$ before seeing evidence $E$ — the **prior probability**.
- $P(E \mid H)$: probability of seeing evidence $E$ if hypothesis $H$ is true — the **likelihood**.
- $P(E)$: probability of seeing evidence $E$ across all possible cases — the **marginal probability**, a normalization factor.
- $P(H \mid E)$: new degree of belief in hypothesis $H$ after seeing evidence $E$ — the **posterior probability**.

:::details Prior / Likelihood / Posterior: an illustrated triangular relationship
These three words are the core of Bayesian inference and are often confused upon first encounter:

**Prior $P(H)$**: the belief you hold before seeing any data. Example: "I guess this coin is fair; the probability of heads is 50%." Its source is domain knowledge, historical data, or uninformative assumptions.

**Likelihood $P(E \mid H)$**: **if hypothesis $H$ is true**, the probability of observing the current evidence $E$. Note the direction — it is not "how likely $H$ is after seeing $E$," but "assuming $H$ is true, how likely $E$ is to appear." Example: "If the coin is fair, the probability of 3 consecutive heads is $1/8$."

**Posterior $P(H \mid E)$**: the updated belief after seeing the evidence. This is the answer you actually want.

**Marginal probability $P(E)$**: the normalization constant that ensures posterior probabilities sum to 1. In practical computation, one often uses $\text{posterior} \propto \text{likelihood} \times \text{prior}$, ignoring this constant.

Memory formula: **new belief ∝ old belief × strength of evidence support**
:::


This is **Bayes' theorem**, or more precisely, the core operation of Bayesian inference.

But writing it as a formula easily makes people miss its logical essence. A clearer way to write it is:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

($\propto$ means proportional to; $P(E)$ is a constant normalization factor and does not change the relative proportions.)

What this formula says is: **belief after seeing evidence is the result of belief before seeing evidence, weighted by evidence**. Evidence acts on the prior through the likelihood function, "pushing" the prior to the posterior.

Returning to the belief chip game, Bayes' formula is not an externally added "statistical trick," but the legality rule for chip movement. You can certainly move chips from one hypothesis to another based on feeling; but as long as you don't move them in proportion to likelihood and prior, you leave arbitrage gaps exploitable by a Dutch book. The coldness of probability theory lies precisely here: it does not guarantee you bet correctly; it only guarantees that you don't contradict yourself within your own betting rules.

### 17.3.5 Hands-On: Doing a Bayesian Update Step by Step

Use the medical test from Thought Exercise 1 to trace a complete update. It's not about plugging into a formula—it's about watching how the chips move.

**Setup**:
- Disease prevalence (prior): $P(\text{sick}) = 0.01$, $P(\text{healthy}) = 0.99$
- Test sensitivity (likelihood): $P(\text{positive} \mid \text{sick}) = 0.90$
- Test specificity: $P(\text{negative} \mid \text{healthy}) = 0.95$, so $P(\text{positive} \mid \text{healthy}) = 0.05$

**Step 1**: Prior chips. Imagine a population of 10,000 people. By the prior, about 100 are sick, 9,900 are healthy.

**Step 2**: A positive test result comes in. Among the 100 sick people, about 90 test positive. Among the 9,900 healthy people, about $9,900 \times 0.05 = 495$ test positive.

**Step 3**: Redistribute belief. The positive-test population is $90 + 495 = 585$ people. Of these, 90 are truly sick. So your updated belief that the person is sick is:
$$P(\text{sick} \mid \text{positive}) = \frac{90}{585} \approx 0.154 = 15.4\%$$

**Verify with the formula**:
$$P(\text{sick} \mid \text{positive}) = \frac{0.90 \times 0.01}{0.90 \times 0.01 + 0.05 \times 0.99} = \frac{0.009}{0.009 + 0.0495} \approx 0.154$$

Both methods agree.

**Key observation**: The test came back positive, but the probability of actually having the disease is only 15.4%. Most people's intuition estimates above 80%. Why the huge gap? Because the disease is extremely rare (1%)—even with high test sensitivity, the majority of positive results are still false positives from healthy people. Simply put, the healthy population is so large (99%) that the 5% false positive rate produces far more positive results (495) than the true positives (90).

**Do a second test**: Treat the 15.4% as the new prior; the second test also comes back positive.

$$P(\text{sick} \mid \text{two positives}) = \frac{0.90 \times 0.154}{0.90 \times 0.154 + 0.05 \times 0.846} = \frac{0.1386}{0.1386 + 0.0423} \approx 0.766$$

After two positives, the probability of having the disease jumps to 76.6%. The prior went from 1% -> 15.4% -> 76.6%—two pieces of evidence caused a dramatic shift in belief.

**What if the base prevalence were 10% (high-risk group)?** After one positive:
$$P(\text{sick} \mid \text{positive}) = \frac{0.90 \times 0.10}{0.90 \times 0.10 + 0.05 \times 0.90} = \frac{0.09}{0.09 + 0.045} = 0.667$$

A high-risk person (10%) after one positive has a 66.7% chance, which is lower than a low-risk person (1%) after two positives (76.6%). Two pieces of cumulative evidence, in the low-risk group, give higher confidence than a single test in the high-risk group. This shows the cumulative power of evidence—enough evidence can overwhelm the prior.

But this example also reveals a default assumption of Bayesian updating: **you observed a *certain* piece of evidence $E$**. You know the test result is "positive," no ambiguity, probability 1. In reality, evidence itself is often uncertain. That is the topic of the next subsection.

::: info Structural analogy of inference rules

Contrasting Bayesian updating with the inference rules of Chapter 14 reveals surprising similarities:

- Formal logic: $\frac{P \to Q \quad P}{Q}$ (modus ponens: consume $P$ and $P \to Q$, obtain $Q$)
- Bayesian: $P(H \mid E) \propto P(E \mid H) \cdot P(H)$ (consume likelihood $P(E \mid H)$ and prior $P(H)$, obtain posterior $P(H \mid E)$)

Both are "using what you have to produce something new"; the difference is: in formal logic, "what you have" are true propositions with $\{0,1\}$ values; in Bayesian inference, "what you have" are degrees of belief with $[0,1]$ values. Bayesian inference is the version of modus ponens expanded to the continuous truth-value domain.
:::

---

## 17.4 The Prior: Inference Never Starts from Zero

Bayesian inference has one aspect that makes many people uncomfortable: you need a prior.

The prior is the belief you already hold before seeing any evidence. Where does this come from? If I know nothing at all, what is the prior?

Frequentists consider this requirement the fatal weakness of the Bayesian method — the prior is subjective; different people can have different priors and obtain different posteriors; who is to say who is right?

The Bayesian answer has two layers.

First layer: **the prior is not arbitrary**. Rational priors are subject to various constraints. The most basic constraint is symmetry: if you know nothing about a situation, you have no reason for the prior to favor any side. This yields the "uninformative prior" — when there is no preference information, assign a uniform prior (for discrete cases) or a maximum-entropy prior (for continuous cases).

Second layer: **the influence of the prior diminishes as evidence accumulates**. This is a mathematical theorem of Bayesian updating: after sufficiently many independent observations, no matter which prior you started from, the posterior will converge to the same position. Subjective priors are temporary; data is objective; rational agents will ultimately reach consensus.

Illustrate with an extreme example. Suppose two people debate whether a certain coin is fair: one person's prior belief is that the probability of heads is 0.99, the other believes it is 0.01. They simultaneously observe this coin tossed 1000 times, with 503 heads. After Bayesian updating, both people's posteriors will concentrate around 0.5 — vastly divergent priors, drowned by evidence.

This convergence property is the source of the Bayesian method's objectivity: not the objectivity of the prior, but the objectivity of the **inference process**.

### 17.4.5 Jeffrey Conditionalization: When Evidence Itself Is Uncertain

Bayes' theorem assumes you observed a **certain event** $E$—"the test result is positive"—with probability 1. But what you see is not always certain. A shadow flickers past the window—it could be a bird, a plane, or a hallucination. Your observation is not "I saw a bird (probability 1)," but "my belief that 'a bird flew by' rose from 0.1 to 0.7."

How do you handle uncertain evidence?

**Jeffrey conditionalization** (Richard Jeffrey, 1965) generalizes Bayesian updating: the evidence is not a certain event $E$ (whose probability jumps to 1), but a change in belief over an evidence partition $\{E_1, E_2, \ldots, E_k\}$—from the old $P_{\text{old}}(E_i)$ to new $Q_{\text{new}}(E_i)$. Then for any hypothesis $H$:

$$P_{\text{new}}(H) = \sum_{i=1}^k P_{\text{old}}(H \mid E_i) \cdot Q_{\text{new}}(E_i)$$

This formula says: when your beliefs over the evidence partition $\{E_i\}$ change, your belief in $H$ changes accordingly—the amount of change is the old conditional probability of $H$ given each $E_i$, multiplied by the new degree of belief in each $E_i$.

**Example**: You hear a noise outside the window ($S$). Your beliefs: it might be a bird ($B$, belief rises to 0.6), might be wind ($W$, belief rises to 0.3), might be something else ($O$, belief 0.1). You care about the proposition "there is a living thing outside the window" ($L$). Known old conditional probabilities: $P(L \mid B)=1$, $P(L \mid W)=0$, $P(L \mid O)=0.5$. Then:

$$P_{\text{new}}(L) = 1 \times 0.6 + 0 \times 0.3 + 0.5 \times 0.1 = 0.65$$

Jeffrey conditionalization is the generalization of Bayesian updating for "uncertain evidence." Bayesian updating is its limiting case—when the new belief in some $E_i$ jumps to 1 and the others to 0.

**Why does this matter?** Because in reality, almost all evidence is uncertain. What you see, hear, and read can rarely be accepted with probability 1. Jeffrey conditionalization enables probability logic to handle "I saw something uncertain"—which is the norm of everyday reasoning, not the exception.

---

## 17.5 Logic and Probability: The Continuization of Truth Values

Return to the basic questions of Chapter 14: soundness and completeness.

In classical logic, these two properties speak of the relationship between syntax ($\vdash$) and semantics ($\vDash$): whatever can be proved is true (soundness), and whatever is true can be proved (completeness).

In probability theory, what do these two layers of relationship become?

**Probability's "soundness" counterpart**: Bayesian updating preserves coherence. If your initial beliefs satisfy the probability axioms, the beliefs after Bayesian updating also satisfy them. Inference does not manufacture internal contradictions; it does not make you simultaneously hold positive probability and negative probability for the same thing. This is the probabilistic version of soundness.

**Probability's "completeness" problem**: what does classical logic's incompleteness (Gödel's theorem) become within the probability framework? This is a more subtle question. Probabilistic inference does not encounter "unprovable propositions" — because every proposition always has a probability, even if it is the a priori given 0.5 (complete uncertainty). But this does not mean that all truths can be discovered by probabilistic inference — it only means that uncertainty is explicitly quantified, rather than logically blocked.

::: info Formal logic and probability: not competitors, but different scales of truth

Formal logic and probability theory are often placed side by side as two "methods of reasoning," as if you must choose only one. But more precisely, they live at different levels: formal logic handles **completely certain information** (a proposition is either true or false in the model); probability theory handles **incomplete information** (a proposition may be true, may be false; my degree of belief is $p$).

A more complete picture is: formal logic is the limiting case of probability theory when the truth-value domain degenerates to $\{0,1\}$. When the probabilities of all propositions are either 0 or 1 (a fully informed agent), Bayesian inference degenerates into Boolean inference. This is not a denial of formal logic, but its generalization.
:::

---

## 17.6 The Chain of Belief Updating: From Inference to Learning

Bayesian updating is a single-step operation: obtain one piece of evidence, update belief once. But inference is usually sequential — you observe evidence one after another, updating each time.

The structure of this chained operation is precisely the formal foundation of machine learning.

Imagine a parameter $\theta$ that determines the behavior of a model (for instance, the true probability of heads of some coin). You have a prior $P(\theta)$ over $\theta$. Then you observe data $D = \{x_1, x_2, \ldots, x_n\}$, updating one at a time:

$$P(\theta \mid x_1) \propto P(x_1 \mid \theta) \cdot P(\theta)$$
$$P(\theta \mid x_1, x_2) \propto P(x_2 \mid \theta) \cdot P(\theta \mid x_1)$$
$$\vdots$$
$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta) = \left(\prod_{i=1}^n P(x_i \mid \theta)\right) \cdot P(\theta)$$

The final posterior $P(\theta \mid D)$ is your belief distribution over the parameter $\theta$ after seeing all the data.

This is **Bayesian learning**: learning is not about finding the "correct" parameter, but about pushing the belief distribution over the parameter from prior to posterior. The parameter is not a point, but a distribution — how much confidence you have in it is told by the width of that distribution.

This framework forms an interesting dialogue with the discussion of overfitting in Chapter 5 of the previous volume: overfitting occurs because the model treats noise in the training data as signal, whereas the Bayesian framework naturally resists overfitting — the regularization term corresponds precisely to the prior's constraint on the parameters. A loose prior corresponds to weak regularization; a sharp prior (concentrated in a specific parameter range) corresponds to strong regularization. Occam's razor — "simpler explanations are preferred" — gains precise mathematical expression in the Bayesian framework: complex models need more data to overcome the prior's preference for simplicity.

::: info The relationship between Maximum A Posteriori (MAP) and Maximum Likelihood Estimation (MLE)

In the Bayesian framework, a commonly used "point estimate" approach is to take the mode of the posterior: $\hat{\theta} = \arg\max_\theta P(\theta \mid D)$, called the maximum a posteriori estimate (MAP). Expanding:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log P(D \mid \theta) + \log P(\theta)\right]$$

If the prior $P(\theta)$ is uniform (all parameters equally likely), $\log P(\theta)$ is constant, and MAP degenerates to maximum likelihood estimation (MLE): $\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta)$. MLE is Bayesian inference with an "unbiased prior." This derivation shows that maximum likelihood estimation is not an independent principle of inference, but a special case of Bayesian inference under a uniform prior.
:::

---

## 17.7 What Probability Cannot Capture

At this point, probability theory seems almost omnipotent: it generalizes logical inference to continuous truth values, explains the necessary form of rational belief, and provides a formal framework for learning.

But it has a fundamental limitation, whose shadow we already glimpsed in Chapter 6 of the previous volume; here it needs to be stated clearly in formal language.

**Probability describes correlation, not causation.**

Consider two variables $X$ and $Y$, whose joint distribution $P(X, Y)$ is fully known. You can compute $P(Y \mid X = x)$ — the conditional distribution of $Y$ given that $X$ takes a certain value. But this conditional probability cannot distinguish among the following three situations:

1. $X$ causes $Y$ (causal: $X \to Y$)
2. $Y$ causes $X$ (causal: $Y \to X$)
3. $X$ and $Y$ are both effects of some common cause $Z$ (confounding: $X \leftarrow Z \to Y$)

All three situations can produce exactly the same joint distribution $P(X, Y)$. Probability — no matter how many times you update, no matter how much data you observe — cannot, from the data alone, distinguish these three situations.

This is not a flaw of the method, but a structural limitation of mathematics: **information about association does not contain information about causal direction**.

::: info Professor Manul comments
This is the sentence most easily skipped in the entire probability theory curriculum — and the most costly one. Countless papers use conditional probability to answer causal questions. Not because the authors are stupid, but because no one drew this wall clearly at the very beginning. Correlation and causation have different mathematical structures; it is not a difference in degree, but a difference in kind. That's all.
:::

If you want to infer causation — to answer "if I intervene on $X$, how will $Y$ change" — you need tools stronger than probability. That tool is precisely the protagonist of Chapter 18: causal calculus (do-calculus) and structural causal models.

---

## Unresolved

**Where does the boundary of subjectivity lie?** Cox's theorem proves the uniqueness of the inference rules, but does not prescribe the prior. Different priors are given to different agents; observing the same data, will they ultimately reach consensus? Under what conditions will they, and under what conditions won't they? This is the "prior selection" problem in Bayesian statistics, for which there is still no universal answer.

**Is quantum probability a generalization of Bayesian probability?** The probability in quantum mechanics — the Born rule — has a similar mathematical structure to Bayesian probability, but the collapse mechanism of quantum states and classical Bayesian updating have essential differences. Does there exist a unified framework that incorporates both classical probability and quantum probability into "the logic of rational belief"? This is the question that Quantum Bayesianism (QBism) attempts to answer; the answer is still under debate.

**Is probability the ceiling of inference?** Section 17.7 has already revealed the answer to this question: no. Probability cannot distinguish correlation from causation — $X$ and $Y$ are highly correlated, but you do not know whether $X$ causes $Y$, or $Y$ causes $X$, or some latent variable $Z$ simultaneously drives both. No matter how much data you observe, no matter how many times you perform Bayesian updates, the answer to this question forever hides beyond the visible range of probability.

This is not a flaw of the method, but a structural fact of mathematics: **observational information does not contain interventional information**. To answer "if I change $X$, what will happen to $Y$," you need a new kind of inference rule — one that formalizes the act of "changing" itself. This is the starting point of Chapter 18.

---

## Exercises

**★ Warm-up**

A medical test for a certain disease has sensitivity 90% and specificity 95%. That is: sick people have a 90% probability of testing positive; healthy people have a 95% probability of testing negative. The disease prevalence in the population is 1%.

First, estimate intuitively: someone tests positive; what is approximately the probability that they actually have the disease? Write down your intuitive answer, then compute the exact value using Bayes' theorem.

$$P(\text{sick} \mid \text{positive}) = \frac{P(\text{positive} \mid \text{sick}) \cdot P(\text{sick})}{P(\text{positive})}$$

(Hint: $P(\text{positive}) = P(\text{positive} \mid \text{sick}) \cdot P(\text{sick}) + P(\text{positive} \mid \text{healthy}) \cdot P(\text{healthy})$. Plug in the numbers and see how much the result differs from your intuition.)

---

**★★ Derivation**

Under the setup of the previous problem:

1. After the first test returns positive, take the first posterior as the new prior, and perform a second independent test, which also returns positive. What is the probability of having the disease now?
2. If this person comes from a high-risk group with disease prevalence 10% (rather than 1%), after one positive test, what is the probability of having the disease?
3. Compare the results of problem 1 (two positives, low-risk group) and problem 2 (one positive, high-risk group). Which scenario yields a higher disease probability? What does this illustrate about the relationship between prior and evidence?

---

**★★★ Challenge**

Cox's theorem proves: any measure of belief satisfying the three requirements of rationality is structurally equivalent to probability. But the first requirement of the theorem is that "belief can be represented by a real-number linear order."

Try to construct an inference scenario that you consider reasonable, in which belief in a certain proposition **cannot be fully expressed by a single real number** — perhaps it needs two numbers (e.g., "at least 0.3, at most 0.8"), or it needs a distribution.

Do such scenarios exist? If they do, does it mean Cox's theorem's premises are not sufficiently general, or does it merely mean that "a certain precisification of belief" is inapplicable in this scenario? Try to distinguish these two possibilities using the language of this chapter — no need to resolve, only need to state the question clearly.

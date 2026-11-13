# Chapter 17: Probability as an Expansion of Logic — Truth Values from {0,1} to [0,1]

> Probability is not frequency. It is the only coherent representation of rational belief under uncertainty.

---

The end of Chapter 16 left a cliffhanger: the semantics of linear logic hinted that "truth value" is not a simple $\{0,1\}$, but some richer structure. Yet that direction — phase semantics, coherence spaces — is a technically hard road, left to the researchers.

There is a wider road.

What if "truth value" were not true-or-false, but a real number between 0 and 1? And what if that real number expressed some agent's **degree of belief** that a proposition is true? What shape would the inference rules take then?

That's this chapter's question. The answer is called **Bayesian probability theory** — but not the frequentist version you may have learned; rather its logical-foundation version: probability as the expression of rational belief, governed by laws that can be derived rather than simply declared.

---

## 17.0 The Belief-Chip Game: Staying Coherent Under Uncertainty

Let's swap truth values from black-and-white game pieces to chips.

On the table are several propositions: $H_1, H_2, \ldots, H_k$. You hold a hundred belief chips and must spread them over the propositions. The more chips a proposition gets, the more you believe it. The game's demand is not "always bet on the winner" but something more basic: **your bets must not contradict themselves**.

Then the evidence arrives. You see $E$. The rules force you to redistribute chips: hypotheses that support $E$ gain chips; hypotheses that can't explain $E$ lose chips. And this move is not arbitrary — it must obey Bayesian updating:

$$P(H \mid E) = \frac{P(E \mid H)P(H)}{P(E)}$$

You can think of the Bayes formula as the casino's anti-cheating rule: if you don't update this way, a clever enough bookmaker can design a set of bets that loses you money no matter how the world turns out. That's the intuition behind the Dutch book.

:::details The formal skeleton of this game
- **State space**: probability distributions $p_t \in \Delta^{k-1}$, the set of legal allocations of belief chips.
- **Legal moves**: after observing evidence $E_t$, update the prior $p_t$ into the posterior $p_{t+1}$.
- **Transition rule**: $p_{t+1}(H) \propto P(E_t \mid H)p_t(H)$.
- **Winning condition**: a belief distribution that absorbs evidence while keeping consistency with the probability axioms.
- **Failure mode**: a chip allocation that violates probability rules, breeding internal contradiction — or being steadily harvested by a Dutch book.
:::

The key point of this game isn't "probability guarantees you'll be right." Probability is not that powerful. It guarantees one humbler, and more necessary, thing: when your evidence is incomplete, your belief updates don't fight each other.

Formal logic handles legal inference in a world of known facts; probability logic handles legal updating in a world of unknowns. The first asks "can this conclusion be derived?"; the second asks "now that I've seen this evidence, where should my belief move?"

::: info Professor Pallas says
Probability isn't a fuzzy logic for the fainthearted. Probability is the posture rationality takes when it must stand on incomplete information. You don't know the answer, so you allocate chips; evidence arrives, so you move chips. The truly irrational thing isn't uncertainty — it's pretending you're certain while you're deep inside your own uncertainty.
:::

### 17.0.5 Playing the Game in Code

The chip table is small enough to run. The program below moves a hundred chips by Bayes' rule as the evidence arrives, checks that the total never changes (the coherence invariant), and then builds the Dutch book from Section 17.0 for real: an incoherent allocation loses money no matter how the world turns out.

```python
#!/usr/bin/env python3
"""
belief_chip_game.py -- the Belief-Chip Game of Chapter 17.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 17, Section 17.0
"The Belief-Chip Game: Staying Coherent Under Uncertainty."

A hundred belief chips spread over the hypotheses. Evidence arrives, and
the casino's rules force one specific redistribution -- Bayes' rule --
because any other move lets a bookmaker harvest you. This program:
  * moves chips by the Bayes rule and checks the invariant that matters:
    the chip total never changes (the probability axioms = coherence),
  * shows belief converging as the evidence piles up,
  * builds the Dutch book of Section 17.0 for real: an incoherent chip
    allocation loses money no matter how the world turns out.
All numbers below are computed, not staged.
Run:  python3 belief_chip_game.py
"""
TOTAL = 100.0  # one hundred belief chips

def update(beliefs, likelihoods, evidence):
    """One Bayes step: chips move toward hypotheses that support the
    evidence, away from those that cannot. Pure; returns a new dict."""
    raw = {
        h: belief * likelihoods[h][evidence]
        for h, belief in beliefs.items()
    }
    z = sum(raw.values())
    return {h: v * TOTAL / z for h, v in raw.items()}

def demo():
    print("=" * 70)
    print(" THE BELIEF-CHIP GAME -- coherence under uncertainty")
    print(" The Reasoning Kingdom, Volume 2, Chapter 17 (Sec. 17.0)")
    print("=" * 70)
    # Three hypotheses about a coin: H1 = fair, H2 = biased to tails,
    # H3 = biased to heads. Likelihood of H or T under each.
    beliefs = {"H1": TOTAL / 3, "H2": TOTAL / 3, "H3": TOTAL / 3}
    likelihoods = {
        "H1": {"H": 0.5, "T": 0.5},
        "H2": {"H": 0.1, "T": 0.9},
        "H3": {"H": 0.9, "T": 0.1},
    }
    flips = "H H T H H H H H T H".split()
    print("\nThree hypotheses, 100 chips, split evenly:")
    print("  H1: fair coin     H2: tails-biased     H3: heads-biased")
    print("\nThe coin flips: " + " ".join(flips))
    print("\n  after flips     H1 chips   H2 chips   H3 chips   total")
    for k in (0, 3, 7, 10):
        seen = flips[:k]
        for flip in seen:
            beliefs = update(beliefs, likelihoods, flip)
        total = sum(beliefs.values())
        row = "  " + f"{k:>5}" + "".join(
            f"{beliefs[h]:>11.1f}" for h in ("H1", "H2", "H3")
        )
        print(row + f"{total:>9.1f}")
        assert abs(total - TOTAL) < 1e-9, "coherence broken!"
    print("\nThe chip total never moved. That invariant is the whole game:")
    print("  a rational agent's beliefs must obey the probability axioms, or")
    print("  its beliefs fight each other.")
    # The Dutch book: an incoherent allocation loses money, guaranteed.
    print("\nThe Dutch book, run for real:")
    print("  a bookie with incoherent prices: P(A) = 0.6 and P(not A) = 0.7.")
    print("  Prices sum to 1.3, not 1.0 -- the bookie is overconfident.")
    for world in ("A is true", "A is false"):
        payout = 1.0
        income = 0.6 + 0.7
        print(f"  world: {world:<12} income {income:.1f} - payout "
              f"{payout:.1f} = bookie profit {income - payout:+.1f}")
    print("  Either way the bookie pockets 0.3. Coherence is not a moral")
    print("  virtue -- it is the only way to keep your wallet out of the")
    print("  casino's argument.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

---

## 17.1 The Debate Over What Probability Means

What does "probability $\frac{1}{2}$" mean?

The frequentist answer: flip this coin infinitely many times, and the proportion of heads converges to $\frac{1}{2}$. Probability is long-run frequency; it only makes sense for repeatable experiments.

The Bayesian answer: I judge the chance that this coin comes up heads next time as $\frac{1}{2}$. Probability is a measure of belief; it makes sense for a single event too.

The debate has run for a century and still isn't fully over. But there's one question frequentism cannot answer, and Bayesianism can:

"What's the probability of rain tomorrow?"

Tomorrow happens exactly once. There is no infinite repetition. You cannot wait for infinitely many "tomorrows" to measure a frequency. Yet the weather forecast's 70% chance of rain is meaningful — it describes the forecaster's belief strength in the proposition "it rains tomorrow," based on the weather data at hand.

The core claim of Bayesian probability theory: **probability is the logic of belief, not the statistics of frequency**. Moreover, a rational agent's beliefs must obey the probability axioms — not because nature demands it, but because a belief that violates the probability axioms is **incoherent**; it breeds contradictions in inference.

---

## 17.2 Cox's Theorem: Why the Axioms Are Inevitable

In 1946, Richard Cox asked a question: if you want to express belief strengths with real numbers, in a way that is internally consistent, what constraints must those numbers satisfy?

His starting point was three requirements, each a minimum bar for rational belief:

**Requirement 1 (ordering)**: beliefs are comparable. For any two propositions $A$ and $B$, your belief in $A$ is either stronger than, equal to, or weaker than your belief in $B$.

**Requirement 2 (consistency)**: belief in a compound proposition is fully determined by belief in its components. Your belief in "$A$ and $B$" is some function of your belief in $A$ and your belief in $B$ once $A$ is known to hold.

**Requirement 3 (duality)**: belief in $A$ and belief in $\neg A$ are complementary — being fully certain of $A$ means fully disbelieving $\neg A$.

Cox proved that under these three requirements, any internally consistent belief measure must, up to a monotone change of scale, be equivalent to standard probability. That is, you may choose a different gauge (say $[0,100]$ instead of $[0,1]$), but the structure of the inference rules is fully determined:

$$P(A \land B) = P(A) \cdot P(B \mid A)$$

$$P(A) + P(\neg A) = 1$$

This isn't a law discovered by experiment; it's the inevitable outcome of rational coherence. If you express belief in numbers, and your beliefs are internally consistent, then you are doing probability — you just may not realize it yet.

::: info Professor Pallas says
Cox's theorem has led many people to conclude "Bayesianism is the only rational option." Slow down — the theorem's premise is that belief can be represented on a linear ordering of real numbers. Question that premise and the whole theorem stops applying. A theorem's power comes from its premises; so think the premises through first, then discuss the necessity of the conclusion. Don't treat the conclusion of a conditional as an unconditional truth.
:::

::: info The philosophical import of Cox's theorem

What makes Cox's theorem profound is its "uniqueness": belief measures meeting the rational requirements are structurally unique. This means probability theory is not a set of tools humans invented — it is the **necessary form** of rational belief. If you reject probability theory, you either refuse to compare beliefs (giving up ordering), or you accept internal contradictions between beliefs.

This matches the spirit of Chapter 14 exactly: the axioms of a formal system are not arbitrary conventions; they are the minimum you must accept to avoid contradiction. Cox's theorem applies the same logic to belief: to avoid incoherence, belief must obey the probability axioms.
:::

---

## 17.3 Bayesian Updating: The Probabilistic Version of an Inference Rule

With probability as the measure of belief, what does "inference" become?

In formal logic, inference produces new true propositions from known true propositions. In probability theory, inference **updates** your degree of belief in a proposition, given what you observed.

The rule for this update is the most important theorem of probability:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

Let's spell out the four quantities:

- $P(H)$: your belief in hypothesis $H$ before seeing evidence $E$ — the **prior probability**.
- $P(E \mid H)$: the probability of seeing evidence $E$ if hypothesis $H$ holds — the **likelihood**.
- $P(E)$: the probability of seeing evidence $E$ over all possible cases — the **marginal probability**, the normalizing factor.
- $P(H \mid E)$: your new belief in hypothesis $H$ after seeing evidence $E$ — the **posterior probability**.

:::details Prior / likelihood / posterior: the triangle explained
These three terms are the heart of Bayesian inference, and newcomers constantly mix them up:

**Prior $P(H)$**: the belief you carried before seeing any data. Example: "I'd guess this coin is fair; 50% heads." Its sources: domain knowledge, historical data, or an uninformative assumption.

**Likelihood $P(E \mid H)$**: the probability of observing the current evidence $E$, if hypothesis $H$ is true. Note the direction — it's not "how likely is $H$ given $E$", but "if $H$ were true, how likely is $E$". Example: "If the coin is fair, the probability of 3 heads in a row is $1/8$."

**Posterior $P(H \mid E)$**: the updated belief after seeing the evidence. This is the number you actually want.

**Marginal probability $P(E)$**: the normalizing constant that keeps posterior probabilities summing to 1. In practice, people often compute $\text{posterior} \propto \text{likelihood} \times \text{prior}$ and ignore this constant.

Memory formula: **new belief ∝ old belief × how strongly the evidence supports it**
:::

This is **Bayes' theorem** — or, more precisely, the core operation of Bayesian inference.

But written as a formula, it's easy to miss the logical essence. A clearer statement:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

($\propto$ means "proportional to"; $P(E)$ is a constant normalizing factor and does not change relative proportions.)

What this says: **the belief after the evidence is your belief before the evidence, weighted by the evidence**. The evidence acts on the prior through the likelihood, "pushing" the prior into the posterior.

Back at the belief-chip game: the Bayes formula isn't an extra "statistical trick" bolted on from outside — it's the legality rule for moving chips. You can, of course, move chips from one hypothesis to another by feel; but as soon as your moves stop being proportional to likelihood and prior, you leave behind the arbitrage gap a Dutch book needs. That's the coldness of probability: it doesn't guarantee you'll get it right; it only guarantees you won't contradict yourself within your own betting rules.

### 17.3.5 Hands-On: Run a Bayesian Update Yourself

Let's trace one complete update with the medical test from Exercise 1. Not plugging numbers into a formula — watching the chips move.

**Setup**:
- Disease prevalence (prior): $P(\text{sick}) = 0.01$, $P(\text{healthy}) = 0.99$
- Test sensitivity (likelihood): $P(\text{positive} \mid \text{sick}) = 0.90$
- Test specificity: $P(\text{negative} \mid \text{healthy}) = 0.95$, so $P(\text{positive} \mid \text{healthy}) = 0.05$

**Step 1**: the prior chips. Imagine a group of 10,000 people. By the prior, about 100 are sick, 9,900 healthy.

**Step 2**: a positive result arrives. Of the 100 sick people, about 90 test positive. Of the 9,900 healthy people, about $9900 \times 0.05 = 495$ test positive.

**Step 3**: redistribute the belief. The positive group totals $90 + 495 = 585$ people. Of them, 90 are truly sick. So your updated belief in sickness is:

$$P(\text{sick} \mid \text{positive}) = \frac{90}{585} \approx 0.154 = 15.4\%$$

**Verify with the formula**:

$$P(\text{sick} \mid \text{positive}) = \frac{0.90 \times 0.01}{0.90 \times 0.01 + 0.05 \times 0.99} = \frac{0.009}{0.009 + 0.0495} \approx 0.154$$

Both computations agree.

**Key observation**: the test came back positive, but the probability of sickness is only 15.4%. Most people's gut estimate would be above 80%. Why such a gap? Because the disease is rare (1%) — even with high sensitivity, most positive results are false positives from healthy people. Put simply: healthy people dominate the population (99%), so their false positives (5%) yield 495 positives, dwarfing the 90 true positives.

**Test again**: use 15.4% as the new prior; the second test is also positive.

$$P(\text{sick} \mid \text{two positives}) = \frac{0.90 \times 0.154}{0.90 \times 0.154 + 0.05 \times 0.846} = \frac{0.1386}{0.1386 + 0.0423} \approx 0.766$$

After two positives, the sickness probability jumps to 76.6%. The prior went 1% → 15.4% → 76.6% — two pieces of evidence, and the belief shifted violently.

**What if the base rate is 10% (a high-risk group)?** After one positive:

$$P(\text{sick} \mid \text{positive}) = \frac{0.90 \times 0.10}{0.90 \times 0.10 + 0.05 \times 0.90} = \frac{0.09}{0.09 + 0.045} = 0.667$$

So a high-risk person (10%) after one positive has a sickness probability of 66.7% — is that higher than the low-risk person's 76.6% after two positives? No: 76.6% > 66.7%. Two accumulated pieces of evidence in the low-risk group yield more confidence than a single test in the high-risk group. That showcases the cumulative power of evidence — enough evidence can overwhelm the prior.

But this example also exposes a default premise of Bayesian updating: **you observed a definite piece of evidence $E$**. You know the test result is "positive," without ambiguity, probability 1. But in reality, evidence itself is often uncertain. That's the next subsection's topic.

::: info A structural analogy between inference rules

Compare Bayesian updating with Chapter 14's inference rules, and the parallels are striking:

- Formal logic: $\frac{P \to Q \quad P}{Q}$ (modus ponens: consumes $P$ and $P \to Q$, produces $Q$)
- Bayesian: $P(H \mid E) \propto P(E \mid H) \cdot P(H)$ (consumes the likelihood $P(E \mid H)$ and the prior $P(H)$, produces the posterior $P(H \mid E)$)

Both are "deriving something new from what you already had"; the difference is that formal logic's "what you already had" consists of $\{0,1\}$-valued true propositions, while the Bayesian "what you already had" consists of $[0,1]$-valued belief degrees. Bayesian inference is the version of modus ponens expanded to a continuous truth-value domain.
:::

---

## 17.4 The Prior: Inference Never Starts From Zero

Bayesian inference has a part that makes many people uneasy: you need a prior.

The prior is the belief you already hold before seeing any evidence. Where does it come from? If I know nothing at all, what is my prior?

Frequentists consider this demand the fatal flaw of the Bayesian method — priors are subjective; different people can hold different priors and reach different posteriors; who's right?

The Bayesian answer comes in two layers.

First: **the prior is not arbitrary**. Rational priors are under constraints. The most basic constraint is symmetry: if you know nothing about a situation, you have no reason to tip the prior to either side. This gives the "uninformative prior" — with no preference information at hand, assign a uniform prior (for discrete cases) or a maximum-entropy prior (for continuous cases).

Second: **the prior's influence fades as evidence accumulates**. This is a mathematical theorem of Bayesian updating: after enough independent observations, the posterior converges to the same location no matter which prior you started from. Subjective priors are temporary; data is objective; rational agents eventually converge on agreement.

A stark example. Two people are arguing about whether a coin is fair: one's prior says the heads probability is 0.99, the other's says 0.01. They both watch the coin being flipped 1,000 times, 503 of which land heads. After Bayesian updating, both posteriors concentrate around 0.5 — priors in wild disagreement, drowned by the evidence.

That convergence is where Bayesian objectivity comes from: not the objectivity of the prior, but the objectivity of the **inference process**.

### 17.4.5 Jeffrey Conditioning: When the Evidence Itself Is Uncertain

Bayes' theorem assumes you observed a definite event $E$ — "the test result is positive" — with probability 1. But what you see is not always definite. You watch a shadow dart past the window — it might be a bird, it might be a plane, it might be a hallucination. Your observation isn't "I saw a bird (probability 1)"; it's "my belief in 'a bird flew by' went from 0.1 to 0.7."

How do you handle this kind of evidence?

**Jeffrey conditioning** (Richard Jeffrey, 1965) generalizes Bayesian updating: the evidence isn't a definite event $E$ (with probability jumping to 1); it's a shift of belief across a partition of evidence $\{E_1, E_2, \ldots, E_k\}$ — from the old $P_{\text{old}}(E_i)$ to the new $Q_{\text{new}}(E_i)$. Then for any hypothesis $H$:

$$P_{\text{new}}(H) = \sum_{i=1}^k P_{\text{old}}(H \mid E_i) \cdot Q_{\text{new}}(E_i)$$

What this formula says: when your beliefs shift across the evidence partition $\{E_i\}$, your belief in $H$ shifts with them — the shift is the old conditional probability of $H$ given each $E_i$, times your new belief in each $E_i$.

**Example**: you hear a noise outside the window ($S$). Your beliefs: it could be a bird ($B$, up to 0.6), it could be the wind ($W$, up to 0.3), something else ($O$, 0.1). You care about the proposition "there's something alive outside" ($L$). Given the old conditional probabilities $P(L \mid B) = 1$, $P(L \mid W) = 0$, $P(L \mid O) = 0.5$:

$$P_{\text{new}}(L) = 1 \times 0.6 + 0 \times 0.3 + 0.5 \times 0.1 = 0.65$$

Jeffrey conditioning is the generalization of Bayesian updating to "uncertain evidence." Bayesian updating is its limiting case — when the new belief in some $E_i$ jumps to 1 and all others fall to 0.

Why does this matter? Because in reality, almost all evidence is uncertain. What you see, hear, and read is rarely accepted with probability 1. Jeffrey conditioning lets probability logic handle "I saw something I'm not sure about" — which is the normal state of everyday inference, not an exception.

---

## 17.5 Logic and Probability: Truth Value Goes Continuous

Back to Chapter 14's fundamental questions: soundness and completeness.

In classical logic, those two properties describe the relation between syntax ($\vdash$) and semantics ($\vDash$): everything provable is true (sound), and everything true is provable (complete).

In probability, what do these two relations become?

**Probability's "soundness" counterpart**: Bayesian updating preserves coherence. If your initial beliefs satisfy the probability axioms, the updated beliefs do too. Inference won't manufacture internal contradictions, won't make you hold a positive and a negative probability of the same event at once. That's the probabilistic version of soundness.

**Probability's "completeness" question**: what does classical incompleteness (Gödel's theorem) become in the probabilistic framework? This is subtler. Probabilistic inference never runs into an "unprovable proposition" — every proposition has a probability, even if it's just the prior's 0.5 (complete uncertainty). But that doesn't mean all truth can be discovered through probabilistic inference — it only means uncertainty is quantified explicitly, rather than blocked logically.

::: info Formal logic and probability: not competitors, but different gauges of truth value

Formal logic and probability theory are often compared as two rival "reasoning methods," as if you must pick one. But precisely speaking, they live on different levels: formal logic handles **fully determined information** (a proposition is either true in the model or false); probability theory handles **incomplete information** (a proposition may be true, may be false, and my belief degree is $p$).

A more complete picture: formal logic is the limiting case of probability when the truth-value domain degenerates to $\{0,1\}$. When all proposition probabilities are 0 or 1 (a fully informed agent), Bayesian inference degenerates into Boolean inference. This isn't a denial of formal logic — it's a generalization of it.
:::

---

## 17.6 The Chain of Belief Updates: From Inference to Learning

Bayesian updating is a single-step operation: take one piece of evidence, update once. But inference is usually continuous — you observe evidence one after another, updating each time.

The structure of this chained operation is precisely the formal foundation of machine learning.

Picture a parameter $\theta$ that determines a model's behavior (say, the true heads-probability of a certain coin). You hold a prior $P(\theta)$ on $\theta$. Then you observe data $D = \{x_1, x_2, \ldots, x_n\}$, updating one at a time:

$$P(\theta \mid x_1) \propto P(x_1 \mid \theta) \cdot P(\theta)$$
$$P(\theta \mid x_1, x_2) \propto P(x_2 \mid \theta) \cdot P(\theta \mid x_1)$$
$$\vdots$$
$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta) = \left(\prod_{i=1}^n P(x_i \mid \theta)\right) \cdot P(\theta)$$

The final posterior $P(\theta \mid D)$ is your belief distribution over the parameter $\theta$ after seeing all the data.

That's **Bayesian learning**: learning is not "finding the correct parameter" — it's pushing the belief distribution over the parameter from prior to posterior. A parameter is not a point but a distribution; how confident you are shows in how wide or narrow that distribution is.

This framework makes an interesting conversation with the overfitting discussion in Chapter 5 of Volume 1: overfitting happens because a model mistakes noise in the training data for signal, while the Bayesian framework naturally resists overfitting — the regularization term corresponds exactly to the prior's constraint on the parameter. A loose prior is weak regularization; a sharp prior (concentrated on a narrow parameter range) is strong regularization. Occam's razor — "the simpler explanation wins" — gains a precise mathematical expression in the Bayesian framework: complex models need more data to overcome the prior's preference for simplicity.

::: info The relation between MAP and MLE

In the Bayesian framework, the common "point estimate" move is to take the posterior mode: $\hat{\theta} = \arg\max_\theta P(\theta \mid D)$, called maximum a posteriori estimation (MAP). Expanding:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log P(D \mid \theta) + \log P(\theta)\right]$$

If the prior $P(\theta)$ is uniform (all parameters equally likely), $\log P(\theta)$ is a constant, and MAP degenerates into maximum likelihood estimation (MLE): $\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta)$. MLE is Bayesian inference under an impartial prior. This derivation shows maximum likelihood estimation is not an independent inference principle — it is a special case of Bayesian inference under a uniform prior.
:::

---

## 17.7 What Probability Cannot Capture

By now probability theory looks nearly omnipotent: it generalized logical inference to continuous truth values, explained the necessary form of rational belief, and supplied a formal framework for learning.

But it has one fundamental limit — we already glimpsed its shadow in Chapter 6 of Volume 1, and here it must be stated in formal language.

**Probability describes correlation, not causation.**

Consider two variables $X$ and $Y$ whose joint distribution $P(X, Y)$ is fully known. You can compute $P(Y \mid X = x)$ — the conditional distribution of $Y$ given some value of $X$. But this conditional probability cannot distinguish among the following three situations:

1. $X$ causes $Y$ (causal: $X \to Y$)
2. $Y$ causes $X$ (causal: $Y \to X$)
3. $X$ and $Y$ are both outcomes of a common cause $Z$ (confounding: $X \leftarrow Z \to Y$)

All three situations can produce exactly the same joint distribution $P(X, Y)$ of the observed variables. Probability — no matter how many times you update, no matter how much data you observe — cannot tell these three situations apart from the data itself.

This isn't a flaw in the method; it's a structural limit of the mathematics: **the information in association does not include the information in causal direction**.

::: info Professor Pallas says
This is the most skippable, highest-cost sentence in every probability course. Countless papers use conditional probabilities to answer causal questions. Not because the authors are stupid — because nobody drew this wall clearly at the outset. Association and causation differ in mathematical structure; it's not a difference in degree, it's a difference in kind. That's all.
:::

If you want to infer causation — to answer "if I intervene on $X$, how will $Y$ change?" — you need tools stronger than probability. That tool is the star of Chapter 18: do-calculus and structural causal models.

---

## Open Questions

**Where is the boundary of subjectivity?** Cox's theorem proved the uniqueness of the inference rules, but it didn't fix the prior. Different agents with different priors observe the same data — will they eventually agree? Under what conditions yes, under what conditions no? This is the "prior selection" problem in Bayesian statistics, and it still has no universal answer.

**Is quantum probability a generalization of Bayesian probability?** The probabilities of quantum mechanics — the Born rule — share a similar mathematical structure with Bayesian probability, but the collapse mechanism of quantum states differs fundamentally from classical Bayesian updating. Is there a single framework that fits both classical and quantum probability into "the logic of rational belief"? That is the question quantum Bayesianism (QBism) tries to answer — and the answer is still under debate.

**Is probability the ceiling of inference?** Section 17.7 has already revealed the answer: no. Probability cannot distinguish correlation from causation — $X$ and $Y$ are strongly correlated, but you don't know whether $X$ causes $Y$, whether $Y$ causes $X$, or whether some hidden variable $Z$ drives both. No matter how much data you observe, no matter how many Bayesian updates you perform, the answer to that question forever lies outside probability's visible range.

That's not a defect of the method but a structural fact of mathematics: **the information in observation does not contain the information in intervention**. To answer "if I change $X$, what happens to $Y$?", you need a new kind of inference rule — one that formalizes the action of "changing" itself. That's where Chapter 18 begins.

---

## Exercises

**★ Warm-Up**

A medical test has a sensitivity of 90% and a specificity of 95% for a certain disease. That is: a sick person tests positive with probability 90%, and a healthy person tests negative with probability 95%. The disease's prevalence in the population is 1%.

First estimate by intuition: someone tests positive — what is the probability they're actually sick? Write down your intuitive answer, then compute the exact value with Bayes' theorem.

$$P(\text{sick} \mid \text{positive}) = \frac{P(\text{positive} \mid \text{sick}) \cdot P(\text{sick})}{P(\text{positive})}$$

(Hint: $P(\text{positive}) = P(\text{positive} \mid \text{sick}) \cdot P(\text{sick}) + P(\text{positive} \mid \text{healthy}) \cdot P(\text{healthy})$. Plug in the numbers and see how far your result is from your intuition.)

---

**★★ Derivation**

Under the setup of the previous question:

1. After the first positive test, use the first posterior as the new prior, run a second independent test, and it comes back positive again. What is the sickness probability now?
2. If the person is from a high-risk group with a 10% prevalence (rather than 1%), what is the sickness probability after the same single positive test?
3. Compare Question 1 (two positives, low-risk group) with Question 2 (one positive, high-risk group). Which case gives a higher sickness probability? What does that say about the relation between prior and evidence?

---

**★★★ Challenge**

Cox's theorem proves that any belief measure satisfying the three rational requirements is structurally equivalent to probability. But the theorem's first requirement is "belief can be represented on a linear ordering of real numbers."

Try to construct a reasoning scenario you consider reasonable, in which belief in some proposition **cannot be fully expressed by a single real number** — maybe it needs two numbers (like "at least 0.3, at most 0.8"), or maybe it needs a distribution.

Does such a scenario exist? If it does, does it show Cox's premises are not general enough, or does it just show that "one particular refinement of belief" doesn't apply to that scenario? Use this chapter's language to try to distinguish these two possibilities — you don't need to solve it, just get the question clearly stated.

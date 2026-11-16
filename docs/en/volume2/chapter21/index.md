# Chapter 21: Learning as Inverse Inference — Generalization Is Another Word for Compression

> The formal meaning of "the model has learned something" is: it has found a shorter description.

---

Chapter 20 ended with a question: can heuristic functions be discovered automatically from data? PAC learning guarantees that learning is possible, but it doesn't say what learning **is** — it only describes a black box and tells you what properties the black box's output has.

This chapter opens the black box.

Learning, in its most abstract sense, is inferring a generalizable law from a batch of observations — examples, data, experience. For this inference process to have a formal meaning, it must answer: compared to the observations themselves, what does the inferred law add, and what does it leave out? How can it say things about examples it has never seen?

The keyword in the answer is **compression**.

---

## 21.0 The Compression Detective Game: Recovering the Law from Samples

Learning is not memorizing data. Memorizing is too easy, and too useless. Real learning is like a detective cracking a case: there's a pile of clues on the table, and you must find an explanation shorter than the clues themselves.

When the game starts, nature hands you a string of samples:

$$S = \{(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n)\}$$

You may submit a hypothesis $h$. This hypothesis must do two things: first, it must explain the data you've seen; second, it must not be too long itself. Because the dumbest "law" always exists: memorize every sample, one by one. Such a law has zero training error, but no generalization ability at all.

The compression detective's scoring rule is:

$$L(h) + L(S \mid h)$$

The first term is how many bits it takes to describe the hypothesis itself; the second is how many more bits it takes to describe the data given the hypothesis. The best hypothesis isn't the most complex one, and isn't the one that hugs the training set — it's the one with the shortest total description length.

:::details The Formal Skeleton of This Game
- **State space**: the sample set $S$, the hypothesis class $\mathcal{H}$, the encoding scheme $L$.
- **Legal moves**: choose a hypothesis $h \in \mathcal{H}$ and use it to compress the data.
- **Transition rules**: work backwards from the observed samples to candidate laws, comparing $L(h) + L(S \mid h)$.
- **Winning condition**: find a hypothesis that compresses the training data and keeps its error low on unseen samples.
- **Failure modes**: memorizing the samples by heart; or choosing too strong an inductive bias, compressing the world into the wrong shape.
:::

This game states the secret of generalization plainly: **you can generalize because you found compressible structure**. If the data were completely random, there'd be no short law to find; if you found a short law that explains a great many samples, it most likely captured some real structure.

But there's danger here too. Compression needs an encoding scheme, and the encoding scheme is the inductive bias. You think you're "learning from the data" — but you've been walking in with a razor in your hand all along: what counts as simple, what counts as complex, your representation system decided that long before you arrived.

::: info Professor Pallas's Cat's Commentary
The cruel thing about the compression detective game: without bias, there is no learning. A completely neutral learner can only recite the samples; to generalize, you must prefer some laws over others. Machine learning classes like to call this "model selection". That's putting it way too lightly. What you're actually doing is choosing what shape you're willing to believe the world has.
:::

### 21.0.5 Playing the Game in Code

The scoring rule is small enough to compute. The program below fits four candidate laws to the ten samples by least squares, scores them by $L(h) + L(S \mid h)$, and then asks every law to predict a sample nobody has seen. The interpolating polynomial scores a perfect training fit — and then predicts nonsense.

```python
#!/usr/bin/env python3
"""
compression_detective.py -- the Compression Detective Game of Chapter 21.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 21, Section 21.0
"The Compression Detective Game: Recovering the Law from Samples."

Nature hands over ten samples; one of them is noisy. You may submit a
hypothesis h. The scoring rule is the one the book writes on the table:

    L(h) + L(S | h)

the bits for the hypothesis itself, plus the bits left over to describe
the data given the hypothesis. This program fits four candidate laws
(constant, linear, quadratic, a degree-9 polynomial that memorizes
everything, and a literal look-up table), scores them all by the rule,
and then tests generalization on a sample nobody has seen.
All fits are computed here (least squares via Gaussian elimination); all
scores and predictions are real.
Honest scope: the bit-costs are toy constants, not an optimal code. The
mechanism -- total description length picks the law that compresses --
is the point, not the coding theorem.
Run:  python3 compression_detective.py
"""
# The hidden law is y = 2x + 1, plus ONE noisy sample at x=5 (20 instead
# of 11). The detective does not know which sample is the noisy one.
XS = tuple(range(10))
YS = (1, 3, 5, 7, 9, 20, 13, 15, 17, 19)
BITS_PER_PARAMETER = 20     # toy cost of writing one parameter down
MEMORY_COST_PER_SAMPLE = 40  # toy cost of memorizing one (x, y) pair


def polyfit(xs, ys, degree):
    """Least-squares polynomial fit of the given degree.
    Normal equations solved by Gaussian elimination with partial
    pivoting. Returns the coefficient tuple (lowest power first)."""
    n = degree + 1
    a = [[0.0] * (n + 1) for _ in range(n)]
    for x, y in zip(xs, ys):
        powers = [x ** k for k in range(2 * degree + 1)]
        for i in range(n):
            for j in range(n):
                a[i][j] += powers[i + j]
            a[i][n] += y * powers[i]
    # Gaussian elimination
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    coeffs = [0.0] * n
    for r in range(n - 1, -1, -1):
        coeffs[r] = (a[r][n] - sum(
            a[r][c] * coeffs[c] for c in range(r + 1, n)
        )) / a[r][r]
    return tuple(coeffs)


def evaluate(coeffs, x):
    return sum(c * x ** k for k, c in enumerate(coeffs))


def score(coeffs):
    """L(h) + L(S|h) for a polynomial law, in toy bits."""
    lh = len(coeffs) * BITS_PER_PARAMETER
    residuals = (evaluate(coeffs, x) - y for x, y in zip(XS, YS))
    ls = sum(r * r for r in residuals)
    return lh + ls


def demo():
    print("=" * 70)
    print(" THE COMPRESSION DETECTIVE GAME -- the razor, scored for real")
    print(" The Reasoning Kingdom, Volume 2, Chapter 21 (Sec. 21.0)")
    print("=" * 70)
    print("\nThe samples on the table:")
    print("  x:", XS)
    print("  y:", YS)
    print("  (one of these is noise; the detective is not told which)")
    print("\nCandidate laws, scored by L(h) + L(S|h):")
    print(f"  {'law':<38}{'params':>6}{'L(h)':>7}{'L(S|h)':>10}{'total':>9}")
    candidates = []
    for degree in (0, 1, 2, 9):
        coeffs = polyfit(XS, YS, degree)
        name = {0: "constant", 1: "linear", 2: "quadratic", 9: "degree-9 (interpolate)"}[degree]
        total = score(coeffs)
        lh = len(coeffs) * BITS_PER_PARAMETER
        ls = total - lh
        candidates.append((name, coeffs, total))
        print(f"  {name:<38}{degree + 1:>6}{lh:>7.0f}{ls:>10.0f}{total:>9.0f}")
    # The look-up table: memorize every sample. Zero error, huge law.
    lookup = dict(zip(XS, YS))
    lh = len(lookup) * MEMORY_COST_PER_SAMPLE
    candidates.append(("memorize every sample", lookup, lh))
    print(f"  {'memorize every sample':<38}{len(lookup):>6}{lh:>7.0f}"
          f"{0:>10.0f}{lh:>9.0f}")
    winner = min(candidates, key=lambda c: c[2])
    print(f"\nWinner by total description length: {winner[0]}")
    print("  The interpolating polynomial has zero training error, and the")
    print("  look-up table is error-free by construction -- but both must")
    print("  carry their own weight in bits. The simplest law that")
    print("  explains the pile wins.")
    # Generalization: a sample nobody has seen yet, x = 10.
    print("\nThe unseen sample: x = 10. What does each law predict?")
    for name, law, _ in candidates:
        if name == "memorize every sample":
            print(f"  {name:<28} no prediction at all (x=10 was never memorized)")
        else:
            print(f"  {name:<28} y = {evaluate(law, 10):>10.1f}")
    print("  The hidden law was y = 2x + 1; its true answer is 21. The")
    print("  interpolator's wild guess is what 'zero training error' buys")
    print("  you: nothing on samples you have not seen.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
```

## 21.1 The Inverse Problem Structure of Learning

In formal logic, inference runs from axioms to theorems: given the premises, derive the conclusion. That's **forward inference**.

Learning runs in exactly the opposite direction: given a batch of theorems (the facts you've observed), infer back the most likely set of axioms (the laws). That's **inverse inference** — reasoning backwards from effects to causes.

This inverse structure has a fundamental difficulty: it is **underdetermined**. A finite batch of observations is usually compatible with infinitely many different laws. You watch the sun rise every day; this is compatible with "the sun rises every day", and also with "the sun rises every day except March 17, 2035", and with countless other laws too.

There's no purely logical solution to this difficulty. You cannot deduce the one correct law from your observations. You need some kind of **inductive bias** — a prior preference over laws, one that favors some class of them among all the laws compatible with the observations.

What the inductive bias is, determines what learning is.

### 21.1.5 The Concrete Face of Underdetermination

With the same data set, three different learners will produce three different laws. This isn't a metaphor — take out a pen and do the arithmetic.

**The data**: $(x,y) = \{(0,0), (1,1), (2,4), (3,9)\}$, four points.

**Learner A (linear regression)**: its hypothesis space is all linear functions $y = ax + b$. Looking at these four points, the best solution it finds is $y = 3x - 0.5$ (least squares). This function isn't exactly right on the training points — at $(2,4)$ it predicts $5.5$, an error of $1.5$ — but it "believes" the law is linear.

**Learner B (quadratic regression)**: its hypothesis space is all quadratic functions $y = ax^2 + bx + c$. Looking at these four points, it finds the exact solution $y = x^2$ — a perfect fit on all training points. It "believes" the law is quadratic.

**Learner C (degree-10 polynomial)**: its hypothesis space is all polynomials of degree at most 10, $y = \sum_{i=0}^{10} a_i x^i$. There are infinitely many degree-10 polynomials through these four points — that's the face of underdetermination. Left unconstrained, the learned law may be a wildly oscillating curve, thrashing madly between the training points.

Who's right? If the true law really is $y = x^2$, then learner B is best. If the true law is $y = 3x$ (with noise added), then learner A is better. Learner C is almost certainly wrong — it's too flexible, and it will treat noise as law.

**The key point**: the three learners can never settle who's right by "looking at more data". Given any finite set of points, there is always a degree-10 polynomial that fits them perfectly. You can be perfect on the training data and terrible on the test data at the same time. The data itself doesn't tell you — choosing a linear, quadratic, or degree-10 polynomial is **your choice**, not the data's.

That's the essence of inductive bias: it isn't learned from the data; it's what you carry into the problem before you ever look at the data. Like a pair of glasses, you put them on before you look. Different glasses, different laws.

::: info Professor Pallas's Cat's Commentary
Most people nod along when they hear "underdetermination", then go back to training their models with default hyperparameters. Underdetermination isn't a theoretical fact you need to "understand" — it's an operational problem that happens during every single training run. The moment you choose a model architecture, underdetermination has already been partly resolved — but you don't necessarily know which bias you chose, or whether it suits the current data. Knowing you have a bias without knowing which one is more dangerous than not knowing there's a bias at all — because you think you're in control, and you're not.
:::

::: info Professor Pallas's Cat's Commentary
Inductive bias isn't a technical option; it's a metaphysical commitment. You choose a linear model, and you're asserting "the law is linear"; you choose a deep network, and you're asserting some structural hypothesis about feature hierarchies. Most machine learning courses call this "tuning hyperparameters", and bury the metaphysical commitment inside an engineering operation. That's dodging a real question, not solving it.
:::

---

## 21.2 Formalizing Occam's Razor

The oldest, most intuitive inductive bias is Occam's razor: prefer the simpler explanation.

But what does "simple" mean?

Kolmogorov complexity gives an answer.

**Definition (Kolmogorov complexity)**: the Kolmogorov complexity $K(x)$ of a string $x$ is the length of the shortest program that generates $x$ (with reference to some fixed universal Turing machine).

$K(x)$ measures $x$'s **intrinsic complexity** — not how long $x$ is, but how hard $x$ is to describe. A string of all zeros, "000...0" (a million zeros), has a tiny $K$ — there's a very short program that generates it ("print a million zeros"). A random string has a $K$ roughly equal to its own length — there's no description shorter than "list the string itself".

:::details Kolmogorov Complexity: Measuring Complexity by the Length of a Compressing Program (prior work: Kolmogorov, 1965)
**Kolmogorov complexity $K(x)$** is the mathematical definition of "how complex is this object" — the complexity equals the length of the shortest program needed to describe it.

Intuitive examples:
- "The first million natural numbers" -> a very short program suffices (`for i in range(1000000): print(i)`), $K$ is tiny
- A randomly generated million digits -> no description shorter than "list the digits themselves", $K \approx$ the string's length

**The important uncomputability**: $K(x)$ is uncomputable — no algorithm can compute the exact value of $K(x)$ for any given $x$ (it's a consequence of the halting problem). You can only compute upper bounds on it (find a program that generates $x$; its length is an upper bound).

**Why still useful?** Even uncomputable, Kolmogorov complexity provides a **theoretical standard** — an absolute benchmark for "simplicity". In practice, MDL is used as an approximation (substituting a real compression algorithm for the shortest program).
:::

In this language, "simple" means "low Kolmogorov complexity", and "prefer the simpler explanation" means "choose the law with the smaller $K$".

### 21.2.5 Hands-On: Estimating Upper Bounds of Kolmogorov Complexity by Hand

$K(x)$ can't be computed exactly, but upper bounds can: find a program that generates $x$, and its length is an upper bound.

**Example 1**: $x =$ "01010101" ("01" repeated 4 times, 8 characters total).
- Program: `for i in range(4): print("01")`
- Program length (with some reasonable encoding): about 20-30 characters (Python syntax overhead + loop body)
- $K(x) \leq$ about 30

**Example 2**: $x =$ "01010101" repeated 1000 times (2000 characters).
- Program: `for i in range(1000): print("01")`
- Program length: about 35 characters (only the loop count changed)
- $K(x) \leq$ about 35

Notice: the data grew from 8 characters to 2000, and the program only grew from about 30 to about 35. That's the power of compression — the longer the law, the bigger the payoff.

**Example 3**: $x =$ the first 1000 digits of $\pi$.
- Program: `mp.pi.n(digits=1000)` (depends on the math library)
- Program length: about 40 characters
- $K(x) \leq$ about 40 (but this relies on the math library; the library itself would have to be counted. We can accept that — the definition of $K$ is relative to a fixed universal Turing machine, and the math library can be part of the machine)

**Example 4**: $x =$ a genuinely random string (1000 digits).
- Program: you can only `print("the first 1000 characters of the random string, verbatim")`
- Program length $\approx 1000$ + a bit of overhead
- $K(x) \approx$ the length of the string itself

These four examples reveal one key intuition: **$K(x)$ measures how compressible $x$ is**. A highly regular string has a compact description, close to $K(x) \ll |x|$; a completely random string can't be compressed, $K(x) \approx |x|$.

But there's a subtle point: the concrete value of $K(x)$ depends on the choice of reference Turing machine. Different machines may give different $K$ values for the same $x$, differing by a constant (the length of the program that simulates the other machine). That's why $K(x)$ is a complexity measure in the **asymptotic sense**, not an absolutely exact quantity. But differing by a constant is fine for most theoretical discussions — we care about the order of magnitude, not the exact value.

::: info Professor Pallas's Cat's Commentary
Estimating an upper bound on Kolmogorov complexity is a crash course in itself: you find a pattern -> you describe the pattern in the shortest program you can -> and the program's length is your boldest claim about "how complex" the string is. But you might also have missed a pattern — the program you found is only an upper bound. The true value of $K(x)$ is the distance between you and the shortest of all possible programs — and you can't be sure you found that shortest one.
:::

This is the heart of the **Minimum Description Length (MDL)** principle:

$$\text{Best hypothesis} = \arg\min_h \left[ L(h) + L(\text{data} \mid h) \right]$$

$L(h)$ is the number of bits needed to describe the hypothesis $h$ itself; $L(\text{data} \mid h)$ is the number of extra bits needed to describe the data under hypothesis $h$. The optimal hypothesis is the one that minimizes the total length of "hypothesis + the data as encoded by the hypothesis".

::: info MDL and Bayes Are the Same Thing
MDL and Bayesian inference are two languages for one idea. If the prior probability of hypothesis $h$ is $P(h) \propto 2^{-L(h)}$ (shorter hypotheses get higher prior — the standard form of Shannon coding), then the maximum a posteriori estimate (MAP) is exactly MDL:

$$\arg\max_h P(h \mid \text{data}) = \arg\max_h [P(\text{data} \mid h) \cdot P(h)] = \arg\min_h [L(\text{data} \mid h) + L(h)]$$

Occam's razor, in the Bayesian language, is the prior's preference for simplicity; in the MDL language, it's the choice of the shortest description length. They are the same mathematical structure wearing different clothes.
:::

---

## 21.3 Generalization Is Compression

The MDL principle reveals a deep equivalence: **generalization ability is equivalent to compression ability**.

Why?

A hypothesis $h$ generalizes, meaning it has captured the law in the data instead of memorizing the noise. A law is structure that can be described compactly; noise is a random component that can't be compressed. If $h$ can "explain" the data using a description shorter than the data itself, then $h$ has truly compressed the data — it found the part of the data that can be captured compactly.

Conversely, a model that perfectly memorizes its training data has done no compression — it just stored the data verbatim. Such a model does perfectly on the training data but has no predictive power over new data, because it found no law; it merely archived examples.

This is precisely the formalization of "overfitting" from Chapter 5 (Volume 1): **overfitting is memory without compression; generalization is effective compression**.

In the language of Kolmogorov complexity, the amount of generalization can be measured: suppose the training data is compressed from $n$ bits down to $k$ bits ($k < n$). Then the compression ratio $n/k$, in a sense, measures the hypothesis's "generalization potential" — how much generalizable law was found, not how many specific cases were memorized.

### 21.3.5 Double Descent: When "More Parameters = Better Generalization" Shook Classic Theory

Classical learning theory (including PAC and VC dimension) predicted a U-shaped curve: model too simple -> underfitting; model just right -> generalizes well; model too complex -> overfitting. But after 2018, researchers observed a counterintuitive phenomenon in large-scale deep learning: **double descent**.

After the parameter count exceeds the amount of training data — entering the "overparameterized" region — the test error doesn't keep rising. It falls again. A model big enough to perfectly memorize the training data instead starts generalizing better.

What does this mean? VC dimension and PAC's worst-case guarantees are still correct — they aren't "wrong". But they describe the worst of all possible data distributions. Real deep learning, on natural data, seems to have some kind of **implicit simplicity preference**: even when the model has enough freedom to memorize every training sample, the interplay of the optimization algorithm (SGD) and the architecture choices (convolutions, attention, layer normalization) still leans toward simpler solutions.

**The overtraining perspective**: after 2023, the practitioner community gradually accepted another classic-intuition-defying fact: for a model with a fixed parameter count, pouring in hundreds or even thousands of times more data — so-called "overtraining" — typically keeps improving generalization, far beyond what classic theory predicts. Models are still learning generalizable laws when the data volume reaches hundreds of times the parameter count.

These two phenomena — double descent and overtraining — point in the same direction: **deep learning's generalization mechanism cannot be fully explained by pure statistical learning theory**. The MDL viewpoint (generalization = compression) offers a more flexible language: the key isn't the parameter count, it's the "effective description length" the model actually uses inside. A 100-billion-parameter model, if its effective degrees of freedom are implicitly compressed by SGD down to the millions, is "simple" in the MDL framework — despite the enormous parameter count.

This viewpoint is still developing; nothing is settled. But it makes one important fact clear: **theoretical tools need to be revisited as practice develops**. PAC and VC dimension won't become "outdated" — they give exact guarantees in the worst case. But the distribution of natural data isn't the worst case, so actual behavior can be far better than the worst-case bound. Keeping straight "under what assumptions does the theory hold" and "under what conditions does practice break those assumptions" is the key to understanding generalization.

::: info Professor Pallas's Cat's Commentary
Double descent slapped classic theory in the face, but classic theory isn't dead. Classic theory says: "there exists some horrible data distribution that sinks your model". Double descent says: "fortunately, natural data isn't that distribution". Both statements are true at once; they don't contradict each other. A good theorist isn't someone who explains everything with theory, but someone who knows where the theory stops being enough — and starts looking for a new one.
:::

::: info Why Deep Neural Networks Can Generalize
This was a mystery in the theory community for a while: deep neural networks often have more parameters than training data, and by traditional learning theory such models should badly overfit. In practice, they usually generalize well. The MDL viewpoint offers a direction: what measures generalization isn't the parameter count but the "effective description length" the model actually uses — a model with ten billion parameters that are highly structured among themselves may have an effective description length far smaller than its parameter count. Compression is what matters, not scale. But as the deep learning research community has found — and it's worth mentioning that the description above is mostly the classic deep representation learning picture, which is not necessarily the gold standard in LLM and modern neural network training — after 2023, the data-to-parameter ratio in deep learning has emphasized overtraining: with a fixed parameter count, we pour hundreds or even thousands of times more data into training a network of the same scale. Recent research suggests that overtraining is an effective form of generalization for deep learning and large language models.
:::

---

## 21.4 Choosing an Inductive Bias

MDL and Kolmogorov complexity give an elegant framework, but there's a practical problem: Kolmogorov complexity is **uncomputable**.

::: info Professor Pallas's Cat's Commentary
So Occam's razor, in its strictest form, is itself undecidable. You cannot write an algorithm that, for any two hypotheses, always finds which is simpler in finite time. Behind the word "simple" hides a halting problem. This doesn't stop you from using approximations of MDL — but you have to know it's an approximation, not the real razor.
:::

Given a string $x$, computing $K(x)$ requires finding the shortest program that generates $x$ — which is equivalent to solving the halting problem, which Chapter 19 already proved undecidable. You can't write an algorithm that computes the Kolmogorov complexity of any input in finite time.

This means MDL, in its purest form, is itself uncomputable. The MDL used in practice is an approximation: substitute some fixed encoding scheme for the universal Turing machine, and substitute a computable description length for the uncomputable Kolmogorov complexity.

Different approximation schemes correspond to different inductive biases:

- **Decision tree learning**: the hypothesis space is decision trees; the description length is the tree's depth or node count. It favors small trees, i.e. simple decision rules.
- **Linear models**: the hypothesis space is linear functions; the description length is the norm of the coefficients. L1 regularization favors sparse coefficients (most are zero); L2 regularization favors small coefficients.
- **Neural networks**: the hypothesis space is networks of a specific architecture; the "description length" is jointly determined by the parameter values and the network structure. This description length is very hard to pin down — a huge blank zone between theory and practice.

Every learning algorithm implicitly chooses an inductive bias — a particular understanding of "what simple means". In most machine learning courses, this choice is handled under the banner of "hyperparameter tuning" — but it's actually a metaphysical commitment about which kind of law should be preferred.

### 21.4.5 Rademacher Complexity: Another Ruler

VC dimension is a combinatorial quantity, and for many modern hypothesis spaces (like deep neural networks) computing the exact VC dimension is very hard. Rademacher complexity offers a different viewpoint — it directly measures a hypothesis class's ability to fit random noise.

Consider a sample set $S = \{z_1, \ldots, z_m\}$, and assign each sample an independent Rademacher variable $\sigma_i \in \{-1, +1\}$ (chosen uniformly at random). The empirical Rademacher complexity of a hypothesis class $\mathcal{H}$ is:

$$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^m \sigma_i h(z_i)\right]$$

Intuition: if $\mathcal{H}$ fits pure random noise well (the $\sigma_i$'s are random $\pm 1$), then $\mathcal{H}$ is complex. Because fitting noise doesn't require "learning a law", just enough degrees of freedom. The higher the Rademacher complexity, the more easily $\mathcal{H}$ overfits.

Relation to VC dimension: VC dimension gives a worst-case bound; Rademacher complexity gives a **data-dependent** bound — it measures complexity on the specific sample set $S$ at hand. In some cases this makes it tighter than VC dimension (the bound is closer to actual performance).

Put VC dimension and Rademacher complexity side by side: VC dimension asks "how many points can you shatter?" (purely combinatorial); Rademacher complexity asks "on this particular data set, how well can you fit random noise?" (data-dependent). They quantify the same thing from different angles: **the hypothesis space's capacity to host random patterns**. The more capacity, the more samples you need; when samples run short, the model may learn noise instead of signal.

---

## 21.5 The Two Difficulties of Inverse Inference

Learning as inverse inference faces two fundamental difficulties, neither of which the data itself can solve.

**The first difficulty: underdetermination**. As we said, finite data is compatible with infinitely many hypotheses, so an inductive bias is required to choose among them. The inductive bias is imported; it is not inferred from the data. The solution to a learning problem depends on the choice of hypothesis space — and that choice lies outside the learning algorithm's jurisdiction.

**The second difficulty: distribution shift**. PAC learning assumes the training data and test data come from the same distribution $\mathcal{D}$. In practice, this assumption is frequently violated — the distribution when training and the distribution at deployment may differ. When distribution shift happens, any generalization guarantee built on the training data may fail.

These two difficulties are fundamental limits of learning theory, not algorithm bugs. They mean: **every learning system works under certain assumptions; outside those assumptions, the guarantees vanish**.

This structure is very similar to the formal systems of Chapter 14: a formal system is sound inside its axioms, with no guarantees beyond them; a learning system is sound inside its assumptions, with no guarantees beyond them. Both systems' reliability depends on an external commitment that cannot be verified from inside.

### 21.5.5 Why This Difficulty Cannot Be Solved by Data

Let's nail this problem down with an extreme example.

Suppose you have a learner whose task is to predict the next element of a sequence. The training data is:

$$2, 4, 6, 8, 10, 12, \ldots$$

Three learners each learn a different law:
- **Learner A**: outputs $2n$ (the evens)
- **Learner B**: outputs $n^2 - n + 2$ (this formula also gives 2, 4, 6, 8... for the first few terms — no: $1^2 - 1 + 2 = 2$, $2^2 - 2 + 2 = 4$, $3^2 - 3 + 2 = 8$... wait, that's wrong: $3^2 - 3 + 2 = 8$, but the sequence's 3rd term is 6. So this one doesn't work)

Let me rewrite it — a correct example matters more:

Sequence: $0, 1, 2, 3, 4, 5, \ldots$
- **Learner A**: $f(n) = n$ (the natural numbers)
- **Learner B**: $f(n) =$ "the $n$-th non-negative integer" (the same as A)
- **Learner C**: $f(n) = n$ for $n \leq 1000$, and $f(n) = n + 1000$ after that

For the first 5 elements (0,1,2,3,4), the three learners agree exactly. You cannot tell them apart from the data. But the prediction for element 1001: A and B predict 1000, C predicts 2000.

What does this example expose? Finite data can never uniquely pin down the law — underdetermination isn't "not enough data yet"; it's "underdetermination so strong that it survives even with limit data". The only way out is to bring in an inductive bias — to assert that C is "more complex" and therefore worse than A. But that assertion didn't come from the data; you brought it in.

And note: if the true law really is C (say the sequence jumps abruptly after element 1000), then your inductive bias — the preference for A — makes you systematically wrong. Inductive bias is not a free lunch: it helps you in some worlds and hurts you in others. And you can't tell from the training data which world you live in.

Back to the compression detective game: the detective never walks onto the scene empty-handed. They carry an encoding scheme with them, a set of presuppositions about "what a short description means". So the same pile of clues leads different detectives to different laws: the linear model sees a line, the decision tree sees branches, the neural network sees hierarchical representations. So-called learning is not the data speaking up on its own; it's the data being forced to answer inside a particular compression language.

This sentence is unpleasant, but it must stay: a learning system's failure is often not that it failed to find a law — it's that it found a law too beautiful, under the wrong rules of the game.

---

## 21.6 The Unified Picture of Learning and Inference

Put this chapter's content into the framework of Volume 2.

Chapter 14 built the syntactic machine of inference. Chapter 15 discovered this machine has inherent limits — some true propositions can't be proven from inside. Chapter 16 modified one structural rule and got a new kind of inference. Chapter 17 extended truth values to $[0,1]$ and got probabilistic inference. Chapter 18 introduced the do operator and got causal inference. Chapter 19 discovered that these inferences have fundamental computational lower bounds. Chapter 20, under cost constraints, defined "good enough" precisely.

What Chapter 21 does is embed "learning" into this framework. Learning is **inverse inference** — starting from observations, infer back to the laws. The MDL principle gives the basic rule of inverse inference: the shortest explanation. The PAC framework gives the quality guarantee of inverse inference: approximately correct with high probability.

But this unified picture is missing one final piece: once the inference system is large and complex enough, its "inference" begins to include propositions about itself — about what it can learn, what it can't, and what its own inductive bias is. At that point, the Gödel structure of Chapter 15 reappears: **a sufficiently strong inference system cannot fully infer itself**.

### 21.6.5 The Parallel Structure: The Incompleteness of Learning

Put the logical incompleteness of Chapters 14–15 and the learning incompleteness of Chapters 20–21 side by side. The two arguments share a deep structure:

| | Formal systems (Chapters 14–15) | Learning systems (Chapters 20–21) |
|---|---|---|
| **Starting point** | Axioms + inference rules | Training data + inductive bias |
| **Doable from inside** | Generate all theorems (syntactic) | Search the hypothesis space (learning) |
| **Needed from outside** | Semantic interpretation (model theory) | Generalization guarantees (PAC/MDL) |
| **Cannot be internalized** | A proof of the system's consistency | Justification of the inductive bias |
| **Boundary behavior** | There exist true but unprovable propositions | There exist concepts that are learnable but need an external bias |

This isn't "kind of similar". This is **the same mathematical structure instantiated in two domains**. In both cases:
1. The legitimate operations inside the system (proving / learning) can continue indefinitely
2. But the system cannot verify, from inside, the **correctness conditions** of those operations (consistency / generalization)
3. The correctness conditions need an **external viewpoint** — formal systems need models, learning systems need distribution assumptions

**The deepest question**: if the inductive bias itself becomes a learning object — meta-learning — can this limit be broken?

Meta-learning indeed learns a bias over task distributions, making adaptation to new tasks faster. But meta-learning has its own inductive bias: it carries assumptions about the task distribution (like "tasks share structure") when evaluating "which bias performs well on the task distribution". If you make meta-learning itself a learning object (meta-meta-learning), this chain has no end — each level needs its own inductive bias, and each level's bias must be imported from outside.

This is fully parallel to Gödel's structure: add $G$ to the axioms -> the new system has its own $G'$ -> it never wins. Make the inductive bias a learning object -> meta-learning has its own meta-bias -> it can never bootstrap itself.

This is the entrance to Chapter 22. Chapter 23 goes on to turn the convergence rate into the log curves of emergence.

---

## Open Questions

**Kolmogorov complexity is uncomputable, so does MDL even mean anything?** The uncomputability of Kolmogorov complexity makes it an ideal, not an algorithm. But mathematics is full of such ideals: optimal decisions, perfect information, infinite precision. An ideal doesn't need to be computable; it needs to serve as a benchmark — telling us how far the actual algorithms sit from the optimum. MDL's computable approximation is a finite approximation of this ideal; the ideal's very existence is what gives "approximation" its meaning.

**Is generalization really equivalent to compression?** The MDL framework suggests: a model that can't compress its training data won't generalize; a model that can, will. But this equivalence doesn't always hold in deep learning practice — some large-scale models generalize fine while also "memorizing" the data. This tension points to an unsolved theoretical question: what exactly is deep learning's generalization mechanism?

**Can inductive bias be learned?** If learning is inverse inference, then can "what makes a good hypothesis space" itself be learned? Meta-learning tries to learn a good inductive bias over task distributions, so models adapt faster to new tasks. But meta-learning has its own inductive bias itself — is this an infinite regress, or does it stop at some level?

---

## Exercises

**★ Warm-up**

Of the two strings below, which has the lower Kolmogorov complexity? Give your intuition and a one-sentence reason — no need to compute concrete values.

1. `0101010101010101010101010101010101010101` (40 characters, alternating 0s and 1s)
2. `1101001000011010110101010001010001000010` (40 random characters)

One more question: a "completely random" string — what is its Kolmogorov complexity roughly? Compared to the length of the string itself?

---

**★★ Derivation**

1. **Compression and generalization**: one model has zero error on the training set, but large error on the test set (overfitting); another has 5% error on training and 5% on testing (good generalization). Describe the difference in MDL terms: which one compressed the data? Which one memorized it? What does each model's "description length" consist of?

2. **Making inductive biases explicit**: linear regression, decision trees, and neural networks (with L2 regularization) — what inductive bias does each learning algorithm imply? Describe each with "what kind of law is preferred". On what type of data does each bias systematically fail?

---

**★★★ Challenge**

Kolmogorov complexity is uncomputable (because it's equivalent to the halting problem). This means: no algorithm can decide, in finite time, which of any two hypotheses is simpler.

But MDL is used in practice, with computable approximations. Try to list at least two computable "description length" approximation schemes, and analyze: what inductive bias does each approximation implicitly assume? When would different approximations give different "optimal hypotheses"?

The point of this problem: translate "hyperparameter choice" into "inductive bias choice", then into "an implicit commitment about what simple means" — after all three translations, does your view of machine learning's "objectivity" change?

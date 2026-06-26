# Chapter 21: Learning as Inverse Inference — Generalization Is Compression by Another Name

> The formal meaning of "the model has learned something" is: it has found a shorter description.

---

Chapter 20 left us with a question: can heuristic functions be automatically discovered from data? PAC learning guarantees the possibility of learning, but does not say what learning **is** — it merely describes a black box, telling you what properties the black box's output has.

This chapter sets out to open that black box.

Learning, in the most abstract sense, is inferring a generalizable regularity from a batch of observations — examples, data, experiences. For this inference process to have formal meaning, it must answer: compared to the observations themselves, what does the inferred regularity add? What does it omit? Why can it say things about unseen examples from examples it has seen?

The keyword of the answer is **compression**.

---

## 21.0 The Compression Detective Game: Reverse-Engineering Regularities from Samples

Learning is not memorizing the data. Memorization is too easy, and too useless. Genuine learning is like a detective solving a case: there is a pile of clues on the table, and you must find an explanation that is shorter than the clues themselves.

At the start of the game, Nature gives you a string of samples:

$$S = \{(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n)\}$$

You may submit a hypothesis $h$. This hypothesis must do two things: first, it must explain the data already seen; second, it must not itself be too long. Because the most foolish "regularity" always exists: memorize all the samples one by one. Such a regularity has zero training error, but no generalization ability whatsoever.

The scoring rule of the compression detective is:

$$L(h) + L(S \mid h)$$

The first term is how many bits are needed to describe the hypothesis itself; the second term is how many additional bits are needed to describe the data after the hypothesis is given. The best hypothesis is not the most complex, nor the one most closely fitting the training set, but the one that makes the total description length shortest.

:::details The Formal Skeleton of This Game
- **State space**: sample set $S$, hypothesis class $\mathcal{H}$, encoding scheme $L$.
- **Legal moves**: choose a hypothesis $h \in \mathcal{H}$, and use it to compress the data.
- **Transition rules**: reverse-engineer candidate regularities from observed samples, compare $L(h)+L(S\mid h)$.
- **Victory condition**: find a hypothesis that can compress the training data and maintain low error on unseen samples.
- **Failure mode**: rote memorization of samples; or choosing an overly strong inductive bias, compressing the world into the wrong shape.
:::

This game states the secret of generalization very plainly: **one can generalize because one has discovered compressible structure**. If the data are completely random, there is no short regularity to be found; if you have found a short regularity that explains a large number of samples, then it very likely captures some genuine structure.

But there is also danger here. Compression requires an encoding scheme, and the encoding scheme is the inductive bias. You think you are "learning from data," but in fact you have always entered the field carrying a razor: what counts as simple, what counts as complex, has long been determined by your representation system.

::: info Professor Pallas's Cat's Commentary
The coldness of the Compression Detective Game lies in this: without bias, there is no learning. A completely neutral learner can only memorize samples; to generalize, one must favor certain regularities. Machine learning classes like to call this "model selection." That is far too light a phrasing. In truth, it is choosing what you are willing to believe the world looks like.
:::

## 21.1 The Inverse Problem Structure of Learning

In formal logic, the direction of inference is from axioms to theorems: given premises, derive conclusions. This is **forward inference**.

The direction of learning is exactly the opposite: given a batch of theorems (observed facts), reverse-engineer the most probable set of axioms (regularities). This is **inverse inference** — pushing back from effects to causes.

This inverse structure has a fundamental difficulty: it is **underdetermined**. A finite batch of observations is typically compatible with infinitely many different regularities. You see the sun rise every day; this is compatible with the regularity "the sun rises every day," also compatible with "the sun rises every day except March 17, 2035," and compatible with infinitely many other regularities as well.

This difficulty has no purely logical solution. You cannot deductively derive the uniquely correct regularity from observations. You need some kind of **inductive bias** — an a priori preference for regularities, favoring a certain class among all regularities compatible with the observations.

What inductive bias is determines what learning is.

### 21.1.5 Concrete Faces of Underdetermination

Using the same dataset, three different learners will give three different regularities. This is not a metaphor; it becomes clear with a simple calculation.

**Data**: $(x,y) = \{(0,0), (1,1), (2,4), (3,9)\}$, four points.

**Learner A (Linear Regression)**: The hypothesis space is all linear functions $y = ax + b$. Seeing these four points, the optimal solution it finds is $y = 3x - 0.5$ (least squares). This function is not exactly accurate on the training points — the predicted value at (2,4) is 5.5, an error of 1.5 — but it "believes" the regularity is linear.

**Learner B (Quadratic Regression)**: The hypothesis space is all quadratic functions $y = ax^2 + bx + c$. Seeing these four points, the exact solution it finds is $y = x^2$ — a perfect fit for all training points. It "believes" the regularity is quadratic.

**Learner C (Degree-10 Polynomial)**: The hypothesis space is all polynomials of degree at most 10: $y = \sum_{i=0}^{10} a_i x^i$. There are infinitely many degree-10 polynomials passing through these four points — this is the face of underdetermination. Without constraints, the learned regularity could be an extremely oscillatory curve, wildly wiggling between the training points.

Who is right? If the true regularity is indeed $y = x^2$, Learner B is optimal. If the true regularity is $y = 3x$ (with added noise), Learner A is better. Learner C is almost certainly wrong — it is too flexible and would treat noise as regularity.

**Key point**: Four people cannot determine who is right by "looking at more data." For any given finite set of points, there always exists a degree-10 polynomial that perfectly fits them. You can achieve perfect performance on the training data while performing terribly on test data. The data itself will not tell you — choosing degree 1, degree 2, or degree 10 is **your choice**, not the data's choice.

This is the essence of inductive bias: it is not learned from data; it is something you bring into the arena before you even look at the data. Like a pair of glasses you put on before you see the data. Different glasses see different regularities.

::: info Professor Manul's Commentary
Many people nod inwardly when they hear "underdetermination," and then continue training their models with default hyperparameters. Underdetermination is not a theoretical fact that needs to be "understood"; it is an operational problem that occurs in every training session. The moment you choose a model architecture, underdetermination has already been partially resolved — but you may not know exactly what bias you chose, nor whether it is appropriate for the current data. Knowing that there is a bias but not knowing which bias you chose is more dangerous than not knowing there is a bias — because you think you are in control, when you are not.
:::

::: info Professor Pallas's Cat's Commentary
Inductive bias is not a technical option; it is a metaphysical commitment. If you choose a linear model, you are conceding that "regularities are linear"; if you choose a deep network, you are conceding some structural assumption about feature hierarchies. Most machine learning classes call this "tuning hyperparameters," burying the metaphysical commitment inside engineering operations. This is evading a genuine problem, not solving it.
:::

---

## 21.2 Formalizing Occam's Razor

The oldest and most intuitive inductive bias is Occam's razor: **simpler explanations are preferred**.

But what does "simple" mean?

Kolmogorov complexity offers an answer.

**Definition (Kolmogorov complexity)**: The Kolmogorov complexity $K(x)$ of a string $x$ is the length of the shortest program that can generate $x$ (with some fixed universal Turing machine as reference).

$K(x)$ measures the **intrinsic complexity** of $x$ — not how long $x$ is, but how hard $x$ is to describe. A string of all zeros "000...0" (a million zeros) has a very small $K$ value — there is a very short program that can generate it ("output a million zeros"). A random string has a $K$ value roughly equal to its length — there is no shorter description than "list the string itself."

:::details Kolmogorov Complexity: Measuring Complexity by "Compressed Program Length" (prior work: Kolmogorov, 1965)
**Kolmogorov complexity $K(x)$** is the mathematical definition of "how complex is this object" — complexity equals the length of the shortest program required to describe it.

Intuitive examples:
- "the first million natural numbers" -> a very short program suffices (`for i in range(1000000): print(i)`), $K$ is very small
- A randomly generated million-digit number -> no shorter description than "list the digits themselves," $K \approx$ string length

**The important uncomputability**: $K(x)$ is uncomputable — there exists no algorithm that can compute the exact value of $K(x)$ for arbitrary $x$ (this is a corollary of the halting problem). You can only compute an **upper bound** (find a program that can generate $x$; its length is an upper bound).

**Why is it still useful?** Even though uncomputable, Kolmogorov complexity provides a **theoretical standard** — an absolute benchmark for measuring "simplicity." In practice, MDL approximations are used (substituting actual compression algorithms for the shortest program).
:::


In this language, "simple" means "small Kolmogorov complexity," and "simpler explanations are preferred" means "choose the regularity with smaller $K$ value."

### 21.2.5 Hands-On: Computing Upper Bounds of Kolmogorov Complexity

$K(x)$ cannot be computed exactly, but upper bounds can be computed — find a program that can generate $x$, and its length is an upper bound.

**Example 1**: $x =$ "01010101" (01 repeated 4 times, 8 characters total).
- Program: `for i in range(4): print("01")`
- Program length (using some reasonable encoding): about 20-30 characters (Python syntax overhead + loop body)
- $K(x) \leq$ about 30

**Example 2**: $x =$ "01010101" repeated 1000 times (2000 characters).
- Program: `for i in range(1000): print("01")`
- Program length: about 35 characters (only changed loop count)
- $K(x) \leq$ about 35

Note: the data grew from 8 characters to 2000 characters, but the program only grew from about 30 to about 35. This is the power of compression — the longer the regularity, the greater the benefit of compression.

**Example 3**: $x =$ the first 1000 digits of $\pi$.
- Program: `mp.pi.n(digits=1000)` (depends on the math library)
- Program length: about 40 characters
- $K(x) \leq$ about 40 (but this depends on the math library, which itself needs to be accounted for. However, we can accept this — the definition of $K$ is relative to a fixed universal Turing machine, and the math library can be part of that Turing machine)

**Example 4**: $x =$ a truly random string (1000 bits).
- Program: can only `print("1000 bits of random string as-is")`
- Program length ≈ 1000 + a little overhead
- $K(x) \approx$ the length of the string itself

These four examples reveal a crucial intuition: **$K(x)$ measures the extent to which $x$ can be compressed**. Highly regular strings have succinct descriptions, approaching $K(x) \ll |x|$; completely random strings cannot be compressed, $K(x) \approx |x|$.

But there is a subtle point: the specific value of $K(x)$ depends on the choice of reference Turing machine. Different Turing machines may give different $K$ values for the same $x$, differing by a constant (the length of the program that simulates the other Turing machine). This is why $K(x)$ is a measure of complexity in an **asymptotic sense**, not an absolutely precise quantity. But differing by a constant is sufficient for most theoretical discussions — we care about orders of magnitude, not exact values.

::: info Professor Manul's Commentary
Computing an upper bound on Kolmogorov complexity is itself a crash course: you find a regularity -> you describe the regularity with the shortest program -> the length of the program is your strongest claim about how "complex" this string is. But you may also have missed a regularity — the program you found is only an upper bound. The true value of K(x) is the gap between you and the shortest of all possible programs — and you can never be sure you have found the shortest one.
:::

This is the core of the **Minimum Description Length (MDL)** principle:

$$\text{Optimal hypothesis} = \arg\min_h \left[ L(h) + L(\text{data} \mid h) \right]$$

$L(h)$ is the number of bits required to describe hypothesis $h$ itself, $L(\text{data} \mid h)$ is the additional number of bits required to describe the data given hypothesis $h$. The optimal hypothesis is the one that makes the total length of "hypothesis + data encoded under the hypothesis" shortest.

::: info The Equivalence of MDL and Bayesian Approaches
The MDL principle and Bayesian inference are the same thing in two languages. If the prior probability of hypothesis $h$ is $P(h) \propto 2^{-L(h)}$ (i.e., shorter hypotheses have higher prior, the standard form of Shannon coding), then the maximum a posteriori estimate (MAP) is exactly equivalent to MDL:

$$\arg\max_h P(h \mid \text{data}) = \arg\max_h [P(\text{data} \mid h) \cdot P(h)] = \arg\min_h [L(\text{data} \mid h) + L(h)]$$

Occam's razor, in Bayesian language, is the prior's preference for simplicity; in MDL language, it is the choice of shortest description length. The two are equivalent mathematical structures, dressed in different clothes.
:::

---

## 21.3 Generalization Is Compression

The MDL principle reveals a profound equivalence: **generalization ability is equivalent to compression ability**.

Why?

A hypothesis $h$ can generalize, meaning it captures regularities in the data, rather than memorizing noise. Regularities are structures that can be succinctly described; noise is a random component that cannot be compressed. If $h$ can "explain" the data using a description shorter than the length of the data itself, this shows that $h$ has truly compressed the data — discovered the part of the data that can be captured succinctly.

Conversely, a model that completely memorizes the training data does not compress — it merely stores the data unchanged. Such a model performs perfectly on training data, but has no predictive ability for new data, because it has discovered no regularities, only stored examples.

This is precisely the formalization of "overfitting" from Chapter 5 (Volume 1): **overfitting is memorization without compression; generalization is effective compression**.

In the language of Kolmogorov complexity, the quantity of generalization can be measured: suppose the hypothesis compresses the training data from $n$ bits to $k$ bits ($k < n$); then the compression ratio $n/k$ in some sense measures the hypothesis's "generalization potential" — how many generalizable regularities have been discovered, rather than how many particular instances have been memorized.

### 21.3.5 Double Descent: When "More Parameters = Better Generalization" Shook Classic Theory

Classic learning theory (including PAC and VC dimension) predicts a U-shaped curve: model too simple -> underfitting; model just right -> good generalization; model too complex -> overfitting. However, after 2018, researchers observed a counterintuitive phenomenon in large-scale deep learning: **Double Descent**.

After the number of parameters exceeds the amount of training data — that is, entering the "overparameterized" regime — test error does not continue to rise, but instead **decreases again**. After the model becomes large enough to completely memorize the training data, it paradoxically begins to generalize better.

What does this mean? VC dimension and PAC guarantees in the worst case are still correct — they are not "wrong." But they describe **the worst of all possible data distributions**. Actual deep learning, on natural data, seems to have some kind of **implicit simplicity preference** — even when the model has enough degrees of freedom to memorize every training sample, the interaction of the optimization algorithm (SGD) and architecture choices (convolutions, attention, layer normalization) still biases toward simpler solutions.

**The perspective of overtraining**: After 2023, the practitioner community gradually accepted another counterintuitive fact: for a model with a fixed parameter count, pouring in hundreds or even thousands of times more data — so-called "overtraining" — can often continuously improve generalization performance, far exceeding the predictions of classic theory. Models continue to learn generalizable regularities even when the data volume reaches hundreds of times the parameter count.

These two phenomena — double descent and overtraining — together point in one direction: **the generalization mechanism of deep learning cannot be fully explained by pure statistical learning theory**. The MDL perspective (generalization = compression) offers a more flexible language: the key is not the number of parameters, but the "effective description length" actually used internally by the model. A hundred-billion-parameter model, if its effective degrees of freedom are implicitly compressed by SGD to the order of millions, is "simple" in the MDL framework — despite the large parameter count.

This perspective is still developing, without a settled conclusion. But it illustrates an important fact: **theoretical tools need to be re-examined as practice evolves**. PAC and VC dimension will not "become obsolete" — they provide precise guarantees in the worst case. But the distribution of natural data is not the worst case, so actual behavior can be far better than worst-case bounds. Distinguishing between "what assumptions a theory holds under" and "under what conditions practice breaks through those assumptions" is the key to understanding generalization.

::: info Professor Manul's Commentary
Double Descent delivered a slap to classic theory, but classic theory is not dead. Classic theory says "there exists some terrible data distribution that will capsize your model"; Double Descent says "fortunately, natural data is not that distribution." Both statements are simultaneously true and not contradictory. A good theorist is not someone who explains everything with a single theory, but someone who knows where the theory starts to fall short and begins looking for a new theory.
:::

::: info Why Deep Neural Networks Can Generalize
This question was for a time a puzzle in theory: the number of parameters in deep neural networks often exceeds the amount of training data; according to traditional learning theory, such models should severely overfit. Yet in practice they often generalize well. The MDL perspective offers a direction: what measures generalization ability is not the number of parameters, but the "effective description length" actually used by the model — a model with a hundred billion parameters but with parameters highly structured among themselves may have an effective description length far smaller than the parameter count. Compression is the key, not scale. However, as the deep learning academic community has discovered, it is worth noting that the preceding description is more characteristic of classical deep representation learning methods; in the training of large language models and modern neural networks, this is not necessarily the gold standard. After 2023, our deep learning data-to-parameter ratios increasingly emphasize over-training — that is, for a given parameter count, we would pour in hundreds or even thousands of times more data to train a neural network of the same scale. The latest research has found that over-training is an effective form of generalization for deep learning and large language models.
:::

---

## 21.4 The Choice of Inductive Bias

MDL and Kolmogorov complexity provide an elegant framework, but there is a practical problem: Kolmogorov complexity is **uncomputable**.

::: info Professor Pallas's Cat's Commentary
So Occam's razor, in its strictest formal version, is itself undecidable. You cannot write an algorithm that, for any two hypotheses, always determines in finitely many steps which is simpler. Hidden behind the word "simple" is a halting problem. This does not prevent you from using MDL approximations — but you must know that what you are using is an approximation, not the true razor.
:::

Given a string $x$, computing $K(x)$ requires finding the shortest program that can generate $x$ — this is equivalent to solving the halting problem, which Chapter 19 has already proved is undecidable. You cannot write an algorithm that, for any input, computes its Kolmogorov complexity in finitely many steps.

This means that MDL, in its purest form, is itself uncomputable. MDL in practice is its approximate version: using some fixed encoding scheme to replace the universal Turing machine, using computable description length to replace uncomputable Kolmogorov complexity.

Different approximation schemes correspond to different inductive biases:

- **Decision tree learning**: the hypothesis space is decision trees; the description length is the depth or node count of the tree. Favors small trees, i.e., simple decision rules.
- **Linear models**: the hypothesis space is linear functions; the description length is the norm of the coefficients. L1 regularization favors sparse coefficients (most coefficients zero), L2 regularization favors small coefficients.
- **Neural networks**: the hypothesis space is a specific-architecture network; the "description length" is jointly determined by parameter values and network structure. This description length is very hard to define precisely; there is a vast gap between theory and practice.

Every learning algorithm implicitly chooses an inductive bias, an understanding of "what is simple." This choice, in the vast majority of machine learning courses, is handled under the name of "hyperparameter tuning" — but it is in fact a metaphysical commitment about "what kind of regularities are preferred."

### 21.4.5 Rademacher Complexity: Another Yardstick

VC dimension is a combinatorial quantity; for many modern hypothesis spaces (such as deep neural networks), computing the VC dimension exactly is very difficult. **Rademacher complexity** provides a different perspective — it directly measures a hypothesis class's ability to fit random noise.

Consider a sample set $S = \{z_1, \ldots, z_m\}$, randomly assign each sample a Rademacher variable $\sigma_i \in \{-1, +1\}$ (chosen independently with equal probability). The empirical Rademacher complexity of a hypothesis class $\mathcal{H}$ is:

$$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^m \sigma_i h(z_i)\right]$$

Intuition: if $\mathcal{H}$ can fit pure random noise well ($\sigma_i$ are random $\pm 1$), then $\mathcal{H}$ is complex. Because fitting noise does not require "learning regularities," only enough degrees of freedom. The higher the Rademacher complexity, the more prone $\mathcal{H}$ is to overfitting.

Relationship with VC dimension: VC dimension gives worst-case bounds; Rademacher complexity gives **data-dependent** bounds — it measures complexity on a given specific sample set $S$. This makes it in some cases tighter than VC dimension (the bounds are closer to actual performance).

Looking at VC dimension and Rademacher complexity side by side: VC dimension asks "how many points can you shatter" (purely combinatorial); Rademacher complexity asks "on this specific dataset, how well can you fit random noise" (data-dependent). Both quantify the same thing from different angles: **the hypothesis space's capacity to accommodate random patterns**. The greater the capacity, the more samples are needed; when samples are insufficient, the model may learn noise rather than signal.

---

## 21.5 Two Difficulties of Inverse Inference

Learning as inverse inference faces two fundamental difficulties, neither of which can be resolved by the data itself.

**The first difficulty: underdetermination**. As described above, finite data are compatible with infinitely many hypotheses; there must be an inductive bias to choose among them. The inductive bias is input, not inferred from the data. The solution to the learning problem depends on the choice of hypothesis space — and the choice of hypothesis space falls outside the jurisdiction of the learning algorithm.

**The second difficulty: distribution shift**. PAC learning assumes that training data and test data come from the same distribution $\mathcal{D}$. But in practical applications, this assumption is often violated — the data distribution at training time and the data distribution at deployment time may differ. When distribution shift occurs, any generalization guarantee based on training data may become void.

These two difficulties are fundamental limitations of learning theory, not defects of algorithms. They indicate: **any learning system operates under certain assumptions; beyond those assumptions, guarantees vanish**.

This structure is highly similar to the formal systems of Chapter 14: a formal system is reliable within its axioms; beyond the axioms, there are no guarantees. A learning system is reliable within its assumptions; beyond the assumptions, there are no guarantees. The reliability of both depends on an external commitment that cannot be verified from within.

### 21.5.5 Why This Difficulty Cannot Be Resolved by Data

Let us nail this problem down with an extreme example.

Suppose you have a learner whose task is to predict the next element of a sequence. The training data is:

$$2, 4, 6, 8, 10, 12, \ldots$$

Three learners each learned different regularities:
- **Learner A**: outputs $2n$ (even numbers)
- **Learner B**: outputs $n^2 - n + 2$ (this formula also gives 2, 4, 6, 8 for the first few terms... no, $1^2-1+2=2$, $2^2-2+2=4$, $3^2-3+2=8$... wait, that is wrong, $3^2-3+2=8$, but the 3rd term of the sequence is 6. So this is incorrect.)

Let me rewrite — a correct example is more important:

Sequence: $0, 1, 2, 3, 4, 5, \ldots$
- **Learner A**: $f(n) = n$ (natural numbers)
- **Learner B**: $f(n) = $ "the $n$-th non-negative integer" (same as A)
- **Learner C**: $f(n) = n$ for $n \leq 1000$, then $f(n) = n + 1000$

For the first 5 elements (0,1,2,3,4), the three learners are **in perfect agreement**. You cannot distinguish them from the data. But for the 1001st element: A and B predict 1000, C predicts 2000.

**What does this example reveal?** Finite data can never uniquely determine a regularity — underdetermination is not "temporarily insufficient data," but "underdetermination strong enough that it does not disappear even with data in the limit." The only way out is to introduce inductive bias — asserting that C is "more complex" and therefore not as good as A. But this assertion itself does not come from the data; you bring it in.

And note: if the true regularity is indeed C (for example, a sudden jump after the 1000th element), then your inductive bias — favoring A — will cause you to systematically err. Inductive bias is not a free lunch: it helps you in some worlds and hurts you in others. And you cannot determine from training data which world you live in.

Returning to the Compression Detective Game, the detective never enters the crime scene empty-handed. He carries some encoding scheme, some presupposition about "what counts as a short description." Thus the same batch of clues, in the eyes of different detectives, will lead to different regularities: linear models see straight lines, decision trees see branches, neural networks see hierarchical representations. So-called learning is not the data speaking for itself, but the data being forced to answer within some compression language.

This sentence is uncomfortable, but it must be kept: the failure of a learning system is often not because it found no regularities, but because it found an overly beautiful regularity under the wrong rules of the game.

---

## 21.6 The Unified Picture of Learning and Inference

Let us place the content of this chapter within the framework of Volume 2.

Chapter 14 established the syntactic machine of inference. Chapter 15 discovered that this machine has intrinsic limitations — certain true propositions cannot be proved from within. Chapter 16 modified one structural rule, obtaining a new kind of inference. Chapter 17 expanded truth values to $[0,1]$, yielding probabilistic inference. Chapter 18 introduced the intervention operator, yielding causal inference. Chapter 19 discovered that the computational cost of these inferences has a fundamental lower bound. Chapter 20, under cost constraints, precisely defined "good enough."

What Chapter 21 does is: embed "learning" into this framework. Learning is **inverse inference** — starting from observations, reverse-engineering regularities. The MDL principle provides the basic principle of inverse inference: the shortest explanation. The PAC framework provides the quality guarantee of inverse inference: probably approximately correct.

But this unified picture still lacks its final piece: when the inference system becomes sufficiently large and complex, its "inferences" begin to include propositions about itself — about what it can learn, what it cannot learn, what its own inductive bias is. At this point, the Gödelian structure of Chapter 15 reappears: **a sufficiently strong inference system cannot fully infer itself**.

### 21.6.5 Parallel Structure: The Incompleteness of Learning

Place the logical incompleteness of Ch14-15 and the learning incompleteness of Ch20-21 side by side. These two lines of argument share a deep structure:

| | Formal System (Ch14-15) | Learning System (Ch20-21) |
|---|---|---|
| **Starting point** | Axioms + Inference rules | Training data + Inductive bias |
| **What can be done internally** | Generate all theorems (syntax) | Search hypothesis space (learning) |
| **Externally required** | Semantic assignment (model theory) | Generalization guarantees (PAC/MDL) |
| **Cannot be internalized** | Proof of system consistency | Justification of inductive bias |
| **Boundary manifestation** | There exist true but unprovable propositions | There exist learnable concepts requiring external bias |

This is not "kind of similar." This is **the same mathematical structure instantiated in two domains**. In both cases:
1. Legitimate operations within the system (proving/learning) can proceed indefinitely
2. But the system cannot internally verify the **correctness conditions** of these operations (consistency/generalization)
3. Correctness conditions require an **external perspective** — formal systems need models, learning systems need distributional assumptions

**The deepest question**: if we make inductive bias itself an object of learning — meta-learning — can we break through this limitation?

Meta-learning does indeed learn a bias over a distribution of tasks, enabling faster adaptation to new tasks. But meta-learning itself also has an inductive bias: it carries assumptions about the task distribution (e.g., "tasks share common structure") regarding "what kind of bias performs well on the task distribution." If we make meta-learning itself an object of learning (meta-meta-learning), this chain has no end — each layer requires its own inductive bias, and each layer's inductive bias must come from external input.

This is fully parallel to the Gödelian structure: adding $G$ as an axiom -> the new system has its own $G'$ -> you can never outrun it. Turning inductive bias into a learning object -> meta-learning has its own meta-bias -> you can never bootstrap your way out.

This is the entrance to Chapter 22. The convergence point of the eight boundaries is in Chapter 25.

---

## Unresolved

**If Kolmogorov complexity is uncomputable, does MDL still have meaning?** The uncomputability of Kolmogorov complexity means it is an ideal, not an algorithm. But mathematics is full of such ideals: optimal decisions, perfect information, infinite precision. Ideals need not be computable; they are needed as benchmarks — telling us how far actual algorithms are from the optimum. The computable approximations of MDL are finite approximations to this ideal; the existence of the ideal itself makes the word "approximation" meaningful.

**Is generalization truly equivalent to compression?** The MDL framework suggests: a model that cannot compress training data will not generalize; a model that can compress will generalize. But this equivalence does not always hold in the practice of deep learning — some large-scale models "memorize" data while also generalizing well. This tension points to an unresolved theoretical problem: what exactly is the generalization mechanism of deep learning?

**Can inductive bias be learned?** If learning is inverse inference, then the matter of "what hypothesis space is good" — can this also be learned? Meta-learning attempts to learn a good inductive bias over a distribution of tasks, enabling the model to adapt faster on new tasks. But meta-learning itself also has an inductive bias — is this an infinite regress, or can it stop at some level?

---

## Exercises

**★ Warm-up**

Of the following two strings, which has lower Kolmogorov complexity? Only an intuitive judgment and a one-sentence justification are required; no specific numerical computation is needed.

1. `0101010101010101010101010101010101010101` (40 characters, alternating 01)
2. `1101001000011010110101010001010001000010` (40 random characters)

Also think about this question: for a "completely random" string, what is its Kolmogorov complexity approximately? How does it compare to the length of the string itself?

---

**★★ Derivation**

1. **Compression and generalization**: One model has zero error on the training set, but large error on the test set (overfitting); another has 5% error on training, also 5% on test (good generalization). Use MDL language to describe the difference: which one compressed the data? Which one memorized the data? What does the "description length of the overfitting model" and the "description length of the generalizing model" each contain?

2. **Making inductive bias explicit**: Linear regression, decision trees, and neural networks (with L2 regularization) — these three learning algorithms each implicitly what inductive bias? Describe in terms of "what kind of regularities are preferentially chosen." Under what type of data would each bias systematically fail?

---

**★★★ Challenge**

Kolmogorov complexity is uncomputable (because it is equivalent to the halting problem). This means: no algorithm can, for any two hypotheses, determine in finitely many steps which is simpler.

Yet MDL is used in practice, with computable approximate versions. Try to list at least two computable "description length" approximation schemes, and analyze: what inductive bias does each approximation implicitly correspond to? Under what circumstances would different approximation schemes give different "optimal hypotheses"?

The purpose of this question is: translate "hyperparameter selection" into "inductive bias selection," then translate that into "implicit commitments about what counts as simple" — after doing these three layers of translation, say whether your view on the "objectivity" of machine learning has changed.

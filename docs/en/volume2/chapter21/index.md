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
- "the first million natural numbers" → a very short program suffices (`for i in range(1000000): print(i)`), $K$ is very small
- A randomly generated million-digit number → no shorter description than "list the digits themselves," $K \approx$ string length

**The important uncomputability**: $K(x)$ is uncomputable — there exists no algorithm that can compute the exact value of $K(x)$ for arbitrary $x$ (this is a corollary of the halting problem). You can only compute an **upper bound** (find a program that can generate $x$; its length is an upper bound).

**Why is it still useful?** Even though uncomputable, Kolmogorov complexity provides a **theoretical standard** — an absolute benchmark for measuring "simplicity." In practice, MDL approximations are used (substituting actual compression algorithms for the shortest program).
:::


In this language, "simple" means "small Kolmogorov complexity," and "simpler explanations are preferred" means "choose the regularity with smaller $K$ value."

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

---

## 21.5 Two Difficulties of Inverse Inference

Learning as inverse inference faces two fundamental difficulties, neither of which can be resolved by the data itself.

**The first difficulty: underdetermination**. As described above, finite data are compatible with infinitely many hypotheses; there must be an inductive bias to choose among them. The inductive bias is input, not inferred from the data. The solution to the learning problem depends on the choice of hypothesis space — and the choice of hypothesis space falls outside the jurisdiction of the learning algorithm.

**The second difficulty: distribution shift**. PAC learning assumes that training data and test data come from the same distribution $\mathcal{D}$. But in practical applications, this assumption is often violated — the data distribution at training time and the data distribution at deployment time may differ. When distribution shift occurs, any generalization guarantee based on training data may become void.

These two difficulties are fundamental limitations of learning theory, not defects of algorithms. They indicate: **any learning system operates under certain assumptions; beyond those assumptions, guarantees vanish**.

This structure is highly similar to the formal systems of Chapter 14: a formal system is reliable within its axioms; beyond the axioms, there are no guarantees. A learning system is reliable within its assumptions; beyond the assumptions, there are no guarantees. The reliability of both depends on an external commitment that cannot be verified from within.

Returning to the Compression Detective Game, the detective never enters the crime scene empty-handed. He carries some encoding scheme, some presupposition about "what counts as a short description." Thus the same batch of clues, in the eyes of different detectives, will lead to different regularities: linear models see straight lines, decision trees see branches, neural networks see hierarchical representations. So-called learning is not the data speaking for itself, but the data being forced to answer within some compression language.

This sentence is uncomfortable, but it must be kept: the failure of a learning system is often not because it found no regularities, but because it found an overly beautiful regularity under the wrong rules of the game.

---

## 21.6 The Unified Picture of Learning and Inference

Let us place the content of this chapter within the framework of Volume 2.

Chapter 14 established the syntactic machine of inference. Chapter 15 discovered that this machine has intrinsic limitations — certain true propositions cannot be proved from within. Chapter 16 modified one structural rule, obtaining a new kind of inference. Chapter 17 expanded truth values to $[0,1]$, yielding probabilistic inference. Chapter 18 introduced the intervention operator, yielding causal inference. Chapter 19 discovered that the computational cost of these inferences has a fundamental lower bound. Chapter 20, under cost constraints, precisely defined "good enough."

What Chapter 21 does is: embed "learning" into this framework. Learning is **inverse inference** — starting from observations, reverse-engineering regularities. The MDL principle provides the basic principle of inverse inference: the shortest explanation. The PAC framework provides the quality guarantee of inverse inference: probably approximately correct.

But this unified picture still lacks its final piece: when the inference system becomes sufficiently large and complex, its "inferences" begin to include propositions about itself — about what it can learn, what it cannot learn, what its own inductive bias is. At this point, the Gödelian structure of Chapter 15 reappears: **a sufficiently strong inference system cannot fully infer itself**.

This is the entrance to Chapter 22.

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

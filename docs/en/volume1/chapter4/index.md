# Chapter 4: The Manifold Hypothesis — The Hidden Order of High-Dimensional Data

> You think data is scattered at random through high-dimensional space. It isn't. It's squeezed onto a thin, curved surface.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. The Absurdity of Dimension

First, a thought experiment.

Drop 100 points at random on a line segment. On average, the distance between neighboring points is about 1/100 of the segment's length — dense, well sampled.

Now put those 100 points into a square. To keep the same sampling density, you need $100^2 = 10000$ points.

A cube? That needs $100^3 = 1000000$ points.

This is the **Curse of Dimensionality** — when Bellman named the phenomenon in 1957, he probably didn't expect it to haunt the entire field of machine learning.

:::details The Curse of Dimensionality: Why Is Everything Sparse in High Dimensions?
The **Curse of Dimensionality** says: as the dimension of data grows, covering the space at the same density requires an exponentially growing number of samples.

An intuitive example:
- 100 points on a 1D line (0 to 1): average spacing = 0.01 — still fairly dense
- To keep the same density in a 2D square, you need 100² = 10,000 points
- A 10D hypercube needs 100¹⁰ = 10²⁰ points — more than the number of atoms in the observable universe

It's not just "we need more data." The deeper issue: in high-dimensional space, **the very concept of distance breaks down** — the distances between all points converge to the same value, and "nearest neighbor" becomes indistinguishable.

That's why many algorithms that work well in low dimensions (k-NN, clustering) perform badly in high dimensions. And the manifold hypothesis (below) offers exactly a way out: real data is not uniformly scattered through high-dimensional space — it has low-dimensional structure.
:::


As dimension grows, the number of samples needed to cover space at equal density explodes exponentially. But that's only the surface of the problem. The truly strange things happen in geometry.

Take two random points inside a $d$-dimensional unit hypersphere. Their distance satisfies:

$$
\mathbb{E}[\|x - y\|] \to \sqrt{2}, \quad d \to \infty
$$

while the variance of the distance goes to zero.

Which means: in high-dimensional space, all distances between points converge to the same value. The nearest neighbor and the farthest neighbor become almost equally far. Distance-based similarity measures — cosine similarity, Euclidean distance, $k$-NN classification — all fail in high dimensions, because "near" has lost its discriminating power.

If you randomly initialize two 1000-dimensional vectors, their cosine similarity is almost surely close to zero. Not because they are semantically opposite, but because in 1000 dimensions, random vectors are almost surely orthogonal.

**What does orthogonality mean?** Two vectors are orthogonal, i.e. $\vec{u} \cdot \vec{v} = 0$, which geometrically means they stand perpendicular to each other and are uncorrelated — the projection of one vector onto the other's direction is zero. In low dimensions, orthogonality is a special relationship that must be deliberately constructed. In high dimensions, orthogonality is the default state, the inevitable product of randomness.

This fact echoes back to us in two places.

**Echo 1: The conceptual space of expert systems.** In Chapter 2, expert systems represented knowledge with discrete symbols — `"is a mammal"`, `"has wings"`, `"can fly"` — each symbol an atom, and the relationships between concepts had to be written explicitly into rules. Look at this again from the viewpoint of high-dimensional geometry: each discrete symbol amounts to a dimension placed on some orthogonal basis. `"Cat"` and `"dog"` have no intrinsic relation in a symbolic system; they are two independent tokens, just as two random unit vectors are almost surely orthogonal. The expert system's fundamental dilemma — its inability to automatically capture continuous similarity between concepts — from this angle, is because it forces the world into an **artificial orthogonal basis**. The world is not orthogonal, but the symbolic system pretends it is.

::: info Professor Pallas's Cat Commentary
The orthogonality of high-dimensional space is not a bug, but a feature. In extremely high dimensions, random vectors are almost surely orthogonal — which means they are naturally decoupled, independent of each other. This decoupling is exactly the precondition for high-dimensional embedding learning to succeed: each dimension can independently characterize a different semantic aspect, without having to "stick together" the way they must in low-dimensional space.

The expert system's problem is precisely that it **deliberately manufactures** this orthogonality — but in an extremely inefficient number of dimensions. It hard-codes each concept as an orthogonal basis vector, creating a finite-dimensional, artificial, discrete orthogonal representation. This representation is conceptually clean, but extremely inefficient in representational power — it cannot capture continuous similarity between concepts, and it cannot exploit the "free lunch" that high-dimensional space's natural orthogonality provides.

What makes real high-dimensional embeddings (such as word vectors) powerful is that they let the data decide how to exploit high-dimensional orthogonality while preserving semantic continuity. Training does not force orthogonality; it lets statistical constraints push the vectors onto a manifold — a surface that is neither a fully orthogonal random desert, nor a fully correlated low-dimensional swamp, but a carefully folded, semantically rich surface.
:::

**Echo 2: The analogy arithmetic of word vectors.** In Chapter 3, we saw the neat equation $\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$ and asked: is this reasoning, or statistical coincidence? Now, with high-dimensional orthogonality as background, we can state its limit more precisely: word vectors can do this arithmetic precisely because training data pushed them away from the orthogonal state of random initialization — the co-occurrence constraints in the corpus pull random vectors into a low-dimensional semantic manifold. The analogy arithmetic works: credit goes to manifold structure, not to the algorithm's wisdom. And when the analogy fails (`doctor - man + woman ≈ nurse`, rather than female doctor), the reason is that the training corpus's bias has warped the manifold — the curvature of the semantic space in that region is not the curvature of "objective concepts", but the curvature of "corpus statistical bias".

These two echoes point to one question: if random high-dimensional space is an orthogonal desert, then "meaningful data" must be learnable precisely because the data is not random — something has compressed it into a low-dimensional structure. That is what the manifold hypothesis is about.

<figure>
<p><img src="/figures/ch04_fig1_curse_of_dimensionality.png" alt="The Curse of Dimensionality: the collapse of the distance distribution of random point pairs as dimension grows" /></p>
<figcaption>The Curse of Dimensionality: the collapse of the distance distribution of random point pairs as dimension grows</figcaption>
</figure>

*Figure 1: Left, the distance distribution of random point pairs in 2, 10, 100, and 1000 dimensions. The higher the dimension, the more concentrated the distribution, variance → 0 — "distance" loses its ability to discriminate. Right, a histogram of pairwise cosine similarities of 1000-dimensional random vectors, almost all piled up near zero.*

<div class="center">

------------------------------------------------------------------------

</div>

This confronts machine learning with a fundamental problem: if data really were high-dimensional random, then any algorithm based on distance or similarity — including neural networks — should not work.

But they do work.

There are two possible explanations:

First, these algorithms get around the limits of high-dimensional geometry in some way we don't fully understand.

Second, the data itself is not high-dimensional random.

The second explanation is right.

<div class="center">

------------------------------------------------------------------------

</div>

## A Pause

Wait — before accepting the conclusion that "the second explanation is right", let's slow down.

Why does natural data gather on a low-dimensional manifold? This is a fact that needs explaining, not an axiom.

One answer: the laws of physics constrain the space of possible states — face variation is governed by the physics of bones, muscles, and lighting, so the possible faces are vastly fewer than the possible pixel combinations.

But this answer smuggles in an assumption: **the world is lawful, and those laws are low-dimensional.**

Why should that assumption hold? If the world's real laws were high-dimensional and chaotic, the manifold hypothesis would collapse — and machine learning shouldn't work.

Then why does it in fact work? Is it because physical laws really are low-dimensional, or because we just happened to test it only on "the problems where the manifold hypothesis holds"?

Leave that question aside for now.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Manifold Hypothesis: Constraints Create Surfaces

The core claim of the **Manifold Hypothesis** is:

> Natural data (images, language, sound) lives in high-dimensional space, but it actually gathers near a smooth surface (a manifold) whose intrinsic dimension is far lower than the ambient dimension of the space it lives in.

"A manifold" is a mathematical term, meaning a topological structure that looks like Euclidean space locally. A 2D surface embedded in 3D space is a manifold — the earth's surface is a manifold, a Möbius strip is a manifold, the surface of a bent pipe is a manifold.

:::details Manifold: Understand It Through the Earth's Surface
A **manifold** is a geometric object defined this way: it looks "flat" locally (like an ordinary Euclidean plane), but can be curved as a whole.

The best example: **the earth's surface**. Stand at any point on earth and look only at the tiny patch beneath your feet — it looks no different from flat ground. That is "locally like Euclidean space". But seen as a whole, the earth's surface is spherical, not planar. The earth's surface is a 2D manifold embedded in 3D space.

In machine-learning terms:
- The 49,152-dimensional space of face images is the analogue of 3D space
- The tens-of-dimensions surface where real faces actually live is the analogue of the earth's surface

**Why does the manifold hypothesis matter?** Because it explains why machine learning works: the data is not high-dimensional random, but gathered on a low-dimensional surface, and algorithms are in fact learning the structure of this low-dimensional surface.
:::


Build intuition with a concrete example:

A $128 \times 128$ color face image, as a vector, has dimension $128 \times 128 \times 3 = 49152$. In that space, the possible pixel combinations number $256^{49152}$ — more orders of magnitude than the atoms in the universe.

But "real faces" occupy only an extremely small region of this space. What parameters determine a real face?

- Identity (whose face)

- Pose (how many degrees the head is turned, the elevation angle)

- Lighting (where the light comes from, how strong it is)

- Expression (smiling? frowning?)

- Age

Roughly a dozen to a few dozen parameters can describe most of the variation in a face. This means the intrinsic dimension of face images is on the order of tens — not 49,152.

Image data is squeezed onto a roughly tens-of-dimensions manifold embedded in 49,152-dimensional space.

Language data is the same. "Valid English sentences" are only a tiny fraction of all possible word sequences. "Sensible meanings" are an even smaller fraction of all valid English sentences. Every layer of constraint is one more fold and compression of the manifold.

<figure>
<p><img src="/figures/ch04_fig2_manifold_embedding.png" alt="Manifold embedding: the structure of a low-dimensional surface in high-dimensional space" /></p>
<figcaption>Manifold embedding: the structure of a low-dimensional surface in high-dimensional space</figcaption>
</figure>

*Figure 2: Left, a 2D manifold (the Swiss Roll dataset) embedded in 3D space — the 3D coordinates look complicated, yet the data is distributed smoothly along two intrinsic dimensions (the unfolded u, v coordinates). Right, manifold learning (UMAP) on the same dataset recovers the intrinsic 2D structure. Color encodes the continuity of the intrinsic coordinates — local neighbors stay neighbors in high-dimensional space.*

<div class="center">

------------------------------------------------------------------------

</div>

Why does natural data gather on a manifold? Not by coincidence — it is the **inevitable result of the generation process**.

Any data is produced by some generation process. Face images are produced by biological processes — DNA, bone structure, skin texture; these parameters are finite and vary continuously. Language is produced by grammar rules and semantic constraints. Music is produced by physics (sound waves) and cultural conventions.

The **constraints** of the generation process are the **curvature** of the manifold. The stronger the constraints, the lower-dimensional and more curved the manifold.

This is a profound fact: the intrinsic dimension of data is not a property inherent to the data, but a projection of the **structure of the world that generates it**.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. Training Is Folding

Now we've reached the most important part of this chapter.

Usually we say neural networks "learn" — learn to recognize cats, learn to translate languages, learn to predict the next word. This phrasing is dangerous. "Learning" suggests some kind of understanding, some process of internalizing knowledge.

A more accurate statement: **neural networks perform manifold transformations.**

Unfold how an autoencoder works:

$$
\text{Input } x \in \mathbb{R}^D \xrightarrow{\text{Encoder } f} \text{Latent code } z \in \mathbb{R}^d \xrightarrow{\text{Decoder } g} \text{Reconstruction } \hat{x} \in \mathbb{R}^D
$$

with $d \ll D$. The training goal is to minimize the reconstruction error $\|x - \hat{x}\|^2$.

What is happening here? The encoder $f$ learns to map high-dimensional inputs to the low-dimensional latent space $z$, and this low-dimensional space happens to be exactly the coordinate system of the data manifold. The decoder $g$ learns to reconstruct high-dimensional inputs from manifold coordinates.

In other words, what the encoder learns is a **parameterization of the manifold** — it finds a set of coordinates that can describe points on the manifold. What the decoder learns is the mapping between those coordinates and the original high-dimensional representation.

It's not just autoencoders. Classification networks do the same thing. When you train a ResNet to recognize image categories, the feature vector before the final fully-connected layer is exactly the manifold coordinates the network learned — only this coordinate system is deliberately designed to be "useful for classification".

Backpropagation is the process of adjusting this coordinate system. Every gradient update changes the way the network maps high-dimensional inputs to the low-dimensional latent space — that is, it changes the parameterization of the manifold.

**The essence of training is folding the manifold structure of the data seen in the past into the curvature of the parameters.**

The word "folding" matters. Not "storage", not "memory" — folding. Training data is not stored item by item in the network — the network doesn't have nearly enough parameters to store every sample of the training set. What happens is: the statistical structure of the data, the geometric features of the manifold the data lives on, are compressed into the configuration of network weights.

$10^6$ face images, each 49,152-dimensional. ResNet-50 has about $2.5 \times 10^7$ parameters. With 25 million numbers, it remembers the manifold structure described by 49.2 billion pixel values. The compression ratio is about 200:1.

This is not learning. This is topological compression.

<figure>
<p><img src="/figures/ch04_fig3_compression_bottleneck.png" alt="Training: the mapping from high-dimensional data to compressed manifold parameters" /></p>
<figcaption>Training: the mapping from high-dimensional data to compressed manifold parameters</figcaption>
</figure>

*Figure 3: Top, the encoder gradually maps high-dimensional data (a 3D point cloud scattered on a 2D manifold) into a 1D latent code. Bottom, the evolution of the latent space during training — as epochs increase, the latent representation goes from chaotic to structured. Color represents the continuous coordinate of the data on the original manifold; continuous color means local structure is preserved.*

<div class="center">

------------------------------------------------------------------------

</div>

There is a deep question here worth pausing on: what is stored in the parameters?

An intuition is: the parameters store "knowledge" — what cats look like, the rules of grammar, the laws of physics.

But the more accurate statement is: the parameters store a **compressed encoding of the training data's manifold structure**.

Once you know this, you can understand why pretraining works — the same manifold (say, "natural images") can be shared by multiple tasks, and the manifold parameterization learned in pretraining can be transferred.

You can also understand why fine-tuning sometimes changes model behavior and sometimes doesn't — it depends on whether the manifold of your new task is a subset of the pretraining manifold.

And you begin to understand why large models hallucinate — the fourth act will cover this.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. The Bill for Compression

Any compression has a cost. This is not an empirical observation; it is an information-theoretic theorem.

Shannon's source coding theorem tells us: the limit of lossless compression is the entropy of the source. Beyond that limit, you must lose information. Neural network compression goes far beyond that limit, so it must be losing information.

What is lost?

**Tally 1: Points off the Manifold**

After training, the network's parameters encode the manifold where the training data lives. When at test time it encounters a point that is not on this manifold — images from a different distribution, language phenomena never seen, counterfactual reasoning scenarios — what does the network do?

It has no alarm saying "this point is not on my manifold". It can only drag the point forcibly to the nearest manifold region, then answer according to that region's logic.

Result: **confident errors**. Not "I don't know", but "it looks like X, so I answer X" — even when X is completely wrong.

This is the geometric root of the Distribution Shift problem: the manifold the model learned and the manifold of the real world disagree outside the training distribution. The larger the shift, the larger the error — but the confidence stays unchanged, because confidence is a function of manifold coordinates, not a function of "how far the point is from the manifold".

**Tally 2: Isometry inside the Manifold**

When compressing, some regions of the manifold are "stretched", others are "squeezed".

Where training data is dense, the network's learned parameterization has high resolution — adjacent inputs map to adjacent latent codes, and local geometry is preserved well.

Where training data is sparse, the network's parameterization has low resolution — adjacent inputs may map to far-apart places in latent space, or far-apart inputs may map to adjacent places.

This means, the network generalizes well where training data is dense and poorly where it is sparse — not random poorness, but patterned: the manifold structure of sparse regions gets "hallucinated", and what fills it in is the most common patterns of the training data, not the true structure of that region.

The result of that hallucinated filling is hallucination.

**Tally 3: Tishby's Information Bottleneck**

In 2017, Naftali Tishby proposed a theoretical framework for this process — the **Information Bottleneck**.

Training a neural network is essentially a trade-off:

$$
\min_{z} \left[ I(X; Z) - \beta \cdot I(Z; Y) \right]
$$

where $X$ is the input, $Y$ the label, $Z$ the intermediate representation, $I(\cdot; \cdot)$ mutual information, and $\beta$ the trade-off parameter.

The first term $I(X; Z)$ measures how much input information the intermediate representation $Z$ keeps — smaller is better, meaning more compression.

The second term $I(Z; Y)$ measures how much information about the label $Z$ keeps — larger is better, meaning more useful for the task.

The optimal solution of this trade-off is a "sufficient statistic" — it keeps all the information needed to predict the label and discards all redundant information.

Tishby's claim is that training happens in two stages — first the "fitting" stage, where mutual information $I(Z; Y)$ increases rapidly as the network learns to capture label-related information; then the "compression" stage, where $I(X; Z)$ starts to fall as the network begins to discard information irrelevant to the label.

This theory is beautiful. But it is also controversial — subsequent research (Saxe et al., 2018) found that the compression stage does not appear in many cases, depending on the choice of activation function. The information bottleneck is a valuable analytical framework, but not a complete theory `-> [Tishby & Schwartz-Ziv, 2017, arXiv:1703.00810]` `-> [Saxe et al., 2018, arXiv:1812.09881]`.

**The final tally**

Add the three tallies together, and the cost of manifold compression is:

1. In-distribution: good generalization, accurate confidence

2. At the edge of distribution: generalization starts to degrade, but confidence remains high (danger zone)

3. Out-of-distribution: poor generalization, yet confidence can still be very high (most dangerous zone)

The network does not know it is standing at the edge of the manifold. It only knows that the current input mapped to some coordinate in the latent space, and then it outputs according to that coordinate's logic. There is no built-in "uncertainty sensor" telling it: you are now making inferences about a region you have never truly understood.

This is not a bug; it is the inevitable structural consequence of manifold compression.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. The Limit of Compression: How Small Can a Manifold Be Squeezed?

The manifold hypothesis says high-dimensional data squeezes onto a low-dimensional manifold. So how low is "low"? How far can compression go?

This is an information-theoretic question, not just a geometric one.

**Rate-distortion theory gives a lower bound**

Shannon's rate-distortion theory tells us: to reconstruct a signal with distortion $D$, you need at least $R(D)$ bits of information. That $R(D)$ is an **information-theoretic lower bound** — no compression algorithm can break through it.

For data on a manifold:

$$
R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(x, \hat{x})] \leq D} I(X; \hat{X})
$$

When you compress a neural network's hidden layer dimension from $d$ to $k$, what you are really doing is lossy compression — some details of the original manifold are inevitably lost.

**Intrinsic dimension**

The **intrinsic dimensionality** of a manifold is the minimum number of coordinates needed to describe it. This quantity can be estimated from the distribution of pairwise distances:

$$
d_{\text{int}} = -\frac{\log n}{\log (r / R)}
$$

where $r$ is the near-neighbor radius and $R$ the maximum radius.

The experimental results are surprising: the intrinsic dimension of ImageNet images is about 40, far smaller than the raw pixel dimension ($224 \times 224 \times 3 \approx 150{,}000$). The intrinsic dimension of language models' representation spaces is about 13–20, far smaller than the hidden dimension (768 or more).

This shows that real-world data really does squeeze onto extremely low-dimensional manifolds.

**But there is a hard floor**

Compression is not unlimited. Once compressing below the intrinsic dimension, distortion rises sharply — you start losing the manifold's own structure, not just noise.

This is why knowledge distillation (compressing a large model into a small one) has a performance cliff: past a certain compression ratio, the model is not "a bit worse" — it "collapses completely". The rate-distortion curve drops precipitously after the elbow.

This hard floor, and how randomization can break through the limits of deterministic compression, is the subject of Chapter 11.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. A Small Pause

Let me take stock of what this chapter did.

In high-dimensional space, random data cannot be learned — distance loses meaning, and sampling requires exponentially many data points. Machine learning works not because it overcomes high-dimensionality, but because the data itself is not high-dimensional random. Natural data gathers on low-dimensional manifolds, and that is the joint result of the physical processes, biological constraints, and grammar rules that generate the data.

Training a neural network is the process of compressing the training data's manifold structure into the parameters. Not storage — compression. The compression ratio is on the order of 100:1 to 1000:1.

Compression has unavoidable costs: points off the manifold are forcibly projected, sparse regions get their structure hallucinated, and under distribution shift confidence and accuracy decouple from each other. These costs are not engineering flaws; they are structural consequences of information-theoretic constraints.

This leads to the question of Chapter 5: if what a model learns is manifold structure rather than causal laws, why does its behavior systematically fail outside the distribution? Where is the difference between statistical correlation and genuine reasoning?

<div class="center">

------------------------------------------------------------------------

</div>


> The manifold tells us where the data is, but not whether the model learned the real laws. In the next chapter, we will see how statistical fitting becomes a trap for reasoning.

## Open Questions

- How well does the manifold a neural network learns actually "match" the manifold of real-world data? We have no good tools for measuring this match.

- Can the intrinsic dimension of a manifold be estimated? What methods exist, and what does each assume? In the activation space of an LLM, what is the intrinsic dimension?

- Does the information lost in "compression" include the structure that reasoning requires? Or does reasoning ability happen to live in the part that was kept?

- If you trained a neural network on completely random data, what kind of "manifold" would it learn? The answer to this thought experiment would tell us how necessary the manifold hypothesis is.

- The two-stage information bottleneck theory (fitting + compression) is still experimentally contested. If the compression stage doesn't exist, what explains neural network generalization?

- **The manifold structure of knowledge**: a knowledge graph is a discrete symbolic manifold, and word vectors are a continuous semantic manifold. How do these two manifolds **map to each other and validate each other**? Is there a unified "knowledge manifold" theory?

- **Manifolds at multiple levels**: the phase-space manifolds of physical systems (life), the logical-space manifolds of symbolic systems (knowledge), the representation-space manifolds of statistical systems (learning) — do **isomorphisms or mapping relations** exist among these different levels of manifolds?

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Compress a Manifold, Then Find Its Bill

The core claim of this chapter is: compression has costs, and the location of those costs is not random — it is patterned. You will verify this by hand with a minimal autoencoder — not to get a good model, but to pinpoint exactly where compression cheated.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Generate Your Manifold Data

Don't use a real dataset — too many distractions. Start from a manifold whose **generation process you fully understand**.

Generate one of the following:

**Option A: A 2D Torus Embedded in 3D**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 2000  # number of sample points

# Uniformly sample the two angular parameters of the torus
t = np.random.uniform(0, 2 * np.pi, N)   # angle around the main axis
phi = np.random.uniform(0, 2 * np.pi, N) # angle around the tube cross-section

r = 3.0    # main ring radius
tube = 1.0 # tube radius

# Torus parametric equations: embedded in 3D space
x = (r + tube * np.cos(phi)) * np.cos(t)
y = (r + tube * np.cos(phi)) * np.sin(t)
z = tube * np.sin(phi)

# Add a small amount of Gaussian noise to simulate measurement error
noise_std = 0.05
x += np.random.normal(0, noise_std, N)
y += np.random.normal(0, noise_std, N)
z += np.random.normal(0, noise_std, N)

# Stack the three coordinates into a data matrix of shape=(N, 3)
data_torus = np.stack([x, y, z], axis=1)
# Save intrinsic coordinates for later coloring (t is the main angle, phi is the tube angle)
intrinsic_t = t
intrinsic_phi = phi

# Visualize (colored by the t value)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=t, cmap='hsv', s=2, alpha=0.6)
plt.colorbar(sc, label='intrinsic coordinate t')
ax.set_title('Torus manifold (Option A)')
plt.tight_layout()
plt.show()
```

**Option B: Swiss Roll (the dataset in this chapter's figure)**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

np.random.seed(42)
N = 2000  # number of sample points

# Generate the standard Swiss Roll with sklearn
# t is the intrinsic coordinate (unfold angle), z is the intrinsic coordinate along the height
data_swiss, t_swiss = make_swiss_roll(n_samples=N, noise=0.1, random_state=42)
# data_swiss has shape=(N, 3); the columns are x, y, z (3D embedding coordinates)

# Or generate it manually (equivalent):
t_manual = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, N)  # unfold angle
x_manual = t_manual * np.cos(t_manual)
z_manual = np.random.uniform(0, 1, N)                        # height (second intrinsic dimension)
y_manual = t_manual * np.sin(t_manual)
# Add a small amount of Gaussian noise
x_manual += np.random.normal(0, 0.1, N)
y_manual += np.random.normal(0, 0.1, N)
z_manual += np.random.normal(0, 0.05, N)

# Visualize the manually generated Swiss Roll (colored by t)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_manual, y_manual, z_manual, c=t_manual, cmap='viridis', s=2)
plt.colorbar(sc, label='intrinsic coordinate t (unfold angle)')
ax.set_title('Swiss Roll manifold (Option B)')
plt.tight_layout()
plt.show()

# Use the sklearn version from here on (more standard)
data = data_swiss          # 3D coordinates, shape=(N, 3)
intrinsic_t = t_swiss      # intrinsic coordinate t, used for coloring
```

Option B has an intrinsic dimension of 2 (t and z), embedded in 3D space.

**Your first question (answer before writing code):** What is this data's "true intrinsic dimension"? If you use PCA for dimensionality reduction, how many principal components do you think you need to keep to retain most of the variance? Write down your prediction; compare it with the actual result later.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Implement a Minimal Autoencoder

Structure definition:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Autoencoder architecture ---
class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        # Encoder: 3 -> 16 -> d (bottleneck dimension)
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),      # 3D input (embedded data points)
            nn.ReLU(),             # non-linear activation
            nn.Linear(16, bottleneck_dim)  # d-dimensional latent code z
        )
        # Decoder: d -> 16 -> 3 (reconstruct the original coordinates)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)       # 3-dimensional reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)    # encode
        x_hat = self.decoder(z)  # decode
        return x_hat, z

# --- Training function ---
def train_autoencoder(data_tensor, bottleneck_dim, n_epochs=300, lr=1e-3):
    """
    Train the autoencoder until the reconstruction error converges.
    Returns the trained model.
    """
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = Autoencoder(bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # loss function: mean squared error (reconstruction error)

    for epoch in range(n_epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = loss_fn(x_hat, batch)  # reconstruction error
            loss.backward()
            optimizer.step()

    return model

# --- Example: use the Swiss Roll data, starting with bottleneck dimension d=2 ---
# (Assume data was generated in the previous step, shape=(N, 3))
data_tensor = torch.tensor(data, dtype=torch.float32)  # convert to a PyTorch tensor

# Train an autoencoder with bottleneck dimension d=2
model_d2 = train_autoencoder(data_tensor, bottleneck_dim=2)
print("Autoencoder for d=2 trained")
```

Train until the reconstruction error converges. No hyperparameter tuning — the default settings are fine.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Systematically Change the Bottleneck Dimension and Record the Bill

Train with d = 1, 2, 3, 4, 5, and for each d record:

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Split the data into train and test sets (80% / 20%)
N = len(data_tensor)
split = int(0.8 * N)
train_data = data_tensor[:split]
test_data  = data_tensor[split:]

results = {}  # stores the results for each bottleneck dimension

for d in [1, 2, 3, 4, 5]:
    # 1. Train the autoencoder
    model = train_autoencoder(train_data, bottleneck_dim=d, n_epochs=300)
    model.eval()

    with torch.no_grad():
        # 2. Compute the average reconstruction error (MSE) on the train set
        train_hat, train_z = model(train_data)
        train_mse = nn.MSELoss()(train_hat, train_data).item()

        # 3. Compute the average reconstruction error on the test set
        test_hat, test_z = model(test_data)
        test_mse = nn.MSELoss()(test_hat, test_data).item()

    results[d] = {'train_mse': train_mse, 'test_mse': test_mse,
                  'train_z': train_z.numpy(), 'model': model}

    # 4. Visualize the latent space (colored by intrinsic coordinate t)
    train_t = intrinsic_t[:split]  # intrinsic coordinates of the train set
    if d == 1:
        # d=1: the latent space is a line, plot a 1D scatter
        fig, ax = plt.subplots(figsize=(8, 2))
        sc = ax.scatter(train_z.numpy()[:, 0],
                        np.zeros(len(train_z)),
                        c=train_t, cmap='viridis', s=5)
        plt.colorbar(sc, label='intrinsic coordinate t')
        ax.set_title(f'd={d} latent space (1D)')
    elif d == 2:
        # d=2: plot a 2D scatter directly
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(train_z.numpy()[:, 0],
                        train_z.numpy()[:, 1],
                        c=train_t, cmap='viridis', s=5)
        plt.colorbar(sc, label='intrinsic coordinate t')
        ax.set_title(f'd={d} latent space (2D)')
    plt.tight_layout()
    plt.show()

    # 5. Sample 20 points from the latent space, decode them, and plot them in 3D
    with torch.no_grad():
        # Uniformly sample the latent space (using the range of the train-set latent codes)
        z_min = train_z.min(dim=0).values
        z_max = train_z.max(dim=0).values
        z_samples = torch.rand(20, d) * (z_max - z_min) + z_min
        decoded_samples = model.decoder(z_samples).numpy()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:split, 0], data[:split, 1], data[:split, 2],
               c='lightblue', s=1, alpha=0.3, label='train data')
    ax.scatter(decoded_samples[:, 0], decoded_samples[:, 1], decoded_samples[:, 2],
               c='red', s=50, zorder=5, label='randomly decoded points')
    ax.set_title(f'd={d} random decoding (randomly sampled latent points)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"d={d}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}")

print("\nComparison across bottleneck dimensions:")
for d, r in results.items():
    print(f"  d={d}: train MSE={r['train_mse']:.4f}, test MSE={r['test_mse']:.4f}")
```

**Your second question:** When d=1, the encoder compressed a 2D manifold into 1D. What did it keep? What did it lose? Does the arrangement of points in the latent space preserve the original manifold's topology (for example, the circular loop relationship)?

**Your third question (the core question):** When d=2, if you color the latent-space points by the original intrinsic coordinates (t and φ, or t and z), do the colors vary continuously? If yes, what does that mean? If not, what does that mean?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Find Out What Happens to Out-of-Distribution Points

Generate a batch of points that are **not on your manifold** — for example, random Gaussian noise points in 3D.

Feed that batch through your autoencoder:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Use the d=2 model for out-of-distribution testing
model = results[2]['model']
model.eval()

# OOD data: sample 100 points from a standard 3D Gaussian distribution
# Note: these points are scattered across the whole 3D space, not restricted to the manifold
ood_data = torch.randn(100, 3)  # standard normal with mean 0 and variance 1

with torch.no_grad():
    # Reconstruct with the autoencoder (encode -> decode)
    ood_reconstructed, ood_z = model(ood_data)

# Compute the reconstruction errors for both datasets
loss_fn = nn.MSELoss(reduction='none')  # compute the error for each sample individually
with torch.no_grad():
    train_hat, _ = model(train_data)
    # MSE of each train sample (averaged over the three dimensions)
    train_errors = loss_fn(train_hat, train_data).mean(dim=1).numpy()
    # MSE of each OOD sample
    ood_errors   = loss_fn(ood_reconstructed, ood_data).mean(dim=1).numpy()

# Compare the reconstruction errors
print(f"Average reconstruction error on train set: {train_errors.mean():.4f} ± {train_errors.std():.4f}")
print(f"Average reconstruction error on OOD points: {ood_errors.mean():.4f} ± {ood_errors.std():.4f}")

# Visualize: plot the reconstruction of the OOD points in 3D
# Observe where they land (whether they are "pulled" toward the manifold)
fig = plt.figure(figsize=(10, 5))

# Left panel: the original OOD points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:500, 0], data[:500, 1], data[:500, 2],
            c='lightblue', s=2, alpha=0.3, label='train manifold')
ax1.scatter(ood_data[:, 0], ood_data[:, 1], ood_data[:, 2],
            c='red', s=30, label='OOD points (original)')
ax1.set_title('OOD points (scattered in 3D space)')
ax1.legend(fontsize=8)

# Right panel: OOD points after reconstruction by the autoencoder
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(data[:500, 0], data[:500, 1], data[:500, 2],
            c='lightblue', s=2, alpha=0.3, label='train manifold')
ax2.scatter(ood_reconstructed[:, 0].numpy(),
            ood_reconstructed[:, 1].numpy(),
            ood_reconstructed[:, 2].numpy(),
            c='orange', s=30, label='OOD points (reconstructed)')
ax2.set_title('OOD points after reconstruction — "pulled" toward the manifold')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

**Your fourth question:** After going through the autoencoder, where do the out-of-distribution points get "pulled"? Are their reconstructions random, or do they systematically land in some region of the training manifold?

This is what Section 4 of this chapter said: "The network has no alarm saying 'this point is not on my manifold'. It can only drag the point forcibly to the nearest manifold region." You just watched this process with your own hands.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 5: Estimate Your Manifold's Intrinsic Dimension

Back to Step 1 — now do a data-driven intrinsic-dimension estimate:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Run PCA on the training data (a numpy array, not a tensor)
train_np = data[:split]  # shape=(N_train, 3)

pca = PCA()               # keep all components (no limit)
pca.fit(train_np)

# Variance ratio explained by each principal component
explained_ratio = pca.explained_variance_ratio_
# Cumulative explained variance curve
cumulative_ratio = np.cumsum(explained_ratio)

# Plot the cumulative variance curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: variance explained by each principal component
axes[0].bar(range(1, len(explained_ratio) + 1), explained_ratio, color='steelblue')
axes[0].set_xlabel('Principal component index')
axes[0].set_ylabel('Explained variance ratio')
axes[0].set_title('Variance explained by each principal component')

# Right panel: cumulative explained variance curve, find the "elbow"
axes[1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio,
             marker='o', color='tomato')
axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of principal components')
axes[1].set_ylabel('Cumulative explained variance ratio')
axes[1].set_title('Cumulative variance curve (find the elbow)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Find how many principal components are needed to explain 95% of the variance
n_components_95 = np.searchsorted(cumulative_ratio, 0.95) + 1
print(f"Principal components needed for cumulative variance >= 95%: {n_components_95}")
print(f"Variance explained by each component: {explained_ratio.round(4)}")
print(f"Cumulative explained variance: {cumulative_ratio.round(4)}")

# Compare with the autoencoder results
print(f"\nIntrinsic dimension estimated by PCA: {n_components_95}")
print("Test MSE of the autoencoder at each bottleneck dimension:")
for d, r in results.items():
    print(f"  d={d}: test MSE={r['test_mse']:.4f}")
print("(The point where the MSE jumps sharply corresponds to the intrinsic dimension estimated by the autoencoder)")
```

Compare this number with the prediction you wrote down in Step 1.

**Your fifth question:** Is the intrinsic dimension you estimated with PCA consistent with the "smallest lossless bottleneck dimension" you got from the autoencoder? If they differ, what do you think causes the difference — PCA's linearity assumption, or the autoencoder's nonlinearity?

<div class="center">

------------------------------------------------------------------------

</div>

### Checklist

After finishing these five steps, you should be able to answer the following three questions — and answer them with your own experimental data, not with this chapter's text:

1. What is the cost of compressing below the manifold's intrinsic dimension? Where does the cost appear?

2. How are out-of-distribution points "handled hallucinatively" after passing through the autoencoder?

3. "Where training data is dense, generalization is good; where it is sparse, generalization is poor" — in your experiment, which part of the manifold does the sparse region correspond to? Is the reconstruction error higher there?

If you only do one thing, do Step 4. It's the shortest experiment, and the one that makes you intuit most directly how "hallucination" happens.

<div class="center">

------------------------------------------------------------------------

</div>

## Summary: From Manifolds to Knowledge Structures — The Deep Connections of the First Four Chapters

As we close the first four chapters, let's review the road we've walked, and the deeper connections between what is new in Chapter 1 and Chapter 2.

### Three Levels of Reasoning Systems

The first four chapters actually describe reasoning systems at **three different levels**, which share a **meta-pattern**:

1. **The Physical Level (New in Chapter 1)**: **Living systems** as open systems
   - Achieve local entropy reduction through energy flows
   - Order maintained at the cost of environmental entropy increase
   - Dissipative structures: self-organization far from equilibrium

2. **The Symbolic Level (New in Chapter 2)**: **Knowledge systems** as symbolic structures
   - Exact reasoning through logical rules
   - Interpretability bought at the cost of construction effort
   - Knowledge graphs: symbolic AI implemented at scale

3. **The Statistical Level (Chapters 3–4)**: **Learning systems** as statistical models
   - Pattern recognition learned from data flows
   - Generalization bought at the cost of computational complexity
   - The manifold hypothesis: the low-dimensional structure of high-dimensional data

### A Unified View: Structure, Cost, Flow

These three levels share a **unified framework**:

| Dimension | Physical level (life) | Symbolic level (knowledge) | Statistical level (learning) |
|------|----------------|----------------|----------------|
| **Core mechanism** | Energy flow | Logical reasoning | Pattern recognition |
| **Form of order** | Physical structural order | Symbolic structural order | Representation-space order |
| **Cost of maintenance** | Environmental entropy | Construction cost | Computational complexity |
| **External input** | Energy / matter | Knowledge / rules | Data / samples |
| **Structural assumption** | Dissipative structures | Closed world | Manifold hypothesis |

### From Discrete to Continuous: The Evolution of Representations

The first four chapters also trace the evolution of **knowledge representations**:

1. **Discrete symbols** (Chapter 2): If-Then rules, knowledge graph triples
2. **Continuous vectors** (Chapter 3): word embeddings, semantic spaces
3. **Low-dimensional manifolds** (Chapter 4): intrinsic dimension, the internal structure of data

**Key insight**: a knowledge graph is a **discrete symbolic manifold**, and word vectors are a **continuous semantic manifold**. Both are trying to capture the world's **structure** — just in different mathematical languages.

### The Meta-Question: Reasoning at Multiple Levels

This leads to a **meta-question**: what is reasoning?

- **Physical answer**: reasoning is a cognitive strategy life uses to fight entropy
- **Symbolic answer**: reasoning is a symbolic system's derivation following logical rules
- **Statistical answer**: reasoning is a statistical model's pattern matching on the manifold

These three answers are **all right, but none is complete**. Real reasoning may be the **coordination of all three**:
- The physical level provides the **existential basis** (why reasoning is needed)
- The symbolic level provides the **precise framework** (how reasoning is formalized)
- The statistical level provides the **implementation mechanism** (how reasoning is computed)

## Exercises: Where Are the Boundaries?

Each level has its own **boundary**:
- Physical level: thermodynamic limit (Landauer's principle)
- Symbolic level: knowledge boundary (closed-world assumption)
- Statistical level: generalization boundary (manifold hypothesis)

**The final question**: when an AI system "reasons", which level is it working on? Or does true intelligence require an **integration that transcends these levels**?

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Tishby, N. & Schwartz-Ziv, M. (2017). *Opening the Black Box of Deep Neural Networks via Information* — the deep-learning version of information bottleneck theory, proposing the two-phase training hypothesis `-> [arXiv:1703.00810]`

- Saxe, A. et al. (2018). *On the Information Bottleneck Theory of Deep Learning* — critique and correction of Tishby's theory, showing the compression phase depends on the choice of activation function `-> [arXiv:1812.09881]`

- Fefferman, C., Mitter, S. & Narayanan, H. (2016). *Testing the Manifold Hypothesis* — mathematical tests of the manifold hypothesis: when it holds and when it doesn't `-> [arXiv:1204.1423]`

- Bengio, Y., Courville, A. & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives* — survey of representation learning, covering the relationship between manifold learning and deep representations `-> [arXiv:1206.5538]`

- Bellman, R. (1957). *Dynamic Programming* — the original source of the Curse of Dimensionality; Bellman named the phenomenon in an optimization context

- Chapter 11 of this book, Sections 5–9: the Collins randomized optimizer — information-theoretic proofs and experiments for $O(1)$ state compression and the safe compression ratio $c_{\text{opt}} \approx 34$ (code in Chapter 11's appendix: `collins_adam.py`)

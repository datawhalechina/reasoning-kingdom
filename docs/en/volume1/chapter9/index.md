# Chapter 9: The Transformer — The Attention Revolution of Dynamic Topology

> Every Attention Head asks the same question: right now, which parts matter to which parts?

<div class="center">

------------------------------------------------------------------------

</div>

## 1. A Paper from June 2017

On June 12, 2017, eight authors at Google Brain led by Ashish Vaswani uploaded a paper to arXiv. Its title was "Attention Is All You Need".

That title is arrogant. At the time, the dominant architecture for machine translation was the RNN (recurrent neural network) and the LSTM (long short-term memory network) — they had ruled the field of NLP (natural language processing, the branch of AI that works with human language) for over twenty years. And this paper said: you don't need any of that. Attention is enough.

Even more arrogant: they were right.

Five months later, NIPS 2017 accepted the paper. Two years after that, BERT and GPT-2 arrived — both built on the Transformer. Three years after that, GPT-3 proved the Transformer can scale to 175 billion parameters.

Today, almost every large language model — from ChatGPT to Claude — is a variant of the Transformer.

But back in June 2017, it was just a radical idea: can you throw away the recurrent structure completely?

### 1.1 The Price of Sequential Dependency

An RNN works in an intuitive way: process the words one by one, left to right. At each step, it combines the current word with the "memory" of what came before, and updates the hidden state.

$$
h_t = f(h_{t-1}, x_t)
$$

The design looks natural — after all, people read sentences left to right, too. But it has two fatal flaws:

:::details RNN and LSTM: what is a recurrent neural network?
**RNN (Recurrent Neural Network)**: a neural network that processes a sequence one time step at a time. Each step takes the current input $x_t$ and the previous hidden state $h_{t-1}$, and outputs a new hidden state $h_t$. Because the hidden state "circulates" from step to step, it's called "recurrent".

Intuition: each time you read one word, you update your mental picture of the sentence, then read the next word — an RNN is mimicking exactly that.

**LSTM (Long Short-Term Memory network)**: an improved RNN. It adds "gates" (forget gate, input gate, output gate) that control what to keep and what to forget, which softens the problem of losing information over long distances.

**The vanishing gradient**: while a network trains, the error signal (the gradient) propagates from the output layer back toward the input layer. If that path is long — say, an RNN processing a long sentence — the gradient keeps being multiplied by numbers smaller than 1, and eventually becomes close to zero. Weights far from the output "learn nothing". The LSTM's gates ease this problem but never fully solve it.
:::

**Defect 1: no parallelization**

To compute the hidden state of word 10, you must first compute the hidden state of word 9. And for word 9, you need word 8's… that chain can't be parallelized.

In the GPU era, that's catastrophic. A GPU has thousands of cores and can run thousands of operations at once. But an RNN forces you to compute serially — one word at a time.

For a sequence of length N, the RNN needs N serial steps. Even if each step takes just 1 millisecond, a sentence of 1000 words needs 1 second.

**Defect 2: information decay over long dependencies**

Suppose you want to translate this sentence:

    "The cat, which we found in the garden last summer, was very friendly."

The subject "cat" and the verb "was" are 12 words apart. The RNN has to carry the information about "cat" across 12 steps.

At every step, the information decays. Even with the LSTM's gates, long-distance transport is still hard — this is another face of the **vanishing gradient** problem.

Back in 1994, Bengio and his colleagues proved that when an RNN learns long-range dependencies, its gradients decay exponentially. The LSTM eased the problem, but never fully solved it.

### 1.2 Vaswani's Bet

Vaswani and colleagues' idea was: **if every word could directly "see" every other word, there would be no need to pass information through a hidden state at all**.

That's the self-attention mechanism. It lets every position in the sequence compute its relevance to all the other positions directly, then combine them with a weighted sum using those relevances.

No recurrence, no sequential dependency, completely parallel.

What's the price? **The complexity goes from $O(N)$ to $O(N^2)$**.

:::details Self-Attention: every word asks every other word "do you matter?"
**Self-attention in one sentence**: while processing a sentence, every word "asks" all the other words "how important are you to me?", then combines everyone's information weighted by the answers.

Intuition: take "The bank failed". When processing "failed", self-attention makes "failed" ask "the", "bank" and every other word how much they matter, finds that "bank" is the most relevant, and leans more on "bank" to understand the context of "failed".

**The price of $O(N^2)$**: with a sequence of length $N$, each word must be compared with the other $N$ words, so there are $N \times N = N^2$ comparisons. At $N=1000$ that's a million; at $N=10000$ it's 100 million. This is why long context windows are expensive.

**Why is it worth it?** Because all the $N^2$ computations run at once on the GPU, while the RNN's $N$ steps can only run one after another. The total time is actually shorter — and there's no information decay at all.
:::

For a sequence of length N, self-attention must compute an $N \times N$ attention matrix — the relevance of every word to every other word. At N=512, that's 262,144 numbers. At N=2048, that's 4,194,304 numbers.

This is a bold bet: trade $O(N^2)$ space and computation for parallelization and a **global view** — the ability of every position to see all the others directly.

The result proved that the bet was worth it.

:::details Pause
"The bet was worth it" — but worth it, and why?

Does the Transformer work because self-attention's design is, structurally, closer to language's "real structure" than the RNN? Or simply because $O(N^2)$ compute is exactly what GPU parallelism eats best, letting it cover up architectural flaws with more data and more compute?

In other words: is the Transformer's success because its inductive bias is right, or because its inductive bias is the weakest — so weak that brute-force data can fill it in? (An inductive bias is the set of built-in assumptions an architecture makes about what the data looks like.)

"Having no inductive bias" is itself an inductive bias. The Transformer assumes that "a word at any position might matter to any other position". Is that assumption right for language? Is it right for every task?

Let's put this question aside for now.
:::

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Math of Self-Attention: An Elegant Symmetry

The heart of the Transformer is the self-attention mechanism. The problem it solves is simple:

**given a sequence, how does every position "see" all the other positions and decide which ones matter?**

The math turns out to be strikingly compact. For an input sequence $X \in \mathbb{R}^{N \times d}$ (N words, each d dimensions), compute three matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

The three matrices play different roles:

- **Query**: "what am I looking for?" — the pattern the current position wants to pay attention to

- **Key**: "what can I offer?" — the indexing information that each position provides

- **Value**: "what is my content?" — the actual information held by each position

Then attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Let me break this formula down step by step.

**Step 1: similarity** — $QK^T \in \mathbb{R}^{N \times N}$

Each query is dot-producted with each key. The element at row $i$, column $j$ is $q_i \cdot k_j$ — how interested position $i$ is in position $j$.

Why is a dot product a good similarity measure? Because $q \cdot k = |q||k|\cos\theta$ — when two vectors point in the same direction, the dot product is at its largest.

**Step 2: scaling** — divide by $\sqrt{d_k}$

This is a key technical detail. When $d_k$ is large, dot products vary a lot — they can be very large positive or very large negative. That saturates the softmax: the biggest score gets probability close to 1, and everything else close to 0.

Under the initialisation assumptions we use here, the dimensions of $Q$ and $K$ are independent and have constant variance, so the variance of the dot product $Q \cdot K$ grows in proportion to the dimension $d$. Scaling by $\frac{1}{\sqrt{d_k}}$ normalises that variance back into a stable range around $O(1)$, keeping softmax out of the saturated and the overly flat regions. During training the representation distribution drifts a little, but layer normalization re-centres the input distribution and residual connections steady the gradient flow, so the model adaptively keeps the attention-scores scale in a reasonable range — and training stays stable.

**Step 3: normalization** — softmax (a function that turns a row of scores into a probability distribution summing to 1) normalizes each row

The scores become a probability distribution. The softmax output in row $i$ is a probability vector that sums to 1: it shows how position $i$ distributes its attention across all the other positions.

**Step 4: weighted sum** — multiply by $V$

Use the attention weights as weights in a sum over the values. The output of position $i$ is:

$$
\text{out}_i = \sum_{j=1}^N \text{attention}_{ij} \cdot v_j
$$

This is a **soft addressing** mechanism: instead of picking one position, it takes a weighted average of all of them, with weights decided by similarity.

**What's beautiful about this mechanism**:

1. **Global view**: every position directly "sees" every other position — no passing through a hidden state needed

2. **Fully parallel**: all positions' outputs compute at the same time, no sequential dependency

3. **Dynamic topology**: the attention matrix is generated on the fly from the input content, not a fixed set of weights

<div class="center">

------------------------------------------------------------------------

</div>

## 3. Try It Yourself: See What Attention Is Looking At

Take out a sheet of paper and work through a simple example by hand.

**Input sentence**: "cat sat mat" (3 words, kept simple)

Suppose each word is already embedded as a 2-dimensional vector:

    cat: [1, 0]
    sat: [0, 1]
    mat: [1, 1]

**Step 1: compute Q, K, V**

To keep it simple, say $W_Q = W_K = W_V = I$ (the identity matrix), so $Q = K = V = X$.

**Step 2: compute the attention scores**

$$
QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
$$

Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.41$:

$$
\text{scores} = \begin{bmatrix} 0.71 & 0 & 0.71 \\ 0 & 0.71 & 0.71 \\ 0.71 & 0.71 & 1.41 \end{bmatrix}
$$

**Step 3: apply softmax** (per row)

For the first row $[0.71, 0, 0.71]$:

$$
\text{softmax} = \frac{[e^{0.71}, e^0, e^{0.71}]}{\sum} \approx [0.42, 0.16, 0.42]
$$

The full attention matrix (approximate):

$$
A = \begin{bmatrix} 0.42 & 0.16 & 0.42 \\ 0.16 & 0.42 & 0.42 \\ 0.27 & 0.27 & 0.46 \end{bmatrix}
$$

Reading the rows:

- **Row 1**: "cat" attends to itself (0.42) and to "mat" (0.42) about equally, less to "sat" (0.16)

- **Row 3**: "mat" attends to every word, but most of all to itself (0.46)

**Step 4: weighted sum**

The output of the first word:

$$
\text{out}_1 = 0.42 \times [1,0] + 0.16 \times [0,1] + 0.42 \times [1,1] = [0.84, 0.58]
$$

This vector blends information from "cat" and "mat", because those attention weights are high.

**Key observation**: the attention matrix $A$ is **generated dynamically** — it isn't a fixed set of weights; it's computed from the input content on the spot. That's what "dynamic topology" means.

<div class="center">

------------------------------------------------------------------------

</div>

## Professor Pallas's Monologue: The Dynamic Topology of Hidden Space

---

Textbooks have one fatal weakness.

They say: "attention models the explicit relations between tokens." They say: "self-attention lets every token directly see every other token."

**That's wrong.**

Let me clear up a core misunderstanding — a misunderstanding that has even seeped into papers at top conferences, sending the whole community down the wrong path on a wrong intuition for years.

### A Problem of Spaces: Hidden Space vs. Token Space

People always think attention operates in "token space". But what is "token space"? Is it the lookup table of embeddings? That space only encodes word forms — it's static, and it lives at the vocabulary level.

The real operation happens in **hidden space**.

When the sequence $X$ enters the Transformer, it first becomes:

$$
X_{\text{hidden}} = \text{LayerNorm}(X_{\text{embed}} + \text{PositionalEncoding})
$$

Positional encoding, layer normalization… none of these are decoration. They turn the static embeddings **into context-dependent hidden representations**.

Then attention makes its entrance:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Note: $Q = X_{\text{hidden}}W_Q$, $K = X_{\text{hidden}}W_K$, $V = X_{\text{hidden}}W_V$.

**$Q$, $K$, $V$ are projections of the hidden representation — not projections of the tokens themselves.**

### What Attention Actually Does: Representation Retrieval in Hidden Space

So what is attention doing?

It isn't comparing "token A" with "token B". It's comparing "position A's hidden representation" with "position B's hidden representation".

In more technical terms:

- **Embedding space**: word form -> static vector

- **Hidden space**: context understanding -> dynamic vector

The attention mechanism is doing **representation retrieval** in hidden space.

$q_i$ is a query: "given my current understanding of the context, what information do I need?" $k_j$ is an index: "given my current understanding of the context, what information do I offer?"

The similarity $q_i \cdot k_j$ measures the match between two **context understandings** — not between two **words**.

### RNN's Hard Bottleneck vs. the Transformer's Dynamic Topology

This difference reveals the essence of the two architectures:

**RNN: the hard bottleneck of explicit-memory recursion**

$$
h_t = f(h_{t-1}, x_t)
$$

- $h_t$ is a hidden vector of fixed dimension

- all historical information must be compressed into that low-dimensional space

- this is a **hard bottleneck** — the information is forced into a smaller dimensionality, like pressing a whole book into a one-page summary

- vanishing/exploding gradients are the inevitable price of this one-dimensional chain structure

- **low-entropy structure**: the connection pattern is fixed, the information capacity is constant

**Transformer: dynamic topology in hidden space**

- every hidden representation $h_i$ projects through QKV to $q_i, k_i, v_i$

- attention weights $\alpha_{ij}$ are computed on the fly and form a **dynamic connection graph**

- no fixed-dimensional bottleneck — information capacity grows with sequence length

- **high-entropy structure**: connection weights are generated dynamically, with huge expressive freedom

### Entropy and Representational Freedom

Let me put it in slightly quantitative terms:

- RNN representational freedom: $O(d_{\text{hidden}})$, independent of sequence length

- attention representational freedom: $O(N \times d_{\text{model}})$, grows linearly with sequence length

But attention isn't just "more storage". Its high entropy comes from **dynamic selectivity**.

For each input, it builds, in real time, the topology that best fits the current context. That adaptivity — building connections on demand, inside hidden space — is where the intelligence actually shows.

### The Pyramid Structure of GPT

This also explains why GPT still works after you prune away some of its layers.

A Transformer isn't a flat structure; it's a **pyramid**:

- **shallow hidden spaces**: encode surface syntax and local dependencies

- **middle hidden spaces**: encode phrase structure and medium-distance dependencies

- **deep hidden spaces**: encode semantic roles, logical relations and long-range dependencies

Each layer is a round of "representation refinement" inside hidden space. Pruning the top layers only loses the most abstract understanding — the foundation below is still there.

### What Dynamic Topology Really Means

Now I can define "dynamic topology" more accurately:

**The attention mechanism is a dynamic connection graph built in real time in hidden space, according to the current context. The connections aren't decided by the tokens — they're decided by the hidden representations of those tokens in that specific context.**

The same word ("bank"):

- in a riverside context, its hidden representation encodes "river, bank, nature"

- in a financial context, its hidden representation encodes "finance, money, business"

So its pattern of attention to other words is completely different.

That's the charm of dynamic topology — **connections are context-dependent, content-driven, and evolve in real time**.

It isn't "all tokens see all other tokens equally". It's "under the current hidden representations, which positions' understandings are most relevant to this position's understanding".

### A Graph-Theory Clarification: Seeing ≠ Reaching

But there's an even deeper misconception, especially common in textbooks.

Those books love to explain attention with "two children becoming friends": every child (token) can see every other child, then decides who to hold hands with. Vivid — but it skips the most important link.

**Seeing is not the same as reaching, and not the same as passing a message.**

In graph-theory terms: treat one layer of the Transformer as a complete graph — every node (position) has an edge pointing at every other node. The edge stands for "attention": I can see your hidden representation, I can take information from you.

But the edge's "bandwidth" is limited. One attention operation is only a shallow integration of information. If you want a deep exchange with a distant node — merging its semantics into your syntax, then fusing that with a third node's pragmatics — you need multi-step propagation.

In a Transformer, one step is one layer.

So:

- **one-layer Transformer**: every position can see every other position, but only one shallow integration is possible

- **two-layer Transformer**: information propagates two steps — A takes from B, integrates, then takes from C

- **L-layer Transformer**: information can walk L steps through the graph, achieving deep, multi-hop exchange

This is why **depth** matters so much.

At the same width ($d_{\text{model}}$):

- shallow networks: short exchange paths, only local or shallow pattern matching

- deep networks: information can "travel" farther through hidden space, integrating across many levels of abstraction

**The length of information exchange ≈ the depth of the network.**

This isn't speculation on my part. It's a basic fact of graph theory: in a graph $G$, the **path length** from node $u$ to node $v$ decides how many "hops" information can make from $u$ to $v$. Attention hands you the edges of the complete graph (reachable in one step), but real semantic integration needs multi-hop paths.

For example:

- Layer 1: learns part-of-speech labels (noun/verb)

- Layer 2: learns phrase structure (noun phrase/verb phrase)

- Layer 3: learns syntactic roles (subject/object)

- Layer 4: learns semantic relations (agent/patient)

- Layer 5: learns pragmatic intent (statement/question/command)

Each layer performs deeper information exchange on top of the previous one. Without enough depth, information can't "travel" to abstract enough levels.

So when you hear "the Transformer lets every token see every token", remember:

- it describes a **one-layer** property

- real **understanding** needs the deep propagation path formed by stacking many layers

- that's why GPT-3 has 96 layers, not 1

**Seeing is free. Reaching costs layers.**

### Back to That One Classic Question

Now we can answer the question we left hanging:

**Is the Transformer's success because its inductive bias is right?**

The answer is: partly.

Its inductive bias isn't "all tokens might be relevant to each other". It's something deeper:

**"In the right hidden space, context-dependent representations can be selectively integrated through dynamic topology."**

This bias is closer to the real structure of language than the RNN's "a sequence is a Markov chain", yet it isn't so weak that it has no bias at all.

It lands exactly on the sweet spot between "flexible enough to capture the complexity of language" and "biased enough to learn efficiently".

---

"Okay, now we can go back to multi-head attention."

<div class="center">

------------------------------------------------------------------------

</div>

## Professor Pallas's Second Monologue: The S×S Matrix — a Search Engine in Hidden Space

Before we continue, one deeper structure still needs to be said clearly. The previous section said attention doesn't operate in token space but in hidden space. But that matrix in hidden space — $QK^T/\sqrt{d_k}$ — what is it, exactly?

**It isn't $N \times N$. It's $S \times S$.**

$S$ is the hidden-state space — the $d$-dimensional manifold shaped over and over by layer normalization, positional encoding and residual connections. The $N$ tokens are just $N$ sample points on this manifold. $QK^T$ isn't computing relations between tokens — it's computing inner products between any two points on the hidden-state manifold. Swap "cat" for "feline": $N$ stays the same, but the hidden states change — the content of the $S \times S$ matrix is completely rebuilt. Switch the language from English to French: $N$ may go from 6 to 7, but the hidden representations of the "cat/sat/mat" material keep roughly the same positions in $S$ — the structure of $S \times S$ is approximately unchanged even though the size of $N \times N$ changed.

**$N$ is appearance; $S$ is essence.** Tokens just sample $N$ points from the manifold $S$. Reading the attention matrix as "lines between tokens" replaces the manifold itself with its sample points.

### Self-Attention as a Search Engine

An information retrieval system has three core components: the query ("what am I looking for"), the index ("what does each document offer"), and the content ("the actual information"). The retrieval process: match queries against indices -> rank -> extract the content of the most relevant documents.

Self-Attention's three projection matrices do exactly the same thing:

- $W_Q$: projects hidden states into query vectors — "in this context, what information do I need?"

- $W_K$: projects hidden states into key vectors — "in this context, what information labels do I offer?"

- $W_V$: projects hidden states into content vectors — "what is my actual information content?"

$QK^T$ computes query-index similarity -> softmax normalizes it into relevance weights -> $\times V$ extracts the content. This isn't a metaphor. Mathematically, Self-Attention is an information retrieval system. The only difference: in classical IR the queries/indices/content are written by humans; in Self-Attention they are learned from data.

Each Transformer layer performs one step of state retrieval in hidden-state space: the current hidden state $H$ is the "information store", and Self-Attention runs a full-store search over these $N$ states — every state issues a query (given the current understanding, what's still needed), offers an index (given the current understanding, what's available), the system scores every query-index pair (the $S \times S$ matrix), and every state pulls information from the most relevant other states (a weighted average of $V$). $L$ layers = $L$ steps of hidden-state-space search.

### Compared with Explicit Search

| | A\* | MCTS | Transformer |
|:--|:---|:-----|:-----------|
| search space | discrete graph | discrete tree | continuous manifold $S$ |
| search operator | $f(n) = g(n) + h(n)$ | UCB + simulation | $QK^T/\sqrt{d_k}$ |
| information integration | accumulated along the path | back-propagated along the tree | weighted average (soft) |
| topology | fixed | fixed | dynamically reshaped (different per layer) |
| guarantee | optimality (admissible $h$) | logarithmic convergence | no theoretical guarantee |

The Transformer's search shares the same skeleton as A\* and MCTS — it hunts for relevant information in a space, makes a "which way to go" decision at every step, and works under a computation budget — but it replaces the hard selection of discrete space with soft weighted averaging in continuous space. This is both its biggest advantage (differentiable, parallelizable, trainable end to end) and its biggest weakness (no theoretical guarantees, prone to dilution by noise, constrained by the Yonglin limit).

### Softmax Temperature and the ADS Barrier: Entropy Control in Space and Time

There's a deep connection here.

Self-Attention's $1/\sqrt{d_k}$ is a **global temperature parameter** — it controls how sharp the attention is along the spatial dimension. Lower temperature: attention concentrates on a few highly relevant states (hard selection). Higher temperature: attention spreads out evenly (no selection at all).

ADS's barrier $\alpha(B_s) = -\log(1 - B_s)$ controls search behavior along the temporal dimension — high entropy shrinks the step ("not sure, go slowly"), low entropy extends it ("sure, take a big step").

Both mechanisms use **entropy to tune the "sharpness" of a search**:

| Dimension | Mechanism | Controlled quantity | Effect |
|:-----|:-----|:-------|:-----|
| Space ($S \times S$ matrix) | softmax temperature $\tau = \sqrt{d_k}$ | entropy of the attention distribution | low $\tau$ -> focus only on the most relevant state |
| Time (search step size) | ADS barrier $\alpha(B_s)$ | entropy of the belief distribution | low entropy -> big steps, high entropy -> brake |

Taken together, the two can each do their own job inside a single search system: control "how wide to look" in the spatial dimension, and "how fast to move" in the temporal dimension. The Transformer's $1/\sqrt{d_k}$ is a fixed global temperature; ADS's $\alpha(B_s)$ is a dynamic local barrier. A natural guess: could the softmax temperature also vary dynamically with $B_s$ — forcing focus in uncertain contexts, and allowing broad integration in certain ones? Today this is still an open question — if ever realized, it would transplant the ADS mechanism into the Transformer, rather than leaving it as a conceptual parallel.

### Multi-Layer S×S and the Yonglin Limit

Each Transformer layer has its own $W_Q^\ell$, $W_K^\ell$, $W_V^\ell$. This means each layer's S×S matrix is computed in a different projected subspace — layer 1 measures relevance in a low-level syntactic subspace, layer 6 in a middle-level semantic subspace, layer 12 in a high-level pragmatic subspace. Every layer searches a different $S_\ell \times S_\ell$, and $S_\ell$ itself keeps evolving (residual connections constantly refresh the hidden states). This is a search whose search space itself evolves.

If we treat CoT as an outer loop stacked on top of the Transformer's S×S search — each generated token runs the hidden state through $L$ S×S searches, then projects it onto the simplex with softmax — then Yonglin's formula says:

$$
\lim_{t \to \infty} \left(\text{S×S-Search}^L \circ \text{softmax}\right)^t = A
$$

The S×S matrix gives the Transformer search power in hidden-state space. But that power is constrained by the statistical structure of the training data — the S×S search happens in subspaces defined by the trained weights, and those weights encode $A$. Every step of the S×S search pulls the hidden state a little closer to $A$. No matter how large $L$ is, no matter how refined the S×S search, the limit of the outer loop is always $A$ — it can search in the neighborhood of $A$, but it cannot escape $A$'s basin of attraction.

### The Unified Lineage of Search in This Book

| Form | Space | Operator | Topology | Guarantee | Chapter |
|:-----|:-----|:-----|:-----|:-----|:-----|
| Explicit discrete search | discrete graph | $f(n) = g(n) + h(n)$ | fixed | optimality (admissible) | Chapter 8 |
| Explicit stochastic search | discrete tree | UCB + simulation | fixed | logarithmic convergence | Chapter 10 |
| Topology-reshaping search | dynamic manifold | $\alpha(B_s)$ barrier | dynamic | locally admissible | Chapters 8, 10, 12 |
| Implicit continuous search | hidden-state manifold $S$ | $QK^T/\sqrt{d_k}$ | dynamic (per layer) | none | Chapter 9 |

These aren't four isolated categories. This is one continuous lineage: from discrete to continuous, from fixed topology to dynamic topology, from selection functions to geometric operators. Four forms, one question: find a path to the goal under uncertainty. The Transformer's S×S matrix is the ultimate form on the continuous end — it doesn't merely search over a continuous space; it continuously redefines the geometry of the search space itself.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. Multi-Head Attention: Division of Labor and Cooperation

A single attention head can only capture one relationship pattern. But the relations in language are diverse:

- **Syntactic relations**: subject-verb, modifier-head noun

- **Semantic relations**: pronoun-antecedent, synonyms

- **Positional relations**: neighboring words, long-distance dependencies

The Transformer solves this with multi-head attention: run several attention heads in parallel, each head learning a different relationship pattern.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Intuition**: split the $d$-dimensional space into $h$ subspaces of $d/h$ dimensions each. Each head does attention in its own subspace, then the results are concatenated.

Observations from experiments (visualization studies of BERT):

- **Head 1**: attends to adjacent words (local syntax)

- **Head 2**: attends to pronouns and their antecedents (coreference resolution)

- **Head 3**: attends to the start and end of the sentence (global structure)

- **Head 4**: attends to punctuation (syntactic boundaries)

Different heads spontaneously learn different linguistic functions, even though nobody tells them to.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. The Price of O(N²): The Thermodynamic Bill for a Global View

Self-Attention's time complexity is $O(N^2d)$, and its space complexity is $O(N^2)$.

This isn't a constant factor you can optimize away. It's the **essential cost of the architecture**.

### 5.1 Why N²?

Computing $QK^T$ requires $N \times N$ dot products, each needing $d$ multiplications: $O(N^2d)$ operations in total.

Storing the attention matrix requires $N \times N$ floating-point numbers. At $N=512$, that's 262,144 numbers. At $N=2048$, that's 4,194,304 numbers.

Compare with other architectures:

- **RNN**: $O(Nd^2)$ — linear in sequence length, but sequential

- **CNN**: $O(Nkd^2)$ — linear in sequence length, but the receptive field is limited to the kernel size $k$

- **Transformer**: $O(N^2d)$ — global view, but quadratic complexity

### 5.2 The Information-Theory View: The Entropy Cost of Full Connectivity

From an information-theory viewpoint, the $O(N^2)$ complexity is the inevitable result of a **fully connected topology**.

The RNN is a **chain topology**: every node only connects to the previous one. The information flow is one-way, left to right. This is a **low-entropy structure** — the connection pattern is deterministic and doesn't need to be stored.

The Transformer is a **fully connected topology**: every node connects to every other node. This is a **high-entropy structure** — the connection weights (the attention matrix) are dynamic and must be stored.

Shannon's information theory tells us: describing the connection weights of a fully connected graph with $N$ nodes requires $\Omega(N^2)$ bits of information — because there are $N^2$ edges.

This isn't an implementation problem. This is an **information-theoretic lower bound**.

### 5.3 The Physics View: The Constraint of Landauer's Principle

Landauer's principle says: erasing 1 bit of information costs at least $k_B T \ln 2$ of energy.

In every forward pass, the Transformer computes and stores $N^2$ attention weights. At the next token, those weights are "erased" (recomputed).

This means: processing a sequence of length $N$ costs the Transformer at least $O(N^2 k_B T \ln 2)$ of energy — this is a **thermodynamic lower bound**.

At $N=2048$, room temperature $T=300K$, that's on the order of $10^{-18}$ joules. It looks tiny — but when you process billions of tokens, it accumulates into a significant energy bill.

GPT-3's training consumed about 1287 MWh of electricity. A fair chunk of that went into computing and storing attention matrices.

### 5.4 Is the Price Worth It?

For sequences shorter than 2048, the answer is yes. The Transformer beats the RNN across the board on machine translation, text generation, question answering and more.

But for long sequences (document-level translation, long-text summarization, genome sequence analysis), $O(N^2)$ becomes the bottleneck.

This gave rise to a line of "efficient Transformer" research:

- **Sparse Attention**: compute only part of the attention, e.g. a local window plus global tokens

- **Linear Attention**: use the kernel trick to reduce complexity to $O(Nd^2)$

- **Reformer**: approximate attention with locality-sensitive hashing

- **Longformer**: mix local windows and global attention

But all these methods have a cost: either sacrificing the global view, or introducing approximation error.

There's no free lunch: a global view and linear complexity — you can only pick one. This isn't an engineering limit. It's a basic constraint of information theory and thermodynamics.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. Linear Attention: Trading Softmax for the Associative Law

The root of $O(N^2)$ is $QK^T$ — that matrix has to be fully computed before softmax can normalize each row.

So what if we drop the softmax?

### 6.1 The Kernel View

Recall standard attention:

$$
\text{out}_i = \frac{\sum_j \exp(q_i \cdot k_j) \cdot v_j}{\sum_j \exp(q_i \cdot k_j)}
$$

The $\exp(q \cdot k)$ inside softmax can be seen as a **kernel function** $\kappa(q, k)$, measuring the similarity between a query and a key.

There's a general approximation scheme for kernel functions: **feature maps**. If there exists a function $\phi(\cdot)$ such that:

$$
\kappa(q, k) \approx \phi(q)^T \phi(k)
$$

then attention can be rewritten as:

$$
\text{out}_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) \cdot v_j}{\sum_j \phi(q_i)^T \phi(k_j)}
$$

**Key move: use the associativity of addition and of matrix multiplication** — swap the parentheses:

$$
\text{out}_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)}{\phi(q_i)^T \left(\sum_j \phi(k_j)\right)}
$$

Now $\sum_j \phi(k_j) v_j^T$ and $\sum_j \phi(k_j)$ can be computed once, in advance, and stored as two fixed matrices. Computing them costs $O(Nd^2)$ — linear in sequence length.

All $N$ queries share these two matrices, so each query costs one matrix-vector multiply, $O(d^2)$.

Total complexity: $O(Nd^2)$ — **linear attention is born**.

### 6.2 Linear Attention's Fatal Flaw: Slipperiness

This sounds perfect. But linear attention has consistently underperformed standard attention in practice — especially when precise, sharp attention matters (for example coreference resolution, keyword matching).

The reason: **softmax naturally produces sharp distributions; the linear kernel can't.**

Softmax's exponential function greatly amplifies large scores and suppresses small ones. The result is highly concentrated attention weights — one position may grab 0.95 of the weight, with everyone else left with 0.05.

The linear kernel $\phi(q)^T\phi(k)$ has no such "sharpening" effect. All positions get more evenly spread weights — everyone gets a say.

That looks "fairer", but that's exactly the problem.

A metaphor: you're at a noisy party trying to hear one person (the signal). Softmax attention lets you focus 95% of your attention on that person, with 5% on everyone else (noise). Linear attention spreads attention evenly over everyone — you can still hear that person, but the noise gets amplified just as much, and the signal gets drowned.

**Linear attention is too smooth — it accumulates noise and signal on equal terms, diluting the sharpness.**

More formally: linear attention computes a single linear read of the Key-Value outer-product matrix $\sum_j \phi(k_j)v_j^T$. This matrix presses all historical information into a fixed-size matrix ($d \times d$ dimensions), like a blackboard with finite capacity. As the sequence grows, new information keeps piling on, and old information gets overwritten.

That's the essence of linear attention: it's actually a **finite-capacity recurrent state**, not a true global view.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. State Space Models: From Linear Attention to Selective Memory

The analysis of linear attention exposes a deeper question: **can we have a global view and linear complexity at the same time?**

The answer is: no — unless you bring in a new dimension: **selectivity**.

That's the core idea behind state space models (SSMs) and Mamba.

### 7.1 Seeing the Shadow of SSM in Linear Attention

The recurring form of linear attention:

$$
S_t = S_{t-1} + \phi(k_t)v_t^T, \quad \text{out}_t = \phi(q_t)^T S_t
$$

where $S_t \in \mathbb{R}^{d \times d}$ is an accumulated "state matrix".

This recurrence is almost identical to the classical **linear state space model** (Linear SSM):

$$
h_t = A h_{t-1} + B x_t, \quad y_t = C h_t
$$

$h_t$ is the hidden state, $A$ the state-transition matrix, $B$ the input projection, $C$ the output projection.

Linear attention's $S_t$ is an SSM with $A = I$ (the identity matrix) — it only knows "accumulate", not "forget", so old information stays in the state forever, causing overwriting and dilution.

**The SSM improvement**: let $A$ differ from the identity matrix, and instead be a matrix that can selectively "decay" old information.

### 7.2 HiPPO: Compressing History with Orthogonal Polynomials

In 2020, Albert Gu and colleagues proposed the HiPPO (High-order Polynomial Projection Operator) framework, which gave a theoretical construction of the $A$ matrix:

$$
A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}
$$

Intuition: HiPPO-$A$ makes the hidden state $h_t$ the **optimal projection** of the historical inputs $x_{1:t}$ onto a basis of Legendre polynomials. It's like approximating a function with the first few terms of a Fourier series — with a limited set of coefficients, you capture the "main components" of the history.

This solves the overwriting problem of linear attention: old information isn't accumulated equally, but compressed by importance into a low-dimensional "summary of history".

### 7.3 Mamba: Selective State Space Models

In 2023, Albert Gu and Tri Dao proposed Mamba. Its core innovation is the **selective mechanism** (Selective SSM, S6).

In a standard SSM, the $A, B, C$ matrices are fixed and independent of the input. That means the model treats every token the same way — no matter how important or irrelevant it is.

Mamba makes $B, C$ and the step size $\Delta$ input-dependent:

$$
B_t = B(x_t), \quad C_t = C(x_t), \quad \Delta_t = \Delta(x_t)
$$

The discretized state transition:

$$
\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t
$$

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t
$$

The point of selectivity:

- $\Delta_t$ large: $\bar{A}_t \approx 0$, the state decays sharply — the model **actively forgets** old information and focuses on the current input

- $\Delta_t$ small: $\bar{A}_t \approx I$, the state barely changes — the model **actively remembers** old information and ignores the current input

This isn't a fixed forgetting curve. It's **content-driven selective memory**. For important tokens, the model lowers the decay rate so their information lingers in the state; for noisy tokens, it forgets faster.

### 7.4 Why Selectivity Solves Linear Attention's Problem

Back to the "slipperiness" problem: linear attention accumulates all information into the state equally, so the signal is diluted by noise.

Mamba solves it with the selective $\Delta_t$:

- sees important information -> $\Delta_t$ large -> state updates hard, old noise gets cleared out

- sees noise -> $\Delta_t$ small -> state barely moves, noise ignored

**This is the counterpart of softmax's "sharpness" in a recurrent architecture**: instead of concentrating attention along the spatial dimension, it selectively remembers along the temporal dimension.

### 7.5 The Cost: Locality Comes Back

Mamba is $O(N)$ in sequence length, parallelizable at training time (via a parallel scan algorithm), and recurrent at inference time (constant $O(d)$ per step).

But it has a cost: **selectivity is local**.

$\Delta_t$ only depends on the current token $x_t$; it can't directly see tokens elsewhere in the sequence. This means Mamba cannot do the Transformer's "leapfrog" long-range matching — its long-range memory is built by progressively compressing history, not by direct retrieval.

For tasks that need exact matching of distant tokens (say, finding a specific word that appeared 1000 tokens ago), Mamba is weaker than the Transformer.

This is unavoidable: **you cannot have both direct global access and linear complexity**. Mamba pays the linear price to buy "indirect access via compressed history".

### 7.6 Transformer, Linear Attention, Mamba: A Three-Way Tradeoff

| Architecture | Time complexity | Space complexity | Long-range ability | Exact-matching ability |
|:--|:--|:--|:--|:--|
| Transformer | $O(N^2d)$ | $O(N^2)$ | direct global access | strong (sharp softmax) |
| Linear attention | $O(Nd^2)$ | $O(d^2)$ | finite-capacity state | weak (smooth, noise-diluted) |
| Mamba (SSM) | $O(Nd)$ | $O(d)$ | compressed-history memory | medium (selective memory) |

There's no free lunch. Global view, linear complexity, exact matching — you can only have at most two of the three at once.

This is a basic constraint of information theory, not an engineering problem.

<div class="center">

------------------------------------------------------------------------

</div>

## 8. Positional Encoding: How the Transformer Understands "Order"

Self-Attention has a strange property: **it's insensitive to word order**.

Shuffle "cat sat mat" into "mat cat sat", and the attention matrix changes (because the row/column order of $QK^T$ changes), but the computation itself is exactly the same. The Transformer has no concept of "the 1st word" or "the 2nd word".

But language is ordered. "cat sat mat" and "mat sat cat" mean different things.

The solution: **positional encoding**. Add position information into the input embeddings:

$$
\text{input} = \text{word}_{\text{embedding}} + \text{position}_{\text{encoding}}
$$

The original Transformer used sinusoidal positional encoding:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

where $pos$ is the position and $i$ the dimension index.

Why sine functions? Because they have an elegant property:

$$
PE_{pos+k} = f(PE_{pos})
$$

Namely: a relative position can be expressed as a linear function of the absolute position. That makes it much easier for the model to learn "relative position relations".

Sinusoidal positional encoding is an absolute-position encoding: it's deterministic and has no learnable parameters, so it adds no extra learning noise. But because it's injected by adding to the token embedding, the content information and the absolute position information inside each token representation are tightly coupled — hard to separate explicitly in the attention computation. Its limitation: it never explicitly models relative positions between tokens; the model has to derive those relations indirectly from absolute positions — an added learning burden.

What's more, this implicit modeling can also hurt the model's long-distance dependence capture and its extrapolation to unseen sequence lengths.

Later work proposed relative positional encodings:

- **RoPE** (rotary position embedding): encodes relative position directly inside the attention computation

- **ALiBi**: adds a position bias to the attention scores — the farther, the lower the weight

These extrapolate better to long sequences — train at length 512, test at length 2048.

<div class="center">

------------------------------------------------------------------------

</div>

## 9. Pseudocode: The Transformer's Complete Forward Pass

Let me formalize the Transformer's core algorithms.

**Algorithm 1: the Self-Attention layer**

```python
import torch
import torch.nn.functional as F

def self_attention(X, W_Q, W_K, W_V):
    # X: (N, d_model), W_Q/W_K/W_V: (d_model, d_k)
    Q = X @ W_Q          # (N, d_k)
    K = X @ W_K          # (N, d_k)
    V = X @ W_V          # (N, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.T / d_k ** 0.5              # (N, N)
    attn   = F.softmax(scores, dim=-1)         # normalize each row
    return attn @ V                            # (N, d_v)
# Time complexity O(N²·d), space complexity O(N²)
```

**Algorithm 2: multi-head attention**

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h  = num_heads
        self.dk = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        B, N, D = X.shape
        # project and split into multiple heads (B, h, N, dk)
        Q = self.W_Q(X).view(B, N, self.h, self.dk).transpose(1, 2)
        K = self.W_K(X).view(B, N, self.h, self.dk).transpose(1, 2)
        V = self.W_V(X).view(B, N, self.h, self.dk).transpose(1, 2)
        attn = F.softmax(Q @ K.transpose(-2, -1) / self.dk**0.5, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, N, D)  # concatenate the heads
        return self.W_O(out)
```

**Algorithm 3: a complete Transformer block**

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        X = self.norm1(X + self.attn(X))                         # attention + residual + LN
        X = self.norm2(X + self.ff2(F.relu(self.ff1(X))))        # FFN + residual + LN
        return X
# A full Transformer = stacking multiple TransformerBlocks
```

<div class="center">

------------------------------------------------------------------------

</div>

## 10. The Philosophy of Attention: Understanding or Retrieval?

The Transformer's success raises a deep question: **is attention "understanding" relations, or "retrieving" relevant information?**

Consider this sentence:

    "The trophy doesn't fit in the suitcase because it is too big."

Humans know "it" refers to "trophy" (not "suitcase"), because we understand the logic of "too big" — the trophy is too big, so it doesn't fit.

Transformer also correctly makes "it"'s attention concentrate on "trophy". But did it really understand the causal logic, or did it just learn a statistical pattern: "it" usually refers to the previous noun, and "it" near "too big" is more likely to refer to the first noun?

There's no simple answer. But there are clues:

**Evidence 1: the interpretability of attention patterns**

Researchers found that certain attention heads do learn linguistically meaningful patterns: subject-verb agreement, modifier-head relations, pronoun-antecedent binding.

These patterns aren't random. They're systematic.

**Evidence 2: fragility under distribution shift**

But when the test data's distribution differs from the training data, these patterns collapse. If during training "it" always referred to the first noun, then at test time a case where "it" refers to the second noun makes the model fail.

This suggests that what the Transformer learned may be **statistical shortcuts**, not deep causal understanding.

**Evidence 3: attention ≠ explanation**

More subtle: the attention weights themselves may not be the real cause of the model's decision. A 2019 study showed that even if you randomly shuffle the attention weights, the model's output sometimes barely changes — because the real information may be encoded in the value vectors, not in the attention distribution.

**Attention is a useful tool, but it is not a sufficient condition for understanding.**

<div class="center">

------------------------------------------------------------------------

</div>

## 11. Visualization: What the Attention Heads Are Looking At

<figure>
<p><img src="/figures/ch09_fig1_attention_heads.png" alt="Figure 9.1: The division of labor among multi-head attention heads" /></p>
<figcaption>Figure 9.1: The division of labor among multi-head attention heads</figcaption>
</figure>

The Transformer defines the infrastructure of reasoning. In the next chapter, we'll see how it combines with search — finding the best reasoning path through state space.

<div class="center">

------------------------------------------------------------------------

</div>

## Open Questions

- **Is attention really "understanding" relations?**

  Or is it just doing complex pattern matching? Consider this test: if you randomly shuffle the attention weights but keep the value vectors unchanged, how much does the model's output change? Jain & Wallace (2019) found: on some tasks, shuffling the attention weights has little effect on the output. What does that say? Do the attention weights not matter, or is the real information encoded in the value vectors?

- **Can the O(N²) cost be fundamentally broken?**

  All "efficient Transformers" either sacrifice the global view or introduce approximation. But is that inevitable? The information-theoretic lower bound says: describing a fully connected graph of $N$ nodes needs $\Omega(N^2)$ bits. But maybe we don't need to store the full attention matrix — maybe there exists an implicit representation that encodes the $N^2$ connection pattern in $O(N)$ space?

- **What is the essence of positional encoding?**

  Why do sine functions work? Why does RoPE (rotary position embedding) extrapolate better to long sequences? What is the optimal encoding of position information? This question is equivalent to: how can you inject order information while keeping permutation invariance?

- **How many heads should multi-head attention have?**

  BERT uses 12 heads, GPT-3 uses 96. How was this number chosen? Is there a theory of the optimal head count? One possible angle: each head learns the relation pattern of one subspace, so the head count should equal the "intrinsic dimension of language relations". But what is that dimension?

- **Can attention weights serve as an interpretability tool?**

  If attention isn't explanation (the argument of Jain & Wallace), how should we understand the model's decision process? A deeper question: does interpretability require causality? That is, what we need isn't "what the model is looking at", but "if I change the input, how does the model's output change"?

- **What is the Transformer's inductive bias?**

  CNNs have a locality bias (the convolution kernel), RNNs have a sequential bias (the recurrent structure). What's the Transformer's inductive bias? Is "having no inductive bias" itself an inductive bias? This question touches the philosophy of architecture as prior: every architecture assumes some structure in the observed data — some assumptions just hide deeper than others.

- **Can the S×S matrix have theoretical guarantees?** (see Professor Pallas's Second Monologue)

  A\* has an admissibility guarantee, MCTS a logarithmic convergence guarantee, ADS locally admissible step conditions. The Transformer's S×S matrix search has no theoretical guarantee at all — it "usually works", but nobody can say precisely under what conditions it fails, what the failure pattern looks like, or where the failure boundary lies. Is there some constraint on $W_Q$, $W_K$ that would give S×S matrix search something like an "admissibility" guarantee? This is the open problem of search guarantees in continuous hidden space.

- **Can a dynamic temperature transplant ADS into the Transformer?** (see Professor Pallas's Second Monologue)

  ADS uses $\alpha(B_s)$ to control step length dynamically along the temporal dimension (brake when entropy is high, accelerate when it's low), and the Transformer uses a fixed $\tau = \sqrt{d_k}$ to control attention sharpness along the spatial dimension. What if the softmax temperature also varied dynamically with the local entropy $B_s$ — forced focus at high uncertainty (low $\tau$), broad integration at high certainty (high $\tau$) — could the adaptive behavior of ADS be reproduced inside the Transformer? That would be a mechanism swap between the log barrier and the softmax temperature under the common theme of "entropy control".

- **How does the spectral structure of the multi-layer S×S matrices relate to reasoning depth?** (see Professor Pallas's Second Monologue)

  The S×S matrix of layer $\ell$ has its own eigenvalue distribution. Large eigenvalues correspond to dominant reasoning directions, small ones to exploratory directions. By tracking how the spectral structure changes with layer depth, can we predict where a reasoning chain starts to drift off the right track? More specifically: does a sudden drop in effective rank signal "the model has stopped reasoning effectively and is outputting the prior" — an early warning of Yonglin convergence?

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Vaswani et al. (2017). *Attention Is All You Need* — the original Transformer paper, which opened a new era of NLP `-> [arXiv:1706.03762]`

- Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — the kernel derivation of linear attention; the associative law buys $O(N)$ complexity, together with the slipperiness cost `-> [arXiv:2006.16236]`

- Gu, A. et al. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections* — optimal history compression with Legendre polynomials; the theoretical foundation of SSMs `-> [arXiv:2008.07669]`

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — the selective SSM; content-driven $\Delta_t$ realizes softmax's sharpness along the temporal dimension `-> [arXiv:2312.00752]`

- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers* — bidirectional Transformer pretraining, demonstrating the power of the attention mechanism `-> [arXiv:1810.04805]`

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding* — RoPE rotary position embedding, improved long-sequence extrapolation `-> [arXiv:2104.09864]`

- Press et al. (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* — the ALiBi position-bias method

- Clark et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention* — attention visualization and interpretability research

- Tay et al. (2020). *Efficient Transformers: A Survey* — a survey of efficient Transformer architectures

- Jain & Wallace (2019). *Attention is not Explanation* — a critical analysis of attention as an explanation tool

- Wiegreffe & Pinter (2019). *Attention is not not Explanation* — a rebuttal to the above, a defense of attention

# Chapter 3: From Symbols to Vectors — The First Liberation of Representation Space

> Subtract "man" from "king", add "woman", and you get "queen". Is that reasoning — or arithmetic?

<div class="center">

------------------------------------------------------------------------

</div>

## 1. That One Sentence

In 2013, Tomas Mikolov wrote an equation in a paper that went on to be cited more often than just about any NLP paper of its era:

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}
$$

It means: take the word vector for "king", subtract the vector for "man", add the vector for "woman" — and the nearest neighbor is "queen".

This was not a carefully designed example. It fell straight out of an experiment. The algorithm learned the relationship on its own; nobody told it "king is to man as queen is to woman".

At that moment, the AI research community collectively gasped.

Why? Because it looked like the machine was doing **analogical reasoning**. Not rule-based reasoning, not logical deduction, but something closer to human intuition: it grasped the essence of a relationship, then carried that relationship over to new objects.

And then the question arrived: is it really reasoning?

That question is what this chapter is about — what vector representations are, why they are more powerful than symbols, where their power comes from, and where it runs out.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Problem with Symbols, Again

In the last chapter, we saw the core trouble with symbolic AI: knowledge has to be written out explicitly, and there is no end to the knowledge of the world.

But there is another problem I didn't stress back then: **a symbol's meaning is bolted on from the outside**.

In an expert system, the string `"is a mammal"` is just a token to the machine — an atom that cannot be broken down any further. It doesn't know how "mammal" relates to "animal", doesn't know how close "mammal" is to "reptile", doesn't know that "cat" is a more typical mammal than "whale".

All of these relationships have to be written into rules explicitly.

    IF  is a mammal
    THEN is an animal  [R_base_1]

    IF  is a mammal
    AND is not a whale
    THEN usually has legs  [R_mammal_legs]

Every relationship needs its own rule. And relationships are endless, so the rules never run out to write.

The root of the problem: **symbolic systems represent concepts with discrete tokens, while the relationships between concepts are continuous, graded, and nested**. The distance between "cat" and "dog" can't be captured by discrete If-Then rules — you can state "a cat is a mammal, a dog is a mammal", but you can't say "a cat is closer to a dog than a snake is".

Vector representations solved this problem. Not completely, but solved the core of it.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. The Distributional Hypothesis: Meaning Hides in the Neighbors

The idea behind vector representations goes back to a line the linguist J.R. Firth wrote in 1957:

> "You shall know a word by the company it keeps."

This intuition is called the **distributional hypothesis**: a word's meaning is determined by the distribution of contexts it appears in.

Words that appear in similar contexts mean similar things. "Cat" and "dog" both show up often in contexts like "I keep a \_\_\_", "feed it", "it meowed" — so they are semantic neighbors.

The intuition can be mechanized. Build a matrix: rows are words, columns are context words (or documents), and each cell records how often the two co-occur (or some transformation of that count). Then each row is a vector representation of a word. Words with similar meanings end up close to each other in high-dimensional space.

That's the original version of vector representations, called **distributional semantic models** (DSMs), around since the 1990s. It works, but it has two problems: the dimension is too high (up to hundreds of thousands), and the vectors are extremely sparse (most cells are zero).

:::details Distributional semantic models (DSM): Word meanings hidden in a co-occurrence matrix
A **distributional semantic model** is a family of methods that represent word meanings as vectors. The core idea: build a big "word × context" table recording which words appear together with which, then treat each row of the table as that word's vector.

- **High-dimensional**: if the corpus has 100,000 different words, the matrix has 100,000 columns, and each word's vector is 100,000-dimensional — very long
- **Sparse**: most word pairs rarely or never co-occur, so most of the matrix is 0 — extremely wasteful to store

Word2Vec (Mikolov 2013) solved this: instead of building the matrix directly, it trains a neural network to **implicitly** compress that co-occurrence information into dense 100–300 dimensional vectors.
:::

Mikolov's 2013 paper went further.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. Word2Vec: Compression Through Prediction

Word2Vec doesn't build a co-occurrence matrix. It trains a neural network with one job: **given a word, predict its context words** (Skip-gram), or **given the context words, predict the center word** (CBOW).

:::details Skip-gram and CBOW: The two training modes of Word2Vec
Word2Vec has two variants, which differ only in the direction of the prediction task:

- **Skip-gram**: feed in the center word, predict the words around it. Example: in the sentence "I like eating apples", feed in "eat" and predict "like" and "apples". Works well on small datasets and does better on rare words.
- **CBOW (Continuous Bag of Words)**: feed in the surrounding words, predict the center word. Feed in "like" and "apples", predict "eat". Trains faster and suits large datasets.

Both are just means to an end; the real goal is the word vectors (embeddings). After training, the prediction head is thrown away and only the embedding matrix is kept.
:::

It sounds ordinary. But the by-product is the point.

While you train this network, every word gets mapped to a low-dimensional vector (usually 100 to 300 dimensions). For the prediction task to do well, those vectors have to encode something essential about each word — because only by understanding the relationships between words can you predict which words will co-occur.

Once training is done, you throw away the prediction head and keep only the embedding matrix. That's your word vectors.

There is something deep here worth stopping on:

**You never taught the network the relationship between "king" and "queen". You gave it no semantic rules. You just made it predict text sequences. And then, once it had learned to predict, you discovered that relationships like "king − man + woman ≈ queen" had emerged on their own.**

Why? Because in natural language, wherever "king" shows up, "queen" shows up in a parallel way — just in the corresponding feminine contexts. To predict accurately, the network must encode that parallelism into the vectors. "Royalty" is one axis of context; "gender" is another. Those two axes correspond to two directions in vector space. Subtracting the "man" vector and adding the "woman" vector is a shift along the gender axis.

Meaning emerged from the distribution.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. The Shallow Logic in Geometry

Let's look at the analogy equation more carefully.

$$
\vec{A} - \vec{B} + \vec{C} \approx \vec{D}
$$

king − man + woman ≈ queen, paris − france + japan ≈ tokyo, walk − walked + swim ≈ swam (the past-tense relationship is encoded too).

Each analogy corresponds to a **translation** in vector space. The shift from "man" to "woman" and the shift from "king" to "queen" are one and the same vector direction — a movement along the gender dimension.

That's a very clean geometric structure. Semantic relationships become directions.

![image](/figures/ch03_fig4_similarity_heatmap.png)

*Figure 4: Pairwise cosine similarity matrix for 12 words. Green = high similarity, red = low/negative similarity. The white lines mark the boundaries of four semantic clusters (royalty, gender, occupation, animal). Observation: similarities within the same cluster are far higher than across clusters; the animal cluster barely overlaps the other three. That is precisely a visual proof that "semantically similar words sit next to each other in vector space".*

<div class="center">

------------------------------------------------------------------------

</div>

## 6. The Bias in Vectors: The Corpus Is the Worldview

What Word2Vec learns is not the truth about language, but the statistics of a corpus. And a corpus is written by people — it carries human bias, historical bias, cultural bias.

This is not a fringe problem. It is the core cost of the distributional hypothesis.

**The Bolukbasi et al. (2016) experiment**

They found that in the Word2Vec trained on Google News:

$$
\vec{\text{man}} - \vec{\text{woman}} \approx \vec{\text{doctor}} - \vec{\text{nurse}}
$$

"Man is to doctor as woman is to nurse" — this analogy holds in vector space because it statistically holds in the training corpus.

A more direct test:

$$
\vec{\text{father}} - \vec{\text{mother}} + \vec{\text{programmer}} \approx \vec{\text{homemaker}}
$$

Not "female programmer" — "homemaker".

That's not a bug in the algorithm. The algorithm correctly learned the statistics of the corpus. The bug is ours: **we mistake statistical regularities for semantic truth**.

**Where do the biases come from?**

The distributional hypothesis says a word's meaning hides in its neighbors. But if the neighbors' distribution is itself biased —

- In news articles, the neighbors of "CEO" are more often "his decision" than "her decision"
- In historical texts, "scientist" co-occurs more often with male names
- In novels, emotion words are distributed unevenly across genders

Word2Vec faithfully learned all of these. It is a mirror — it reflects the corpus, not the world.

This exposes a fundamental limit of the distributional hypothesis:

Vector spaces learn **descriptive** statistical regularities, while reasoning needs **normative** truth. "'X and Y often co-occur in the corpus' does not equal 'X and Y are causally related in the world'", and it doesn't equal "this co-occurrence is right".

You can debias word vectors (Bolukbasi's paper proposes a linear projection method), but that's a band-aid — it treats the symptom, not the cause. The root problem: **starting from statistical correlation, you can never reach semantic truth**.

This limit still exists in the Transformer and large language model (LLM) era — a large language model is a model trained on vast amounts of text to predict the next word. Bigger corpora, bigger and wider biases — but the underlying logic of the distributional hypothesis hasn't changed.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. The Illusion of Analogies

Let me come back to that equation:

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}
$$

Is this really reasoning?

Let's take it apart.

For the equation to hold, the difference between the "king" and "man" vectors must be approximately the same as the difference between the "queen" and "woman" vectors.

$$
\vec{\text{King}} - \vec{\text{Man}} \approx \vec{\text{Queen}} - \vec{\text{Woman}}
$$

What is that? It's a statement about **matching shift directions**. A geometric regularity — not logical reasoning.

![image](/figures/ch03_fig2_analogy_arithmetic.png)

*Figure 2: A geometric view of the vector analogy. The four words form an approximate parallelogram in 2D PCA space — man→king and woman→queen are two nearly parallel edges, while man→woman and king→queen are another such pair. That's what "the gender shift vector stays consistent across the royal word pair" means. The cosine similarity of King − Man + Woman comes out near 0.995.*

In logic, reasoning needs **rule + facts → conclusion**. "All kings are men" plus "Elizabeth is a queen" gives you "Elizabeth is not a king" — that is reasoning, because there is transitivity, variable binding, and quantifiers in play.

Vector arithmetic has none of that. It only says: these words' relative positions in semantic space display some parallel structure.

More importantly, this equation fails in plenty of cases.

Fournier et al. (2020) systematically analyzed the limits of analogy tests and found that the celebrated "analogy equations" lean heavily on a convenient evaluation scheme — the input words themselves are excluded from the candidate answers, while the nearest neighbor is often one of the input words. After changing the evaluation, a large number of reported "analogy capabilities" drop significantly `-> [Fournier et al., 2020, arXiv:2010.03446]`.

There is an even more direct example. Try this analogy:

    doctor - man + woman ≈ ?

On some training corpora, what you get is not "female doctor", but "nurse".

Because in those corpora, "doctor" co-occurs more often with masculine contexts, and "nurse" more often with feminine ones. What the model learned is not the neutral meaning of the profession "doctor", but its **statistical bias** in that particular corpus.

Vector representations don't capture truth. They capture the statistical distribution of a corpus. If the corpus has biases, the vectors have biases. This is not a bug — it's how the thing works.

![image](/figures/ch03_fig3_analogy_bias.png)

*Figure 3: Left, the royal analogy succeeds — the result vector of King − Man + Woman lands very close to Queen (cosine ≈ 0.995), with a clean parallelogram structure. Right, the occupational analogy fails — Doctor − Man + Woman points not to "female doctor" but to Nurse (cosine ≈ 0.865). The reason: in the training corpus, Doctor leans toward masculine contexts and Nurse toward feminine ones. Same arithmetic, different biased outcomes.*

<div class="center">

------------------------------------------------------------------------

</div>

## 8. The Limit of the Distributional Hypothesis

Let's think harder about where the boundary of the distributional hypothesis lies.

The hypothesis says meaning is determined by contextual distribution. In most cases that's true. But it implies: **if two words have exactly the same contextual distribution in a text, the model will treat them as having the same meaning**.

"I like cats" and "I like dogs" — these two sentences are perfectly symmetric, so "cat" and "dog" look highly similar in distribution. And yes, they are semantically close; the model is right.

But:

"The earth orbits the sun" and "the sun orbits the earth"

If you look only at word distributions, "earth" and "sun" appear in very similar contexts — they both occur often in sentences about astronomy, gravity, and orbital motion. Yet of these two sentences, one is true and one is false.

Distributional semantic models have no concept of **truth value**. They don't know which propositions are true and which are false. What they know is which words appear in which contexts — a statistical regularity, not a causal model of the world.

This is why vector representations, though they perform well on many tasks, haven't solved reasoning. They solve the **representation** problem: representing concepts in a richer, more continuous way so that similar concepts sit close geometrically. But "close" is not "logically related", and "statistical co-occurrence" is not "causal correlation".

Wittgenstein once said something that fits remarkably well here: **the meaning of language is its use**. The distributional hypothesis captures part of that intuition — a word's meaning does show up in its use. But use is statistical. Is understanding also statistical? That is an open question.

<div class="center">

------------------------------------------------------------------------

</div>

## 9. From Word Vectors to Sentences: The Costs Mount Up

Word2Vec gives you **static word vectors** — one fixed vector per word, no matter what context it appears in.

The word "bank" gets the same static vector in "I went to the bank to withdraw money" and in "wildflowers grew all over the river bank". That is one of its core limitations.

ELMo (2018) and later BERT (2018) introduced **contextualized word vectors** — the same word gets different vectors in different contexts. A major step forward. But it brought another cost: you no longer have a tidy, offline, look-up table of word vectors; you must run the whole network every time.

:::details ELMo and BERT: What are contextualized word vectors?
**Static word vectors** (like Word2Vec): one vector per word, wherever it appears. Whether "apple" means the fruit or Apple the company, the vector is the same.

**Contextualized word vectors** (ELMo, BERT): the same word gets a different vector in different sentences, because the whole sentence goes into the network, and each word's representation is shaped by its context.

- **ELMo** (2018, Allen AI): uses a bidirectional LSTM to produce context-sensitive word vectors, serving them as features to downstream tasks
- **BERT** (2018, Google): pretrains a bidirectional language model with a Transformer Encoder, then fine-tunes it on downstream tasks

The cost: no more look-up; every inference must run the entire network, greatly increasing the compute. That's the eternal trade-off between accuracy and efficiency.
:::

The deeper problem: **the meaning of a sentence is not the simple combination of its word vectors**.

"Cat bites dog" and "dog bites cat" — average the word vectors, and you get almost identical sentence vectors, because the same three words were used. But these sentences describe opposite events.

Word meanings can be encoded in vectors, but sentence meaning depends on **structure** — who is the subject, who is the object, and the argument roles of the verb. Pure vector arithmetic captures none of that structure.

This problem pushed toward the invention of the Transformer — using attention to dynamically handle word-to-word relations rather than simply stacking them. But that's the story of Chapter 9.

<div class="center">

------------------------------------------------------------------------

</div>

## 10. A Small Pause

Let me take stock.

From symbols to vectors: a genuine paradigm shift. It solved one of symbolic AI's core problems: **the representation of meaning**. Concepts are no longer meaningless atoms; they now have positions, distances, and directions in vector space. Similar concepts draw close, and relationships are encoded as directional shifts.

That delivered real capabilities: semantic search, analogy reasoning (of a sort), semantic clustering, and the foundation for transfer learning.

But it did not solve reasoning — for three reasons:

First, **it captures statistical regularities, not causal structure**. The corpus has biases, so the vectors have biases. The corpus has correlations, so the vectors have correlations. But correlation is not causality, and distribution is not truth.

Second, **it has no compositionality**. Simple linear combinations of word vectors cannot capture the structural meaning of a sentence. "Cat bites dog" and "dog bites cat" are identical under a bag-of-words view — a fundamental flaw.

Third, **the analogy equation looks like reasoning, but it is not**. It's a parallel structure in geometry, not a transitive derivation in logic. When it breaks down (say, with socially biased analogies), it quietly hands you a wrong answer, with no warning at all.

Vector representation is one prerequisite for reasoning — but it is not reasoning itself.

<div class="center">

------------------------------------------------------------------------

</div>

## 11. One Deeper Question Still

The deep assumption behind the distributional hypothesis is: **language can carry meaning, and that meaning can be extracted from patterns of language use**.

That's an assumption worth questioning.

John Searle's Chinese Room thought experiment (1980) is the most famous attack in this direction. Picture a person who knows no Chinese shut in a room with a thick rule book: for every Chinese symbol that comes in, it tells him which Chinese symbol to send out. From the outside, he seems to understand Chinese and can answer questions in Chinese — but he has no idea what any of the symbols mean. He is just mechanically executing rules.

:::info
**Professor Pallas says**
This thought experiment is a mirror showing the awkward truth about symbol manipulation: you can imitate understanding perfectly while understanding nothing. What word-vector models do is, in a sense, very much like this Chinese Room — they extract extremely fine-grained statistical patterns from vast text, and those patterns produce correct behavior on many tasks. But do they "understand" the meanings of the words? There is no simple answer, because first we would have to define what "understanding" means.
:::

But there is a more practical test: **distribution shift**. Put the model outside its training distribution and make it handle combinations never seen in the training corpus — what happens? If its performance collapses, you have good reason to suspect it only memorized the distribution instead of truly understanding language.

That is the core question of Chapter 5. For now, just hold onto the cliffhanger.

<div class="center">

------------------------------------------------------------------------

</div>


> Vectors capture similarity, but not causality. In the next chapter, we enter the geometry of high-dimensional space — the manifold hypothesis reveals why data is not randomly distributed.

## Open Questions

- Is the analogy equation `vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)` evidence of reasoning, or a statistical coincidence? It succeeds for some analogies and fails for others — what does that inconsistency tell us?

- Where is the boundary of the distributional hypothesis? Are there meanings that could never be learned from textual distributions? Color? Pain? Bodily sensation?

- Word vectors capture social bias — is that a defect, or a feature? If you delete the bias from word vectors, do you also destroy some real structure of the world?

- What would you get if you trained word2vec on pure noise text? What does this thought experiment reveal?

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Pierce the Illusion of Analogies with Vector Arithmetic

This chapter made a big claim: the analogy equation looks like reasoning, but it's only a parallel geometric structure. You need to verify this distinction with your own hands — not to read it, but to feel the moment it collapses.

You need access to a pretrained word-vector model. The simplest option: load Google's publicly released word2vec model with Python's `gensim` library, or use `fasttext`'s pretrained vectors.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Verify the Famous Equation

Start with the most classic analogy:

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx ?
$$

Implement the logic from Section 5:

```python
import numpy as np

# Assume model is a loaded gensim KeyedVectors object
# Example of loading it:
#   from gensim.models import KeyedVectors
#   model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def analogy_query(model, positive_word, minus_word, plus_word, topn=5):
    """
    Perform a vector analogy query: vec(positive_word) - vec(minus_word) + vec(plus_word)
    model: a gensim KeyedVectors object
    Returns: a list of nearest neighbor words with their cosine similarity scores
    """
    # Compute the target vector: vec("king") - vec("man") + vec("woman")
    target_vector = (model[positive_word]
                     - model[minus_word]
                     + model[plus_word])

    # Find the words in the vocabulary with the highest cosine similarity to the target vector
    # gensim's similar_by_vector can automatically exclude the input words (via the negative parameter)
    results = model.similar_by_vector(
        target_vector,
        topn=topn + 3  # fetch a few extra, so the input words can be filtered out manually
    )

    # Exclude positive_word, minus_word, plus_word themselves
    exclude = {positive_word.lower(), minus_word.lower(), plus_word.lower()}
    filtered = [(word, score) for word, score in results
                if word.lower() not in exclude][:topn]

    return filtered


# ── Run the classic analogy: king - man + woman ≈ ? ───────────────────────
results = analogy_query(model, "king", "man", "woman")

print("Analogy query: king - man + woman ≈ ?")
print("-" * 35)
for rank, (word, score) in enumerate(results, 1):
    print(f"  Rank {rank}: {word:15s}  cosine similarity = {score:.4f}")
```

You should get "queen", with a cosine similarity near 0.7 to 0.9.

**Your first question:** Does this result surprise you? Write down your intuition before continuing: does this look "like reasoning" or "like arithmetic"? Don't read past this point until you've answered.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Test Systematically, Then Look for Counterexamples

Pick 3 pairs from each of the four categories below and test them in turn:

**Category 1: gender** - the man -> woman analogy applied to: king, doctor, professor, nurse, engineer……

**Category 2: capitals** - france -> paris applied to: china, japan, germany, usa……

**Category 3: tense** - the go -> went analogy applied to: run, swim, eat, think……

**Category 4: antonyms** - the good -> bad analogy applied to: big, hot, fast, happy……

For each test case, record: - what you expected - the top 5 nearest neighbors actually returned - the cosine similarity scores

**Your second question:** Which category has the highest success rate? Which the lowest? Can you guess why — is it a problem with the training corpus, or is this kind of relationship not suited to being encoded as a linear shift in the first place?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Reproduce the Occupational Bias Experiment Yourself

Test the failure case described in Section 7:

    doctor - man + woman ≈ ?

Then extend the test: - engineer - man + woman ≈ ? - nurse - woman + man ≈ ? - CEO - man + woman ≈ ? - janitor - woman + man ≈ ?

Record every result.

**Your third question:** For the word pairs whose output is visibly biased, what is your explanation? Are these biases a bug in the word-vector system, or an honest reflection of the true statistical structure of the training corpus? If the latter, what does "debiasing" even mean?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Find the Geometric Conditions of the Analogy Equation

Do a more precise test: when does the analogy equation hold exactly, and when does it fail badly?

Set up the following experiment:

For a set of analogy pairs you choose, compute two quantities:

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def analogy_precision(model, analogies):
    """
    Measure analogy precision in bulk.
    analogies: a list of (A, B, C, D) tuples, each encoding an analogy of the form A:B :: C:D.
    Returns: detailed measurements for each analogy.
    """
    results = []

    for (A, B, C, D) in analogies:
        # Compute the predicted vector: vec(B) - vec(A) + vec(C)
        predicted_vector = model[B] - model[A] + model[C]

        # Get the whole vocabulary ranked by similarity (input words excluded after filtering)
        exclude = {A.lower(), B.lower(), C.lower()}
        neighbors = model.similar_by_vector(predicted_vector, topn=len(model.key_to_index))
        # Filter out the input words to get a ranked word list
        filtered_neighbors = [(word, score) for word, score in neighbors
                               if word.lower() not in exclude]

        # Actual nearest neighbor rank: the position of D in the similarity ranking
        rank_of_D = None
        for idx, (word, score) in enumerate(filtered_neighbors, 1):
            if word.lower() == D.lower():
                rank_of_D = idx
                break

        # Compute offset consistency: the cosine similarity of vec(B) - vec(A) and vec(D) - vec(C)
        # This directly measures whether the two pairs have the same "relationship direction"
        offset_AB = model[B] - model[A]
        offset_CD = model[D] - model[C]
        offset_consistency = cosine_similarity(offset_AB, offset_CD)

        results.append({
            "analogy": f"{A}:{B} :: {C}:{D}",
            "top1_hit": rank_of_D == 1,           # whether D ranks first (exact hit)
            "top5_hit": rank_of_D is not None and rank_of_D <= 5,  # whether D appears in the top 5
            "rank_of_D": rank_of_D,
            "offset_consistency": offset_consistency,  # offset consistency (a geometric measure)
        })

    return results


# ── Example test suite ────────────────────────────────────────────────────
test_analogies = [
    ("man",    "king",    "woman",  "queen"),    # gender x royalty
    ("france", "paris",   "japan",  "tokyo"),    # country x capital
    ("go",     "went",    "run",    "ran"),      # verb tense
    ("good",   "bad",     "big",    "small"),    # antonyms
    ("man",    "doctor",  "woman",  "nurse"),    # occupational bias case
]

precision_results = analogy_precision(model, test_analogies)

print(f"{'Analogy':30s} {'Top-1 hit':^8s} {'Top-5 hit':^8s} {'Rank':^6s} {'Offset consistency':^10s}")
print("-" * 70)
for r in precision_results:
    print(f"{r['analogy']:30s} "
          f"{'✓' if r['top1_hit'] else '✗':^8s} "
          f"{'✓' if r['top5_hit'] else '✗':^8s} "
          f"{str(r['rank_of_D']):^6s} "
          f"{r['offset_consistency']:^10.4f}")
```

**Your fourth question (the core question):** When an analogy fails, is the "offset consistency" score high or low? That is, does the geometric structure (same shift direction) hold, while the nearest-neighbor retrieval still fails? If so, does the failure lie in the geometry, or in the retrieval method? Which claim from Section 7 does this illustrate?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 5: Build an Example Where Cosine Similarity Lies

Cosine similarity measures direction, not length. Find two words whose cosine similarity is very high but that you feel are semantically far apart — or the reverse: low cosine similarity, but you feel they should be semantically close.

```python
def find_counterintuitive_similarities(model, word, topn=20, low_threshold=0.3):
    """
    For a given word W:
    1. Find the topn words with the highest cosine similarity, and look for any that feel out of place
    2. Find words with cosine similarity below low_threshold, and check whether any are semantically related yet geometrically distant
    model: a gensim KeyedVectors object
    """
    print(f"=== Cosine similarity analysis for word '{word}' ===\n")

    # Find the topn words with the highest cosine similarity
    top_neighbors = model.most_similar(word, topn=topn)
    print(f"Top {topn} words by cosine similarity (look for any that feel out of place):")
    for w, score in top_neighbors:
        print(f"  {w:20s}  similarity = {score:.4f}")

    print()

    # Check whether any low-similarity words still seem semantically related to you
    # Method: compute the similarity directly for a set of candidate words
    candidate_words = ["science", "art", "mathematics", "philosophy",
                       "literature", "music", "history", "biology"]
    print(f"Cosine similarity of candidate words (below {low_threshold} but semantically related?):")
    for candidate in candidate_words:
        if candidate in model:
            sim = model.similarity(word, candidate)
            flag = "  <- semantically related but distant?" if sim < low_threshold else ""
            print(f"  {candidate:20s}  similarity = {sim:.4f}{flag}")


# ── Example run ───────────────────────────────────────────────────────────
find_counterintuitive_similarities(model, word="physics", topn=20, low_threshold=0.3)
```

**Your fifth question:** What type of semantic relationship is cosine similarity failing to capture? How does this connect to Section 8, "The Limit of the Distributional Hypothesis"?

<div class="center">

------------------------------------------------------------------------

</div>

### Checklist

When you finish this exercise, you should have three things:

1.  A table: every analogy you tested, success/failure, and your explanation

2.  A bias case: a result you reproduced yourself that made you uncomfortable, and your position on it

3.  A counterintuitive cosine-similarity case: a real gap between vector geometry and linguistic intuition

The third one matters most. That gap is the distance between vector representations and true reasoning.

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space* — the original word2vec paper: the CBOW and Skip-gram architectures, `-> [arXiv:1301.3781]`

- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality* — negative sampling, phrase vectors, and compositionality, `-> [arXiv:1310.4546]`

- Fournier, Dupoux & Dunbar (2020). *Analogies minus analogy test* — a critical analysis of analogy tests, showing the standard evaluation is flawed, `-> [arXiv:2010.03446]`

- Firth, J.R. (1957). *A synopsis of linguistic theory 1930-55* — the philosophical starting point of the distributional hypothesis: "You shall know a word by the company it keeps"

- Searle, J. (1980). *Minds, Brains, and Programs* — the Chinese Room thought experiment, a philosophical attack on whether symbol manipulation can produce understanding

- Bolukbasi et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* — the classic empirical study of bias in word embeddings

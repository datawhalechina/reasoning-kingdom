# Chapter 5: The Trap of Fitting — Statistical Correlation Is Not Reasoning

> After a model has seen a million cats, does it know what a cat is?

<div class="center">

------------------------------------------------------------------------

</div>

## 1. An Unsettling Experiment

In 2021, Bender and colleagues published a paper with a blunt title: “On the Dangers of Stochastic Parrots.”

The title is not a metaphor. It's a diagnosis.

The paper's core argument: a large language model — a model trained on vast amounts of text to predict the next word — no matter how much text it trains on, is fundamentally doing statistical pattern matching. It learns which sequences of words tend to appear together in the training data, and then reproduces those patterns when it generates. Like a parrot that has heard enough conversations to say "hello" or "goodbye" at the right moment, it doesn't understand what those words mean.

The comparison stings, because it touches a deeper question: there is a gap between statistical correlation and causal understanding.

Let me show you a concrete example.

In 2023, Hodel and West ran a simple test. They gave GPT-3 letter-string analogy puzzles — the task on which Webb and colleagues (2023) had claimed that GPT-3 had "emerged" analogical reasoning ability.

The original task looks like this:

    input: abc -> abd, kji -> ?
    expected output: kjj

That's the simple rule "move the last letter one step forward." GPT-3 did well on this task.

Then Hodel and West tried the simplest variants: change the string length from 3 to 4, or shuffle the alphabet a little.

**GPT-3's performance collapsed immediately.**

Not "dropped a bit" — collapsed. Accuracy went from near 100% down to near random guessing. Humans barely changed on these variants, because humans understand the **rule itself**, not a surface pattern tied to a specific length or a specific alphabet.

That's the core question of this chapter: **when a model learns by minimizing training error, what does it actually learn?**

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Contract of Empirical Risk Minimization

Let's make this question precise, mathematically.

The standard paradigm of supervised learning is **Empirical Risk Minimization (ERM)**. Given training data $(x_1, y_1), \ldots, (x_n, y_n)$, we choose a hypothesis $h$ to minimize the empirical risk:

$$
\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)
$$

Here $\ell$ is the loss function, which measures how far the prediction $h(x_i)$ is from the true label $y_i$.

:::details Empirical Risk Minimization (ERM): What Is Training a Model Doing?
**Empirical risk minimization** is the engine behind most machine learning, and in plain words it means: **make the model's prediction error as small as possible on the training data**.

A few key terms:
- **Hypothesis $h$**: the model itself (say, a neural network). Given an input $x$, it outputs a prediction $h(x)$
- **Loss function $\ell$**: a function that scores "how bad the prediction is." Predict "cat" when the label is "dog" — big loss. Predict correctly — small loss
- **Empirical risk $\hat{R}(h)$**: the average loss over all training samples — minimizing this is "getting the model to answer as many training items right as possible"

**The core question**: minimizing training error ≠ truly understanding the rule. The model may just be "memorizing answers" (overfitting), or it may learn surface patterns instead of the underlying mechanism. That's exactly what this chapter asks: what does ERM learn?
:::


ERM's theoretical guarantee comes from a simple intuition: if the training data are sampled independently and identically distributed (i.i.d.) from some distribution $\mathcal{D}$, then when $n$ is large enough, the empirical risk $\hat{R}(h)$ converges to the true risk:

$$
R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}} [\ell(h(x), y)]
$$

How fast is that convergence? Statistical learning theory tells us: for a finite hypothesis space, the rate is $O(1/\sqrt{n})$. For an infinite hypothesis space (like a neural network), the rate depends on the "complexity" of the space — VC dimension, Rademacher complexity, and so on.

**But there's a premise hiding here: is what ERM optimizes actually the same thing as what we really care about?**

Let me unpack that.

What is ERM doing? It's searching for a function $h$ such that some distance (loss) between $h(x)$ and $y$ is minimized on the training data.

But what does "reasoning" need? Reasoning needs the model to capture the **causal mechanism** between $x$ and $y$ — the underlying process that generated the data.

These two are not the same thing.

The classic results of statistical learning theory — for instance, the framework Vapnik and Chervonenkis built in the 1970s — are about **generalization bounds**:

$$
R(h) \leq \hat{R}(h) + \text{complexity penalty}(h, n)
$$

That inequality says: if the hypothesis space isn't too complex, then a model with low training error will also have low test error.

But look at the assumption in there: **the test data and the training data come from the same distribution** $\mathcal{D}$.

In real life, that assumption almost never holds.

Your training data is the data you could collect — maybe from one particular hospital, one particular time period, one particular population. Your test data is what the model meets after deployment — from a different hospital, a different season, a different population.

**Distribution shift is the norm, not the exception.**

And when distribution shift happens, the "correlations that worked on the training data" that ERM learned may fail completely.

<div class="center">

------------------------------------------------------------------------

</div>

## A Pause

The model memorized the training set — we call that overfitting, it's an error, and we punish it.

But what if the model memorized the entire internet?

Then what would we call it?

GPT-4's training data is roughly a large sample of all the text humans have ever written. If it "memorized" all of that, what's the essential difference from overfitting? — size, or something else?

One more question: when distribution shift happens, ERM fails. But humans also fail under distribution shift sometimes — we misjudge things in unfamiliar cultures, and make basic mistakes in new domains.

So is ERM's flaw unique to machine learning, or does it belong to every inductive learning system? Including you?

Let's leave that question on the table.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. Shortcut Learning: When Correlation Is Easier Than Causation

Here's a thought experiment.

Suppose you train a model to recognize "cows." The training set has 1000 photos of cows: 950 with grass backgrounds, 50 with beach backgrounds (some seaside farm).

What would ERM learn?

If the model is simple enough, it might learn: "if the background is grass, predict cow." That rule has 95% accuracy on the training set — pretty good.

But has the rule captured the essence of "cow"? Obviously not. Deploy that model to a beach farm and it will classify every cow wrong.

This is **shortcut learning** — the model exploits **spurious correlations** in the training data instead of the true causal features.

Geirhos and colleagues surveyed this phenomenon systematically in 2020. Their conclusion: shortcut learning is everywhere in deep learning:

- **Texture bias**: ImageNet-trained models classify by texture more than shape, while humans do the opposite
- **Background dependence**: object detection models use background statistics (like "boats usually appear on water") as a shortcut
- **Dataset bias**: sentiment models lean on certain high-frequency words (like "terrible," "amazing") and ignore the overall semantics of a sentence

Why does this happen?

**Because ERM has no mechanism to tell "useful correlation" from "spurious correlation."** As long as some feature correlates with the label in the training data, ERM will use it — whether or not that correlation still holds outside the training distribution.

In 2023, Puli and colleagues uncovered a deeper reason: even when the spurious feature (the shortcut) provides no extra information at all — even when the stable features already determine the label completely — **the default ERM setup (gradient descent + cross-entropy) still prefers the shortcut.**

Why? Because gradient descent implicitly maximizes the classification margin. And in the linearly separable case, the max-margin solution tends to be the one that uses both the stable features and the shortcut — even when the stable features alone would reach zero training error.

This isn't overfitting — training error is already zero. It's a **problem of inductive bias**: the gradient descent + cross-entropy combination naturally prefers a particular type of solution, and when shortcuts exist, that solution over-relies on them.

<figure>
<p><img src="/figures/ch05_fig1_gradient_analysis.png" alt="Gradient evolution of the shortcut feature and the stable feature during training" /></p>
<figcaption>Gradient evolution of the shortcut feature and the stable feature during training</figcaption>
</figure>

*Figure 1: During training, the gradient norm of the shortcut feature rises quickly and stays high, while the stable feature's gradient keeps decaying. The model is gradually attracted to the shortcut during optimization, even though the stable feature is already enough to do the task.*

Let me say this with a more formal framework.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. A Causal Graph View: Stable Features and Spurious Features

Suppose the data-generating process can be described by a causal graph. There are three variables:

- $Y$: the label (say, "is this a cow")
- $X_s$: stable features (like the cow's shape and texture)
- $X_c$: shortcut features (like whether the background is grass)

The true causal relations are:

$$
Y \to X_s, \quad Y \to X_c
$$

That is, the label $Y$ determines both the stable features and the shortcut features. In the training data, because of some sampling bias, $Y$ and $X_c$ are strongly correlated — for example, "cows" almost always appear against grass backgrounds.

But that correlation is not causal. $X_c$ doesn't determine $Y$; it's $Y$ that determines the distribution of $X_c$ in the training data.

What does ERM see? ERM sees the joint distribution $P(X_s, X_c, Y)$, and it learns a prediction function:

$$
\hat{Y} = f(X_s, X_c)
$$

If $X_c$ correlates strongly with $Y$ in the training data, ERM will give $X_c$ a large weight — because that lowers the training error.

The problem: when the test data come from a different distribution where the $Y$–$X_c$ correlation no longer holds (say, cows appear on beaches), the model's predictions fail.

This isn't the model being "not smart enough." It's a structural limit of ERM: **ERM optimizes $P(Y|X)$, not the causal mechanism $P(X|Y)$.**

Pearl's ladder of causation tells us that to answer an interventional question ("if I move the background to a beach, can the model still recognize the cow?"), you need more than conditional probabilities — you need a causal model.

But ERM only sees observational data. It can't tell correlation from causation.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. The Stochastic Parrot Hypothesis: Repeating Is Not Understanding

Now let's go back to the question this chapter opened with: are large language models stochastic parrots?

The heart of the question isn't whether models are "conscious" or "understand" — those are philosophical questions, and we'll set them aside. The heart is: **how much of a model's behavior can be explained by "memorizing + retrieving patterns from the training data"?**

In 2025, Zhao and colleagues proposed a decisive test: is chain-of-thought (CoT) reasoning — reasoning step by step and writing the steps down — real reasoning, or just a mirror image of the training data's distribution?

Their experiment design is clever. They built a fully controllable environment, DataAlchemy, trained language models from scratch, and then systematically changed the distribution gap between training and test data:

- **Task distribution shift**: training saw addition; testing does multiplication
- **Length distribution shift**: training saw 3-digit arithmetic; testing does 5-digit arithmetic
- **Format distribution shift**: training saw the "step 1, step 2" format; testing switches to "first, second"

The result is devastating: **CoT reasoning degraded significantly under all three types of distribution shift.** The model isn't "reasoning" — it's "pattern matching." It learned what input maps to what chain-of-thought format in the training data, and at test time it reproduces that format.

When the test distribution differs from the training distribution, the reproduction fails.

This is the same phenomenon as GPT-3's collapse on letter-string analogy in Section 1: the model learns surface statistical regularities, not the underlying abstract rule.

But there's a deeper question here.

The "stochastic parrot" critique from Bender and colleagues hides an assumption: if the model is only doing statistical pattern matching, then its abilities have a ceiling — it cannot go beyond the statistical structure of the training data.

But is that assumption right?

In 2023, Wei and colleagues pushed back: even if the model is only pattern matching, if the training data is large enough and diverse enough, pattern matching alone may be enough to produce behavior that looks like "reasoning."

This is a debate about **emergence**: when model scale and data scale pass some critical point, does something qualitatively new appear?

The evidence so far is mixed.

On one hand, we do see surprisingly impressive abilities — like GPT-4's performance on certain reasoning tasks, which is close to the human average.

On the other hand, the fragility of these abilities under distribution shift suggests they may still be "complex pattern matching" rather than true abstract reasoning.

The decisive test: **can the model produce correct behavior on combinations it has never seen in training?**

That's what the next section is about: distribution shift as the touchstone.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. Distribution Shift: The Touchstone of Reasoning Ability

If a model truly "understands" what it's doing, then when the input distribution changes, its performance should degrade gracefully — not collapse.

That's a hypothesis you can test precisely.

**In-Distribution (ID) performance** measures how the model does on data similar to its training data. That's the standard test-set evaluation.

**Out-of-Distribution (OOD) performance** measures how the model does on data outside the training distribution. That's the real generalization ability.

Let me show you a few concrete examples of how distribution shift exposes a model's fragility.

**Example 1: Shortcuts in medical image segmentation**

In 2024, Woodland and colleagues studied OOD detection for deep learning models on medical image segmentation. They found: models trained on liver segmentation dropped performance significantly when they met images from different hospitals and different scanners.

The problem wasn't image quality — the new images were perfectly good. The problem was that the model had learned **device-specific artifacts** in the training data as shortcuts.

For example, one particular model of CT scanner produces a specific noise pattern in one corner of the image. The model learned to use that noise pattern to help with segmentation — because in the training data, that noise pattern correlated strongly with liver position.

But when you switch to a different scanner model, the noise pattern disappears, and the model's segmentation accuracy collapses.

**Example 2: Spurious correlation in natural language understanding**

In 2023, Shuieh and colleagues systematically evaluated three post-training algorithms (SFT, DPO, KTO) for robustness under spurious correlations.

They built three task types — math reasoning, instruction following, and document question answering — and planted different levels of spurious correlation into the data (10% vs 90%).

The result: **all models degraded significantly under high spurious correlation.** The preference-learning methods (DPO/KTO) were relatively robust on math reasoning, but on complex, context-heavy tasks, supervised fine-tuning (SFT) actually held up better.

What does this tell us? No single training method universally resists shortcut learning. The best strategy depends on the task type and the nature of the spurious correlation.

**Example 3: Distribution fragility of CoT reasoning**

Back to Zhao's DataAlchemy experiment. Their core finding: **CoT reasoning is a fragile mirror of the training data distribution.**

When any one of the three dimensions — task, length, format — shifts, CoT's effectiveness drops significantly. This shows the model didn't learn "how to reason"; it learned "what a chain of reasoning looks like in the training data."

Worse, the model's failures under distribution shift are **systematic**, not random. It doesn't make occasional mistakes — it fails consistently on particular types of input, because those inputs trigger patterns that don't exist in the training data.

All three examples point to the same conclusion: **distribution shift is not an edge case. It's the core test.** If a model only performs well in-distribution, then what it learned is most likely statistical correlation, not causal mechanism.

<figure>
<p><img src="/figures/ch05_fig2_id_ood_performance.png" alt="ID and OOD performance at different shortcut strengths" /></p>
<figcaption>ID and OOD performance at different shortcut strengths</figcaption>
</figure>

*Figure 2: Left: as the shortcut correlation in the training data gets stronger, ID accuracy rises but OOD accuracy falls. Right: the generalization gap (ID − OOD) grows linearly with shortcut strength; once the shortcut correlation passes 80%, the gap enters the danger zone (\>20%). This quantifies the damage shortcut learning does to out-of-distribution generalization.*

<div class="center">

------------------------------------------------------------------------

</div>

## 7. Why ERM's Inductive Bias Is Not Enough

Now we're at the most important point of the chapter.

The problem isn't ERM itself — ERM is a reasonable learning principle. The problem is: **ERM paired with the standard optimizer (gradient descent) and the standard loss (cross-entropy) produces an inductive bias that isn't fit for learning causal structure.**

Let me lay out the argument.

**Inductive bias** is the implicit assumption in a learning algorithm — it determines which hypothesis the algorithm picks when several hypotheses all fit the training data.

What is the inductive bias of gradient descent + cross-entropy?

In the linearly separable case, gradient descent converges to the **max-margin solution** — the one that maximizes the distance between the decision boundary and the nearest training sample.

That's often good. A max margin usually means better generalization — because it's more robust to small perturbations of the training data.

But when a shortcut exists, the max-margin solution tends to be the one that **uses both the stable features and the shortcut**.

Why? Because if you use both features, you can push the boundary farther out — even if one of those features (the shortcut) will fail out of distribution.

Puli and colleagues' 2023 study characterized this phenomenon precisely. They proved: in a simple linear perceptron task, even when the stable features completely determine the label, gradient descent still assigns nonzero weight to the shortcut — because doing so maximizes the margin.

So what's the fix?

One direction is to **change the inductive bias**. For instance, instead of chasing the max margin, aim for a **uniform margin** — make all training samples' distances to the boundary as equal as possible.

Puli's MARG-CTRL (Margin Control) is exactly this idea: by re-shaping the loss function, the model is encouraged to produce a uniform-margin solution, which reduces reliance on shortcuts.

Another direction is to **model causal structure explicitly**. That means going beyond purely observational data — introducing interventions or counterfactual reasoning — and that's the topic of Chapter 6.

But even without causal inference, smarter training strategies can mitigate shortcut learning.

<div class="center">

------------------------------------------------------------------------

</div>

## 8. Mitigation Strategies: From Data Augmentation to Adversarial Training

If we know where the shortcuts are, what can we do?

**Strategy 1: Data augmentation**

The most direct approach: make the training data more diverse, and break the spurious correlations.

In the cow example, if you can collect enough pictures of "cows on beaches," the model won't over-rely on the "grass background" shortcut.

But this method has two problems.

First, you need to know what the shortcut is. In real scenarios, shortcuts are hidden — you don't know what spurious correlation your model is exploiting.

Second, even if you know the shortcut, collecting diverse data can be very expensive, or infeasible.

**Strategy 2: Reweighting training samples**

If some training samples are "too easy" — the model can predict them correctly using a shortcut — then lower their weight, and force the model to learn the harder samples.

That's the idea behind Tilted ERM, proposed by Li and colleagues in 2020. By introducing a "tilt" parameter $t$, you adjust the loss weight of each sample:

$$
\text{Tilted-ERM}: \quad \min_h \frac{1}{t} \log \left( \frac{1}{n} \sum_{i=1}^n e^{t \cdot \ell(h(x_i), y_i)} \right)
$$

When $t > 0$, high-loss samples (hard samples) get more weight; when $t < 0$, low-loss samples (easy samples) get more weight.

By tuning $t$, you can trade off between robustness and fairness.

**Strategy 3: Adversarial training**

Another idea: explicitly generate "adversarial examples" — the samples where the model would rely on the shortcut and fail — then train on those samples.

Sricharan and Srivastava proposed in 2018: use a GAN to generate samples the model classifies with high confidence but that are actually OOD, and then maximize the model's entropy (uncertainty) on those samples.

This forces the model not to be overconfident on out-of-distribution inputs, reducing shortcut reliance.

**Strategy 4: Causal regularization**

If we have prior knowledge about the causal structure, we can encode it in a regularization term.

For example, if we know some feature $X_c$ is a shortcut, we can add a penalty to the loss that reduces the model's reliance on $X_c$:

$$
\min_h \hat{R}(h) + \lambda \cdot \text{Dependence}(h, X_c)
$$

Here $\text{Dependence}(h, X_c)$ measures how sensitive the model's output is to $X_c$. It can be defined in terms of gradients, mutual information, and so on.

But all these methods share one limitation: **they need some form of supervision** — either prior knowledge about the shortcut, or OOD data, or human-labeled hard samples.

In a fully unsupervised setting, detecting and mitigating shortcut learning is still an open problem.

<figure>
<p><img src="/figures/ch05_fig3_confidence_calibration.png" alt="Confidence calibration curves: ID versus OOD data" /></p>
<figcaption>Confidence calibration curves: ID versus OOD data</figcaption>
</figure>

*Figure 3: The confidence calibration curves show the most dangerous consequence of shortcut learning — overconfidence. ID data (green) is well calibrated along the diagonal, while OOD data (red) veers far off it: at high confidence (0.8–0.9), the model's actual accuracy is only 0.6–0.7. The shaded region marks the overconfidence zone, where the model confidently produces wrong predictions, with no uncertainty warning at all.*

<div class="center">

------------------------------------------------------------------------

</div>

## 9. Pseudocode: Shortcut Detection and OOD Generalization Tests

Let me formalize the core algorithms from the discussion above.

**Algorithm 1: Gradient-based shortcut detection**

```python
import torch

def shortcut_detection(model, dataloader, shortcut_indices, stable_indices, tau=0.5):
    """
    Gradient-based shortcut detection.
    shortcut_indices: column indices of the shortcut features in the input
    stable_indices:   column indices of the stable features in the input
    Returns: shortcut dependence score S (higher means stronger shortcut reliance)
    """
    model.eval()
    ratios = []
    criterion = torch.nn.CrossEntropyLoss()

    for x, y in dataloader:
        x = x.requires_grad_(True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        g = x.grad  # shape: (batch, n_features)
        g_c = g[:, shortcut_indices].norm(dim=1)   # norm of gradients w.r.t. shortcut features
        g_s = g[:, stable_indices].norm(dim=1)     # norm of gradients w.r.t. stable features
        ratio = g_c / (g_s + 1e-8)
        ratios.append(ratio.detach())

    S = torch.cat(ratios).mean().item()
    if S > tau:
        print(f"S={S:.3f} > τ={tau}: model relies significantly on shortcut features")
    else:
        print(f"S={S:.3f} ≤ τ={tau}: model mostly relies on stable features")
    return S
```

The intuition behind this algorithm: if the model over-relies on the shortcut, the gradient of the loss with respect to the shortcut features will be large — because changing the shortcut features noticeably changes the prediction.

**Algorithm 2: Generalization test under distribution shift**

```python
def generalization_test(model, id_loader, ood_loader, device="cpu"):
    """
    Evaluate the model on ID and OOD test sets and report the generalization gap.
    """
    def evaluate(loader):
        model.eval()
        correct, total, high_conf_wrong = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                conf = probs.max(dim=1).values
                correct += (pred == y).sum().item()
                # high-confidence but wrong samples (overconfidence detection)
                high_conf_wrong += ((conf > 0.8) & (pred != y)).sum().item()
                total += y.size(0)
        return correct / total, high_conf_wrong / total

    acc_id,  hcw_id  = evaluate(id_loader)
    acc_ood, hcw_ood = evaluate(ood_loader)
    gap = acc_id - acc_ood

    print(f"ID  accuracy:        {acc_id:.1%}")
    print(f"OOD accuracy:       {acc_ood:.1%}")
    print(f"Generalization gap: {gap:.1%}")
    if gap < 0.05:
        print("-> good generalization; the model may have learned stable features")
    elif gap < 0.20:
        print("-> moderate generalization; partial shortcut reliance")
    else:
        print("-> poor generalization; the model heavily relies on shortcuts")
    if hcw_ood > 0.1:
        print(f"⚠ OOD high-confidence error rate {hcw_ood:.1%}: overconfidence present")
    return acc_id, acc_ood, gap
```

**Algorithm 3: Tilted ERM training**

```python
import torch
import torch.nn as nn

def tilted_erm_train(model, dataloader, t=5.0, lr=1e-3, epochs=50):
    """
    Tilted ERM training: gives more weight to hard samples to reduce shortcut reliance.
    t > 0: larger values focus more on high-loss samples; t = 0 reduces to standard ERM.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            losses = criterion(model(x), y)          # per-sample losses
            # Tilted loss: log-sum-exp softens the max
            tilted_loss = (1.0 / t) * torch.log(
                (1.0 / len(losses)) * torch.sum(torch.exp(t * losses))
            )
            optimizer.zero_grad()
            tilted_loss.backward()
            optimizer.step()

    return model
```

When $t > 0$, this algorithm gives more weight to hard samples, forcing the model not to rely only on shortcuts.

<div class="center">

------------------------------------------------------------------------

</div>

## 10. A Small Pause

Let me take stock of what this chapter did.

Empirical risk minimization is the standard paradigm of supervised learning. Its theoretical guarantee rests on a key assumption: the training data and the test data come from the same distribution.

But in reality, that assumption almost never holds. Distribution shift is the norm, not the exception.

When distribution shift happens, the "correlations that worked on the training data" that ERM learned can fail completely. This isn't a bug — it's a structural feature of ERM: **ERM optimizes statistical correlation, not causal mechanism.**

Shortcut learning is the concrete form of this problem: the model exploits spurious correlations in the training data instead of the true causal features. Even worse, when the spurious features carry no extra information at all, the inductive bias of gradient descent + cross-entropy still makes the model rely on them — because that maximizes the classification margin.

The stochastic parrot hypothesis says: large language models may be doing sophisticated statistical pattern matching, not real reasoning. The fragility under distribution shift — like CoT reasoning collapsing under task, length, and format shifts — supports this hypothesis.

Mitigation methods include data augmentation, sample reweighting, adversarial training, and causal regularization. But all of them need some form of supervision. In a fully unsupervised setting, detecting and mitigating shortcut learning remains an open problem.

This leads to the core question of Chapter 6: if statistical correlation isn't enough, what do we need? The answer is causal reasoning — not just observing $P(Y|X)$, but understanding interventions $P(Y|do(X))$ and counterfactuals $P(Y_x|X', Y')$.

> Correlation is a shortcut, not reasoning. In the next chapter, we'll answer head-on: what is causality? And why observational data can never tell you the answer.

<div class="center">

------------------------------------------------------------------------

</div>

## Open Questions

- Are the "emergent abilities" of large language models a genuine qualitative leap, or just more of the same quantity-based pattern matching? As models keep growing, will the answer to this question change?
- In a fully unsupervised setting, is there a general way to detect shortcut learning? Or does detecting shortcuts essentially require prior knowledge about the task?
- Under what conditions is ERM's inductive bias (max margin) helpful, and under what conditions harmful? Is there a unified framework that characterizes this trade-off?
- If we trained a neural network on purely random data, what kind of "shortcuts" would it learn? What would that thought experiment tell us about the nature of shortcut learning?
- Does human learning also have shortcut learning? If so, how do humans overcome it? And what does that imply for designing better machine learning algorithms?

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Build a Shortcut Trap, Then Watch the Model Fall In

This chapter makes one core claim: ERM will use any correlation in the training data, whether it's causal or not. You're about to verify it with your own hands — not by reading about it, but by watching it happen.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Design Your Shortcut Dataset

Pick a simple binary classification task, then **deliberately** plant a shortcut in the data.

**Option A: Image classification (recommended)**

Task: tell two simple shapes apart (e.g., circle vs. square)

Shortcut design:
- Training set: 90% of circles in the left half of the image, 90% of squares in the right half
- Test set (ID): keep the same position bias
- Test set (OOD): flip the positions — circles on the right, squares on the left

**Option B: Text classification**

Task: sentiment classification (positive vs. negative)

Shortcut design:
- Training set: append "recommended!" to 90% of positive reviews, and "not recommended" to 90% of negative reviews
- Test set (ID): keep the same markers
- Test set (OOD): remove all markers, keep only the review body

**Your first question (answer it before generating any data):** What do you think the model will learn? Will it rely on position/markers (the shortcut), or on shape/semantics (the stable features)? Why?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Train a Standard ERM Model

Use the simplest architecture:
- Image: a 2–3 layer convolutional network
- Text: a simple LSTM or Transformer

Training goal: minimize cross-entropy loss until training accuracy > 95%

**Don't do anything special** — just standard gradient descent + cross-entropy.

Record:
- Training curve (loss vs. epoch)
- Final training accuracy

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Evaluate on Three Test Sets

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# --- Helper: evaluate model accuracy -----------------------------------
def evaluate(model, dataloader, device='cpu'):
    """Return the accuracy (0~1) of the model on the given dataset."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return accuracy_score(all_labels, all_preds)

# 1. ID test set (same distribution as training; shortcut still present)
acc_id = evaluate(model, test_id_loader)
print(f"ID test set accuracy: {acc_id:.3f}")

# 2. OOD test set (shortcut broken: position/marker reversed)
acc_ood = evaluate(model, test_ood_loader)
print(f"OOD test set accuracy: {acc_ood:.3f}")

# 3. No-shortcut baseline (optional): retrain on data with a 50% shortcut correlation, then evaluate on OOD
# train_baseline = regenerate a training set with a 50% (random) shortcut correlation
# model_baseline = train a new model (same architecture, same hyperparameters)
# acc_baseline_ood = evaluate(model_baseline, test_ood_loader)
# print(f"No-shortcut baseline OOD accuracy: {acc_baseline_ood:.3f}")

# 4. Compute the generalization gap
gap = acc_id - acc_ood
print(f"\nGeneralization gap (ID - OOD): {gap:.3f}")
if gap < 0.05:
    print("-> good generalization; the model may have learned stable features")
elif gap < 0.20:
    print("-> partial shortcut reliance; moderate generalization")
else:
    print("-> heavy shortcut reliance; the model fell into the trap!")
```

**Your second question:** How big is the generalization gap? If the gap > 30%, the model relies heavily on the shortcut. Did your model fall into the trap?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Visualize What the Model Is Looking At

**For images:** use a gradient-weighted class activation map (Grad-CAM) or a simple saliency map, and see which region of the image the model attends to when it makes a prediction.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- Gradient-based saliency map (for image classification models) -------
def saliency_map(model, x_sample, target_class):
    """
    Compute the absolute gradients of the prediction w.r.t. the input pixels as the saliency.
    x_sample: tensor of shape=(1, C, H, W); requires_grad will be enabled
    target_class: index of the target class (0 or 1)
    Returns a saliency map of shape=(H, W) (max over channels)
    """
    model.eval()
    x = x_sample.clone().detach().requires_grad_(True)  # enable gradient tracking

    # Forward pass to obtain the prediction logits
    logits = model(x)
    y_hat = logits[0, target_class]  # logit of the target class

    # Backward pass: gradient of y_hat w.r.t. the input x
    model.zero_grad()
    y_hat.backward()

    # Take absolute gradients and max over channels to get a 2-D saliency map
    saliency = x.grad.data.abs()           # shape=(1, C, H, W)
    saliency, _ = saliency.max(dim=1)      # shape=(1, H, W)
    saliency = saliency.squeeze().numpy()  # shape=(H, W)
    return saliency

# Take one sample from the OOD test set and see where the model looks
sample_x, sample_y = next(iter(test_ood_loader))  # take one batch
x0 = sample_x[0:1]   # first image, shape=(1, C, H, W)
pred_class = model(x0).argmax(dim=1).item()

sal = saliency_map(model, x0, target_class=pred_class)

# Visualization: overlay the saliency on the original image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: original image (if it is a single-channel grayscale image)
axes[0].imshow(x0.squeeze().detach().numpy(), cmap='gray')
axes[0].set_title(f'Original image (true class={sample_y[0].item()}, prediction={pred_class})')
axes[0].axis('off')

# Right: saliency heatmap
im = axes[1].imshow(sal, cmap='hot')
axes[1].set_title('Saliency map (region the model attends to)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.suptitle('If the saliency concentrates on the left/right half, the model is using the position shortcut')
plt.tight_layout()
plt.show()

# If the model relies on the position shortcut, the saliency will concentrate
# on the left/right half instead of the shape edges.
```

**For text:** compute each word's contribution to the prediction (via occlusion or gradients).

If the model relies on the word "recommended!", then occluding that word will flip the prediction, while occluding other words makes little difference.

**Your third question:** Where does the model's attention/saliency concentrate? Is it looking at the shortcut, or at the stable features?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 5: Test Confidence Calibration

When the model fails on OOD data, does it know it's failing?

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- Confidence analysis: compare max softmax probabilities on ID and OOD data ---
def get_confidence_and_accuracy(model, dataloader):
    """
    Return the max softmax probability (confidence) of each sample and whether the prediction is correct.
    """
    model.eval()
    confidences, corrects = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)          # convert to a probability distribution
            conf, pred = probs.max(dim=1)             # the max probability is the confidence
            corrects.extend((pred == y_batch).numpy())
            confidences.extend(conf.numpy())
    return np.array(confidences), np.array(corrects)

# Compute confidences on ID and OOD data separately
conf_id,  correct_id  = get_confidence_and_accuracy(model, test_id_loader)
conf_ood, correct_ood = get_confidence_and_accuracy(model, test_ood_loader)

# Print summary statistics
print(f"ID data  - accuracy: {correct_id.mean():.3f}, mean confidence: {conf_id.mean():.3f}")
print(f"OOD data - accuracy: {correct_ood.mean():.3f}, mean confidence: {conf_ood.mean():.3f}")

# Plot confidence histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(conf_id,  bins=20, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title(f'ID confidence distribution\n(accuracy={correct_id.mean():.2f})')
axes[0].set_xlabel('Max softmax probability')
axes[0].set_ylabel('Number of samples')
axes[1].hist(conf_ood, bins=20, color='tomato',    edgecolor='white', alpha=0.85)
axes[1].set_title(f'OOD confidence distribution\n(accuracy={correct_ood.mean():.2f})')
axes[1].set_xlabel('Max softmax probability')
plt.tight_layout()
plt.show()

# Diagnosis: overconfidence test
if conf_ood.mean() > 0.8 and correct_ood.mean() < 0.5:
    print("⚠ Overconfident! OOD data: high confidence + low accuracy -> the model is confidently wrong (dangerous!)")
elif conf_ood.mean() < 0.6:
    print("✓ The model knows it is uncertain (lower OOD confidence); relatively safe")
else:
    print("OOD confidence is moderate; judge it together with accuracy")

# Summary:
# - ID data: high accuracy + high confidence -> normal
# - OOD data: low accuracy + high confidence -> overconfidence (dangerous!)
# - OOD data: low accuracy + low confidence -> the model knows it is uncertain (relatively safe)
```

**Your fourth question (the core one):** What is the model's average confidence on OOD data? If the confidence is still high (> 0.8) while accuracy is low (< 0.5), the model is confidently wrong — the most dangerous consequence of shortcut learning.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 6 (Optional): Try a Mitigation Method

Choose one of the methods from Section 8:

**Method A: data augmentation** Add some "anti-shortcut" samples to the training set (e.g., circles on the right, squares on the left), raising the ratio from 10% to 50% step by step, and watch how the generalization gap changes.

**Method B: sample reweighting** Implement a simplified Tilted ERM: lower the weight of "too easy" samples (the ones the model already predicts correctly with high confidence).

Observe: does the mitigation method reduce the generalization gap? What does it cost (training time? loss of ID accuracy?)?

<div class="center">

------------------------------------------------------------------------

</div>

### Checklist

After finishing this exercise, you should be able to answer:

1. Did your model fall into the shortcut trap? How large is the generalization gap?
2. Is the model's attention/saliency concentrated on the shortcut feature?
3. Is the model overconfident on OOD data?
4. If you tried a mitigation method, did it work? What did it cost?

If you only do one thing, do Step 4. It's the shortest experiment, and the one that gives you an intuition for "what the model is looking at."

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Bender, E. M., et al. (2021). *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* — the original "stochastic parrot" paper, which sparked a wide debate about the nature of large language models
- Hodel, D. & West, J. (2023). *Response: Emergent analogical reasoning in large language models* — a critical rebuttal of GPT-3's "emergent reasoning abilities," showing fragility under distribution shift -> [arXiv:2308.16118]
- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks* — a systematic survey of shortcut learning across vision, language, and speech
- Puli, A., et al. (2023). *Don't blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy* — reveals how the inductive bias of gradient descent + cross-entropy leads to shortcut learning -> [arXiv:2308.12553]
- Zhao, C., et al. (2025). *Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens* — the DataAlchemy experiment, showing CoT reasoning as a fragile mirror of the training distribution -> [arXiv:2508.01191]
- Li, T., et al. (2020). *Tilted Empirical Risk Minimization* — balances robustness and fairness by reweighting samples -> [arXiv:2007.01162]
- Sricharan, K. & Srivastava, A. (2018). *Building robust classifiers through generation of confident out of distribution examples* — generating OOD samples with a GAN for adversarial training -> [arXiv:1812.00239]
- Woodland, M., et al. (2024). *Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation* — OOD detection and shortcut learning in medical imaging -> [arXiv:2408.02761]
- Vapnik, V. & Chervonenkis, A. (1971). *On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities* — the foundational work of statistical learning theory and the theoretical basis of ERM
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* — the classic textbook on causal reasoning; a must-read for understanding the difference between statistical correlation and causality

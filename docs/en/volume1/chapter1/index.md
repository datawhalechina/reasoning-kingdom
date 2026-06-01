# Chapter 1: Against Entropy Increase — Reasoning as a Survival Strategy

> A system that cannot predict the next moment will be destroyed by the next moment.

<div class="center">

------------------------------------------------------------------------

</div>

## I. The Universe Has a Direction

I want to first tell you a story about an ice cube.

You put an ice cube into a cup of warm water. You wait a while. The ice melts, the water cools. You've seen this a thousand times, you think it's boring, you think it's taken for granted. But if you really sit down and think about it, this event hides a puzzle that troubled physicists for nearly two hundred years.

Have you ever seen this process in reverse? A cup of warm water, suddenly, a corner spontaneously freezes into ice, while the rest of the water becomes hotter? You haven't. Not because it violates any law of mechanics — Newton's equations are time-symmetric, running forward and backward both hold. It's because the Second Law of Thermodynamics prescribes a **direction** for the universe.

The name of this direction is entropy increase.

Entropy, roughly speaking, is the "degree of disorder" of a system, or more precisely, the logarithm of the number of microstates corresponding to a macrostate. Ice melting into warm water is going from an ordered state with extremely few possibilities to a disordered state with extremely many possibilities. This process happens spontaneously because the latter is overwhelmingly more probable statistically. The reverse process is not impossible, just extremely improbable — on the timescale of the age of the universe, you might not even wait for it to happen spontaneously once.

:::details The Second Law of Thermodynamics and Entropy: You Can Read This Even Without Understanding Physics
**Entropy** is a quantity in physics that measures the "degree of disorder." You don't need to memorize the formula, just remember an intuition: things naturally and spontaneously always go from order to disorder — a cup will shatter, but the shards won't reassemble themselves; a room will get messy, but won't tidy itself up. This one-way direction is the essence of the Second Law of Thermodynamics.

"Entropy increase" means "the degree of disorder increases." The total entropy of the entire universe is constantly increasing, and this is a physical fact that has been verified countless times in experiments.

Why can living organisms maintain local order? Because they are not isolated systems — they rely on consuming external energy (food, sunlight) to maintain internal order, while discharging more heat (waste heat) to the outside. When the total account is tallied, the entropy of the universe as a whole is still increasing.
:::


So the universe has a direction: toward more chaos, toward more disorder, toward higher entropy.

Now, in this universe surging toward chaos, there are some very strange things.

They are called living organisms.

<div class="center">

------------------------------------------------------------------------

</div>

## II. Existence Against the Current

A living cell is a miracle in the thermodynamic sense.

Not because it violates the Second Law — it doesn't. A cell fights local entropy increase by discharging more entropy to the outside. It consumes ordered chemical energy (glucose, ATP), degrades it into disordered waste heat, and discharges it to the environment. From the perspective of the entire system (cell + environment), total entropy is still increasing. But inside the cell, astonishing orderliness is maintained: precisely folded proteins, precisely regulated ion gradients, precisely transcribed genetic information.

Schrödinger, in his 1944 booklet *What is Life?*, used a beautiful phrase: organisms live by consuming "negative entropy." They take in low-entropy things from the environment, discharge high-entropy waste, and use this difference to maintain their own order.

But have you noticed — maintaining internal order is only a necessary condition for being alive, not a sufficient condition.

A crystal is also highly ordered. The lattice of table salt, the hexagon of a snowflake, the regular arrangement of quartz. Under suitable conditions, they can maintain their structure indefinitely, without consuming energy, without discharging waste heat. Yet we wouldn't say a crystal is "alive."

The reason is: the order of a crystal is **static**. It doesn't need to respond to the outside, doesn't need to react to changes, doesn't need to predict the future. It just exists passively. As long as the environment doesn't exceed its physical limits (doesn't melt, doesn't shatter), it can maintain itself.

The problem that organisms face is completely different. The outside world is dynamic, changing, sometimes hostile. Predators will appear, temperature will suddenly change, food sources will disappear, competitors will arrive. A living system must not only maintain internal order, but also **respond appropriately to changes in the external world** — and respond before the changes cause damage.

This requires **prediction**.

<div class="center">

------------------------------------------------------------------------

</div>

## III. Prediction Is Not a Luxury, It's a Necessity

Let me change the angle to talk about this.

Imagine a very simple organism — a single-celled bacterium, swimming in an environment with a food gradient. Where the food concentration is high, it should stay; where the food concentration is low, it should leave. But the bacterium's cell body is very small, it cannot simultaneously sense the concentration difference between two places — it can only sense the concentration at its current position.

How does it know whether it's moving toward a higher concentration or toward a lower concentration?

The answer is: it remembers the concentration from one second ago and compares it with the current concentration. If now is higher than just now, continue forward; if now is lower than just now, change direction.

This is a very primitive form of prediction — predicting "if I continue in this direction, the concentration will continue to rise." It doesn't even count as true prediction, just a simple temporal difference. But it's already enough for the bacterium to navigate in a food gradient.

Now stretch the timescale longer, increase the organism's complexity.

A frog waiting for a mosquito. It doesn't wait until the mosquito flies into its mouth to react — its visual system tracks the trajectory in advance, its tongue is already launched before the mosquito arrives. There is prediction here: prediction of the mosquito's trajectory, estimation of the time difference, calibration of the timing of the tongue launch.

A chimpanzee watching another chimpanzee's gaze to judge whether it will attack next. There is more complex prediction here: inference of another individual's intention, recognition of behavioral patterns, even some rough model of "what the other is thinking."

The fineness of predictive ability is positively correlated with the cognitive complexity of organisms. This is not accidental. **Organisms that can better predict the world survive better, reproduce more, and pass their predictive mechanisms to offspring.** Evolution, in a certain sense, is a screening process about predictive ability.

Karl Friston's **Free Energy Principle** formalizes this intuition. His core argument is: any system capable of maintaining its own existence must, mathematically, be performing some form of inference — minimizing "surprise" about sensory input. High surprise means the world deviates greatly from the model's predictions, means the system failed to accurately predict what was about to happen, means danger. Thus, surviving is inferring `→ [Friston et al., 2020, arXiv:2002.04501]`.

:::details What is the Free Energy Principle? (Prior Work: Friston, 2020)
The **Free Energy Principle** is a theoretical framework proposed by neuroscientist Karl Friston, attempting to use a unified mathematical principle to explain the behavior of all living systems.

The core idea is very intuitive: a living system must constantly fight against "being hit by the unexpected" — if the world does something completely unexpected, the system may not survive. Therefore, all systems that can survive long-term are mathematically equivalent to systems that "constantly minimize surprise," that is, systems that "constantly make the world conform more to their own expectations."

The "free energy" here is a term borrowed from thermodynamics, but in this context, it is an information-theoretic quantity, representing "the upper bound of the gap between model predictions and actual sensations." Minimizing free energy ≈ making predictions more accurate.

You don't need to master the formulas, just remember: this principle says that **inference and prediction are not luxury functions of the brain, but mathematical necessities for any system that can maintain itself**.
:::


And so, prediction goes from being a luxury to being a necessity for survival.

<div class="center">

------------------------------------------------------------------------

</div>

## Pause for a Moment

Bacteria are doing temporal differences, frogs are doing trajectory prediction, chimpanzees are building intention models — they are all "predicting," but we wouldn't say bacteria are "thinking."

So, between prediction and thinking, is there a clear boundary? Or is it just a difference in complexity, quantitative change triggering qualitative change?

There's also a more uncomfortable question: can a system make **perfect predictions** while **completely not understanding the world**?

If yes — then what exactly is the use of "understanding"?

Let's set this question aside for now. Continue moving forward.

<div class="center">

------------------------------------------------------------------------

</div>

## IV. But Prediction Is Not Equal to Understanding

I want to stop here for a moment, because there's a trap that's easy to fall into.

Predictive ability and understanding ability are not the same thing.

Let me give you an example.

An excellent weather forecaster, after seeing specific cloud formations, pressure distributions, and wind direction data, can predict that it will rain tomorrow. His prediction may be very accurate. But if you ask him "why does this kind of cloud formation lead to rainfall," he needs to answer using fluid mechanics, thermodynamics, and phase-change dynamics of water vapor — this is a question at another level, much deeper than "will it rain tomorrow."

Or a more extreme example: a trained neural network, after learning from large amounts of historical data that "after a certain pattern appears, a certain result will occur," can then very accurately predict that result. But does it understand "why"? Does it know the underlying causal mechanism? This is a question we won't fully answer even by the end of this book, but we'll get closer and closer to it.

This distinction is important because it tells us: **prediction accuracy cannot serve as a proxy for depth of understanding**.

A system can achieve extremely accurate predictions while completely not understanding what it's predicting. It captures patterns in a statistical sense but has not built a causal model of the world. When conditions change — when it encounters situations outside the training distribution — its predictions will collapse, and the collapse is often in unpredictable, strange ways.

Then, what is true "understanding"?

This requires us to establish a hierarchy.

<div class="center">

------------------------------------------------------------------------

</div>

## V. From Reflex to Reasoning: A Story of Hierarchy

I want to use an analogy to introduce this hierarchy.

Imagine you are learning to drive.

At the very beginning, every action of yours is conscious: clutch, shift gears, steering wheel, accelerator — you have to think about these four things simultaneously, and also watch the road. This is the stage of extremely heavy cognitive load.

A few months later, you drive to and from work while thinking about what to eat for dinner. Shifting gears, braking, turning — these actions become automatic, descending from conscious reasoning into some kind of reflex.

But if a child suddenly runs out, your consciousness will instantly switch back, making a very non-automatic decision — where to steer? Hard brake or swerve around? This is high-level reasoning happening within milliseconds.

Here, three different processing levels are running simultaneously. They don't replace each other, but coexist, activated in different situations.

Now let me unfold this analogy.

**Level 1: Reflex**. Input triggers output, no intermediate model, no representation of the world. The knee-jerk reflex: the doctor taps your knee with a small hammer, your leg lifts. You didn't think "the leg should lift up," the entire circuit is closed at the spinal cord level, the cerebral cortex doesn't even need to participate. There is input and output here, but no reasoning.

**Level 2: Associative Learning**. The system establishes statistical associations between stimuli. Pavlov's dog: bell rings, saliva is secreted. There is learning here — the association is established through experience, not hard-coded. But it's still association, not a model. The dog doesn't know "the bell means food is about to appear," it is just executing a reinforced mapping. If the food changes location, or the time interval between the bell and food changes, this mapping will get confused, but the dog cannot infer "why."

**Level 3: Generative Models**. This is a fundamental transformation. The system no longer passively waits for input and then reacts, but actively maintains an internal representation of the world, continuously generating predictions about upcoming input.

The **predictive coding** framework proposed by neuroscientists describes precisely this level. The basic working mode of the brain is not processing sensory input bottom-up, but continuously generating predictions top-down, and then only processing the **error** between predictions and actual sensations. You walk into a dark room, your brain is not waiting for photons in a blank state — it is already predicting what might be in this room, predicting where the chair is, predicting the texture of the walls. Prediction error is the information that is truly processed in large quantities `→ [Sennesh et al., 2022, arXiv:2208.10601]`.

:::details Predictive Coding: The Brain Is Not a Camera (Prior Work: Rao & Ballard, 1999 et al.)
Traditional intuition believes the brain is like a camera: light comes in, the visual cortex processes it, you see an image. **Predictive coding** overturns this picture.

This framework argues: the brain's main job is to **continuously generate predictions top-down**, and then only pass "the parts where the prediction is wrong" (prediction error) upward for higher levels to update the model. Most of the time, a considerable portion of what you "see" is filled in by your own brain.

One piece of evidence: visual illusions. Even if you know an image is an illusion, you still can't stop being deceived — because your high-level model has solidified a prediction, which is more "stubborn" than a single sensory input.

This theory explains why the brain only consumes 20 watts: most of the time it only needs to process the small portion of signals that are "wrong," rather than the full amount of sensory data.
:::


This architecture explains many phenomena: why you are almost insensitive to things within expectations and extremely sensitive to unexpected things; why you can still clearly hear the other person talking in a noisy environment (your model is helping to fill in information); why visual illusions persist even when you know they are illusions — because your high-level model has solidified a prediction that is harder to overturn than a single sensory input.

**Level 4: Causal Models**. Generative models may still only be capturing statistical correlations, not causal mechanisms. A true causal model can answer questions at three levels — Pearl's Causal Ladder: If there are dark clouds in the sky, what is the probability of rain (observation)? If I artificially create dark clouds, will it rain (intervention)? If there had been no dark clouds in the sky that day, would yesterday's rain still have happened (counterfactual)? These three questions require three completely different abilities; pure statistical correlations can only answer the first. We will go deep into this distinction in Chapter 6.

**Level 5: Meta-Reasoning**. The ability to reason about one's own reasoning. Knowing where your own model is reliable and where it's unreliable. A child finishes a math problem, feels it's wrong, redoes it — they are monitoring their own reasoning process. A scientist asks themselves "can this experiment rule out alternative hypotheses" — they are reasoning about whether their own reasoning method is sufficient.

We don't know where current AI systems stand in this hierarchy. This question will run through this book.

<div class="center">

------------------------------------------------------------------------

</div>

## VI. Information Has Mass, Reasoning Has a Cost

Before continuing, I want to insert a physicist's interlude here.

In 1961, IBM physicist Rolf Landauer published a paper demonstrating something that sounded very strange at the time: **erasing one bit of information necessarily releases at least $k_B T \ln 2$ of heat into the environment**. Here $k_B$ is the Boltzmann constant, $T$ is the environmental temperature.

This conclusion is called Landauer's Principle. Its implication is: **information is not free**. Processing information, especially erasing information, has an unavoidable physical cost. There is a deep connection between information and thermodynamics `→ [Chattopadhyay et al., 2025, arXiv:2506.10876]`.

:::details Landauer's Principle: How Much Energy Does It Cost to Delete a Bit? (Prior Work: Landauer, 1961)
**Landauer's Principle** says: physically, erasing one bit of information requires releasing at least $k_B T \ln 2$ joules of heat into the environment (at room temperature, about $3 \times 10^{-21}$ joules, very small but not zero).

$k_B$ is the Boltzmann constant (about $1.38 \times 10^{-23}$ J/K), $T$ is the absolute temperature (in Kelvin), $\ln 2 \approx 0.693$. The specific numerical value of this formula is not important; what matters is: **this number is not zero**.

Why is this important? Because it means information processing has a physical cost; reasoning is not a cost-free abstract operation, but a real physical process. This principle was later experimentally confirmed and also resolved the "Maxwell's demon" paradox that had troubled physicists for nearly a hundred years (see main text).
:::


This resolves a paradox that had troubled physicists for nearly a hundred years — Maxwell's demon. Maxwell proposed a thought experiment in 1867: a small demon sits on a partition, observes the speed of each gas molecule, lets fast molecules pass to one side, and slow molecules stay on the other side. Gradually, one side gets hot, the other cold — as if a temperature difference is created without consuming energy. This violates the Second Law of Thermodynamics.

Landauer pointed out: in the process of observing each molecule, the demon must erase the information of the previous molecule to process the next. And erasing information must release heat. This heat precisely compensates for the temperature difference it wanted to exploit. The demon is not a winner; it is just paying the bill in a hidden way.

Why do I tell this story here?

Because it tells us: **any system that performs reasoning pays a physical price**. Every instance of information processing, every model update, every calculation of prediction error — these are all real physical processes, all have a thermodynamic cost. Reasoning is not free; the universe does not permit free reasoning.

A far-reaching corollary: the reasoning architecture of biological systems, under evolutionary pressure, should be highly energy-efficient. The brain's predictive coding architecture — only processing prediction errors, not raw input — is near-optimal in both information theory and thermodynamics. Your brain consumes about 20 watts, roughly the power of a dim light bulb, but the computation it performs is something no existing AI system can replicate at the same energy consumption. This is not accidental; this is the result of four billion years of evolutionary pressure.

<div class="center">

------------------------------------------------------------------------

</div>

## VII. Reasoning Requires a Starting Point

Let's take one more step forward.

Bayesian inference is the most precise framework we currently have for "how to update beliefs based on evidence":

$$
P(h \mid e) = \frac{P(e \mid h) \cdot P(h)}{P(e)}
$$

You have a hypothesis $h$ about the world, you observe evidence $e$, then you update your belief about $h$, obtaining the posterior probability $P(h \mid e)$.

:::details Bayes' Formula: How to Read This Equation?
This formula says: **the probability that hypothesis $h$ is true after seeing evidence $e$** (posterior probability), equals the combination of three quantities:

- $P(h)$: **prior probability** — your initial belief about this hypothesis before seeing the evidence (e.g., "the probability of heads is 50%")
- $P(e \mid h)$: **likelihood** — if hypothesis $h$ is true, the probability of observing evidence $e$ (e.g., "if the coin is fair, the probability of tossing 3 consecutive heads is 1/8")
- $P(e)$: **normalization constant** — the total probability of observing $e$ under all possible hypotheses, used to ensure the result is a valid probability

Intuitive version: **new belief = old belief × degree of evidence support ÷ normalization**

Core idea: you don't start inference from scratch; you **update** your belief based on prior beliefs, using new evidence. The stronger the prior, the more evidence is needed to change it.
:::


But there is a $P(h)$ in this formula — the prior probability. It is the belief about $h$ that you already held before seeing the evidence. Bayesian updating does not start from zero; it always departs from a prior.

Where does the prior come from?

In practice, priors come from past experience, from previous inferences, from domain knowledge, from — in the case of biological systems — hardwired assumptions accumulated through evolution. But if you trace all the way back, there is always some initial belief accepted without justification.

This is not a defect; it is a structural feature of reasoning itself: **you must start from somewhere, and the place where you start cannot be fully justified by your own reasoning process.**

For biological systems, evolution solved this problem for them, at least partially: those individuals holding "good priors about the world" survived, reproduced, and passed their neural system structures to offspring. Natural selection is a meta-optimization process about prior fitness.

But note a subtlety: this optimization process itself is not reasoning. Natural selection is blind random variation plus selection pressure; it doesn't know what it's doing, it just preserves what happens to work. The foundation of biological systems' reasoning ability is optimized by a process that does not reason.

This is something worth stopping to think about.

For machine learning systems, training data plays a similar role. The model distills statistical regularities from the training data, and these regularities become implicit priors for reasoning. But the training data itself is not neutral — it comes from a specific distribution, carries specific assumptions, reflects a specific world. When the system encounters situations outside the training distribution, where will these implicit priors break? In what way will they break?

**Any reasoning system has one or more anchor points that cannot be fully reached by its own reasoning process.** You can update your priors. But the rules of updating themselves depend on a deeper meta-prior. You can question your assumptions. But the way you question depends on a set of logical rules you haven't questioned. This is a recursion with an endpoint — the endpoint is some starting belief where judgment is suspended.

The quality of the anchor point determines how far reasoning can go.

We will return here with more formal language in Chapter 15. For now, just remember this sentence.

<div class="center">

------------------------------------------------------------------------

</div>

## VIII. A Small Pause

We've covered a lot. Let me stop here and sort out the threads of this chapter.

 We started from the Second Law of Thermodynamics: the universe moves toward chaos, and living organisms go against the current, maintaining their own order by discharging entropy to the outside. But maintaining static order is not enough — the outside is dynamic, changing, sometimes hostile. So organisms need **prediction**, need to respond before changes happen.

 Predictive ability has levels: from the simplest temporal difference, to associative learning, to generative models, to causal reasoning, to meta-reasoning. Each level is more powerful than the previous, but also more costly — Landauer's Principle tells us that information processing has a physical cost; reasoning is not free.

 And all reasoning requires a starting point, a prior anchor point that cannot be fully justified by reasoning itself. The source of this anchor point, for organisms, is evolution; for machines, it's training data — but neither is perfect; both carry their own blind spots.

 These three constraints — **cost, hierarchy, anchor points** — are the foundation for understanding reasoning.

---

## IX. The Physical Reasoning of Life: The Cost and Implications of Entropy Reduction

Before concluding this chapter, let's return to the starting point, but go deeper.

Schrödinger's concept of "negative entropy" in *What is Life?* reveals a profound physical fact: life does not violate the Second Law of Thermodynamics, but is a special system that **uses energy gradients to achieve local order**. This local entropy reduction comes at the cost of greater entropy increase in the environment — living systems, by continuously taking in energy and matter (negative entropy) from the outside, maintain their own ordered structures, while discharging more disorder (waste heat, waste) into the environment.

### Dissipative Structures: Order Far from Equilibrium

Belgian physicist Prigogine's **dissipative structure theory** provides a mathematical framework for this: under conditions far from thermodynamic equilibrium, open systems, through continuous energy-matter flow, can spontaneously form and maintain ordered structures. Life is precisely the ultimate expression of such dissipative structures.

This physical mechanism has important implications for reasoning systems:

1. **Analogy between energy flow and information flow**: Life needs energy flow to maintain physical order; reasoning systems need information flow to maintain "cognitive order." Without continuous information input (perception, learning), reasoning systems will "cognitively degrade," just as life without energy input will die.

2. **The cost of local order**: Life's local entropy reduction comes at the cost of environmental entropy increase; AI's "understanding" comes at the cost of computational complexity. Both embody the fundamental principle that **order requires paying a price**.

3. **Self-organization and emergence**: The origin of life is a self-organization phenomenon caused by energy flow in the universe — an ordered structure formed through energy input in a local region under the law of entropy increase. Similarly, "understanding" in AI systems may also be a self-organized emergent phenomenon in data flow.

### From Physical Reasoning to Cognitive Reasoning

This perspective lets us see the **physical foundation** of reasoning:

- **Physical layer**: Living systems fight entropy increase through energy flow
- **Information layer**: Cognitive systems fight "cognitive entropy increase" (confusion, ignorance) through information flow
- **Computational layer**: AI systems achieve pattern recognition and prediction through computational flow

The three layers share the same **meta-pattern**: all require continuous external input (energy/information/data), all exchange local order at some cost (environmental entropy increase/computational complexity), and all may, under suitable conditions, give rise to emergent complex behavior.

### Reflection Question: What is the "Energy" of Reasoning?

If life needs physical energy to maintain survival, what "energy" does reasoning need to maintain effectiveness?

Possible answers: **information diversity** (to avoid overfitting), **computational resources** (to achieve complex inference), **time** (learning process), **cognitive load** (attention allocation)... None of these are free; all have their own "thermodynamic cost."

---

 Next, starting from Chapter 2, we will trace the history: what was it like when humans first attempted to turn reasoning into a mechanical process? Where did those early attempts succeed, where did they fail, and what did they leave behind?

<div class="center">

------------------------------------------------------------------------

</div>


> Reasoning requires representation. In the next chapter, we will see how humans used symbols and rules to build the first generation of reasoning machines — and why they ultimately failed.

---

## Thought Echoes: From Thermodynamics to Machine Learning

Readers who have reached here might ask: what exactly is the relationship between thermodynamic entropy and information-theoretic entropy?

**The answer is: the mathematical form is highly consistent; the physical meanings are each independent.**

In 1948, when Shannon founded information theory, he consulted von Neumann about what term to use to describe the uncertainty of information. Von Neumann said: "Call it 'entropy' — after all, not many people understand thermodynamics, and in a debate, you'll always have the upper hand."

This suggestion was adopted. Thus we have:
- Thermodynamic entropy: $S = -k_B \sum p_i \ln p_i$, describing the degree of disorder of physical systems
- Information entropy: $H(X) = -\sum p(x) \log_2 p(x)$, describing the uncertainty of information  
- Cross-entropy: $H(p,q) = -\sum p(x) \log q(x)$, describing the divergence between two distributions

When you train a neural network to minimize cross-entropy, you are doing two things:
1. **Measuring the gap**: calculating the "distributional distance" between the model's prediction $q(x)$ and the real world $p(x)$
2. **Fighting against disorder**: through gradient descent, moving $q(x)$ from "a chaotic state far from $p(x)$" toward "an ordered state close to $p(x)$"

This echoes the core proposition of this chapter: **reasoning is a survival strategy against entropy increase**. In the physical world, life fights thermodynamic entropy increase through metabolism; in the information world, intelligent systems fight the entropy increase of predictive distributions through learning.

The formal similarity is not accidental. It reveals that when different domains face the fundamental challenge of "from disorder to order," they converge on similar mathematical structures.

## Unresolved

- What are the essential similarities and differences between the brain's predictive coding architecture and the reasoning structure of large language models?

- What does Landauer's Principle mean in practice for AI systems — are we approaching the thermodynamic limits of reasoning?

- Evolution selected "good priors" for organisms, but evolution itself does not reason — this means the foundation of biological cognitive systems is optimized by a blind process. How reliable is this foundation?

- **Life as an open system**: If life achieves local entropy reduction through energy flow, through what "flow" do AI systems achieve "cognitive order"? Information flow, data flow, or computational flow? What is the thermodynamic cost of these "flows"?

- **Dissipative structures and AI**: Prigogine's dissipative structure theory describes how systems far from equilibrium maintain order. Can the AI training process (gradient descent, parameter updates) be viewed as a kind of "cognitive dissipative structure"? Is the model after training stabilization in a "cognitive equilibrium state"?

<div class="center">

------------------------------------------------------------------------

</div>

## Hands-On: Build a Minimal Bayesian Predictor

This chapter talked about three things: reasoning has costs, reasoning has hierarchy, reasoning needs anchor points. Now it's your turn to turn these three things into something runnable.

Not asking you to write a big system. Asking you to use the minimum amount of code to personally experience where these three constraints bite.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Decide What to Predict

Choose a domain where you can obtain sequential data — the simpler the better.

A few suggestions: - Toss a coin you're unsure is fair, record the heads/tails sequence - Observe the hourly visitor count of a certain website, encode as "high traffic/low traffic" - Record your own mood state over a day (good/neutral/bad), for several consecutive days

**Your first question:** What is the "prior" of this sequence? Before seeing any data, what assumptions do you have about the distribution of this sequence? Write it out — even a simple statement like "I think heads and tails are each 50%."

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Implement Bayesian Updating

No external libraries. Use the formula given in Section VII:

$$
P(h \mid e) = \frac{P(e \mid h) \cdot P(h)}{P(e)}
$$

The logic you need to implement is:

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Step 1: Define the hypothesis space ──────────────────────────────────────────
# Assume the probability θ of the coin landing heads takes the following discrete values
hypotheses = np.arange(0.1, 1.0, 0.1)   # [0.1, 0.2, ..., 0.9]

# ── Step 2: Assign prior probabilities to each hypothesis ──────────────────────────────
# Uniform prior: assume each θ is equally likely at the start
prior = np.ones(len(hypotheses)) / len(hypotheses)

# ── Step 3: Bayesian update function ──────────────────────────────────────
def bayesian_update(prior, hypotheses, observation):
    """
    Update the posterior distribution based on a single observation.
    observation: 1 means Heads, 0 means Tails
    """
    # a. Compute the likelihood of observing this result under each hypothesis P(e | h_i)
    if observation == 1:
        likelihoods = hypotheses           # Heads: likelihood = θ
    else:
        likelihoods = 1.0 - hypotheses    # Tails: likelihood = 1 - θ

    # b. Compute unnormalized posterior: P(e | h_i) × P(h_i)
    unnormalized = likelihoods * prior

    # c. Normalize: divide by the sum of all unnormalized posteriors, ensuring probabilities sum to 1
    posterior = unnormalized / unnormalized.sum()

    return posterior

# ── Step 4: Simulate an observation sequence and update step by step ──────────────────────────────
# Example: manually input a sequence of heads/tails (1=heads, 0=tails)
observations = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                1, 1, 0, 1, 1]

top_hypothesis_history = []   # Record the index of the highest-probability hypothesis each round
top_posterior_history  = []   # Record the highest posterior probability value each round

current_prior = prior.copy()

for i, obs in enumerate(observations):
    current_prior = bayesian_update(current_prior, hypotheses, obs)

    # Record the highest-probability hypothesis
    best_idx = np.argmax(current_prior)
    top_hypothesis_history.append(hypotheses[best_idx])
    top_posterior_history.append(current_prior[best_idx])

    print(f"Observation {i+1:2d}: {'Heads' if obs==1 else 'Tails'} | "
          f"Most likely θ = {hypotheses[best_idx]:.1f} | "
          f"Posterior probability = {current_prior[best_idx]:.3f}")

# ── Plot line chart: change of the highest posterior probability with number of observations ──────────
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(observations) + 1), top_posterior_history,
         marker='o', linewidth=2)
plt.xlabel("Observation Number")
plt.ylabel("Posterior Probability of the Most Likely Hypothesis")
plt.title("Bayesian Updating: Convergence of Posterior Probability with Observations")
plt.ylim(0, 1)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
```

After completing this, look at the line chart: the horizontal axis is the observation sequence number, the vertical axis is the current posterior probability of the "most likely hypothesis."

**Your second question:** How many observations do you need before the posterior probability starts to stabilize? What is the relationship between this number and the choice of prior?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Experience the Prediction Hierarchy

Using the Bayesian update you implemented, make predictions at three different levels:

**Level 1 (Reflex level)**: What is the next observation most likely to be? Directly output the prediction of the current highest-probability hypothesis.

**Level 2 (Generative model level)**: Not just predict the next one; predict the expected distribution of the next 10. Your Bayesian predictor already has a model about "what parameter controls this sequence" — use it to generate expectations for 10 steps.

**Level 3 (Meta-reasoning level)**: How confident are you in your prediction? What is your posterior entropy? The more uniform the posterior (higher entropy), the more uncertain you are about the hypothesis space, and the more conservative your predictions should be.

```python
import numpy as np

def posterior_entropy(posterior):
    """
    Compute the entropy (Shannon entropy) of the posterior distribution.
    Higher entropy means greater uncertainty about the hypothesis space; lower entropy means beliefs are more concentrated.
    posterior: numpy array containing the posterior probabilities of each hypothesis
    """
    # Filter out entries with zero probability (log(0) is meaningless)
    p = posterior[posterior > 0]
    # Compute Shannon entropy: H = -∑ P(h_i) × log(P(h_i))
    entropy = -np.sum(p * np.log(p))
    return entropy

# Example usage: assume current posterior distribution
example_posterior = np.array([0.05, 0.1, 0.2, 0.4, 0.15, 0.05, 0.03, 0.01, 0.01])
h = posterior_entropy(example_posterior)
print(f"Posterior entropy = {h:.4f} (closer to 0 means more certain, max is about {np.log(len(example_posterior)):.4f})")
```

**Your third question:** In what situation should the meta-information from Level 3 ("I'm uncertain") override the direct prediction from Level 1? Can you find a concrete example?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Find Your Anchor Point

Now go back to what Section VII said: "Any reasoning system has an anchor point that cannot be fully reached by its own reasoning process."

For your Bayesian predictor, the anchor point is the prior.

Do an experiment: using three different priors — a uniform prior, a prior strongly biased toward a certain hypothesis, and a completely wrong prior — run the same observation sequence separately.

Record: - Under the three priors, how many observations are needed for the posteriors to reach "almost the same" result? - If the data volume is limited (e.g., only 20 observations), how large is the gap in final beliefs given by the three priors?

**Your fourth question (and the hardest question of this experiment):** If your prior is "completely wrong," can Bayesian updating correct it? How much data is needed? What does this mean for an AI system pre-trained on wrong training data?

<div class="center">

------------------------------------------------------------------------

</div>

### Verification Criteria

You don't need to achieve perfection. You need to achieve:

- Can run through a complete "prior → observation → posterior update" loop

- Can clearly explain what your prior is and why you set it that way

- Can find a case where your system fails — a situation not covered by your hypothesis space

The third point is the most important. Expert systems failed because they encountered the boundaries of their rule base; where will your Bayesian predictor encounter its own boundaries?

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- [Friston et al., 2020] — Free Energy Principle: Markov Blankets, Bayesian Mechanics and Approximate Inference, arXiv:2002.04501

- [Sennesh et al., 2022] — Temporal-Average Formalization of Active Inference, Connecting Optimal Control and Neurophysiology, arXiv:2208.10601

- [Chattopadhyay et al., 2025] — Landauer's Principle and Computational Thermodynamics: A Survey, arXiv:2506.10876

- [Tlusty, 2010] — Rate-Distortion Theory of Biomolecular Codes: Life as an Information Channel, arXiv:1007.4471

- Schrödinger, E. (1944). *What is Life?* — Life and Negative Entropy, the philosophical starting point of this chapter's ideas

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why* — A popular exposition of the Causal Ladder, essential preparatory reading before Chapter 6

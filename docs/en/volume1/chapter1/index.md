# Chapter 1: Fighting Entropy — Reasoning as a Survival Strategy

> A system that cannot predict the next moment will be destroyed by the next moment.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. The Universe Has a Direction

I want to start by telling you a story about an ice cube.

You drop an ice cube into a glass of warm water. You wait a while. The ice melts, and the water cools down. You have seen this happen a thousand times. It's boring. It's obvious. But if you actually sit down and think about it, this little event hides a puzzle that has bothered physicists for almost two hundred years.

Have you ever seen it run backwards? A glass of warm water, and suddenly some corner of it freezes by itself while the rest gets hotter? No. Not because this breaks any law of mechanics — Newton's equations are symmetric in time; they hold just as well running backwards. No, the reason is that the second law of thermodynamics gives the universe a direction.

That direction is called entropy increase.

Entropy, loosely speaking, is the "amount of disorder" of a system — or more precisely, the logarithm of the number of microscopic states that match one macroscopic state. Ice melting into warm water goes from an ordered state with very few possibilities to a disordered state with very many possibilities. The process happens by itself because the latter is overwhelmingly more likely, in statistical terms. The reverse process is not impossible. It's just extremely improbable — on the timescale of the age of the universe, you might never see it happen once.

:::details The Second Law of Thermodynamics and Entropy: No Physics Background Needed to Follow This
**Entropy** is the quantity in physics that measures "how disordered something is." You don't need to remember the formula — just keep one intuition: things naturally go from order to disorder. A cup breaks, and the pieces never put themselves back together. A room gets messy, and it never cleans itself up. That one-way-ness is the essence of the second law of thermodynamics.

"Entropy increase" simply means "disorder increases." The total entropy of the universe keeps growing — a physical fact verified by experiments countless times.

Why can living things stay ordered locally? Because they are not isolated systems. They maintain internal order by consuming outside energy (food, sunlight) while dumping even more heat (waste heat) into the outside world. When we do the total accounting, the entropy of the universe as a whole still increases.
:::

So the universe has a direction: toward more mess, toward more disorder, toward higher entropy.

And now, inside this universe that is running toward disorder, there are some very strange things.

They are called living creatures.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. Creatures Swimming Against the Current

A living cell is a miracle in thermodynamic terms.

Not because it breaks the second law — it doesn't. A cell fights local entropy increase by exporting even more entropy to the outside world. It consumes ordered chemical energy (glucose, ATP), degrades it into disordered waste heat, and pushes it out. Looking at the whole system (cell + environment), total entropy still rises. But inside the cell, staggering order is maintained: proteins folded with precision, ion gradients regulated with precision, genetic information transcribed with precision.

In his 1944 little book *What Is Life?*, Schrödinger used a lovely phrase: living organisms feed on "negative entropy." They take in low-entropy things from the environment, push out high-entropy waste, and use the difference to keep themselves in order.

But did you notice — maintaining internal order is only a necessary condition for being alive. It is not sufficient.

A crystal is also highly ordered. The lattice of table salt, the six-fold symmetry of a snowflake, the regular arrangement of quartz. Under the right conditions they can keep their structure forever, without consuming energy, without dumping waste heat. Yet we would never say a crystal is "alive."

The reason: a crystal's order is **static**. It doesn't need to respond to the outside world, react to change, or predict the future. It just passively exists. As long as the environment doesn't exceed its physical limits (no melting, no shattering), it persists.

Living creatures face a completely different problem. The outside world is dynamic, changing, and sometimes hostile. Predators appear. Temperatures swing. Food sources vanish. Rivals arrive. A living system must not only keep its internal order — it must also **respond properly to changes in the outside world** — and it must respond before the change does damage.

That requires **prediction**.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. Prediction Is Not a Luxury. It's a Necessity.

Let me look at this from another angle.

Imagine a very simple organism: a single-cell bacterium swimming in an environment with a food gradient. Where the food concentration is high, it should stay. Where it is low, it should leave. But the bacterium is tiny. It cannot sense the concentration at two places at once — it can only sense the concentration right where it is.

How does it know whether it is moving toward higher concentration or toward lower?

The answer: it remembers the concentration from one second ago and compares it with the present. If the concentration now is higher than a moment ago, keep going. If it's lower, change direction.

That is a very primitive kind of prediction — predicting "if I keep going this way, the concentration will keep rising." It hardly deserves the name prediction. It's just a simple time difference. But it's enough for the bacterium to navigate the food gradient.

Now stretch the timescale and raise the complexity.

A frog waits for a mosquito. It doesn't wait until the mosquito flies into its mouth. Its visual system tracks the trajectory in advance, and its tongue is already flicking out before the mosquito arrives. There is prediction here: a prediction of the mosquito's path, an estimate of the time gap, a calibration of when to strike.

A chimpanzee watches another chimpanzee's eyes, trying to judge whether it is about to attack. There is more complex prediction here: an inference about another individual's intentions, recognition of behavior patterns, even a rough model of "what the other one is thinking."

How refined an organism's prediction is tracks how complex its cognition is. This is no accident. **Creatures that can predict the world better survive better, reproduce more, and pass their prediction mechanisms to their descendants.** Evolution, in a sense, is a filtering process on predictive ability.

Karl Friston's **free energy principle** formalizes this intuition. His core claim is: any system that can keep existing must, mathematically, be doing some form of inference — minimizing the "surprise" of its sensory input. High surprise means the gap between the world and the model's predictions is big; it means the system failed to predict what was about to happen; it means danger. So, to survive is to infer `-> [Friston et al., 2020, arXiv:2002.04501]`.

:::details What Is the Free Energy Principle? (Prior Work: Friston, 2020)
The **free energy principle** is a theoretical framework proposed by the neuroscientist Karl Friston. It tries to explain the behavior of every living system with one unified mathematical principle.

The core idea is very intuitive: a living system must keep fighting off "being ambushed by the unexpected." If the world does something completely unexpected, the system may fail to survive. So every system that survives long term is mathematically equivalent to a system that "constantly minimizes surprise" — in other words, a system that "constantly makes the world match its expectations."

"Free energy" here is a word borrowed from thermodynamics, but in this context it is an information-theoretic quantity: it means "an upper bound on the gap between the model's predictions and actual sensations." Minimizing free energy ≈ making predictions more accurate.

You don't need to master the formula. Just remember: this principle says that **inference and prediction are not luxury functions of the brain; they are a mathematical necessity for any system that maintains itself**.
:::

And just like that, prediction went from a luxury to a survival necessity.

<div class="center">

------------------------------------------------------------------------

</div>

## A Pause

The bacterium does time differences. The frog predicts trajectories. The chimpanzee builds models of intentions. All of them are "predicting," but we would not say the bacterium is "thinking."

So is there a clear boundary between prediction and thinking? Or is it only a difference in complexity — a case where quantity turns into quality?

And here is a more uncomfortable question: can a system make **perfect predictions** while **not understanding the world at all**?

If it can — then what, exactly, is "understanding" good for?

Set the question aside for now. Keep walking.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. But Prediction Is Not Understanding

I want to stop here, because there is a trap you can easily fall into.

Predictive ability and understanding are not the same thing.

Let me give you an example.

A good weather forecaster can look at the shape of certain clouds, the pressure distribution, and the wind data, and predict tomorrow's rain. His prediction may be very accurate. But ask him "why do these clouds cause rain?" and he needs fluid dynamics, thermodynamics, and the phase-change dynamics of water vapor to answer. That is a different level of question — much deeper than "will it rain tomorrow."

Or a more extreme example: a trained neural network can learn, from a lot of historical data, that "when a certain pattern appears, a certain result follows," and then predict that result very accurately. But does it understand "why"? Does it know the causal mechanism behind it? This is a question we will not fully answer by the end of this book — but we will get closer and closer to it.

This distinction matters, because it tells us: **the accuracy of a prediction cannot stand in as a proxy for the depth of its understanding**.

A system can be extremely accurate at prediction while not understanding what it predicts at all. It has captured patterns in a statistical sense, but it has not built a causal model of the world. When the conditions change — when it meets something outside its training distribution — its predictions collapse. And the way they collapse is often unpredictable, and strange.

So what is real "understanding"?

That requires us to build a ladder.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. From Reflex to Reasoning: A Story of Levels

I want to introduce this ladder with an analogy.

Imagine learning to drive a car.

At first, every action is conscious: clutch, gear shift, steering wheel, gas pedal — you have to think about all four at once while also watching the road. This is a phase of enormous cognitive load.

A few months later, you drive to work while thinking about what to have for dinner. Changing gears, braking, turning — those actions have become automatic, having dropped from conscious reasoning into a kind of reflex.

But if a child suddenly runs out into the street, your consciousness snaps back in an instant, and you make a very non-automatic decision: which way to turn the wheel? Hard brake or swerve around? This is high-level reasoning happening in milliseconds.

Here three different levels of processing are running at the same time. They do not replace each other. They coexist, and get activated in different situations.

Now let me expand the analogy.

**Level 1: Reflex.** Input triggers output. No intermediate model. No representation of the world. The knee-jerk reflex: the doctor taps your knee with a little hammer and your leg kicks. You didn't think "my leg should kick." The whole loop closes at the spinal cord — the cerebral cortex doesn't even need to join in. Here there is input and output, but no reasoning.

**Level 2: Associative learning.** The system builds statistical associations between stimuli. Pavlov's dog: the bell rings, and saliva flows. There is learning here — the association was built through experience, not hard-coded. But it is still an association, not a model. The dog doesn't know "the bell means food is coming." It just executes a mapping that reinforcement has strengthened. If the food moves somewhere else, or the gap between bell and food changes, the mapping gets confused — but the dog cannot infer "why."

**Level 3: A generative model.** This is a fundamental shift. The system no longer passively waits for input and reacts. It actively maintains an internal representation of the world and continuously generates predictions about the input that is about to arrive.

The framework neuroscientists call **predictive coding** describes exactly this level. The brain's basic mode of operation is not bottom-up processing of sensory input — it is top-down generation of predictions, and then processing only the **errors** between prediction and actual sensation. You walk into a dark room. Your brain is not sitting in a blank state waiting for photons. It is already predicting what might be in the room, predicting where the chair is, predicting the texture of the walls. The prediction errors are the information that actually gets processed in bulk `-> [Sennesh et al., 2022, arXiv:2208.10601]`.

:::details Predictive Coding: The Brain Is Not a Camera (Prior Work: Rao & Ballard, 1999, and Others)
The traditional intuition is that the brain is like a camera: light comes in, the visual cortex processes it, and you see the image. **Predictive coding** overturns that picture.

The framework says: the brain's main job is to keep generating predictions from the top down, and then pass only the "part that got wrong" (the prediction error) upward, for higher levels to update their model. Much of what you "see," most of the time, is partly filled in by your own brain.

One piece of evidence: visual illusions. Even when you know an image is an illusion, you can't stop being fooled — because your higher-level model has already locked in a prediction, and it is more stubborn than a single bout of sensory input.

This theory also explains why the brain needs only about 20 watts: most of the time it only has to process the small part of the signal that went "wrong," not the whole stream of raw sensation.
:::

This architecture explains a lot: why you barely notice what you expected and go on high alert for surprises; why you can still make out someone's words in a noisy room (your model helps fill in the missing information); why visual illusions persist even when you know they're illusions — because your higher-level model has locked in a prediction, and it is harder to overturn than a single bout of sensory input.

**Level 4: A causal model.** A generative model may still only be capturing statistical correlation, not causal mechanism. A true causal model can answer three levels of question — Pearl's ladder of causation: if there are dark clouds in the sky, what is the chance of rain (observation)? If I artificially make dark clouds, will it rain (intervention)? If it hadn't been cloudy that day, would yesterday's rain still have happened (counterfactual)? These three questions need three completely different abilities, and pure statistical correlation can answer only the first. We'll go deep into this distinction in Chapter 6.

**Level 5: Meta-reasoning.** The ability to reason about one's own reasoning. Knowing where your own model is reliable and where it is not. A child finishes a math problem, feels it might be wrong, and recomputes — that child is monitoring their own reasoning process. A scientist asks "does this experiment rule out the alternative hypotheses?" — the scientist is reasoning about whether his own reasoning methods are adequate.

We don't know where today's AI systems stand on this ladder. This question will run through the whole book.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. Information Has a Cost, and Reasoning Pays It

Before we go on, I want to insert a physicist's interlude.

In 1961, IBM physicist Rolf Landauer published a paper arguing something that sounded odd at the time: **erasing one bit of information inevitably releases at least $k_B T \ln 2$ of heat into the environment**. Here $k_B$ is Boltzmann's constant and $T$ is the temperature of the environment.

This result is called the Landauer principle. Its meaning: **information is not free**. Processing information — especially erasing it — has an unavoidable physical cost. There is a deep connection between information and thermodynamics `-> [Chattopadhyay et al., 2025, arXiv:2506.10876]`.

:::details The Landauer Principle: How Much Energy Does Deleting One Bit Cost? (Prior Work: Landauer, 1961)
The **Landauer principle** says: physically, erasing one bit of information requires releasing at least $k_B T \ln 2$ joules of heat into the environment (at room temperature, about $3 \times 10^{-21}$ joules — extremely small, but not zero).

$k_B$ is Boltzmann's constant (about $1.38 \times 10^{-23}$ J/K), $T$ is absolute temperature in kelvins, and $\ln 2 \approx 0.693$. The exact number doesn't matter. What matters is: **this number is not zero**.

Why does that matter? Because it means information processing carries a physical cost. Reasoning is not a cost-free abstraction. It is a real physical process. The principle has since been confirmed experimentally, and it also resolved the famous "Maxwell's demon" paradox that had troubled physicists for nearly a hundred years (see the main text).
:::

This resolves a paradox that troubled physicists for nearly a hundred years: Maxwell's demon. In 1867, Maxwell proposed a thought experiment. A little demon sits on a partition between two boxes. It watches the speed of each gas molecule, and lets the fast ones pass to one side while keeping the slow ones on the other. Slowly, one side gets hot and the other cold — as if you could create a temperature difference with no energy cost. This breaks the second law of thermodynamics.

Landauer pointed out: to process each molecule, the demon must get rid of the previous molecule's information — and erasing information requires releasing heat. That heat exactly compensates for the temperature difference the demon was trying to exploit. The demon is not a winner. It is just paying the bill in a hidden way.

Why tell this story here?

Because it tells us: **any system that reasons pays a physical cost for it**. Every bit of information processing, every model update, every computation of a prediction error — these are real physical events with thermodynamic costs. Reasoning is not free. The universe does not allow free reasoning.

And one profound corollary: the reasoning architecture of biological systems, under evolutionary pressure, should be extremely energy-efficient. The brain's predictive coding architecture — processing only prediction errors rather than raw input — is close to optimal, in both information theory and thermodynamics. Your brain uses about 20 watts, roughly the power of a dim lightbulb, but the computation it performs cannot be matched at the same energy cost by any existing AI system. This is no accident. It is the result of four billion years of evolutionary pressure.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. Reasoning Needs a Starting Point

One more step forward.

Bayesian inference is the most precise framework we have for "how to update a belief given evidence":

$$
P(h \mid e) = \frac{P(e \mid h) \cdot P(h)}{P(e)}
$$

You have a hypothesis $h$ about the world, you observe evidence $e$, and you update your belief in $h$ to get the posterior probability $P(h \mid e)$.

:::details Bayes' Rule: How to Read This Equation
The formula says: **the probability that hypothesis $h$ is true after seeing evidence $e$** (the posterior probability) equals a combination of three quantities:

- $P(h)$: the **prior probability** — your initial belief in the hypothesis before seeing any evidence (for example, "the chance this coin is fair is 50%")
- $P(e \mid h)$: the **likelihood** — if hypothesis $h$ is true, the probability of observing evidence $e$ (for example, "if the coin is fair, the chance of 3 heads in a row is 1/8")
- $P(e)$: the **normalization constant** — the total probability of observing $e$ across all possible hypotheses, which makes sure the result is a legal probability

Intuition version: **new belief = old belief × support from the evidence ÷ normalization**

The core idea: you don't start inference from zero. You start from existing beliefs and update them with new evidence. The stronger the prior, the more evidence needed to move it.
:::

But this formula has a $P(h)$ in it — the prior. It is the belief you hold about $h$ before seeing evidence. Bayesian updating never starts from nothing. It always starts from a prior.

Where does the prior come from?

In practice, the prior comes from past experience, from earlier inferences, from domain knowledge — and, for biological systems, from built-in assumptions laid down by evolution. But trace it all the way back, and you always hit some initial belief that was accepted without argument.

This is not a flaw. It is a structural feature of reasoning itself: **you have to begin somewhere, and the place you begin cannot be fully proven by your own reasoning process.**

For biological systems, evolution solved this problem — at least partly: individuals holding "good priors about the world" survived, reproduced, and passed their nervous-system structures to their descendants. Natural selection is a meta-optimization process over the fitness of priors.

But notice one subtlety: this optimization process is not itself reasoning. Natural selection is blind random variation plus screening pressure. It doesn't know what it's doing. It just keeps whatever happens to work. The foundation of biological reasoning ability was optimized by a process that does not reason.

That is something worth stopping to think about.

For machine learning systems, training data plays a similar role. The model distills statistical regularities out of the data, and those regularities become the implicit priors of its reasoning. But training data is not neutral — it comes from a particular distribution, carries particular assumptions, and reflects a particular world. When the system meets something outside its training distribution, where do those implicit priors break? And in what way?

**Every reasoning system has one or more anchors that its own reasoning process cannot fully reach.** You can update your priors. But the rules of updating rest on a deeper meta-prior you have not updated. You can question your assumptions. But the way you question rests on a set of logic rules you never questioned. This is a recursion with an end — and the end is some starting belief held in suspension.

The quality of your anchor decides how far your reasoning can go.

We will come back here, in more formal language, in Chapter 15. For now, just remember this sentence.

<div class="center">

------------------------------------------------------------------------

</div>

## 8. A Small Pause

We've covered a lot. Let me stop here and sort out the threads of this chapter.

We started from the second law of thermodynamics: the universe runs toward disorder, and living creatures swim against the current, exporting entropy to the outside to keep their own order. But keeping static order is not enough — the outside world is dynamic, changing, sometimes hostile. So living things need **prediction**: they must respond before the change arrives.

Predictive ability comes in levels: from the simplest time difference, to associative learning, to generative models, to causal reasoning, to meta-reasoning. Each level is more powerful than the one before — and more expensive. The Landauer principle tells us that information processing carries a physical cost; reasoning is not free.

And all reasoning needs a starting point: a prior anchor that reasoning itself can never fully prove. For living things, that anchor comes from evolution. For machines, from training data. Neither is perfect, and both carry their own blind spots.

These three limits — **cost, levels, anchor** — are the base from which to understand reasoning.

<div class="center">

------------------------------------------------------------------------

</div>

## 9. The Physical Reasoning of Life: The Cost and Lesson of Entropy Reduction

Before closing this chapter, let's go back to the start — but deeper.

The "negative entropy" idea from Schrödinger's *What Is Life?* reveals a deep physical fact: life does not break the second law of thermodynamics. Life is a special kind of system that **uses energy gradients to achieve local order**. That local entropy reduction is paid for with a bigger entropy increase in the environment — living systems keep taking energy and matter (negative entropy) from the outside, sustaining their own ordered structure while dumping more disorder (waste heat, waste matter) into the environment.

### Dissipative Structures: Order Away from Equilibrium

The Belgian physicist Ilya Prigogine's **theory of dissipative structures** provides the mathematical framework: away from thermodynamic equilibrium, an open system that keeps energy and matter flowing through it can spontaneously form and sustain ordered structure. Life is the ultimate expression of a dissipative structure.

This physical mechanism carries important lessons for reasoning systems:

1. **The analogy between energy flow and information flow**: life needs an energy flow to sustain physical order; a reasoning system needs an information flow to sustain "cognitive order." Without continuous information input (perception, learning), a reasoning system suffers "cognitive decay" — just as life dies without energy input.

2. **The cost of local order**: life's local entropy reduction is paid for by environmental entropy increase; AI's "understanding" is paid for by computational complexity. Both express the same founding principle: **order must be paid for**.

3. **Self-organization and emergence**: the origin of life is a self-organization phenomenon driven by energy flow in the universe — ordered structures forming in local regions under entropy's law through energy input. Similarly, "understanding" in AI systems may be a self-organizing emergent phenomenon in the flow of data.

### From Physical Reasoning to Cognitive Reasoning

This angle shows us the **physical foundation** of reasoning:

- **Physical layer**: life systems fight entropy increase through energy flow
- **Information layer**: cognitive systems fight "cognitive entropy increase" (chaos, ignorance) through information flow
- **Computational layer**: AI systems achieve pattern recognition and prediction through computation flow

The three layers share one **meta-pattern**: each needs continuous outside input (energy/information/data), each buys local order at some cost (environmental entropy increase / computational complexity), and each can, under the right conditions, emerge into complex behavior.

<div class="center">

------------------------------------------------------------------------

</div>

## Exercises: What Is the "Energy" of Reasoning?

If life needs physical energy to survive, what "energy" does reasoning need to keep working?

Possible answers: information diversity (avoiding overfitting), computational resources (for complex inference), time (the learning process), cognitive load (attention allocation)… None of these is free. Each has its own "thermodynamic cost."

<div class="center">

------------------------------------------------------------------------

</div>

Starting in Chapter 2, we'll trace history: when humans first tried to turn reasoning into a mechanical process, what did it look like? Where did those early attempts succeed, where did they fail, and what did they leave behind?

Reasoning needs representation. In the next chapter, we'll see how humans used symbols and rules to build the first generation of reasoning machines — and why they finally failed.

<div class="center">

------------------------------------------------------------------------

</div>

:::details Thought Echoes: From Thermodynamics to Machine Learning
Readers who have come this far might ask: what is the real relationship between the entropy in thermodynamics and the entropy in information theory?

The answer: their mathematical forms nearly coincide, but their physical meanings are independent.

In 1948, when Shannon founded information theory, he asked John von Neumann what to call the uncertainty of information. Von Neumann said: "Just call it entropy. Nobody really understands thermodynamics, and you'll always win the argument anyway."

The advice was taken. So we now have:

- Thermodynamic entropy: $S = -k_B \sum p_i \ln p_i$ — describes the disorder of a physical system
- Information entropy: $H(X) = -\sum p(x) \log_2 p(x)$ — describes the uncertainty of information
- Cross entropy: $H(p, q) = -\sum p(x) \log q(x)$ — describes the difference between two distributions

When you train a neural network by minimizing cross entropy, you are doing two things:

1. **Measuring the gap**: computing the "distribution distance" between the model's predictions $q(x)$ and the real world $p(x)$
2. **Fighting disorder**: using gradient descent to move $q(x)$ from "a chaotic state far from $p(x)$" toward "an ordered state close to $p(x)$"

This echoes this chapter's core claim: **reasoning is a survival strategy against entropy increase**. In the physical world, life fights thermodynamic entropy increase through metabolism. In the information world, intelligent systems fight the entropy increase of prediction distributions through learning.

The similarity in form is no coincidence. It reveals that when different fields face the same fundamental challenge — going from disorder to order — they converge on similar mathematical structures.
:::

## Open Questions

- How are the brain's predictive coding architecture and the reasoning structure of large language models similar and different, at bottom?

- What does the Landauer principle mean in practice for AI systems — are we approaching the thermodynamic limit of reasoning?

- Evolution picked "good priors" for living things, but evolution itself does not reason — this means the foundation of biological cognition was optimized by a blind process. How reliable is that foundation?

- **Life as an open system**: if life achieves local entropy reduction through energy flow, through what kind of "flow" do AI systems achieve "cognitive order"? Information flow, data flow, or computation flow? And what is the thermodynamic cost of these "flows"?

- **Dissipative structures and AI**: Prigogine's theory of dissipative structures describes how systems far from equilibrium maintain order. Can AI training (gradient descent, parameter updates) be seen as a kind of "cognitive dissipative structure"? Is a trained, stable model sitting in a "cognitive equilibrium state"?

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Build a Minimal Bayesian Predictor

This chapter said three things: reasoning has a cost, reasoning has levels, and reasoning needs an anchor. Now it's your turn to turn those three into something that runs.

You don't have to write a big system. Use a minimum of code, and feel with your own hands where those three limits bite.

<div class="center">

------------------------------------------------------------------------

</div>

### Step 1: Decide What You'll Predict

Pick a domain where you can get a sequence of data — the simpler, the better.

A few suggestions:

- Flip a coin you're not sure is fair, and record the heads/tails sequence
- Watch a website's hourly visitor count, and encode it as "high traffic / low traffic"
- Record your own mood through one day (good / okay / bad), for several days in a row

**Your first question:** what is the "prior" of this sequence? Before you see any data, what do you assume about the sequence's distribution? Write it down — even a statement as simple as "I think heads and tails are each 50%."

<div class="center">

------------------------------------------------------------------------

</div>

### Step 2: Implement the Bayesian Update

No external libraries. Use the formula from Section 7:

$$
P(h \mid e) = \frac{P(e \mid h) \cdot P(h)}{P(e)}
$$

The logic you need to implement is:

```python
import numpy as np
import matplotlib.pyplot as plt
# --- Step 1: define the hypothesis space ---
# The coin's head probability theta takes the following discrete values
hypotheses = np.arange(0.1, 1.0, 0.1)   # [0.1, 0.2, ..., 0.9]
# --- Step 2: assign prior probabilities ---
# Uniform prior: every theta is equally likely at the start
prior = np.ones(len(hypotheses)) / len(hypotheses)
# --- Step 3: Bayesian update function ---
def bayesian_update(prior, hypotheses, observation):
    """
    Update the posterior distribution after a single observation.
    observation: 1 = Head, 0 = Tail
    """
    # a. likelihood of this observation under each hypothesis, P(e | h_i)
    if observation == 1:
        likelihoods = hypotheses           # Head: likelihood = theta
    else:
        likelihoods = 1.0 - hypotheses    # Tail: likelihood = 1 - theta
    # b. unnormalized posterior: P(e | h_i) × P(h_i)
    unnormalized = likelihoods * prior
    # c. normalize: divide by the sum so probabilities add to 1
    posterior = unnormalized / unnormalized.sum()
    return posterior
# --- Step 4: simulate the observation sequence and update step by step ---
# Example: a manually entered sequence (1 = Head, 0 = Tail)
observations = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                1, 1, 0, 1, 1]
top_hypothesis_history = []   # index of the highest-probability hypothesis
top_posterior_history  = []   # value of the highest posterior probability
current_prior = prior.copy()
for i, obs in enumerate(observations):
    current_prior = bayesian_update(current_prior, hypotheses, obs)
    # record the best hypothesis so far
    best_idx = np.argmax(current_prior)
    top_hypothesis_history.append(hypotheses[best_idx])
    top_posterior_history.append(current_prior[best_idx])
    print(f"obs {i+1:2d}: {'Head' if obs==1 else 'Tail'} | "
          f"best theta = {hypotheses[best_idx]:.1f} | "
          f"posterior = {current_prior[best_idx]:.3f}")
# --- Plot: posterior of the best hypothesis vs. observation number ---
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(observations) + 1), top_posterior_history,
         marker='o', linewidth=2)
plt.xlabel("observation number")
plt.ylabel("posterior of the best hypothesis")
plt.title("Bayesian update: posterior converging with observations")
plt.ylim(0, 1)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
```

After running it, look at the line chart: the x-axis is observation number, the y-axis is the posterior of the currently "most likely hypothesis."

**Your second question:** how many observations does it take before the posterior starts to stabilize? And how does that number relate to your choice of prior?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 3: Feel the Levels of Prediction

Use the Bayesian update you implemented, and make predictions at three different levels:

**Level 1 (reflex level):** what is the most likely next observation? Just output the prediction of the current highest-probability hypothesis.

**Level 2 (generative model level):** don't just predict the next one — predict the expected distribution for the next 10 observations. Your Bayesian machine already contains a model of "what parameter controls this sequence" — use it to generate a 10-step expectation.

**Level 3 (meta-reasoning level):** how confident are you in your own prediction? What is the entropy of your posterior? The more uniform the posterior (higher entropy), the more uncertain you are over the hypothesis space — and the more conservative your predictions should be.

```python
import numpy as np
def posterior_entropy(posterior):
    """
    Shannon entropy of the posterior distribution.
    Higher entropy = more uncertainty over hypotheses; lower = sharper belief.
    posterior: numpy array of posterior probabilities over hypotheses
    """
    # drop zero-probability entries (log(0) is undefined)
    p = posterior[posterior > 0]
    # Shannon entropy: H = -sum P(h_i) * log(P(h_i))
    entropy = -np.sum(p * np.log(p))
    return entropy
# Example: suppose the current posterior is
example_posterior = np.array([0.05, 0.1, 0.2, 0.4, 0.15, 0.05, 0.03, 0.01, 0.01])
h = posterior_entropy(example_posterior)
print(f"posterior entropy = {h:.4f} (closer to 0 = more certain; max ~ {np.log(len(example_posterior)):.4f})")
```

**Your third question:** in what situations should the meta-information from Level 3 ("I'm not sure") override the direct prediction from Level 1? Can you find a concrete example?

<div class="center">

------------------------------------------------------------------------

</div>

### Step 4: Find Your Anchor

Now go back to what Section 7 said: "Every reasoning system has an anchor that its own reasoning process cannot fully reach."

For your Bayesian machine, the anchor is the prior.

Run an experiment: run the same observation sequence through three different priors — a uniform prior, a prior strongly biased toward one hypothesis, and a completely wrong prior.

Record:

- with the three priors, how many observations does the posterior need to reach "almost the same" results?
- if the data is limited (say, only 20 observations), how far apart are the final beliefs from the three priors?

**Your fourth question (and the hardest of this exercise):** if your prior is completely wrong, can Bayesian updating fix it? How much data does it take? And what does this mean for an AI system pretrained on wrong training data?

<div class="center">

------------------------------------------------------------------------

</div>

### Check Your Work

You don't need perfection. You need:

- a complete loop of "prior -> observation -> posterior update" that actually runs

- the ability to say clearly what your prior is, and why you set it that way

- one case that breaks your system — a situation your hypothesis space does not cover

The third point is the most important one. Expert systems failed because they hit the boundary of their rule base. Where will your Bayesian machine hit its own boundary?

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- [Friston et al., 2020] — The free energy principle: Markov blankets, Bayesian mechanics, and approximate inference, arXiv:2002.04501

- [Sennesh et al., 2022] — A time-averaged formalization of active inference, connecting optimal control with neurophysiology, arXiv:2208.10601

- [Chattopadhyay et al., 2025] — A review of the Landauer principle and computational thermodynamics, arXiv:2506.10876

- [Tlusty, 2010] — Rate-distortion theory of biomolecular codes: life as an information channel, arXiv:1007.4471

- Schrödinger, E. (1944). *What is Life?* — life and negative entropy, the philosophical starting point of this chapter's ideas

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why* — a lay account of the ladder of causation; required reading before Chapter 6

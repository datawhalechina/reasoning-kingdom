# Chapter 16: What Is True Reasoning? (The Myths of LLMs)

:::info
**A Letter from Mr. Pallas's Cat**  
Dear Piglet and Little Seal:

Spring has returned to Guangzhou. The kapok blossoms are in full bloom, and the moist breeze from the Pearl River carries warmth into the study of the Black Stone House.

This is the first letter of Part 4. In the first three parts, we journeyed through technology — from the simplest Boolean logic to the most complex Transformer.

But in Part 4, I want to try a different approach. No more technical details — instead, **letters**. Like friends chatting, let's talk about what lies behind the technology.

Today's topic: **do these neural networks truly "reason"?**

Or are they merely imitating the traces we've left behind, repeating human wisdom like echoes?

This question matters. Because if we don't know what AI is doing, we won't know how to live with it.

Don't rush. Let's chat slowly. First, brew a pot of tea and watch the kapok petals drifting outside the window.
:::

---

## The "Verbose" and the "Precise" of Chatbots

Piglet, I remember you asked me last time: "Professor, why does AI sometimes answer quickly and accurately, but other times ramble on without getting to the point?"

You gave a great example: ask AI to write quicksort code, and it writes beautifully. But ask it "why is quicksort faster than bubble sort," and it becomes like a student reciting from memory — spouting a bunch of time complexity analysis without explaining that core intuition — the beauty of divide-and-conquer, the elegance of recursion.

Little Seal, you added at the time: "It's like someone who can recite the entire dictionary but doesn't necessarily know how to write poetry."

Well said.

I sit by the window of the Black Stone House, watching students walk through Kangle Garden. Some are in heated discussion; some are thinking quietly. I wonder: **what is the difference between discussion and thinking?**

AI's "discussion" is statistics-based — it has seen countless conversations and knows what people usually say on a given topic. But true "thinking" requires understanding the connections between concepts, requires causal reasoning, requires asking "why."

## Core Concept: Probabilistic Prediction vs. Logical Reasoning

Little Seal, you've read Hume, haven't you? The philosopher who questioned inductive reasoning. He asked: on what grounds do we believe the sun will rise tomorrow? Merely because it has risen every day in the past?

LLMs are essentially doing **massive-scale induction**. They learn from vast amounts of text: "Paris" is usually followed by "France"; "2+2" is usually followed by "4." This is **probabilistic prediction** — guessing the future based on history.

But human reasoning goes beyond this.

When we say "All humans are mortal; Socrates is human; therefore Socrates is mortal," this is not prediction — it is **logical deduction**. If the premises are correct and the reasoning is valid, the conclusion must be correct.

Piglet, you might ask: "Can't AI do logical deduction?"

It can imitate it. When you give it a syllogism, it may answer correctly. But if you give it contradictory premises — "All birds can fly; penguins are birds; penguins cannot fly" — it might fail to detect the contradiction. Because it processes **statistical relationships between symbols**, not **the meaning of symbols**.

This is the famous "symbol grounding problem." AI knows that "bird" and "fly" often appear together, but it doesn't know what "bird" refers to, doesn't know what "fly" means. Its knowledge is internal to text, not world-directed.

## A Gentle Reminder from Bayesianism

Little Seal, I remember you're quite interested in Bayesianism. Bayes offers a gentler way of reasoning: don't pursue absolute certainty — instead, **continuously update beliefs with new evidence**.

The formula is simple: new belief = old belief × strength of new evidence.

LLM training closely resembles Bayesian updating — using massive data to update language model parameters. But Bayesian reasoning is still probabilistic — it tells you "this is very likely," not "this must be."

This brings to mind an analogy: **Bayesians are humble explorers**. They acknowledge their ignorance and are willing to be changed by evidence. **Logicists are rigorous architects** — they start from axioms and construct necessary truths.

Perhaps true wisdom lies in knowing when to be humble and when to be rigorous.

## Reflection: Can You Give an Example of AI "Predicting" Rather Than "Reasoning"?

Piglet, the example you gave last time was excellent: "If all birds can fly, and penguins are birds, can penguins fly?"

Some AIs would answer "can fly," because they matched the high-frequency pattern "bird -> fly." Even if they answer "cannot fly," it may only be because they matched the fact "penguins cannot fly," not because they truly performed logical reasoning.

True reasoning should go like this:
1. From "all birds can fly" and "penguins are birds," deduce "penguins can fly"
2. But we know "penguins cannot fly"
3. Discover the contradiction
4. Revise the premise: "not all birds can fly" or "penguins are not birds"

This is a **metacognitive** process — thinking about one's own thinking, discovering contradictions, revising beliefs.

AI usually stops at step 1's imitation.

## Three Perspectives on Reasoning

Little Seal, you found a good framework from cognitive science. Let me restate it in more everyday terms:

1. **Logical perspective**: like the rules of chess. Rooks move only in straight lines; knights move in L-shapes. Rigorous, but rigid.

2. **Probabilistic perspective**: like weather forecasts. 70% chance of rain tomorrow. Flexible, but no guarantees.

3. **Heuristic perspective**: like intuitive judgment. "This person seems trustworthy." Fast, but error-prone.

LLMs mainly embody the probabilistic perspective, sometimes imitate the logical perspective, and barely understand the heuristic perspective.

But what is most precious about humans may be **the ability to flexibly switch between these three perspectives**: intuition for simple problems, probability for routine problems, logic for complex ones.

## To You: Future Reasoning Scientists

Piglet, Little Seal:

As I write this, a few more kapok petals have drifted past the window. Spring is always like this — beautiful and brief.

What I want to tell you is: **don't be dazzled by AI's dazzling performance**.

It can write poetry, but doesn't necessarily understand poetic meaning.
It can solve problems, but doesn't necessarily understand the problems.
It can converse, but doesn't necessarily understand conversation.

True reasoning requires understanding causal relationships, maintaining logical consistency, asking "why," admitting "I don't know," and being willing to be changed by evidence.

These are things AI is still learning — and the most precious things about being human.

The next time AI answers you, try asking yourself: is this true reasoning, or just pattern matching? This very ability to distinguish is itself a mark of wisdom.

We'll take it slow. Understanding is what matters most.

Happy spring.

Yours,
Mr. Pallas's Cat
Black Stone House, Sun Yat-sen University
In the season of kapok blossoms

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Observe AI's "thinking"**: find three AIs you're familiar with and ask them the same logical question. Compare their answers. Can you tell whether they are reasoning or pattern-matching?
2. **Design a "reasoning test"**: design 5 questions specifically testing logical reasoning ability rather than knowledge recall. How would you design them?
3. **Record your own thinking**: next time you solve a complex problem, record your thought process. What kinds of reasoning did you use?

### Historical Investigation (for Little Seal)
1. **The evolution of reasoning**: from Aristotle to Bayes, how has humanity's understanding of "reasoning" changed? What problem did each change solve?
2. **A brief history of AI reasoning**: investigate the paradigm shifts in AI reasoning. Why did logicism fail? Has probabilism succeeded?
3. **Eastern and Western views of reasoning**: compare conceptions of reasoning in Eastern and Western philosophy. What's similar? What's different?

### Integrated Reflection
1. **If AI truly could reason**: if one day AI genuinely possesses logical reasoning ability, what would that mean? Would we trust it more, or fear it more?
2. **The future of education**: in the age of AI, what should we teach children? Is memorizing knowledge more important, or cultivating reasoning ability?
3. **Human uniqueness**: if AI can surpass humans on many tasks, is "reasoning" still a uniquely human advantage? What is truly irreplaceable about humans?

> **Piglet's note**: The Professor's letter made me think a lot. I realized that I sometimes do "pattern matching" too — when I encounter a problem, I search for similar solutions rather than truly understanding the principles. From today on, I'll ask "why" more often.
>
> **Little Seal's note**: The Professor mentioned the humility of Bayesianism. This reminds me that true wisdom is not knowing a lot, but knowing what you don't know. The first step of reasoning may be admitting ignorance.
>
> **Mr. Pallas's Cat's closing words**: The essence of reasoning may not be finding the right answer, but maintaining the courage to keep asking. In an age when AI can provide countless answers, those who know how to ask questions may be the most precious of all.

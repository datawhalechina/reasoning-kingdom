# Chapter 2: The Dawn of Symbols — The First Modeling of Causality

> If you could write the world as If-Then, you would own the world. The problem is, you can't finish writing it.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. A Bet

In 1965, Herbert Simon said a sentence that would be quoted for almost sixty years after:

> "Within twenty years, machines will be able to do any work a human can do."

Simon was a Nobel laureate in economics, one of the founders of cognitive science, and an extremely smart person. He wasn't talking nonsense. He had evidence. He had seen the most thrilling thing of his era: a machine, for the first time, beginning to **reason** like a human.

Not compute. Reason.

Logic Theorist — written by Newell and Simon in 1956 — used a set of axioms and inference rules to automatically prove 36 of the 38 theorems in Chapter 2 of Russell's *Principia Mathematica*. One of its proofs was even more concise than Russell's own version. Simon wrote to a friend at the time: "We have invented a thinking machine."

From that moment, for about thirty years, the mainstream paradigm of artificial intelligence stood on one bet so simple it bordered on beautiful:

**If you can represent the world's knowledge as symbols and its reasoning rules as logic, then a machine can reason.**

This chapter's story is about how that bet was made, where it won, and why it eventually lost.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Aesthetics of If-Then

Let me first explain what the core of this paradigm is.

A rule-based reasoning system has a very simple structure. You have two parts:

**Knowledge Base**: a set of If-Then rules describing the world's regularities. **Inference Engine**: given a question, it derives the answer along chains of rules.

Classic example — MYCIN, built at Stanford in 1972. MYCIN diagnosed bacterial blood infections. Its knowledge base held about 600 rules that looked like this:

```
IF  Gram stain of bacterium = positive
AND  shape of bacterium = spherical
AND  growth pattern of bacterium = clustered
THEN species of bacterium = Staphylococcus aureus (confidence 0.7)
```

The inference engine takes a concrete case, plugs the known observations into the rules, and pushes forward, rule after rule, until it reaches a diagnosis and a recommended treatment.

MYCIN's performance was stunning. In 1979, Stanford ran a blind test — 10 real cases were given to MYCIN and to 5 medical specialists, and then 8 infectious disease authorities graded the quality of all the answers without being told which one came from a machine. The result: MYCIN ranked in the **front ranks** of all participants, ahead of several human doctors.

That was 1979.

You can see why Simon's generation was so optimistic. If you can write an expert's knowledge of a domain into rules, the machine becomes an expert in that domain. And it never gets tired, never makes careless mistakes, never loses its judgment after a night shift.

This idea has a kind of **pure aesthetics**. The world is complex, but behind the complexity there are regularities. The regularities can be explicitly expressed. Once expressed, a machine can operate on them.

The problem is that "writing it into rules" is much harder than it looks.

<div class="center">

------------------------------------------------------------------------

</div>

## 3. The Knowledge Bottleneck

Expert systems had a problem that gave everybody a headache. They gave it a name: the **Knowledge Acquisition Bottleneck**.

:::details Knowledge Acquisition Bottleneck: Why Is Expert Knowledge So Hard to Write as Rules?
"Knowledge acquisition bottleneck" means: **turning the knowledge in a human expert's head into rules a computer can process is itself extraordinarily hard.**

There are two layers of reason:
1. **Experts can't say what they do**: much expert judgment has been automated down into the unconscious; the expert can't describe it explicitly anymore (see "tacit knowledge" below).
2. **Knowledge depends on context**: the same piece of information means different things in different situations, and rule systems can't handle that dependency gracefully.

This was the most central practical difficulty of the whole expert system era, and it ultimately brought the technology down.
:::

In other words: writing an expert's knowledge into rules is, by itself, almost an impossible task.

Go find a surgeon with thirty years of experience and ask: "How do you decide a patient needs surgery?" You'll get standard answers: look at the imaging results, look at the blood indicators, look at how long the symptoms have lasted. But when you write all of that into rules, the system performs badly. Because what he didn't tell you are the thirty years of **intuition**: the way he frowns when the patient says a certain thing, that hard-to-name "feeling" when two indicators show up together, the specific patterns of a certain kind of patient in a certain season.

These things are real knowledge. They are what makes him a top-tier surgeon. But he can't say them. Not because he doesn't want to, but because that knowledge **has settled into the bottom of his cognitive system** — he can't access an explicit form of it himself.

Cognitive science calls this **tacit knowledge** — you know how to ride a bicycle, but you can't explain to someone who can't ride what precise mechanical adjustments you're making while riding.

:::details Tacit Knowledge: The Part of Knowledge You Can't Put Into Words
**Tacit knowledge** is a concept from the philosopher Michael Polanyi, who argued in 1958: **"We know more than we can tell."**

Typical examples:
- Riding a bicycle — you can ride, but can you describe, frame by frame, every adjustment you're making?
- Face recognition — you recognize an old friend at a glance, but can you say which pixel features triggered the recognition?
- Expert judgment — a traditional Chinese physician "reads the complexion at a glance"; an experienced surgeon "feels something is off"

The existence of tacit knowledge is why expert systems were destined to hit the "knowledge that can never be fully written down" problem: the most valuable part of expert experience is precisely the part the experts themselves can't describe explicitly.
:::

So the life of a knowledge engineer (the person who builds the knowledge base) was miserable. They spent enormous time interviewing experts, got some rules, put them into the system, tested, found uncovered edge cases, went back to ask again, modified, re-tested. The process was slow, expensive, and had no end.

You wrote 1,000 rules, and found you needed another 1,000. You wrote 10,000, and found new cases triggering situations that the previous 1,000 rules had never anticipated.

The world is not closed. Rules can never be fully written.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. Logic Gates Are the Simplest Form of Causality

Before going on with the failure of expert systems, I want to insert a physics footnote here.

When we talk about "rule reasoning," we are actually talking about a very specific physical process.

A logic gate — an AND gate, an OR gate, a NOT gate — what does it do? Given inputs, it produces a deterministic output. This is the simplest, purest physical realization of a causal relationship.

An AND gate: if A=1 and B=1, then output=1. Otherwise output=0. That is one If-Then rule, etched into silicon, executed by the flow of electrons.

All expert systems and all logical reasoning, at the bottom, are physical implementations like this. Causality is made concrete in circuits: one state causes another state, with no exceptions, no probability, no ambiguity.

There is something profound here worth stopping to think about.

A computer is not "thinking." What it does is execute a sequence of deterministic causal transformations. At each step, input decides output, with no uncertainty. The intuition of McCarthy's and Minsky's generation was: if we can break a thought process down into enough deterministic steps, then a machine can reproduce thought.

That intuition is right in a way. A mathematical proof can be decomposed into a sequence of symbol operations, and every symbol operation is deterministic. MYCIN's reasoning is deterministic — given the same facts, it always gives the same answer. Chess move search is deterministic. A whole large class of "precisely defined problems" can be handled this way.

But "thought" is not just a collection of precisely defined problems.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. The Closed World Assumption

Here we need to formally introduce a concept, because it is the root of all this chapter's problems.

It is called the **closed world assumption** (CWA).

:::details The Closed World Assumption: What You Don't Know, the System Treats as Nonexistent
**Closed world assumption** is the default working mode of rule-based systems: **any fact not clearly recorded in the knowledge base is treated as "false" or "nonexistent."**

Compare two worldviews:
- **Closed world**: not written down = doesn't exist (the logic of database-style queries)
- **Open world**: not written down = I don't know (closer to reality)

This assumption is reasonable inside a closed, bounded domain (like a flight database). But in an open, real world it is dangerous: the system mistakes "I've never seen this situation" for "this situation doesn't exist," and then confidently gives a wrong answer.

Today's large language models — models trained on vast text to predict the next word — have a similar problem: the boundary of training data is, in a sense, a new form of "closed world." This question runs through this entire book.
:::

In a rule-based system, you can only use the knowledge you put in. Anything you didn't explicitly state, the system treats as "nonexistent" or "false."

Here's an example. You build a database with every flight you know. Someone asks "are there direct flights from Beijing to Mars?" Under the closed world assumption, since there is no such record in the database, the system answers: no. That answer is right.

But someone asks "is there a flight from Beijing to Shanghai this afternoon?" The database has none, so the system answers: no. And maybe the database is just incomplete — maybe the flight exists, you just didn't record it.

In both cases the system answers in exactly the same way, but one is a correct inference and the other is a wrong one. The closed world assumption cannot distinguish "explicitly nonexistent" from "unknown whether it exists."

In expert systems, this problem is worse. Your rule base covers 70% of the doctor's knowledge. What about the remaining 30%? The system won't say "I don't know." It will climb along its existing rules and derive an answer it can derive. That answer may be completely wrong — and the system is full of confidence about it.

This is a very dangerous failure mode.

It doesn't get confused. It just crashes — and crashes with confidence.

<div class="center">

------------------------------------------------------------------------

</div>

## 6. The Descendants of MYCIN

Let's continue the story.

After MYCIN, the expert system idea was pushed into almost every domain you could think of. Oil exploration (PROSPECTOR, 1978 — one of its cases found a molybdenum deposit worth about one hundred million dollars). Loan approval (plenty of banking systems). Production scheduling (DEC's XCON, saving the company tens of millions of dollars a year).

In the 1980s, this was a red-hot industry. There were companies selling expert system development tools. There was a dedicated engineering job called "knowledge engineer." Universities opened dedicated courses. In 1982, the Japanese government launched an ambitious "Fifth Generation Computer Systems" project, aiming to build a supercomputer based on logic reasoning, expected to break through to AI within ten years. The US government got scared and created DARPA's Strategic Computing Initiative to respond.

Sound familiar? Every few decades, this story happens once more.

Then, slowly, the cracks appeared.

Maintenance cost. In a large expert system, the interactions between rules become extremely complex. You modify one rule and find it breaks the reasoning chains of three others. The expert system starts to become brittle — not at the edge cases, but in its core region, where it should have been strongest, collapsing from internal contradictions.

Scalability. Going from 600 rules to 6,000 rules makes system behavior unpredictable. The knowledge engineer can no longer track, in his head, the interactions between all rules, let alone understand the integrated behavior after merging them.

Adaptability. MYCIN was written in 1972. Medicine moves on. New pathogens, new tests, new drugs. Every update requires the knowledge engineer to re-enter the system, find the affected rules, and edit them by hand — while being careful not to break anything else. This is not a system that can learn on its own. It is a **static snapshot** that needs humans to maintain it forever.

By the late 1980s, Japan's Fifth Generation project was in trouble. The limits of expert systems became clearer and clearer. In 1987, the expert system hardware market collapsed, and the AI industry entered its second "winter."

<div class="center">

------------------------------------------------------------------------

</div>

## 7. A Simple Expert System

So that this isn't just abstract history, let me show you what an actual miniature expert system looks like.

It's a small system that diagnoses "what animal is this" — first given in a 1981 textbook by Winston and Horn, a classic teaching example of expert systems:

The facts you input: `["has fur", "eats meat", "is tan", "has dark spots", "has claws", "forward-facing eyes"]`

The reasoning process:

```
R1 fired: ['has fur'] -> is a mammal
R5 fired: ['is a mammal', 'eats meat'] -> is a carnivore
R6 fired: ['is a mammal', 'has claws', 'forward-facing eyes'] -> is a carnivore  [already known]
R8 fired: ['is a carnivore', 'is tan', 'has dark spots'] -> is a cheetah

Final conclusion: is a cheetah
```

This logic is clear, traceable, and explainable. At every step you can say which rule it came from. This is one of the expert system's big strengths — **transparency**.

But now you ask: what if this animal has dark spots and black stripes at the same time?

The system doesn't know how to handle the situation. Both R8 and R9 are partly satisfied, but the two rules give different conclusions — a cheetah or a tiger? The outcome depends on the implementation: sometimes it reports the first matching conclusion, sometimes it falls into a conflict, sometimes it produces two contradictory conclusions.

**It doesn't get confused. It doesn't say "I don't know." It gives you an answer.**

And that answer might be utterly meaningless.

<figure>
<p><img src="/figures/ch02_fig1_reasoning_chain.png" alt="Reasoning chain: the forward-chaining derivation for the cheetah case" /></p>
<figcaption>Reasoning chain: the forward-chaining derivation for the cheetah case</figcaption>
</figure>

*Figure 1: The DAG structure of MYCIN-style forward-chaining reasoning. Each node is a fact or conclusion; each edge is a rule that fired. The reasoning path is clear and traceable — the expert system's biggest strength, and its ceiling.*

<div class="center">

------------------------------------------------------------------------

</div>

## 8. Why It Failed — The Real Reason

Now I'll say something deeper than "the knowledge can never be fully written."

The core assumption of expert systems is: **reasoning can be decomposed into independent, enumerable rules, and the interactions between rules are finite and predictable.**

That assumption holds inside very restricted domains. MYCIN faced bacterial infection diagnosis — a domain with clear borders, standard protocols, and a finite set of variables. There, the knowledge base is close to complete and the reasoning close to closed.

But in the open, real world, the assumption is fundamentally wrong.

Human reasoning is not a rule system. When you walk into a room you've never seen, you don't access a rule base about "rooms" and go through it rule by rule. What your brain does is map the new scene into an internal, highly compressed world model — a model built over decades, updated through countless experiences. This model is **continuous** and **generative**. It can interpolate, it can extrapolate, and it can produce reasonable guesses on new situations, even when the situation never appeared among its "seen" samples.

Rule systems don't have that ability. Rule systems are **discrete** and **enumerative**; they can only reason inside the regions explicitly covered. When they hit the boundary of the rule base, they don't guess. They crash.

There is a deeper philosophical issue here. In 1972, the philosopher Hubert Dreyfus raised it in his book *What Computers Can't Do*: human intelligence depends, to a large extent, on our intuitive sense of "what matters" — and that sense is **situated**, built from experiences we accumulated as embodied creatures living in the world. You cannot extract situated knowledge into rules detached from situations, because it does not exist in rule form in the first place.

The AI community laughed at Dreyfus at the time. It turned out he was mostly right.

<figure>
<p><img src="/figures/ch02_fig2_knowledge_boundary.png" alt="Knowledge boundary: the coverage of the closed world assumption and its blind spots" /></p>
<figcaption>Knowledge boundary: the coverage of the closed world assumption and its blind spots</figcaption>
</figure>

*Figure 2: The boundary between the region the knowledge base covers (known) and what lies beyond (unknown). Inside the known region, the expert system reasons precisely. At the boundary, it doesn't say "I don't know" — instead it follows the nearest known rule and gives a confidently wrong answer. The closed world assumption equates "not in the base" with "doesn't exist."*

<div class="center">

------------------------------------------------------------------------

</div>

## 9. What It Left Behind

I don't want this chapter's ending to sound like a failure story. Because that isn't the whole picture.

Where expert systems were good, they were genuinely good. Problems with limited rule counts, clear domain boundaries, and knowledge that can be fully enumerated — XCON really did save DEC tens of millions of dollars. In loan approval systems, if the rules are written well, biases can be explicitly audited and corrected — far more transparent than the hidden biases inside black-box systems. Medical decision support systems still use rule engines of some kind today, on formatted subtasks with well-defined rules.

More importantly, the expert system era left some genuinely important legacies:

**Clean problem decomposition**. Breaking a complex problem into subproblems that can be handled independently is a truly valuable ability. Modern software engineering still does this.

**Interpretability of reasoning**. You can trace every step of the system's reasoning — a rare luxury in today's AI systems. Interpretability became a hot topic again in the deep learning era, and expert systems were the best at it.

**Knowledge representation theory**. The knowledge representation theory built around expert systems later grew into ontology, description logic, and the Semantic Web — things that still run in many industrial systems today.

:::details Ontology, Description Logic, Semantic Web: How Do These Three Things Relate?
These three words are three layers of "how knowledge is structured in computers":

- **Ontology**: in computer science, it means "formally defining the system of concepts and their relations in a domain." For example, the medical ontology SNOMED defines the hierarchical relations between "disease," "symptom," and "drug."
- **Description logic**: the logical language used to build ontologies — more precise than natural language, more computationally tractable than full first-order logic. OWL (Web Ontology Language) is based on description logic.
- **Semantic Web**: the vision proposed by Tim Berners-Lee — web data should be readable by people and "understandable" by machines (machines can grasp the semantic relations in the data). RDF and OWL are its core technologies.

These three form a chain from concept to application: ontology is the goal, description logic is the tool, and the Semantic Web is the application.
:::

And then, the most important legacy of all:

**A clear counterexample about the boundary of reasoning**. Expert systems proved, thoroughly and empirically: **if reasoning depends on a complete explicit knowledge base, then how far your reasoning can go depends entirely on how complete the base is**. The world is open. The knowledge base is closed. The tension between the two is the fundamental limit of any rule-based system.

The flip side of this lesson is exactly where AI research turned in the following decades: don't try to write out all the rules explicitly — instead let the machine **learn** those regularities from data, including all the tacit knowledge humans can't articulate.

That turn was not easy, nor fully successful. But that's the next chapter's story.

<figure>
<p><img src="/figures/ch02_fig3_rule_network.png" alt="Rule network: the three-layer structure of an expert system knowledge base" /></p>
<figcaption>Rule network: the three-layer structure of an expert system knowledge base</figcaption>
</figure>

*Figure 3: The full rule network of the animal-identification expert system. The left layer is raw observed facts (input); the middle layer is intermediate conclusions (mammal, bird, carnivore…); the right layer is the final classification. The links between nodes are rule dependencies. This figure also shows the paradigm's limit: when the number of rules grows from 10 to 10,000, nobody can keep this graph in their head.*

<div class="center">

------------------------------------------------------------------------

</div>

## 10. A Small Pause

Let me sort out this chapter.

The era of symbolic AI and expert systems gave us one very clean lesson: **explicitly expressing the world's knowledge as rules is, in principle, an impossible task**. Not because we didn't try hard enough, but because the structure of knowledge itself is not the form of rules — it is situated, tacit, continuous, and generative.

A logic gate is the simplest physical realization of causality. A rule system is the most direct modeling of causal reasoning. Where they are good, they work very well. But the world is much wider than that.

The closed world assumption is the fatal flaw of this paradigm: inside what you know, it is strong; at the edge of what you know, it fails gracefully; and you never find out where the edge is, because you don't know what you don't know.

This is a pattern worth remembering — not just about expert systems, but about any reasoning system. Every reasoning system has its own closed world boundary: it explicitly or implicitly assumes that some things don't exist, or some situations don't happen. When reality touches that boundary, the reasoning fails in some way.

The failure modes differ, but the lesson is the same: **reasoning needs a world model, and the boundary of the world model is the boundary of reasoning.**

In Chapter 3, we'll start talking about vectors, about continuous representation spaces, about the revolution in representation learning. The step from discrete symbols to continuous vectors is a fundamental leap. But the new problems this leap brings may be no fewer than the old ones it solves.

> Symbols failed, but they left a question behind: what is meaning? In the next chapter, we'll see how vectors give "meaning" a set of geometric coordinates.

<div class="center">

------------------------------------------------------------------------

</div>

## Symbols at Scale: From Expert Systems to Knowledge Graphs

Before closing this chapter, let's see how the legacy of symbolism continues in modern AI — in the form of **knowledge graphs**.

### Knowledge Graphs: Symbolism at Scale

A knowledge graph is essentially a **scaled-up, structured version** of an expert system's knowledge base:

- **Entities** replace the "concepts" of the expert system
- **Relations** replace the "conditions" of If-Then rules
- **Triples** (head entity — relation — tail entity) form the basic unit of knowledge

But knowledge graphs face challenges similar to those of expert systems:

1. **Construction cost**: they need heavy manual labeling or complex extraction algorithms
2. **Coverage problem**: no knowledge graph can ever cover all knowledge
3. **Dynamic updates**: the world changes; knowledge graphs need continuous updating
4. **Knowledge representation**: how do you represent fuzzy, probabilistic, context-dependent knowledge?

### The Conversation Between Knowledge Graphs and LLMs

A frontier direction in current AI research is combining **symbols (knowledge graphs) with statistics (large models)**. This isn't a simple replacement — it's a **complementary relationship**:

| Dimension | Knowledge graph | LLM |
|------|----------|--------|
| **Knowledge representation** | Explicit, structured, discrete | Implicit, distributed, continuous |
| **Interpretability** | High: clear triple structure | Low: black-box vector operations |
| **Reasoning mode** | Symbolic logic reasoning | Statistical pattern matching |
| **Knowledge acquisition** | Human-built / extracted | Learned implicitly from text |
| **Dynamism** | Expensive to update | Adapts via fine-tuning |

### Combining Strategies: RAG and Knowledge Injection

The main ways they are combined today:

1. **Retrieval-augmented generation (RAG)**: use the knowledge graph as external memory, giving the large model factual grounding
2. **Knowledge injection**: convert the graph's triples into training data and inject them into the large model
3. **Symbol-guided generation**: use the graph's structure to constrain the model's output
4. **Hybrid reasoning**: symbolic reasoning for certain knowledge, statistical reasoning for uncertain knowledge

### The Deep Problem: What Fundamentally Separates Symbols and Statistics?

This combination faces fundamental challenges:

1. **Representation gap**: discrete symbols vs. continuous vectors
2. **Reasoning gap**: logical deduction vs. probabilistic inference
3. **Knowledge gap**: explicit statements vs. implicit implications
4. **Update gap**: local updates vs. global retraining

**The key question**: knowledge graphs are "slow but accurate"; large models are "fast but fuzzy." How do you make them work together, not simply glue them together?

### Exercises: A Modern Revival of Symbolism?

Does combining knowledge graphs with large models mean some kind of **revival** of symbolism? Or is it merely **borrowing** symbolic forms inside a statistical framework?

An even deeper question: if large models can "learn" knowledge from text, why do we still need explicit knowledge graphs? A possible answer: **interpretability, verifiability, updatability** — indispensable in critical applications (medicine, law, finance).

<div class="center">

------------------------------------------------------------------------

</div>

## Open Questions

- Do today's large language models, in some implicit way, reintroduce the closed world assumption? Is the distribution of training data a new form of "knowledge base boundary"?

- Interpretability is a real advantage of expert systems. But are interpretability and correctness the same thing? Being able to explain a reasoning process is not the same as the reasoning process being right.

- Dreyfus's critique has one core: situated knowledge cannot be made explicit. Do large language models capture situated knowledge in some way? Or are they just doing very high-dimensional pattern matching?

- **Combining symbols and statistics**: knowledge graphs supply structured knowledge; large models supply generalization. Is this combination a "1+1>2" synergy, or an "oil and water" splice? What fundamental problems does true fusion need to solve?

- **Double representation of knowledge**: the same fact is a discrete triple in a knowledge graph and a continuous vector distribution in a large model. How do the two verify and complement each other? When they conflict, who should you trust?

<div class="center">

------------------------------------------------------------------------

</div>

## Try It Yourself: Build Your Own Expert System, Then Watch It Crash

This chapter's lesson: rules can't be finished, boundaries can't be found, and failure arrives silently. Reading alone won't make these anything more than abstract maxims. You need to build it with your hands, and then watch it crash at the boundary with your own eyes.

### Step 1: Pick a Domain You Actually Know

Don't pick medicine. Don't pick law. Pick a domain you have real experience with — the kind where you can say "this is what you should do in this situation."

A few suggestions:

- Judge the flavor style of a dish (Sichuan / Cantonese / Japanese…)
- Diagnose a common computer problem (network / hardware / software…)
- Judge whether an article is worth a careful read (domain / source / keywords…)
- Rate the difficulty level of a programming problem (easy / medium / hard)

Once you've chosen it, write down on paper: **what are the final conclusions of this domain?** (a list of categories, no more than 10.)

### Step 2: Write Your Rule Base

Using the format shown in Section 7 of this chapter, write at least 15 If-Then rules.

```python
# Rule data structure: each rule is a dictionary
# 'conditions' : list of conditions (ALL must hold to trigger)
# 'conclusion' : rule conclusion (a string)
# 'name'       : rule name, for tracing the reasoning path

# Example: a rule base for classifying cuisines
rules = [
    # --- middle-level rules: ingredient features ---
    {"name": "R1",  "conditions": ["has fish mint"],
     "conclusion": "uses Sichuan ingredients"},
    {"name": "R2",  "conditions": ["has chili pepper", "has Sichuan pepper"],
     "conclusion": "mala style"},
    {"name": "R3",  "conditions": ["has sashimi"],
     "conclusion": "uses Japanese ingredients"},
    {"name": "R4",  "conditions": ["has miso"],
     "conclusion": "uses Japanese ingredients"},
    {"name": "R5",  "conditions": ["has fermented black beans", "has steamed-fish soy sauce"],
     "conclusion": "Cantonese seasoning"},
    {"name": "R6",  "conditions": ["has ginger and scallion"],
     "conclusion": "Cantonese base"},

    # --- middle-level rules: cooking style ---
    {"name": "R7",  "conditions": ["stir-fried over high heat"],
     "conclusion": "high-heat stir-fry style"},
    {"name": "R8",  "conditions": ["steamed"],
     "conclusion": "light cooking style"},
    {"name": "R9",  "conditions": ["charcoal-grilled"],
     "conclusion": "barbecue style"},

    # --- terminal rules: conclude the cuisine ---
    {"name": "R10", "conditions": ["uses Sichuan ingredients", "mala style"],
     "conclusion": "is Sichuan cuisine"},
    {"name": "R11", "conditions": ["uses Japanese ingredients"],
     "conclusion": "is Japanese cuisine"},
    {"name": "R12", "conditions": ["Cantonese seasoning", "light cooking style"],
     "conclusion": "is Cantonese cuisine"},
    {"name": "R13", "conditions": ["Cantonese base", "Cantonese seasoning"],
     "conclusion": "is Cantonese cuisine"},
    {"name": "R14", "conditions": ["barbecue style", "uses Sichuan ingredients"],
     "conclusion": "is Sichuan-style barbecue"},
    {"name": "R15", "conditions": ["high-heat stir-fry style", "Cantonese seasoning"],
     "conclusion": "is Cantonese stir-fry"},
]
```

While writing, pay attention to two things:

**a. Write middle-level rules, not just rules that go straight to a conclusion.**

The bad way:

```
IF has fish mint AND has chili pepper THEN is Sichuan cuisine
```

The good way:

```
IF has fish mint THEN uses Sichuan ingredients
IF has chili pepper AND has Sichuan pepper THEN mala style
IF uses Sichuan ingredients AND mala style THEN is Sichuan cuisine
```

Middle levels make the reasoning chain explainable, and they make problems surface more clearly.

**b. Try reasoning backward from the conclusion, not just forward from the conditions.**

Ask yourself: "to conclude 'it's Sichuan food', what's the minimum set of premises? Is it enough? Are there exceptions?"

### Step 3: Implement the Forward-Chaining Engine

This chapter already gave you the complete logic of the engine:

```python
def forward_chain(rules, initial_facts):
    """
    Forward-chaining inference engine.
    rules: list of rules, each with 'conditions', 'conclusion', 'name'.
    initial_facts: set of initially observed facts.
    Returns: (final fact set, reasoning trace list)
    """
    # known facts = the observed input set
    known_facts = set(initial_facts)
    reasoning_trace = []  # record which rule fired (the reasoning trace)

    # repeat inference until no new fact is added
    while True:
        new_fact_added = False  # whether a new fact was added this round

        for rule in rules:
            # check whether all rule conditions are in the known fact set
            conditions_met = all(cond in known_facts for cond in rule["conditions"])

            if conditions_met and rule["conclusion"] not in known_facts:
                # add the rule conclusion to the known fact set
                known_facts.add(rule["conclusion"])
                # record which rule fired (your reasoning trace)
                reasoning_trace.append(
                    f"fired {rule['name']}: {rule['conditions']} -> {rule['conclusion']}"
                )
                new_fact_added = True

        # stop if this round added nothing new
        if not new_fact_added:
            break

    # output: known fact set + reasoning trace
    return known_facts, reasoning_trace


# --- Example run ---
# Initial observations: the dish features on the table
initial_observations = {"has fish mint", "has chili pepper", "has Sichuan pepper", "stir-fried over high heat"}

final_facts, trace = forward_chain(rules, initial_observations)

print("=== Reasoning trace ===")
for step in trace:
    print(" ", step)

print("\n=== Final conclusions ===")
conclusions = [f for f in final_facts if f.startswith("is ")]
for c in conclusions:
    print(" ", c)
```

You need no GUI, no database, no config file. The simplest implementation: store rules in a list, store known facts in a set, and use one loop to do the reasoning.

### Step 4: Build Three Test Cases — Let One Pass, Let Two Fail

**Case A: the normal case.** A typical case your rule base fully covers. Feed in observations, get the right conclusion, complete reasoning trace. This proves the system "works."

**Case B: the boundary case.** Build a fuzzy case sitting between two categories — one where the premises of two different conclusions are both satisfied. Watch the system's behavior: does it report both conclusions? Pick the first match? Or quietly give a wrong answer?

**Your first question:** is your system's behavior in Case B "acceptable failure" or "dangerous silent failure"? What's the difference between the two?

**Case C: an out-of-distribution case.** Build a case your rule base never anticipated — for example, a fusion dish with both Sichuan and Japanese ingredients; or a computer whose fault crosses all known categories.

Watch: what does the system give you? Does it say "I don't know"? Or does it climb along some nearest rule chain and hand you an answer that looks plausible but is actually wrong?

**Your second question:** which rules did Case C's input trigger? What situations were those rules designed for? What specific flaw of the closed world assumption does the way the system misuses them reveal?

### Step 5: Do the Accounting

How many rules did you write? How much of the domain do they cover?

Estimate: how much knowledge of this domain is still outside your rules? And is what's missing missing because you don't know it, or because you "know it but can't say it"?

**Your third question (and the hardest):** if you had to convert all of your knowledge of this domain into rules, how many would you need? Is that number finite?

### Check Your Work

When you finish this exercise, you should be able to answer:

- Where the boundary of your rule base is — not the boundary you guessed, but the one you hit with your own hands
- Whether the way your system fails at the boundary matches the "confident error" described in this chapter
- Why knowledge engineers say this is "an almost impossible task"

For the third one, you don't need to explain it in theory. You just need to feel, once, in your own little domain, why.

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Shortliffe, E. H. (1976). *MYCIN: Computer-Based Medical Consultations* — the founding case study of expert systems

- Dreyfus, H. L. (1972). *What Computers Can't Do* — the philosophical critique of the symbolic AI paradigm, mocked at the time, later proven profound

- Reiter, R. (1978). "On Closed World Databases." *Logic and Data Bases* — the formal definition of the closed world assumption; the original source for this chapter's core concept

- Minsky, M. (1975). "A Framework for Representing Knowledge." *The Psychology of Computer Vision* — frame systems, another attempt to solve the knowledge representation problem

- Winston, P. H. & Horn, B. K. P. (1981). *LISP* — contains the classic animal-identification expert system; the original inspiration for this chapter's scripts

- [Bader et al., 2004, arXiv:cs/0408069] — neuro-symbolic integration: the unifying challenge of connectionism and first-order logic knowledge representation; the modern echo of the expert system era's problems

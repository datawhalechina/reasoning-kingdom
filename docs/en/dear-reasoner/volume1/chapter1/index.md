# Chapter 1: Telegraph, Flashlight, and the Origin of Logic

:::info
**Mr. Pallas's Cat's Warm Welcome**  
Hey there, future reasoning scientists! Today we start with the simplest "yes" and "no." Don't underestimate these two little fellows — all reasoning, at its core, is nothing more than different permutations of them. Like the "dot" and "dash" of the telegraph, or the "on" and "off" of a flashlight (I often use mine to find glasses that have fallen into the sofa crack). Behind the binary lies infinite possibility — like painting the entire world with only black and white.
:::

---

## Core Question: How Do We Represent the World?

Piglet put down the telegraph model, his eyes sparkling. "I've been thinking about something: **how do we fit the entire world into a computer?** I mean, the world is so big — mountains, rivers, cats, dogs — and computers only understand 0 and 1. How is that even possible!"

It was a spring afternoon in Kangle Garden at Sun Yat-sen University, the scent of kapok blossoms drifting in through the window. Sunlight filtered through the branches of a century-old banyan tree, casting dappled shadows across the study of the Black Stone House. This study was situated in a red-brick building completed in 1914, with thick walls, arched windows, and books piled beside the fireplace — one of the oldest buildings on campus, once the residence of the president of Lingnan University.

By the window, Little Seal looked up from his *History of Logic*, adjusting his glasses. "Piglet, that's a great question. Throughout history, humans have used language, painting, music, mathematical symbols... all sorts of ways to represent the world. But the computer's way of representing things — well, it's a bit special."

Mr. Pallas's Cat set down his gongfu tea set, the fragrance of tea spreading through the air. "You've raised a fundamental question. To represent the world, we must first **simplify it** — like turning an oil painting into pixel art. Today, we begin with the simplest form of representation — binary expression. Don't worry, it may sound simple, but its power is beyond imagination."

## Lessons from the Telegraph: A Binary Universe of Dot and Dash

Piglet fiddled with the telegraph model again, his fingertips tapping the key lightly, like a child discovering a new toy.

"Dot... dash... dot-dash-dot..." He listened intently to the simple binary rhythm, as if he could hear the same signals that had flowed through transoceanic cables a century ago. "Professor, this is amazing! The telegraph only uses two sounds — 'dot' and 'dash' — but in the 19th century people used them to send countless messages: from war news to business orders, from 'I love you, Mom' to 'I've discovered a new element'!"

By the window, Little Seal set down his book and said gently, "Morse code uses combinations of dots and dashes to represent different letters. This incredibly minimalist system of representation can transcend the limits of time and space. Think about it — the same dot-dash sounds carry different meanings in London and New York."

Mr. Pallas's Cat nodded, his eyes curving into crescent moons. "Yes, this is the core of what we explore today: **how to represent the most complex world using the most minimal elements**. Like building a castle with Lego bricks — there are only a few shapes of bricks, but castles can take endless forms."

He picked up the flashlight from the table and pressed the switch.

**On**. Press again.

**Off**.

"Only two states," said Mr. Pallas's Cat, the flashlight beam drawing circles on the wall. "On, or off. Yes, or no. 1, or 0. The dot-dash of the telegraph, the on-off of the flashlight — these seemingly simple binary choices are the starting point of how we represent the world. By the way, I've had this flashlight for three years and haven't changed the batteries once — impressive quality."

---

## Three Cornerstones: The Birth of AND, OR, NOT

Piglet stared at the flashlight, his brow furrowing like a little old man. "But Professor, what complex things can you do with just on and off? You can't write *Harry Potter* with nothing but on and off, can you?"

Mr. Pallas's Cat smiled. "Great question! But let's not rush — we'll start with the three most basic logical relationships. They're like the ABCs of the alphabet — simple, but capable of forming every word." He pulled several small switches and wires from a drawer, moving with the practiced ease of a magician.

### The First Relationship: AND

Mr. Pallas's Cat connected two switches in series and attached a small light bulb. "Look, now for the light to turn on, **both** switches must be closed." He demonstrated as he spoke. "Switch A **AND** switch B must both be closed. It's like... hmm, like having to press the right two numbers on a combination lock at the same time for it to open."

Piglet's eyes lit up. "I get it! Like how both Little Seal and I have to agree for something to pass! Or like a microwave — the door has to be closed **AND** the timer has to be set before it starts heating!"

:::info
**Mr. Pallas's Cat's Commentary**  
The AND relationship is the foundation of cooperation — and the nemesis of procrastination, because all conditions must be met. In logic, we write it as `A ∧ B` (read as "A and B"). Only when A **and** B are both true (1) is the result true. A personal gripe: it's like making plans to have dinner with friends — everyone has to say "yes." If just one person says "I'm working late," the whole plan falls apart.
:::

| A | B | A ∧ B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

### The Second Relationship: OR

Next, Mr. Pallas's Cat connected the switches in parallel, his movements sprightly as if playing the piano. "Now, as long as **at least one** switch is closed, the light turns on." He continued, "Either switch A **OR** switch B closed is enough. It's like... hmm, like when you're hungry and there's an apple **or** a banana in the fridge — eat whichever you like."

Little Seal nodded. "This reminds me of voting — as long as one vote is in favor, the proposal can pass. Or like an emergency exit — any door opening allows escape."

Piglet chimed in excitedly: "And Wi-Fi passwords! Enter the right password **OR** connect to the guest network — either way, you can get online!"

:::info
**Mr. Pallas's Cat's Commentary**  
The OR relationship embraces possibility — it's a very easygoing fellow. `A ∨ B` is true when A **or** B is true. Note that logical OR is inclusive — when both A and B are true, the result is still true. It's like asking "Want to watch a movie or have hotpot this weekend?" — you can answer "Both!" Of course, in real life, "or" is sometimes exclusive, like "turn left or turn right" — you can only pick one. The logical world is far more forgiving than the real world!
:::

| A | B | A ∨ B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

### The Third Relationship: NOT

Finally, Mr. Pallas's Cat demonstrated a single-pole double-throw switch, like a magician revealing a prop. "This switch is special," he said, his eyes twinkling with mischief. "When you flip it one way, the light is on; flip it the other way, the light is off — it always gives the opposite state. It's like a rebellious child: you say east, and it insists on going west."

Piglet jumped in: "That's NOT! On becomes off, off becomes on! Like antonyms — hot becomes cold, tall becomes short!"

Little Seal added: "Or like negation — 'I like cats' becomes 'I don't like cats.' But you have to be careful — sometimes a double negation becomes an affirmation, like 'I don't not like it' actually means you do like it."

:::info
**Mr. Pallas's Cat's Commentary**  
NOT is the mirror image of logic, and a habitual contrarian. `¬A` (read as "not A") turns true into false and false into true. It is the foundation of all negation and inversion. A personal gripe: the symbol ¬ looks like a little hat, but when you put it on top of A, it completely transforms it. In programming, we often use `!A`, which looks more like it's shouting "No!" But be careful about overusing NOT — a triple negation like `¬¬¬A` takes even me a while to figure out.
:::

| A | ¬A |
|---|----|
| 0 | 1 |
| 1 | 0 |

---

## Discretization Thinking: Binary Slices of a Continuous World

Little Seal closed his book, looking thoughtful. "Professor, I have a question. The real world is continuous — light has brightness, sound has volume, temperature has degrees. But computers only process 0 and 1. Isn't a lot of information lost in between? Like turning a rainbow into a black-and-white photograph?"

"That's exactly the crux," said Mr. Pallas's Cat approvingly, like a teacher who has just heard a student ask a great question. "We have an important mode of thinking: **discretization thinking**. But don't worry — although some detail is lost, what we gain is clarity and tractability."

He walked to the window and pointed at the parasol tree outside. "Look at that tree. If we ask 'Is it autumn now?' the answer can be 'yes' or 'no.' But autumn itself is a continuous process — leaves go from green to yellow and then fall, with no clear dividing line. You can't point at a particular leaf and say 'Look! Autumn begins at this very moment!'"

"So we have to **draw a boundary**," Piglet said, suddenly understanding, gesturing in the air with his fingers. "Like saying 'autumn begins when more than half the leaves have turned yellow'! Or like exams — 60 is a pass, 59.9 is a fail, even though they're only 0.1 apart!"

"Precisely," said Mr. Pallas's Cat, nodding and returning to his seat. "This is **binarization** — turning a continuous quantity into discrete 0 and 1. Kind of like dividing dumplings into 'cooked' and 'not cooked,' even though there's a 'half-cooked' state in between — but we usually only care about the final result."

:::detail
**Advanced: The Mathematical Principle of Discretization**

The core of discretization is **threshold setting** — drawing that "passing line." Let a continuous variable x (e.g., temperature) and a threshold θ (e.g., 38°C), the discretization result is:

$$
f(x) = 
\begin{cases}
1 & \text{if } x \geq \theta \quad (\text{fever — take medicine}) \\
0 & \text{otherwise} \quad (\text{fine — drink more water})
\end{cases}
$$

But the choice of threshold θ is critical; choosing poorly can lead to absurd outcomes. Too strict (say, defining fever as ≥40°C) delays treatment; too lenient (35°C already counted as fever) makes healthy people take medicine unnecessarily. It's like exam cutoffs — set too high and no one passes, set too low and everyone gets full marks.

Modern digital systems are smarter, using **multi-level discretization** (e.g., 8 bits for 256 levels, like using 256 shades of gray instead of just black and white) and **dithering** (deliberately adding a little noise to make transitions smoother) to reduce quantization error. But the earliest computers weren't this refined — they really only recognized 0 and 1, like a stubborn old man who sees everything in black and white.

Historically, Claude Shannon, in his 1937 master's thesis *A Symbolic Analysis of Relay and Switching Circuits*, applied Boolean algebra to circuit design for the first time. Think about it — a master's student's work laid the foundation for the entire digital age! This tells us: good ideas don't care about your age; they only care whether someone is willing to think and to act.
:::

---

## Orthogonal Computation Graphs: Seeing the Flow of Logic

Mr. Pallas's Cat walked to the whiteboard and picked up a marker, like an artist preparing to paint. "Now, let's use orthogonal computation graphs to visualize these logical operations. It's like taking an X-ray of logical thinking — we can 'see' the flow of thought."

The diagram he drew was as orderly as a circuit board, right-angled lines crisp and clean, like the perfect work of someone with obsessive tendencies.

![Boolean Logic Orthogonal Computation Graph](/figures/boolean_logic_ortho.svg)

"Look," said Mr. Pallas's Cat, pointing at the diagram and tapping lightly with the marker, "A and B enter from the left, like two children holding hands and walking into an amusement park. They pass through three logic gates — ∧, ∨, ¬ — like playing at three different rides, and finally produce three outputs. Data flows from left to right, like products on an assembly line, or like reading a book — from the letters on the left to the meaning on the right."

Piglet leaned in for a closer look, his nose almost touching the whiteboard. "These ∧, ∨, ¬ symbols are so concise! ∧ looks like a little rooftop, ∨ like a valley, ¬ like a little hat! I can clearly see the function of each gate, as clearly as reading a map!"

"That's the design philosophy of orthogonal computation graphs," said Mr. Pallas's Cat, with a hint of pride. "**Right-angle orthogonality** emphasizes regularity — I don't like crooked, wiggly lines; they're unpleasant to look at. **Left to right** matches reading habits — we don't read Chinese from right to left, after all. **Automatic spacing** maintains visual balance — let Graphviz, that clever little tool, decide the best layout on its own. I once tried manually adjusting the spacing, and it only got messier. Better to leave it to specialized tools."

---

## The Magic of Combination: From Simple Gates to Complex Functions

Little Seal looked at the computation graph and suddenly had a question, his eyes behind his glasses glinting with curiosity. "Professor, these simple gates — ∧, ∨, ¬ — are like three musical notes. Can they be combined to produce more complex 'music'? Like Beethoven's Fifth?"

"Of course!" Mr. Pallas's Cat's eyes flashed, as if he had discovered treasure. "This is the charm of digital circuits — using simple building blocks to construct complex castles. For example, we can use AND, OR, and NOT to build an **XOR gate**, which has a very cool symbol: ⊕."

He wrote on the whiteboard, his handwriting flowing:

$$
A \oplus B = (A \land \neg B) \lor (\neg A \land B)
$$

"Translated into plain language: 'A is true and B is not, or A is not true and B is.'" Mr. Pallas's Cat explained. "XOR means 'output 1 when the two inputs are different, 0 when they are the same.' It is crucial in many cryptographic algorithms — because its behavior is somewhat 'unpredictable,' making it well-suited for ciphers."

| A | B | A ⊕ B |
|---|---|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Piglet jumped up excitedly. "I could use this to make a rock-paper-scissors game! Two people make their moves, different means someone wins, same means it's a draw! Or like a stairway light — two switches controlling one light, pressing either switch changes the light's state! Professor, isn't the XOR gate kind of 'rebellious'? It specifically looks for differences!"

:::detail
**Advanced: De Morgan's Laws and Logic Minimization**

The British mathematician Augustus De Morgan discovered two important laws; they are like the "Transformers" of the logical world:

$$
\neg(A \land B) = \neg A \lor \neg B
$$
$$
\neg(A \lor B) = \neg A \land \neg B
$$

In plain language: **"NOT (A AND B)" equals "NOT A OR NOT B"**; **"NOT (A OR B)" equals "NOT A AND NOT B."** This is a bit like double negation turning back into affirmation, but a little more involved.

These laws allow us to convert between AND and OR and, with the help of NOT, to achieve logic minimization. For example, that complicated XOR expression can be transformed through De Morgan's laws into a more compact form:

$$
A \oplus B = \neg(\neg(A \land \neg B) \land \neg(\neg A \land B))
$$

(Dizzy looking at it? It's okay — I have to stare at it for a while myself before it makes sense.)

This kind of minimization saves transistors and boosts efficiency in circuit design. Modern chip design makes heavy use of such optimization techniques — after all, with billions of transistors on a single chip, every saving counts. It's like packing a suitcase: rolling up your clothes lets you fit in a few more items. De Morgan's laws are the "organizing and tidying" technique of the logical world.

By the way, De Morgan also founded the London Mathematical Society. He had poor eyesight and was nearly blind in his later years, but that didn't stop him from doing complex logical reasoning in his head. So the next time you find math difficult, think of this nearly-blind mathematician.
:::

---

## The Miracle of NAND: One Gate, Infinite Possibilities

Mr. Pallas's Cat drew a new symbol on the whiteboard: ⊼ (NAND).

"This is the **NAND gate**," he said. "`A ⊼ B = ¬(A ∧ B)`. It has a magical property: **using only NAND gates, you can implement ALL logical functions**."

Piglet's eyes went wide. "Using only one kind of gate? How is that possible!"

"Watch this," said Mr. Pallas's Cat, drawing several NAND gate connections:

$$
\neg A = A ⊼ A
$$
$$
A ∧ B = \neg(A ⊼ B) = (A ⊼ B) ⊼ (A ⊼ B)
$$
$$
A ∨ B = \neg(\neg A ∧ \neg B) = \cdots
$$

"Modern chips make extensive use of NAND gates," Mr. Pallas's Cat continued, "because they are simple to manufacture and fast. From smartphones to supercomputers, at the bottom level it's all complex combinations of these basic gates."

:::detail
**Advanced: Proof of NAND's Functional Completeness**

NAND is a **functionally complete** set of logic gates. The proof goes as follows:

1. **Implement NOT**: `¬A = A ⊼ A`
2. **Implement AND**: `A ∧ B = ¬(A ⊼ B) = (A ⊼ B) ⊼ (A ⊼ B)`
3. **Implement OR**: using De Morgan's laws, `A ∨ B = ¬(¬A ∧ ¬B)`

Since {AND, OR, NOT} is functionally complete and NAND can generate all three, NAND alone is functionally complete.

This discovery is profoundly significant. Theoretically, we can construct all digital circuits using the **same physical structure**. This simplifies chip manufacturing and lays the foundation for reconfigurable computing (such as FPGAs).
:::

---

## Mental Model: Three Levels of Discretization Thinking

Mr. Pallas's Cat returned to his seat and took a sip of tea. "Let's summarize today's most important mental model: **discretization thinking**."

He wrote three levels on the whiteboard:

### Level 1: Drawing Boundaries
"The world is inherently continuous, but we **choose boundaries**. What counts as 0? What counts as 1? This choice determines how we understand the world."

### Level 2: Building Combinations
"Using simple gates to compose complex functions. Like using a finite alphabet to write infinite stories, or a finite set of notes to compose endless melodies."

### Level 3: Abstract Leaps
"Eventually, we stop thinking about individual gates and focus on **functional modules**. It's like no longer thinking 'AND gate connected to OR gate' but instead thinking 'this is an adder.'"

Little Seal mused, "Isn't this the power of **abstraction**? From the concrete to the general, from components to systems."

"Exactly," Mr. Pallas's Cat smiled. "Abstraction allows us to think about more complex things without drowning in details."

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary**  
1. **Three Cornerstones of Boolean Logic**: AND, OR, NOT are the foundation of all digital computation  
2. **Discretization Thinking**: Transforming the continuous world into discrete states — the core cognitive tool of computer science  
3. **Complexity from Simplicity**: Finite basic elements can build infinite possibilities  
4. **Visual Understanding**: Orthogonal computation graphs let us "see" the flow of logic  
5. **NAND Completeness**: A single gate type can implement all logical functions, embodying the elegance and simplicity of mathematics
:::

---

## Code Practice: Implementing Boolean Logic in Python

"Let's use Python code to practice Boolean logic," said Mr. Pallas's Cat. "Code not only helps us understand abstract concepts, but also lets us 'run' logical reasoning."

### Basic Logic Gate Implementation

```python
# The three cornerstones of Boolean logic: AND, OR, NOT
def logic_and(a, b):
    """AND gate: output is True only when both inputs are True"""
    return a and b

def logic_or(a, b):
    """OR gate: output is True as long as at least one input is True"""
    return a or b

def logic_not(a):
    """NOT gate: turns True into False, False into True"""
    return not a

# Testing basic logic gates
print("AND gate tests:")
print(f"True AND True = {logic_and(True, True)}")    # True
print(f"True AND False = {logic_and(True, False)}")  # False
print(f"False AND False = {logic_and(False, False)}") # False

print("\nOR gate tests:")
print(f"True OR True = {logic_or(True, True)}")      # True
print(f"True OR False = {logic_or(True, False)}")    # True
print(f"False OR False = {logic_or(False, False)}")  # False

print("\nNOT gate tests:")
print(f"NOT True = {logic_not(True)}")               # False
print(f"NOT False = {logic_not(False)}")             # True
```

### Combinational Logic: Building Complex Functions from Basic Gates

```python
# Building XOR and a voting machine from basic gates
def logic_xor(a, b):
    """XOR gate: output True when inputs differ, False when they are the same"""
    # XOR = (A AND NOT B) OR (NOT A AND B)
    return logic_or(
        logic_and(a, logic_not(b)),
        logic_and(logic_not(a), b)
    )

def voting_machine(votes):
    """Voting machine: output True when two or more switches are on
    
    Parameters:
        votes: list of boolean values, representing each voter's choice
    
    Returns:
        True if yes votes >= 2, False otherwise
    """
    # Count the number of True values
    true_count = sum(1 for vote in votes if vote)
    return true_count >= 2

# Testing combinational logic
print("\nXOR gate tests:")
print(f"True XOR True = {logic_xor(True, True)}")    # False
print(f"True XOR False = {logic_xor(True, False)}")  # True
print(f"False XOR False = {logic_xor(False, False)}") # False

print("\nVoting machine tests:")
print(f"[True, True, False] -> {voting_machine([True, True, False])}")  # True
print(f"[True, False, False] -> {voting_machine([True, False, False])}") # False
print(f"[False, False, False] -> {voting_machine([False, False, False])}") # False
```

### Proving the Functional Completeness of NAND

```python
def logic_nand(a, b):
    """NAND gate: the negation of AND"""
    return not (a and b)

# Building all other logic gates using NAND gates only
def nand_not(a):
    """NOT from NAND: NOT A = NAND(A, A)"""
    return logic_nand(a, a)

def nand_and(a, b):
    """AND from NAND: A AND B = NOT(NAND(A, B))"""
    return nand_not(logic_nand(a, b))

def nand_or(a, b):
    """OR from NAND: A OR B = NAND(NOT A, NOT B)"""
    return logic_nand(nand_not(a), nand_not(b))

def nand_xor(a, b):
    """XOR from NAND (requires 5 NAND gates)"""
    # XOR = NAND(NAND(A, NAND(A, B)), NAND(B, NAND(A, B)))
    nand_ab = logic_nand(a, b)
    nand_a_ab = logic_nand(a, nand_ab)
    nand_b_ab = logic_nand(b, nand_ab)
    return logic_nand(nand_a_ab, nand_b_ab)

# Testing NAND completeness
print("\nTesting all logic gates built from NAND:")
a, b = True, False
print(f"NOT from NAND: {nand_not(a)} (should be {logic_not(a)})")
print(f"AND from NAND: {nand_and(a, b)} (should be {logic_and(a, b)})")
print(f"OR from NAND: {nand_or(a, b)} (should be {logic_or(a, b)})")
print(f"XOR from NAND: {nand_xor(a, b)} (should be {logic_xor(a, b)})")
```

### Mental Model: Discretization Thinking Framework

```python
def discretize_continuous(value, threshold):
    """Discretize a continuous value into a boolean
    
    Parameters:
        value: continuous value (e.g., temperature, brightness)
        threshold: if value exceeds threshold, return True; otherwise False
    
    Returns:
        discrete boolean value
    """
    return value > threshold

# Application: Temperature alarm system
def temperature_alarm(current_temp, danger_threshold):
    """Temperature alarm: triggers when temperature exceeds the danger threshold
    
    Parameters:
        current_temp: current temperature
        danger_threshold: danger temperature threshold
    
    Returns:
        whether an alarm is needed (True/False)
    """
    return discretize_continuous(current_temp, danger_threshold)

# Testing discretization thinking
print("\nDiscretization thinking tests:")
temp = 37.5  # current temperature
threshold = 38.0  # danger threshold
print(f"Temperature {temp}°C, threshold {threshold}°C, need alarm? {temperature_alarm(temp, threshold)}")

temp = 38.5
print(f"Temperature {temp}°C, threshold {threshold}°C, need alarm? {temperature_alarm(temp, threshold)}")
```

"Remember," Mr. Pallas's Cat summarized, "Boolean logic is the perfect embodiment of **discretization thinking**. By concretizing these abstract concepts in code, we can truly understand how to use simple 0s and 1s to represent a complex world."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Circuit Design**: Using three switches, design a "voting machine" — the light turns on when two or more switches are on. Draw the truth table and computation graph. (Use logic gates!)

2. **Logic Challenge**: Implement XOR using only NAND gates. What is the minimum number of NAND gates required?

3. **Real-World Applications**: Observe the electronic devices around you and guess where Boolean logic is at work. (Hint: elevator floor selection, microwave timer, traffic light control...)

### Historical Investigation (for Little Seal)
1. **Figure Study**: George Boole was a self-taught mathematician. Research his life and consider: how can someone without formal education make major contributions to science?

2. **Tracing the Thread**: From Aristotle's syllogisms, to Boole's algebra, to Shannon's circuit theory — how did logic progressively move toward formalization and practical application?

3. **Cultural Impact**: How did Boolean logic influence 20th-century philosophical movements? (Hint: logical positivism, analytic philosophy)

### Integrated Reflection
1. **Imagining the Limits**: If logic were not binary (0/1) but ternary (true/false/possible), or even infinitely-valued (fuzzy logic), what would the world look like? What problems could such "computers" solve?

2. **Ethical Reflection**: When we transform human decisions (such as medical diagnoses, legal judgments) into Boolean logic, what is lost? What is gained?

3. **Creative Exercise**: Design a "logic story" — use AND, OR, NOT to describe an everyday scenario (e.g., "weekend outing plan").

---

## Coming Up Next

Outside the window, the sky was growing dark, and warm lamplight filled the Black Stone House.

"Today we began with the dot and dash of the telegraph," said Mr. Pallas's Cat. "In the next chapter, we face a more down-to-earth question: **reasoning comes at a cost**."

Piglet asked curiously, "A cost? What kind of cost?"

"The cost of time, the cost of space," Mr. Pallas's Cat said slowly. "If you had to find one book among all the books in the universe, how long would a brute-force search take?"

Little Seal flipped through his pages. "This introduces the concept of computational complexity. Historically, when did people first become aware of this problem?"

Mr. Pallas's Cat smiled. "We'll take it slow. See you in the next chapter."

---

> **Piglet's note**: I successfully built XOR out of NAND gates! Even though it took 5 gates, the moment I saw the light come on, it felt like creating a world.
>
> **Little Seal's note**: I read Boole's biography and was amazed that he simultaneously studied logic, probability, and linguistics. True thinkers never set limits on themselves.
>
> **Mr. Pallas's Cat's closing words**: You see, starting from the simplest 0 and 1, we have already taken the first step in reasoning science. Remember: complexity is not the starting point; it is simplicity, after countless permutations and combinations, that emerges. We'll take it slow — understanding is what matters most.

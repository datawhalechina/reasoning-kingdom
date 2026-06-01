# Chapter 3: Turing's Paper Tape (Computability)

:::info
**Mr. Pallas's Cat's Warm Welcome**  
After exploring how to represent the world and how to describe the world's structure, we arrive at an even more fundamental question: **where are the boundaries of reasoning?** Are there forms of thought that no machine can realize? Are there problems that no algorithm can solve? Today, we explore the limits of computation.
:::

---

## Core Question: Where Are the Boundaries of Reasoning?

Another dusk in Kangle Garden. The evening breeze from the Pearl River carried moisture through the study of the Black Stone House. Outside, kapok blossom season was nearing its end; a few late-blooming flowers stubbornly held their red against the twilight. Inside the study, newly arrived books were piled beside the fireplace, and the cover of one — *Alan Turing: The Enigma* — glowed softly under the lamplight.

Piglet was using his laptop to test the running times of various algorithms, with time-complexity charts flickering on the screen. Little Seal was browsing the latter half of *A History of Logic*, his fingertips pausing on the page about "Hilbert's Program."

"Professor," Piglet suddenly looked up, "I have a question. Last chapter we said some problems cost too much to compute, like exponentially complex problems. But are there problems that **no amount of computing power** can solve?"

Little Seal gently closed his book. "Among the 23 mathematical problems Hilbert posed in 1900, the second was the completeness of the axiomatic system of arithmetic. He believed that all true mathematical propositions could be derived from axioms."

Mr. Pallas's Cat set down his gongfu tea set, the honeyed fragrance of Fenghuang Dancong drifting through the air. "You've raised the most profound philosophical question in computer science: **where are the boundaries of reasoning?**"

He walked to the whiteboard and wrote two questions:

1. **What is computable?**
2. **What is uncomputable?**

"In the previous two sessions, we discussed **how to represent the world** (Boolean logic) and **how to describe the world** (time/space structures)," Mr. Pallas's Cat said, turning to face them. "Today, we ask: **which parts of this world can we compute?**"

## Turing's Insight: What Can Machines Do?

Piglet thought for a moment. "Can computers solve all mathematical problems? Like proving a theorem?"

"Mathematicians in the early 20th century were wondering the same thing," said Mr. Pallas's Cat. "David Hilbert proposed an ambitious program: **to construct a complete, consistent axiomatic system for all of mathematics**. This meant:
1. **Completeness**: all true mathematical propositions can be derived from axioms
2. **Consistency**: no contradictory conclusions can be derived
3. **Decidability**: there exists a mechanical procedure to determine the truth or falsity of any proposition"

Little Seal added: "This is the so-called 'decision problem' (Entscheidungsproblem). Hilbert believed that all problems in mathematics could be solved through computation."

"But Kurt Gödel gave a shocking answer in 1931," Mr. Pallas's Cat continued. "His **incompleteness theorems** proved: in any sufficiently powerful formal system, there always exist propositions that can neither be proved nor disproved."

Piglet's eyes lit up. "So mathematics itself has undecidable problems!"

"Yes," Mr. Pallas's Cat nodded. "But this raises a deeper question: **is the decision problem itself decidable?** Does there exist an algorithm that can determine the truth or falsity of any mathematical proposition?"

"This is exactly the question Alan Turing pondered in 1936," said Mr. Pallas's Cat, walking to the bookshelf and taking down *Alan Turing: The Enigma*. "He was only 24 years old, at King's College, Cambridge."

## The Turing Machine: A Thought Experiment

Mr. Pallas's Cat drew an infinitely long paper tape on the whiteboard, divided into squares.

"Turing imagined a minimalist model of computation — the **Turing machine**," he explained. "It has only a few components:
1. **An infinitely long tape**: divided into squares, each square can hold one symbol (usually 0 or 1)
2. **A read/write head**: can read the current square, write a new symbol, and move left or right
3. **A state register**: stores the current state
4. **A rule table**: based on the current state and the symbol read, determines what to write, how to move, and what state to enter"

![Turing Machine Diagram](/figures/turing_machine_ortho.svg)

Piglet stared at the diagram. "This machine is so simple! Can it do anything complex?"

"That is Turing's profound insight," said Mr. Pallas's Cat. "**Any computable problem can be solved by a Turing machine**. Conversely, if a Turing machine cannot solve a problem, that problem is **uncomputable**."

Little Seal mused: "Professor, what is the relationship between this model and real computers?"

"The Turing machine is a **universal model of computation**," Mr. Pallas's Cat explained. "All of today's computers — from smartphones to supercomputers — are equivalent in computing power to Turing machines. This is the core of the **Church-Turing thesis**: a computable function is precisely one that can be computed by a Turing machine."

:::info
**Mr. Pallas's Cat's Commentary**  
The simplicity of the Turing machine is its strength. It defines the essence of computation with the fewest basic elements (tape, read/write head, states). It's like how Boolean logic defines all logical operations with the simplest AND, OR, NOT — **universality emerges from simplicity**.
:::

## The Halting Problem: Drawing the Boundary of Reason

"Now, let's return to our original question," said Mr. Pallas's Cat. "Do uncomputable problems exist? Turing answered affirmatively with the **halting problem**."

He wrote on the whiteboard:

**The Halting Problem**: Given a Turing machine M and an input w, determine whether M will **halt** (eventually stop running) or **run forever** (loop infinitely) on input w.

"That sounds simple enough," Piglet said. "Just write a program to check it, right?"

"Let's try," Mr. Pallas's Cat smiled. "Suppose there exists a program `HALT(M, w)` that can correctly determine whether any program M will halt on input w."

He wrote pseudocode on the whiteboard:

```python
def HALT(M, w):
    """If program M halts on input w, return True; otherwise return False"""
    # Assume this function exists
    return ...  # True or False
```

"Now, let's construct a new program called `PARADOX`," Mr. Pallas's Cat continued writing:

```python
def PARADOX(P):
    if HALT(P, P):  # if program P halts on its own input
        while True:  # enter an infinite loop
            pass
    else:  # if program P does not halt on its own input
        return  # halt immediately
```

Piglet frowned in thought. "This program is a bit strange... it does the opposite of what `HALT` says."

"Exactly!" Mr. Pallas's Cat's eyes flashed. "Now ask: does `PARADOX(PARADOX)` halt?"

1. Suppose `HALT(PARADOX, PARADOX)` returns `True` (thinks `PARADOX` halts on itself)
   - Then `PARADOX(PARADOX)` sees `HALT` returned `True`, so it enters an infinite loop — **it does NOT halt**
   - Contradiction! `HALT`'s judgment was wrong

2. Suppose `HALT(PARADOX, PARADOX)` returns `False` (thinks `PARADOX` does not halt on itself)
   - Then `PARADOX(PARADOX)` sees `HALT` returned `False`, so it halts immediately
   - Another contradiction! `HALT`'s judgment was wrong again

Little Seal drew a deep breath. "No matter what `HALT` judges, it leads to a contradiction. So such a program **cannot exist**."

"Precisely," said Mr. Pallas's Cat slowly. "The halting problem is **undecidable**. This is a fundamental limitation — not because we aren't clever enough, not because computers aren't fast enough, but because **the very nature of the problem makes it impossible to solve algorithmically**."

:::detail
**Advanced: The Philosophical Implications of the Halting Problem**

The undecidability of the halting problem carries profound philosophical significance:

1. **The paradox of self-reference**: the core of the `PARADOX` program is **self-reference** — it attempts to judge its own behavior. This recalls ancient logical paradoxes:
   - "This statement is false" (the liar paradox)
   - "The barber shaves all those, and only those, who do not shave themselves" (Russell's paradox)

2. **Intrinsic limits of computation**: the halting problem proves that some problems are **in principle** unsolvable by computation. This draws a boundary for computing power.

3. **A computational version of Gödel's incompleteness theorems**: Turing essentially rediscovered Gödel's insight in the language of computation. The undecidability of the halting problem corresponds to the incompleteness of formal mathematical systems.

4. **The final answer to Hilbert's Program**: there is no general algorithmic solution to the decision problem (Entscheidungsproblem). The automation of mathematics has its fundamental limits.

Historically, Turing's paper *On Computable Numbers, with an Application to the Entscheidungsproblem* and Alonzo Church's lambda calculus independently solved Hilbert's decision problem at roughly the same time, but Turing's machine model had a greater impact due to its intuitiveness and physical realizability.
:::

---

## Mental Model: Problem Classification — Understanding the Territory of Computation

Mr. Pallas's Cat returned to his seat and took a sip of tea. "Let's summarize today's most important mental model: **problem classification**."

He drew a classification diagram on the whiteboard:

```
All Problems
    ├── Computable Problems
    │    ├── Efficiently solvable (class P)
    │    ├── Possibly efficiently solvable (class NP)
    │    └── Theoretically solvable but practically infeasible (e.g., extremely large exponential-time problems)
    │
    └── Uncomputable Problems
         ├── The Halting Problem
         ├── Problems related to Gödel's incompleteness
         └── Various undecidable problems equivalent to the halting problem
```

"This classification framework helps us understand," Mr. Pallas's Cat explained, "that not all problems are created equal. Some problems:
1. **Are easy to solve** (like sorting, searching)
2. **Are hard but solvable** (like certain NP-complete problems)
3. **Are in principle unsolvable** (like the halting problem)

"Understanding this is understanding **the boundaries of reasoning**."

### Levels of Computability

Little Seal pointed at the diagram and asked: "Professor, within computable problems, are there finer classifications?"

"Yes," Mr. Pallas's Cat nodded. "This is the subject of **computability theory**. For example:
- **Recursively enumerable problems**: solutions can be verified in finite time, but the process of finding a solution may never terminate
- **Recursive problems**: solutions can be found in finite time
- **Complexity classes**: P, NP, EXP, etc., which we discussed in the previous chapter

"These classifications form the **complexity landscape** of computation, helping us understand the intrinsic difficulty of different problems."

:::info
**Mr. Pallas's Cat's Commentary**  
Problem classification is not about setting limits on thought, but about **wisely allocating cognitive resources**. Knowing what is computable lets us focus on possible problems; knowing what is uncomputable helps us avoid futile struggles. This is a hallmark of intellectual maturity.
:::

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary: Understanding the Boundaries of Reasoning**  
1. **The Turing machine model**: defines the essence of computation with the simplest tape, read/write head, and states — the theoretical foundation of all modern computers  
2. **The halting problem is undecidable**: fundamental limitations exist that make certain problems impossible to solve algorithmically  
3. **Problem classification thinking**: distinguishing between the computable and the uncomputable, the easy and the hard — a crucial framework for rational thought  
4. **The trap of self-reference**: when a system tries to judge itself, it often leads to paradox and undecidability  
5. **Accepting the limits of computation**: acknowledging the boundaries of machines and algorithms is not failure, but the beginning of wisdom
:::

---

## Code Practice: Implementing Computability and Problem Classification in Python

"Let's use code to explore the boundaries of computability," said Mr. Pallas's Cat. "From Turing machine simulation to the halting problem proof — using code to make abstract concepts concrete."

### Turing Machine Simulator (Simplified)

```python
class SimpleTuringMachine:
    """A simplified Turing machine simulator
    
    A Turing machine consists of:
    - An infinitely long tape (simulated with a Python list for its finite portion)
    - A read/write head (current position)
    - A state register (current state)
    - A rule table (state transition function)
    """
    
    def __init__(self, initial_tape=''):
        """Initialize the Turing machine
        
        Parameters:
            initial_tape: initial tape content (string, e.g., '1011')
        """
        # Tape: simulated with a list, each element is a character
        self.tape = list(initial_tape)
        # Read/write head position
        self.head_position = 0
        # Current state: 'q0' is the initial state, 'halt' is the halting state
        self.current_state = 'q0'
        # Rule table: format (current_state, current_symbol) -> (new_state, new_symbol, move_direction)
        self.rules = {}
        
    def add_rule(self, current_state, read_symbol, new_state, write_symbol, move):
        """Add a state transition rule
        
        Parameters:
            current_state: current state
            read_symbol: symbol being read
            new_state: new state
            write_symbol: symbol to write
            move: direction to move ('L' left, 'R' right, 'S' stay)
        """
        self.rules[(current_state, read_symbol)] = (new_state, write_symbol, move)
    
    def get_current_symbol(self):
        """Get the symbol at the current position of the read/write head
        
        If the position exceeds the current tape range, return blank symbol '_'
        """
        if 0 <= self.head_position < len(self.tape):
            return self.tape[self.head_position]
        else:
            # Extend the tape (simulating an infinite tape)
            if self.head_position >= len(self.tape):
                self.tape.extend(['_'] * (self.head_position - len(self.tape) + 1))
            else:  # head_position < 0
                # Insert blank symbols at the beginning
                insert_count = -self.head_position
                self.tape = ['_'] * insert_count + self.tape
                self.head_position = 0  # adjust position
            return '_'
    
    def step(self):
        """Execute one step of computation
        
        Returns:
            True: continue executing
            False: halt
        """
        # If already in the halting state, do nothing
        if self.current_state == 'halt':
            return False
        
        # Read the current symbol
        current_symbol = self.get_current_symbol()
        
        # Look up the rule
        key = (self.current_state, current_symbol)
        if key not in self.rules:
            # No rule defined — halt
            self.current_state = 'halt'
            return False
        
        # Execute the rule
        new_state, write_symbol, move = self.rules[key]
        
        # Write the new symbol
        if 0 <= self.head_position < len(self.tape):
            self.tape[self.head_position] = write_symbol
        
        # Update state
        self.current_state = new_state
        
        # Move the read/write head
        if move == 'L':
            self.head_position -= 1
        elif move == 'R':
            self.head_position += 1
        # 'S' means stay
        
        return True
    
    def run(self, max_steps=1000):
        """Run the Turing machine until it halts or reaches the maximum number of steps
        
        Parameters:
            max_steps: maximum number of execution steps
        
        Returns:
            number of steps executed
        """
        steps = 0
        while self.step() and steps < max_steps:
            steps += 1
            # Prevent infinite loop
            if steps >= max_steps:
                print(f"Warning: reached max steps {max_steps}, may be in an infinite loop")
                break
        return steps
    
    def get_tape_string(self, visible_range=10):
        """Get a string representation of the tape content
        
        Parameters:
            visible_range: how many characters on each side of the read/write head to display
        
        Returns:
            tape content as a string
        """
        start = max(0, self.head_position - visible_range)
        end = min(len(self.tape), self.head_position + visible_range + 1)
        tape_part = self.tape[start:end]
        
        # Mark the read/write head position
        result = []
        for i, symbol in enumerate(tape_part):
            pos = start + i
            if pos == self.head_position:
                result.append(f"[{symbol}]")
            else:
                result.append(symbol)
        
        return ''.join(result)
```

### Example: Binary Addition with a Turing Machine

```python
def create_binary_adder():
    """Create a Turing machine that performs binary addition
    
    Function: compute the sum of two binary numbers
    Input format: a#b (e.g., '101#110' means 5+6)
    Output: binary representation of a+b
    """
    tm = SimpleTuringMachine('101#110')  # 5 + 6 = 11 (binary 1011)
    
    # Rule 1: scan right to left, process corresponding digits
    # State q0: initial state, move right to find '#'
    tm.add_rule('q0', '0', 'q0', '0', 'R')
    tm.add_rule('q0', '1', 'q0', '1', 'R')
    tm.add_rule('q0', '#', 'q1', '#', 'R')
    
    # State q1: move right to find the end
    tm.add_rule('q1', '0', 'q1', '0', 'R')
    tm.add_rule('q1', '1', 'q1', '1', 'R')
    tm.add_rule('q1', '_', 'q2', '_', 'L')  # reached the end, start processing leftward
    
    # State q2: process addition leftward
    tm.add_rule('q2', '0', 'q3', '0', 'L')  # current digit is 0
    tm.add_rule('q2', '1', 'q4', '1', 'L')  # current digit is 1
    
    # States q3-q9: handle various carry cases (simplified implementation)
    # ... (a full implementation would require more states and rules)
    
    # Final state: halt
    tm.add_rule('halt', '_', 'halt', '_', 'S')
    
    return tm

# Test the Turing machine simulator
print("Turing Machine Simulator Test")
print("=" * 50)

# Create a simple Turing machine: flip all 0s to 1s and all 1s to 0s
tm = SimpleTuringMachine('1011001')
tm.add_rule('q0', '0', 'q0', '1', 'R')
tm.add_rule('q0', '1', 'q0', '0', 'R')
tm.add_rule('q0', '_', 'halt', '_', 'S')

print(f"Initial tape: {tm.get_tape_string()}")
print(f"Initial state: {tm.current_state}")
print(f"Head position: {tm.head_position}")

steps = tm.run(max_steps=100)
print(f"Executed {steps} steps")
print(f"Final tape: {tm.get_tape_string()}")
print(f"Final state: {tm.current_state}")
```

### Simulating the Halting Problem in Python

```python
def halting_problem_simulation():
    """Demonstration of the halting problem
    
    Suppose we try to write a function halts(P, x) that determines whether program P halts on input x.
    We will show why such a function cannot exist.
    """
    print("\n" + "=" * 50)
    print("Halting Problem Simulation")
    print("=" * 50)
    
    # Hypothetical "halting oracle" function (cannot actually exist)
    def halts(program_code, input_data):
        """Hypothetical function: determines whether a program halts on an input
        
        Note: this function cannot actually be implemented correctly!
        We only use it to demonstrate the paradox.
        """
        # In practice, we cannot implement this function
        # Here we only simulate its "hypothetical existence"
        print(f"  Calling halts(program, '{input_data}')...")
        
        # Simple simulation: if the program contains 'while True:', assume it does not halt
        # This is only for demonstration; in reality, correct judgment is impossible
        if 'while True:' in program_code:
            print(f"  Detected infinite loop, judging as 'does not halt'")
            return False
        else:
            print(f"  No obvious infinite loop detected, judging as 'halts'")
            return True
    
    # A program that loops infinitely
    infinite_loop_code = '''
def infinite_program(x):
    while True:
        x = x + 1
    return x
'''
    
    # A program that terminates normally
    finite_program_code = '''
def finite_program(x):
    return x * 2
'''
    
    # Test the hypothetical halts function
    print("\nTest 1: judging a normal program")
    result1 = halts(finite_program_code, "test")
    print(f"  Result: {result1}")
    
    print("\nTest 2: judging an infinite-loop program")
    result2 = halts(infinite_loop_code, "test")
    print(f"  Result: {result2}")
    
    # Construct the paradox program
    print("\n" + "=" * 50)
    print("Constructing the Paradox Program: Mimicking the Halting Problem Proof")
    print("=" * 50)
    
    # Code for the paradox program P
    paradox_program_code = '''
def paradox_program(input_str):
    # Get its own source code (simplified representation)
    my_code = get_my_code()  
    
    # Call the hypothetical halts function to check whether it itself halts
    if halts(my_code, input_str):
        # If halts says I will halt... I enter an infinite loop!
        while True:
            pass  # infinite loop
    else:
        # If halts says I won't halt... I halt immediately!
        return "I halted!"
'''
    
    print("\nLogic of the paradox program:")
    print("1. Program P first calls halts(P, x) to check whether it itself halts")
    print("2. If halts returns True (saying P will halt), then P enters an infinite loop")
    print("3. If halts returns False (saying P won't halt), then P halts immediately")
    print("\nA contradiction emerges:")
    print("- If halts correctly judges that P will halt → P actually does NOT halt (enters infinite loop)")
    print("- If halts correctly judges that P won't halt → P actually DOES halt (returns immediately)")
    print("\nConclusion: no algorithm exists that can correctly determine, for all programs, whether they halt.")

# Run the halting problem simulation
halting_problem_simulation()
```

### Self-Reference and Paradox

```python
def self_reference_paradox():
    """Self-reference paradox demonstration
    
    Shows the paradoxes that arise when a system tries to judge itself.
    """
    print("\n" + "=" * 50)
    print("Self-Reference Paradox Demonstration")
    print("=" * 50)
    
    # The Liar Paradox
    print("\n1. The Liar Paradox:")
    liar_statement = "This statement is false."
    print(f"  Statement: '{liar_statement}'")
    print("  Analysis:")
    print("  - If this statement is true → then what it says, that it is false, must hold → contradiction")
    print("  - If this statement is false → then what it says, that it is false, is itself false → it is true → contradiction")
    
    # The Barber Paradox
    print("\n2. The Barber Paradox:")
    print("  Rule: 'The barber shaves all those, and only those, who do not shave themselves'")
    print("  Question: does the barber shave himself?")
    print("  Analysis:")
    print("  - If the barber shaves himself → he belongs to 'those who shave themselves' → he should not shave himself")
    print("  - If the barber does not shave himself → he belongs to 'those who do not shave themselves' → he should shave himself")
    
    # Self-reference in computing
    print("\n3. Self-Reference in Computing:")
    
    # Example: self-modifying code
    self_modifying_code = '''
def self_modifying():
    # Get its own source code
    import inspect
    code = inspect.getsource(self_modifying)
    
    # Modify itself (conceptual demonstration)
    print("Original code length:", len(code))
    # Modifying runtime code in practice is complex; this is just a concept
'''
    
    print("  The possibility of self-modifying code:")
    print("  - A program can read its own source code")
    print("  - A program can modify its own behavior")
    print("  - This leads to self-replicating programs such as viruses and worms")
    
    # Gödel's incompleteness as a programming analogy
    print("\n4. Gödel's Incompleteness as a Programming Analogy:")
    print("  Consider a mathematical proof-checking program:")
    print("  - Input: mathematical statement + proof")
    print("  - Output: 'valid' or 'invalid'")
    print("  Gödel constructed a statement G: 'This statement has no proof'")
    print("  If G has a proof → G is false (because it says it has no proof)")
    print("  If G has no proof → G is true (because it says it has no proof)")
    print("  Conclusion: there exist statements that can be neither proved nor disproved")
    
    print("\nMr. Pallas's Cat's reflections:")
    print("Self-reference reveals the fundamental limits of logical systems.")
    print("When we try to use a system to judge the system itself, we always encounter a boundary.")
    print("This is not failure, but the deepening of understanding.")

# Run the self-reference demonstration
self_reference_paradox()

print("\n" + "=" * 50)
print("Computability Reflection Questions:")
print("1. Can you design a program that determines whether another program outputs 'Hello, World!'?")
print("2. Why can virus detection software never detect all viruses?")
print("3. If human thought is a form of computation, what do Gödel's theorems imply for human cognition?")
```

"Through this code," Mr. Pallas's Cat summarized, "we not only understand the operation of Turing machines, but also grasp **the boundaries of computation**. Knowing what is computable lets us focus on possible problems; knowing what is uncomputable helps us avoid futile struggles. This is the mark of maturing rational thought."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Simulation experiment**: try simulating a simple Turing machine with paper and pen. Design a rule table that makes it compute binary addition on the tape. Experience the abstract meaning of the infinite tape.

2. **Paradox construction**: mimicking the proof of the halting problem, construct disproofs for other "undecidable" problems. For example: can we design a program `CORRECT(P, w)` that determines whether program P runs correctly on input w?

3. **Real-world correspondences**: find "undecidable" phenomena around you. For example: can you write a program that determines whether another piece of code contains malicious behavior? (Hint: this has deep connections to the halting problem.)

### Historical Investigation (for Little Seal)
1. **Figure study**: the relationship between Alan Turing's wartime work (breaking the Enigma code) and his theoretical work. How did theoretical insight influence practical application?

2. **Intellectual lineage**: from Leibniz's "universal characteristic" to Hilbert's decision problem, to Turing's theory of computability — how did humanity's dream of "mechanical reasoning" evolve?

3. **Cultural impact**: how did Turing's life and tragedy influence the development of computer science? How was his contribution recognized during his life and after his death?

### Integrated Reflection
1. **Philosophical reflection**: if human thought is also a form of "computation," what do Gödel's and Turing's results imply for human cognition? Are there thinking problems that humans also cannot solve?

2. **Imagining the limits**: suppose there existed "hyper-Turing machines" (such as certain models of quantum computers) — could they solve the halting problem? How would this change our understanding of computation?

3. **Creative exercise**: design a "computability fable" — use a story to explain the difference between computable problems, hard problems, and undecidable problems. (Hint: compare them to puzzles of different difficulty levels.)

4. **Ethical consideration**: when AI systems face undecidable ethical dilemmas (such as the "trolley problem" in autonomous driving), what guidance can Turing's theory offer?

---

## Coming Up Next

Night had fallen deeply; the sound of ships' horns drifted faintly from the Pearl River. Kangle Garden was completely immersed in darkness, with only the lamplight in the Black Stone House study shining like a lone island.

Piglet was still drawing state-transition diagrams for Turing machines on paper, trying to understand how finite rules on an infinite tape could generate infinite possibilities.

Little Seal closed *Alan Turing: The Enigma* and said quietly, "Professor, if some problems are uncomputable, what do we do? Give up?"

Mr. Pallas's Cat paused in tidying the tea set. "No," he said slowly. "Quite the opposite. Knowing where the boundary lies allows us to **choose our battleground wisely**."

He walked to the window and gazed out at Kangle Garden in the night. "In the next chapter, we return to the domain of the computable, to explore the simplest search strategies — **linear and binary**."

Piglet looked up. "Like searching for a book on a sorted shelf?"

"Exactly," Mr. Pallas's Cat smiled. "From the limits of computation back to the starting point of computation, from the undecidable back to the simplest, most fundamental algorithms. Sometimes, the simplest strategy is the most powerful."

Outside the window, the bell tower of the Sun Yat-sen University Library chimed ten. The sound of the bell echoed through the night, as if measuring time — the one dimension from which no computation can escape.

---

> **Piglet's note**: I tried writing a small program to check whether another program contains an infinite loop, only to find that it itself fell into an infinite loop! Is this the magic of self-reference?
>
> **Little Seal's note**: I read about Turing's life and was stunned that he made such profound contributions at age 24. The insight of genius often blooms at its youngest.
>
> **Mr. Pallas's Cat's closing words**: Remember, knowing what is uncomputable is as important as knowing what is computable. The power of reason lies not only in what we can do, but in knowing what we cannot do. We'll take it slow — understanding is what matters most.

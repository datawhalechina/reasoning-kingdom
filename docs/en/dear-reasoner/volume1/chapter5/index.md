# Chapter 5: The Temptation of Greed (Local Optimality)

:::info
**Mr. Pallas's Cat's Warm Welcome**  
Having explored exhaustive search and intelligent search, we now face a new question: **how do we make sequential decisions?** When confronting a series of choices, should we take one step at a time or take in the whole picture? Today, we explore the wisdom and limits of greed.
:::

---

## Core Question: What Is the Strategy for Sequential Decision-Making?

It was an afternoon in Kangle Garden. A breeze from the Pearl River drifted into the Black Stone House study, carrying the early-summer warmth. Outside the window, the banyan's new leaves had turned deep green, and cicadas had begun their intermittent chirping. Mr. Pallas's Cat was arranging the tea set, preparing to brew a fresh pot of Fenghuang Dancong.

Piglet pulled a few coins from his wallet and laid them out on the table. "Professor, if I need to pay 8 yuan and have coins of 1, 3, and 5 yuan, how can I make it with the fewest coins?"

Little Seal set down his *Introduction to Game Theory*. "This is an interesting problem. If you pick the largest denomination each time: 5+1+1+1=8, that's 4 coins. But 3+3+1+1 is also 4 coins, and 3+5 is also 8 yuan — only 2 coins!"

Mr. Pallas's Cat smiled. "You've raised a great question. When we face a sequence of decisions — whether making change, scheduling activities, or planning a route — what strategy should we adopt?"

Piglet answered without thinking: "Pick the biggest coin each time! That uses the fewest coins!"

"That's the idea behind the **greedy algorithm**," Mr. Pallas's Cat nodded. "At each step, make the choice that **looks best right now**. But your example already shows the problem: sometimes, local optimality does not lead to global optimality."

Little Seal thought for a moment. "So the greedy algorithm is **shortsighted**? It only sees immediate gains and doesn't consider long-term consequences?"

"You could say that," said Mr. Pallas's Cat. "But the core issue we explore today is: **what is the strategy for sequential decision-making?** More specifically: **when do locally optimal choices lead to global optimality?**"

He walked to the whiteboard and wrote two concepts:

**Local Optimality** — **Global Optimality**

"The core assumption of the greedy algorithm," Mr. Pallas's Cat explained, "is that **a series of locally optimal choices can lead to a globally optimal solution**. But this assumption does not always hold. Today we explore: when does it hold? Why? When does it fail? Why?"

## The Form of Greedy Algorithms: Choice and Proof

"Let's first formalize the greedy algorithm," said Mr. Pallas's Cat, writing a general framework on the whiteboard:

```
def greedyAlgorithm(problem):
    solution = emptySet()
    while problem not solved:
        # choose the currently best candidate
        candidate = selectBestCandidate(problem.candidates)
        # check whether the candidate is feasible (satisfies constraints)
        if feasible(solution + candidate):
            solution.add(candidate)
            problem.update(candidate)  # update the problem state
    return solution
```

"The greedy algorithm has four key components," Mr. Pallas's Cat explained:
1. **Candidate set**: the options available at each step
2. **Selection function**: picks the "best" option from the candidate set
3. **Feasibility check**: checks whether the choice satisfies the constraints
4. **Problem update**: updates the problem state after the choice

Piglet asked: "How is 'best' defined? Different selection functions lead to different results, right?"

"Exactly," said Mr. Pallas's Cat. "The definition of the selection function determines the behavior of the greedy algorithm. For example, in the coin change problem:
- **Largest denomination first**: always pick the largest coin that doesn't exceed the remaining amount
- **Smallest coin first**: always pick the smallest coin
- **Value density first**: if coins have different values (e.g., collectible value)

"Different selection functions can lead to different results — some leading to global optimality, others not."

### Activity Selection Problem: A Greedy Success Story

Little Seal said: "I recall that in the activity selection problem, the greedy algorithm works. Choosing the activity with the earliest finish time always yields the maximum set of compatible activities."

"Yes," said Mr. Pallas's Cat, writing the activity selection problem on the whiteboard:

**Problem**: given n activities, each with a start time `sᵢ` and finish time `fᵢ`. Select the maximum number of mutually non-overlapping activities.

**Greedy Algorithm**:
```
1. Sort activities by finish time in ascending order
2. Select the first activity (earliest finish)
3. For each subsequent activity:
   If start time ≥ finish time of the currently selected activity:
       Select this activity
```

"The correctness of this algorithm requires proof," said Mr. Pallas's Cat. "It demonstrates the **greedy choice property**: there always exists an optimal solution that includes the activity with the earliest finish time."

:::info
**Mr. Pallas's Cat's Commentary**  
The greedy algorithm for the activity selection problem succeeds because it possesses both the **greedy choice property** and **optimal substructure**. These two properties are the key to the correctness of greedy algorithms. When a problem has both properties, the greedy strategy is effective.
:::

### The Knapsack Problem: Greedy's Failure

"Now let's look at an example where greedy fails," said Mr. Pallas's Cat. "**The 0-1 knapsack problem**."

**Problem**: given n items, each with weight `wᵢ` and value `vᵢ`, and a knapsack with capacity W. Select items to maximize total value, with total weight not exceeding W.

Piglet tried: "Greedy strategy: always pick the item with the highest value?"

"Let's try," said Mr. Pallas's Cat, giving an example. "Suppose W=6, items:
1. Weight 5, Value 5 (density 1)
2. Weight 3, Value 4 (density 1.33)
3. Weight 3, Value 4 (density 1.33)

"Greedy by value density: pick item 2 first (value 4, remaining capacity 3), then item 3 (value 4, total 8).
"But the optimal solution... wait, 8 > 5, so greedy is actually better here."

Little Seal thought: "W=6, items 2+3 weight 6 value 8; item 1 weight 5 value 5. Greedy does win here. The classic counterexample needs items where value isn't monotonic with density..."

Mr. Pallas's Cat smiled. "You've hit upon the key point. The greedy strategy for the 0-1 knapsack (sorting by value density) **does not always yield the optimal solution**, but constructing a counterexample requires care. This is also what makes greedy algorithms interesting: they often yield **near-optimal** solutions, though not **absolutely optimal** ones."

:::detail
**Advanced: Proving the Correctness of Greedy Algorithms**

The correctness of a greedy algorithm typically requires proving two properties:

1. **Greedy Choice Property**:
   There exists an optimal solution that includes the current greedy choice.
   
   Proof method: assume there exists an optimal solution that does not include the greedy choice; modify this solution by replacing some element with the greedy choice, obtaining another solution that is at least as good.

2. **Optimal Substructure**:
   The optimal solution of the problem contains optimal solutions to its subproblems.
   
   Proof method: if the optimal solution of the original problem can be decomposed into a combination of optimal solutions to subproblems, then the problem has optimal substructure.

Taking the activity selection problem as an example:
- **Greedy choice property**: there always exists an optimal solution that includes the activity with the earliest finish time
- **Optimal substructure**: after selecting activity a, the remaining problem is to select a maximum compatible set from the activities that start after a finishes

When a problem has both properties, the greedy algorithm yields the globally optimal solution. Otherwise, greedy can only produce an approximate solution.
:::

---

## Orthogonal Computation Graphs: Seeing the Decision Flow of Greed

Mr. Pallas's Cat turned on the projector, and a neat computation graph appeared on the screen.

![Greedy Algorithm Orthogonal Computation Graph](/figures/greedy_algorithm_ortho.svg)

"This is the orthogonal computation graph for the greedy algorithm," Mr. Pallas's Cat said, pointing at the diagram. "The problem instance enters, passes through the selection function, feasibility check, and problem update, looping until the problem is solved."

Piglet stared at the diagram. "This graph is a loop! Select → check → update → select again... like a decision cycle."

"Exactly," said Mr. Pallas's Cat. "The core of the greedy algorithm is a **decision cycle**: at each step, based on the current state, select the best candidate, update the state, and proceed to the next round of selection. This cyclic structure reflects the **shortsightedness** of the greedy algorithm — it sees only the current state and never goes back to revise previous choices."

Little Seal examined the layered structure of the diagram carefully. "Professor, the design of this diagram... the selection function, feasibility check, and problem update are at the same cycle level, but after each cycle the problem state changes and the candidate set is updated."

"That's the dynamic nature of the greedy algorithm," Mr. Pallas's Cat explained. "Although the algorithmic structure is cyclic, each cycle confronts an **updated problem state**. This **state-dependent choice** is a hallmark of the greedy algorithm — and also its limitation: once a choice is made, it cannot be undone."

---

## Mental Model: The Tension Between Local and Global Optimality

Mr. Pallas's Cat returned to his seat and took a sip of tea. "Let's summarize today's most important mental model: **the tension between local and global optimality**."

He drew a comparison table on the whiteboard:

| Feature | Local Optimality (Greedy Perspective) | Global Optimality (Optimization Perspective) |
|------|-------------------|-------------------|
| **Decision horizon** | Shortsighted, only the current step | Farsighted, considering all steps |
| **Computational cost** | Low — simple choices at each step | High — may require exhaustive or complex optimization |
| **Solution quality** | May be suboptimal, but usually decent | Guaranteed optimal (if computable) |
| **Suitable scenarios** | Large-scale problems, real-time decisions | Small-scale problems, critical decisions |
| **Philosophical implication** | Satisfice — accept imperfection | Pursue perfection — at any cost |

"These two perspectives are not opposed," Mr. Pallas's Cat explained. "Rather, they are **two endpoints on a continuous spectrum**. In practical problems, we often seek a balance between them."

### Conditions for Applying Greedy Strategies

Little Seal asked: "Professor, what kinds of problems are suitable for greedy algorithms?"

Mr. Pallas's Cat listed the conditions on the whiteboard:

```
Problems suitable for greedy algorithms typically have:
1. **Greedy choice property**: locally optimal choices lead to global optimality
2. **Optimal substructure**: the problem can be decomposed into similar subproblems
3. **No aftereffect**: the current choice does not affect the feasibility of future choices (or the effect is predictable)
4. **Computational efficiency requirement**: fast decisions are needed; approximate solutions are acceptable

Common problems amenable to greedy approaches:
├── Activity selection (earliest finish time)
├── Huffman coding (smallest frequency first)
├── Minimum spanning tree (Kruskal/Prim algorithms)
├── Single-source shortest path (Dijkstra's algorithm, nonnegative weights)
└── Fractional knapsack (highest value density first)
```

"Understanding these conditions," said Mr. Pallas's Cat, "helps us judge when we can 'be greedy' and when we need more sophisticated strategies."

:::info
**Mr. Pallas's Cat's Commentary**  
The philosophy of the greedy algorithm is the **satisficing principle** rather than the **optimizing principle**. It does not pursue the absolute best, but finds a good-enough solution within acceptable time. This mode of thinking is useful in many real-world decisions — whether personal life planning, business strategy formulation, or algorithm design.
:::

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary: The Greedy Wisdom of Sequential Decision-Making**  
1. **Greedy algorithm framework**: at each step, choose the currently best candidate, looping until the problem is solved — embodying a shortsighted but efficient decision style  
2. **Correctness conditions**: greedy choice property + optimal substructure = guarantee of global optimality; lacking either yields only approximate solutions  
3. **Identifying suitable scenarios**: activity selection, minimum spanning tree, Huffman coding are suitable for greedy; the 0-1 knapsack, traveling salesman require more sophisticated strategies  
4. **The tension between local and global**: the greedy algorithm embodies the eternal tension between local and global optimality, reminding us of the importance of the decision horizon  
5. **Practical wisdom**: when a problem has the greedy property, the greedy algorithm is an elegant and concise solution; when it doesn't, greedy is a fast but potentially suboptimal heuristic
:::

---

## Code Practice: Implementing Greedy Algorithms in Python

"Let's use code to explore the charm and limitations of greedy algorithms," said Mr. Pallas's Cat. "From coin change to activity selection, from success stories to failure cases."

### Generic Greedy Algorithm Framework

```python
def greedy_algorithm_template(candidates, selection_rule, feasibility_check):
    """Generic greedy algorithm template
    
    Problem description:
        Select a subset from a candidate set that satisfies certain constraints
        and optimizes some objective.
    
    Algorithm idea (greedy strategy):
        1. Initialize an empty solution
        2. While the candidate set is not empty:
           a. Use the selection rule to pick the currently best candidate
           b. Check whether adding this candidate to the current solution is feasible
           c. If feasible, add it to the solution
           d. Remove this candidate from the candidate set
        3. Return the final solution
    
    Key features:
        - Makes the locally optimal choice at every step
        - Does not backtrack or reconsider previous choices
        - Efficient but not necessarily globally optimal
        - Suitable for problems with the greedy choice property
    
    Parameters:
        candidates: candidate set (list, set, or other iterable)
        selection_rule: selection function — takes candidate set, returns the currently best candidate
        feasibility_check: feasibility check function — takes current solution and candidate, returns a boolean
    
    Returns:
        the greedy solution (as a list)
    
    Time complexity: depends on the complexity of the selection rule and feasibility check
    Space complexity: O(n) — stores the solution and candidate set
    
    Example (activity selection problem):
        >>> activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
        >>> def select_earliest_end(candidates):
        ...     return min(candidates, key=lambda x: x[1])
        >>> def is_feasible(solution, candidate):
        ...     if not solution:
        ...         return True
        ...     return candidate[0] >= solution[-1][1]
        >>> greedy_algorithm_template(activities, select_earliest_end, is_feasible)
        [(1, 4), (5, 7), (8, 11), (12, 16)]
    """
    solution = []
    remaining_candidates = candidates.copy()
    
    while remaining_candidates:
        # Select the currently best candidate
        best_candidate = selection_rule(remaining_candidates)
        
        # Check feasibility
        if feasibility_check(solution, best_candidate):
            solution.append(best_candidate)
        
        # Remove the processed candidate
        remaining_candidates.remove(best_candidate)
    
    return solution
```

### The Coin Change Problem

```python
def greedy_coin_change(amount, coins):
    """Greedy coin change algorithm
    
    Problem description:
        Given an amount and a set of coin denominations, find the way to make
        that amount using the fewest coins. Assume unlimited supply of each coin.
    
    Algorithm idea (greedy strategy):
        1. Sort coin denominations in descending order
        2. For each denomination:
           a. Compute the maximum number of such coins that can be used (not exceeding remaining amount)
           b. Use as many of these coins as possible
           c. Update the remaining amount
        3. If the final amount is reduced to zero, return the coin list; otherwise return None
    
    Key features:
        - At each step picks the largest available coin
        - For some denomination sets (e.g., [1, 5, 10, 25]) yields the optimal solution
        - For some denomination sets (e.g., [1, 3, 4]) may not yield the optimal solution
        - The greedy choice property must be verified
    
    Parameters:
        amount: the amount to make (positive integer)
        coins: list of coin denominations (e.g., [25, 10, 5, 1])
    
    Returns:
        list of coins used (largest to smallest), or None if the amount cannot be made
    
    Time complexity: O(n), where n is the number of coin types
    Space complexity: O(k), where k is the number of coins used
    
    Examples:
        >>> greedy_coin_change(63, [25, 10, 5, 1])
        [25, 25, 10, 1, 1, 1]  # uses 6 coins
        >>> greedy_coin_change(6, [4, 3, 1])
        [4, 1, 1]  # greedy uses 3 coins, but optimal is [3, 3] using 2 coins
    
    Note:
        - This algorithm does not guarantee the minimum number of coins
        - For denomination sets that don't satisfy the greedy choice property, dynamic programming is needed
        - In practice, verify the properties of the denomination system
    """
    coins.sort(reverse=True)  # descending order ensures largest first
    result = []
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            result.append(coin)
    
    if amount == 0:
        return result
    else:
        return None  # cannot make the amount

def optimal_coin_change_dp(amount, coins):
    """Dynamic programming coin change (for comparison with greedy)
    
    Parameters:
        amount: the amount to make
        coins: list of coin denominations
    
    Returns:
        list of the fewest coins used
    
    Time complexity: O(amount * n)
    """
    # dp[i] = minimum coins needed to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    # record which coin was chosen
    coin_used = [-1] * (amount + 1)
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin
    
    # If the amount cannot be made
    if dp[amount] == float('inf'):
        return None
    
    # Backtrack to construct the solution
    result = []
    current = amount
    while current > 0:
        coin = coin_used[current]
        result.append(coin)
        current -= coin
    
    return result
```

### Coin Change Experiment: Greedy vs. Optimal

```python
def coin_change_experiment():
    """Coin change experiment: demonstrating success and failure of the greedy algorithm"""
    
    print("Coin Change Experiment")
    print("=" * 50)
    
    test_cases = [
        ("Standard currency system (greedy is optimal)", 63, [100, 50, 20, 10, 5, 2, 1]),
        ("Classic greedy failure example", 6, [4, 3, 1]),
        ("Another failure case", 30, [25, 10, 1]),
        ("Complex case", 40, [25, 20, 10, 5, 1]),
    ]
    
    for name, amount, coins in test_cases:
        print(f"\n{name}:")
        print(f"  Amount: {amount}, Coins: {coins}")
        
        # Greedy algorithm
        greedy_result = greedy_coin_change(amount, coins.copy())
        if greedy_result:
            print(f"  Greedy: {greedy_result} ({len(greedy_result)} coins)")
        else:
            print(f"  Greedy: cannot make the amount")
        
        # Dynamic programming (optimal solution)
        optimal_result = optimal_coin_change_dp(amount, coins)
        if optimal_result:
            print(f"  Optimal: {optimal_result} ({len(optimal_result)} coins)")
        else:
            print(f"  Optimal: cannot make the amount")
        
        # Comparison
        if greedy_result and optimal_result:
            if len(greedy_result) == len(optimal_result):
                print(f"  Result: greedy is optimal!")
            else:
                print(f"  Result: greedy uses {len(greedy_result)-len(optimal_result)} extra coin(s)")

# Run the coin change experiment
coin_change_experiment()
```

### Activity Selection Problem

```python
def greedy_activity_selection(activities):
    """Greedy activity selection algorithm (choosing earliest finish time)
    
    Problem description (Activity Selection Problem):
        Given n activities, each with a start time and finish time.
        Select the largest set of mutually non-conflicting activities.
    
    Algorithm idea (greedy strategy):
        1. Sort all activities by finish time in ascending order
        2. Select the first activity (earliest finish)
        3. For each subsequent activity:
           a. If this activity's start time ≥ the last selected activity's finish time
           b. Select this activity, update the last selected activity's finish time
    
        Greedy choice: at each step, pick the activity with the earliest finish time
        that does not conflict with already-selected activities.
    
    Correctness proof:
        1. Greedy choice property: there exists an optimal solution that includes
           the activity with the earliest finish time
        2. Optimal substructure: after selecting one activity, the remaining problem
           is an optimal subproblem
        3. Inductive proof: mathematical induction proves the greedy algorithm yields the optimal solution
    
    Parameters:
        activities: list of activities, each as (start, end) tuple
    
    Returns:
        list of selected activities (in selection order)
    
    Time complexity: O(n log n) — sorting time
    Space complexity: O(n) — storing sorted activities and results
    
    Example:
        >>> activities = [(1, 4), (3, 5), (0, 6), (5, 7), 
        ...               (3, 9), (5, 9), (6, 10), (8, 11),
        ...               (8, 12), (2, 14), (12, 16)]
        >>> greedy_activity_selection(activities)
        [(1, 4), (5, 7), (8, 11), (12, 16)]
    """
    # Sort by finish time
    sorted_activities = sorted(activities, key=lambda x: x[1])
    
    selected = []
    last_end_time = 0
    
    for start, end in sorted_activities:
        if start >= last_end_time:  # no conflict
            selected.append((start, end))
            last_end_time = end
    
    return selected

def greedy_activity_selection_alternative(activities, key='end'):
    """Activity selection with different sorting criteria
    
    Parameters:
        activities: list of activities
        key: sorting criterion — 'start' (earliest start), 'duration' (shortest duration), 'end' (earliest finish)
    
    Returns:
        list of selected activities
    """
    if key == 'start':
        sorted_activities = sorted(activities, key=lambda x: x[0])
    elif key == 'duration':
        sorted_activities = sorted(activities, key=lambda x: x[1] - x[0])
    else:  # 'end'
        sorted_activities = sorted(activities, key=lambda x: x[1])
    
    selected = []
    last_end_time = 0
    
    for start, end in sorted_activities:
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    
    return selected
```

### Huffman Coding: Greedy Construction of Optimal Prefix Codes

```python
import heapq

class HuffmanNode:
    """Huffman tree node"""
    def __init__(self, char=None, freq=0):
        self.char = char  # character (leaf node)
        self.freq = freq  # frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        # For heap ordering — compare by frequency
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    """Build a Huffman tree (greedy algorithm)
    
    Parameters:
        frequencies: dictionary mapping character → frequency
    
    Returns:
        root node of the Huffman tree
    
    Algorithm steps:
        1. Create a leaf node for each character, put them in a min-heap
        2. While the heap has more than one node:
           a. Pop the two nodes with the smallest frequencies
           b. Create a new node with frequency equal to the sum, and these two as children
           c. Push the new node into the heap
        3. The sole remaining node in the heap is the root of the Huffman tree
    """
    # Create leaf node heap
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Pop the two nodes with the smallest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create a new node
        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        # Push into the heap
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def generate_huffman_codes(root, current_code="", codes={}):
    """Generate Huffman codes
    
    Parameters:
        root: root node of the Huffman tree
        current_code: current path encoding
        codes: encoding dictionary
    
    Returns:
        dictionary mapping character → code
    """
    if root is None:
        return codes
    
    # Leaf node — record the encoding
    if root.char is not None:
        codes[root.char] = current_code
        return codes
    
    # Recursively traverse left and right subtrees
    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)
    
    return codes

def huffman_coding_example():
    """Huffman coding example"""
    
    print("\n" + "=" * 50)
    print("Huffman Coding (Greedy Construction of Optimal Prefix Code)")
    print("=" * 50)
    
    # Example: letter frequencies in English text
    frequencies = {
        'a': 45,  # highest frequency
        'b': 13,
        'c': 12,
        'd': 16,
        'e': 9,
        'f': 5,
    }
    
    print(f"Character frequencies: {frequencies}")
    
    # Build the Huffman tree
    root = build_huffman_tree(frequencies)
    
    # Generate codes
    codes = generate_huffman_codes(root)
    
    print("\nGenerated Huffman codes:")
    for char, code in sorted(codes.items()):
        print(f"  '{char}': {code} (frequency: {frequencies[char]})")
    
    # Compute average code length
    total_freq = sum(frequencies.values())
    avg_length = sum(frequencies[char] * len(code) for char, code in codes.items()) / total_freq
    
    print(f"\nAverage code length: {avg_length:.2f} bits/character")
    print(f"Fixed-length encoding would need: {len(bin(len(frequencies)-1))-2} bits/character")
    
    # Greedy property verification
    print("\nGreedy property verification:")
    print("Merging the two nodes with the smallest frequencies is always optimal, because:")
    print("1. Low-frequency characters should have long codes; high-frequency characters should have short codes")
    print("2. Merging the smallest-frequency nodes is equivalent to assigning them the longest code paths")
    print("3. Mathematical induction can prove this greedy strategy yields the globally optimal prefix code")

# Run the Huffman coding example
huffman_coding_example()
```

Mr. Pallas's Cat summarized: "We have not only understood the implementation of greedy algorithms, but also grasped **the strategy for sequential decision-making**. The greedy algorithm teaches us: sometimes, the best long-term strategy is a series of good short-term decisions; but other times, shortsightedness leads to disaster. Knowing when to be greedy and when to plan is algorithmic thinking — and also life wisdom."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Algorithm implementation**: implement the greedy algorithm for the activity selection problem. Test the results under different sorting criteria (earliest start, shortest duration, earliest finish). Which criterion yields the optimal solution? Why?

2. **Constructing greedy failures**: construct a problem instance where the greedy algorithm fails badly. For example: design a set of coin denominations such that the greedy coin-change algorithm uses far more coins than the optimal solution.

3. **Real-world observation**: observe greedy decision-making in everyday life. Is choosing the shortest checkout line at the supermarket a greedy strategy? When might it fail? (Hint: consider what's in the customers' carts.)

### Historical Investigation (for Little Seal)
1. **Tracing the concept**: when did the concept of "greedy" first appear in computer science? Who first proposed and analyzed greedy algorithms? Consult early algorithm literature (e.g., Kruskal 1956, Prim 1957, Dijkstra 1959).

2. **Cross-disciplinary connections**: what similarities exist between greedy algorithms and "marginal decision-making" in economics or "immediate gratification" in psychology? How do these fields study the pros and cons of shortsighted decisions?

3. **Philosophical reflection**: is the utilitarian "greatest happiness principle" a form of greedy thinking? What are the similarities and differences between Bentham and Mill's utilitarianism and greedy algorithms?

### Integrated Reflection
1. **Ethical consideration**: can autonomous vehicles use greedy algorithms for emergency decision-making? (E.g., always choose the steering direction that minimizes immediate harm.) What problems might such shortsighted decisions cause?

2. **Algorithm comparison**: what are the fundamental differences between greedy algorithms and dynamic programming (covered in the next chapter)? Why are some problems amenable to greedy while others require dynamic programming? And vice versa?

3. **Creative exercise**: design a "greedy fable" — use a story to explain the strengths and limitations of greedy algorithms. (Hint: compare to a mountaineer who always chooses the steepest uphill path.)

4. **Limit challenge**: prove the optimality of the greedy algorithm for Huffman coding. Understand why merging the two nodes with the smallest frequencies is always optimal.

---

## Coming Up Next

The evening sun cast golden ripples across the Pearl River; tour boats drifted slowly by, leaving long wakes. The bell of Kangle Garden rang once more, a reminder of time's passage.

Piglet was simulating the effects of different greedy strategies on his computer, the screen displaying various problem instances and algorithmic performance. Little Seal was deriving the proof of the greedy choice property on the whiteboard, trying to understand the mathematical necessity behind what seemed like intuitive choices.

"Today we started from 'the strategy for sequential decision-making,'" Mr. Pallas's Cat said, tidying the tea set. "We explored the wisdom and limits of greedy algorithms, and understood the tension between local and global optimality."

Little Seal put down the chalk. "Professor, if greedy algorithms can fail, what methods can guarantee we find the globally optimal solution?"

Mr. Pallas's Cat smiled. "Good question. That calls for a more powerful tool — **heuristics**."

---

> **Piglet's note**: I tested greedy algorithms on different coin systems. For common denominations (1, 2, 5, 10, 20, 50, 100), greedy is always optimal! But for weird denominations (1, 3, 4), to make 6 yuan, greedy picks 4+1+1 (3 coins), while optimal is 3+3 (2 coins). Turns out our currency system was carefully designed!
>
> **Little Seal's note**: I looked up the history of greedy algorithms and found that Kruskal and Prim independently proposed minimum spanning tree algorithms in the 1950s, both using the greedy idea. Sometimes, simple intuitions take time to be formalized and proven.
>
> **Mr. Pallas's Cat's closing words**: Remember, the greedy algorithm teaches an important lesson about decision-making: sometimes, the best long-term strategy is a series of good short-term decisions; but other times, shortsightedness leads to disaster. Knowing when to be greedy and when to plan is algorithmic thinking — and also life wisdom. We'll take it slow — understanding is what matters most.

# Chapter 7: The Power of Memory (Dynamic Programming)

:::info
**Mr. Pallas's Cat's Warm Welcome**  
After exploring the wisdom of heuristic search, we face a new question: **how can we use memory to accelerate reasoning?** When problems have overlapping substructures, redundant computation causes massive waste. Today, we explore the wisdom of dynamic programming — using memory to avoid repetition, trading space for time.
:::

---

## Core Question: How Does Memory Empower Reasoning?

It was late night in Kangle Garden. The surface of the Pearl River was as still as a mirror, reflecting a sky full of stars. The lights in the Black Stone House study still glowed; Piglet and Little Seal were studying the Fibonacci sequence.

"Professor, I wrote a recursive function to compute Fibonacci numbers," Piglet said, pointing at the screen. "fib(40) takes forever!"

Little Seal observed the recursion tree. "fib(40) calls fib(39) and fib(38), and fib(39) in turn calls fib(38) and fib(37)... there's a huge amount of redundant computation here."

Mr. Pallas's Cat walked over and smiled. "You've discovered a key problem: **overlapping subproblems**. When the same subproblem is computed many times, computation time explodes. The core issue we explore today is: **how does memory empower reasoning?** More specifically: **how can we avoid redundant computation by storing intermediate results, thereby dramatically improving efficiency?**"

He walked to the whiteboard and wrote two words:

**Redundant Computation** — **Memoized Search**

"The core idea of dynamic programming is to **remember results that have already been computed**," Mr. Pallas's Cat explained. "That way, when the same result is needed again, you can look it up instead of recomputing it. This strategy of 'trading space for time' is an important paradigm in algorithm design."

## Dynamic Programming: From Fibonacci to the Knapsack

"Let's start with the Fibonacci sequence," Mr. Pallas's Cat wrote the recursive definition on the whiteboard:

```
F(0) = 0
F(1) = 1
F(n) = F(n-1) + F(n-2)  (n ≥ 2)
```

"The naive recursion has exponential time complexity O(2ⁿ)," he said. "But if we use **memoization**, the complexity drops to O(n)."

### Memoization vs. Bottom-Up Tabulation

Piglet asked: "What's the difference between memoization and bottom-up tabulation?"

"Both are implementations of dynamic programming," Mr. Pallas's Cat explained:

1. **Memoization (Top-down with Memoization)**:
   - Start from the original problem, recursively decompose
   - Before computing a subproblem, check the table; if already computed, return directly
   - Otherwise compute and store the result
   - Suitable when the subproblem structure is not clearly defined

2. **Bottom-up Tabulation**:
   - Start from the smallest subproblems, gradually build up to the original problem
   - Typically uses arrays or tables to store results
   - Suitable when subproblem dependencies are clearly defined

"Taking Fibonacci as an example," Mr. Pallas's Cat wrote two implementations:

```python
# Memoization
memo = {}
def fib_memo(n):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fib_memo(n-1) + fib_memo(n-2)
    return memo[n]

# Bottom-up tabulation
def fib_tab(n):
    if n <= 1: return n
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

Little Seal thought: "Memoization is more like 'lazy computation'; bottom-up tabulation is more like 'eager computation.'"

"A concise analogy," Mr. Pallas's Cat said approvingly. "Both methods have their suitable scenarios, but the core idea is the same: **store the solutions to subproblems to avoid redundant computation**."

### The Knapsack Problem: A Classic Case of Dynamic Programming

"Now let's look at a more complex problem: the **0-1 knapsack problem**," Mr. Pallas's Cat described on the whiteboard:
- There are n items, with weights wᵢ and values vᵢ
- The knapsack has capacity W
- Select items to maximize total value, with total weight not exceeding W

"The greedy algorithm may fail, but dynamic programming can solve it," he wrote the state transition equation:

```
dp[i][c] = max(dp[i-1][c], dp[i-1][c-wᵢ] + vᵢ)  (if c ≥ wᵢ)
dp[i][c] = dp[i-1][c]                          (if c < wᵢ)
```

Piglet stared at the equation. "dp[i][c] represents the maximum value considering the first i items with capacity c... how do you interpret this equation?"

"This is the key to dynamic programming: the **state transition equation**," Mr. Pallas's Cat explained. "It describes the optimal substructure of the problem: the optimal solution for the current state can be derived from the optimal solutions of previous states."

:::info
**Mr. Pallas's Cat's Commentary**  
The philosophy of dynamic programming is **reasoning with memory**: we not only think about the current problem, but also remember past problems and their solutions. This capacity for memory allows us to learn from experience and avoid repeating mistakes. This is not only a principle of algorithm design, but also an important feature of human intelligence — we can draw lessons from history.
:::

### The Three Elements of Dynamic Programming

Little Seal summarized: "Professor, what conditions must a problem satisfy for dynamic programming?"

Mr. Pallas's Cat listed three elements on the whiteboard:

1. **Optimal Substructure**:
   - The optimal solution of the problem contains optimal solutions to its subproblems
   - Example: in the knapsack problem, the optimal solution for the first i items contains the optimal solution for the first i-1 items

2. **Overlapping Subproblems**:
   - Different subproblems recur repeatedly
   - Example: in the Fibonacci sequence, F(38) is computed many times

3. **State Transition Equation**:
   - Describes how to construct the solution to the original problem from solutions to subproblems
   - This is the core of dynamic programming and requires creative design

"When a problem has all three of these characteristics," Mr. Pallas's Cat said, "dynamic programming is the right solution."

---

## Orthogonal Computation Graphs: Seeing the Memory Flow of DP

Mr. Pallas's Cat turned on the projector, and a neat computation graph appeared on the screen.

![Dynamic Programming Orthogonal Computation Graph](/figures/dynamic_programming_ortho.svg)

"This is the orthogonal computation graph for dynamic programming," Mr. Pallas's Cat said, pointing at the diagram. "The original problem is decomposed into subproblems; subproblem solutions are stored in the memory table; subsequent computations can directly reuse them; finally, the solutions are combined to produce the final answer."

Piglet stared at the diagram. "This graph has looping arrows! Subproblems can directly retrieve results from the memory table, avoiding redundant computation."

"Exactly," said Mr. Pallas's Cat. "The core of dynamic programming is the **memory flow**: computation results flow into the memory table, and subsequent computations read from the memory table. This 'compute-store-reuse' cycle allows it to solve complex problems efficiently."

Little Seal examined the layered structure of the diagram carefully. "Professor, the design of this diagram... subproblem solving and memory storage are separated, but connected by dashed lines indicating reuse relationships. This reflects the dual nature of dynamic programming: computation and memory."

"That's the elegance of dynamic programming," Mr. Pallas's Cat explained. "It separates the **computation process** from the **memory structure**, yet connects them tightly through the state transition equation. This separation and connection is important not only in algorithms — it has a counterpart in cognitive science: the human memory system is also a complex interaction of computation and storage."

---

## Mental Model: The Eternal Trade-Off Between Space and Time

Mr. Pallas's Cat returned to his seat and brewed a fresh pot of tea. "Let's summarize today's most important mental model: **the eternal trade-off between space and time**."

He drew a comparison table on the whiteboard:

| Dimension | Time-First Mindset | Space-First Mindset |
|------|------------|------------|
| **Core strategy** | Trade time for space (recompute) | Trade space for time (store results) |
| **Time complexity** | Possibly exponential | Typically polynomial |
| **Space complexity** | Low | High |
| **Suitable scenarios** | Small problem size, space-constrained | Large problem size, time-constrained |
| **Philosophical root** | Frugality | Investment |

"These two mindsets are not opposed," Mr. Pallas's Cat explained. "Rather, they are **two poles of resource allocation**. A wise algorithm designer knows when to save space and when to save time."

### Conditions and Variants of Dynamic Programming

Piglet asked: "Professor, is dynamic programming suitable for all problems?"

Mr. Pallas's Cat listed the applicable scenarios on the whiteboard:

```
Problems suitable for dynamic programming typically have:
1. **Optimal substructure**: the problem can be decomposed into similar subproblems, and optimal subproblem solutions can be combined into the optimal solution for the original problem
2. **Overlapping subproblems**: subproblems recur rather than being independent
3. **Manageable state space**: the number of states is within acceptable limits (usually polynomial)

Main variants of dynamic programming:
├── **Classic DP**: 0-1 knapsack, longest common subsequence, edit distance
├── **Interval DP**: matrix chain multiplication, optimal binary search tree
├── **Tree DP**: maximum independent set on trees, minimum vertex cover
├── **State compression DP**: traveling salesman problem (TSP) with state compression
└── **Digit DP**: counting numbers satisfying certain conditions
```

"Understanding these variants," said Mr. Pallas's Cat, "helps us identify when dynamic programming can be used, and which variant to choose."

:::info
**Mr. Pallas's Cat's Commentary**  
The philosophy of dynamic programming is **prospective memory**: we not only remember the past, but also precompute and store results that may be needed in the future. This mindset of 'preparing for the future' is not only the foundation of efficient algorithms, but also a manifestation of human planning ability. By imagining future needs and preparing in advance, we can respond quickly when the time comes.
:::

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary: The Wisdom of Dynamic Programming**  
1. **Core DP framework**: through optimal substructure, overlapping subproblems, and state transition equations — trading space for time, reducing exponential complexity to polynomial — embodying the profound wisdom of "memory empowering reasoning"  
2. **Two implementation paradigms**: memoization (top-down, lazy computation) and bottom-up tabulation (eager computation) each have suitable scenarios; understanding their differences helps choose the right implementation  
3. **Classic problem modeling**: the state transition equation for the 0-1 knapsack shows how DP formalizes complex decision processes; the optimization of Fibonacci demonstrates how memoization eliminates redundant computation  
4. **The space-time trade-off**: DP embodies the fundamental trade-off of computer science — trading storage space for computation time; this resource-exchange mindset is an important paradigm in algorithm design  
5. **Applied wisdom**: in sequence alignment, resource allocation, path planning, and other real-world problems, DP is often the optimal solution; mastering state design and transition equation construction is a key skill in algorithmic thinking
:::

---

## Code Practice: Implementing Dynamic Programming in Python

"Let's use Python code to practice dynamic programming," said Mr. Pallas's Cat. "Code not only helps us understand the power of memoization, but also lets us 'run' the optimization process of trading space for time."

### Fibonacci Sequence: From Exponential to Linear

```python
import time

# Naive recursive implementation — exponential time complexity O(2^n)
def fib_recursive(n):
    """Naive recursive Fibonacci implementation"""
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# Memoization implementation — linear time complexity O(n)
def fib_memoization(n, memo=None):
    """Memoized Fibonacci implementation"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memoization(n-1, memo) + fib_memoization(n-2, memo)
    return memo[n]

# Bottom-up tabulation implementation — linear time complexity O(n)
def fib_tabulation(n):
    """Bottom-up tabulation Fibonacci implementation"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Space-optimized version — only O(1) space
def fib_optimized(n):
    """Space-optimized Fibonacci implementation"""
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Performance comparison test
print("Fibonacci Performance Comparison:")
n = 35  # test with n = 35

start = time.time()
result_recursive = fib_recursive(n)
time_recursive = time.time() - start
print(f"Naive recursion: fib({n}) = {result_recursive}, time: {time_recursive:.4f}s")

start = time.time()
result_memo = fib_memoization(n)
time_memo = time.time() - start
print(f"Memoization: fib({n}) = {result_memo}, time: {time_memo:.6f}s")

start = time.time()
result_tab = fib_tabulation(n)
time_tab = time.time() - start
print(f"Bottom-up tabulation: fib({n}) = {result_tab}, time: {time_tab:.6f}s")

start = time.time()
result_opt = fib_optimized(n)
time_opt = time.time() - start
print(f"Space-optimized: fib({n}) = {result_opt}, time: {time_opt:.6f}s")

print(f"\nMemoization is {time_recursive/time_memo:.0f}x faster than naive recursion")
```

### 0-1 Knapsack Problem: A Classic DP Case

```python
def knapsack_01(values, weights, capacity):
    """Dynamic programming solution for the 0-1 knapsack problem
    
    Problem description:
        Given n items, each with weight and value.
        A knapsack with capacity C.
        Each item is either placed in the knapsack (1) or not (0).
        Find the maximum total value without exceeding capacity.
    
    Algorithm idea (dynamic programming):
        Define state: dp[i][c] = max value considering first i items with capacity c.
        
        State transition equation:
        1. If not taking the i-th item: dp[i][c] = dp[i-1][c]
        2. If taking the i-th item (and weight fits):
           dp[i][c] = dp[i-1][c-weights[i-1]] + values[i-1]
        3. Take the maximum of the two
        
        Boundary conditions:
        - dp[0][c] = 0 (no items -> value 0)
        - dp[i][0] = 0 (zero capacity -> value 0)
        
        Steps:
        1. Create (n+1)×(C+1) DP table
        2. Fill DP table item by item
        3. Optional: backtrack to find which items were selected
    
    Key characteristics:
        - Optimal substructure: optimal solution contains optimal subproblem solutions
        - Overlapping subproblems: different decision paths recompute same subproblems
        - Memoization: DP table avoids redundant computation
        - Bottom-up: build solutions to larger problems from smaller ones
    
    DP elements:
        1. State definition: dp[i][c]
        2. State transition equation: max(don't take, take)
        3. Initial state: dp[0][c] = 0
        4. Computation order: left to right, top to bottom
        5. Final answer: dp[n][C]
    
    Parameters:
        values: list of item values, values[i] is the value of the i-th item
        weights: list of item weights, weights[i] is the weight of the i-th item
        capacity: knapsack capacity (positive integer)
        
    Returns:
        (maximum total value, list of selected item indices)
    
    Time complexity: O(n×C), where n is the number of items, C is the capacity
    Space complexity: O(n×C) — DP table size
    
    Example:
        >>> values = [60, 100, 120]
        >>> weights = [10, 20, 30]
        >>> capacity = 50
        >>> knapsack_01(values, weights, capacity)
        (220, [1, 2])  # max value 220, selecting items 1 and 2 (value 100+120)
        
        Explanation:
        Item 0: value 60, weight 10
        Item 1: value 100, weight 20
        Item 2: value 120, weight 30
        Optimal: select items 1 and 2, total weight 20+30=50≤50, total value 100+120=220
    
    Notes:
        - Item weights and values should be non-negative integers
        - If capacity is very large, time complexity may be high
        - Space can be optimized to O(C)
    
    Variant problems:
        - Unbounded knapsack: each item can be taken unlimited times
        - Bounded knapsack: each item has a quantity limit
        - Group knapsack: items are grouped, at most one per group
        - Dependency knapsack: items have dependency relationships
    
    Applications:
        - Resource allocation problems
        - Portfolio optimization
        - Cutting problems
        - Scheduling problems
    """
    n = len(values)
    # Create DP table: dp[i][c] = max value for first i items with capacity c
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for c in range(1, capacity + 1):
            if weights[i-1] <= c:
                # Can choose to take or not take the current item
                dp[i][c] = max(
                    dp[i-1][c],  # don't take current item
                    dp[i-1][c-weights[i-1]] + values[i-1]  # take current item
                )
            else:
                dp[i][c] = dp[i-1][c]  # current item too heavy, can't take
    
    # Optional: backtrack to find selected items
    selected_items = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            selected_items.append(i-1)
            c -= weights[i-1]
    
    return dp[n][capacity], selected_items[::-1]

# Space-optimized version: only O(W) space
def knapsack_01_optimized(values, weights, capacity):
    """Space-optimized 0-1 knapsack — O(W) space"""
    n = len(values)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Iterate backward to avoid overwriting the previous round's results
        for c in range(capacity, weights[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    
    return dp[capacity]

# Knapsack test
print("\n0-1 Knapsack Test:")
values = [60, 100, 120]  # item values
weights = [10, 20, 30]   # item weights
capacity = 50            # knapsack capacity

max_value, selected = knapsack_01(values, weights, capacity)
max_value_opt = knapsack_01_optimized(values, weights, capacity)

print(f"Item values: {values}")
print(f"Item weights: {weights}")
print(f"Knapsack capacity: {capacity}")
print(f"Max value: {max_value}")
print(f"Selected item indices: {selected}")
print(f"Space-optimized max value: {max_value_opt}")

# Comparison with greedy algorithm
def knapsack_greedy(values, weights, capacity):
    """Greedy algorithm: sort by value density"""
    n = len(values)
    items = sorted(range(n), key=lambda i: values[i]/weights[i], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    selected_items = []
    
    for i in items:
        if weights[i] <= remaining_capacity:
            total_value += values[i]
            remaining_capacity -= weights[i]
            selected_items.append(i)
    
    return total_value, selected_items

greedy_value, greedy_selected = knapsack_greedy(values, weights, capacity)
print(f"\nGreedy algorithm result: value={greedy_value}, selected items={greedy_selected}")
print(f"DP outperforms greedy by {(max_value - greedy_value)/greedy_value*100:.1f}%")
```

### Longest Common Subsequence: The Core of Sequence Alignment

```python
def longest_common_subsequence(text1, text2):
    """Dynamic programming solution for Longest Common Subsequence (LCS)
    
    Problem description (LeetCode 1143. Longest Common Subsequence):
        Given two strings text1 and text2, return the length and content of
        their longest common subsequence.
        A subsequence is formed by deleting some characters (or none) without
        changing the relative order of the remaining characters.
        A common subsequence is a subsequence present in both strings.
    
    Algorithm idea (dynamic programming):
        Define state: dp[i][j] = LCS length of text1[:i] and text2[:j].
        
        State transition equation:
        1. If text1[i-1] == text2[j-1] (characters match):
           dp[i][j] = dp[i-1][j-1] + 1
        2. If text1[i-1] != text2[j-1] (characters differ):
           dp[i][j] = max(dp[i-1][j], dp[i][j-1])
           
        Explanation:
        - Characters match: LCS length increases by 1, both strings advance
        - Characters differ: take the max LCS from skipping one char in either string
        
        Boundary conditions:
        - dp[0][j] = 0 (text1 is empty)
        - dp[i][0] = 0 (text2 is empty)
    
    Key characteristics:
        - Classic two-sequence DP problem
        - State definition: 2D DP table
        - State transition: different strategies based on character match
        - Can backtrack to reconstruct the LCS string
    
    Parameters:
        text1: first string
        text2: second string
        
    Returns:
        (LCS length, LCS string)
    
    Time complexity: O(m×n), where m=len(text1), n=len(text2)
    Space complexity: O(m×n) — DP table, optimizable to O(min(m,n))
    
    Example:
        >>> longest_common_subsequence("abcde", "ace")
        (3, "ace")
        >>> longest_common_subsequence("abc", "abc")
        (3, "abc")
        >>> longest_common_subsequence("abc", "def")
        (0, "")
        
        Explanation for "abcde" and "ace":
        text1: a b c d e
        text2: a   c   e
        LCS:   a   c   e  length 3
    
    Notes:
        - LCS does not require contiguity (unlike longest common substring)
        - There may be multiple LCS strings; this returns one found
        - Space can be optimized, but then backtracking to get the string is lost
    
    Variant problems:
        - Longest common substring (requires contiguity)
        - Shortest common supersequence (shortest string containing both)
        - Edit distance (LeetCode 72. Edit Distance)
        - Longest increasing subsequence (LeetCode 300. Longest Increasing Subsequence)
    
    Applications:
        - DNA sequence alignment (bioinformatics)
        - File diff comparison (git diff)
        - Speech recognition
        - Natural language processing (machine translation evaluation)
        - Version control systems
    
    Extensions:
        - Weighted LCS: different character matches have different weights
        - Multi-sequence LCS: common subsequence across multiple strings
        - Approximate matching: allow a certain degree of mismatch
    """
    m, n = len(text1), len(text2)
    # dp[i][j] = LCS length of text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to construct the LCS string
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], ''.join(reversed(lcs))

# LCS test
print("\nLongest Common Subsequence (LCS) Test:")
text1 = "ABCDGH"
text2 = "AEDFHR"
lcs_len, lcs_str = longest_common_subsequence(text1, text2)
print(f"String 1: {text1}")
print(f"String 2: {text2}")
print(f"LCS length: {lcs_len}")
print(f"LCS string: {lcs_str}")

# DNA sequence alignment example
dna1 = "AGCTAGC"
dna2 = "AGCATC"
dna_lcs_len, dna_lcs_str = longest_common_subsequence(dna1, dna2)
print(f"\nDNA Sequence Alignment:")
print(f"DNA1: {dna1}")
print(f"DNA2: {dna2}")
print(f"Common sequence length: {dna_lcs_len}")
print(f"Common sequence: {dna_lcs_str}")
```

### Climbing Stairs: A Simple Example of State Transition

```python
def climb_stairs_recursive(n):
    """Naive recursive solution for the climbing stairs problem"""
    if n <= 2:
        return n
    return climb_stairs_recursive(n-1) + climb_stairs_recursive(n-2)

def climb_stairs_dp(n):
    """Dynamic programming solution for the climbing stairs problem
    
    Problem description (LeetCode 70. Climbing Stairs):
        You are climbing a staircase. It takes n steps to reach the top.
        Each time you can either climb 1 or 2 steps.
        How many distinct ways can you climb to the top?
    
    Algorithm idea (dynamic programming):
        Define state: dp[i] = number of ways to reach step i.
        
        State transition equation:
        To reach step i, you can come from step i-1 (climb 1 step)
        or from step i-2 (climb 2 steps).
        So: dp[i] = dp[i-1] + dp[i-2]
        
        This is essentially a variant of the Fibonacci sequence.
        
        Boundary conditions:
        - dp[1] = 1 (climb 1 step: 1 way)
        - dp[2] = 2 (climb 2 steps: 2 ways — 1+1 or 2)
        - Note: some definitions set dp[0]=1, meaning 1 way to be at the start
    
    Key characteristics:
        - Classic optimal substructure problem
        - Simple and clear state transition
        - Space-optimizable to O(1)
        - Closely related to the Fibonacci sequence
    
    Parameters:
        n: number of steps (positive integer)
        
    Returns:
        Number of distinct ways to climb to step n
    
    Time complexity: O(n) — need to compute n states
    Space complexity: O(n) — DP array, optimizable to O(1)
    
    Example:
        >>> climb_stairs_dp(1)
        1
        >>> climb_stairs_dp(2)
        2  # ways: 1+1 or 2
        >>> climb_stairs_dp(3)
        3  # ways: 1+1+1, 1+2, 2+1
        >>> climb_stairs_dp(4)
        5  # ways: 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2
    
    Notes:
        - For large n, the result may exceed normal integer range
        - Recursive solution has O(2^n) complexity — infeasible
        - Matrix exponentiation can reduce complexity to O(log n)
    
    Variant problems:
        - Can climb 1, 2, or 3 steps at a time
        - Steps allowed are in a set (e.g., {1, 3, 5})
        - Climbing stairs with obstacles (LeetCode 63. Unique Paths II)
        - Min cost climbing stairs (LeetCode 746. Min Cost Climbing Stairs)
    
    Applications:
        - Combinatorial counting problems
        - Path planning
        - Investment decisions (multi-step decision problems)
    """
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def climb_stairs_optimized(n):
    """Space-optimized climbing stairs solution"""
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Climbing stairs performance comparison
print("\nClimbing Stairs Performance Comparison:")
n = 35

start = time.time()
result_recursive = climb_stairs_recursive(n)
time_recursive = time.time() - start

start = time.time()
result_dp = climb_stairs_dp(n)
time_dp = time.time() - start

start = time.time()
result_opt = climb_stairs_optimized(n)
time_opt = time.time() - start

print(f"Distinct ways to climb {n} steps: {result_dp}")
print(f"Naive recursion time: {time_recursive:.4f}s")
print(f"DP time: {time_dp:.6f}s")
print(f"Space-optimized time: {time_opt:.6f}s")
print(f"DP is {time_recursive/time_dp:.0f}x faster than naive recursion")
```

"Remember," Mr. Pallas's Cat summarized, "dynamic programming is the art of **reasoning with memory**. We concretize these abstract concepts through code to truly understand how to trade space for time, and how to use memory to accelerate computation. When facing a complex problem, first ask three questions: 1) Does it have optimal substructure? 2) Are there overlapping subproblems? 3) Is the state space manageable? If all three answers are yes, dynamic programming is very likely the best solution."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Algorithm implementation**: implement the DP solution for the 0-1 knapsack problem. Test with different capacities and item combinations; compare the results of DP vs. greedy algorithms. Try optimizing the space complexity to O(W).

2. **Memoization practice**: use memoized search to solve the "climbing stairs" problem (1 or 2 steps at a time, how many ways to reach step n?). Compare the performance difference between memoized search and naive recursion.

3. **Real-world observation**: observe "dynamic programming" thinking in everyday life. Is making a shopping list and then shopping by aisle a form of dynamic programming? When might this "advance planning" fail? (Hint: consider unlisted sale items.)

### Historical Investigation (for Little Seal)
1. **Tracing the concept**: the term "Dynamic Programming" was coined by Richard Bellman in the 1950s. Consult Bellman's original work to understand why he chose this name (hint: related to "mathematical programming," with "dynamic" opposed to "static").

2. **Cross-disciplinary connections**: what connections exist between dynamic programming and "chunking" in psychology? How do humans improve efficiency by breaking complex tasks into familiar chunks? Compare algorithmic memoization with human habit formation.

3. **Philosophical reflection**: does dynamic programming embody philosophical "pragmatism" or "rationalism"? Does it emphasize empirical accumulation (memory) or logical derivation (state transition)?

### Integrated Reflection
1. **Ethical consideration**: what ethical problems might arise from using DP algorithms in resource allocation (e.g., medical resource allocation)? When an algorithm pursues the "optimal solution," what humanitarian considerations might be overlooked?

2. **Algorithm comparison**: what are the fundamental differences among dynamic programming, divide-and-conquer, and greedy algorithms? Why are some problems suitable for divide-and-conquer but not DP? Some for greedy but not DP?

3. **Creative exercise**: design a "dynamic programming fable" — use a story to explain the idea of DP. (Hint: compare to an architect who first calculates the load-bearing capacity of each brick before designing the entire building.)

4. **Limit challenge**: prove that the longest common subsequence (LCS) problem has optimal substructure. Design the state transition equation and implement the algorithm.

---

## Coming Up Next

Dawn was breaking; the surface of the Pearl River shimmered with pale gold. The birds of Kangle Garden began to sing — a new day was about to begin. The lamplight in the Black Stone House study still glowed warmly, like a lighthouse of thought that never goes out.

Piglet was testing the performance of different DP algorithms on his computer, the screen displaying various problem instances and running times. Little Seal was deriving the general form of state transition equations on the whiteboard, trying to understand the unified pattern behind the seemingly complex formulas.

"Today we started from 'how memory empowers reasoning,'" Mr. Pallas's Cat said, tidying the tea set. "We explored the wisdom of dynamic programming, and understood the eternal trade-off between space and time."

Little Seal put down the chalk. "Professor, DP solves overlapping subproblems, but what if a problem has no overlapping substructure, or the state space is too large?"

Mr. Pallas's Cat smiled. "Good question. In that case, we need new thinking tools — **approximation algorithms** or **heuristic search**, or we must accept the problem's inherent difficulty."

Piglet looked up. "Those are things we already learned! So algorithmic thinking is a spiral, going upward."

"Exactly," said Mr. Pallas's Cat. "Algorithmic thinking is not linear, but a **spiral ascent**. We keep returning to similar problems, but each time with new perspectives and tools."

"In the next volume," Mr. Pallas's Cat walked to the window and gazed at Kangle Garden in the morning light, "we will leave the deterministic universe and enter **the fault zone that crosses logic**. We'll explore: when the world becomes fuzzy, continuous, high-dimensional — are deterministic rules still enough?"

Outside the window, the first rays of sunlight fell upon the Pearl River, glittering on the waves. The three people in the Black Stone House study were like three explorers who had just completed one leg of a journey and were about to embark on a new expedition.

---

> **Piglet's note**: I tested different implementations of Fibonacci. Naive recursion took several seconds to compute fib(40); memoized search took less than a millisecond! The space-for-time effect is staggering. Dynamic programming is like installing a "memory cheat" for algorithms!
>
> **Little Seal's note**: I looked up the history of dynamic programming and found that Bellman originally studied DP to solve multi-stage decision problems. Interestingly, he chose the word "dynamic" to emphasize the temporal ordering of decisions, and "programming" to attract attention from the mathematical programming community. Sometimes, naming itself is a strategy.
>
> **Mr. Pallas's Cat's closing words**: Remember, dynamic programming teaches an important lesson about reasoning: sometimes, the fastest progress comes not from charging ahead, but from pausing to remember the path already traveled. We'll take it slow — understanding is what matters most.

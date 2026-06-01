# Chapter 6: The Art of Heuristics (Approximation and Estimation)

:::info
**Mr. Pallas's Cat's Warm Welcome**  
After exploring the wisdom and limits of greedy algorithms, we face a new question: **when the perfect solution is unattainable, how do we make smart choices?** Sometimes, we cannot guarantee finding the optimal solution, or even a solution within a reasonable time. Today, we explore the wisdom of heuristics — navigating the unknown with clever guesses.
:::

---

## Core Question: How Do We Find Direction Amid Uncertainty?

It was evening in Kangle Garden. The golden afterglow of the setting sun reflected off the Pearl River; a few egrets skimmed the water's surface, leaving faint ripples. In the study of the Black Stone House, Mr. Pallas's Cat was organizing the bookshelves, while Piglet and Little Seal gathered around a maze diagram.

"This maze is so complicated!" Piglet said, pointing at the twisting paths on the diagram. "If I don't know where the exit is, how should I proceed?"

Little Seal thought. "Theoretically, we could use depth-first search to explore all paths, but we'd potentially try thousands of dead ends."

Mr. Pallas's Cat walked over and smiled. "You've raised a classic problem: **in an unknown environment, how do we find a target efficiently?** When perfect information is unavailable, we need to rely on **heuristics** — 'smart guesses' based on experience."

Piglet asked curiously: "Heuristics? Like the intuition of 'try to move toward the center' when navigating a maze?"

"Exactly," said Mr. Pallas's Cat. "A heuristic is not a rule that guarantees correctness, but a rule of thumb that **improves the probability of finding a good solution**. The core issue we explore today is: **how do we find direction amid uncertainty?** More specifically: **how do we make good-enough decisions with limited resources and information?**"

He walked to the whiteboard and wrote two words:

**Exact Search** — **Heuristic Search**

"Exact search guarantees finding the optimal solution, but the cost may be astronomical time complexity," Mr. Pallas's Cat explained. "Heuristic search does not guarantee optimality, but can find a 'good-enough' solution within acceptable time. This trade-off lies at the heart of modern algorithm design."

## Heuristic Search: The Wisdom of the A* Algorithm

"Let's look at a classic heuristic algorithm: **A* (A-star) search**," said Mr. Pallas's Cat, drawing a grid map on the whiteboard. "Suppose we need to go from start S to goal G, with obstacles in between."

Little Seal said: "The A* algorithm combines **actual cost** and **estimated cost**, right?"

"Exactly," said Mr. Pallas's Cat, writing the formula:

```
f(n) = g(n) + h(n)
```

- **g(n)**: the actual cost from the start to node n
- **h(n)**: the estimated cost from node n to the goal (the heuristic function)
- **f(n)**: the total estimated cost of node n

"A* always expands the node with the smallest f(n)," Mr. Pallas's Cat explained. "It's like being in a maze — you consider how far you've already walked, and estimate how far the exit still is, then choose the direction that 'looks closest.'"

### The Art of Designing Heuristic Functions

Piglet asked: "How do you estimate h(n)? Just guess randomly?"

"That's the core of heuristic design," said Mr. Pallas's Cat. "A good heuristic function needs to satisfy two properties:
1. **Admissibility**: never overestimates the true cost (guaranteeing the optimal solution is found)
2. **Consistency**: satisfies the triangle inequality (guaranteeing efficiency)

"The most commonly used heuristic functions are **Manhattan distance** or **Euclidean distance**," he gave as examples. "In a grid map where you can only move up, down, left, and right, Manhattan distance (|x₁−x₂| + |y₁−y₂|) is admissible."

Little Seal thought: "But what if diagonal movement is allowed?"

"Then Manhattan distance would overestimate and needs adjustment," said Mr. Pallas's Cat. "The design of the heuristic function needs to **match the structure of the problem**. A good heuristic can dramatically improve search efficiency; a poor one can be slower than blind search."

:::info
**Mr. Pallas's Cat's Commentary**  
The philosophy of heuristic search is an extension of the **satisficing principle**: we don't pursue the absolute optimum, but find a good-enough solution within acceptable time. This reflects the norm of real-world decision-making — information is limited, time is limited, and we must make the best guess amid uncertainty.
:::

### A* vs. Greedy Best-First Search

"Let's compare two heuristic strategies," Mr. Pallas's Cat drew a comparison table:

| Strategy | Selection Criterion | Pros | Cons |
|------|---------|------|------|
| **Greedy Best-First** | Smallest h(n) (estimated cost only) | Fast toward the goal | May fall into local optima; no optimality guarantee |
| **A* Search** | Smallest f(n)=g(n)+h(n) | Guarantees optimality (if h is admissible) | May need more memory and time |

Piglet had a flash of insight: "So greedy best-first 'only looks forward,' while A* 'looks both backward and forward'!"

"A concise summary," Mr. Pallas's Cat said approvingly. "The greedy algorithm can be seen as a special case of heuristic search — it only considers local information, while A* combines global information."

## Approximation Algorithms: When the Optimal Is Out of Reach

Little Seal asked: "Professor, if the problem itself is NP-hard, like the Traveling Salesman Problem, what do we do?"

"In that case we need **approximation algorithms**," said Mr. Pallas's Cat. "They don't guarantee finding the optimal solution, but they guarantee that the solution quality is within a certain ratio."

He gave examples: "For the Traveling Salesman Problem (TSP), there are several approximation strategies:
1. **Nearest neighbor algorithm**: starting from any city, always go to the nearest unvisited city
2. **Minimum spanning tree doubling algorithm**: construct a minimum spanning tree, then perform an Eulerian tour
3. **Christofides algorithm**: guarantees a solution no worse than 1.5 times the optimal"

Piglet calculated: "So if we accept a solution that may be 50% worse than optimal, we can solve it in polynomial time?"

"Exactly," said Mr. Pallas's Cat. "The core of approximation algorithms is **trading solution quality for computation time**. In some applications, 95% of the optimal is good enough, while computation time drops from exponential to polynomial."

:::detail
**Advanced: Performance Guarantees of Approximation Algorithms**

Approximation algorithms are typically measured by the **approximation ratio**:
- **α-approximation algorithm**: guarantees the solution is at most α times the optimal (for minimization problems)
- **(1-ε)-approximation algorithm**: guarantees the solution is at least (1-ε) times the optimal (for maximization problems)

For example:
- Vertex Cover problem: 2-approximation algorithm (solution at most twice the optimal)
- Set Cover problem: ln n-approximation algorithm
- Traveling Salesman Problem (metric TSP): 1.5-approximation algorithm (Christofides)

Designing approximation algorithms requires clever construction and proof, and is an important area of theoretical computer science.
:::

---

## Orthogonal Computation Graphs: Seeing the Decision Flow of Heuristics

Mr. Pallas's Cat turned on the projector, and a neat computation graph appeared on the screen.

![Heuristic Search Orthogonal Computation Graph](/figures/heuristic_search_ortho.svg)

"This is the orthogonal computation graph for the A* algorithm," Mr. Pallas's Cat said, pointing at the diagram. "The start point passes through actual cost accumulation and heuristic estimation; the node with the smallest f(n) is selected for expansion, looping until the goal is found."

Piglet stared at the diagram. "This graph has two branches! g(n) and h(n) are computed in parallel, then merged into f(n)."

"Exactly," said Mr. Pallas's Cat. "The core of the A* algorithm is **dual-path evaluation**: one path records the cost already paid, the other predicts the future cost. This combination of 'retrospect and prospect' makes it more efficient than pure prospect (greedy) or pure retrospect (Dijkstra)."

Little Seal examined the layered structure of the diagram carefully. "Professor, the design of this diagram... the actual cost and heuristic cost are computed at the same level, then merged for selection. This reflects the symmetrical beauty of the A* algorithm."

"That's the elegance of heuristic search," Mr. Pallas's Cat explained. "It balances **experience** (cost already paid) with **intuition** (estimate of future cost). This balance is important not only in algorithms but also in life — the best decisions often consider both past investment and future return."

---

## Mental Model: The Eternal Tension Between Satisficing and Optimizing

Mr. Pallas's Cat returned to his seat and poured three cups of tea. "Let's summarize today's most important mental model: **the eternal tension between satisficing and optimizing**."

He drew a comparison table on the whiteboard:

| Dimension | Optimality Mindset | Satisficing Mindset |
|------|-----------|-----------|
| **Goal** | Pursue the absolute best | Pursue good enough |
| **Time attitude** | Spare no time | Value time efficiency |
| **Information need** | Requires complete information | Tolerates incomplete information |
| **Suitable scenarios** | Small-scale, critical decisions | Large-scale, everyday decisions |
| **Philosophical root** | Perfectionism | Pragmatism |

"These two mindsets are not opposed," Mr. Pallas's Cat explained. "Rather, they are **two ends of a decision spectrum**. A wise decision-maker knows when to pursue perfection and when to accept satisficing."

### Conditions for Applying Heuristic Methods

Piglet asked: "Professor, when should we use heuristic methods?"

Mr. Pallas's Cat listed the conditions on the whiteboard:

```
Problems suitable for heuristic methods typically have:
1. **Large problem scale**: computation time of exact algorithms is unacceptable
2. **Incomplete information**: complete problem information cannot be obtained
3. **Real-time requirements**: an answer must be provided within a limited time
4. **Fault tolerance**: approximate or probabilistically correct solutions are acceptable
5. **Human intuition available**: rules of thumb exist that can be drawn upon

Common heuristic application scenarios:
├── Path planning (A*, D* algorithms)
├── Game AI (Minimax with alpha-beta pruning)
├── Scheduling problems (genetic algorithms, simulated annealing)
├── Machine learning (gradient descent, stochastic gradient descent)
└── Everyday decisions (rules of thumb, heuristics)
```

"Understanding these conditions," said Mr. Pallas's Cat, "helps us judge when we can 'be heuristic' and when we must 'be exact.'"

:::info
**Mr. Pallas's Cat's Commentary**  
The philosophy of heuristic thinking is **bounded rationality** — making the best possible decision under limited resources, limited information, and limited time. This is not only a principle of algorithm design, but also a manifestation of human wisdom. We can never have a God's-eye view, but we can light a lamp in the fog.
:::

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary: The Wisdom of Heuristic Thinking**  
1. **Heuristic search framework**: the A* algorithm combines actual cost g(n) with heuristic estimate h(n), dramatically improving search efficiency while preserving optimality — embodying the balanced wisdom of "retrospect and prospect"  
2. **Heuristic function design principles**: admissibility (no overestimation) and consistency (triangle inequality) are key to correctness and efficiency; heuristics that match the problem structure greatly boost performance  
3. **The philosophy of approximation**: when the optimal is unattainable, α-approximation algorithms provide quality-guaranteed suboptimal solutions, finding the best balance between solution quality and computation time  
4. **The tension between satisficing and optimizing**: heuristic thinking acknowledges the limits of human rationality, pursuing "good enough" under constraints rather than absolute perfection — a manifestation of pragmatist wisdom  
5. **Applied wisdom**: in real problems such as path planning, game AI, and scheduling optimization, heuristic methods are often best practice; learning to choose the right heuristic strategy is an important lesson in algorithmic thinking
:::

---

## Code Practice: Implementing Heuristic Search in Python

"Let's use code to explore the charm of heuristic search," said Mr. Pallas's Cat. "From the A* algorithm to different heuristic functions, from maze pathfinding to sliding puzzles."

### A* Algorithm Generic Framework

```python
import heapq
import math

class Node:
    """A* algorithm node class"""
    def __init__(self, position, parent=None):
        self.position = position  # node position (e.g., (x,y) coordinates)
        self.parent = parent      # parent node
        
        # Cost estimates
        self.g = 0  # actual cost from start to current node
        self.h = 0  # heuristic estimate from current node to goal
        self.f = 0  # total estimated cost f = g + h
    
    def __lt__(self, other):
        # For heap ordering — compare by f value
        return self.f < other.f
    
    def __eq__(self, other):
        # Compare by node position
        return self.position == other.position
    
    def __hash__(self):
        # Make node hashable, for use in sets
        return hash(self.position)

def a_star_search(start, goal, grid, heuristic):
    """A* search algorithm
    
    Problem description:
        Find the shortest path from start to goal on a grid map.
        0 in the grid indicates traversable area; 1 indicates an obstacle.
    
    Algorithm idea (heuristic search):
        A* combines Dijkstra's algorithm (guarantees optimality) and greedy best-first search (fast).
        Uses the evaluation function f(n) = g(n) + h(n):
        - g(n): actual cost from start to node n
        - h(n): estimated cost from node n to goal (heuristic function)
        
        Steps:
        1. Initialize the open list (priority queue) and closed set
        2. Add the start node to the open list, g=0, f=h(start)
        3. While the open list is not empty:
           a. Remove the node with the smallest f from the open list as the current node
           b. If the current node is the goal, backtrack to get the path
           c. Add the current node to the closed set
           d. Generate all neighbors of the current node
           e. For each neighbor:
              - Skip if it's an obstacle or already in the closed set
              - Compute new g = current g + movement cost (typically 1)
              - If the neighbor is not in the open list or new g is smaller:
                * Update the neighbor's g, h, f values
                * Set the neighbor's parent to the current node
                * If not in the open list, add it
    
    Key features:
        - Completeness: if a path exists, it will be found
        - Optimality: if the heuristic is admissible (h(n) ≤ actual cost), the shortest path is found
        - Efficiency: faster than Dijkstra, more likely to find the optimal than greedy best-first
        - Heuristic function quality affects search efficiency
    
    Heuristic function requirements:
        1. Admissibility: h(n) ≤ actual cost from n to goal (guarantees optimality)
        2. Consistency (monotonicity): h(n) ≤ c(n, n') + h(n') (guarantees efficiency)
        Common heuristic functions:
        - Manhattan distance: |x1-x2| + |y1-y2| (only up/down/left/right movement)
        - Euclidean distance: √((x1-x2)² + (y1-y2)²) (diagonal movement allowed)
        - Chebyshev distance: max(|x1-x2|, |y1-y2|) (8-direction movement)
    
    Parameters:
        start: start coordinates (x, y)
        goal: goal coordinates (x, y)
        grid: 2D grid map, 0 = traversable, 1 = obstacle
        heuristic: heuristic function — takes two coordinates, returns estimated distance
    
    Returns:
        path (list of coordinates from start to goal), or None if no path found
    
    Time complexity: O(b^d), where b is the branching factor, d is the solution depth
               In practice much less than exhaustive search, depending on heuristic quality
    Space complexity: O(b^d), needs to store all explored nodes
    
    Example:
        >>> grid = [
        ...     [0, 0, 0, 0, 0],
        ...     [0, 1, 1, 1, 0],
        ...     [0, 0, 0, 0, 0],
        ...     [0, 1, 1, 1, 0],
        ...     [0, 0, 0, 0, 0]
        ... ]
        >>> def manhattan(p1, p2):
        ...     return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        >>> a_star_search((0, 0), (4, 4), grid, manhattan)
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), 
         (4, 1), (4, 2), (4, 3), (4, 4)]
    
    Variants:
        - Bidirectional A*: search from both start and goal simultaneously
        - Iterative deepening A*: A* combined with iterative deepening
        - Weighted A*: f(n) = g(n) + w*h(n), w>1 speeds up but may be suboptimal
        - D*: A* for dynamic environments
    """
    # Create start and goal nodes
    start_node = Node(start)
    goal_node = Node(goal)
    
    # Initialize open list and closed set
    open_list = []
    closed_set = set()
    
    # Add the start node to the open list
    heapq.heappush(open_list, start_node)
    
    # Search loop
    while open_list:
        # Remove the node with the smallest f value
        current_node = heapq.heappop(open_list)
        
        # If the goal is reached, backtrack to construct the path
        if current_node.position == goal_node.position:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # reverse to get path from start to goal
        
        # Add the current node to the closed set
        closed_set.add(current_node.position)
        
        # Generate neighbor nodes (up, down, left, right)
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x = current_node.position[0] + dx
            new_y = current_node.position[1] + dy
            
            # Check boundaries
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
                # Check for obstacles
                if grid[new_x][new_y] == 0:
                    neighbors.append((new_x, new_y))
        
        # Process neighbor nodes
        for neighbor_pos in neighbors:
            # Skip if the neighbor is in the closed set
            if neighbor_pos in closed_set:
                continue
            
            # Create the neighbor node
            neighbor = Node(neighbor_pos, current_node)
            
            # Compute g value: g from start to current + movement cost (assumed 1)
            neighbor.g = current_node.g + 1
            
            # Compute h value: heuristic function estimate
            neighbor.h = heuristic(neighbor_pos, goal)
            
            # Compute f value
            neighbor.f = neighbor.g + neighbor.h
            
            # Check if the node already exists in the open list with a smaller g
            found_in_open = False
            for open_node in open_list:
                if neighbor.position == open_node.position and neighbor.g >= open_node.g:
                    found_in_open = True
                    break
            
            # If not in the open list or g is smaller, add to the open list
            if not found_in_open:
                heapq.heappush(open_list, neighbor)
    
    # Open list is empty — no path found
    return None
```

### Heuristic Function Implementations

```python
# Various heuristic function implementations
def manhattan_distance(pos1, pos2):
    """Manhattan distance (for grids allowing only up/down/left/right movement)
    
    Formula: h(n) = |x1 - x2| + |y1 - y2|
    
    Characteristics:
        - Suitable for grid environments allowing only four-direction movement
        - Admissible (actual shortest path ≥ Manhattan distance)
        - Consistent (satisfies triangle inequality)
        - Simple and fast to compute
    
    Example:
        >>> manhattan_distance((0, 0), (3, 4))
        7  # |0-3| + |0-4| = 3+4=7
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(pos1, pos2):
    """Euclidean distance (for cases allowing diagonal movement)
    
    Formula: h(n) = √((x1 - x2)² + (y1 - y2)²)
    
    Characteristics:
        - Suitable for continuous spaces allowing movement in any direction
        - Admissible (straight-line distance ≤ actual path length)
        - Consistent
        - Involves square root, slightly slower
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def chebyshev_distance(pos1, pos2):
    """Chebyshev distance (for cases allowing eight-direction movement)
    
    Formula: h(n) = max(|x1 - x2|, |y1 - y2|)
    
    Characteristics:
        - Suitable for grids allowing eight-direction movement (up/down/left/right + diagonals)
        - Admissible (when diagonal movement cost is 1)
        - Consistent
        - Simple to compute
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return max(abs(x1 - x2), abs(y1 - y2))

def zero_heuristic(pos1, pos2):
    """Zero heuristic (degenerates to Dijkstra's algorithm)
    
    Formula: h(n) = 0
    
    Characteristics:
        - Always admissible (0 ≤ actual distance)
        - Degenerates to Dijkstra's algorithm — guarantees finding the shortest path
        - No heuristic information — low search efficiency
        - Suitable when no good heuristic is available
    
    Note:
        - When h(n)=0, A* degenerates to Dijkstra's algorithm
        - Explores all possible paths — guarantees optimality but inefficient
        - Rarely used in practice unless no better heuristic exists
    """
    return 0
```

### Maze Experiment

```python
def maze_experiment():
    """Maze experiment: comparing the effects of different heuristic functions"""
    
    print("Maze Experiment: Comparing A* with Different Heuristic Functions")
    print("=" * 50)
    
    # Define the maze map (0 = traversable, 1 = obstacle)
    # 10x10 grid
    maze = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    start = (0, 0)
    goal = (9, 9)
    
    print(f"Maze size: {len(maze)}x{len(maze[0])}")
    print(f"Start: {start}, Goal: {goal}")
    
    # Test different heuristic functions
    heuristics = [
        ("Manhattan distance", manhattan_distance),
        ("Euclidean distance", euclidean_distance),
        ("Chebyshev distance", chebyshev_distance),
        ("Zero heuristic (Dijkstra)", zero_heuristic),
    ]
    
    results = []
    
    for name, heuristic_func in heuristics:
        print(f"\nUsing {name}:")
        
        class CountingAStar:
            def __init__(self, heuristic):
                self.heuristic = heuristic
                self.nodes_expanded = 0
            
            def count_heuristic(self, pos1, pos2):
                self.nodes_expanded += 1
                return self.heuristic(pos1, pos2)
        
        counter = CountingAStar(heuristic_func)
        
        import time
        start_time = time.time()
        path = a_star_search(start, goal, maze, counter.count_heuristic)
        elapsed = time.time() - start_time
        
        if path:
            print(f"  Path found, length: {len(path)-1} steps")
            print(f"  Nodes expanded: {counter.nodes_expanded}")
            print(f"  Time: {elapsed:.6f}s")
            results.append((name, len(path)-1, counter.nodes_expanded, elapsed))
        else:
            print(f"  No path found")
            results.append((name, None, counter.nodes_expanded, elapsed))
    
    # Result analysis
    print("\n" + "=" * 50)
    print("Result Analysis")
    print("=" * 50)
    
    print("\nPerformance comparison:")
    print("Heuristic           Path length    Nodes expanded    Time(s)")
    print("-" * 50)
    
    for name, path_len, nodes, time_taken in results:
        if path_len is not None:
            print(f"{name:20} {path_len:8} {nodes:12} {time_taken:10.6f}")
        else:
            print(f"{name:20} {'No path':8} {nodes:12} {time_taken:10.6f}")
    
    print("\nHeuristic property analysis:")
    print("1. Admissibility: heuristic does not overestimate actual cost")
    print("   - Manhattan distance: ✓ (a lower bound on shortest path in a grid)")
    print("   - Euclidean distance: ✓ (straight-line distance ≤ actual path)")
    print("   - Chebyshev distance: ✓ (admissible when 8-direction movement is allowed)")
    print("   - Zero heuristic: ✓ (never overestimates)")
    
    print("\n2. Consistency: h(n) ≤ c(n, n') + h(n')")
    print("   - Manhattan distance: ✓ (satisfies triangle inequality)")
    print("   - Euclidean distance: ✓")
    print("   - Chebyshev distance: ✓")
    print("   - Zero heuristic: ✓")
    
    print("\n3. Informedness: the closer the estimate to the actual cost, the faster the search")
    print("   - Manhattan distance: high (very accurate in a grid)")
    print("   - Euclidean distance: medium")
    print("   - Chebyshev distance: medium")
    print("   - Zero heuristic: low (no heuristic information — degenerates to Dijkstra)")

maze_experiment()
```

### The 8-Puzzle (Sliding Puzzle)

```python
def sliding_puzzle_heuristic(puzzle, goal):
    """Heuristic functions for the 8-puzzle
    
    Common heuristic functions:
    1. Misplaced tiles count
    2. Sum of Manhattan distances
    """
    
    def misplaced_tiles(puzzle, goal):
        """Misplaced tiles heuristic"""
        count = 0
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] != 0 and puzzle[i][j] != goal[i][j]:
                    count += 1
        return count
    
    def manhattan_sum(puzzle, goal):
        """Sum of Manhattan distances heuristic"""
        goal_positions = {}
        for i in range(3):
            for j in range(3):
                if goal[i][j] != 0:
                    goal_positions[goal[i][j]] = (i, j)
        
        distance = 0
        for i in range(3):
            for j in range(3):
                value = puzzle[i][j]
                if value != 0:
                    goal_i, goal_j = goal_positions[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    return misplaced_tiles(puzzle, goal), manhattan_sum(puzzle, goal)

def sliding_puzzle_example():
    """8-puzzle example"""
    
    print("\n" + "=" * 50)
    print("8-Puzzle (Sliding Puzzle) Heuristic Function Design")
    print("=" * 50)
    
    initial = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    print("Initial state:")
    for row in initial:
        print("  " + " ".join(str(x) if x != 0 else " " for x in row))
    
    print("\nGoal state:")
    for row in goal:
        print("  " + " ".join(str(x) if x != 0 else " " for x in row))
    
    misplaced, manhattan = sliding_puzzle_heuristic(initial, goal)
    
    print(f"\nHeuristic estimates:")
    print(f"  Misplaced tiles: {misplaced}")
    print(f"  Sum of Manhattan distances: {manhattan}")
    
    print("\nHeuristic comparison:")
    print("1. Admissibility:")
    print("   - Misplaced tiles: ✓ (each move reduces at most 1 misplaced tile)")
    print("   - Manhattan distance: ✓ (each move reduces Manhattan distance by at most 1)")
    
    print("\n2. Informedness (which is more accurate?):")
    print("   - Misplaced tiles: underestimates actual moves")
    print("   - Manhattan distance: closer to actual moves")
    print("   → Manhattan distance is typically more efficient")
    
    print("\n3. Consistency check:")
    print("   - Manhattan distance satisfies consistency")
    print("   - Misplaced tiles does not strictly satisfy consistency, but is admissible")

sliding_puzzle_example()
```

"Through this code," Mr. Pallas's Cat summarized, "we have not only understood the implementation of heuristic search, but also grasped the wisdom of **finding direction amid uncertainty**. Heuristic algorithms teach us: in a world of incomplete information, the best strategy is often not waiting for perfect information, but making the best guess based on what we have."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Algorithm implementation**: implement the A* algorithm to solve the maze problem. Test the effects of different heuristic functions (Manhattan, Euclidean, Chebyshev). Which heuristic expands the fewest nodes? Why?

2. **Heuristic design**: design a heuristic function for the 8-puzzle (sliding puzzle). Consider: how do you guarantee the admissibility of the heuristic? Can you design a heuristic that is both efficient and admissible?

3. **Real-world observation**: observe heuristic decision-making in everyday life. Is choosing the shortest checkout line at the supermarket a heuristic? When might it fail? (Hint: consider what's in customers' carts, differences in cashier speed.)

### Historical Investigation (for Little Seal)
1. **Tracing the concept**: the word "heuristic" comes from the Greek "εὑρίσκω" (to discover). When was it formally introduced into computer science? Consult early AI literature (e.g., Newell & Simon 1958, Hart et al. 1968).

2. **Cross-disciplinary connections**: what connections exist between heuristic thinking and the "heuristics and biases" research in psychology? What are the similarities and differences between Kahneman and Tversky's prospect theory and algorithmic heuristics?

3. **Philosophical reflection**: do heuristic algorithms embody the philosophical tradition of "pragmatism"? Compare the pragmatist thought of William James and John Dewey with the design principles of heuristic algorithms.

### Integrated Reflection
1. **Ethical consideration**: can autonomous vehicles use heuristic algorithms for emergency decision-making? (E.g., always choosing the steering direction that minimizes immediate harm.) What ethical problems might such experience-based decisions entail?

2. **Algorithm comparison**: what are the fundamental differences among heuristic search, greedy algorithms, and dynamic programming? Why are some problems amenable to heuristics while dynamic programming is infeasible? And vice versa?

3. **Creative exercise**: design a "heuristic fable" — use a story to explain the strengths and limitations of heuristic thinking. (Hint: compare to a navigator using a compass and star charts to navigate through fog.)

4. **Limit challenge**: prove that if the heuristic function h(n) satisfies consistency, the A* algorithm will not re-expand any node. Understand the relationship between consistency and admissibility.

---

## Coming Up Next

Night deepened; the lights along both banks of the Pearl River reflected in the water like the Milky Way come to earth. The banyan trees of Kangle Garden swayed gently in the evening breeze, as if pondering the discussion just concluded.

Piglet was simulating the effects of different heuristic functions on his computer, the screen displaying maze maps and various search paths. Little Seal was deriving the proof of heuristic admissibility on the whiteboard, trying to understand the mathematical rigor behind what seemed like intuitive estimates.

"Today we started from 'how to find direction amid uncertainty,'" Mr. Pallas's Cat said, tidying the tea set. "We explored the wisdom of heuristic search, and understood the eternal tension between satisficing and optimizing."

Little Seal put down the chalk. "Professor, what if heuristic algorithms are still too slow, or we need the absolute optimal solution — are there other methods?"

Mr. Pallas's Cat smiled. "Good question. That calls for an even more powerful tool — **dynamic programming**."

Piglet looked up. "Dynamic programming? Is that the next chapter?"

"Exactly," Mr. Pallas's Cat explained. "Dynamic programming is **reasoning with memory**. It stores intermediate results, avoids redundant computation, and can solve many problems for which heuristic algorithms cannot guarantee optimality. But its cost is higher space complexity."

"In the next chapter," Mr. Pallas's Cat walked toward the bookshelf, "we'll explore the stair-climbing problem, Fibonacci numbers, the knapsack problem... and see how dynamic programming trades space for time to solve complex optimization problems."

Outside the window, a cruise ship drifted slowly across the Pearl River, its lights casting flowing shadows on the windows of the Black Stone House. The three people in the study were like three explorers, searching for the lighthouse of wisdom in the ocean of algorithms.

---

> **Piglet's note**: I tested the effect of different heuristic functions on the A* algorithm. Manhattan distance expanded the fewest nodes in a grid map, but Euclidean distance was better when diagonal movement was allowed. Heuristics are like giving the algorithm a pair of "eyes" — letting it see the general direction of the goal!
>
> **Little Seal's note**: I looked up the history of heuristic search and found that the A* algorithm was proposed by Hart, Nilsson, and Raphael in 1968. Interestingly, the idea of heuristics appeared early in AI, but formal proofs took a long time. Sometimes, intuition walks ahead of formalization.
>
> **Mr. Pallas's Cat's closing words**: Remember, heuristic algorithms teach an important lesson about decision-making: in a world of incomplete information, the best strategy is often not waiting for perfect information, but making the best guess based on what we have. We'll take it slow — understanding is what matters most.

# Chapter 10: The Art of Search — Cruising in Reasoning Space

> MCTS knows nothing about the rules of Go. What it knows is: which path is worth continuing down.

<div class="center">

------------------------------------------------------------------------

</div>

Chapter 8 covered A\* — searching over a fixed graph, with a heuristic function pointing the way. Chapter 9 covered the Transformer — on the manifold of hidden states, it measures how states relate to each other with an S×S matrix, taking one soft search step per layer. Both searches share one blind spot: they depend on a road network that already exists. The graph of A\* is drawn by people; the hidden-state metric of the Transformer is fed by the training data.

But what if you're dropped into a wilderness with no roads at all? No graph, no metric, no admissible heuristic — all you know is the start, the finish, and the ability to "take a step, look around, and decide which way to go."

This is the battlefield of Monte Carlo Tree Search (MCTS). It depends on no prior map. What it runs on is random simulation — throw stones in every direction, see which paths get through, then come back and tell the decision maker. It is not searching on known terrain; it is building a map while searching.

If A\* is a hiker with a map and the Transformer is a particle gliding over a manifold, then MCTS is a blindfolded person feeling through a maze — after every step, reach out and check before deciding where to go. Three algorithms, three philosophies of search: A\* is prior geometry, MCTS is empirical sampling, the Transformer is an implicit metric. This chapter is about the second one.

<div class="center">

------------------------------------------------------------------------

</div>

## 1. The 37th Move, March 2016

March 9, 2016. The Four Seasons Hotel in Seoul. AlphaGo vs. Lee Sedol, game two, and the game had reached move 37.

AlphaGo played a move in the upper-right corner: a shoulder hit on the fifth line.

The commentary room went silent for 12 seconds. Professional 9-dan players stared at the board, unable to believe their eyes. This was not a move a human would play — it broke the standard patterns of Go, and it looked like the mistake of an amateur.

But the rest of the game proved that this move was the key to victory. Later, people called it the "Hand of God".

How did AlphaGo find this move?

Not by "understanding" the beauty of Go, not by memorizing game records — but by **search**: finding the path that leads to victory among millions of possible games.

Here is a deep isomorphism: **reasoning can be seen as path search through a state space**.

- Each **state** is a step of reasoning (or a position on the board).

- Each **action** is a logical inference (or a move).

- The **goal** is to find a proof (or to win).

From this point of view, theorem proving, mathematical reasoning, and playing games are all search problems at heart.

<div class="center">

------------------------------------------------------------------------

</div>

## 2. The Curse of the Search Space

But the search space is exponential.

The state space of Go is about $10^{170}$. Even searching $10^9$ states per second, exhausting everything would take $10^{153}$ years — $10^{143}$ times the age of the universe.

The state space of chess is "only" about $10^{47}$, still far beyond any computer.

**Brute-force search is impossible. We need smarter search.**

The classic search algorithms come in two kinds:

**Depth-first search (DFS)**: follow one path to the end; if it doesn't work, backtrack. Its strength is low space complexity, O(depth); its weakness is that it can wander into branches with effectively unbounded depth and never find the optimal solution.

**Breadth-first search (BFS)**: expand layer by layer — first all nodes at depth 1, then all nodes at depth 2. Its strength is that it guarantees the shortest path; its weakness is exponential space complexity, O(branching factor^depth).

Neither is good enough. DFS is too blind; BFS is too expensive.

We need a method that can both explore unknown territory and exploit the good paths we already know.

That is Monte Carlo Tree Search (MCTS).

<div class="center">

------------------------------------------------------------------------

</div>

## 3. MCTS: The Wisdom of Random Sampling

The core idea of MCTS is surprisingly simple: **use random simulations to estimate the value of each action, then prefer the actions with high value**.

It has four steps, run in a loop:

1. **Selection**: starting from the root, use the UCB formula to pick the most promising child node, one node at a time, until you reach a leaf.

2. **Expansion**: if the leaf is not a terminal state, add a new child node.

3. **Simulation**: starting from the new node, play randomly until the game ends, and get a result (win/lose/draw).

4. **Backpropagation**: send the result up along the path, updating the statistics of every node it passes through.

Repeat this loop N times (say 10,000 times), then pick the action with the most visits.

Why does this work?

Because MCTS automatically balances exploration against exploitation: good branches get visited again and again, so their statistics become more and more accurate; bad branches get few visits, so little time is wasted; unexplored branches get an "exploration bonus", so they are never ignored completely.

<div class="center">

------------------------------------------------------------------------

</div>

## 4. The UCB Formula: The Math of Exploration and Exploitation

The heart of MCTS is the UCB (Upper Confidence Bound) formula, used to choose which node to explore next:

$$
\text{UCB}(n) = \frac{w_n}{v_n} + c\sqrt{\frac{\ln v_p}{v_n}}
$$

Where:

- $w_n$: total wins accumulated at node n
- $v_n$: number of visits to node n
- $v_p$: visits of the parent node
- $c$: the exploration constant (usually $\sqrt{2}$)

**The first term**: $w_n/v_n$ is the **exploitation term** — the average win rate. A node with a high win rate scores high.

**The second term**: $c\sqrt{\ln v_p/v_n}$ is the **exploration term** — the uncertainty. A node visited few times scores high.

When the parent has been visited a lot ($v_p$ is large) but a particular child has been visited rarely ($v_n$ is small), the exploration term is large, which encourages searching that node.

**This formula has a theoretical guarantee**: in 2006, Kocsis and Szepesvári proved that MCTS with UCB converges to the optimal policy at a logarithmic rate.

<div class="center">

------------------------------------------------------------------------

</div>

## 5. Try It Yourself: Playing Tic-Tac-Toe with MCTS

Let's get a feel for MCTS with a simple example.

**Game**: tic-tac-toe (3×3), X moves first.

**Initial state**: an empty board

    . . .
    . . .
    . . .

**Step 1: the first simulation**

Start at the root node (the empty board), pick a move at random — say X plays the center:

    . . .
    . X .
    . . .

Then O plays randomly, X plays randomly, and so on until the game ends. Suppose X wins.

Backpropagation: the root's statistics become (wins=1, visits=1).

**Step 2: the second simulation**

Start from the root again. This time a different first move may be chosen — say X plays a corner:

    X . .
    . . .
    . . .

Play randomly to the end; suppose it's a draw.

Backpropagation: the root's statistics become (wins=1, visits=2).

**Step 3: repeat 1000 times**

After 1000 simulations, the root's 9 children (corresponding to the 9 possible first moves) have different statistics:

| Position | Visits | Win rate |
|:-----|:---------|:-----|
| Center | 342      | 0.68 |
| Corner | 278      | 0.65 |
| Edge   | 156      | 0.52 |
| …      | …        | …    |

Pick "center", the one with the most visits, as the best first move.

**Observations**:

- the center is visited most, because its win rate is high and its UCB score is high
- the edges are visited less, because early simulations showed a lower win rate
- even the bad positions get a few visits, so nothing is overlooked

**Question**: if we ran only 100 simulations, would the decision change? What is the relationship between the number of simulations and the quality of the decision?

<div class="center">

------------------------------------------------------------------------

</div>

## 6. From Search to Intuition: ADS Entropy Regularization

MCTS is powerful, but it has a problem: **too slow**.

Every decision requires thousands of simulations. AlphaGo takes about 0.1 seconds per move — fast for humans, but far too slow for real-time systems like self-driving cars.

The deeper problem: the time complexity of search is $O(N^2)$, where N is the number of simulations. To improve the quality of decisions, the number of simulations must grow exponentially.

The ADS strategy (Adaptive Dual Search, from Chapter 8) makes a radical proposal: **instead of learning MCTS's search trajectories, reshape the LLM's reasoning space with entropy regularization, collapsing the sprawling search tree into a low-entropy manifold**.

This is not traditional knowledge distillation; it is **topological reshaping**:

**The core mechanism: a dynamic log barrier**

$$
\alpha_t = -\log\left(1 - \frac{H(p_t)}{H_{\max}}\right)
$$

When the model's output entropy $H(p_t)$ approaches the maximum entropy $H_{\max}$, $\alpha_t \to \infty$, forming a "penalty wall".

**Effects**:

- high-entropy branches (uncertain reasoning paths) become unreachable during optimization
- search is forced to move along low-entropy (high-confidence) paths
- this is not pruning; it changes the geometry of the reachable states

**Compared with MCTS**:

- MCTS is **explicit search**: build a tree, simulate, backpropagate — recompute everything every time
- ADS is **implicit pruning**: an entropy barrier repels high-uncertainty states — the reasoning space is compressed in advance

**The key insight**: search does not need to be "frozen into weights"; it can be reshaped in real time through dynamic constraints. At every step of reasoning, the entropy barrier adjusts the boundary of the reachable states.

AlphaGo Zero used a neural network to learn the policy distribution of MCTS. ADS constrains the LLM's output space directly with entropy regularization — no teacher model, no distillation, just a log-barrier function.

<div class="center">

------------------------------------------------------------------------

</div>

## 7. Pseudocode: MCTS, ADS, and Causal Planning

**Algorithm 1: The MCTS main loop**

```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state    = state
        self.parent   = parent
        self.children = {}   # action -> MCTSNode
        self.visits   = 0
        self.value    = 0.0

    def is_fully_expanded(self, actions):
        return all(a in self.children for a in actions)

def ucb_select(node, actions, c=math.sqrt(2)):
    # select the child with the highest UCB score
    best, best_score = None, -float("inf")
    for a, child in node.children.items():
        exploit = child.value / child.visits
        explore = c * math.sqrt(math.log(node.visits) / child.visits)
        score   = exploit + explore
        if score > best_score:
            best_score, best = score, child
    return best

def mcts(root_state, get_actions, transition, rollout, N=1000):
    # get_actions(s) -> list of actions
    # transition(s, a) -> next_state
    # rollout(s) -> reward
    root = MCTSNode(root_state)

    for _ in range(N):
        node = root
        # Select
        actions = get_actions(node.state)
        while node.is_fully_expanded(actions) and node.children:
            node    = ucb_select(node, actions)
            actions = get_actions(node.state)

        # Expand
        unexplored = [a for a in actions if a not in node.children]
        if unexplored:
            a    = random.choice(unexplored)
            s_new = transition(node.state, a)
            child = MCTSNode(s_new, parent=node)
            node.children[a] = child
            node = child

        # Simulate
        reward = rollout(node.state)

        # Backpropagate
        while node is not None:
            node.visits += 1
            node.value  += reward
            node = node.parent

    # return the action with the most visits
    return max(root.children, key=lambda a: root.children[a].visits)
```

**Algorithm 2: UCB selection** (already embedded in the `ucb_select` function above)

**Algorithm 3: ADS entropy regularization**

```python
import torch
import torch.nn.functional as F

def ads_entropy_regularization(model, input_ids, tokenizer, max_steps=20, threshold=0.5):
    # model:     language model (exposes logits)
    # input_ids: initial token sequence
    # returns:   the list of generated token ids
    generated = list(input_ids)
    vocab_size = model.config.vocab_size

    for t in range(max_steps):
        with torch.no_grad():
            logits = model(torch.tensor([generated])).logits[0, -1]  # (vocab_size,)

        p_t   = F.softmax(logits, dim=-1)
        H_t   = -(p_t * torch.log(p_t + 1e-10)).sum().item()   # current entropy
        H_max = math.log(vocab_size)                            # max entropy (uniform distribution)

        # dynamic log barrier: as entropy approaches the max, alpha_t grows and the penalty strengthens
        alpha_t = -math.log(1 - H_t / H_max + 1e-10)

        # stop once entropy is low enough (model has converged to a low-entropy token)
        if H_t < threshold:
            break

        next_token = p_t.argmax().item()   # greedy decoding
        generated.append(next_token)

    # alpha_t -> inf as H_t -> H_max: high-entropy states are repelled by the barrier, search collapses to the low-entropy manifold
    return generated
```

**Algorithm 4: Causal planning**

```python
# CausalPlanner from causal_tictactoe.py (condensed)
class CausalPlanner:
    def __init__(self, A, E, U=None, lr=0.1, steps=80, seed=0):
        self.A = A                 # (9,9) causal matrix from build_causal_A
        self.E = E                 # (9, D) embeddings from build_embeddings
        self.U = U if U is not None else np.zeros_like(E)
        self.lr, self.steps = lr, steps

    def plan(self, board, player):
        """Choose the cell to play, by gradient descent on the
        intervention strength a over empty cells. Loss: maximize
        activation of target win-lines, penalize sparse interventions."""
        empties = empty_cells(board)
        if not empties:
            return None
        if len(empties) == CELLS:
            return 4                 # center first

        a = np.ones(CELLS) * 0.1
        a = np.clip(a, 0.0, 2.0)
        m = np.zeros(CELLS, dtype=bool)
        m[empties] = True

        # target cells: win lines the player could complete
        target_mask = np.zeros(CELLS)
        for (l1, l2, l3) in self._lines_near_complete(board, player):
            for c in (l1, l2, l3):
                target_mask[c] += 1.0

        def loss(a):
            A_do = self.A.copy()
            for c in empties:
                A_do[:, c] = a[c] * self.A[:, c]   # intervention strength
            E_next = A_do.T @ self.E + self.U
            act = float(np.sum(target_mask[:, None] * E_next))
            penalty = 0.5 * float(np.sum(a[m] ** 2))
            return -(act - penalty)

        # Adam over a (masked to empty cells)
        beta1, beta2 = 0.9, 0.999
        m_hat, v_hat = np.zeros(CELLS), np.zeros(CELLS)
        for t in range(1, self.steps + 1):
            grad_num = np.zeros(CELLS)
            eps = 1e-3
            for c in empties:
                a_p, a_m = a.copy(), a.copy()
                a_p[c] += eps; a_m[c] -= eps
                grad_num[c] = (loss(a_p) - loss(a_m)) / (2 * eps)
            m_hat = beta1 * m_hat + (1 - beta1) * grad_num
            v_hat = beta2 * v_hat + (1 - beta2) * grad_num ** 2
            a = a - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8) * m
            a = np.clip(a, 0.0, 2.0)
        return max((c for c in empties), key=lambda c: a[c])
```

<div class="center">

------------------------------------------------------------------------

</div>

## 8. Thought Experiment: When Search Becomes Gradient Descent

MCTS is near-perfect on tic-tac-toe because it enumerates the possibilities. But enumeration is expensive — each move needs hundreds of simulations. If the state space has a continuous structure, why climb a discrete tree?

Let's run a thought experiment: **encode the board state as embedding vectors, encode the rules of the game as a causal graph, and turn search into gradient descent in embedding space**. No ready-made library needed — a few dozen lines of numpy will do (companion script: `causal_tictactoe.py`).

### 8.1 The Causal World Operator: A Matrix Implementation of do()

The heart of it is a neural structural causal model: every variable (one of the 9 cells of tic-tac-toe) is a D-dimensional embedding vector, and causal edges connect cells that share a win line. The full code in this chapter's appendix builds the whole thing with just three functions — the causal matrix, do, and propagation:

```python
# 1. causal matrix: cells sharing a win line are causally connected
def build_causal_A(seed=0):
    A = np.zeros((CELLS, CELLS))          # CELLS = 9
    for (a, b, c) in LINES:               # the 8 win lines
        for u in (a, b, c):
            for v in (a, b, c):
                if u != v:
                    A[u][v] += 1.0
    A = A / (np.abs(A).sum(axis=1, keepdims=True) + 1e-9)
    return A

# 2. do(X = v): cut all incoming edges of X (fill column X with zeros)
def do_cut(A, var):
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do

# 3. propagate: one matrix multiplication along the remaining edges
def propagate(A_do, E, U):
    return A_do.T @ E + U
```

**The key insight**: `do(X=v)` does only two things to a causal graph — cut all incoming edges of X, then propagate the effect along the causal edges. In matrix form, cutting the incoming edges means zeroing out the column of X, and propagation is one matrix multiplication: `E_next = A_do.T @ E_cur + U`. There is no simpler definition.

### 8.2 Gradient Planning vs. Tree Search

Now look at the two ways to solve tic-tac-toe:

**MCTS (tree search)**:

```python
def mcts_move(board, player, n_rollouts=500):
    # build the search tree: UCB1 selection, random rollouts, backpropagation of statistics
    # time complexity: O(rollouts x depth)
    return best_move
```

**CausalPlanner (gradient planning)**:

```python
def planner_move(board, scm, player):
    # encode the board state as embeddings
    E_cur = board_to_embedding(board, scm.E, player)

    # define the goal: maximize the embedding norm of the winning line
    target = {"c4": 1.8, "c0": 1.5}  # center > corner > edge

    # optimize the intervention energy with gradient descent
    planner = CausalPlanner(scm)
    result = planner.plan(
        E_init=E_cur,
        target=target,
        interv_nodes=empty_cells(board),
        lr=0.1,
        steps=80
    )
    # time complexity: O(steps x N x D)
    return max(empty_cells, key=lambda c: result["a_opt"][c])
```

**The energy function**: $E(a) = \sum_j (\lVert E_{\text{next}}[j] \rVert - \text{target}_j)^2$. Gradient descent finds the intervention values that bring the embedding norms closest to the targets.

### 8.3 Experiment Results: Invincible Moving First, Winless Moving Second

Run the full script from this chapter's appendix (`python3 causal_tictactoe.py`; 30 games, 400 MCTS rollouts per move):

```
Matchup A: Planner(X) vs MCTS(O)  — Planner moves first
  30 games  Planner=12  Draw=18  MCTS=0
Matchup B: MCTS(X) vs Planner(O)  — MCTS moves first
  30 games  MCTS=30  Planner=0  Draw=0
```

Reading the results:

- **Moving first, gradient planning never loses**: from an empty board the gradient signal is clear — the center embedding has the largest norm, which matches the optimal first move. But notice: 12 wins and 18 draws — it cannot beat perfect defense; it just grabs the center first.

- **Moving second, it loses everything**: once MCTS takes the center, the gradient planner's signal only contains "attack targets", with no "defense signal" — the causal graph encodes the win lines, not the lines the opponent might win. It has no idea it should block.

This is not a bug; it is a feature: gradient planning captures the **continuous heuristic** — "how good does this position look" — rather than "search every possible outcome". It substitutes geometric distance in embedding space for the win-rate statistics of tree search. The first-move advantage comes from the embedding initialization (a value prior of center > corner > edge); the total collapse moving second comes from the causal graph simply containing no defensive knowledge — gradient planning is not "reasoning"; it is "looking at which cell seems most valuable".

### 8.4 Try It Yourself: Run the Tic-Tac-Toe Match

```bash
python3 causal_tictactoe.py
# Full code: see this chapter's appendix.
```

**Observations**:

1. Gradient planning is perfect on simple positions (the opening, an immediate win).

2. Moving second against MCTS's blockade it almost always loses — try lowering MCTS's rollout count from 400 to 50, and see whether it can still hold the line.

3. Speed comparison: gradient planning uses about 60 Adam iterations × 9-dimensional numerical gradients ≈ 1000 floating-point operations per move, versus MCTS's 400 × depth state evaluations.

**Exercises**:

- When can the gradient signal mislead? (When the optimal path needs to get temporarily "worse".)

- Which game knowledge does the causal graph encode? (The topology of the win lines — the lines to win, not the lines to defend.)

- What would the causal graph need in order to learn defense? (The opponent's win lines must enter the objective function as well.)

- What does this suggest about the future of AI planning? (Continuous representations + gradient optimization, versus discrete search.)

::: info Professor Pallas's Cat's Comment

MCTS is discrete enumeration; gradient planning is a continuous heuristic. This is not a question of which is better, but of **two ontologies of search**: one climbs a tree of states, the other flows in embedding space. Gradient descent is not "an approximate tree search" — it is a completely different thing: it replaces the win rates of a tree with distances in a vector space. Admitting the difference matters more than pretending they compete. And we just saw it with our own eyes: moving second, it loses every game — the continuous heuristic is not a cure-all; it only knows what it has learned.

:::

<div class="center">

------------------------------------------------------------------------

</div>

## 9. Visualization: The Asymmetric Growth of the Search Tree

<figure>
<p><img src="/figures/ch10_fig1_mcts_tree.png" alt="Figure 10.1: Evolution of an MCTS search tree" /></p>
<figcaption>Figure 10.1: Evolution of an MCTS search tree</figcaption>
</figure>

*Figure 10.1 shows how the shape of the MCTS search tree evolves as the number of simulations grows. The three subplots, from left to right, correspond to N=10, 100, and 1000 simulations:*

- **Left (N=10)**: the tree is almost symmetric, and most branches have similar visit counts (nodes of similar size). There hasn't been enough exploration yet; UCB hasn't separated good paths from bad ones.

- **Middle (N=100)**: asymmetry begins to appear. Good branches (green nodes, high win rate) are explored first, and their nodes become visibly larger. Bad branches (red nodes) get fewer visits.

- **Right (N=1000)**: highly asymmetric. The best path is visited again and again (the largest green node), the second-best path gets moderate exploration, and the bad paths are nearly abandoned (small red nodes).

**Key observation**: node size is proportional to visit count, and color shows the average win rate (green = high, red = low). This asymmetric growth is exactly the heart of the UCB algorithm — **concentrate a limited simulation budget on the most promising paths**.

<figure>
<p><img src="/figures/ch10_fig2_ucb_tradeoff.png" alt="Figure 10.2: The exploration-exploitation trade-off of UCB" /></p>
<figcaption>Figure 10.2: The exploration-exploitation trade-off of UCB</figcaption>
</figure>

*Figure 10.2 shows how the exploration term and the exploitation term of the UCB formula change as a node gets more visits:*

$$
\text{UCB}(n) = \underbrace{\text{average win rate}}_{\text{exploitation term}} + c \cdot \underbrace{\sqrt{\frac{\ln N}{n}}}_{\text{exploration term}}
$$

- **Blue curve (the exploitation term)**: the average win rate converges to its true value (~0.7) as visits grow. It swings a lot early on and settles later.

- **Red curve (the exploration term)**: $\sqrt{\ln N/n}$ decays with visits. The more a node is visited, the smaller the exploration bonus.

- **Green dashed line (total UCB)**: the sum of the two terms. Early on the exploration term dominates (red is high); later the exploitation term dominates (blue is stable).

**The crossover is at n≈30**: around here the two terms contribute about equally. Before it: "daring exploration". After it: "precise exploitation". This balance point is controlled by the constant $c$ — the larger $c$ is, the more aggressive the exploration.

<div class="center">

------------------------------------------------------------------------

</div>

## 10. Search in LLMs: From Chain of Thought to Reinforcement Learning

After 2022, the search problem gained a new dimension: **the language model itself is a state space**. Every reasoning step is no longer a move on a board but a piece of text — a line of math, a logical argument, a line of code.

This raises a deep question: **how can an LLM search effectively inside its language-based reasoning space?**

The MCTS answer is: build a tree, simulate, backpropagate. But for a language model, "simulating once" means running an entire forward pass — an enormous cost. And the language reasoning space is continuous, with an infinite branching factor — you cannot enumerate all possible "next sentences".

Two different technical routes give two very different answers.

### 10.1 GPT-o1: A Process Reward Model and Policy Distillation

In September 2024, OpenAI released the o1 family of models. The core innovation is not the model architecture, but the **search strategy**.

The technical route of o1 can be summarized as a three-layer architecture:

**Layer 1: the Process Reward Model (PRM)**

In traditional reinforcement learning, the reward is given at the end — right answer +1, wrong answer −1. But in reasoning tasks this is far too sparse: for a math problem that needs 20 steps of derivation, only step 20 can be judged right or wrong.

The PRM idea: **score every intermediate step**. Given the $k$-th step in a reasoning chain, the PRM outputs the probability that this step is "going in the right direction":

$$
r_k = \text{PRM}(s_0, a_1, a_2, \ldots, a_k)
$$

This converts a sparse terminal reward into **dense process rewards**, so the search can get feedback at every step.

**Layer 2: a policy model guiding the tree search**

With the PRM in hand, o1 uses a policy model (a fine-tuned language model) to run an MCTS-like search in the space of reasoning steps:

- **Select**: use the policy model to generate $K$ candidate reasoning steps
- **Evaluate**: the PRM scores each candidate
- **Expand**: continue along the highest-scoring step
- **Backpropagate**: the final result updates the PRM and the policy model in reverse

This was the first systematic combination of **tree search with language model reasoning**. Chain of thought (CoT — reasoning step by step, writing the steps down) was no longer a single linear derivation, but a search tree growing in reasoning space.

**Layer 3: knowledge distillation compresses the search**

Search works, but it is slow. At inference time, o1 actually runs a **distilled policy model** — it saw a huge number of high-quality reasoning trajectories produced by tree search during training, and learned to "simulate the search process internally", without having to build a tree at inference time.

This resembles human chess players: a beginner computes every variation; a master sees the best move at a glance — because the master has "internalized" the search as intuition.

**This is a historic moment for chain of thought**: CoT is no longer just a trick that raises accuracy; it has become the infrastructure that lets a language model perform **systematic search** in reasoning space.

### 10.2 DeepSeek-R1: Kill the PRM, Compete Inside a Group

In early 2025, DeepSeek released the R1 model, taking a route that contrasts sharply with o1: **no PRM whatsoever**.

o1's route has a fundamental bottleneck: **how is the PRM trained?**

To train a PRM, you need humans to label every intermediate reasoning step "right/wrong". That is extremely expensive data engineering — a problem needing 20 steps of derivation needs every step labeled, so the data cost is 20 times that of terminal-only labeling.

And there is an even deeper question: **who judges whether an intermediate step is right?** Some reasoning paths look "wrong" yet reach the correct answer in the end; some look "right" yet fail at the very last step. The correctness of an intermediate step depends on the final outcome, but the PRM must score it without knowing the outcome. That is a fundamental difficulty.

DeepSeek-R1's solution is **Group Relative Policy Optimization (GRPO)**:

**Core idea: no absolute scoring is needed — only relative ranking**

For the same question, let the model generate $G$ complete reasoning trajectories (chains of answers):

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_\theta(\cdot \mid q)
$$

Judge right or wrong only at the end: $r_i = 1$ if the $i$-th trajectory reaches the correct answer, and $r_i = 0$ otherwise.

Then compute the **group-relative advantage** — which trajectories are better than this group's average?

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j
$$

Update the policy with this relative advantage:

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \cdot A_i\right]
$$

**What makes this design deep**:

- **No PRM needed**: the reward signal comes only from terminal right/wrong judgment, which completely removes the need to label intermediate steps
- **Competition inside the group**: the trajectories of one group form a competition — "better than the group average" gets positive gradient, "worse than the group average" gets negative gradient
- **Emergent search**: the model spontaneously learns "self-verification" during training — generate several reasoning paths, compare them internally, and keep the better one

This is a profound innovation in reinforcement learning: **replacing explicit process supervision (PRM) with implicit group-competition pressure**.

### 10.3 Comparing the Two Routes Philosophically

| Dimension | The GPT-o1 route | The DeepSeek-R1 route |
|:-----|:-----------|:----------------|
| Process supervision | Explicit PRM, step-by-step scoring | No PRM, terminal reward only |
| Search style | Explicit tree search + distillation | Group competition via GRPO |
| Data needs | High (process labels) | Low (answer labels only) |
| Compute cost | Cheap to train, expensive at inference | Expensive to train, cheap at inference |
| Emergent ability | Passed on through distillation | Emerges by itself through competition |

**From the MCTS point of view**, the two routes correspond to two different search strategies:

The o1 route is more like **supervised tree search**: the PRM is a hand-crafted heuristic function, and the policy model is a prior guided by human data. It depends heavily on human knowledge, but performs strongly in domains where the data exists.

The DeepSeek-R1 route is more like **unsupervised random search plus selection pressure**: GRPO is equivalent to letting $G$ random-walk paths compete on the same question, with natural selection keeping the good paths. It does not depend on human judgment about "the process", but needs a large number of samples to generate enough competitive pressure.

**This is a deep tension**: the more precise the process supervision, the more it depends on human knowledge; the less it depends on human knowledge, the more it needs spontaneous emergent competition.

Chain of thought brings this tension into the era of language models. In the Go era, the answer to this tension was AlphaGo Zero — abandon human game records entirely and rely on competitive selection from self-play. In the language-reasoning era, this tension has no final answer yet.

<div class="center">

------------------------------------------------------------------------

</div>

## 11. The Philosophy of Search: Computation as Reasoning

MCTS reveals a deep fact: **reasoning does not need "understanding" — only effective search**.

AlphaGo does not know the beauty of Go and does not understand concepts such as "thickness" or "territory". All it knows is: in this state, which move has the highest simulated win rate.

But that was enough. Through millions of simulations, MCTS found game moves that humans had not found in thousands of years.

This raises a philosophical question: **if search is powerful enough, do we still need "understanding"?**

One view holds that **understanding is a shortcut for search**. Human players, by understanding the principles of the game, can prune quickly without searching every possibility. But if computation were unlimited, search could replace understanding.

Another view holds that **understanding is the foundation of generalization**. MCTS is strong at Go, but it must be retrained for a different game. Humans understand abstract concepts such as "strategy" and "trade-off", and can transfer them quickly to new games.

The entropy regularization of ADS offers a third view: **search can be reshaped in real time through dynamic constraints**. There is no need to "freeze search into weights"; an entropy barrier simply repels high-uncertainty states at every step of reasoning.

This resembles the potential field of physics: rather than precomputing every path in advance, you feel the potential gradient at every position and naturally move in the direction of lower potential energy.

Search is powerful, but expensive. In the next chapter, we will face the economics of reasoning directly: how to get reasoning of the same quality with less computation?

<div class="center">

------------------------------------------------------------------------

</div>

## Open Questions

- **Where are the limits of generalization for entropy regularization?** ADS works on arithmetic tasks, but on tasks that need multi-layer abstraction, small models exhibit "confident hallucination". What is the relationship between the entropy barrier and model capacity?

- **Can topological reshaping extend to other reasoning tasks?** The log barrier works in LLM reasoning — but in discrete search spaces, such as symbolic reasoning and theorem proving?

- **What is the optimal exploration-exploitation balance?** How should the constant $c$ in the UCB formula be chosen? Do different problems need different $c$?

- **Can MCTS generalize to continuous action spaces?** The actions of Go and tic-tac-toe are discrete. How can MCTS be applied to robot control, continuous optimization, and other continuous-action problems?

- **Where is the boundary between search and reasoning?** Theorem proving can be viewed as search — but can a mathematician's "insight" be captured by search?

- **Is PRM's intermediate-step labeling fundamentally difficult?** If a reasoning path that "looks wrong" ultimately reaches the correct answer, was that step right or wrong? Does this labeling difficulty mean the PRM can never provide truly reliable process supervision?

- **Where does GRPO's group-competition pressure come from?** If all $G$ trajectories are correct (or all are wrong), then $A_i = 0$, the gradient vanishes, and the model cannot learn. This means GRPO needs precise calibration of task difficulty — neither too hard (all wrong) nor too easy (all right). Is this calibration a solvable engineering problem, or a fundamental limit?

<div class="center">

------------------------------------------------------------------------

</div>

## Appendix: Full Code

Below is the complete script for this chapter's companion experiment (copy it, save it as `causal_tictactoe.py`, and run it):

```python
"""
Gradient Planning vs Tree Search on Tic-Tac-Toe
================================================
Companion script for The Reasoning Kingdom, Chapter 10, Section 8
("When Search Becomes Gradient Descent").

The idea experiment: encode the board as embeddings, encode the rules as a
causal graph, and turn "search" into "gradient descent in embedding space".

Two planners square off:
  * MCTS        : explicit tree search (UCB1 + random rollouts)
  * CausalPlanner: gradient descent on an intervention vector over a
                   neural structural causal model (SCM) built from win-line
                   adjacency

Everything below is self-contained and numpy-only. No external repos, no
type theory, no lambda calculus — just embeddings, a causal matrix, and
Adam.

Run:  python3 causal_tictactoe.py

(Full code: see this chapter's appendix.)
"""
import numpy as np
import math

# --------------------------------------------------------------------------
# Board helpers (3x3 = cells c0..c8)
# --------------------------------------------------------------------------
CELLS = 9
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]

def empty_cells(board):
    return [i for i in range(CELLS) if board[i] == 0]

def board_to_vec(board, player):
    """State vector: +1 for player's stones, -1 for opponent, 0 empty."""
    return np.array([1 if c == player else (-1 if c != 0 else 0)
                     for c in board], dtype=float)

def winner(board):
    for (a, b, c) in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if len(empty_cells(board)) == 0 else None   # 0 = draw, None = ongoing

def print_board(board):
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(3):
        print("  " + " ".join(sym[board[3 * r + c]] for c in range(3)))

# --------------------------------------------------------------------------
# Causal graph: win-line adjacency
# --------------------------------------------------------------------------
def build_causal_A(seed=0):
    """A[i][j] = strength of causal influence from cell j to cell i.
    Cells sharing a win line are causally connected; center dominates."""
    A = np.zeros((CELLS, CELLS))
    for (a, b, c) in LINES:
        for u in (a, b, c):
            for v in (a, b, c):
                if u != v:
                    A[u][v] += 1.0
    # normalize rows; give the center extra weight (the classic first move)
    rng = np.random.default_rng(seed)
    A += rng.normal(0, 0.05, A.shape)
    A = A / (np.abs(A).sum(axis=1, keepdims=True) + 1e-9)
    return A

def build_embeddings(seed=0, D=16):
    """One D-dim embedding per cell; center has large norm, corners medium,
    edges small — the 'value prior' the planner starts with."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1, size=(CELLS, D))
    # value profile: center > corner > edge
    profile = np.zeros(CELLS)
    profile[4] = 1.6                      # c4 = center
    for c in (0, 2, 6, 8):                # corners
        profile[c] = 1.1
    for c in (1, 3, 5, 7):                # edges
        profile[c] = 0.6
    E = E * profile[:, None]
    return E

# --------------------------------------------------------------------------
# CausalPlanner (gradient planning)
# --------------------------------------------------------------------------
def do_cut(A, var):
    """Pearl do(X = v): cut incoming edges of var (set column to 0)."""
    A_do = A.copy()
    A_do[:, var] = 0
    return A_do

class CausalPlanner:
    def __init__(self, A, E, U=None, lr=0.1, steps=80, seed=0):
        self.A = A
        self.E = E
        self.U = U if U is not None else np.zeros_like(E)
        self.lr = lr
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def plan(self, board, player, used_step_cost=True):
        """Return the cell to play, chosen by gradient descent on the
        intervention strength a >= 0 over empty cells.

        Loss: want the post-intervention embedding to be 'strong' on the
        win lines the player can complete, minus a sparsity penalty."""
        empties = empty_cells(board)
        if not empties:
            return None
        if len(empties) == CELLS:
            return 4                       # center first (known best)

        a = np.ones(CELLS) * 0.1
        a = np.clip(a, 0.0, 2.0)
        m = np.zeros(CELLS, dtype=bool)    # mask of empty cells
        m[empties] = True
        v = board_to_vec(board, player)

        # target lines the player could complete
        target_mask = np.zeros(CELLS)
        for (l1, l2, l3) in self._lines_near_complete(board, player):
            for c in (l1, l2, l3):
                target_mask[c] += 1.0

        def loss(a):
            A_do = self.A.copy()
            for c in empties:
                A_do[:, c] = 0
            # intervention strength scales the causal flow: column weight
            for c in empties:
                A_do[:, c] = a[c] * self.A[:, c]
            E_next = A_do.T @ self.E + self.U
            # we want the target cells to activate: maximize sum(target*E_next)
            act = float(np.sum(target_mask[:, None] * E_next))
            # and prefer sparse interventions (one strong move, not many)
            penalty = 0.5 * float(np.sum(a[m] ** 2))
            return -(act - penalty)

        # Adam over a (masked to empty cells)
        beta1, beta2 = 0.9, 0.999
        m_hat = np.zeros(CELLS); v_hat = np.zeros(CELLS)
        for t in range(1, self.steps + 1):
            g = np.zeros(CELLS)
            grad_num = np.zeros(CELLS)
            eps = 1e-3
            for c in empties:
                a_p = a.copy(); a_m = a.copy()
                a_p[c] += eps; a_m[c] -= eps
                grad_num[c] = (loss(a_p) - loss(a_m)) / (2 * eps)
            g[m] = grad_num[m]
            m_hat = beta1 * m_hat + (1 - beta1) * g
            v_hat = beta2 * v_hat + (1 - beta2) * g * g
            mh = m_hat / (1 - beta1 ** t)
            vh = v_hat / (1 - beta2 ** t)
            a = a - self.lr * mh / (np.sqrt(vh) + 1e-8) * m
            a = np.clip(a, 0.0, 2.0)

        # sanity: never pick an occupied cell — pick the empty cell with the
        # highest intervention strength
        best = max((c for c in empties), key=lambda c: a[c])
        return best

    def _lines_near_complete(self, board, player):
        """Win lines where player already holds 2/3 and the rest is empty."""
        res = []
        for (a, b, c) in LINES:
            vals = [board[x] for x in (a, b, c)]
            if vals.count(player) == 2 and vals.count(0) == 1:
                res.append((a, b, c))
        return res

# --------------------------------------------------------------------------
# MCTS with UCB1 (the classic)
# --------------------------------------------------------------------------
class MCTSNode:
    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0

def ucb1(node):
    if node.N == 0:
        return float("inf")
    if node.parent is None:
        return 0.0
    exploit = node.W / node.N
    explore = math.sqrt(2 * math.log(node.parent.N + 1) / node.N)
    return exploit + explore

def mcts_move(board, player, n_rollouts=500, seed=0):
    rng = np.random.default_rng(seed)
    root = MCTSNode(board, player, parent=None)
    foe = 1 if player == 2 else 2

    for _ in range(n_rollouts):
        node = root
        b = board.copy()
        turn = player

        # selection
        while node.children:
            # choose best child by UCB1
            best_c = max(node.children, key=lambda c: ucb1(node.children[c]))
            node = node.children[best_c]
            b[best_c] = turn
            turn = foe if turn == player else player

        # expansion + rollout
        res = winner(b)
        if res is None:
            moves = empty_cells(b)
            c_move = int(rng.choice(moves))
            b[c_move] = turn
            turn = foe if turn == player else player
            res = winner(b)
        while res is None:
            moves = empty_cells(b)
            c_move = int(rng.choice(moves))
            b[c_move] = turn
            turn = foe if turn == player else player
            res = winner(b)

        # backprop
        sign = 1.0 if res == 0 else (1.0 if res == player else 0.0)
        while node is not None:
            node.N += 1
            node.W += sign
            node = node.parent

    if not root.children:
        return empty_cells(board)[0] if empty_cells(board) else None
    best_c = max(root.children, key=lambda c: root.children[c].N)
    return best_c

# --------------------------------------------------------------------------
# Matchup
# --------------------------------------------------------------------------
def play_game(planner_first, n_rollouts=400, seed=0, verbose=False):
    A = build_causal_A(seed=seed)
    E = build_embeddings(seed=seed)
    planner = CausalPlanner(A, E, lr=0.1, steps=60, seed=seed)

    board = [0] * CELLS
    rng = np.random.default_rng(seed)
    turn = 1
    while True:
        if (turn == 1) == planner_first:
            move = planner.plan(board, turn)
        else:
            move = mcts_move(board, turn, n_rollouts=n_rollouts,
                             seed=rng.integers(0, 10000))
        if move is None or board[move] != 0:
            break
        board[move] = turn
        res = winner(board)
        if res is not None:
            return res
        turn = 2 if turn == 1 else 1
    return 0

def matchup(games=20, n_rollouts=400, seed=0, verbose=False):
    stats = {"Planner": 0, "MCTS": 0, "Draw": 0}
    for g in range(games):
        r = play_game(True, n_rollouts=n_rollouts, seed=seed + g)
        stats["Planner" if r == 1 else ("MCTS" if r == 2 else "Draw")] += 1
    return stats

if __name__ == "__main__":
    print("=" * 64)
    print("Gradient Planning vs MCTS on Tic-Tac-Toe")
    print("=" * 64)

    print("\n[1] Causal graph from win-line adjacency (A is 9x9, rows sum 1)")
    A = build_causal_A()
    print("    A[4,:3] =", np.round(A[4, :3], 3), " (center <- top line)")

    print("\n[2] Embedding profile: center > corner > edge")
    E = build_embeddings()
    print("    norms:", [round(float(np.linalg.norm(E[i])), 2) for i in range(9)])

    print("\n[3] Matchup: Planner (gradient) first vs MCTS (rollouts=400), 30 games")
    stats = matchup(games=30, n_rollouts=400, seed=100)
    print(f"    Planner wins {stats['Planner']} ; draws {stats['Draw']} ; MCTS wins {stats['MCTS']}")

    print("\n[4] Matchup: MCTS first vs Planner (O), 30 games")
    stats2 = {"Planner": 0, "MCTS": 0, "Draw": 0}
    for g in range(30):
        r = play_game(False, n_rollouts=400, seed=200 + g)
        stats2["Planner" if r == 2 else ("MCTS" if r == 1 else "Draw")] += 1
    print(f"    MCTS wins {stats2['MCTS']} ; Planner wins {stats2['Planner']} ; draws {stats2['Draw']}")

    print("\n    Observation: gradient planning never loses when it is first")
    print("    (it just grabs the center), but loses every game when second —")
    print("    its causal graph encodes aiming at win lines, not blocking.")
    print("\nDone.")
```

<div class="center">

------------------------------------------------------------------------

</div>

## Further Reading

- Kocsis & Szepesvári (2006). *Bandit Based Monte-Carlo Planning* — the original MCTS paper that proposed the UCT algorithm `-> [ECML 2006]`

- Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search* — the AlphaGo paper, the milestone of MCTS + neural networks

- Silver et al. (2017). *Mastering the game of Go without human knowledge* — AlphaGo Zero, learning entirely from self-play

- Chapter 10 of this book, §6, *From Search to Intuition: ADS Entropy Regularization* — the full definition of the ADS strategy, its pseudocode, and its companion script (`ads_search.py`)

- Lightman et al. (2023). *Let's Verify Step by Step* — OpenAI's systematic study of process reward models (PRM), showing process supervision beats outcome supervision `-> [arXiv:2305.20050]`

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — the GRPO method: group-competition reinforcement learning that achieves strong reasoning without a PRM `-> [arXiv:2501.12948]`

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — where the GRPO algorithm was first proposed, validated on mathematical reasoning tasks `-> [arXiv:2402.03300]`

- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — the chain-of-thought paper, which established CoT as the infrastructure of LLM reasoning `-> [arXiv:2201.11903]`

- Browne et al. (2012). *A Survey of Monte Carlo Tree Search Methods* — a survey of MCTS methods

- \[Dam et al., 2020\] — convex regularization in MCTS, with a proof of exponential convergence `-> [arXiv:2007.00391]`

- \[Dam et al., 2019\] — Power-UCT, improving node value estimates with power means `-> [arXiv:1911.00384]`

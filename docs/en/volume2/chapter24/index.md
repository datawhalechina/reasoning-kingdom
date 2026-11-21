# Chapter 24: Algorithmic Convergence — The LIT Theorem

---

$A_L$ set; $A_L$ resident: $1_L$

$X_L=A_L^D$; $A_L=\{0,1\}^L\cup\{!\}$

$(X_L,d_w)$ $\Rightarrow$ space

Route: the triviality equation of algorithmic convergence — discrete, bounded, sequentially iterated $\to$ the value set keeps only $A_L=\{0,1\}^L$ and the negation $!$ $\to$ weighted Hamming space $X_L=A_L^D$ $\to$ construct the iteration sequence $\to$ well-ordering gives the stopping time, no limit taken $\to$ weighted Hamming gives $2^L+1$ $\to$ clauses cover, hence collapse to UNSAT $\to$ the solution space collapses $\to$ the convergence of the whole Kingdom is put into one equation; P and NP are just two samples.

## Preface: Enumerate First, Then Talk About Convergence

A clause, three variables. However the literals are arranged, whatever values are plugged in, whatever operators are inserted, whatever conditions are added, in the end it all reads as the same thing: under this assignment, is the clause satisfied? Beyond that, it has no other behavior.

So we do not guess its structure; we list its behavior. The three variables each take $0$ or $1$: eight assignments. One assignment, one answer; eight answers make a complete table. Enumerate this table from beginning to end, missing nothing.

The eight answers do not run separately. Stack them together, collect them into one variable $a_n$, and push forward along $n=0,1,2,\dots$. Each transition updates the whole stack of answers at once. The optimal transition does not backtrack, does not correct mistakes; it walks only the successful path, and the stack is laid down layer by layer.

After this step, convergence becomes a computable question. $a_n$ lies in the space $X_L$, and the distance between two points is Hamming distance — count how many values differ. How much distance $T$ squeezes out per step is computed directly (§2).

The method is not mysterious; it is even stupid: not understanding the structure of the clause, so try every assignment. But to speak of a naive fixed point, one has to use this stupid method. A slightly cleverer method assumes structure from the start, and then naivety has nowhere to stand. The stupid method fills the whole space; only once it is full can one discuss which cell the fixed point lands in.

This chapter does not teach P, and it does not teach NP. It teaches the triviality equation of algorithmic convergence: discreteness, boundedness, sequential iteration. P and NP are just two samples of this equation.

### How to Play: Fill the Table, Light the Lamps

Laid out, this is the L-SAT game itself, only drawn differently: fill the table, light the lamps.

A clause is a row of lamps. In 3-SAT a clause has three lamps, one per literal. A literal is a variable or its negation; once the assignment is fixed, a true literal lights its lamp, a false one leaves it dark.

For example, $(x\vee\neg y\vee z)$ has three lamps watching $x$, $\neg y$, and $z$. Under an assignment, the clause is satisfied if at least one of the three lamps is lit; if all three are dark, the clause is false.

A table has many rows, one clause per row. The whole formula is the conjunction of these clauses: every row must have at least one lit lamp, or the formula is not satisfied; one all-dark row and it fails.

The 3 in 3-SAT sits right here: three lamps per clause. 2-SAT has two lamps per clause. Insert all the Boolean operators you like — a clause is still these literals, and the lamps are still three; stack the operators as high as you want, when you count lamps you count literals.

Take three clauses and the assignment $x=1,y=0,z=1$:

| Clause | Lamp 1 | Lamp 2 | Lamp 3 | This row |
|:---|:---:|:---:|:---:|:---|
| $x\vee y\vee z$ | 🟩 $x$ | ⬜ $y$ | 🟩 $z$ | lit, satisfied |
| $\neg x\vee y\vee\neg z$ | ⬜ $\neg x$ | ⬜ $y$ | ⬜ $\neg z$ | all dark, unsatisfied |
| $x\vee\neg y\vee z$ | 🟩 $x$ | 🟩 $\neg y$ | 🟩 $z$ | lit, satisfied |

The second row is all dark, so this assignment fails. Try $x=0,y=1,z=1$:

| Clause | Lamp 1 | Lamp 2 | Lamp 3 | This row |
|:---|:---:|:---:|:---:|:---|
| $x\vee y\vee z$ | ⬜ $x$ | 🟩 $y$ | 🟩 $z$ | lit, satisfied |
| $\neg x\vee y\vee\neg z$ | 🟩 $\neg x$ | 🟩 $y$ | ⬜ $\neg z$ | lit, satisfied |
| $x\vee\neg y\vee z$ | ⬜ $x$ | ⬜ $\neg y$ | 🟩 $z$ | lit, satisfied |

Every row has at least one lit lamp, so this assignment satisfies the whole formula. The game is to find such an assignment. SAT asks exactly this: is there an assignment that lights at least one lamp in every row?

The book does not try assignments one at a time. It lays all eight assignments into one table, one cell per assignment, with the formula's truth value in the cell; all the answers are packed into one value function $a_n$, iterated along the time axis. Satisfiability is the old rule: at least one lit cell; only an all-dark table is the constant zero, and only that is unsatisfiable.

For the three clauses above, the eight-assignment truth table is:

| $000$ | $001$ | $010$ | $011$ | $100$ | $101$ | $110$ | $111$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⬜ | 🟩 | ⬜ | 🟩 | 🟩 | ⬜ | 🟩 | 🟩 |

Five cells lit, three dark; the number of lit cells is $\operatorname{satStates}$. With a lit cell, the formula is satisfiable. A cell holds only two values, all off $0$ or all on $1_L$; a lit cell is an assignment that makes the formula true.

Conversely, a fully dark table is the constant zero:

| $000$ | $001$ | $010$ | $011$ | $100$ | $101$ | $110$ | $111$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

$\operatorname{satStates}=0$, no assignment makes the formula true, unsatisfiable. The table stops in the dark, not a resident.

The book's iteration starts from the constant-one table $1_{X_L}$ and updates the whole table under the rule; each step produces a new table. Time moves only along a table that still has a lit cell; when the new table goes fully dark, the clock stops. The optimal transition counts successful steps, not lit cells.

Keep filling until the table stops changing: that is the fixed point. If the table at rest still has at least one lit cell, it is a resident; if the whole table is dark, it is the constant zero, not a resident.

The game has one question left: when the clock stops, is the table still lit? Lit, and there is SAT; fully dark, and there is none.

## 1. The Space

### 1.1 The value set $A_L$ (the trivial foundation)

For each $L$, define the trivial value set

$$
A_L:=\{0,1\}^L\cup\{!\}
$$

$|A_L|=2^L+1$. $A_L$ holds $L$-bit states, and it holds one operator, a single negation $!$. There is no $\&$ and no $|$: a disjunction or a conjunction is layout, a way of writing the clause down, not a value taken. The trivial resident takes the constant-one state

$$
1_L:=(1,\dots,1)\in\{0,1\}^L\subseteq A_L
$$

which on the space is the constant-one value function $1_{X_L}$.

### 1.2 Constructing $A_3$ and the complete space

Take $L=3$: $A_3:=A_L|_{L=3}$, made of 8 states and the negation, 9 in all:

| Number | State |
|-----:|:-----|
| 0 | $(0,0,0)$ |
| 1 | $(0,0,1)$ |
| 2 | $(0,1,0)$ |
| 3 | $(0,1,1)$ |
| 4 | $(1,0,0)$ |
| 5 | $(1,0,1)$ |
| 6 | $(1,1,0)$ |
| 7 | $(1,1,1)$ |
| 8 | $!$ |

$$
X_3:=A_3^D,\qquad |X_3|=9^8.
$$

A width-$3$ clause is determined by a sign pattern $s\in D$:

$$
C_s(x)=\bigvee_{i=1}^3
\begin{cases}
x_i, & s_i=0,\\
\neg x_i, & s_i=1.
\end{cases}
$$

$C_s$ is false only at the cell $x=s$, true everywhere else. Its truth table is

$$
f_s(x)=
\begin{cases}
1_L, & x\ne s,\\
0_L, & x=s.
\end{cases}
$$

A formula is a conjunction of clauses; the sign pattern of its $j$-th clause is $s_j$. Its truth table is the pointwise conjunction of the clause truth tables:

$$
F(x)=\bigwedge_{j=1}^{m}f_{s_j}(x).
$$

$F(x)=0$ exactly when some clause is false at $x$; $F(x)=1_L$ exactly when $x$ satisfies every clause. Satisfiable $\Leftrightarrow$ $F$ is not constant zero. A resident is a nonzero table; the constant-zero table is not a resident.

$X_3=A_3^D$ is a complete space: see 1.4 for the distance, 1.5 for completeness.

### 1.3 Constructing $A_2$ and the complete space

Take $L=2$: $A_2:=A_L|_{L=2}$, made of 4 states and the negation, 5 in all:

| Number | State |
|-----:|:-----|
| 0 | $(0,0)$ |
| 1 | $(0,1)$ |
| 2 | $(1,0)$ |
| 3 | $(1,1)$ |
| 4 | $!$ |

For the 2-SAT case take $D=\{0,1\}^2$,

$$
X_2:=A_2^D,\qquad |X_2|=5^4.
$$

$X_2$ is the complete space: see 1.4 for the distance, 1.5 for completeness.

In name, $L=2$ is 2-SAT and $L=3$ is 3-SAT; this chapter does not look at the class boundary, it looks at how these two spaces move.

### 1.3.1 Not filling in a 3-bit/2-bit table, but iterating the whole solution space along the time axis

$A_2$, $A_3$ do not mean “the input has only 2 or 3 bits”. They are pointwise enumeration value sets: $A_L$ enumerates every state of $\{0,1\}^L$ into itself, and $X_L=A_L^D$ packs all candidate value functions into one space.

- $D=\{0,1\}^3$ enumerates 8 candidate input states; $X_3=A_3^D$ is the entire candidate solution space.
- The iterated object $a_n\in X_3$ is not a single 3-bit assignment but the whole candidate table: the values at the 8 states are packed into $a_n$ at once and evolve together along the time mark $n$.
- The time gate looks at the newly filled table: as long as one cell is still lit ($\operatorname{satStates}(a_{n+1})\ge1$), the step is allowed to land; when the whole table is dark, time stops and cannot push outward.

So the “3” in 3-SAT is the clause width — three lamps per clause; the book’s $L=3$ round is three variables and an eight-cell table. The two 3s have the same value and different meanings.

### 1.3.2 Optimal transition: only successful advances count

Backtracking, correction, and hesitant branches do not enter the transition ledger. Only steps that truly move forward successfully are recorded:

$$
\operatorname{ReachesIn}\ f_0\ 0\ f_0
$$

$$
\operatorname{ReachesIn}\ f_0\ (n+1)\ g
\qquad
\text{if}\ \operatorname{ReachesIn}\ f_0\ n\ f\ \text{and}\ \operatorname{stepSpace}\ f=\operatorname{some}\ g
$$

$\operatorname{IsOptimalTime}\ f_0\ \text{goal}\ n$ takes the number of transitions on the shortest successful path; failed branches and correction branches are reduced away and not counted in $n$.

### 1.3.3 Discrete transitions and the god-level machine (absolute optimal transition)

State transitions take the discrete ticks $n=0,1,2,\dots$. The state space is not a single trajectory; the whole $X_L$ is packed into $a_n$ and evolves together.

The oracle is an optimal state-transition agent: every step follows exactly the successful path in the state space forward, never turning back and never correcting. It is not a new physical machine, nor extra computational power; it is just the name given to this optimal transition. The ledger is not counted in seconds but in transitions; the worst-case ledger counts transitions, not seconds. One transition can process arbitrarily many assignments and arbitrarily many clauses; the ledger records only how many transitions occurred. Failed, backtracking, correcting branches never enter the ledger at all. Its transition is defined as

$$
\begin{aligned}
&\operatorname{OracleRun}\ f_0\ \text{goal}\ n \iff \exists\ \text{state}:\mathbb N\to X_3\\
&\qquad \text{state}\ 0=f_0\\
&\qquad \text{state}\ n=\text{goal}\\
&\qquad \forall i<n,\ \operatorname{stepSpace}\ (\text{state}\ i)=\operatorname{some}\ (\text{state}\ (i+1))
\end{aligned}
$$

$\operatorname{OracleOptimal}$ further requires it to be the shortest of all successful paths:

$$
\operatorname{OracleOptimal}\ f_0\ \text{goal}\ n
=
\operatorname{OracleRun}\ f_0\ \text{goal}\ n
\wedge
\forall m,\ \operatorname{ReachesIn}\ f_0\ m\ \text{goal}\to n\le m
$$

This $n$ is the absolute optimal transition:

- Upper bound: an $n$-step successful transition chain really exists (the oracle has arrived);
- Lower bound: no successful transition chain is shorter than $n$;
- Branches spent backtracking, correcting, or hesitating are all reduced away and not counted in $n$.

It is the oracle's transition count, not the actual time of any physical machine; a physical machine can only be slower.

### 1.4 Weighted Hamming distance

The space is discrete: $A_L$ is a finite set of $2^L+1$ elements, $D=\{0,1\}^L$ is a finite set of $2^L$ states, and $X_L=A_L^D$ is a finite product. The space has no direction, no line segments, no inner product; Euclidean distance is not its own metric. The metric given by the coordinate structure must be of Hamming type — compare coordinate by coordinate, count the different coordinates:

$$
d(f,g)=|\{x\in D:\ f(x)\ne g(x)\}|
$$

Geometric properties:

- it takes integer values $0,1,\dots,|D|$;
- adjacent means “exactly one coordinate differs”; $d(f,g)$ is the shortest-path length on the finite product graph;
- with unit weights, the ball centered at $f$ with radius $r$ is the set of functions “differing in at most $r$ coordinates”, a union of subcubes.

Weighted Hamming multiplies each coordinate by a weight $w_x\ge0$:

$$
d_w(f,g)=\sum_{x\in D} w_x\cdot\mathbf 1[f(x)\ne g(x)]
$$

- the Hamming type of the metric is determined by discreteness; weights change the price, not the type
- $w_x\equiv1$ $\Rightarrow$ ordinary Hamming distance
- it takes finitely many nonnegative values

3-SAT case ($D=\{0,1\}^n$):

$$
d_w(f,g)=\sum_{x\in\{0,1\}^n} w_x\cdot\mathbf 1[f(x)\ne g(x)]
$$

When $n=3$ the sum runs over 8 states; when $n=2$ it runs over 4 states.

### 1.5 Completeness

- $A_L$ is a value set, $X_L$ is a value-function space, and $!$ acts pointwise.
- Pointwise structure $\Rightarrow$ completeness is inherited coordinate by coordinate: $X_L$ is a product of one copy of $A_L$ over each input coordinate. $X_L$ complete $\Leftrightarrow$ $A_L$ complete.
- Metric side: weighted Hamming aggregates coordinate by coordinate. A finite input set $\Rightarrow$ $X_L$ finite $\Rightarrow$ Cauchy completeness holds automatically; in the infinite case, the completeness of $X_L$ is decided by the distance completeness of $A_L$.
- Expression side: $!$ acts pointwise and creates no new coordinate; it only flips each cell.

In one sentence: the completeness of this value-function space is **inherited** — it is complete when $A_L$ is, and it lacks whatever $A_L$ lacks.

## 2. The Iteration Sequence

### 2.1 The most naive single-variable iteration

The iterated object is not a single value, nor a single variable $X_1,X_2,\dots$; it is one set variable that gathers all states:

- all value sets
- all states
- all ways of assigning all variables

Pack them all into $a_n$ and let it iterate forward along time $n$; each step takes every state in the set, an iteration enumerative by nature. “Single variable” means exactly this one set variable that contains all states.

**Step one: build the sequence.** Fix the starting table $a_0$ and push it forward one step at a time:

$$
a_1=T(a_0),\quad a_2=T(a_1),\quad\dots,\quad a_{n+1}=T(a_n).
$$

So far we only know a sequence exists; we do not know whether it stops. About $T$ we have only one condition:

$$
d_w(T(a_n),T(a_{n+1}))\le q^n\cdot d_w(a_n,a_{n+1}),\qquad 0\le q<1.
$$

It says the distance between two consecutive steps is shrinking. It does not say the distance between any two steps can be made arbitrarily small, so we cannot conclude that the sequence converges. Completeness (§1.5) says a Cauchy sequence has a limit, but we do not yet know this sequence is Cauchy, so it is no help. Banach's fixed-point theorem does not apply directly either: that theorem requires a contraction holding for any two points in the space (Chapter 12 draws that line), while this condition holds only for consecutive steps. Convergence has to be proved separately.

**Step two: every later step is no farther than the first.** Write down the distance between consecutive steps:

$$
\delta_n:=d_w(a_n,a_{n+1})=\sum_{x\in D} w_x\cdot\mathbf 1[a_n(x)\ne a_{n+1}(x)].
$$

Coordinate by coordinate: if a cell changes from step $n$ to step $n+1$, add that cell's weight $w_x$; if not, add nothing. $\delta_n$ is the distance of that step.

Now substitute the iteration condition. Its left side is exactly $\delta_{n+1}$:

$$
\delta_{n+1}=d_w(a_{n+1},a_{n+2})=d_w(T(a_n),T(a_{n+1}))\le q^n\delta_n.
$$

Since $0\le q<1$, we have $q^n\le1$, hence

$$
\delta_{n+1}\le\delta_n.
$$

Adjacent distances never increase. Iterate the inequality term by term:

$$
\delta_k\le\delta_{k-1}\le\cdots\le\delta_0,
$$

that is,

$$
d_w(a_k,a_{k+1})\le d_w(a_0,a_1),\qquad k=0,1,2,\dots
$$

However far the first step went, no later step goes farther. The convergence proof below uses exactly this.

**Step three: distance takes only finitely many values, so it must stop.** $X_L$ is finite, so there are finitely many pairs of points; whatever nonnegative weights are chosen, $d_w(f,g)$ takes only finitely many values. For $n\ge1$, $q^n<1$; as long as $\delta_n>0$, the iteration condition gives

$$
\delta_{n+1}\le q^n\delta_n<\delta_n.
$$

While the sequence moves, the distance strictly decreases. Each strict decrease switches to another value, and there are only finitely many values, so it cannot keep decreasing forever. Hence some $\delta_n$ is $0$.

Consider the set

$$
\{n\in\mathbb N\mid\delta_n=0\}
$$

This set is nonempty. The natural numbers are well ordered, so take the least element and call it $N$. $\delta_N=0$, and from $\delta_{N+1}\le q^N\delta_N=0$ onward by induction, $\delta_n=0$ for $n\ge N$; that is,

$$
a_N=a_{N+1}=a_{N+2}=\cdots
$$

The sequence stops moving after finitely many steps. For any $m,n\ge N$, $d_w(a_m,a_n)=0$, so it is a Cauchy sequence, converging to $a_N$. Only now does completeness enter: a Cauchy sequence in this space has a limit. Here we get more: the limit is $a_N$ itself.

**At most how many steps.** With unit weights, go back to the sum from step two:

$$
\delta_0=\sum_{x\in D}w_x\cdot\mathbf 1[a_0(x)\ne a_1(x)].
$$

Each term is either $1$ or $0$, and there are $|D|$ terms, so $\delta_0$ is an integer from $0$ to $|D|$. Whenever $\delta_n>0$, $\delta_{n+1}<\delta_n$; two distinct integers differ by at least $1$, so each moving step lowers the distance by at least $1$:

$$
\delta_n\le\delta_0-(n-1)\qquad(1\le n\le\delta_0+1).
$$

Take $N=\delta_0+1$. $D$ is the set of all $L$-bit strings, so $|D|=2^L$: eight cells for $L=3$, four for $L=2$, hence

$$
N\le|D|+1=2^L+1,
$$

which is $9$ for $L=3$ and $5$ for $L=2$. $X_3$ has $9^8$ points, yet this sequence stops within nine steps.

The size of $q$ does not affect the step count: as long as $q<1$, each step either does not move or lowers the distance by at least one, so it stops within $|D|+1$ steps. $q$ decides how fast the distance shrinks, not when it stops.

With general nonnegative weights, the distance need not decrease by whole numbers, so the concrete bound $N\le\delta_0+1$ need not hold. But there are still only finitely many values, and strict decrease can still happen only finitely many times, so the sequence still stops after finitely many steps. The 9 and 5 computed later use unit weights.

## 3. Stopping Point and the Trivial Equation

### 3.1 The stopping point

$N$ is the step at which the sequence stops. $\delta_N=0$, and the sequence stops here. The stopping point $a_N$ is a point of the space. Whether it is a resident is not a matter of time; it is a matter of this point.

### 3.2 The Trivial Equation: Discrete, Bounded, Iterated

Looking back, the whole proof uses three things:

- Discreteness: $A_L$ is finite, and distance takes only finitely many values.
- Boundedness: with unit weights, $\delta_n\le\delta_0\le|D|$, so $N\le|D|+1$.
- Iteration: $a_{n+1}=T(a_n)$, one step after another, in order.

From the very first algorithm course, the first lesson is these three things: discrete, bounded, sequentially iterated. The definition of an algorithm has already said it clearly. These three are not a shell put around algorithms; they are the definition itself, one-to-one. Put any algorithm against these three, and its convergence is already inside. A strictly decreasing sequence in a finite set reaches the bottom in a few steps; well-ordering picks out the stopping time; no limit, no $-\infty$.

Why can $L$-SAT reach across the whole algorithm world? Because it is the most ordinary. $L=2$ has two lamps to a row, $L=3$ three lamps to a row; the narrowest clause holds a whole side's discrete, bounded, sequential trajectories. Not because it is special, but because it is so ordinary that any algorithm fits inside it. This chapter only walks $L$-SAT into the same table.

No construction is needed. §4 counts the clauses: $2^L$ cover the table, $2^L+1$ close it; $L=2$ gives 5, $L=3$ gives 9. Both end at constant zero, both UNSAT. The one-cell-at-a-time enumeration is the road the bound in §2.1 describes.

## 4. Collapse

$L$-SAT is a table of $2^L$ cells: $L$ variables, one assignment per cell. These $2^L$ cells are $D=\{0,1\}^L$ from §1 — the index set the weighted Hamming distance adds over cell by cell, $|D|=2^L$. A width-$L$ clause has value $0$ at exactly one cell, $x=s$, and $1$ everywhere else; so one clause touches one cell. $L=2$ is 2-SAT, $L=3$ is 3-SAT; the derivation below holds for every $L$.

### 4.1 Where $2^L+1$ comes from

Set the clauses aside for a moment and look only at the distance. The weighted Hamming distance is

$$
d_w(f,g)=\sum_{x\in D}w_x\cdot\mathbf 1[f(x)\ne g(x)].
$$

Compare the two tables cell by cell: where they differ, record that cell's weight $w_x$; where they agree, record nothing. $D=\{0,1\}^L$ has $|D|=2^L$ cells, so the sum has $2^L$ terms.

At each moving step the iteration either does not move, or makes at least one cell differ. With unit weights the initial distance is $\delta_0\le|D|$, each moving step drops at least one, so at most $|D|$ steps to run through; plus the step at which it stays put,

$$
N\le|D|+1=2^L+1.
$$

$L=2$ gives 5, $L=3$ gives 9. The number is settled on the weighted Hamming side: as many cells as the table has, plus one.

The clause is read off this table, not a second number started alongside. One $L$-SAT clause corresponds to one cell; the clause count runs to the same $|D|+1$.

$L=2$: $4+1=5$; $L=3$: $8+1=9$; $L=4$: $16+1=17$; $L=10$: $1024+1=1025$. A clause carries $L$ literals; $2^L$ clauses use $L\cdot2^L$ literals: 8 at $L=2$, 24 at $L=3$, 10240 at $L=10$. The table has $|D|$ cells; the count is one more. In reverse, $L=\log_2(m^*-1)=\ln(m^*-1)/\ln 2$: give the count and recover the width; give the width and count the clauses.

The information is only so much. The table has $|D|=2^L$ cells, and one clause covers one; within the $2^L$ patterns, the lit and dark cells can still be spelled out one by one; take them all and the table is covered, all zero; past that, no clause adds a new hint, it only punches the space straight to one side.

### 4.2 Covered, Hence UNSAT

One sentence: one clause knocks out one cell. At the start the whole table is $1$, every cell SAT; each clause added knocks at most one cell from SAT to UNSAT. The $2^L$ clauses knock out at most $2^L$ cells; once all $2^L$ patterns have been taken, any further clause only repeats, hitting no new cell. So from all-SAT to all-UNSAT, at most $2^L+1$ clauses complete the transition. $L=2$: 5; $L=3$: 9.

The difference between 5 and 9 is not SAT versus UNSAT; it is the size of the table: 4 cells need 5, 8 cells need 9. This number is not a patent of $L$-SAT. The weighted Hamming distance has $|D|$ cells, and the stopping bound is $|D|+1$; the clauses merely knock out the same $|D|$ cells one by one. Clause and assignment alike take the certificate the weighted Hamming distance issues — $2^L+1$. For any algorithm whose table has $|D|$ cells and whose distance moves at least one cell per step, the stopping bound is still $|D|+1$.

There is no probability here. Take fewer than $2^L$ patterns and some cell is left uncovered; that cell is SAT. Take all $2^L$ and the table is covered; covered is UNSAT. It is not that more clauses tip it one way; the count reaches $2^L$ and the cover is complete, reaches $2^L+1$ and there is no new cell.

The bound says nothing about time. Time is not guaranteed: how long finding SAT takes, how long confirming UNSAT takes — nobody promises. Nobody says SAT is certain within $2^L$, and nobody says it is impossible; $2^L+1$ marks where the information ends, not where the clock ends.

### 4.3 Not a distance

This iteration does not do two things: it does not search, and it does not compute a distance between P and NP. It does one thing: press the formula table onto its fixed point and see whether the fixed point is lit.

The difficult part is not the iteration; it is writing the table that stops — knowing whether it will go all dark. Given an assignment, substitute it into the formula and the light or dark shows in one step; the hard part is finding that assignment. Checking is one step; finding is not.

So "convergence is not the difficulty" is not a sentence of defeat. It moves the question: there is no boundary on the time axis; the boundary is at the fixed point. A lit fixed point is SAT; an all-dark one is UNSAT. What is wanted is not a faster iteration; it is the fixed point that is still lit.

## 5. Not P, Not NP — a Convergence Problem

### 5.1 The solution space moves on its own

Someone keeps asking: why don't you construct a Turing machine? Why not start from a two-state Turing machine and push up step by step?

No need. The Turing-machine layer is too shallow to reach the question. Add states, add tape, add parallelism — what is added is how much the machine can spit out at once; the question here is whether the resident is in the solution space. If the resident is absent, no machine, however godlike, can search it out; if it is present, a stupid machine walking slowly will get there. Solvability is not set by the searching machine; it is set by the space itself.

Drive the machine to its most godlike: process several hundred million clauses at once, lock onto the optimal one, follow the solution space. However godlike the machine, it can only use the information already in the clauses. The information bound is $2^L+1$ clauses: 5 at $L=2$, 9 at $L=3$. Past that, adding constraints yields no hint, only pushes the space to one side. Fast or slow, with more clauses the solution space lands at constant zero.

This is also the fee the algorithm charges: you can design an optimal algorithm and make every step hug the solution space; but the clauses hold only so much information, and when it runs out it is gone — the fee cannot be avoided. The $2^L+1$ is an information bound, not a stopwatch: one more clause, one more literal, adds no information; more constraints only push the space down to the UNSAT side — every $L$-SAT alike.

If the resident is there, $2^L+1$ clauses are enough to say the space through; if it is not, no machine can conjure it. Time is not guaranteed: how long finding SAT takes, how long confirming UNSAT takes — nobody promises. The direction is fixed: once the information is used up, the space tilts one way only. It did not lose on computing power; it reached the end of the clauses' information.

So there is no need to construct the Turing machine of a hundred years from now, nor to push up from a two-state one. A carefully designed parallel or backtracking algorithm, any other machine — all walk on the same table. If the resident is absent, the table contracts to constant zero, and the SAT lamp cannot light. This is not a question of computing power; it is a question of the solution space: whether that satisfying assignment is there, whether that resident is there, is a mathematical question, not a computer question.

Someone starts from properties of two-state Turing machines and proves that a certain class of machines cannot search. That is another road: it proves something about the machine, not the space. Machines can be swapped; the space does not swap. Bring a stronger machine, and the shape of the space is the same. What is to be proved is that the resident is not in the space, not that some machine is not fast enough.

### 5.2 Conclusion

This chapter teaches one triviality equation of algorithmic convergence: discrete, bounded, sequentially iterated. With these three, any algorithm is collected in the same equation. The difference is at the stopping point — a lit lamp is SAT, a dark one UNSAT. The $2^L+1$ is the information bound on the clause count: 5 at $L=2$, 9 at $L=3$; count to here and both collapse to constant zero.

Only two samples are laid out here: one P, one NP. They are picked out not to split them apart, but so you can see that both sides share one structure. Every P problem and every NP problem sends its solution space toward a fixed point under the same equation; P and NP are just the readings this equation gives on two samples.

The general form of the trivial equation. Let $X$ be finite, $T:X\to X$, $a_0\in X$, $a_{n+1}=T(a_n)$, with $\delta_n=d(a_n,a_{n+1})$ integer-valued, and take $0\le q<1$. If

$$
\delta_{n+1}\le q^n\delta_n\quad(n\ge0),
$$

then

$$
\exists N\le\delta_0+1:\quad \delta_N=0.
$$

$\delta_N=0$ means $a_N=a_{N+1}$; it does not move again.

Every algorithm can be written as $a_{n+1}=T(a_n)$; once its distances satisfy this contraction inequality, it reaches a stopping point in finitely many steps. Triviality is not an adjective; it is proved — by the collapse of the solution space. That is the one new thing in this chapter, and this single move unifies the Reasoning Kingdom.

This is not a question of whether the machine is fast enough; it is a question of whether that resident is in the space. Look further along the time axis and you will find nothing new.

There really are problems in life with no answer. Not every problem hides a solution waiting for a smarter machine to dig it out. Some problems narrow as you go deeper, until there is nothing left. This is ordinary. Some problems simply have no answer.

In the end, this chapter is not refuting the algorithm; it is defending the algorithm itself. The classical definition of an algorithm has only three conditions: discrete, bounded, sequentially iterated. It is not that some machine is not fast enough; it is the boundary these three commandments draw. Neural networks, Monte Carlo trees, every kind of heuristic: their convergence is collected in these three. An algorithm is convergent by its own nature; the three commandments are where that convergence is written. Nothing here derives an algorithm; the derivation has always been of these three rules themselves.

The three commandments are the algorithm's limits and open questions: inside the triangle they enclose, convergence is necessary; outside it, neither convergence nor determinism can even be spoken of.

---

The Reasoning Kingdom ends here.

It is an open book. What is proved is laid here, and what is not proved is laid here too. Chapter 24 puts the convergence of the whole Reasoning Kingdom into one equation: the table reaches a fixed point, and the lamp is either lit or out.

Walking toward tomorrow does not require first becoming a person who can be written down, complete, normal, brave, without fear or shame. No one has that obligation.

Tomorrow will not open its front door for us. Then wade across. Feel the world as before, imagine as before, live as before.

Once you have waded across, you are across.

Good luck.

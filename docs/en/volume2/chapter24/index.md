# Chapter 24: Algorithmic Convergence — The LIT Theorem

---

$A_L$ set; $A_L$ resident: $1_L$

$X_L=A_L^D$; $A_L=\{0,1\}^L\cup\{!\}$

$(X_L,d_w)$ $\Rightarrow$ space

Route: the triviality equation of algorithmic convergence — discrete, bounded, sequentially iterated $\to$ the value set keeps only $A_L=\{0,1\}^L$ and the negation $!$ $\to$ weighted Hamming space $X_L=A_L^D$ $\to$ construct the iteration sequence $\to$ division, change of base, proper fraction $\to$ well-ordering gives the stopping time, no limit taken $\to$ the formula map $T_S$ freezes in one step $\to$ a lit stopping point is SAT, a dark one UNSAT $\to$ both ideal times are $1$, the difference is in the stopping point $\to$ the solution space collapses; P and NP are just two samples.

## Preface: Enumerate First, Then Talk About Convergence

A clause, three variables. However the literals are arranged, whatever values are plugged in, whatever operators are inserted, whatever conditions are added, in the end it all reads as the same thing: under this assignment, is the clause satisfied? Beyond that, it has no other behavior.

So we do not guess its structure; we list its behavior. The three variables each take $0$ or $1$: eight assignments. One assignment, one answer; eight answers make a complete table. Enumerate this table from beginning to end, missing nothing.

The eight answers do not run separately. Stack them together, collect them into one variable $a_n$, and push forward along ideal time $n=0,1,2,\dots$. Each step updates the whole stack of answers at once. Ideal time does not backtrack, does not correct mistakes; it walks only the successful path, and the stack is laid down layer by layer.

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

The book's iteration starts from the constant-one table $1_{X_L}$ and updates the whole table under the rule; each step produces a new table. Time moves only along a table that still has a lit cell; when the new table goes fully dark, the clock stops. Ideal time counts successful steps, not lit cells.

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

A formula is a conjunction of clauses; its sign-pattern set is $S\subseteq D$. Its truth table is

$$
F_S(x)=\bigwedge_{s\in S}f_s(x)=
\begin{cases}
1_L, & x\notin S,\\
0_L, & x\in S.
\end{cases}
$$

Satisfiable $\Leftrightarrow$ $F_S$ is not constant zero $\Leftrightarrow$ $S\ne D$. A resident is a nonzero table; the constant-zero table is not a resident.

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

### 1.3.2 Optimal time: only successful advances count

Backtracking, correction, and hesitant branches do not enter the time ledger. Only steps that truly move forward successfully are recorded:

$$
\operatorname{ReachesIn}\ f_0\ 0\ f_0
$$

$$
\operatorname{ReachesIn}\ f_0\ (n+1)\ g
\qquad
\text{if}\ \operatorname{ReachesIn}\ f_0\ n\ f\ \text{and}\ \operatorname{stepSpace}\ f=\operatorname{some}\ g
$$

$\operatorname{IsOptimalTime}\ f_0\ \text{goal}\ n$ takes the number of steps on the shortest successful path; failed branches and correction time are reduced away and not counted in $n$.

### 1.3.3 Discrete time and the god-level machine (absolute optimal time)

Time takes the discrete ticks $n=0,1,2,\dots$. The state space is not a single trajectory; the whole $X_L$ is packed into $a_n$ and evolves together.

Oracle time is a name: it refers to the most ideal time, with no correction at all, evolving entirely along the successful path of the state space. It is not a new physical machine, nor extra computational power; it is just the name given to this most ideal time. Imagine an oracle as this ideal evolver: every step follows exactly the successful path in the state space, and failed, backtracking, correcting branches never enter the time ledger at all. Its time is defined as

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

This $n$ is the absolute optimal time:

- Upper bound: an $n$-step successful path really exists (the oracle has arrived);
- Lower bound: no successful path is shorter than $n$;
- Time spent backtracking, correcting, or hesitating is all reduced away and not counted in $n$.

It is the oracle's time, not the actual time of any physical machine; a physical machine can only be slower.

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

Construct the iteration sequence on the complete space $(X,d)$:

$$
a_{n+1}=T(a_n)
$$

Iteration condition:

$$
d(T(a_n),T(a_{n+1}))\le q^n\cdot d(a_n,a_{n+1})
$$

Prove that it really is a Cauchy sequence. The proof uses one fact only: distances are integers, and take only finitely many possible values.

Let $\delta_n:=d(a_n,a_{n+1})$. The iteration condition is

$$
\delta_{n+1}=d(a_{n+1},a_{n+2})\le q^n\,d(a_n,a_{n+1})=q^n\delta_n
$$

Since $0\le q<1$, we have $q^n\le1$, hence

$$
\delta_{n+1}\le\delta_n
$$

The distance between adjacent terms never increases.

First look at the first rung. $\delta_0=d(a_0,a_1)$ is Hamming distance: it counts how many coordinates of $a_0$ and $a_1$ differ. The space has $|D|$ coordinates in all, so

$$
d(a_n,a_{n+1})=\delta_n\le\delta_0\le|D|
$$

$L=3$: $|D|=8$, $\delta_0$ is an integer from $0$ to $8$. $L=2$: $|D|=4$, $\delta_0$ is an integer from $0$ to $4$.

Tighten one notch. For $n\ge1$, $q^n<1$, so as long as $\delta_n>0$, we have $\delta_{n+1}<\delta_n$ — at least one notch drops each time, no ties. Integers only decrease, and after $\delta_0$ notches they hit the floor:

$$
\delta_n\le\delta_0-(n-1)\qquad(1\le n\le\delta_0+1)
$$

Take $N=\delta_0+1$: for $L=3$, $N\le9$; for $L=2$, $N\le5$. Hence $\delta_N=0$, and from this term on

$$
a_N=a_{N+1}=a_{N+2}=\dots
$$

The sequence no longer moves; it is constant. A constant sequence is Cauchy, and it sits at this point; for all $m,n\ge N$, $d(a_m,a_n)=0$.

Check with 9: $A_3$ has $8+1=9$ values, $X_3$ has $9^8$ points, yet this path is finished within ten steps.

Weighted Hamming is the same ledger, only each notch has a different price: $d_w(a_n,a_{n+1})\le d_w(a_0,a_1)$, the distance falls in finitely many values, drops notch by notch, and reaches the bottom in finitely many steps.

### 2.2 The division form of the iteration condition

Let

$$
\Delta_n:=d(T(a_n),T(a_{n+1}))
$$

$$
\delta_n:=d(a_n,a_{n+1})
$$

Divide the multiplier on the right side of the iteration condition over to the left:

$$
\frac{\Delta_n}{\delta_n}\le q^n
$$

- left side: image distance $\Delta_n$ divided by preimage distance $\delta_n$
- right side: exponential function form
- $0\le q<1$

Why divide the multiplier over to the left? $\Delta_n$ and $\delta_n$ are both distances, of the same dimension; their quotient is a pure number, measuring how many times $T$ squeezed the distance at this step. $q^n$ is also a pure number. Comparing two pure numbers does not depend on what unit distances are measured in. Written as the product $\Delta_n\le q^n\delta_n$ it means the same thing, but the multiplication hides what the multiple means; division puts it in plain sight.

Division also writes the hypothesis into the denominator: only when $\delta_n>0$ and the sequence is still moving forward is $\Delta_n/\delta_n$ a number. Once the sequence freezes, $a_n=a_{n+1}$, the numerator $\Delta_n=d(a_{n+1},a_{n+2})\le q^n\cdot0=0$ is also $0$, and the ratio degenerates to $0/0$ — not large, not small, but undefined. So the division form covers only the stretch where the sequence is still moving. This will reappear unchanged in log space.

## 3. Log Space

### 3.1 A proper fraction

Start from the division form of §2.2:

$$
\frac{\Delta_n}{\delta_n}\le q^n,\qquad 0\le q<1.
$$

The base on the right is below $1$, so a larger exponent gives a smaller value. For $n\ge1$, $q^n\le q<1$, hence

$$
\frac{\Delta_n}{\delta_n}<1,\qquad \Delta_n<\delta_n.
$$

The numerator is smaller than the denominator: a proper fraction. It says one thing only — while the sequence is still moving, this step's image distance is strictly shorter than the previous one. No logarithms, no change of base; write that down first.

For $q=0$ it is even more direct: the right side is $0$, the proper fraction forces $\Delta_n=0$, and the sequence stops in one step. Below, assume $0<q<1$.

### 3.2 Change of base

The proper fraction says it shrinks, not how fast. Take a logarithm and the speed shows.

Change of base:

$$
\log_b x=\frac{\log_c x}{\log_c b}
$$

With $c=e$,

$$
\log_b x=\frac{\ln x}{\ln b}.
$$

The base changes, the value does not; when the base is below $1$, the logarithm reverses order.

Take $b=q$ and take the logarithm of the proper fraction. While the sequence moves the numerator is positive; $\Delta_n=0$ means the next step stops, so there is no need to look at a logarithm. Hence

$$
\log_q\frac{\Delta_n}{\delta_n}
=\frac{\ln(\Delta_n/\delta_n)}{\ln q}
\ge n.
$$

The inequality flips because $\ln q<0$. The change of base does one thing: it pulls $q^n$ down to $n$. The exponent comes off the shoulder and becomes one step on the ground.

The natural-log form says the same:

$$
\ln\Delta_n-\ln\delta_n\le n\cdot\ln q.
$$

The right side is linear in $n$, with slope $\ln q<0$. Each step translates the log coordinate down by $|\ln q|$; $n$ steps, down by $n|\ln q|$. Products become sums through $\ln(xy)=\ln x+\ln y$ alone. The exponential curve was bent; in log coordinates the bend is gone, and only this line is left.

### 3.2.1 The floor

The line keeps going down, but there is no need to let it run to infinity.

Distance takes only finitely many values: unweighted, $\delta_n$ is an integer from $0$ to $|D|$; weighted, one of finitely many nonnegative values. While the sequence moves, $\delta_{n+1}<\delta_n$. A strictly decreasing sequence in a finite set has only a few steps to take. At most $|D|$ steps, and it lands on $0$.

Now use well-ordering. The set

$$
\{n\in\mathbb N\mid\delta_n=0\}
$$

is nonempty; $\mathbb N$ is well ordered, so it has a least element, call it $N$. $N$ is the step at which it lands on the floor, a natural number.

What it drops to can also be written down: not an invisible limit, but $\delta_N=0$, and $a_N=a_{N+1}=\cdots$. The floor is the stopping point $a_N$ itself.

### 3.3 The least stopping time

$$
N\le\delta_0+1.
$$

For $L=3$, $N\le9$; for $L=2$, $N\le5$. At $N$, $a_N=a_{N+1}=\dots$, and the sequence freezes.

### 3.4 The stopping point

$N$ is the step at which the sequence stops. $\delta_N=0$, and the division form $\Delta_N/\delta_N$ becomes $0/0$, undefined; the ledger ends here. The stopping point $a_N$ is a point of the space. Whether it is a resident is not a matter of time; it is a matter of this point.

### 3.5 The Trivial Equation: Discrete, Bounded, Iterated

The ledger is closed. Looking back, the whole path uses three things:

- Discreteness: $A_L$ is finite, and distance takes only finitely many values.
- Boundedness: $\delta_n\le\delta_0\le|D|$; at most $|D|$ steps to the floor.
- Iteration: $a_{n+1}=T(a_n)$, one step after another, in order.

From the very first algorithm course, the first lesson is these three things: discrete, bounded, sequentially iterated. The definition of an algorithm has already said it clearly. These three are not a shell put around algorithms; they are the definition itself, one-to-one. Put any algorithm against these three, and its convergence is already inside. A strictly decreasing sequence in a finite set reaches the bottom in a few steps; well-ordering picks out the stopping time; no limit, no $-\infty$.

Why can two small problems reach across the whole algorithm world? Because they are the most ordinary. 3-SAT has three lamps to a row, 2-SAT two lamps to a row; the narrowest clause holds a whole side's discrete, bounded, sequential trajectories. Not because they are special, but because they are so ordinary that any algorithm fits inside them. This chapter only walks both sides into the same table.

No construction is needed. §5 lays $D$ on the table, takes the union of the sign patterns killed by the clauses as $S$, substitutes into $T_S$, and expands: $a_1=F_S$, $a_2=F_S$; the least stopping time is $1$ for both 2-SAT and 3-SAT.

## 4. Collapse

Now put the formula into this iteration. The sign-pattern set of the formula is $S\subseteq D$; define

$$
T_S(f)(x)=
\begin{cases}
f(x), & x\notin S,\\
0, & x\in S.
\end{cases}
$$

Cells killed by the formula are zeroed; the rest stay. There is no $\&$ and no $|$ here; only the sign pattern $S$, the zeroing, and the existing values.

Start from the constant-one table $a_0=1_{X_L}$. Compute two steps:

$$
a_1=T_S(a_0)=F_S,
$$

$$
a_2=T_S(a_1)=F_S.
$$

$T_S$ is idempotent and stops at the second step. Take $Q=0$, $N=1$. $T_S\circ T_S=T_S$ is what the trivial equation looks like on one formula: it does not care what $S$ looks like, what $L$ is, or how wide the clauses are; substitute and it holds, and the moment it holds it stops.

That is the collapse: the whole candidate table is pressed onto the formula's truth table in one step. Cells in $S$ go dark; cells outside $S$ stay lit. Whether the fixed point $F_S$ is a resident depends on one thing:

- $F_S=0$ $\Leftrightarrow$ $S=D$ $\Leftrightarrow$ every assignment is killed $\Leftrightarrow$ unsatisfiable;
- $F_S\ne0$ $\Leftrightarrow$ some assignment is not killed $\Leftrightarrow$ satisfiable.

Constant zero is not a resident. If the table stops in the dark, there is no SAT; if it is still lit, SAT sits in one of those cells. The iteration does not search and does not try and error; it projects the formula onto the table, all at once.

## 5. 2-SAT and 3-SAT: Substitute and Compute

Now substitute 2-SAT and 3-SAT directly. No need to invent problems: $D$ is the assignment table, $S$ is the union of the cells killed by the clauses, and $T_S$ only zeroes them. Substitute, expand, count the steps.

### 5.1 2-SAT: $D=\{0,1\}^2$

Take the 2-SAT formula

$$
F=(x_1\vee x_2)\wedge(x_1\vee\neg x_2),
$$

whose truth table is

$$
h(x)=(x_1,x_1).
$$

Both clauses are false only when $x_1=0$, so $S=\{00,01\}$. Let $D=\{00,01,10,11\}$. From $a_0=1_X$,

$$
a_1=T_S(a_0)=F_S=h,
$$

$$
a_2=T_S(a_1)=h.
$$

Distances:

$$
\delta_0=d(1_X,h)=|\{00,01\}|=2,
$$

$$
\delta_1=d(h,h)=0.
$$

One step to the bottom, $Q=0$, $N=1$. $h$ is lit on $10$ and $11$, so it is a resident; those two assignments satisfy the formula. The $2$ in $L=2$ is the number of variables; the $2$ in standard 2-SAT is the clause width. Same numeral, different meaning.

In general, 2-SAT is the same two substitutions. For $n$ variables, $D=\{0,1\}^n$; a width-2 clause kills $2^{n-2}$ assignments, and $S$ is the union of those cells. $T_S$ zeroes and only zeroes, hence is idempotent; $a_1=F_S$, $a_2=F_S$, $N=1$.

### 5.2 3-SAT: $D=\{0,1\}^3$

Take

$$
F=(x\vee y\vee z)\wedge(\neg x\vee\neg y\vee\neg z).
$$

Its sign-pattern set is $S=\{000,111\}$: $F$ is false only at these two cells and true on the other six. $F_S$ arrives in one step and is a resident; $Q=0$, $N=1$.

Now take $S=D$, killing all eight cells. $F_S=0$, again one step; the fixed point is constant zero, not a resident, UNSAT. $N$ is still $1$.

In general, 3-SAT is the same. For $n$ variables, $D=\{0,1\}^n$; a width-3 clause kills $2^{n-3}$ assignments, and after taking the union into $S$, the same $T_S$ gives $a_1=F_S$, $a_2=F_S$, $N=1$.

The clause width goes from 2 to 3; it changes how many cells each clause kills, not the number of steps. The least stopping time for $L=2$ and $L=3$ is $1$; time does not tell them apart, the fixed point does. Let the number of variables go from 2 to 3 to three hundred million: $T_S$ is still idempotent, and one formula still freezes in one step. How big the table is only changes how long $F_S$ takes to write down, not the number of steps to the fixed point.

### 5.3 Not a distance

This iteration does not do two things: it does not search, and it does not compute a distance between P and NP. It does one thing: press the formula table onto its fixed point and see whether the fixed point is lit.

The difficult part is not the iteration. The iteration is one step. The difficult part is writing $F_S$ — knowing, before the formula is unpacked, which cell is killed and which is still lit. Given an assignment, substitute it into the formula and the light or dark shows in one step; the hard part is finding that assignment. Checking is one step; finding is not.

So "both times are 1" is not a sentence of defeat. It moves the question: there is no boundary on the time axis; the boundary is at the fixed point. A lit fixed point is SAT; an all-dark one is UNSAT. What is wanted is not a faster iteration; it is the fixed point that is still lit.

## 6. Not P, Not NP — a Convergence Problem

### 6.1 The solution space moves on its own

Someone keeps asking: why don't you construct a Turing machine? Why not start from a two-state Turing machine and push up step by step?

No need. The Turing-machine layer is too shallow to reach the question. Add states, add tape, add parallelism — what is added is how much the machine can spit out at once; the question here is whether the resident is in the solution space. If the resident is absent, no machine, however godlike, can search it out; if it is present, a stupid machine walking slowly will get there. Solvability is not set by the searching machine; it is set by the space itself.

Drive the machine to its most godlike: spit out several hundred million states at once, lock onto the optimal one, follow it; every step follows the width of a three-literal clause, every step ideal. This is a monster machine. At the end the monster machine still does not find the resident. It did not lose on computing power; it arrived at constant zero, the table stopped in the dark, and the SAT lamp never lit. The evolution of the solution space lands on a fixed point; the monster machine at most accelerates its own death.

So there is no need to construct the Turing machine of a hundred years from now, nor to push up from a two-state one. A carefully designed parallel or backtracking algorithm, any other machine — all walk on the same table. The table contracts, contracts to constant zero, and the SAT lamp cannot light. This is not a question of computing power; it is a question of the solution space: whether that satisfying assignment is there, whether that resident is there, is a mathematical question, not a computer question.

Someone starts from properties of two-state Turing machines and proves that a certain class of machines cannot search. That is another road: it proves something about the machine, not the space. Machines can be swapped; the space does not swap. Bring a stronger machine, and the shape of the space is the same. What is to be proved is that the resident is not in the space, not that some machine is not fast enough.

### 6.2 Conclusion

This chapter teaches one triviality equation of algorithmic convergence: discrete, bounded, sequentially iterated. With these three, any algorithm is collected in the same equation. $T_S\circ T_S=T_S$; both ideal times are $1$, and the difference is in the cell where they stop — a lit lamp is SAT, a dark one UNSAT.

Only two samples are laid out here: one P, one NP. They are picked out not to split them apart, but so you can see that both sides share one structure. Every P problem and every NP problem sends its solution space toward a fixed point under the same equation; P and NP are just the readings this equation gives on two samples.

The general form of the trivial equation. Let $\delta_0=d(a_0,a_1)$:

$$
\begin{aligned}
&\forall\,(X,d,q,T,a_0),\quad X\ \text{finite},\quad 0\le q<1,\\
&\qquad a_{n+1}=T(a_n),\quad d(Ta_n,Ta_{n+1})\le q^n d(a_n,a_{n+1})\quad(\forall n\ge1)\\
&\Longrightarrow\quad \exists N\le\delta_0+1:\quad a_N=T(a_N)=a_{N+1}=\cdots
\end{aligned}
$$

The formula map is its cleanest instance: $T_S\circ T_S=T_S$, $N=1$. Every algorithm can be written as $a_{n+1}=T(a_n)$; once its distance ledger satisfies this contraction inequality, it reaches a stopping point in finitely many steps. Triviality is not an adjective; it is proved — by the collapse of the solution space. That is the one new thing in this chapter, and this single move unifies the Reasoning Kingdom.

This is not a question of whether the machine is fast enough; it is a question of whether that resident is in the space. Look further along the time axis and you will find nothing new.

There really are problems in life with no answer. Not every problem hides a solution waiting for a smarter machine to dig it out. Some problems narrow as you go deeper, until there is nothing left. This is ordinary. Some problems simply have no answer.

In the end, this chapter is not refuting the algorithm; it is defending the algorithm itself. The classical definition of an algorithm has only three conditions: discrete, bounded, sequentially iterated. It is not that some machine is not fast enough; it is the boundary these three commandments draw. Neural networks, Monte Carlo trees, every kind of heuristic: their convergence is collected in these three. An algorithm is convergent by its own nature; the three commandments are where that convergence is written. Nothing here derives an algorithm; the derivation has always been of these three rules themselves.

The three commandments are the algorithm's limits and open questions: inside the triangle they enclose, convergence is necessary; outside it, neither convergence nor determinism can even be spoken of.

---

The Reasoning Kingdom ends here.

It is an open book. What is proved is laid here, and what is not proved is laid here too. The conclusion of Chapter 24 stops at the collapse of the solution space: the table reaches a fixed point, and the lamp is either lit or out.

Walking toward tomorrow does not require first becoming a person who can be written down, complete, normal, brave, without fear or shame. No one has that obligation.

Tomorrow will not open its front door for us. Then wade across. Feel the world as before, imagine as before, live as before.

Once you have waded across, you are across.

Good luck.

# Chapter 24: An Inquiry into the NP Problem — The LIT Theorem

---

$A_L$ set; $A_L$ resident: $1_L$

$X_L=(A_L,\&,|,!)$ semiring

$(X_L,d)$ $\Rightarrow$ space

Route: define the space and the distance $\to$ construct the iteration sequence $\to$ division $\to$ change of base into log space $\to$ take $n\to\infty$ $\to$ use “a discrete space contains no $-\infty$” to derive a contradiction $\to$ prove the discreteness lemma $\to$ construct the P resident explicitly $\to$ prove by contradiction that no NP resident exists $\to$ return to ideal time and give the upper bound.

## Preface: Enumerate First, Then Talk About Convergence

A clause, three variables. However the literals are arranged, whatever values are plugged in, whatever operators are inserted, whatever conditions are added, in the end it all reads as the same thing: under this assignment, is the clause satisfied? Beyond that, it has no other behavior.

So we do not guess its structure; we list its behavior. The three variables each take $0$ or $1$: eight assignments. One assignment, one answer; eight answers make a complete table. Enumerate this table from beginning to end, missing nothing.

The eight answers do not run separately. Stack them together, collect them into one variable $a_n$, and push forward along ideal time $n=0,1,2,\dots$. Each step updates the whole stack of answers at once. Ideal time does not backtrack, does not correct mistakes; it walks only the successful path, and the stack is laid down layer by layer.

After this step, convergence becomes a computable question. $a_n$ lies in the space $X_L$, and the distance between two points is Hamming distance — count how many values differ. How much distance $T$ squeezes out per step is computed directly (§2).

The method is not mysterious; it is even stupid: not understanding the structure of the clause, so try every assignment. But to speak of a naive fixed point, one has to use this stupid method. A slightly cleverer method assumes structure from the start, and then naivety has nowhere to stand. The stupid method fills the whole space; only once it is full can one discuss which cell the fixed point lands in.

## 1. The Space

### 1.1 The value set $A_L$ (the trivial foundation)

For each $L$, define the trivial value set

$$
A_L:=\{0,1\}^L\cup\{\&,|,!\}
$$

$|A_L|=2^L+3$. $A_L$ does only two things: it holds $L$-bit states, and it holds three pointwise operators. The trivial resident takes the constant-one state

$$
1_L:=(1,\dots,1)\in\{0,1\}^L\subseteq A_L
$$

which on the space is the constant-one value function $1_{X_L}$.

### 1.2 Constructing $A_3$ and the complete space

Take $L=3$: $A_3:=A_L|_{L=3}$, made of 8 states and 3 operators, 11 in all:

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
| 8 | $\&$ |
| 9 | $\vert$ |
| 10 | $!$ |

$X_3$ is the value-function space: an element $f:D\to A_3$ is a value function, and $\&$, $\vert$, $!$ act on it pointwise.

For the 3-SAT case take $D=\{0,1\}^n$. A value function is just the evaluation map:

$$
\Phi:\{0,1\}^n\to A_3
$$

When $n=3$, $D$ has 8 states; $|A_3|=11$, so $X_3=A_3^D$ has $11^8$ value functions. Satisfiable $\Leftrightarrow$ $\Phi$ is not the constant-zero function — in the space this is the subset “nonzero functions”, shaped like a cone / hyperplane, with no clause syntax inside.

$X_3=A_3^D$ is a complete space: see 1.4 for the distance, 1.5 for completeness.

### 1.3 Constructing $A_2$ and the complete space

Take $L=2$: $A_2:=A_L|_{L=2}$, made of 4 states and 3 operators, 7 in all:

| Number | State |
|-----:|:-----|
| 0 | $(0,0)$ |
| 1 | $(0,1)$ |
| 2 | $(1,0)$ |
| 3 | $(1,1)$ |
| 4 | $\&$ |
| 5 | $\vert$ |
| 6 | $!$ |

For the 2-SAT case take $D=\{0,1\}^2$; $X_2:=A_2^D$ is the complete space: see 1.4 for the distance, 1.5 for completeness.

$L=2$: 2-SAT, $\in\mathrm P$; $L=3$: 3-SAT, $\in\mathrm{NP}$ (Cook–Levin).

### 1.3.1 Not filling in a 3-bit/2-bit table, but iterating the whole solution space along the time axis

$A_2$, $A_3$ do not mean “the input has only 2 or 3 bits”. They are pointwise enumeration value sets: $A_L$ enumerates every state of $\{0,1\}^L$ into itself, and $X_L=A_L^D$ packs all candidate value functions into one space.

- $D=\{0,1\}^3$ enumerates 8 candidate input states; $X_3=A_3^D$ is the entire candidate solution space.
- The iterated object $a_n\in X_3$ is not a single 3-bit assignment but the whole candidate table: the values at the 8 states are packed into $a_n$ at once and evolve together along the time mark $n$.
- The time gate counts only “exactly 3 satisfied states”: when $\operatorname{satStates}(a_n)=3$, the whole space is allowed to advance to $a_{n+1}$; otherwise time stops and cannot push outward.

So “3” is the exactly-3 satisfied states on the current time slice, not the input length; “2” likewise: there are at most 2 satisfied states, a third cannot be conjured, and the 3-time gate is forever closed.

### 1.3.2 Optimal time: only successful advances count

Backtracking, correction, and hesitant branches do not enter the time ledger. Only steps that truly pass the gate and move forward successfully are recorded:

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

The space is discrete: $A_L$ is a finite set of $2^L+3$ elements, $D=\{0,1\}^L$ is a finite set of $2^L$ states, and $X_L=A_L^D$ is a finite product. The space has no direction, no line segments, no inner product; Euclidean distance is not its own metric. The metric given by the coordinate structure must be of Hamming type — compare coordinate by coordinate, count the different coordinates:

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

- the Hamming type of the metric is determined by discreteness; weights do not change the type
- $w_x\equiv1$ $\Rightarrow$ ordinary Hamming distance
- general value sets: $d_w(f,g)=\sum_{x\in D} w_x\cdot\delta(f(x),g(x))$, with $\delta$ the discrete metric on $A_3$

3-SAT case ($D=\{0,1\}^n$):

$$
d_w(f,g)=\sum_{x\in\{0,1\}^n} w_x\cdot\mathbf 1[f(x)\ne g(x)]
$$

When $n=3$ the sum runs over 8 states; when $n=2$ it runs over 4 states.

Norm form ($A_3=\{0,1\}$):

$$
\|f\|_w=\sum_{x\in D} w_x\cdot f(x)
$$

$$
d_w(f,g)=\|f\oplus g\|_w
$$

### 1.5 Completeness

- $A_L$ is a value set, $X_L$ is a value-function space, and $\&$, $\vert$, $!$ act pointwise.
- Pointwise structure $\Rightarrow$ completeness is inherited coordinate by coordinate: $X_L$ is a product of one copy of $A_L$ over each input coordinate. $X_L$ is a semiring / lattice / complemented $\Leftrightarrow$ $A_L$ is; $X_L$ complete $\Leftrightarrow$ $A_L$ complete.
- Metric side: weighted Hamming aggregates coordinate by coordinate. A finite input set $\Rightarrow$ $X_L$ finite $\Rightarrow$ Cauchy completeness holds automatically; in the infinite case, the completeness of $X_L$ is decided by the distance completeness of $A_L$.
- Expression side: $\{\&,\vert,!\}$ pointwise generate the function algebra on $A_L$; that they generate all value functions rests on $A_L$ itself being a Boolean value set.

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

The sequence no longer moves; it is constant. A constant sequence is Cauchy, and its limit is this point; for all $m,n\ge N$, $d(a_m,a_n)=0$.

Check with 11: $A_3$ has $8+3=11$ values, $X_3$ has $11^8$ points, yet this path is finished within ten steps.

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

### 3.1 The change-of-base formula

Change of base:

$$
\log_b x=\frac{\log_c x}{\log_c b}
$$

Replace the argument with the numerator $\Delta_n$ and the base with the denominator $\delta_n$:

$$
\log_{\delta_n}\Delta_n=\frac{\log_c\Delta_n}{\log_c\delta_n}
$$

Move the division inequality into log space:

$$
\begin{aligned}
&\ln\left(\frac{\Delta_n}{\delta_n}\right)\le n\cdot\ln q\\
\iff\ &\ln\Delta_n-\ln\delta_n\le n\cdot\ln q
\end{aligned}
$$

- the exponential form $q^n$ becomes the linear term $n\cdot\ln q$ in log space
- the division expression is isomorphic to the change-of-base formula: numerator $\Delta_n$ $\leftrightarrow$ argument, denominator $\delta_n$ $\leftrightarrow$ base

The change-of-base formula is itself a division. The numerator is the argument, the denominator the base; the ratio in §2.2 and the quotient on the log scale are the same thing, and written as a subtraction it is $\ln\Delta_n-\ln\delta_n$.

Turning a product into a sum relies on one property only: $\ln(xy)=\ln x+\ln y$. So $q^n$ in log coordinates is not “looks like” a straight line; it is a straight line: each step translates the coordinate down by $|\ln q|$; after $n$ steps, down by $n|\ln q|$. In the log coordinate $y$, the map at this step is

$$
y\mapsto y+n\cdot\ln q
$$

Scaling has become translation. And $n\ln q$, as a function of $n$, is linear: it passes through the origin, its slope $\ln q$ is constant, and it travels as far as $n$ travels. In log space the exponential has no curvature left; only this linear term and its slope remain.

### 3.2 A proper fraction

$0\le q<1$ $\Rightarrow$ $\ln q<0$. Hence

$$
\ln\Delta_n-\ln\delta_n\le n\cdot\ln q<0
$$

that is, $\Delta_n<\delta_n$: the ratio is a proper fraction.

### 3.2.1 Why this term goes to $-\infty$

On the right, $n\ln q$ is $n$ times a constant, with $\ln q<0$, so each step drops by a fixed amount $|\ln q|$. The steps have no end, the spacing never changes, and it passes through every real number on its way down, with no floor; no finite upper bound can stop it, so it can only be written as $-\infty$.

The left side is tied below the right side by $\le$. The right side has no bottom, so the left side's upper bound has no bottom either; it falls along with it.

The left side had other places it could go. $\ln\Delta_n-\ln\delta_n$ is the difference of the logs of two distances, and distances take only finitely many values: integers $0,1,\dots,|D|$ in the unweighted case, finitely many nonnegative numbers in the weighted case. When nonzero, its log is some finite number; the log sends exactly one distance to $-\infty$, namely $0$:

$$
\ln 0=-\infty
$$

So the full set of possible values on the left is a finite set plus the single point $-\infty$. No number in the finite set can stay below $n\ln q$ all the way; only $-\infty$ can follow it to the end. The limit therefore records the fixed point at the far end of the coordinate: log space has no lower slot.

### 3.3 Taking the limit

Let $n\to\infty$:

$$
\lim_{n\to\infty}(\ln\Delta_n-\ln\delta_n)\le\lim_{n\to\infty}n\cdot\ln q=-\infty
$$

The upper bound runs off to $-\infty$, so:

- the limit is at $-\infty$
- the fixed point is also at $-\infty$

## 4. Contradiction

But this space is a discrete space, and **it has no $-\infty$ entry**.

On one side the fixed point is at $-\infty$; on the other, the space has no $-\infty$:

$\Rightarrow$ a contradiction, directly falsified.

## 5. The P Resident and the NP Counterexample

The conclusions below all live inside the discrete space $(X,d)$ of §1.

### 5.1 Discreteness lemma

$X=A_3^D$, $|D|=8$, $|A_3|=11$, so $|X|=11^8$ is finite. Weighted Hamming distance takes only finitely many values; let

$$
\delta_{\min}:=\min\{d(f,g):\ f,g\in X,\ d(f,g)>0\}>0
$$

Let $\{a_n\}$ be a Cauchy sequence. Take $N$ such that $d(a_m,a_n)<\delta_{\min}$ for $m,n\ge N$. Then

$$
d(a_m,a_n)=0\qquad(m,n\ge N)
$$

$$
a_N=a_{N+1}=a_{N+2}=\dots
$$

That is, the limit of the Cauchy sequence is a concrete point $a^*$ in $X$.

### 5.2 $L_{\mathrm{SAT}}$ and the P resident

$L_{\mathrm{SAT}}$: every clause has exactly $L$ literals.

- $L=2$: 2-SAT, $\in\mathrm P$;
- $L=3$: 3-SAT, $\in\mathrm{NP}$ (Cook–Levin).

The chain has to be made clear. 3-SAT is NP-hard, and that is not something we constructed. To be precise, it is NP-complete: it lies in NP and is NP-hard. This conclusion is older than this book: Cook (and Levin) proved that every problem in NP has a polynomial-time reduction to SAT; from SAT to 3-SAT only two mechanical things happen — long clauses are split into clauses of three literals (adding auxiliary variables where needed), and short clauses are padded with repeated literals. Join the two stretches, and every problem in NP reduces to 3-SAT. Someone else welded the chain; 3-SAT is tied at its end, and NP-hardness is a property the chain gives it, not a property our step gives it.

What we construct here is the space and the residents, not NP-hardness. Constructing is not the problem: to prove existence, giving a construction is a proof. The point is to say which stretch is borrowed. The reduction chain is borrowed from Cook–Levin; the space, the iteration sequence, and the P resident are ours. Seeing “construct 3-SAT” should not suggest that the act of construction automatically produces NP-hardness — the hardness comes with the old chain, not with the new construction. This constructive proof is about who lives in the space and who does not.

The P resident is constructed on the $L=2$ side; the mature resident on the $L=3$ side is constructed directly along the Cook reduction chain.

For the 2-SAT case take $L=2$: $D=\{0,1\}^2=\{00,01,10,11\}$, 4 states. Plug in the $L=2$ values: $A_2$ has $4$ states plus $3$ operators, $7$ values in all; the space $X_2=A_2^D$ has $7^4$ points. The resident uses only the $4$ state values — write $B:=\{0,1\}^2\subset A_2$, $0:=(0,0)$, $1:=(1,1)$, with $\wedge$ acting bitwise on $B$; the whole orbit lies in $B^D\subset X_2$, $4^4=256$ points. Hamming distance is an integer lying between $0$ and $4$.

Derive from scratch. First fix the multiplier:

$$
h:D\to B,\qquad h(x_1,x_2):=(x_1,x_1)
$$

It reads only the first bit and writes it into both slots. The values at the four inputs, listed one by one:

| $x$ | $00$ | $01$ | $10$ | $11$ |
|:--|:--:|:--:|:--:|:--:|
| $h(x)$ | $00$ | $00$ | $11$ | $11$ |

Read one bit, write two; cost $O(1)$, on the P side; and $h\in B^D\subset X_2$.

Next define the operator as pointwise conjunction:

$$
T_h(f)(x):=f(x)\wedge h(x)\qquad(x\in D)
$$

Take the constant-one function as the starting point:

$$
a_0:=1_X,\qquad a_0(x)=11\quad(x\in D)
$$

First step, computed pointwise: one two-bit conjunction for each of the four inputs:

$$
\begin{aligned}
a_1(00)&=11\wedge h(00)=11\wedge00=00\\
a_1(01)&=11\wedge h(01)=11\wedge00=00\\
a_1(10)&=11\wedge h(10)=11\wedge11=11\\
a_1(11)&=11\wedge h(11)=11\wedge11=11
\end{aligned}
$$

Going through the four values one by one: both bits of $11\wedge v$ are $1$, so the conjunction lets each bit through unchanged ($1\wedge0=0$, $1\wedge1=1$); the result is the value of $h$ at that point. Hence

$$
a_1=h
$$

Second step, again pointwise:

$$
\begin{aligned}
a_2(00)&=a_1(00)\wedge h(00)=00\wedge00=00\\
a_2(01)&=a_1(01)\wedge h(01)=00\wedge00=00\\
a_2(10)&=a_1(10)\wedge h(10)=11\wedge11=11\\
a_2(11)&=a_1(11)\wedge h(11)=11\wedge11=11
\end{aligned}
$$

The two slots of $h$ are equal, so at each input the conjunction is between two identical two-bit values; bitwise $0\wedge0=0$, $1\wedge1=1$, and the result is itself again: $h\wedge h=h$. Hence $a_2=a_1$, and every step after is the same:

$$
a_n=h\qquad(n\ge1)
$$

The distance ledger is settled coordinate by coordinate:

$$
d(a_0,a_1)=|\{x\in D:\ 11\ne h(x)\}|=|\{00,01\}|=2
$$

$$
d(a_1,a_2)=|\{x\in D:\ h(x)\ne h(x)\}|=0
$$

That is, $\delta_0=2$, $\delta_1=0$. $\delta_n$ is an integer between $0$ and $4$; starting from $\delta_0=2$ it never increases ($0\le q<1\Rightarrow q^n\delta_n\le\delta_n$), reaches $0$ at step $1$, and stays $0$ thereafter. Iteration condition $\Delta_n\le q^n\delta_n$: for $n=0$, $0\le2$; for $n\ge1$, $0\le0$; it holds everywhere.

Therefore

$$
a^*_P:=h\in X_2
$$

The P resident is constructed: frozen after one step, with limit the explicit point $h$ in the space.

This resident is special in two places. Its limit is the point $h$ in $X_2$, not $-\infty$: after one step every bit is fixed, and further steps only repeat. And it satisfies exactly $2$ states — $h(x)=11$ holds only on $10$ and $11$; the other two inputs give $00$. The 3-time gate opens only after counting the third satisfied state; among all behaviors of $L=2$ a third cannot be produced, so the gate is forever closed. The P resident lives inside this closed gate.

### 5.3 The NP counterexample

Suppose for contradiction that an NP resident $a^*_{NP}$ exists, with fixed point $-\infty$. By §3.3 and §4:

$$
a^*_{NP}=-\infty\notin X
$$

The iteration sequence $\{a_n\}$ on the NP side satisfies the condition of §2.1 and freezes within $N\le9$ steps:

$$
a_N=a_{N+1}=\dots=a^*\in X
$$

Hence $\lim a_n=a^*\in X$, which contradicts $\lim a_n=a^*_{NP}=-\infty\notin X$.

So no NP resident exists; the counterexample object is $-\infty$ itself, which is not in the space.

### 5.4 Conclusion

$$
\begin{aligned}
\text{P-resident}:&\quad a^*_P=h\in X_2&&\text{(}L=2\text{, constructed explicitly in 5.2)}\\
\text{NP-resident}:&\quad a^*_{NP}=-\infty\notin X&&\text{(}L=3\text{, proved nonexistent in 5.3)}
\end{aligned}
$$

$$
\Rightarrow\ \mathrm P\ne\mathrm{NP}
$$

## 6. Back to Ideal Time

### 6.1 Oracle time is already in hand

People speak of NP-hard problems under ideal time as if ideal time were something extra, out of reach. The ideal time in this book is not extra. It is the oracle time of §1.3.3, and also the time $n$ of the iteration sequence in §2. Where the sequence stops is where ideal time stops.

It has already been proved. The $q^n$ in the iteration condition stops growing after the first step: for $n\ge1$, $q^n\le q$. Write this uniform decay constant as

$$
Q:=q
$$

Then every further step multiplies the distance between adjacent terms by at most $Q$:

$$
\delta_{n+1}\le Q\,\delta_n\qquad(n\ge1)
$$

Multiply through to step $n$:

$$
\delta_n\le\delta_1\,Q^{n-1}\qquad(n\ge1)
$$

$\delta_1$ is the adjacent distance after the first step, no larger than the number of coordinates $|D|$. Hence

$$
\delta_n\le|D|\,Q^{n-1}
$$

This inequality is the ideal-time ledger. $Q$ is the decay constant, $|D|$ the size of the space; both are written in the space. One transformation computes it.

### 6.2 How much remains, how much time

Suppose a distance $\varepsilon$ remains, that is, we require $\delta_n<\varepsilon$. From the inequality above, it suffices that

$$
|D|\,Q^{n-1}<\varepsilon
$$

Take logarithms to base $1/Q$ on both sides ($1/Q>1$, so the direction does not change):

$$
n-1>\log_{1/Q}\frac{|D|}{\varepsilon}
$$

that is,

$$
n>1+\log_{1/Q}\frac{|D|}{\varepsilon}
$$

The right side uses only three numbers: the size of the space $|D|$, the tolerance $\varepsilon$, and the decay constant $Q$. All three are written in the space; none is foreign. This is the oracle-time upper bound corresponding to the remaining distance $\varepsilon$. When $\varepsilon\ge|D|$, no further walking is needed; when $Q=0$, it freezes in one step.

This upper bound is a certificate. The certificate need not carry the starting gap $\delta_0$; one $Q$ is enough. $Q$ is the decay constant of this sequence; $\delta_0$ is only where one particular run started.

### 6.3 The time upper bound for the ideal state

The ideal state is $\delta=0$. Distances in the space take finitely many values; let $\delta_{\min}$ be the smallest positive distance among them (§5.1): $\delta_n<\delta_{\min}$ can only imply $\delta_n=0$. Substitute $\varepsilon=\delta_{\min}$:

$$
n^*_{\text{ideal}}
\le
\left\lceil
1+\log_{1/Q}\frac{|D|}{\delta_{\min}}
\right\rceil
$$

Plug in the numbers for $L=3$: $|D|=8$, $\delta_{\min}=1$, hence

$$
n^*_{\text{ideal}}
\le
\left\lceil
1+\log_{1/Q}8
\right\rceil
=
\left\lceil
1+\frac{\ln 8}{\ln(1/Q)}
\right\rceil
$$

$Q$ is the decay constant, fixed; the right side is therefore a constant. Ideal time is finite, and can be counted out.

This connects with §3. The inequality in log space has content only while $\delta_n>0$ and the sequence is still moving (§2.2); once the sequence stops, it has reached its end. Ideal time is the length of this stretch of inequality. To call ideal time out of reach is to treat it as a limit; here it is a string of rungs that can be counted to the end.

### 6.4 Two kinds of oracle time

On the P resident's side, the certificate can be issued on the spot. After the first step $\delta_1=0$; take the decay constant $Q=0$: the distance is already squeezed to zero, and further steps only repeat. Ideal time $1$, down in a single step. The multiplier is the closed-form $h$, which reads only the first bit; the candidate table $X_2$ has $7^4$ points, and it touches only two, $a_0$ and $h$. Oracle time is counted linearly in the number of rows, polynomial.

The NP side cannot issue the certificate. A certificate needs a stopping point inside the space for $Q$ to have somewhere to be written; $-\infty$ is not in the space, and the pen cannot land. If one insists on recording an oracle time for it, only enumeration remains: spread out the whole stack of candidate tables; for $L=3$, $X_3$ has $11^8\approx2.1\times10^8$ points. Each increase of $L$ by $1$ grows $A_L$ from $7$ values to $11$ and then to $19$, and $X_L$ from $7^4$ to $11^8$ and then to $19^{16}$ — base and exponent grow together, exponents stacked on exponents. The upper bound of this ledger is an explosive exponential geometry.

The two lines are different. The P resident's certificate carries a $Q$ (here $0$), and the ideal time is $1$, polynomial-linear; the NP resident cannot even issue a certificate, and only the size of the whole stack of candidate tables remains, an explosive exponential geometry. Put $\mathrm P\ne\mathrm{NP}$ on the time axis, and it says these two lines never touch.

---

The book ends here.

It is an open book. What is proved is laid here, and what is not proved is laid here too. The conclusion of Chapter 24 stops at $\mathrm P\ne\mathrm{NP}$; there is no next page.

Walking toward tomorrow does not require first becoming a person who can be written down, complete, normal, brave, without fear or shame. No one has that obligation.

Tomorrow will not open its front door for us. Then wade across. Feel the world as before, imagine as before, live as before.

Once you have waded across, you are across.

Good luck.

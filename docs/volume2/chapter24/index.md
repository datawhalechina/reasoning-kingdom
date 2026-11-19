# 第24章：NP 问题的探讨——LIT 定理

---

$A_L$ 集合；$A_L$ 居民：$1_L$

$X_L=(A_L,\&,|,!)$ 半环

$(X_L,d)$ $\Rightarrow$ 空间

路线：定义空间与距离 $\to$ 构造迭代列 $\to$ 除法 $\to$ 换底进入对数空间 $\to$ 取 $n\to\infty$ $\to$ 用「离散空间里没有 $-\infty$」导出矛盾 $\to$ 证离散性引理 $\to$ 显式构造 P 居民 $\to$ 反证 NP 居民不存在。

## 1. 空间

### 1.1 值集合 $A_L$（平凡基础）

对每个 $L$，定义平凡值集合

$$
A_L:=\{0,1\}^L\cup\{\&,|,!\}
$$

$|A_L|=2^L+3$。$A_L$ 只做两件事：装 $L$ 位状态，装三个逐点算子。平凡居民取常一状态

$$
1_L:=(1,\dots,1)\in\{0,1\}^L\subseteq A_L
$$

到空间上就是常一值函数 $1_{X_L}$。

### 1.2 构造 $A_3$ 与完备空间

取 $L=3$：$A_3:=A_L|_{L=3}$，由 8 个状态和 3 个算子组成，共 11 个：

| 编号 | 状态 |
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

$X_3$ 是值函数空间：元素 $f:D\to A_3$ 是值函数，$\&$、$\vert$、$!$ 对它逐点作用。

3-SAT 情形取 $D=\{0,1\}^n$。值函数就是求值映射：

$$
\Phi:\{0,1\}^n\to A_3
$$

$n=3$ 时 $D$ 有 8 个状态；$|A_3|=11$，所以 $X_3=A_3^D$ 有 $11^8$ 个值函数。可满足 $\Leftrightarrow$ $\Phi$ 不是常零函数——在空间里是「非零函数」这个子集，形态是一个锥 / 超平面，里面没有子句语法。

$X_3=A_3^D$ 是完备空间：距离见 1.4，完备性见 1.5。

### 1.3 构造 $A_2$ 与完备空间

取 $L=2$：$A_2:=A_L|_{L=2}$，由 4 个状态和 3 个算子组成，共 7 个：

| 编号 | 状态 |
|-----:|:-----|
| 0 | $(0,0)$ |
| 1 | $(0,1)$ |
| 2 | $(1,0)$ |
| 3 | $(1,1)$ |
| 4 | $\&$ |
| 5 | $\vert$ |
| 6 | $!$ |

2-SAT 情形取 $D=\{0,1\}^2$，$X_2:=A_2^D$ 是完备空间：距离见 1.4，完备性见 1.5。

$L=2$：2-SAT，$\in\mathrm P$；$L=3$：3-SAT，$\in\mathrm{NP}$（Cook–Levin）。

### 1.3.1 不是 3 位/2 位填表，而是整个解空间沿时间轴迭代

$A_2$、$A_3$ 不是「输入只有 2 位或 3 位」。它们是逐点枚举的值集合：$A_L$ 把 $\{0,1\}^L$ 的每个状态都枚举进去，$X_L=A_L^D$ 再把所有候选值函数包进同一个空间。

- $D=\{0,1\}^3$ 枚举 8 个候选输入状态；$X_3=A_3^D$ 是整个候选解空间。
- 迭代对象 $a_n\in X_3$ 不是单个 3 位赋值，而是整张候选表：8 个状态的值同时被包在 $a_n$ 里，沿时间刻度 $n$ 一起演化。
- 时间门只数「恰好 3 个满足状态」：$\operatorname{satStates}(a_n)=3$ 时，整个空间才允许推进到 $a_{n+1}$；否则时间停住，无法往外推。

所以「3」是当前时间片上恰好满足的 3 个状态，不是输入长度；「2」同理，最多只有 2 个满足状态，变不出第 3 个，3-时间门永远关闭。

### 1.3.2 最优时间：只计成功推进

回溯、改错、踌躇不前的分支不进时间账。只记录真正过门、成功向前的步骤：

$$
\operatorname{ReachesIn}\ f_0\ 0\ f_0
$$

$$
\operatorname{ReachesIn}\ f_0\ (n+1)\ g
\qquad
\text{若}\ \operatorname{ReachesIn}\ f_0\ n\ f\ \text{且}\ \operatorname{stepSpace}\ f=\operatorname{some}\ g
$$

$\operatorname{IsOptimalTime}\ f_0\ \text{goal}\ n$ 取的是最短成功路径的步数；失败分支和改错时间被归约掉，不计入 $n$。

### 1.3.3 离散时间与神级机器（绝对最优时间）

时间取离散刻度 $n=0,1,2,\dots$。状态空间不是单条轨迹，而是整个 $X_L$ 被包在 $a_n$ 里一起演化。

Oracle 时间（神谕时间）是一个名词：它指没有任何改错、完全贴着状态空间成功路径演化的最理想时间。它不是新增的物理机器，也不是额外的计算能力；它只是给这个最理想时间起的名字。设想一个神谕（oracle）作为这个理想演化者：每一步都完全贴着状态空间里成功的那条路径走，失败、回溯、改错的分支根本不进入时间账。它的时间定义成

$$
\begin{aligned}
&\operatorname{OracleRun}\ f_0\ \text{goal}\ n \iff \exists\ \text{state}:\mathbb N\to X_3\\
&\qquad \text{state}\ 0=f_0\\
&\qquad \text{state}\ n=\text{goal}\\
&\qquad \forall i<n,\ \operatorname{stepSpace}\ (\text{state}\ i)=\operatorname{some}\ (\text{state}\ (i+1))
\end{aligned}
$$

$\operatorname{OracleOptimal}$ 再要求它是所有成功路径中最短的：

$$
\operatorname{OracleOptimal}\ f_0\ \text{goal}\ n
=
\operatorname{OracleRun}\ f_0\ \text{goal}\ n
\wedge
\forall m,\ \operatorname{ReachesIn}\ f_0\ m\ \text{goal}\to n\le m
$$

这个 $n$ 是绝对最优时间：

- 上界：确实存在一条 $n$ 步成功路径（神谕到了）；
- 下界：任何成功路径都不短于 $n$；
- 回溯、改错、踌躇不前的时间全部归约掉，不计入 $n$。

它是神谕的时间，不是任何物理机器的实际时间；物理机器只能比它慢。

### 1.4 加权汉明距离

空间是离散的：$A_L$ 是 $2^L+3$ 个元素的有限集，$D=\{0,1\}^L$ 是 $2^L$ 个状态的有限集，$X_L=A_L^D$ 是有限积。空间里没有方向、线段和内积，欧几里得距离不是它自身的度量；坐标结构给出的度量必须是汉明型——逐坐标比较，数不同的坐标：

$$
d(f,g)=|\{x\in D:\ f(x)\ne g(x)\}|
$$

几何性质：

- 取值为整数 $0,1,\dots,|D|$；
- 相邻即「恰好一个坐标不同」，$d(f,g)$ 是有限积图上的最短路径长度；
- 单位权时，以 $f$ 为心、半径 $r$ 的球是「至多 $r$ 个坐标不同」的函数集，是子立方体的并。

加权汉明在每个坐标上乘权重 $w_x\ge0$：

$$
d_w(f,g)=\sum_{x\in D} w_x\cdot\mathbf 1[f(x)\ne g(x)]
$$

- 度量的汉明型由离散性决定，权重不改变这个型
- $w_x\equiv1$ $\Rightarrow$ 普通汉明距离
- 一般值集合：$d_w(f,g)=\sum_{x\in D} w_x\cdot\delta(f(x),g(x))$，$\delta$ 为 $A_3$ 上的离散度量

3-SAT 情形（$D=\{0,1\}^n$）：

$$
d_w(f,g)=\sum_{x\in\{0,1\}^n} w_x\cdot\mathbf 1[f(x)\ne g(x)]
$$

$n=3$ 时求和跑完 8 个状态；$n=2$ 时求和跑完 4 个状态。

范数形式（$A_3=\{0,1\}$）：

$$
\|f\|_w=\sum_{x\in D} w_x\cdot f(x)
$$

$$
d_w(f,g)=\|f\oplus g\|_w
$$

### 1.5 完备性

- $A_L$ 是值集合，$X_L$ 是值函数空间，$\&$、$\vert$、$!$ 逐点作用。
- 逐点结构 $\Rightarrow$ 完备性按坐标继承：$X_L$ 是每个输入坐标上一份 $A_L$ 的积。$X_L$ 是半环 / 格 / 有补 $\Leftrightarrow$ $A_L$ 是；$X_L$ 完备 $\Leftrightarrow$ $A_L$ 完备。
- 度量侧：加权汉明是逐坐标聚合。输入集有限 $\Rightarrow$ $X_L$ 有限 $\Rightarrow$ 柯西完备自动成立；无限时，$X_L$ 的完备性由 $A_L$ 的距离完备性决定。
- 表达侧：$\{\&,\vert,!\}$ 逐点生成 $A_L$ 上的函数代数；能生成全部值函数，靠的是 $A_L$ 本身是布尔值集。

一句话：这个值函数空间的完备性是**继承来的**——$A_L$ 完备它就完备，$A_L$ 缺什么它就缺什么。

## 2. 迭代列

### 2.1 最朴素的单变量迭代

迭代对象不是单个值，也不是单个变量 $X_1,X_2,\dots$，而是把全部状态收进同一个集合变量：

- 所有值集合
- 所有状态
- 所有变量的全部取值方式

全部包进 $a_n$，让它按时间 $n$ 往前迭代；每一步把集合里的状态全部取到，这是穷举性质的迭代。所谓「单变量」，指的就是这一个包住全部状态的集合变量。

在完备空间 $(X,d)$ 上构造迭代列：

$$
a_{n+1}=T(a_n)
$$

迭代条件：

$$
d(T(a_n),T(a_{n+1}))\le q^n\cdot d(a_n,a_{n+1})
$$

证明它是柯西列分两步：

1. 相邻两项的差有界：

   $$
   d(a_{n+1},a_n)\le d(a_0,a_1)
   $$

2. 再证出 $T$ 是压缩映射，把这些差按几何级数压到 $0$。

### 2.2 迭代条件的除法形式

记

$$
\Delta_n:=d(T(a_n),T(a_{n+1}))
$$

$$
\delta_n:=d(a_n,a_{n+1})
$$

把迭代条件右边的乘子除到左边：

$$
\frac{\Delta_n}{\delta_n}\le q^n
$$

- 左边：像距离 $\Delta_n$ 除以原像距离 $\delta_n$
- 右边：指数函数形式
- $0\le q<1$

## 3. 对数空间

### 3.1 换底公式

换底公式：

$$
\log_b x=\frac{\log_c x}{\log_c b}
$$

把真数替换为分子 $\Delta_n$，底数替换为分母 $\delta_n$：

$$
\log_{\delta_n}\Delta_n=\frac{\log_c\Delta_n}{\log_c\delta_n}
$$

除法不等式换到对数空间：

$$
\begin{aligned}
&\ln\left(\frac{\Delta_n}{\delta_n}\right)\le n\cdot\ln q\\
\iff\ &\ln\Delta_n-\ln\delta_n\le n\cdot\ln q
\end{aligned}
$$

- 指数形式 $q^n$ 在对数空间变成线性项 $n\cdot\ln q$
- 除法式与换底公式同构：分子 $\Delta_n$ $\leftrightarrow$ 真数，分母 $\delta_n$ $\leftrightarrow$ 底数

### 3.2 真分数

$0\le q<1$ $\Rightarrow$ $\ln q<0$。于是

$$
\ln\Delta_n-\ln\delta_n\le n\cdot\ln q<0
$$

即 $\Delta_n<\delta_n$：这个比值是一个真分数。

### 3.3 取极限

令 $n\to\infty$：

$$
\lim_{n\to\infty}(\ln\Delta_n-\ln\delta_n)\le\lim_{n\to\infty}n\cdot\ln q=-\infty
$$

上界跑到 $-\infty$，所以：

- 极限在 $-\infty$
- 不动点也在 $-\infty$

## 4. 矛盾

但这个空间是离散空间，**没有 $-\infty$ 这一项**。

一边不动点在 $-\infty$，一边空间里没有 $-\infty$：

$\Rightarrow$ 前后矛盾，直接证伪。

## 5. P 居民与 NP 反例

以下结论都在 §1 的离散空间 $(X,d)$ 内。

### 5.1 离散性引理

$X=A_3^D$，$|D|=8$，$|A_3|=11$，故 $|X|=11^8$ 有限。加权汉明距离只取有限个值，令

$$
\delta_{\min}:=\min\{d(f,g):\ f,g\in X,\ d(f,g)>0\}>0
$$

设 $\{a_n\}$ 是柯西列。取 $N$，使 $m,n\ge N$ 时 $d(a_m,a_n)<\delta_{\min}$。于是

$$
d(a_m,a_n)=0\qquad(m,n\ge N)
$$

$$
a_N=a_{N+1}=a_{N+2}=\dots
$$

即柯西列的极限是 $X$ 中的一个具体点 $a^*$。

### 5.2 $L_{\mathrm{SAT}}$ 与 P 居民

$L_{\mathrm{SAT}}$：每个子句恰好 $L$ 个字面量。

- $L=2$：2-SAT，$\in\mathrm P$；
- $L=3$：3-SAT，$\in\mathrm{NP}$（Cook–Levin）。

P 居民取 $L=2$ 一侧构造；$L=3$ 一侧的成熟居民直接沿 Cook 规约链构造。

2-SAT 情形取 $n=2$，$D=\{0,1\}^2$（4 个状态）。记 $B:=\{0,1\}^2\subset A_2$（4 个值），$0:=(0,0)$，$1:=(1,1)$，$\wedge$ 在 $B$ 上逐比特作用；值函数空间 $X_2:=B^D$。

定义

$$
h:D\to B,\qquad h(x):=(x_1,x_1)
$$

即只读第一个比特：

| $x$ | $00$ | $01$ | $10$ | $11$ |
|:--|:--:|:--:|:--:|:--:|
| $h(x)$ | $00$ | $00$ | $11$ | $11$ |

$h$ 的计算代价是 $O(1)$，属于 P 一侧；且 $h\in B^D\subseteq X_2$。

取

$$
T_h(f):=f\wedge h\qquad\text{（逐点）}
$$

$$
a_0:=1_X\qquad\text{（常一函数）}
$$

迭代：

$$
\begin{aligned}
a_1&=T_h(a_0)=1_X\wedge h=h\\
a_2&=T_h(a_1)=h\wedge h=h\\
a_n&=h\qquad(n\ge1)
\end{aligned}
$$

因此

$$
a^*_P:=h\in X_2
$$

P 居民构造完成：一步后冻结，极限是显式点 $h$。迭代条件 $\Delta_n\le q^n\cdot\delta_n$ 在 $n\ge1$ 时两边为 $0$，在 $n=0$ 时 $\Delta_0=0\le\delta_0$，处处成立。

$L=3$ 一侧的成熟居民沿 Cook 规约链（CIRCUIT-SAT $\to$ SAT $\to$ 3-SAT）构造。

### 5.3 NP 反例

反设存在 NP 居民 $a^*_{NP}$，其不动点为 $-\infty$。由 §3.3、§4：

$$
a^*_{NP}=-\infty\notin X
$$

NP 侧的迭代列 $\{a_n\}$ 满足 §2 的条件，是柯西列。由 5.1，存在 $N$：

$$
a_N=a_{N+1}=\dots=a^*\in X
$$

故 $\lim a_n=a^*\in X$，这与 $\lim a_n=a^*_{NP}=-\infty\notin X$ 矛盾。

所以 NP 居民不存在；反例对象就是 $-\infty$ 本身，它不在空间里。

### 5.4 结论

$$
\begin{aligned}
\text{P-resident}:&\quad a^*_P=h\in X_2&&\text{（}L=2\text{，5.2 显式构造）}\\
\text{NP-resident}:&\quad a^*_{NP}=-\infty\notin X&&\text{（}L=3\text{，5.3 反证不存在）}
\end{aligned}
$$

$$
\Rightarrow\ \mathrm P\ne\mathrm{NP}
$$

# 第23章：永霖推论——智能与涌现的对数曲线

> 一个系统收敛得多快，只由一个数决定。

---

[第12章](/volume1/chapter12/)已经证明了收敛：在切空间谱界与步长窗口下，推断算子 $F$ 是压缩映射，巴拿赫不动点定理保证迭代收敛到先验锚点 $A$。那里的能量面是

$$
E(p)=D_{\mathrm{KL}}(A\|p).
$$

收敛有了，“方向”也有了。但“最终会收敛”和“第几步收敛”是两件事。

这一章只做一件事：**把收敛速率这个数算出来，然后看它能推出什么。**

结论会反复出现同一个对象：

$$
\boxed{\ \rho^*=\frac{\kappa_T-1}{\kappa_T+1}\ }
$$

其中 $\kappa_T=\theta_{\max}/\mu_T$ 是切空间 Hessian 的条件数。这个数决定误差曲线的斜率，决定到达精度 $\varepsilon$ 需要多少步，也决定“涌现”是不是真的。除此之外，本章不需要第二个标量。

为什么叫“推论”？因为收敛性的发动机仍然是巴拿赫不动点定理，能量面仍然是信息论里的 KL 散度。本章没有换发动机，只是把发动机的功率读出来。

这不是智能的全部。但任何以概率单纯形为输出空间的智能系统，都绕不开这一段几何。

> **永霖推论 = 信息论的能量面 + 巴拿赫的完备性 + 一个速率。**

:::info 波兰学派的提示
本章的完备性来自巴拿赫不动点定理，而它出自两次世界大战之间的波兰数学学派。学派的故事，以及“为什么要把收敛放进完备空间”的方法论，放在[第12章](/volume1/chapter12/)的波兰学派注记里。本章只使用这个结果，不复述历史。
:::

---

## 23.1 一个数：压缩率

推断算子写成

$$
F(p)=\Pi_{\mathcal{X}}\left(p-\eta P_T\nabla E(p)\right),
$$

其中 $E(p)=D_{\mathrm{KL}}(q\|p)$，$P_T=I-\frac1K\mathbf{1}\mathbf{1}^\top$ 是到切空间

$$
T=\left\{v\in\mathbb{R}^K:\ \mathbf{1}^\top v=0\right\}
$$

的正交投影。在不动点 $A=q$ 处，

$$
\nabla^2E(A)=\operatorname{diag}\!\left(\frac1{q_i}\right),
\qquad
H_T=P_T\,\nabla^2E(A)\,P_T.
$$

$H_T$ 是切空间 Hessian。它在 $T$ 上对称半正定，有 $K-1$ 个特征值。记最小和最大分别为

$$
\mu_T=\lambda_{\min}(H_T),\qquad \theta_{\max}=\lambda_{\max}(H_T).
$$

$F$ 在 $A$ 处的切空间导数是

$$
J_T=I_T-\eta H_T.
$$

它的特征值是 $1-\eta\lambda_i$。定义**压缩率**

$$
\rho(\eta)=\max_i|1-\eta\lambda_i|.
$$

这是本章唯一的主角。

**第一个数学性质：$\rho(\eta)$ 不是上界，而是精确的谱半径。**

$J_T$ 在 $T$ 上对称，所以算子范数等于谱半径：

$$
\|J_T\|_{T\to T}=\rho(\eta).
$$

于是对 $A$ 附近的初始点，线性化误差的渐近速率精确等于 $\rho(\eta)$：

$$
\lim_{t\to\infty}\left(\frac{\|p_t-A\|_2}{\|p_0-A\|_2}\right)^{1/t}=\rho(\eta).
$$

（若初始误差在主导特征空间上的分量不为零；否则只会更快。）这里没有“近似压缩”“大概收敛”之类的余地。压缩率就是谱半径，谱半径就是对数曲线的斜率。

![图 23.1：单纯形内部的几何压缩——进入锚点邻域后，每一步把误差乘上同一个因子 $\rho^*$](/figures/ch23_simplex_orbit.png)

图 23.1 把“压缩”画在概率单纯形上。轨道从 $p_0$ 出发，每次迭代都朝锚点 $A=q$ 靠近。刚进入锚点邻域时，误差的收缩比例会短暂波动；一旦进入线性化有效的范围，每一步就都乘以同一个数 $\rho^*\approx0.454$。这就是“几何收敛”四个字的几何含义：不是每次减少同样的距离，而是每次按同样的比例缩短。

---

## 23.2 只有两个端点有效

$H_T$ 有 $K-1$ 个特征值。一个自然的担心是：算速率是不是需要知道全部谱？

不需要。

**引理 23.1.** 设 $\mu_T\le\lambda_i\le\theta_{\max}$。对任意 $\eta>0$，

$$
\max_i|1-\eta\lambda_i|
=\max\left(|1-\eta\mu_T|,\ |1-\eta\theta_{\max}|\right).
$$

**证明.** 固定 $\eta$，函数 $g(\lambda)=|1-\eta\lambda|$ 是 $\lambda$ 的凸函数（绝对值函数与仿射函数的复合）。凸函数在区间上的最大值必在端点取得。对每个 $\lambda_i$ 都有 $g(\lambda_i)\le\max(g(\mu_T),g(\theta_{\max}))$，且端点本身就是特征值，所以等号成立。$\square$

这个引理把 $K-1$ 维的谱压缩成两个数：**最小值 $\mu_T$ 和最大值 $\theta_{\max}$。** 中间有多少特征值、分布在哪里，都不影响压缩率。

它还给出稳定窗口的精确边界：

$$
\rho(\eta)<1
\iff
0<\eta<\frac{2}{\theta_{\max}}.
$$

左边是因为 $1-\eta\lambda_i>-1$ 对所有 $i$ 成立；右边是因为最快模式 $1-\eta\theta_{\max}$ 最先越过 $-1$。边界是精确的，不是“大约在这个范围”。

---

## 23.3 最优速率：精确公式

现在选一个步长，让 $\rho(\eta)$ 最小。这是两个端点分支的极小极大问题：

$$
\rho^*=\inf_{\eta>0}\ \max\left(|1-\eta\mu_T|,\ |1-\eta\theta_{\max}|\right).
$$

最优点在两个分支相等处：

$$
1-\eta\mu_T=\eta\theta_{\max}-1.
$$

解出

$$
\eta^*=\frac{2}{\mu_T+\theta_{\max}},
\qquad
\rho^*=\frac{\theta_{\max}-\mu_T}{\theta_{\max}+\mu_T}
=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T}.
$$

**定理 23.2（永霖推论的速率形式）.** 在稳定窗口内，$\rho(\eta)$ 的最小值是 $\rho^*$，唯一最小值点是 $\eta^*$。并且

$$
|1-\eta^*\mu_T|=|1-\eta^*\theta_{\max}|=\rho^*.
$$

也就是说，最优步长让最慢模式和最快模式以完全相同的速度收缩。

这个速率有四个精确性质，每一个都只需要一行代数：

1. **范围。** $0\le\rho^*<1$。当 $\kappa_T=1$（所有切向特征值相等）时 $\rho^*=0$，一步到位；当 $\kappa_T\to\infty$ 时 $\rho^*\to1$，收敛无限慢。
2. **单调。** $\dfrac{d\rho^*}{d\kappa_T}=\dfrac{2}{(\kappa_T+1)^2}>0$。条件数越大，速率越差，没有例外。
3. **可逆。** $\kappa_T=\dfrac{1+\rho^*}{1-\rho^*}$。速率与条件数是一一对应的：知道速率就等于知道条件数，反之亦然。在这个意义上，$\rho^*$ 是条件数的完整不变量。
4. **只依赖端点。** 由引理 23.1，$\rho^*$ 只由 $\mu_T,\theta_{\max}$ 决定。中间的谱形状不进入公式。

第 3 条值得多看一眼。它说明“收敛多快”与“能量面多各向异性”不是两个独立的问题：它们是同一个数的两种写法。

![图 23.2：一个标量的三种读法——精确 V 形、速率与条件数的互逆、窗口的精确线性夹逼](/figures/ch23_rate_properties.png)

面板 (a) 是 $\rho(\eta)$ 的 V 形。左侧由慢模式 $|1-\eta\mu_T|$ 控制，右侧由快模式 $|1-\eta\theta_{\max}|$ 控制，交点就是 $(\eta^*,\rho^*)$。虚线是稳定边界 $2/\theta_{\max}$。

面板 (b) 是 $\rho^*=(\kappa_T-1)/(\kappa_T+1)$。它单调、有界、可逆。

面板 (c) 是窗口。下一节会把它写成精确不等式。

![图 23.3：速率–条件数双曲线——两个方向读同一个关系](/figures/ch23_rate_hyperbola.png)

图 23.3 把这个关系单独画出来。左图是 $\rho^*=(\kappa_T-1)/(\kappa_T+1)$ 作为 $\kappa_T$ 的函数：它是一支双曲线，水平渐近线是 $\rho^*=1$。右图是它的反函数 $\kappa_T=(1+\rho^*)/(1-\rho^*)$：当 $\rho^*\to1$ 时，条件数竖直地冲向无穷。两张图说的是同一件事——速率越好，条件数越接近 1；速率接近 1，条件数就爆炸。基线 $q=(0.70,0.20,0.10)$ 在两张图上对应同一个点 $(2.663,0.454)$。

---

## 23.4 Hessian：$\mu_T$ 和 $\theta_{\max}$ 从哪里来

速率只需要两个端点，但这两个端点仍然要从矩阵里算。先看一张具体的图。取 $K=6$，锚点

$$
q=(0.35,\ 0.25,\ 0.15,\ 0.12,\ 0.08,\ 0.05).
$$

![图 23.4：Hessian 的三种读法——原始矩阵、切空间投影、以及切空间谱](/figures/ch12_special_hessian_matrix.png)

### 面板 (a)：原始 Hessian——每个答案的“局部硬度”

$$
\nabla^2E(q)=\operatorname{diag}\!\left(\frac1{q_1},\ldots,\frac1{q_K}\right)
=\operatorname{diag}(2.86,\ 4.00,\ 6.67,\ 8.33,\ 12.50,\ 20.00).
$$

第 $i$ 个对角元是 $1/q_i$。答案越稀有，这个方向越陡。对角线之外全是零：如果概率可以独立变化，各个答案之间没有耦合。

但概率不能独立变化。它们必须加起来等于 1。这个约束会把非对角元“种”进去。

### 面板 (b)：切空间 Hessian——约束把答案耦合起来

投影之后，$H_T$ 的每个元素是

$$
(H_T)_{ij}
=\frac{\delta_{ij}}{q_i}
-\frac{1}{K}\left(\frac{1}{q_i}+\frac{1}{q_j}\right)
+\frac{1}{K^2}\sum_{k=1}^{K}\frac{1}{q_k}.
$$

读法有三条：

1. **每一行加起来为零。** 全 1 方向是法向，$H_T\mathbf{1}=0$。所有答案同时加同一个数，信念不变。
2. **对角元为正，总迹变小。** $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i=45.30$，小于原始迹 $54.36$。约束吃掉了一部分总曲率。
3. **非对角元大多为负，但不是全部为负。** 两个最稀有的答案之间耦合很强地为负（$q_5$ 与 $q_6$ 之间是 $-3.91$）：抬高一个，必须压低另一个。两个最常见的答案之间有一个小的正耦合（$q_1$ 与 $q_2$ 之间是 $+0.37$）。严格的陈述只有两条：行和为零，且在切空间上半正定。

一个漂亮的特例：当 $q$ 均匀时，$1/q_i=K$，于是

$$
H_T=K\,P_T=K\left(I-\frac1K\mathbf{1}\mathbf{1}^\top\right).
$$

这是完全图 $K_K$ 的图拉普拉斯矩阵，特征值是 $0$（法向）和 $K$（切向，$K-1$ 重）。均匀信念的切空间几何，就是“每个答案与每个其他答案都有一条等权边”的图。

### 面板 (c)：切空间谱——只需要两端

对这个锚点，切空间谱是

$$
\lambda=(3.32,\ 5.24,\ 7.57,\ 11.13,\ 18.03).
$$

所以

$$
\mu_T=3.32,\qquad \theta_{\max}=18.03,\qquad \kappa_T=5.43.
$$

- 最小特征值对应“在前两个最可能的答案之间重新分配质量”的慢模式；
- 最大特征值对应“修正最稀有答案”的快模式；
- 中间的三个特征值，对最优速率没有贡献。这就是引理 23.1 的实际含义。

![图 23.5：切空间谱与原始对角线的交错；只有两个端点进入速率](/figures/ch23_hessian_spectrum.png)

图 23.5 面板 (a) 把久期方程的“交错”画了出来：灰色是原始对角线 $d_i=1/q_i$，蓝色是切空间特征值。每个蓝色根都落在相邻两个灰色 level 之间，$d_{(1)}<\lambda_1<d_{(2)}<\cdots$。面板 (b) 换成 $K=6$ 的三组锚点：均匀锚点的五个特征值全相等（$\kappa_T=1$）；基线锚点给出 $\kappa_T=5.43$；接近顶点的锚点把所有特征值一起推高，但比值只到 $5.70$——切空间条件数被约束卡在 $K=6$ 附近。中间那些特征值改变的是谱的形状，不改变 $\rho^*$。

:::details 引理：切空间谱由一个标量方程决定

设 $D=\operatorname{diag}(d_1,\ldots,d_K)$，$d_i=1/q_i$，$H_T=P_TDP_T$。对 $x\in T$，$H_Tx=\lambda x$ 等价于 $P_TDx=\lambda x$。因为 $x\in T$，$P_Tx=x$，所以 $Dx-\lambda x=c\mathbf{1}$，即 $x_i=c/(d_i-\lambda)$。再用 $\sum_ix_i=0$，得到**久期方程**

$$
\sum_{i=1}^{K}\frac{1}{d_i-\lambda}=0.
$$

它的 $K-1$ 个根就是切空间谱，并与 $d_i$ 交错：

$$
d_{(1)}<\lambda_1<d_{(2)}<\cdots<\lambda_{K-1}<d_{(K)}.
$$

求 $\mu_T$ 与 $\theta_{\max}$ 只需要在最小的两个 $d_i$ 之间、最大的两个 $d_i$ 之间各二分一次。对 $K=3$，久期方程化为

$$
3s_3\lambda^2-2s_2\lambda+1=0,
\qquad
s_2=q_1q_2+q_1q_3+q_2q_3,\quad s_3=q_1q_2q_3,
$$

这正是[第12章](/volume1/chapter12/)“直接证据”一节里出现的二次方程。

:::

:::details 边界：切空间条件数饱和到 $K$

原始 Hessian 在边界上会爆炸：$q_{\min}\to0$ 时 $1/q_{\min}\to\infty$。但切空间谱被约束夹住了。取对称顶点极限

$$
q(t)=\frac{1-t}{K}\mathbf{1}+t\,e_1,\qquad t\to1.
$$

全空间条件数 $q_{\max}/q_{\min}\sim1/(1-t)$ 发散，切空间条件数却满足

$$
\kappa_T\to K.
$$

数值上，$K=5,10,50,200,1000$ 分别给出 $\kappa_T=5.0000,10.0000,50.0000,199.9998,999.9990$。原因是 $K-1$ 个稀有答案的 $d_i$ 相等，久期方程在那一段只拉出一个根 $d_{\text{rare}}/K$，其余 $K-2$ 个根重合。约束把“一个极小概率”的爆炸，摊成了“$K$ 个答案之间的耦合”。

![图 23.6：全空间条件数发散，切空间条件数饱和到 $K$](/figures/ch23_kappa_saturation.png)

图 23.6 面板 (a) 是 $K=5$ 的对称顶点极限：红线是全空间条件数，随 $t\to1$ 对数发散；蓝线是切空间条件数，稳稳地贴在 $K=5$ 上。面板 (b) 把谱按 $\mu_T$ 归一化：最小值恒为 1，其余 $K-1$ 个特征值一起升到 $K$。发散没有消失，它被约束重新分配成了“$K$ 个答案之间”的耦合。

:::

---

## 23.5 速率就是对数曲线的斜率

先把线性化模型写出来。误差递推是精确的：

$$
\|p_t-A\|_2=\rho^t\|p_0-A\|_2.
$$

两边取自然对数：

$$
\ln\|p_t-A\|_2=\ln\|p_0-A\|_2+t\ln\rho.
$$

这是 $t$ 的一次函数。**半对数坐标上，它是一条直线，斜率精确等于 $\ln\rho$。** 换成双对数坐标，它不再是直线——因为 $t=e^s$ 会把一次函数变成指数函数。所以“几何收敛在双对数上是一条直线”这句话是错的；正确的说法是“在半对数上是一条直线”。非线性轨道会有高阶项，但主导特征值给出的渐近斜率仍然是 $\ln\rho$。

![图 23.7：半对数坐标下是直线，双对数坐标下是曲线](/figures/ch23_semilog_loglog.png)

图 23.7 用同一组数据画了两次。面板 (a) 是半对数坐标：三条几何轨道都是严格的直线，斜率分别等于 $\ln\rho$。步长太小（浅蓝）斜率平缓；步长最优（深蓝）斜率最陡；步长过大（红）虽然慢模式被压得很快，却被最快模式拖住，反而更慢。面板 (b) 换成双对数坐标：同一条几何轨道弯成一条曲线，不再是直线；灰色虚线是一条真正的幂律 $t^{-1.5}$，它在双对数上才是直线。所以“log-log 直线”属于幂律，“半对数直线”才属于几何收敛。

以速率 $\rho^*$ 衰减时，到达精度 $\varepsilon$ 的连续步数是

$$
N_{\mathrm{real}}(\varepsilon)=\frac{\ln(\|p_0-A\|_2/\varepsilon)}{\ln(1/\rho^*)},
\qquad
N(\varepsilon)=\left\lceil N_{\mathrm{real}}(\varepsilon)\right\rceil.
$$

取 $\|p_0-A\|_2=1$ 记 $L=\ln(1/\rho^*)$，则 $N_{\mathrm{real}}=\ln(1/\varepsilon)/L$。下面给出 $L$ 的精确夹逼。

**定理 23.3（窗口的精确线性界）.** 对 $\kappa_T>1$，

$$
\frac{\kappa_T-1}{2}\ln\frac1\varepsilon
\ \le\ N_{\mathrm{real}}(\varepsilon)\ \le\
\frac{\kappa_T}{2}\ln\frac1\varepsilon.
$$

**证明.** 令 $x=2/(\kappa_T-1)>0$。由 $(\kappa_T+1)/(\kappa_T-1)=1+x$，有

$$
L=\ln\frac{\kappa_T+1}{\kappa_T-1}=\ln(1+x).
$$

两个标准不等式给出

$$
\frac{2x}{2+x}\le\ln(1+x)\le x.
$$

代入 $x=2/(\kappa_T-1)$：

$$
\frac{2}{\kappa_T}\le L\le\frac{2}{\kappa_T-1}.
$$

取倒数再乘 $\ln(1/\varepsilon)$，即得结论。$\square$

这两个界都是精确的，不是渐近近似。它们把窗口夹在两条斜率分别为 $(\kappa_T-1)/2$ 和 $\kappa_T/2$ 的直线之间。图 23.2 面板 (c) 画的就是这个夹逼：蓝线是精确的 $N$，阴影是两条精确界。

对基线 $q=(0.70,0.20,0.10)$：

$$
\mu_T=2.9904,\quad \theta_{\max}=7.9620,\quad \kappa_T=2.6625,
$$

$$
\eta^*=0.18261,\quad \rho^*=0.45393,\quad N_{\mathrm{real}}(0.1)=2.915,
$$

即需要 $N=\lceil2.915\rceil=3$ 步。

精确界给出

$$
\frac{2.6625-1}{2}\ln10=1.914
\ \le\ 2.915\ \le\
\frac{2.6625}{2}\ln10=3.065.
$$

窗口落在界内，并且上下界只差 $1.15$ 步。这不是“大概 3 步”，而是“精确地夹在 1.914 与 3.065 之间”。

---

## 23.6 涌现：速率变，或者只是读数变

误差曲线是 $\rho^t$。如果 $\rho$ 不变，这条曲线就是精确的几何曲线，没有拐点。

所以，一个“突然会了”的拐点只有两种可能：

1. **读数变了，速率没变。** 指标是非线性的。比如准确率是 $\mathbf{1}[\text{误差}<\varepsilon]$，一条光滑的误差曲线穿过阈值时，准确率会看起来像阶跃。Schaeffer 等人（2023）系统地论证了这类“涌现”可以完全由度量选择制造出来。
2. **速率真的变了。** 这只能通过 $\mu_T$ 或 $\theta_{\max}$ 的变化发生：一个新的慢模式进入谱的低端，或者一个新的快模式进入高端。此时 $\rho^*$ 本身改变，误差曲线真的会弯。

用本章的标量，诊断规则只有一条：

> 先测 $\rho^*$ 随规模的变化。如果 $\rho^*$ 不变，而 benchmark 跳了，那是读数；如果 $\rho^*$ 跳了，那才是动力学意义上的涌现。

![图 23.8：两种“涌现”——读数跳变，或斜率真的改变](/figures/ch23_emergence.png)

图 23.8 把判据画出来。面板 (a) 中 loss 在 log-log 上是一条直线，速率没有变；但准确率是一个阈值指标，曲线穿过阈值时看起来像阶跃。面板 (b) 中 $\rho^*$ 本身从 $0.82$ 降到 $0.45$，半对数误差曲线的斜率变了——同样的步数，误差差了几个数量级。前者是读数，后者是动力学。

这比“看曲线像不像涌现”要硬。它把“涌现”从一个形容词，变成了一个可以测量的谱事件。Wei 等人（2022）描述的涌现现象是真实存在的经验事实；本章只是给了一个判据，去区分其中哪些来自速率，哪些来自读出。

:::details 一个必须承认的边界：本章没有定义“智能”
“智能与涌现的对数曲线”这个标题容易被读成“本章解释了智能”。它没有。

本章解释的是一类学习动力学：当一个系统以最小化 KL 为目标、以概率单纯形为输出空间时，它的误差曲线服从什么数学。它不回答“什么叫理解”“什么叫意识”“为什么某些能力只在足够大的模型上出现”。那些问题需要别的工具，也可能需要目前还不存在的工具。

把“永霖推论”当成智能的定义，是错误的；把它当成智能系统必须偿还的一笔几何账，是合适的。
:::

---

## 23.7 结语：一个数

整章只有一个数：

$$
\rho^*=\frac{\kappa_T-1}{\kappa_T+1},
\qquad
\kappa_T=\frac{\theta_{\max}}{\mu_T},
\qquad
\kappa_T=\frac{1+\rho^*}{1-\rho^*}.
$$

它精确地决定四件事：

1. **斜率。** 半对数误差曲线的斜率是 $\ln\rho^*$。
2. **步数。** 到达 $\varepsilon$ 的步数精确地夹在 $\frac{\kappa_T-1}{2}\ln\frac1\varepsilon$ 与 $\frac{\kappa_T}{2}\ln\frac1\varepsilon$ 之间。
3. **稳定性。** 稳定窗口是 $0<\eta<2/\theta_{\max}$，最优步长是 $\eta^*=2/(\mu_T+\theta_{\max})$。
4. **涌现的判据。** $\rho^*$ 不变而指标跳变，是读数；$\rho^*$ 跳变，才是动力学。

[第12章](/volume1/chapter12/)说：推理会收敛到先验锚点 $A$，而不是真实答案 $A^*$。本章补上了“多快”：

> 多快，由一个条件数决定；条件数由切空间 Hessian 的两个端点决定；两个端点由矩阵决定。除此之外，没有第二个标量。

“Scaling Law 不是玄学”这句话现在可以立住了，而且不需要等某个私人模型来作证。GPT 能把一条 loss 曲线压到 2.5，一个小 FFN 也许只能压到 9：它们的具体下界不同，但共享同一条 log-log 曲线。模型只是往下探的工具，不是曲线的发明者。只要一个系统做了单纯形投影、在单纯形空间里运动，并以最小熵（KL）为目标训练，它就会呈现这套几何判据。换句话说，**任何满足这三个条件的架构，都能出示同一张“几何判据证书”；架构决定的是证书上的数值下界，而不是证书的格式。**

---

## 悬而未决

**镜像下降的精确速率是什么？** 本章的压缩率建立在欧氏投影梯度上。真实模型更接近 KL 镜像下降，而 KL 不是度量，不能直接套巴拿赫。是否存在一个 Bregman 版本的谱半径公式？如果有，它是否仍然只依赖两个端点？这是开放的。

**能不能在大规模模型上直接测 $\mu_T$ 和 $\theta_{\max}$？** 本章的谱是在 $K=3$ 到 $2048$ 的显式分布上算的。真实 LLM 的词表很大，Hessian 无法显式存储。随机迹估计、Lanczos、隐式微分能不能给出可靠的端点估计？这会把“涌现诊断”从理论变成实验。

**$\rho^*$ 什么时候随规模改变？** 本章给出了判据，但没有给出 $\rho^*(N)$ 的模型。如果慢模式的出现有临界规模，那个临界点由什么决定？这是“涌现”真正需要回答的问题。

---

## 思考题

**★ 热身**

1. 对 $K=3$、$q=(1/3,1/3,1/3)$，验证 $H_T=3P_T$，写出它的全部特征值，并算出 $\rho^*$。
2. 对 $q=(0.70,0.20,0.10)$，用久期方程算出 $\mu_T,\theta_{\max},\kappa_T$，再算 $\eta^*,\rho^*$ 与 $N(0.1)$。

**★★ 推导**

1. 证明引理 23.1：凸函数 $g(\lambda)=|1-\eta\lambda|$ 在区间上的最大值在端点取得。
2. 证明 $\operatorname{tr}(H_T)=(1-1/K)\sum_i1/q_i$。
3. 证明定理 23.2：$\eta^*=2/(\mu_T+\theta_{\max})$ 是唯一最小值点，$\rho^*=(\kappa_T-1)/(\kappa_T+1)$，且两个端点分支在最优点的值相等。
4. 证明定理 23.3 的两个界：$\frac{2x}{2+x}\le\ln(1+x)\le x$。

**★★★ 挑战**

1. 对 $K=3$，把久期方程写成 $3s_3\lambda^2-2s_2\lambda+1=0$，并验证它与[第12章](/volume1/chapter12/)的二次方程一致。
2. 设计一个实验，只通过测量 $\rho^*$ 随规模的变化，区分“读出效应”和“谱分岔”。说明每种情况下你会看到什么，以及如何排除混淆变量。
3. 尝试构造一个反例，说明在 KL 镜像下降下，欧氏压缩率 $\rho^*$ 不再是精确速率；再说明需要加什么条件才能恢复一个精确的速率公式。

---

## 附录：核心计算脚本

下面这段代码只做一件事：给定锚点 $q$，用久期方程算切空间谱的两个端点，再算精确速率、最优步长和窗口。它不训练任何模型。

```python
import numpy as np

def tangent_spectrum(q, iters=90):
    """Eigenvalues of P_T diag(1/q) P_T on T={sum=0}, via secular equation."""
    d = np.sort(1.0 / np.asarray(q, float))
    K = len(d)
    roots = np.empty(K - 1)
    for j in range(K - 1):
        lo, hi = d[j], d[j + 1]          # one root in each interval
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = np.sum(1.0 / (d - mid))  # sum 1/(d_i - lambda)
            if f > 0: hi = mid
            else:     lo = mid
        roots[j] = 0.5 * (lo + hi)
    return roots


def rate(q, eps=0.1, e0=1.0):
    lam = tangent_spectrum(q)
    mu, theta = lam.min(), lam.max()
    kappa = theta / mu
    eta_star = 2.0 / (mu + theta)
    rho_star = (kappa - 1.0) / (kappa + 1.0)
    n_real = np.log(e0 / eps) / np.log(1.0 / rho_star)
    n_steps = np.ceil(n_real)
    lo = (kappa - 1.0) / 2.0 * np.log(e0 / eps)
    hi = kappa / 2.0 * np.log(e0 / eps)
    return dict(mu_T=mu, theta_max=theta, kappa_T=kappa,
                eta_star=eta_star, rho_star=rho_star,
                n_real=n_real, n_steps=n_steps, n_lo=lo, n_hi=hi)


print(rate([0.70, 0.20, 0.10]))
# mu_T=2.9904, theta_max=7.9620, kappa_T=2.6625,
# eta_star=0.18261, rho_star=0.45393, n_real=2.915, n_steps=3,
# n_lo=1.914, n_hi=3.065

print(rate([0.35, 0.25, 0.15, 0.12, 0.08, 0.05]))
# kappa_T=5.4268
```

---

**参考文献**

- Banach, Stefan (1922). “Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.” *Fundamenta Mathematicae* 3(1), 133–181. DOI: 10.4064/fm-3-1-133-181.——压缩映射原理的原始文献
- Schaeffer, Rylan; Miranda, Brando; and Koyejo, Sanmi (2023). “Are Emergent Abilities of Large Language Models a Mirage?” arXiv:2304.15004.——涌现的度量伪影
- Wei, Jason; Tay, Yi; Bommasani, Rishi; Raffel, Colin; Zoph, Barret; Borgeaud, Sebastian; Yogatama, Dani; Bosma, Maarten; Zhou, Denny; Metzler, Donald; Chi, Ed H.; Hashimoto, Tatsunori; Vinyals, Oriol; Liang, Percy; Dean, Jeff; and Fedus, William (2022). “Emergent Abilities of Large Language Models.” arXiv:2206.07682.——涌现现象的系统描述
- Amari, Shun-ichi (2016). *Information Geometry and Its Applications*. Applied Mathematical Sciences 194. Springer Japan. DOI: 10.1007/978-4-431-55978-8.——Bregman 几何、Fisher 信息与自然梯度
- Nesterov, Yurii (2018). *Lectures on Convex Optimization*, 2nd edition. Springer Optimization and Its Applications 137. Springer. DOI: 10.1007/978-3-319-91578-4.——强凸、光滑、步长与条件数
- Boyd, Stephen; and Vandenberghe, Lieven (2004). *Convex Optimization*. Cambridge University Press. DOI: 10.1017/CBO9780511804441.——投影的非扩张性与收敛率
- 本书[第12章](/volume1/chapter12/)，永霖公式——永霖定理的全局版本、切空间谱界与直接证据

---

正文到这里结束。后面是写给读者的[跋文](/volume2/afterword)。

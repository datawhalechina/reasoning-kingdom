# 第13章：推理的边界——以及我们为什么必须接受它

> 图灵问：机器能思考吗？更难的问题是：我们能知道它们真的在思考吗？

<div class="center">

------------------------------------------------------------------------

</div>

## 一、边界确实存在

这本书从第一章的熵讲起，一路走过符号系统、词向量、神经网络、Transformer、搜索和复杂度。进步是真的，限制也是真的。

现在问一个更深的问题：**推理本身有没有边界？**

有。而且它不来自工程没做好，它是逻辑上的必然。

第七章的停机问题：有些问题根本判不了。扔给一个程序一个输入，没有算法能断定它会不会停。

第十二章的永霖公式：推理链最后收回到先验锚点，对象层闭合，元层断开。

两个结果指向同一件事：任何足够强的推理系统，都兜着自己解不了的问题。

数学真理，不是失败。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、哥德尔的回声

1931年，哥德尔证明了一个定理，震得数学界半天没缓过神：任何足够强的形式系统，都含有自己证不了的真命题。

往简单里说。

假设有个形式系统 $S$，它有两样东西：

- 一组公理，系统点头承认的起点
- 一组推理规则，规定怎么从公理推出新命题

系统还得够强。强到能算基本算术，强到能谈“证明”这回事。到了这一步，哥德尔就能在里面造出一个命题 $G$：

$$
G = \text{“G 在系统 S 中不可证明”}
$$

问一句：$G$ 是真的吗？

**情况1**：$G$ 可证明。那 $G$ 说的“我不可证明”就是假话。一个靠得住的形式系统证出了假命题，系统塌了。

**情况2**：$G$ 不可证明。那 $G$ 说的就是真话。$G$ 真，系统证不出来。

结论：有一个真命题 $G$，系统 $S$ 证不了它。

把它念一遍。

“命题 $G$ 在系统 $S$ 里不可证明。”它可证，这句话就是撒谎；它不可证，这句话就是真话。它不描述世界，它描述自己。你判断它是真是假的那一刻，已经掉进它挖的坑。

逻辑没出错，是自指要收的账。任何足够强的系统都带着这样一个坑：“足够强”的意思就是，它不只会谈论对象，还会谈论自己怎么谈论对象。

### 为什么“足够强”会带来自指？

哥德尔不只是写下一个悖论。他把“证明”编码成算术。命题、证明序列、推理规则，全都编上号，变成自然数之间的关系。一个本来只会谈加法和乘法的系统，忽然能谈“这个命题能不能被证明”。

危险在这儿。系统一旦能编码自己的语法，就能造出指向自己的命题。自指不是外加的功能，是表达力够强之后长出来的副作用。

所以不完备不是“系统还不够强”。正相反：系统越强，越躲不开自己的影子。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、哥德尔的边界：不完备性的形状

哥德尔不完备定理最要紧的地方，不在“有命题证不出来”这个结论。在它露出的结构：系统能在内部自洽地生成无穷多个合法步骤，却没法从内部担保，这些步骤走到了系统之外的真理。

一个形式系统可以写成一个三元组：

$$
S = (\mathcal{A}, \mathcal{R}, \vdash)
$$

$\mathcal{A}$ 是公理集合，$\mathcal{R}$ 是推理规则，$\vdash$ 是可证明关系。$G$ 的诡异在哪儿？它把系统的元层问题塞回对象层。表面上看是个普通命题，实际在问“这个系统能不能证明我”。

由此有三个后果：

**后果一：自我验证有天花板。** 一个系统能用 $S$ 证出许多定理。可你要它从内部担保“真理由此到齐了”，它就栽了。这不是它的毛病，是每个足够强的系统共有的形状。

**后果二：内部自洽不等于外部为真。** 系统能在边界内生出无穷多个看起来对的证明，然后把这份自洽当成世界的真相。真正危险的不是犯错，是它没法用自己证明自己对。

**后果三：边界之内，活照样干。** 系统给自己担保不了，外面就得有人看着：机器证明检查器、独立验证器、人工审查。完备性不是系统一个人的事，是人和系统分着干的事。

:::details 技术细节：形式系统的抽象足够宽敞，可以容纳一个推理系统
把一个推理系统摆到 $S = (\mathcal{A}, \mathcal{R}, \vdash)$ 旁边：

- $\mathcal{A}$：系统认账的起点——公理，或者训练分布和先验
- $\mathcal{R}$：合法的推导规则——推理规则，或者隐状态的转移规则
- $\vdash$：可证明关系——一个推导步骤收不收

这么摆的好处：谈“系统边界”不必留在数学里，也不用假定推理系统和数学系统是同一种东西。哥德尔谈的是形式系统，不是神经网络。他的教训是结构性的：任何靠自己的规则生成合法步骤的系统，都没法用自己证明自己完备。
:::

:::details 一个容易混淆的点：$G$ 和 $A$ 不是一回事
哥德尔命题 $G$ 把系统的证明能力变成一个内部对象，逼系统回答：“我能不能证明我证不了的东西？”

推理模型的先验锚点 $A$ 是另一回事。模型不会说“我跳不出训练分布”。它给出一个看起来合理、读起来顺、内部自洽的答案。那个答案就是元层限制投到对象层的影子。

两句话像，机制不一样。$G$ 的不可证明，来自形式系统的自指能力；$A$ 的收敛，来自推理动力学的吸引域。哥德尔和永霖各管各的，一个推不出另一个。

能带走的一句：任何足够强的推理系统，都把自己的极限编进结构里。哥德尔只是把这句话推得最远、最准的那一个。
:::

<div class="center">

------------------------------------------------------------------------

</div>

## 四、动手试试：构造一个不可判定问题

让我们亲手体验停机问题的不可判定性。

**步骤1：假设存在一个停机判定器**

**步骤2：构造对角化程序**

下面的 Python 让这个思想实验变得具体。我们先定义一个“假想的”停机判定器（现实中无法精确实现），再构造把它逼入矛盾的对角化程序，最后用穷举模拟验证矛盾必然发生。

```python
import sys
import threading

# ------------------------------------------------------------------
# Step 1: simulate an "approximate halting checker" with a timeout
#   A true HALT cannot exist; here we approximate it with a finite timeout:
#   if the program returns within `timeout` seconds, we judge it "halts",
#   otherwise we judge it "does not halt".
# ------------------------------------------------------------------
def approximate_halt(func, timeout=0.05):
    """
    Approximate whether func() halts within `timeout` seconds.
    Returns True for "judged halting", False for "judged non-halting".
    """
    result = [None]

    def runner():
        try:
            func()
            result[0] = True   # function returned normally -> halts
        except Exception:
            result[0] = True   # raising an exception also counts as halting

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    # still running after the timeout -> judged as "does not halt"
    return result[0] is True

# ------------------------------------------------------------------
# Step 2: construct the diagonalization program DIAG
#   The logic of DIAG:
#     If HALT(DIAG) returns True (judges that DIAG will halt)
#         -> DIAG enters an infinite loop (actually does not halt)
#     Otherwise (judged not to halt)
#         -> DIAG returns immediately (actually halts)
# ------------------------------------------------------------------
def make_diag(halt_checker):
    """Factory function: build a diagonalization program from the given halt_checker"""
    def diag():
        # First ask the "checker": will I (diag) halt?
        will_halt = halt_checker(diag)
        if will_halt:
            # The checker says I will halt -> I stubbornly loop forever
            while True:
                pass
        else:
            # The checker says I will not halt -> I stubbornly return at once
            return

    return diag

# ------------------------------------------------------------------
# Step 3: ask whether DIAG(DIAG) will halt, revealing the contradiction
# ------------------------------------------------------------------
print("=" * 55)
print("Undecidability of the halting problem: the diagonalization argument")
print("=" * 55)

diag = make_diag(approximate_halt)

# First predict with the checker
prediction = approximate_halt(diag)
print(f"\nChecker prediction: DIAG will {'halt' if prediction else 'not halt'}")

# Then actually observe DIAG's behavior using a timeout
actual_halted = approximate_halt(diag, timeout=0.1)
print(f"Actual observation: DIAG {'halted' if actual_halted else 'did not halt (timeout)'}")

# Check whether a contradiction arises
if prediction != actual_halted:
    print("\n-> Contradiction! The checker's prediction is the opposite of DIAG's actual behavior.")
else:
    # Note: the timeout approximation may make the two observations
    # "coincidentally" agree; this is only the error of an approximate
    # checker and does not affect the rigorous mathematical proof.
    print("\n-> Note: the timeout approximation may mask the contradiction, but the rigorous proof")
    print("   guarantees the exact HALT inevitably leads to a contradiction (see the analysis below).")

print("\n--- Rigorous logical analysis ---")
print("Case A: HALT(DIAG, DIAG) = True  -> DIAG loops forever -> actually does not halt -> contradiction!")
print("Case B: HALT(DIAG, DIAG) = False -> DIAG returns at once -> actually halts   -> contradiction!")
print("\nConclusion: an exact halting checker HALT is logically impossible.")
print("      This is not a matter of algorithmic cleverness, but a logical necessity.")
```

**步骤3：询问 DIAG(DIAG)**

现在问：DIAG(DIAG) 会停机吗？

- **如果 HALT(DIAG, DIAG) 返回 True**：
  - 也就是说 DIAG(DIAG) 会停机
  - 但按照 DIAG 的定义，它会进入无限循环
  - 矛盾！
- **如果 HALT(DIAG, DIAG) 返回 False**：
  - 也就是说 DIAG(DIAG) 不停机
  - 但按照 DIAG 的定义，它会立即返回（停机）
  - 矛盾！

**结论**：HALT 不可能存在。

**这个矛盾告诉我们什么？**

不是我们还没有找到足够聪明的算法，而是**没有任何算法能解决停机问题**。这是逻辑必然，不是技术限制。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、推理王国的地图

退后一步。

把前面十三章摊开。第一章讲熵，讲世界为什么需要预测。第二章讲符号，讲人怎么把预测写成能操作的规则。第三章到第六章，神经网络把符号压成向量，把向量推成状态，再把状态织成 Transformer 的注意力流。第七章讲复杂度：不是每次搜索都能漂亮收场。第八章讲启发式：精确拿不到，就得有代价地妥协。第九章到第十一章，注意力、搜索、随机化各把边界往前推了一点，边界本身没动。第十二章的永霖公式收尾：连“继续推理”都有收敛的尽头。

所有这些，落在一张地图上。

**中心：可判定问题。**

这里是王国最亮的城区。排序、最短路、动态规划、线性代数，还有一堆工程优化问题住在这儿。不见得容易，路是清楚的：给了输入，算法会停；答案能查；复杂度能估。

P 类是中心的主干道，多项式时间可解。NP 类是更硬的街区：答案好验，找起来可能贵得离谱。第七章的 $\mu(L, d)$ 相图说，NP 内部也不是每条街一样黑。可解不是开关，是一整片连续的地形。

**雾带：相变区域。** 参数一过阈值，行为就骤变，这叫相变。

再往外，是雾。

OpenXOR 的 $\alpha \approx 4.26$ 是一片典型雾带：实例从“几乎总能解”翻脸成“几乎总解不了”。Collins 的 $c_{\text{opt}} \approx 34$ 也在这里：压缩从安全走成危险。CoT 的有效推理窗口同样是雾带：推理从有产出的展开，滑成对先验的复读。

雾带最要紧的一点：它不是墙。你不能干脆划一句“这边可解，那边不可解”。在雾里，参数动一点，后果差很多。多一个约束，多一点压缩，多走几步推理，任何一样都可能让系统从清醒滑进幻觉。

**外圈：不可判定，不完备。**

再往外，王国的边疆就黑了。

停机问题：有些问题没有通用判定算法。哥德尔命题：有些真命题在系统内部证不出来。永霖公式：有些推理链在模型内部完成不了元层自证。三件事各是自己的边界，不是同一件事的三种说法。

它们不是同一条定理，却朝一个方向：系统一旦强到能表达自己、模拟自己、回头反思自己，就一定会生出自己的盲点。

:::details 三种边界的区别
- **复杂度边界**：解得出来，可能太慢。例子：NP 难搜索。
- **相变边界**：不是突然解不了，是在某段参数里陡然变难。例子：3-SAT 的 $\alpha \approx 4.26$。
- **逻辑边界**：不是慢，是根本没有通用算法。例子：停机问题和哥德尔不完备。

把这三种边界搅在一起，会犯两种错：把逻辑上的不可能当成努力不够，或者把工程上的难处过早说成哲学的命。
:::

地图上，推理的边界不是一条线，是一片有梯度的雾。雾的尽头没有更大的模型，只有逻辑自己的地平线。

上卷想说的就是这个。推理不是一架一直往上爬的梯子，是在三片地形里找路：可判定的、相变的、不完备的。知道自己站在哪儿，比闷头冲管用。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、伪代码：边界的形式化

**算法1：停机问题的对角化证明**

```text
Halting-problem undecidability proof:

Assumption: a decider HALT(P, x) exists

Construct the diagonal program:
def DIAG(P):
  if HALT(P, P):
    loop_forever()   # infinite loop
  else:
    return           # halt

Ask: HALT(DIAG, DIAG) = ?

Case 1: HALT(DIAG, DIAG) = True
  -> DIAG(DIAG) halts
  -> but DIAG definition: if HALT returns True, loop forever
  -> contradiction!

Case 2: HALT(DIAG, DIAG) = False
  -> DIAG(DIAG) does not halt
  -> but DIAG definition: if HALT returns False, halt
  -> contradiction!

Conclusion: HALT does not exist
```

![对角化论证：停机问题不可判定性的可视化](/figures/ch15_fig1_diagonalization.png)

*图1：对角化论证的矩阵视角。行是程序 P，列是输入 x，每个格子记录 HALT(P, x) 的结果（H = 停机，L = 循环）。对角线上的格子 HALT(P, P) 定义了 DIAG：对角线为 H 的地方，DIAG 循环；为 L 的地方，DIAG 停机。DIAG 必然出现在某一行中——但那一行在对角线上的值与 DIAG 的定义矛盾。这个矛盾是逻辑的必然，不是技术的限制。*

**算法2：永霖公式的元层断裂**

```text
Meta-layer break detection (AI system S, reasoning task T):

Input:  AI system, reasoning task
Output: object-level conclusion, meta-layer limit

1. Object-level reasoning:
   C = S.reason(T)

2. Meta-layer verification:
   ask: "can S prove C is correct?"

3. Incompleteness detection:
   if S cannot verify C at the meta layer:
     # there exists a true proposition C* that is true at
     # the object layer but unprovable at the meta layer
     return (C, "meta-layer break")

4. Return (C, "object layer closed, meta layer broken")
```

*注：步骤3中的“存在”不是哥德尔的证明，而是永霖公式的直接推论——模型用产生推理的同一套参数来验证推理，所以一旦锚点 $A$ 不是 $A^*$，验证就会永久错位。*

<div class="center">

------------------------------------------------------------------------

</div>

## 七、推理的未来：在边界之内能走多远

承认边界，不等于认输。边界反倒指出了方向：在边界之内，能走多远？

---

### 7.1 从不可能到“概率上可能”

永霖公式说的是元层断裂：模型验证自己的推理，用的是产生那套推理的同一批参数，跳不出去。

但不是非黑即白。不是“完全行”和“完全不行”，是一条概率谱：

- **短链**（3–5 步）：元层断裂几乎不碍事，模型能自己纠错
- **中链**（10–20 步）：误差开始攒起来，需要外部校准点
- **长链**（50 步以上）：先验锚点占上风，自我纠错快没用了

对系统设计的意思很直接：别让模型一个人跑超过有效窗口的长链。打断、校准、重启，这些补偿少不了。

---

### 7.2 外部验证器：打破元层封闭

永霖公式的根子在自指：拿推理验证推理。

跳出这个圈，得请一个**独立的外部验证器**：

- **形式化验证**：Lean、Coq 这类机器证明。模型写草稿，形式系统一步步独立核对。验证器和模型不共享参数，断裂靠外力接上。
- **符号执行**：代码推理里，让解释器真的去跑模型写的代码，拿运行结果当验证信号。
- **多模型辩论**：两个各训各的模型互挑毛病，系统性的偏差就被压下去了。

它们走的是同一条路：把元层从模型肚子里搬到外面。内外之间没有参数共享，自指的环给切断了。

这个缺口有实测的样子。在真实模型上，唯一推得动信念的力量来自外面。一份独立检测器的报告能改变模型的信念（第 14.7 节：检测器可靠度为 0.90 时，模型对贝叶斯后验的追踪中位 KL 是 0.02）。模型自己把信念重读 128 步，$P(\text{YES})$ 冻在锚点 0.485（第 14.8 节）。外部检查点做得到内部动力学做不到的事。断裂不是修不了，它要一股外力来修。

---

### 7.3 扩展上下文：更长的有效推理窗口

有效推理窗口 $t_{\text{conv}}$ 不是定死的。它看模型多大、任务什么结构、怎么训。

这几年几个方向：

- **长程 CoT 训练**：DeepSeek-R1 的 GRPO 发现，组内一竞争，模型自己就写出更长、结构更整的推理链。窗口在训练中被拉长了。
- **记忆增强**：把中间步骤挪到外部存储，工具调用、向量库都行，模型不必把中间状态全揣在激活值里。
- **递归精炼**：同一道题反复推，每次都拿上一次的输出当新起点，像数值方法里的迭代收敛。

---

### 7.4 接受边界，才是设计的起点

停机问题：没有通用的判定算法。P≠NP 要是成立，就没有通用的高效搜索。永霖公式：没有完整的自我验证。

三条边界围出一个设计空间：

| 边界 | 工程补偿 |
|:-----|:--------|
| 停机问题不可判定 | 有界计算、超时终止 |
| NP 难搜索 | 启发式、近似算法、随机化 |
| 元层断裂 | 外部验证器、工具调用、多模型辩论 |

这本书里每个工具，A\*、MCTS、ADS、Collins、Transformer，都是某条边界之内的工程补偿。它们不破边界，只在边界里找最好的活法。

推理最诚实的定义大概就是这个：在约束之内找最好的路径。

## 八、接受边界，而不是否认边界

这本书绕来绕去，核心就一句：推理有边界，这些边界是根本的。

这听着悲观。其实松绑了。

认下边界，就不追不可能的完美了：

- 不再指望 AI 把所有问题解决掉
- 不再相信更大的模型能抹掉所有限制
- 不再把不完备当成毛病

换个问法，问点有用的：

- 在边界之内，能做多好？
- 怎么认出一个问题在界内还是界外？
- 不确定的时候，怎么选得最好？

第七章的 $\mu(\alpha)$ 相图：可解不是二元，是概率。相变点附近，一个问题悬在可解和不可解中间。

第八章的启发式：最优拿不到，就要“足够好”。难的是还得量出这个“足够好”到底多好。

第十章的 MCTS：推理不一定要“理解”，要的是有效的搜索。

第十一章的 Collins：随机化能绕开确定性的下界，拿概率保证换效率。

第十二章的永霖公式：CoT 值钱的地方在收敛之前的有效窗口，不在总步数。

这些都是边界之内的智慧。

这里有坑。

用启发式、近似、外部验证器在界内做优化，本身也可能变成另一种“物的傲慢”：把边界当成要征服的工程问题，而不是要认的逻辑事实。技术系统最会的一件事，就是把什么问题都改写成“这儿还能不能再优化一点”。可有些地方，问题不在优化得不够，在忘了问：为了什么优化？

推理真正的边界不在技术，在目的。推理为了什么？只为准确率，那在界内接着优化就够了。为了理解，为了人看懂世界、看懂自己、看懂彼此，边界就不是墙，是路标。它说：到这儿为止，再往前不是推理的地盘。

剩下的是幽默感、审美、道德勇气、爱的能力。算法学不像，也不该被算法吞掉。

<div class="center">

------------------------------------------------------------------------

</div>

## 九、边界即可能性

边界更像起点，不像终点。

知道有些问题判不了，就不必把时间花在找完美算法上，转身去用近似、启发式、概率保证。

知道推理会收到先验锚点，就别盲目加步数，去优化有效窗口。

知道全局感受野和线性复杂度不能兼得，就别追一个不存在的架构，在权衡里找最优点。

边界给可能性划了地方。

界内，地方大得很：

- 更好的启发函数
- 更快的搜索算法
- 更紧的近似保证
- 更巧的资源分配

界外，是更深的理论问题：

- 边界推得动吗？
- 不同问题的边界，共不共享一套数学结构？
- 人的推理受不受同样的边界管？

<div class="center">

------------------------------------------------------------------------

</div>

## 悬而未决

这本书从悬着的问题开头，也用悬着的问题收尾。

- **推理的边界是死的，还是能推的？** P vs NP 证得出来吗？还是它自己就不可判定？
- **能不能造一个“足够弱”的系统，绕开不完备？** 系统不够强，就撞不上哥德尔命题。可这样的系统还有用吗？
- **认了极限之后干什么？** 在界内优化，还是试着推边界？
- **人的推理，边界在哪？** 我们受不受永霖公式管？受的话，怎么翻过去？
- **不确定性是毛病，还是本来就这样？** 也许推理就是拿不完整的信息做最好的猜测，本来就不求绝对真理。

这些问题没有答案。问出来，本身就是推理在干活。

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- 本书第12章，永霖公式——推理不完备性的理论证明（含推导与脚本）
- Gödel, K. (1931). *Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I*——不完备定理的原始论文
- Turing, A. (1936). *On Computable Numbers, with an Application to the Entscheidungsproblem*——停机问题与可计算性
- Turing, A. (1950). *Computing Machinery and Intelligence*——图灵测试：机器能思考吗？
- Penrose, R. (1989). *The Emperor's New Mind*——哥德尔定理与人工智能的哲学讨论
- \[Hamkins & Nenu, 2024\]——停机问题的历史澄清 `-> [arXiv:2310.07927]`
- Hofstadter, D. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid*——关于自指、递归与意识的经典著作

<div class="center">

------------------------------------------------------------------------

</div>

## 尾声

推理王国的边界不是墙，是一条地平线。

站上去看，前面不是尽头，是更宽的未知。

这本书想画出它的轮廓：从熵和预测到符号，从词向量到神经网络，从 Transformer 到复杂度，从搜索到推理，最后到边界。

地平线总在前面。

一个答案带出新问题。一条边界底下，往往还有更深的结构。

推理的旅程没有终点，只有越看越深。

迷人的地方就在这儿。

祝你好运。

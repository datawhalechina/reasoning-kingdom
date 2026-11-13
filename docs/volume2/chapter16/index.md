# 第16章：线性逻辑与资源——每个假设只能用一次

> 经典逻辑默认资源是无限的。这是一个谎言，而且是一个代价高昂的谎言。

---

第15章留下了一个令人不安的结论：只要系统足够强，就一定存在它无法证明的真命题；系统本身也无法担保自己的一致性。这是哥德尔的礼物——或者说，他的诅咒。

但“足够强”到底是什么意思？强从哪里来？

一个形式系统之所以强大，部分原因在于它的假设可以被任意重复使用。你知道 $P$，就可以把 $P$ 用上一百次——它不会磨损，不会耗尽，永远在那里等着你。这个能力对应第14章里悄悄列出的一条结构规则：**收缩**。收缩说的是：一份假设和两份假设，在推断中是等价的。多出来的那一份可以折叠掉。

这条规则看起来是常识。但它是可以质疑的。

1987 年，让-伊夫·吉拉尔（Jean-Yves Girard）问了一个简单的问题：如果拿掉收缩规则，会发生什么？

发生的事情远比他最初预期的要多。他得到的不只是一个削弱版的经典逻辑，而是一个全新的逻辑——**线性逻辑**——它描述的是一个资源会被消耗的世界。

---

## 16.0 一次性厨房游戏：每种食材只能用一次

经典逻辑像一个奇怪的厨房：你只有一枚鸡蛋，却可以同时煎出一百个蛋卷。因为在经典逻辑里，假设不会被消耗。你把 $A$ 用了一次，$A$ 还在；再用一次，它还在。这对“知识”说得通，对“资源”却荒谬。

所以我们换一个厨房。

桌上摆着几张食材卡：鸡蛋 $A$、面粉 $B$、黄油 $C$。每条菜谱都是一条线性蕴含，例如

$$A \otimes B \multimap D$$

意思是：花掉一份 $A$ 和一份 $B$，得到一道菜 $D$。一旦花掉，食材卡就从桌面上消失。你不能凭空变出一枚鸡蛋，也不能把已经用掉的黄油拿回来——除非某张卡上带着一个特殊标记 $!A$，表示它不是普通资源，而是可以反复调用的知识。

这就是线性逻辑的游戏感：推理不再只是“从真到真”，而是“在资源预算之内完成一次转换”。

:::details 这个游戏的形式化骨架
- **状态空间**：资源的多重集，例如 $\Gamma = \{A, A, B, !C\}$。重复出现的 $A$ 是两份真实资源，不是一个符号写了两遍。
- **合法走法**：按线性规则消耗资源、创造新资源；只有 $!A$ 可以被复制或丢弃。
- **转移规则**：如果有一条菜谱 $A \otimes B \multimap D$，那么状态 $\Gamma, A, B$ 可以转移到 $\Gamma, D$。
- **胜利条件**：用有限资源构造出目标资源 $G$。
- **失败模式**：目标看似可推导，但你在中途悄悄复用了某个已经被花掉的假设。
:::

这个游戏迫使读者看见经典逻辑藏起来的东西：**推理往往带有物理代价**。在程序里，内存不是无限的；在通信协议里，已经发出的消息不能同时留在原处；在量子系统里，未知量子态不能被任意克隆。

线性逻辑不是经典逻辑的怪异变体。它是一个提醒：经典逻辑预设了一个资源多得离谱的世界，在那里资源可以免费复制。真实世界没有这么慷慨。

::: info 兔狲教授评
一次性厨房游戏的规则很简单：用过就是用过了。许多抽象逻辑一落到现实里，就会迎头撞上这句话。你不能用一张五美元钞票买两杯咖啡，也不能把两个从不释放内存的模型塞进同一块显存。经典逻辑忘了这一点；线性逻辑把账单递回给你。
:::

## 16.0.5 用代码玩这个游戏

这个厨房小到可以只用几行代码跑起来。下面的程序把桌面保存为一个不可变的卡片元组，按照菜谱允许的方式做菜，并且在你试图复用已经花掉的食材时，公开地拒绝你。那次拒绝就是整个游戏。

```python
#!/usr/bin/env python3
"""
one_shot_kitchen.py -- the One-Shot Kitchen Game of Chapter 16.
Book reference: The Reasoning Kingdom, Volume 2, Chapter 16, Section 16.0
"The One-Shot Kitchen Game: Each Ingredient Can Be Used Only Once."
Classical logic is a strange kitchen: one egg, a hundred omelets at once,
because assumptions are never consumed. Linear logic is the kitchen where
an ingredient card vanishes the moment a recipe spends it. This program
runs that kitchen:
  * the table is a multiset of ingredient cards (an immutable sorted tuple),
  * a recipe consumes its inputs and produces one dish,
  * a card marked !C is knowledge, not a resource: it stays on the table,
  * the kitchen refuses to cook with a card that is already spent.
That refusal is the whole point. The failure mode of Section 16.0 is
"somewhere along the way you silently reused an assumption that had
already been spent." Here the reuse is not silent: it is a visible error.
Run:  python3 one_shot_kitchen.py
"""
# A recipe: (needed cards, dish produced, cards that are reusable knowledge)
# A x B -o D  : spend one egg and one flour, get one dish D.
# !C -o cup   : knowledge C can be called on again and again.
RECIPES = (
    (("A", "B"), "D", frozenset()),
    (("!C",), "cup", frozenset({"!C"})),
)

def cook(table, recipe):
    """Apply one recipe to the table if every needed card is present.
    Returns (new_table, dish), or None when a needed card is already
    spent. Pure: the old table is never touched.
    """
    needed, produced, reusable = recipe
    left = list(table)
    for card in needed:
        if card in reusable:
            continue                # knowledge is not consumed
        if card not in left:
            return None             # the card was spent earlier
        left.remove(card)
    left.append(produced)
    return tuple(sorted(left)), produced

def demo():
    print("=" * 70)
    print(" THE ONE-SHOT KITCHEN GAME -- resources that run out")
    print(" The Reasoning Kingdom, Volume 2, Chapter 16 (Sec. 16.0)")
    print("=" * 70)
    table = ("!C", "A", "B")
    print(f"\nTable at start: {table}")
    print("  recipes:  A x B -o D   (spend egg and flour, get dish D)")
    print("            !C -o cup   (!C is knowledge: never spent)")
    result = cook(table, RECIPES[0])
    assert result is not None
    table, dish = result
    print(f"\nCook the omelet:      {table}   (A and B are gone; {dish} appeared)")
    refused = cook(table, RECIPES[0])
    assert refused is None
    print("Cook it again:        refused.")
    print("  the egg card is spent. Classical logic would hand you the same")
    print("  egg back for free -- the contraction rule is a photocopier for")
    print("  assumptions. This kitchen does not own a photocopier.")
    table, cup = cook(table, RECIPES[1])
    assert cup == "cup"
    print(f"\nUse knowledge !C:     {table}")
    table, cup2 = cook(table, RECIPES[1])
    print(f"Use knowledge !C again: {table}")
    print("  !C never left the table. Knowledge can be used again and again;")
    print("  a resource cannot.")
    print("\nThe point of the game: reasoning often carries a physical cost.")
    print("  You cannot buy two cups of coffee with one five-dollar bill, and")
    print("  you cannot clone an unknown quantum state. Classical logic assumed")
    print("  a world obscenely rich in resources; the real world is not that")
    print("  generous. Linear logic hands the bill back to you.")
    print("=" * 70)

if __name__ == "__main__":
    demo()
```

## 16.1 收缩规则在做什么

回忆收缩规则的形式：

$$\frac{\Gamma, A, A \vdash B}{\Gamma, A \vdash B}$$

横线上：上下文里有两份 $A$，可以推出 $B$。横线下：只有一份 $A$，仍然可以推出 $B$。

这条规则说的是：**你知道 $A$，就可以把它用任意多次，而不需要为每一次使用“支付”一份 $A$**。

在经典逻辑里，这完全合理。知识不是物质——你知道“$1 + 1 = 2$”，用上一百次，它还在那里，一点也没变少。

但世界上并不是所有“前提”都是知识。

考虑这句话：“我有一张五美元钞票，它可以买一杯咖啡。”

这是一次完全合法的命题推断：有一张五美元钞票 → 可以买咖啡。

但如果你对这个命题应用收缩，就会得到：有一张五美元钞票，可以买两杯咖啡。这显然错了。那张钞票已经被花掉了。

经典逻辑对这类错误无能为力，因为它不区分“可以自由复用的知识”和“使用后即被消耗的资源”。线性逻辑的出发点正是这一点：**这两类东西根本不同，需要不同的符号和不同的规则**。

::: info 兔狲教授评
经典逻辑处理知识，线性逻辑处理资源——这个区分是真实的，不是花哨的哲学。你写程序时，内存会耗尽；量子态不能被克隆；在协议里，消息发出去就没了。经典逻辑对这些场景的建模从一开始就是错的。不是近似地错，而是根本地错。
:::

::: info 逻辑是在描述世界，还是在规范推断？

这里有一个值得停下来想一想的哲学分歧。有人会说：逻辑不是在描述资源消耗；它谈论的是真值，而真值不会因为被使用而“耗尽”。这个观点没有错——经典逻辑确实是关于真值的。

但线性逻辑采取另一种立场：逻辑规范的是**推断的合法性**，而推断发生在一个有代价的世界里。不是每一条“已知”都能免费复用。如果你想使用一份资源，就必须为它付出代价。这不是对经典逻辑的否定，而是对它的推广——经典逻辑是线性逻辑加上了“资源可以被任意复制”这一特殊假设之后的退化版本。
:::

---

## 16.2 线性逻辑的符号体系

拿走收缩，逻辑的结构就变了。最明显的变化是：你不能再随意“合并”两个假设，因为它们各自的数量开始变得重要。

线性逻辑引入了一套与经典逻辑平行、但含义不同的连接词。

**乘法连接词 $\otimes$（张量，tensor）**：$A \otimes B$ 表示“你同时拥有 $A$ 和 $B$”，而且两份资源都是真实的，各占一个位置。要从 $A \otimes B$ 推出任何结论，你必须把 $A$ 和 $B$ 都消耗掉。它对应“并行使用”——两件事同时发生。

**加法连接词 $\&$（with）**：$A \,\&\, B$ 表示“你握有 $A$ 这个选项和 $B$ 这个选项，可以挑一个来用”。但你只能挑一个——资源的命运在你选择的那一刻就决定了。它对应“选择”——两件事里只有一件发生。

**乘法析取 ⅋（par）**：线性逻辑的对偶连接词，对应乘法合取的否定形式。

**加法析取 $\oplus$（plus）**：$A \oplus B$ 表示“你拥有 $A$ 或 $B$ 中的一个，但你不知道是哪个”——这表达的是信息缺失，而不是选择上的富足。

:::details 线性逻辑的四个连接词：通过点菜来理解（承接 Girard, 1987）
线性逻辑里多出来的连接词，最容易通过餐厅点菜的类比来把握：

| 符号 | 读法 | 类比 | 含义 |
|------|------|------|------|
| $A \otimes B$ | tensor（张量） | 套餐：汉堡**和**薯条，两样都给你 | 拥有两份资源，必须都消耗 |
| $A \,\&\, B$ | with | 今日特选：汉堡**或**沙拉——你来选 | 你握有选择权，但只能挑一个 |
| $A \oplus B$ | plus | 厨师随机递给你汉堡或沙拉；你不知道是哪个 | 你得到其中一个，但不知道是哪个 |
| $A \multimap B$ | 线性蕴含 | 用一份 $A$ 换一份 $B$；$A$ 被消耗 | 资源转换，而不是知识推导 |

**为什么要区分 $\otimes$ 和 $\&$？** 在经典逻辑里，“拥有 $A$ 和 $B$”只有一种说法（$A \land B$），因为资源可以复制：“同时拥有两者”和“能够选择”是一回事。线性逻辑去掉了复制，这两种“拥有”就必须被精确地区分开。
:::

## 16.2.5 实操：在线性逻辑里写一个证明

让我们用一种序贯演算（sequent calculus）的风格来写一个线性逻辑证明。一个 sequent 的形式是 $\Gamma \vdash \Delta$，读作：资源集合 $\Gamma$ 可以被转换成资源集合 $\Delta$。

在线性 sequent calculus 里，恒等公理只有一条：

$$\frac{}{A \vdash A}\ (\text{id})$$

但关键的差别在于**上下文管理**：每一条推断规则都精确说明它消耗什么、产生什么。没有弱化（你不能把一条假设丢进空气里），没有收缩（你不能复制一条假设）。

下面是一个完整的证明示例——证明 $A \otimes B \multimap B \otimes A$（“同时拥有 $A$ 和 $B$”可以转换为“同时拥有 $B$ 和 $A$”——在线性逻辑里，交换这两者不是公理；它是一条你可以证明的定理）：

```
Proof tree:

                ————— (id)             ————— (id)
                A ⊢ A                 B ⊢ B
               ——————————— (⊗R)     ——————————— (⊗R)
               A, B ⊢ A ⊗ B          B, A ⊢ B ⊗ A
               ————————————— (⊗L, bundle A and B)
               A ⊗ B ⊢ A ⊗ B         ——————————————— (exchange? Not allowed!)
```

等等——这里刚刚暴露了一个问题。在标准的线性 sequent calculus 里，交换（重排上下文）究竟是一条隐含的结构规则，还是一条显式的规则？

答案是：**这取决于你用的是哪一种线性逻辑变体**。在经典线性逻辑（CLL）里，交换被保留为一条结构规则——你可以自由重排上下文中的资源。在非交换线性逻辑（NLL）里，交换被去掉，资源的左右顺序具有物理含义（比如时间顺序）。

在保留交换的版本里，这个证明很简单：

1. 从 $A \vdash A$ 和 $B \vdash B$ 出发。
2. 用 $\otimes R$ 得到 $A, B \vdash A \otimes B$ 和 $B, A \vdash B \otimes A$。
3. 用交换重排第二个：$A, B \vdash B \otimes A$。
4. 用 $\multimap R$ 从 $A \otimes B \vdash A \otimes B$ 和 $A, B \vdash B \otimes A$……不对，等等——更直接的路线是用 $\otimes L$：
5. 要证明 $A \otimes B \vdash B \otimes A$，先假设你有 $A \otimes B$（也就是 $A$ 和 $B$ 被捆在一起），用 $\otimes L$ 把它们拆开再重新组合。最后用 $\multimap R$。

在没有交换的版本里，$A \otimes B \vdash B \otimes A$ 是**不可证的**！因为资源的位置信息已经被写进了逻辑本身——你不能“自由地”重排 $A$ 和 $B$ 的顺序。

这个练习的重点不在于公理的编号，而在于第一手体验：**拿掉一条结构规则，整个逻辑的定理集合就变了**。这不是微调；这是根本性的改变。

::: info 兔狲教授评
在保留交换的版本里，这个证明只需要两步；拿掉交换，它就变成不可证的。大多数逻辑课从不让你看到这种对比——它们把交换当成空气；你呼吸着空气，却不知道空气存在。线性逻辑的工作之一，就是让空气变得可见。
:::

::: info 为什么需要这么多连接词？

经典逻辑只需要两个基本连接词（比如 $\neg$ 和 $\to$）就能表达一切。线性逻辑需要四个，感觉冗余。但这四个无法合并，因为它们在资源意义上确实不同：同时拥有两者，与能够做出选择——经典逻辑用收缩规则抹平了这一差异。既然知识可以被无限制地复制，“同时拥有”和“能够选择”就没有区别。一旦收缩消失，这个差异就必须被精确地写出来，否则逻辑就会丢失信息。
:::

---

## 16.3 Bang：可以重复使用的资源

但如果每一条假设都只能用一次，我们就无法谈论真正的知识——那些你可以正当地反复引用的事实。线性逻辑用一个特殊符号解决了这个问题：bang 模态 $!A$（读作“当然 $A$”）。

$!A$ 的意思是：$A$ 是一份**无限资源**。拥有一份 $!A$，就意味着你可以随时从中取出一个 $A$，取多少次都行，而且它不会有任何损耗。

形式上，$!A$ 服从一组特殊规则：

- **弱化**：$!A \vdash \mathbf{1}$（$!A$ 可以被丢弃——你不一定要用它）
- **收缩**：$!A \vdash !A \otimes !A$（$!A$ 可以被复制——你可以把它用两次）
- **推广**：如果 $!A \vdash B$，那么 $!A \vdash !B$（从无限资源推出的结论本身也是无限的）

有了 bang，线性逻辑就恢复了对经典推理的模拟：如果每一条假设都带上 $!$，整个系统就退化为经典逻辑。bang 是线性逻辑与经典逻辑之间的桥梁——它精确地标出哪些假设可以按经典意义自由使用，哪些是真正的一次性资源。

::: info 兔狲教授评
很多人到这里会说：“好，所以 bang 就是把资源变成知识。”且慢——你有没有意识到，在线性逻辑里，“可以被重复使用”是一项必须被明确声明的**特权**，而不是默认权利？这个颠覆比大多数人意识到的要彻底得多。在经典逻辑里，每一条假设默认都带着一个看不见的 $!$——而你从来不知道。
:::

---

## 16.4 蕴含现在的含义不同了

一旦收缩消失，蕴含 $A \to B$ 也分裂成两种必须区分的含义。

线性蕴含 $A \multimap B$（读作“$A$ 线性蕴含 $B$”，或“$A$ 变换为 $B$”）的意思是：**恰好消耗一份 $A$，产生一份 $B$**。

这不是知识的传递——这是**资源的转换**。一张五美元钞票 $\multimap$ 一杯咖啡。这正是线性逻辑能够正确表达的那类推断——用过之后，五美元钞票消失了，换成了咖啡。

相比之下，经典逻辑的 $A \to B$ 不消耗 $A$：你知道 $A$，推出 $B$，而 $A$ 还留在那里供以后使用。要在线性逻辑里表达经典蕴含，你会写成 $!A \multimap B$：从无限的 $A$ 中取出一份，产生一个 $B$，而 $!A$ 本身毫发无损地留下来。

这一区分揭示了“知识”与“资源”的本质差异：知识是 $!A$ 型的——不会因为使用而减少；资源是裸 $A$ 型的——使用就会被消耗。经典逻辑不做这种区分，因为它假设每一条前提都是知识。

---

## 16.5 三个现实世界的对应

线性逻辑不是逻辑学家的纯思想实验。它精确地对应着三个重要的现实。

**内存管理。** 在编程中，内存是一种资源：分配一块，用完就必须释放，否则就会泄漏。经典逻辑关于内存的直觉是错的——它假设你“知道”一块内存，就可以永远引用它，但一块被释放的内存已经不存在了。线性类型系统（比如 Rust 的所有权系统）的核心正是线性逻辑：每个变量恰好被拥有一次，交出所有权就是花掉一条假设，借用则是使用一个带 $!$ 装饰的临时视图。线性逻辑给出了 Rust 所有权机制的形式语义。

**量子计算。** 量子力学有一条著名的禁令：**不可克隆定理**。一个未知的量子态不能被完美复制。用逻辑的语言说，这正是“不允许收缩”——一个量子比特是线性资源；你不能从一个 $\psi$ 造出两份 $\psi$。量子计算的计算模型天然地生活在线性逻辑之中；一个量子电路就是一串线性推断。

**并发协议。** 两个进程之间的通信协议可以被精确地编码为线性逻辑命题。协议中“一方发送、另一方接收”恰恰是线性蕴含：资源从一侧转移到另一侧——不可复制，不可丢弃。会话类型（session types）正是这一对应的工程实现，它让编译器在类型检查阶段验证协议的正确性——逻辑保证协议不会死锁，也不会丢失消息。

## 16.5.5 实操：Rust 的所有权系统如何镜像线性逻辑

这不是比喻。Rust 的类型检查器在编译期执行的规则，在线性逻辑里都有精确的对应物。下面用一个具体的 Rust 程序来说明。

**例1：移动语义 = 线性蕴含**

```rust
let x = String::from("Hello");  // x owns the string
let y = x;                      // ownership moves from x to y
// println!("{}", x);           // compile error! x has been "consumed"
// error[E0382]: use of moved value: `x`
```

对应的线性逻辑命题：

$$\text{String("Hello")} \multimap \text{String@y}$$

资源被隐式地从 x 移动到 y，x 不再可用。在线性逻辑里，这就是：消耗一份资源，产生一份资源，原来那份消失。

**例2：不可变借用 = $!A$ 的一个临时视图**

```rust
let x = String::from("Hello");
let r1 = &x;  // immutable borrow — can read, cannot modify
let r2 = &x;  // multiple immutable borrows may coexist
println!("{} {}", r1, r2);  // ✓
println!("{}", x);          // ✓ x is still valid
```

对应的线性逻辑命题：

$$!A \vdash A \otimes A \otimes \cdots$$

$!A$（无限资源）允许从中取出多个只读视图。借用结束时，x 的所有权不受影响。

**例3：可变借用 = 对线性资源的独占访问**

```rust
let mut x = String::from("Hello");
let r = &mut x;  // mutable borrow — exclusive access
r.push_str(", world");
// let r2 = &x;  // compile error! cannot hold a mutable and an immutable borrow at once
// error[E0502]: cannot borrow `x` as immutable because it is also borrowed as mutable
```

对应的线性逻辑是：可变借用临时“拿走”了对资源的独占修改权——与移动不同，借出的资源在借用结束后会还回来；与不可变借用不同，可变借用是独占的，所以两个可变借用不能共存。这是线性逻辑与仿射类型（affine types，即资源至多使用一次的类型）之间的交界地带。

**例4：函数调用 = 消耗一个参数**

```rust
fn consume(s: String) {  // consumes the input String
    println!("{}", s);
    // s is dropped here
}
let x = String::from("Hello");
consume(x);               // x is moved into the function parameter
// println!("{}", x);     // compile error! x has been consumed
```

对应的线性逻辑命题：

$$\frac{\Gamma \vdash \text{String("Hello")} \multimap \mathbf{1}}{\Gamma, \text{String("Hello")} \vdash \mathbf{1}}$$

函数体消耗了这份资源，什么也没有产生（$\mathbf{1}$，对应 Rust 的 `()` 类型）；资源在调用发生时被转移，调用者不再拥有它。

**四个例子的总结：**

| Rust 操作 | 线性逻辑 | 解释 |
|-----------|---------|------|
| 移动 `let y = x` | $A \multimap B$ | 资源从一个地方移动到另一个地方 |
| 不可变借用 `&x` | $!A \vdash A$ | 从无限资源中取出一个只读视图 |
| 可变借用 `&mut x` | 独占的线性访问 | 临时转移独占修改权；借用结束时归还 |
| 函数调用 `f(x)` | $\multimap$ 消去 | 资源被函数体消耗 |

这一对应绝非巧合。Rust 的核心类型系统（所有权、借用、生命周期）建立在线性类型之上——它的类型检查器在编译期执行的逻辑，正是线性逻辑的推断规则。Rust 编译器并不是“从线性逻辑中汲取了灵感”；它**实现了**线性逻辑。

::: info Curry-Howard 对应的一个延伸

第14章提到，证明与程序在结构上同构（这个对应关系会在第22章正式展开）。线性逻辑把这个对应又推进了一步：线性证明对应**线性程序**——其中每个变量恰好使用一次的程序。这不是偶然的巧合；这是因为线性逻辑从一开始就是为了捕捉“计算是一个消耗资源的过程”这一直觉而设计的。Girard 最初正是在分析 System F（多态 λ-演算）的证明结构时发现了线性逻辑——他在逻辑的深处发现了资源的痕迹。
:::

---

## 16.6 两种真值：占有与选择

线性逻辑还引出了一个更根本的哲学转变：**“真”不再只有一种含义**。

在经典逻辑里，如果 $A$ 为真，它就是真的——只有一种情况，到此为止。

在线性逻辑里，“拥有 $A$”可以是两种不同的事情：

- **张量式占有**：$A \otimes B$——你同时拥有 $A$ 和 $B$，这是两份独立的资源。
- **加法式占有**：$A \,\&\, B$——你有两个选项，可以挑一个来用。

这两种“拥有”在资源经济学中具有完全不同的价值。拥有一个苹果和一个梨，与拥有“在苹果和梨之间做选择的权利”并不相同——前者是两份资源；后者是一份资源外加一项选择权。

经典逻辑把它们混为一谈，因为在真值层面上，“$A$ 和 $B$ 都为真”等价于“$A$ 为真或 $B$ 为真，而且两者都为真”——这是把同一件事换了一种说法。但一旦资源有了数量，这两件事就截然不同了。

这个区分将在第17章以一种完全出乎意料的方式重新出现：当真值本身变成一个连续区间 $[0, 1]$ 时，“同时拥有”与“选择其一”之间的差异，会变成概率的两种不同模式——乘法规则与贝叶斯更新。

---

## 16.7 过渡：在真值变成连续之前——三值逻辑作为热身

第17章将做一件激进的事：把真值从 $\{0, 1\}$ 扩张到 $[0, 1]$。在此之前，我们先做一个小实验：如果真值有三个值而不是两个，会发生什么？

波兰逻辑学家 Łukasiewicz 在 1920 年提出了三值逻辑。真值集合是 $\{0, \frac{1}{2}, 1\}$，其中 $\frac{1}{2}$ 表示“未定”。连接词规则如下：

**否定**：$\neg 0 = 1$，$\neg \frac{1}{2} = \frac{1}{2}$，$\neg 1 = 0$

**合取**：$a \land b = \min(a, b)$

**析取**：$a \lor b = \max(a, b)$

**蕴含**：$a \to b = \min(1, 1 - a + b)$

（蕴含规则是三值逻辑里最反直觉的部分，请多盯它一会儿。当 $a = 1, b = 0$ 时得到 $0$——与经典逻辑一致；当 $a = \frac{1}{2}, b = 0$ 时得到 $\frac{1}{2}$——“从不确定推出假，结论仍然不确定”。）

**练习：** 使用上面的规则，计算下列每个命题在这个真值赋值下的值：$P = \frac{1}{2}$（未定），$Q = 0$（假）。

1. $P \to Q$
2. $\neg Q \to \neg P$（逆否命题）
3. $P \lor \neg P$（排中律）

第三个最有意思：在经典逻辑里，$P \lor \neg P$ 是重言式，永远为真。在三值逻辑里，当 $P = \frac{1}{2}$ 时，$\neg P = \frac{1}{2}$，所以 $P \lor \neg P = \max(\frac{1}{2}, \frac{1}{2}) = \frac{1}{2}$。

**排中律——在三值逻辑里，不再是重言式。**

这不是技术细节。它意味着：当你扩张“真值”时，你其实在修改自己愿意接受的推断规则。三值逻辑拒绝排中律；经典逻辑接受它。两者没有哪个更“正确”——它们是两种不同的选择，是对“什么使推断合法”的两种不同理解。

回到一次性厨房游戏：本章真正改变的不是菜谱的名字——而是厨房的物理法则。经典逻辑的厨房假设食材可以被无限复制；线性逻辑把食材重新变成消耗品；三值逻辑走得更远，它暗示即便是“熟了/没熟”这样的二元判断，也可以被第三种状态打断。状态空间每变一次，合法走法的集合就变一次。

第17章要做的，是把这个方向推到极致：真值不是三个点，而是一个连续区间 $[0, 1]$。到那时，推断规则会变成什么样子？

---

## 悬而未决

**线性逻辑里的“真”是什么？** 在经典逻辑里，语义很清楚：一个命题要么真要么假，真值表能给你一切。在线性逻辑里，资源的“语义”是什么？你如何定义一个模型，使得“$A$ 为真”意味着某种具体的资源占有？答案是相位语义（phase semantics）和相干空间（coherence spaces），但它们指向同一个方向：真值不再是 $\{0, 1\}$，而是某种更丰富的结构。

**“资源”在多大程度上是一个精确的数学概念？** 线性逻辑给了我们一个形式框架，但“资源”本身仍然是一个直觉性的词。一份资源是什么？什么时候复制一份资源是合法的？这些问题在不同的应用领域有不同的答案；线性逻辑提供的，是一个能够容纳这些不同答案的框架——而不是某一个单一答案。

**如果连交换规则也去掉，会发生什么？** 第14章列出了三条结构规则：弱化、收缩、交换。线性逻辑去掉了收缩（有时也去掉弱化）。如果连交换也去掉——让假设的顺序变得重要——你得到的就是**有序逻辑**（ordered logic），它对应着带有时间顺序的资源使用，比如消息必须按顺序发送和接收的通信协议。这个方向仍在展开之中。

**如果“真”本身是一个程度问题，推断规则应该怎样改变？** 线性逻辑通过给资源计数，把 $\{0, 1\}$ 真值沿着一个维度撑开了——假设之间有了“副本数量”的差别，但每一份副本仍然是确定的真或假。然而还有另一个方向：真不是“1 份还是 2 份”，而是“0.3 还是 0.7”。这个方向不再是关于资源——而是关于不确定性。不确定性下推断规则的推广，正是第17章要讲的内容。

---

## 思考题

**★ 热身**

用线性逻辑的资源直觉，判断下列推断是否成立。不需要形式证明——只需用“五美元钞票买咖啡”这样的资源例子，清楚说明它为什么对或为什么错。

1. $A \otimes B \vdash A$（同时拥有 $A$ 和 $B$——你能单独推出 $A$ 吗？）
2. $A \,\&\, B \vdash A$（握有“在 $A$ 和 $B$ 之间选择”的权利——你能推出 $A$ 吗？）
3. $!A \vdash A$（无限的 $A$——你能从中取出一个 $A$ 吗？）
4. $A \vdash !A$（一个 $A$——它能变成无限的 $A$ 吗？）

第 1 题与第 2 题的对比，是本章最重要的区分。第 4 题是一旦去掉收缩规则就彻底失效的推断。

---

**★★ 推导**

考虑下面三个推断。判断每一个在线性逻辑中是否成立，并写出你的理由。

1. $A \multimap B,\ A \multimap B \vdash A \multimap B$（两份“$A$ 变成 $B$”的能力——它们能合并成一份吗？）
2. $!(A \multimap B),\ !A \vdash !B$（一条无限的转换规则，加上无限的输入——能产出无限的输出吗？）
3. $A \otimes B \multimap C \vdash A \multimap (B \multimap C)$（消耗 $A$ 和 $B$ 产生 $C$——等价于先消耗 $A$、再消耗 $B$ 产生 $C$ 吗？）

第 1 题涉及收缩。第 2 题涉及 bang 的推广规则。第 3 题是线性逻辑版的柯里化——想一想它在资源意义上是否真的成立。

---

**★★★ 挑战**

Rust 的所有权系统实现了线性类型：每个值恰好有一个所有者，转移所有权意味着花掉一条假设，借用则对应使用一个带 $!$ 装饰的临时视图。

试着把下列每一个 Rust 操作都表达为一条对应的线性逻辑推断：

- 把变量 `x` 的所有权移动给函数 `f`（移动语义）
- 把 `x` 的不可变引用传给函数 `g`（不可变借用）
- 一个函数的返回值（消耗输入，产生输出）

不必做到完全精确——但要说清楚这一点：哪些操作对应消耗性推断（线性蕴含 $\multimap$），哪些对应可复用的视图（bang $!$）？这个练习的目的是把语言设计的直觉翻译成逻辑的语言——在翻译的过程中，你会发现哪些部分对得上，哪些部分对不上。

# 第3章：从符号到向量——表示空间的第一次解放

> 把“国王”减去“男人”，加上“女人”，你就得到了“女王”。这是推理——还是算术？

<div class="center">

------------------------------------------------------------------------

</div>

## 一、那句话

2013 年，Tomas Mikolov 在一篇论文里写下了一个方程。后来，这个方程被引用的频率，大概超过了那个时代任何一篇 NLP 论文：

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}
$$

意思是：把“国王”的词向量减去“男人”的向量，再加上“女人”的向量——最近邻就是“女王”。

这不是一个被精心设计出来的例子。它是实验直接跑出来的结果。算法自己学会了这个关系；没有人告诉它“国王之于男人，犹如女王之于女人”。

那一刻，AI 研究界集体倒吸了一口气。

为什么？因为这看起来像是机器在做**类比推理**。不是规则推理，不是逻辑演绎，而是更接近人类直觉的某种东西：它抓住了关系的本质，然后把这种关系迁移到新的对象上。

然后，问题来了：它真的在推理吗？

这一章讲的就是这个问题——什么是向量表示，它为什么比符号更强大，它的力量从哪里来，又在哪里止步。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、符号的问题，重说一遍

在上一章，我们看到了符号 AI 的核心困境：知识必须被显式地写出来，而关于世界的知识写不完。

但还有一个问题我当时没有强调：**符号的意义是从外面硬装上去的**。

在专家系统里，`"is a mammal"` 这个字符串对机器来说只是一个 token——一个无法再分解的原子。它不知道“哺乳动物”和“动物”是什么关系，不知道“哺乳动物”离“爬行动物”有多近，也不知道“猫”比“鲸”更像一个典型的哺乳动物。

所有这些关系都必须被显式地写进规则。

    IF  is a mammal
    THEN is an animal  [R_base_1]

    IF  is a mammal
    AND is not a whale
    THEN usually has legs  [R_mammal_legs]

每一种关系都需要自己的规则。而关系是无穷无尽的，规则永远写不完。

问题的根源在于：**符号系统用离散的 token 表示概念，而概念之间的关系是连续的、有程度的、嵌套的**。“猫”和“狗”之间的距离，无法用离散的 If-Then 规则捕获——你可以说“猫是哺乳动物，狗是哺乳动物”，但你说不出“猫比蛇更接近狗”。

向量表示解决了这个问题。不是完全解决，但解决了核心部分。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、分布假设：意义藏在邻居里

向量表示背后的想法，可以追溯到语言学家 J.R. Firth 在 1957 年写下的一句话：

> “你可以通过一个词交往的朋友来认识它。”（You shall know a word by the company it keeps.）

这个直觉叫做**分布假设**（distributional hypothesis）：一个词的意义，由它出现的上下文分布决定。

出现在相似上下文里的词，意义相近。“猫”和“狗”都经常出现在“我养了一只 \_\_\_”“给它喂食”“它叫了”这样的上下文里——所以它们是语义近邻。

这个直觉可以被机械化。建一个矩阵：行是词，列是上下文词（或文档），每个格子记录两者共现的频率（或对计数的某种变换）。那么每一行就是一个词的向量表示。意义相近的词，最终会在高维空间里彼此靠近。

这是向量表示的原始版本，叫做**分布语义模型**（distributional semantic models，DSM），从 1990 年代就存在了。它有效，但有两个问题：维度太高（可达几十万维），向量极其稀疏（大多数格子是零）。

:::details 分布语义模型（DSM）：词义藏在共现矩阵里
**分布语义模型**是一类把词的意义表示为向量的方法。核心思路是：建一张大的“词 × 上下文”表，记录哪些词和哪些词一起出现，然后把这张表的每一行当作该词的向量。

- **高维**：如果语料里有 10 万个不同的词，矩阵就有 10 万列，每个词的向量就是 10 万维的——非常长
- **稀疏**：大多数词对很少或从不共现，所以矩阵里大部分是 0——存储极其浪费

Word2Vec（Mikolov 2013）解决了这个问题：不直接构建矩阵，而是训练一个神经网络，把这种共现信息**隐式地**压缩成 100–300 维的稠密向量。
:::

Mikolov 2013 年的论文走得更远。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、Word2Vec：通过预测来压缩

Word2Vec 不构建共现矩阵。它训练一个神经网络，只做一件事：**给定一个词，预测它的上下文词**（Skip-gram），或者**给定上下文词，预测中心词**（CBOW）。

:::details Skip-gram 和 CBOW：Word2Vec 的两种训练模式
Word2Vec 有两个变体，区别只在于预测任务的方向：

- **Skip-gram**：输入中心词，预测它周围的词。例如，在句子“I like eating apples”里，输入“eat”，预测“like”和“apples”。在小数据集上表现好，对罕见词效果更好。
- **CBOW（Continuous Bag of Words）**：输入周围的词，预测中心词。输入“like”和“apples”，预测“eat”。训练更快，适合大数据集。

两者都只是达到目的的手段；真正的目标是词向量（嵌入）。训练完成后，预测头被丢掉，只保留嵌入矩阵。
:::

这听起来很普通。但副产品才是重点。

训练这个网络时，每个词都会被映射到一个低维向量（通常是 100 到 300 维）。要让预测任务做得好，这些向量必须编码每个词的某种本质——因为只有理解词与词之间的关系，你才能预测哪些词会共现。

训练结束后，你丢掉预测头，只保留嵌入矩阵。这就是你的词向量。

这里有一些深刻的东西值得停下来想一想：

**你从未教过网络“国王”和“女王”之间的关系。你没有给它任何语义规则。你只是让它预测文本序列。然后，等它学会预测之后，你发现“king − man + woman ≈ queen”这样的关系自己涌现了出来。**

为什么？因为在自然语言里，“国王”出现的地方，“女王”也会以平行的方式出现——只是出现在相应的女性语境里。为了准确预测，网络必须把这种平行性编码进向量。“皇室”是一个上下文轴，“性别”是另一个。这两个轴对应向量空间里的两个方向。减去“男人”向量、加上“女人”向量，就是沿性别轴做一次平移。

意义，从分布中涌现。

<div class="center">

------------------------------------------------------------------------

</div>

## 五、几何中的浅层逻辑

让我们更仔细地看这个类比方程。

$$
\vec{A} - \vec{B} + \vec{C} \approx \vec{D}
$$

king − man + woman ≈ queen，paris − france + japan ≈ tokyo，walk − walked + swim ≈ swam（过去时的关系也被编码了）。

每一个类比都对应向量空间里的一次**平移**。从“男人”到“女人”的位移，和从“国王”到“女王”的位移，是同一个向量方向——沿性别维度的一次移动。

这是非常干净的几何结构。语义关系变成了方向。

![image](/figures/ch03_fig4_similarity_heatmap.png)

*图4：12 个词的两两余弦相似度矩阵。绿色=高相似，红色=低/负相似。白线划定了四个语义簇的边界（皇室、性别、职业、动物）。观察：同一簇内的相似度远高于跨簇；动物簇与其他三簇几乎没有重叠。这正是“语义相近的词在向量空间里彼此相邻”的可视化证明。*

<div class="center">

------------------------------------------------------------------------

</div>

## 六、向量的偏见：语料即世界观

Word2Vec 学到的不是关于语言的真理，而是语料库的统计。而语料库是人写的——它带着人类的偏见、历史的偏见、文化的偏见。

这不是一个边缘问题。这是分布假设的核心代价。

**Bolukbasi et al. (2016) 的实验**

他们发现，在 Google News 上训练的 Word2Vec 里：

$$
\vec{\text{man}} - \vec{\text{woman}} \approx \vec{\text{doctor}} - \vec{\text{nurse}}
$$

“男人之于医生，如同女人之于护士”——这个类比在向量空间里成立，是因为它在训练语料里统计上成立。

更直接的测试：

$$
\vec{\text{father}} - \vec{\text{mother}} + \vec{\text{programmer}} \approx \vec{\text{homemaker}}
$$

不是“女程序员”，而是“家庭主妇”。

这不是算法里的 bug。算法正确地学到了语料的统计。bug 在我们这里：**我们把统计规律误当成了语义真相**。

**偏见从哪里来？**

分布假设说，一个词的意义藏在它的邻居里。但如果邻居的分布本身就是有偏的——

- 在新闻报道里，“CEO”的邻居更多是“他的决定”，而不是“她的决定”
- 在历史文本里，“科学家”更常与男性名字共现
- 在小说里，情感词在不同性别之间的分布不均匀

Word2Vec 忠实地学到了这一切。它是一面镜子——照出的是语料，不是世界。

这暴露了分布假设的一个根本极限：

向量空间学到的是**描述性的**统计规律，而推理需要的是**规范性的**真理。“‘X 和 Y 在语料里经常共现’不等于‘X 和 Y 在世界上有因果关系’”，也不等于“这种共现是正确的”。

你可以对词向量做去偏见处理（Bolukbasi 的论文提出了一种线性投影方法），但那只是创可贴——治标不治本。根本问题在于：**从统计相关性出发，你永远无法抵达语义真相**。

这个极限在 Transformer 和大语言模型（LLM）时代依然存在——大语言模型是在海量文本上训练、用来预测下一个词的模型。语料更大，偏见也更大、更广——但分布假设的底层逻辑没有改变。

<div class="center">

------------------------------------------------------------------------

</div>

## 七、类比的幻觉

让我回到那个方程：

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}
$$

这真的是推理吗？

我们把它拆开来看。

要让这个等式成立，“国王”和“男人”两个向量之间的差，必须近似等于“女王”和“女人”两个向量之间的差。

$$
\vec{\text{King}} - \vec{\text{Man}} \approx \vec{\text{Queen}} - \vec{\text{Woman}}
$$

这是什么？这是一个关于**偏移方向相匹配**的陈述。是几何上的规律性——不是逻辑推理。

![image](/figures/ch03_fig2_analogy_arithmetic.png)

*图2：向量类比的几何视图。四个词在二维 PCA 空间中形成一个近似平行四边形——man→king 和 woman→queen 是两条几乎平行的边，而 man→woman 和 king→queen 是另一对这样的边。这就是“性别位移向量在皇室词对上保持一致”的意思。King − Man + Woman 的余弦相似度接近 0.995。*

在逻辑里，推理需要**规则 + 事实 → 结论**。“所有国王都是男人”加上“伊丽莎白是女王”，推出“伊丽莎白不是国王”——这是推理，因为其中有传递性、变量绑定和量词。

向量算术里这些都没有。它只说明：这些词在语义空间里的相对位置，呈现出某种平行结构。

更重要的是，这个等式在很多情况下都会失败。

Fournier et al.（2020）系统分析了类比测试的局限，发现那些著名的“类比等式”严重依赖一个取巧的评估方案——把输入词本身排除在候选答案之外，而最近邻往往就是输入词之一。改变评估方式后，大量被报告的“类比能力”显著下降 `-> [Fournier et al., 2020, arXiv:2010.03446]`。

还有一个更直接的例子。试试这个类比：

    doctor - man + woman ≈ ?

在某些训练语料上，你得到的不是“女医生”，而是“护士”。

因为在那些语料里，“医生”更常与男性语境共现，“护士”更常与女性语境共现。模型学到的不是“医生”这个职业的中立含义，而是它在特定语料里的**统计偏见**。

向量表示不捕获真理。它捕获的是语料库的统计分布。语料有偏见，向量就有偏见。这不是 bug——这就是它的工作方式。

![image](/figures/ch03_fig3_analogy_bias.png)

*图3：左图，皇室类比成功——King − Man + Woman 的结果向量非常接近 Queen（余弦 ≈ 0.995），平行四边形结构清晰。右图，职业类比失败——Doctor − Man + Woman 不指向“女医生”，而指向 Nurse（余弦 ≈ 0.865）。原因是：在训练语料里，Doctor 偏向男性语境，Nurse 偏向女性语境。同样的算术，不同的偏见结果。*

<div class="center">

------------------------------------------------------------------------

</div>

## 八、分布假设的极限

让我们更深入地想一想，分布假设的边界在哪里。

这个假设说，意义由上下文分布决定。大多数情况下这是对的。但它意味着：**如果两个词在一段文本里的上下文分布完全相同，模型就会把它们当作意义相同。**

“I like cats”和“I like dogs”——这两句话完全对称，所以“cat”和“dog”在分布上看起来高度相似。是的，它们在语义上确实接近；模型是对的。

但是：

“The earth orbits the sun”和“the sun orbits the earth”

如果只看词的分布，“earth”和“sun”出现在非常相似的上下文里——它们都经常出现在关于天文、引力、轨道运动的句子里。然而这两句话，一句是真的，一句是假的。

分布语义模型没有**真值**的概念。它不知道哪些命题是真的、哪些是假的。它知道的只是哪些词出现在哪些上下文里——这是一种统计规律，不是关于世界的因果模型。

这就是为什么向量表示虽然在许多任务上表现很好，却没有解决推理。它们解决的是**表示**问题：用更丰富、更连续的方式表示概念，让相似的概念在几何上彼此靠近。但“靠近”不是“有逻辑关系”，“统计共现”也不是“因果相关”。

Wittgenstein 说过一句话，用在这里再贴切不过：**语言的意义就是它的使用**。分布假设抓住了这个直觉的一部分——一个词的意义确实体现在它的使用里。但使用是统计性的。理解也是统计性的吗？这是一个悬而未决的问题。

<div class="center">

------------------------------------------------------------------------

</div>

## 九、从词向量到句子：代价不断累积

Word2Vec 给出的是**静态词向量**——每个词一个固定向量，不管它出现在什么上下文里。

“bank”这个词，在“I went to the bank to withdraw money”和“wildflowers grew all over the river bank”里得到的是同一个静态向量。这是它的核心局限之一。

ELMo（2018）和后来的 BERT（2018）引入了**上下文化词向量**——同一个词在不同上下文里得到不同的向量。这是重大的一步。但它带来了另一个代价：你不再有一张整洁的、可离线查询的词向量表；每次都必须运行整个网络。

:::details ELMo 和 BERT：什么是上下文化词向量？
**静态词向量**（如 Word2Vec）：每个词一个向量，无论它出现在哪里。“apple”指水果还是指 Apple 公司，向量都一样。

**上下文化词向量**（ELMo、BERT）：同一个词在不同句子里得到不同的向量，因为整个句子都进入网络，每个词的表示都由它的上下文塑造。

- **ELMo**（2018，Allen AI）：用双向 LSTM 生成上下文敏感的词向量，作为特征提供给下游任务
- **BERT**（2018，Google）：用 Transformer Encoder 预训练一个双向语言模型，再在下游任务上微调

代价：不再能查表；每次推理都必须运行整个网络，计算量大幅增加。这就是准确率与效率之间永恒的权衡。
:::

更深层的问题是：**句子的意义不是词向量的简单组合。**

“Cat bites dog”和“dog bites cat”——把词向量取平均，你会得到几乎相同的句子向量，因为用的是同样的三个词。但这两句话描述的是相反的事件。

词的意义可以被编码进向量，但句子的意义取决于**结构**——谁是主语，谁是宾语，动词的论元角色是什么。纯粹的向量算术没有捕获其中任何一点结构。

这个问题推动了 Transformer 的发明——用注意力动态处理词与词之间的关系，而不是把它们简单堆叠起来。但那是第九章的故事。

<div class="center">

------------------------------------------------------------------------

</div>

## 十、一个小小的停顿

让我盘点一下。

从符号到向量：一次真正的范式转变。它解决了符号 AI 的一个核心问题：**意义的表示**。概念不再是无意义的原子；它们在向量空间里有了位置、距离和方向。相似的概念彼此靠近，关系被编码为方向上的位移。

这带来了真实的能力：语义搜索、类比推理（某种意义上）、语义聚类，以及迁移学习的基础。

但它没有解决推理——原因有三：

第一，**它捕获的是统计规律，不是因果结构**。语料有偏见，向量就有偏见。语料有相关性，向量就有相关性。但相关不是因果，分布不是真理。

第二，**它没有组合性**。词向量的简单线性组合无法捕获句子的结构意义。“Cat bites dog”和“dog bites cat”在词袋视角下完全相同——这是一个根本缺陷。

第三，**类比方程看起来像推理，但其实不是**。它是几何上的平行结构，不是逻辑上的传递推导。当它失效时（比如带有社会偏见的类比），它会不声不响地给你一个错误答案，没有任何警告。

向量表示是推理的一个前提条件——但它本身不是推理。

<div class="center">

------------------------------------------------------------------------

</div>

## 十一、还有一个更深的问题

分布假设背后的深层假设是：**语言能够承载意义，而且这种意义可以从语言使用的模式中被提取出来。**

这是一个值得质疑的假设。

John Searle 的中文房间思想实验（1980）是这个方向上最著名的攻击。想象一个不懂中文的人被关在房间里，手里有一本厚厚的规则书：每送进来一个中文符号，规则书就告诉他该送出哪个中文符号。从外面看，他似乎懂中文，能用中文回答问题——但他根本不知道那些符号是什么意思。他只是在机械地执行规则。

:::info
**兔狲教授说**  
这个思想实验是一面镜子，照出了符号操作的尴尬真相：你可以完美地模仿理解，同时什么都不理解。词向量模型所做的事，在某种意义上非常像这个中文房间——它们从海量文本中提取极其精细的统计模式，这些模式在许多任务上产生正确的行为。但它们“理解”这些词的意义吗？没有简单的答案，因为首先我们必须定义“理解”是什么意思。
:::

但有一个更实际的测试：**分布偏移**。把模型放到它的训练分布之外，让它处理训练语料里从未见过的组合——会发生什么？如果它的性能崩塌，你就有充分理由怀疑：它只是记住了分布，而不是真正理解了语言。

这是第五章的核心问题。现在，先把这个悬念留住。

<div class="center">

------------------------------------------------------------------------

</div>


> 向量捕获相似性，但不捕获因果。下一章，我们进入高维空间的几何——流形假设揭示了为什么数据不是随机分布的。

## 悬而未决

- 类比方程 `vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)` 是推理的证据，还是统计的巧合？它对某些类比成立，对另一些失败——这种不一致性告诉了我们什么？

- 分布假设的边界在哪里？是否存在永远无法从文本分布中学到的意义？颜色？疼痛？身体感觉？

- 词向量捕获社会偏见——这是缺陷，还是特性？如果你从词向量中删除偏见，是否也破坏了世界的某些真实结构？

- 如果你用纯噪声文本训练 word2vec，会得到什么？这个思想实验揭示了什么？

<div class="center">

------------------------------------------------------------------------

</div>

## 动手试试：用向量算术戳破类比的幻觉

这一章提出了一个重要的主张：类比方程看起来像推理，但它只是一个几何上的平行结构。你需要亲手验证这个区别——不是读懂它，而是感受它崩溃的那一刻。

你需要一个预训练词向量模型。最简单的选择：用 Python 的 `gensim` 库加载 Google 公开发布的 word2vec 模型，或者使用 `fasttext` 的预训练向量。

<div class="center">

------------------------------------------------------------------------

</div>

### 第一步：验证那个著名的方程

从最经典的类比开始：

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx ?
$$

用第五节的逻辑来实现：

```python
import numpy as np

# Assume model is a loaded gensim KeyedVectors object
# Example of loading it:
#   from gensim.models import KeyedVectors
#   model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def analogy_query(model, positive_word, minus_word, plus_word, topn=5):
    """
    Perform a vector analogy query: vec(positive_word) - vec(minus_word) + vec(plus_word)
    model: a gensim KeyedVectors object
    Returns: a list of nearest neighbor words with their cosine similarity scores
    """
    # Compute the target vector: vec("king") - vec("man") + vec("woman")
    target_vector = (model[positive_word]
                     - model[minus_word]
                     + model[plus_word])

    # Find the words in the vocabulary with the highest cosine similarity to the target vector
    # gensim's similar_by_vector can automatically exclude the input words (via the negative parameter)
    results = model.similar_by_vector(
        target_vector,
        topn=topn + 3  # fetch a few extra, so the input words can be filtered out manually
    )

    # Exclude positive_word, minus_word, plus_word themselves
    exclude = {positive_word.lower(), minus_word.lower(), plus_word.lower()}
    filtered = [(word, score) for word, score in results
                if word.lower() not in exclude][:topn]

    return filtered


# ── Run the classic analogy: king - man + woman ≈ ? ───────────────────────
results = analogy_query(model, "king", "man", "woman")

print("Analogy query: king - man + woman ≈ ?")
print("-" * 35)
for rank, (word, score) in enumerate(results, 1):
    print(f"  Rank {rank}: {word:15s}  cosine similarity = {score:.4f}")
```

你应该会得到“queen”，余弦相似度接近 0.7 到 0.9。

**你的第一个问题：** 这个结果让你意外吗？在继续之前，先写下你的直觉：这看起来“像推理”还是“像算术”？在你回答之前，不要往下读。

<div class="center">

------------------------------------------------------------------------

</div>

### 第二步：系统性地测试，然后寻找反例

从下面四类中每类挑 3 对，依次测试：

**类别 1：性别**——把 man -> woman 的类比应用到：king、doctor、professor、nurse、engineer……

**类别 2：首都**——france -> paris，应用到：china、japan、germany、usa……

**类别 3：时态**——go -> went 的类比，应用到：run、swim、eat、think……

**类别 4：反义词**——good -> bad 的类比，应用到：big、hot、fast、happy……

对每个测试案例，记录：- 你预期是什么 - 实际返回的前 5 个最近邻 - 余弦相似度分数

**你的第二个问题：** 哪一类成功率最高？哪一类最低？你能猜到原因吗——是训练语料的问题，还是这类关系本来就不适合被编码为线性位移？

<div class="center">

------------------------------------------------------------------------

</div>

### 第三步：亲手复现职业偏见实验

测试第七节描述的那个失败案例：

    doctor - man + woman ≈ ?

然后扩展测试：- engineer - man + woman ≈ ? - nurse - woman + man ≈ ? - CEO - man + woman ≈ ? - janitor - woman + man ≈ ?

记录每一个结果。

**你的第三个问题：** 对于那些输出明显有偏的词对，你的解释是什么？这些偏见是词向量系统的 bug，还是训练语料真实统计结构的如实反映？如果是后者，那么“去偏见”到底意味着什么？

<div class="center">

------------------------------------------------------------------------

</div>

### 第四步：找到类比方程的几何条件

做一个更精确的测试：类比方程什么时候精确成立，什么时候严重失效？

搭一个这样的实验：

对于一组你自己选择的类比对，计算两个量：

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def analogy_precision(model, analogies):
    """
    Measure analogy precision in bulk.
    analogies: a list of (A, B, C, D) tuples, each encoding an analogy of the form A:B :: C:D.
    Returns: detailed measurements for each analogy.
    """
    results = []

    for (A, B, C, D) in analogies:
        # Compute the predicted vector: vec(B) - vec(A) + vec(C)
        predicted_vector = model[B] - model[A] + model[C]

        # Get the whole vocabulary ranked by similarity (input words excluded after filtering)
        exclude = {A.lower(), B.lower(), C.lower()}
        neighbors = model.similar_by_vector(predicted_vector, topn=len(model.key_to_index))
        # Filter out the input words to get a ranked word list
        filtered_neighbors = [(word, score) for word, score in neighbors
                               if word.lower() not in exclude]

        # Actual nearest neighbor rank: the position of D in the similarity ranking
        rank_of_D = None
        for idx, (word, score) in enumerate(filtered_neighbors, 1):
            if word.lower() == D.lower():
                rank_of_D = idx
                break

        # Compute offset consistency: the cosine similarity of vec(B) - vec(A) and vec(D) - vec(C)
        # This directly measures whether the two pairs have the same "relationship direction"
        offset_AB = model[B] - model[A]
        offset_CD = model[D] - model[C]
        offset_consistency = cosine_similarity(offset_AB, offset_CD)

        results.append({
            "analogy": f"{A}:{B} :: {C}:{D}",
            "top1_hit": rank_of_D == 1,           # whether D ranks first (exact hit)
            "top5_hit": rank_of_D is not None and rank_of_D <= 5,  # whether D appears in the top 5
            "rank_of_D": rank_of_D,
            "offset_consistency": offset_consistency,  # offset consistency (a geometric measure)
        })

    return results


# ── Example test suite ────────────────────────────────────────────────────
test_analogies = [
    ("man",    "king",    "woman",  "queen"),    # gender x royalty
    ("france", "paris",   "japan",  "tokyo"),    # country x capital
    ("go",     "went",    "run",    "ran"),      # verb tense
    ("good",   "bad",     "big",    "small"),    # antonyms
    ("man",    "doctor",  "woman",  "nurse"),    # occupational bias case
]

precision_results = analogy_precision(model, test_analogies)

print(f"{'Analogy':30s} {'Top-1 hit':^8s} {'Top-5 hit':^8s} {'Rank':^6s} {'Offset consistency':^10s}")
print("-" * 70)
for r in precision_results:
    print(f"{r['analogy']:30s} "
          f"{'✓' if r['top1_hit'] else '✗':^8s} "
          f"{'✓' if r['top5_hit'] else '✗':^8s} "
          f"{str(r['rank_of_D']):^6s} "
          f"{r['offset_consistency']:^10.4f}")
```

**你的第四个问题（核心问题）：** 当一个类比失败时，“偏移一致性”分数是高还是低？也就是说，几何结构（相同的位移方向）是否仍然成立，而最近邻检索却失败了？如果是这样，失败到底在几何上，还是在检索方法上？这说明了第七节的哪个论断？

<div class="center">

------------------------------------------------------------------------

</div>

### 第五步：构造一个让余弦相似度撒谎的例子

余弦相似度衡量的是方向，不是长度。找两个词，它们的余弦相似度很高，你却觉得它们在语义上相距很远——或者反过来：余弦相似度很低，但你觉得它们应该在语义上很接近。

```python
def find_counterintuitive_similarities(model, word, topn=20, low_threshold=0.3):
    """
    For a given word W:
    1. Find the topn words with the highest cosine similarity, and look for any that feel out of place
    2. Find words with cosine similarity below low_threshold, and check whether any are semantically related yet geometrically distant
    model: a gensim KeyedVectors object
    """
    print(f"=== Cosine similarity analysis for word '{word}' ===\n")

    # Find the topn words with the highest cosine similarity
    top_neighbors = model.most_similar(word, topn=topn)
    print(f"Top {topn} words by cosine similarity (look for any that feel out of place):")
    for w, score in top_neighbors:
        print(f"  {w:20s}  similarity = {score:.4f}")

    print()

    # Check whether any low-similarity words still seem semantically related to you
    # Method: compute the similarity directly for a set of candidate words
    candidate_words = ["science", "art", "mathematics", "philosophy",
                       "literature", "music", "history", "biology"]
    print(f"Cosine similarity of candidate words (below {low_threshold} but semantically related?):")
    for candidate in candidate_words:
        if candidate in model:
            sim = model.similarity(word, candidate)
            flag = "  <- semantically related but distant?" if sim < low_threshold else ""
            print(f"  {candidate:20s}  similarity = {sim:.4f}{flag}")


# ── Example run ───────────────────────────────────────────────────────────
find_counterintuitive_similarities(model, word="physics", topn=20, low_threshold=0.3)
```

**你的第五个问题：** 余弦相似度没有捕获哪一类语义关系？这和第八节“分布假设的极限”有什么联系？

<div class="center">

------------------------------------------------------------------------

</div>

### 检验标准

完成这个练习后，你应该有三样东西：

1.  一张表：你测试过的每一个类比、成功/失败，以及你的解释

2.  一个偏见案例：你亲手复现的、让你不舒服的结果，以及你对它的立场

3.  一个反直觉的余弦相似度案例：向量几何与语言直觉之间真实存在的裂缝

第三样最重要。那条裂缝，就是向量表示与真正推理之间的距离。

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space* — word2vec 原始论文：CBOW 与 Skip-gram 架构，`-> [arXiv:1301.3781]`

- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality* — 负采样、短语向量与组合性，`-> [arXiv:1310.4546]`

- Fournier, Dupoux & Dunbar (2020). *Analogies minus analogy test* — 对类比测试的批判性分析，表明标准评估有缺陷，`-> [arXiv:2010.03446]`

- Firth, J.R. (1957). *A synopsis of linguistic theory 1930-55* — 分布假设的哲学起点：“你可以通过一个词交往的朋友来认识它”

- Searle, J. (1980). *Minds, Brains, and Programs* — 中文房间思想实验，对符号操作能否产生理解的哲学攻击

- Bolukbasi et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* — 词嵌入偏见问题的经典实证研究

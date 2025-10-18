# 词嵌入与位置编码

### 1. 输入处理：词嵌入与位置编码

-   **词嵌入（Input Embedding）**：与之前的模型一样，输入文本首先通过词嵌入层，将每个词或词元（token）转换为一个固定维度的向量。
-   **位置编码（Positional Encoding）**：自注意力机制本身无法感知序列中词语的顺序，为了解决这个问题，模型必须引入一种能表达单词位置信息的方式。位置编码的核心思想是：为输入序列中的每个位置创建一个唯一的、固定维度的向量（位置向量），然后将这个位置向量与对应位置的词嵌入向量相加，得到一个既包含语义又包含位置信息的新向量，作为模型真正的输入。

    **正弦/余弦编码公式**

    原始论文提出了一种使用不同频率的正弦和余弦函数来生成位置向量的方案：
    $$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
    $$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
    - `pos` 是单词在序列中的位置。
    - `i` 是位置向量中的维度索引。
    - `d_model` 是嵌入维度。

    这个公式最巧妙的一点在于，它使得模型能够轻易地学习到**相对位置信息**。因为对于任何固定的偏移量 `k`，`PE(pos+k)` 都可以表示为 `PE(pos)` 的一个线性变换。这意味着模型可以学会识别“后面第 k 个词”这样的相对关系，而无论当前词的位置 `pos` 在哪里。

## 附录：位置编码的线性变换特性

这个结论之所以成立，是因为位置编码使用了**三角函数的和角公式**。

### 1. 回顾公式

首先，我们有两个公式，分别用于计算位置向量中的偶数维度 (`2i`) 和奇数维度 (`2i+1`)：

-   $PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{2i/d_{\text{model}}}})$
-   $PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{2i/d_{\text{model}}}})$

为了方便，我们把频率项（分母）设为 $\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}} $。
那么公式就变成：

-   $PE_{(pos, 2i)} = \sin(pos \cdot \omega_i)$
-   $PE_{(pos, 2i+1)} = \cos(pos \cdot \omega_i)$

### 2. 证明线性关系

现在，我们想知道 `PE(pos+k)` 和 `PE(pos)` 之间的关系。我们来看 `pos+k` 位置的编码：

-   $PE_{(pos+k, 2i)} = \sin((pos+k) \cdot \omega_i)$
-   $PE_{(pos+k, 2i+1)} = \cos((pos+k) \cdot \omega_i)$

这里就是关键！我们可以使用三角函数的**和角公式**：
-   $\sin(A+B) = \sin(A)\cos(B) + \cos(A)\sin(B)$
-   $\cos(A+B) = \cos(A)\cos(B) - \sin(A)\sin(B)$

把 $A = pos \cdot \omega_i$ 和 $B = k \cdot \omega_i$ 代入：

$PE_{(pos+k, 2i)} = \sin(pos \cdot \omega_i)\cos(k \cdot \omega_i) + \cos(pos \cdot \omega_i)\sin(k \cdot \omega_i)$
$PE_{(pos+k, 2i+1)} = \cos(pos \cdot \omega_i)\cos(k \cdot \omega_i) - \sin(pos \cdot \omega_i)\sin(k \cdot \omega_i)$

现在，我们把 $PE_{(pos, ...)}$ 的定义代回去：

$PE_{(pos+k, 2i)} = PE_{(pos, 2i+1)} \cdot \sin(k \cdot \omega_i) + PE_{(pos, 2i)} \cdot \cos(k \cdot \omega_i)$
$PE_{(pos+k, 2i+1)} = PE_{(pos, 2i+1)} \cdot \cos(k \cdot \omega_i) - PE_{(pos, 2i)} \cdot \sin(k \cdot \omega_i)$

我们可以把这个关系写成矩阵乘法的形式。对于每一个维度对 `(2i, 2i+1)`：

$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(k \cdot \omega_i) & \sin(k \cdot \omega_i) \\ -\sin(k \cdot \omega_i) & \cos(k \cdot \omega_i) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

这个 2x2 的矩阵就是一个**旋转矩阵**。

**结论**：
`PE(pos+k)` 在 `(2i, 2i+1)` 这两个维度上的值，可以通过将 `PE(pos)` 在相同维度上的值乘以一个 2x2 的矩阵得到。这个矩阵**只依赖于偏移量 `k`**，而与原始位置 `pos` 无关。

将所有维度对的变换组合在一起，就可以得到一个大的、与 `pos` 无关的变换矩阵 `M_k`，使得：

$PE_{pos+k} = M_k \cdot PE_{pos}$

这就是**线性变换**的定义。

### 3. 这为什么重要？

这个特性意味着，对于模型来说，从位置 `pos` 移动到 `pos+k` 的“操作”是**一致的**，无论 `pos` 是 5 还是 50。

当自注意力机制计算两个词（比如在 `pos` 和 `pos+k`）之间的关联度时，它会用到这两个词的 Query 和 Key 向量，而这些向量都包含了相应的位置编码。因为 `PE(pos+k)` 和 `PE(pos)` 之间存在固定的线性关系，模型可以非常容易地学会**只关注相对位置 `k`**，而不是绝对位置 `pos`。

这使得模型具有了**位置不变性**，能够将学到的关于相对位置的知识（例如，“动词后面的名词很可能是宾语”）应用到句子的任何地方。

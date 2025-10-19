# Transformer 推理过程详解

为了更好地理解 Transformer 的内部工作机制，我们将通过一个简化的示例句子，一步步推演其在推理（或称生成）过程中的计算。

**示例任务**：将英文句子 "The cat sat" 翻译成中文。

我们将重点关注编码器和解码器如何协同工作，以自回归的方式（一次一个词）生成翻译结果。

## 1. 编码器计算过程

编码器的任务是处理整个输入句子，并为每个词生成一个上下文相关的表示。这个过程在推理和训练中是相同的。

**示例句子**："The cat sat"

### a. 词元化 (Tokenization)

首先，输入句子会被拆分成词元（tokens）。

-   **词元**：`[The, cat, sat]`

### b. 词嵌入 (Input Embedding)

每个词元都会被转换成一个固定维度的向量。假设我们的词嵌入维度 $d_{model} = 4$。

-   `Embedding(The)` = $[0.1, 0.2, 0.3, 0.4]$
-   `Embedding(cat)` = $[0.5, 0.6, 0.7, 0.8]$
-   `Embedding(sat)` = $[0.9, 0.8, 0.7, 0.6]$

### c. 位置编码 (Positional Encoding)

由于自注意力机制本身无法感知词元在序列中的位置信息，我们需要添加位置编码。

-   `PositionalEncoding(0)` (for "The") = $[0.0, 0.1, 0.2, 0.3]$
-   `PositionalEncoding(1)` (for "cat") = $[0.4, 0.5, 0.6, 0.7]$
-   `PositionalEncoding(2)` (for "sat") = $[0.8, 0.9, 1.0, 1.1]$

**输入到编码器的向量 (Input to Encoder)**：

-   $X_{\text{The}} = \text{Embedding(The)} + \text{PositionalEncoding(0)} = [0.1, 0.3, 0.5, 0.7]$
-   $X_{\text{cat}} = \text{Embedding(cat)} + \text{PositionalEncoding(1)} = [0.9, 1.1, 1.3, 1.5]$
-   $X_{\text{sat}} = \text{Embedding(sat)} + \text{PositionalEncoding(2)} = [1.7, 1.7, 1.7, 1.7]$

## 2. 多头自注意力机制 (Multi-Head Self-Attention)

这是 Transformer 的核心。我们将以一个注意力头为例进行推演。假设每个头的维度 $d_k = d_v = d_{model} / \text{num\_heads} = 4 / 1 = 4$。

### a. 生成 Query, Key, Value 向量

每个输入向量 $X$ 都会通过三个不同的线性变换（乘以权重矩阵 $W_Q, W_K, W_V$）来生成对应的 Query (Q)、Key (K) 和 Value (V) 向量。

-   Query 矩阵

$$
W_Q =
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.5 & 0.6 & 0.7 & 0.8 \\
0.9 & 1.0 & 1.1 & 1.2 \\
1.3 & 1.4 & 1.5 & 1.6
\end{bmatrix}
$$

-   Key 矩阵
$$
W_K =
\begin{bmatrix}
1.6 & 1.5 & 1.4 & 1.3 \\
1.2 & 1.1 & 1.0 & 0.9 \\
0.8 & 0.7 & 0.6 & 0.5 \\
0.4 & 0.3 & 0.2 & 0.1
\end{bmatrix}
$$

-   Value 矩阵
$$
W_V =
\begin{bmatrix}
0.2 & 0.4 & 0.6 & 0.8 \\
1.0 & 1.2 & 1.4 & 1.6 \\
1.8 & 2.0 & 2.2 & 2.4 \\
2.6 & 2.8 & 3.0 & 3.2
\end{bmatrix}
$$

**计算 Q, K, V 向量** (以 "The" 为例):

-   $Q_{\text{The}} = X_{\text{The}} \cdot W_Q = [1.5, 1.7, 1.9, 2.1]$
-   $K_{\text{The}} = X_{\text{The}} \cdot W_K = [1.0, 0.9, 0.8, 0.7]$
-   $V_{\text{The}} = X_{\text{The}} \cdot W_V = [3.0, 3.4, 3.8, 4.2]$

("cat" 和 "sat" 的 Q, K, V 向量计算方式相同。)

### b. 计算注意力分数 (Attention Scores)

每个 Query 向量会与所有 Key 向量进行点积，然后除以 $\sqrt{d_k}$ 进行缩放。以 $Q_{\text{cat}}$ 为例：

-   $Q_{\text{cat}} = X_{\text{cat}} \cdot W_Q = [5.5, 6.1, 6.7, 7.3]$
-   $K_{\text{The}} = [1.0, 0.9, 0.8, 0.7]$
-   $K_{\text{cat}} = X_{\text{cat}} \cdot W_K = [5.0, 4.5, 4.0, 3.5]$
-   $K_{\text{sat}} = X_{\text{sat}} \cdot W_K = [10.2, 9.5, 8.8, 8.1]$

**计算缩放后分数 (Scaled Scores)**：

-   $\text{ScaledScore}(Q_{\text{cat}}, K_{\text{The}}) = (Q_{\text{cat}} \cdot K_{\text{The}}) / 2 = 10.73$
-   $\text{ScaledScore}(Q_{\text{cat}}, K_{\text{cat}}) = (Q_{\text{cat}} \cdot K_{\text{cat}}) / 2 = 53.65$
-   $\text{ScaledScore}(Q_{\text{cat}}, K_{\text{sat}}) = (Q_{\text{cat}} \cdot K_{\text{sat}}) / 2 = 116.07$

### c. Softmax 归一化

将分数通过 Softmax 函数转换为注意力权重。

-   $\text{AttentionWeights}(Q_{\text{cat}}) = \text{softmax}([10.73, 53.65, 116.07]) \approx [0.0001, 0.0009, 0.9990]$

### d. 计算加权和 (Weighted Sum)

将注意力权重与对应的 Value 向量相乘并求和。

-   $V_{\text{The}} = [3.0, 3.4, 3.8, 4.2]$
-   $V_{\text{cat}} = X_{\text{cat}} \cdot W_V = [5.5, 6.1, 6.7, 7.3]$
-   $V_{\text{sat}} = X_{\text{sat}} \cdot W_V = [10.2, 11.0, 11.8, 12.6]$

**输出向量 (Output for $Q_{\text{cat}}$)**：

$\text{Output}_{\text{cat}} = 0.0001 \cdot V_{\text{The}} + 0.0009 \cdot V_{\text{cat}} + 0.9990 \cdot V_{\text{sat}} \approx [10.1953, 11.0008, 11.7864, 12.5870]$

## 3. 残差连接与层归一化 (Add & Norm)

自注意力层的输出会与原始输入 $X_{\text{cat}}$ 进行残差连接，然后进行层归一化。

-   **残差连接**：$Y_{\text{cat}} = X_{\text{cat}} + \text{Output}_{\text{cat}}$
-   **层归一化**：对 $Y_{\text{cat}}$ 进行归一化处理。

## 4. 位置前馈网络 (Position-wise Feed-Forward Network)

经过 Add & Norm 后的向量会独立地通过一个两层的全连接前馈网络。

经过以上步骤，编码器为 "The cat sat" 输出一个上下文向量序列 $C = [C_{\text{The}}, C_{\text{cat}}, C_{\text{sat}}]$。

---

## 5. 解码器推理过程：自回归生成

解码器的任务是根据编码器的输出 $C$ 和已生成的部分序列，逐步生成下一个词元。

### 步骤 1: 生成第一个词

#### a. 解码器输入

解码器在第一个时间步只接收一个特殊的 `<sos>` (start of sentence) 标记作为输入。

-   `Embedding(<sos>)` 和 `PositionalEncoding(0)` 被计算并相加，得到 $X'_{\text{<sos>}}$。

#### b. 带掩码的多头自注意力

解码器首先对自己当前的输入（只有 `<sos>`）进行自注意力计算。此时，掩码没有实际作用，因为它只能关注自己。

#### c. 编码器-解码器注意力 (交叉注意力)

-   **Query (Q)**：来自解码器前一个子层（带掩码自注意力层）的输出。
-   **Key (K) 和 Value (V)**：来自**编码器的最终输出** $C = [C_{\text{The}}, C_{\text{cat}}, C_{\text{sat}}]$。

解码器使用它的 Query 来关注编码器输出的整个输入序列信息，判断哪个输入词对于生成第一个目标词最重要。

#### d. 输出层

经过前馈网络、残差连接和层归一化后，解码器最终输出一个 logits 向量。该向量通过 Softmax 函数转换为词汇表中每个词的概率。

-   $P(\text{next token}) = \text{softmax}(Output_{\text{logits}})$

假设概率最高的词是 "猫"。模型选择 "猫" 作为第一个输出。

### 步骤 2: 生成第二个词

#### a. 解码器输入

现在，解码器的输入是之前已生成的所有词元序列，即 `[<sos>, 猫]`。

-   对 `<sos>` 和 "猫" 分别进行词嵌入和位置编码，得到 $X'_{\text{<sos>}}$ 和 $X'_{\text{猫}}$。

#### b. 带掩码的多头自注意力

解码器对 `[<sos>, 猫]` 序列进行自注意力计算。在这里，**掩码** 发挥关键作用：
-   在计算 `<sos>` 的表示时，它只能关注自己。
-   在计算 "猫" 的表示时，它可以关注 `<sos>` 和 "猫" 自己。
掩码确保了在预测当前词时，不会使用到未来的信息。

#### c. 编码器-解码器注意力

与步骤 1 类似，解码器在 "猫" 这个位置的输出会生成一个新的 Query，再次去关注编码器的输出 $C$，以获取生成下一个词所需的信息。

#### d. 输出层

模型在当前位置（"猫" 之后）预测下一个词的概率分布。假设概率最高的词是 "坐着"。

### 步骤 3: 迭代生成

解码器会重复这个过程：将新生成的词元（"坐着"）添加到输入序列中，然后预测下一个词，直到生成一个特殊的 `<eos>` (end of sentence) 标记或达到预设的最大长度。

## 总结

在推理过程中，解码器通过一个自回归循环，一步步构建输出序列。它结合了带掩码的自注意力（关注已生成的输出）和交叉注意力（关注原始输入序列），从而能够生成连贯且与输入相关的文本。

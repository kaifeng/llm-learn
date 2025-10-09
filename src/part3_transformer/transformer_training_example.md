# Transformer 训练过程详解

为了更好地理解 Transformer 如何学习，我们将通过一个简化的示例，一步步推演其在训练过程中的计算。训练的目标是调整模型参数，使其能够根据输入序列预测出正确的目标序列。

**示例任务**：将英文句子 "The cat sat" 翻译成中文 "猫 坐着 <eos>"。

我们将重点关注模型如何在一个训练步骤中处理这对句子。

## 1. 编码器计算过程

编码器的作用与推理时完全相同：为输入序列的每个词元生成一个富含上下文的表示。

**输入句子**："The cat sat"

### a. 词元化 (Tokenization)

-   **词元**：`[The, cat, sat]`

### b. 词嵌入 (Input Embedding)

每个词元被转换成一个向量。假设 $d_{model} = 4$。

-   `Embedding(The)` = $[0.1, 0.2, 0.3, 0.4]$
-   `Embedding(cat)` = $[0.5, 0.6, 0.7, 0.8]$
-   `Embedding(sat)` = $[0.9, 0.8, 0.7, 0.6]$

### c. 位置编码 (Positional Encoding)

添加位置信息。

-   `PositionalEncoding(0)` = $[0.0, 0.1, 0.2, 0.3]$
-   `PositionalEncoding(1)` = $[0.4, 0.5, 0.6, 0.7]$
-   `PositionalEncoding(2)` = $[0.8, 0.9, 1.0, 1.1]$

**输入到编码器的向量**：

-   $X_{\text{The}} = [0.1, 0.3, 0.5, 0.7]$
-   $X_{\text{cat}} = [0.9, 1.1, 1.3, 1.5]$
-   $X_{\text{sat}} = [1.7, 1.7, 1.7, 1.7]$

### d. 多头自注意力及后续层

编码器通过多头自注意力、残差连接、层归一化和前馈网络处理这些输入向量。这个过程与推理时完全一致。最终，编码器为输入序列 "The cat sat" 输出一个上下文向量序列 $C = [C_{\text{The}}, C_{\text{cat}}, C_{\text{sat}}]$。这个输出将用于解码器的每一个时间步。

---

## 2. 解码器训练过程：教师强制 (Teacher Forcing)

训练与推理的关键区别在于解码器的处理方式。训练时，我们使用一种称为 **教师强制 (Teacher Forcing)** 的技术，即直接向解码器提供完整的目标序列，而不是像推理时那样一次生成一个词。这使得模型可以在每个位置上并行计算损失，极大地提高了训练效率。

**目标序列**：`[<sos>, 猫, 坐着]`

### a. 解码器输入

解码器接收整个目标序列作为输入，并对其进行词嵌入和位置编码。

-   **输入词元**：`[<sos>, 猫, 坐着]`
-   **词嵌入与位置编码**：与编码器类似，每个词元都会被转换为一个向量，并加上其对应的位置编码。
    -   $X'_{\text{<sos>}} = \text{Embedding(<sos>)} + \text{PositionalEncoding(0)}$
    -   $X'_{\text{猫}} = \text{Embedding(猫)} + \text{PositionalEncoding(1)}$
    -   $X'_{\text{坐着}} = \text{Embedding(坐着)} + \text{PositionalEncoding(2)}$

### b. 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)

这是教师强制的核心。解码器在处理目标序列时，必须防止它“偷看”未来的词元。例如，在预测 "坐着" 时，模型只能使用 `<sos>` 和 "猫" 的信息，而不能使用 "坐着" 本身的信息。

这是通过一个 **前瞻掩码 (Look-ahead Mask)** 实现的。该掩码会作用于自注意力机制的得分矩阵上，将所有未来位置的注意力得分设置为一个极大的负数（例如 $-\infty$），这样在 Softmax 之后，这些位置的注意力权重就变成了 0。

**注意力得分矩阵（应用掩码前）**:
$$
\begin{bmatrix}
\text{score}(Q_1, K_1) & \text{score}(Q_1, K_2) & \text{score}(Q_1, K_3) \\
\text{score}(Q_2, K_1) & \text{score}(Q_2, K_2) & \text{score}(Q_2, K_3) \\
\text{score}(Q_3, K_1) & \text{score}(Q_3, K_2) & \text{score}(Q_3, K_3) \\
\end{bmatrix}
$$
(其中 1, 2, 3 分别对应 `<sos>`, `猫`, `坐着`)

**应用掩码后的得分矩阵**:
$$
\begin{bmatrix}
\text{score}(Q_1, K_1) & - \infty & - \infty \\
\text{score}(Q_2, K_1) & \text{score}(Q_2, K_2) & - \infty \\
\text{score}(Q_3, K_1) & \text{score}(Q_3, K_2) & \text{score}(Q_3, K_3) \\
\end{bmatrix}
$$

经过 Softmax 后，模型在每个位置的输出只依赖于之前和当前位置的词元。

### c. 编码器-解码器注意力 (Encoder-Decoder Attention)

这一步与推理时类似。解码器的每个位置都会计算一个 Query 向量，并用它去关注编码器输出的所有 Key-Value 对（即 $C = [C_{\text{The}}, C_{\text{cat}}, C_{\text{sat}}]$）。这使得解码器在生成每个目标词元时，都能从整个输入序列中提取相关信息。

由于解码器的所有位置是并行计算的，所以这一步也是并行完成的。

### d. 前馈网络与输出层

经过所有子层后，解码器为目标序列中的每个位置都生成一个最终的输出向量。这些向量会通过一个最终的线性层和 Softmax 函数，得到在每个位置上预测下一个词的概率分布。

-   **位置 1 (`<sos>`) 的输出** -> 预测 "猫" 的概率分布
-   **位置 2 (`猫`) 的输出** -> 预测 "坐着" 的概率分布
-   **位置 3 (`坐着`) 的输出** -> 预测 `<eos>` 的概率分布

## 3. 损失计算与反向传播

训练的核心是让模型的预测尽可能接近真实标签。

-   **目标标签**：`[猫, 坐着, <eos>]` (即目标序列向左移动一位)
-   **损失函数**：通常使用 **交叉熵损失 (Cross-Entropy Loss)**。对于每个位置，我们计算模型预测的概率分布与真实标签（一个 one-hot 向量）之间的损失。

**总损失 (Total Loss)** = $\sum_{i} \text{CrossEntropyLoss}(\text{Prediction}_i, \text{Label}_i)$

计算出总损失后，通过 **反向传播 (Backpropagation)** 算法计算损失函数关于模型所有参数（如 $W_Q, W_K, W_V$ 等）的梯度。最后，使用优化器（如 Adam）根据这些梯度来更新参数。

重复这个过程成千上万次，模型就会逐渐学会如何将输入序列正确地翻译成目标序列。

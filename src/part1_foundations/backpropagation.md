# 反向传播的梯度计算

反向传播（Backpropagation）是神经网络训练的核心算法，其本质是利用链式法则计算损失函数对各层参数的梯度。
下面通过一个具体的两层神经网络实例，展示梯度计算的全过程。

### 一、神经网络结构与参数设定
**网络结构** ：输入层（2节点）→ 隐藏层（2节点）→ 输出层（1节点）
**激活函数** ：隐藏层和输出层均使用sigmoid函数：
$\sigma(x)=1+e^{-x}1​,\sigma'(x)=\sigma(x)(1−\sigma(x))$

**输入数据** ：$X = [1, 2]$，真实标签：$y = 0.5$

**初始参数** ：
隐藏层权重：$$W_1 = \begin{bmatrix}0.5 & 0.6 \\ 0.7 & 0.8\end{bmatrix}$$，偏置：$b_1 = [0.1, 0.2]$
输出层权重：$W_2 = [0.9, 0.1]$，偏置：$b_2 = 0.3$

### 二、前向传播计算（Forward Propagation）

1. **隐藏层输入与输出**

隐藏层加权和：

$z_1 = X \cdot W_1 + b_1 = [1, 2] \cdot \begin{bmatrix}0.5 & 0.6 \\ 0.7 & 0.8\end{bmatrix} + [0.1, 0.2] = [1 \times 0.5 + 2 \times 0.7 + 0.1, \, 1 \times 0.6 + 2 \times 0.8 + 0.2] = [2.0, 2.4]$

隐藏层激活值：
$a_1 = \sigma(z_1) = \left[\frac{1}{1+e^{-2.0}}, \frac{1}{1+e^{-2.4}}\right] \approx [0.8808, 0.9084]$

2. **输出层输入与输出**

输出层加权和：

$z_2 = a_1 \cdot W_2^T + b_2 = [0.8808, 0.9084] \cdot [0.9, 0.1]^T + 0.3 = 0.8808 \times 0.9 + 0.9084 \times 0.1 + 0.3 \approx 1.1835$

输出层预测值：

$\hat{y} = \sigma(z_2) = \frac{1}{1+e^{-1.1835}} \approx 0.7692$

### 三、反向传播计算（Backpropagation）

**损失函数** ：均方误差（MSE）

$L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.7692 - 0.5)^2 \approx 0.035$

1. **计算输出层的误差项（δ₂）**

损失对输出的导数：

$\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y) = 0.7692 - 0.5 = 0.2692$

输出对加权和的导数（sigmoid导数）：

$\frac{\partial \hat{y}}{\partial z_2} = \sigma'(z_2) \approx 0.7692 \times (1 - 0.7692)\approx 0.1776$

输出层误差项（链式法则）：

$\delta{_2} = \frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \approx 0.2692 \times 0.1776 \approx 0.0478$

2. **计算隐藏层到输出层的权重梯度（∂L/∂W₂）和偏置梯度（∂L/∂b₂）**

权重梯度：$\frac{\partial L}{\partial W_2} = \delta{_2} \cdot a_1 = 0.0478 \times [0.8808, 0.9084] \approx [0.0421, 0.0434]$

偏置梯度：$\frac{\partial L}{\partial b_2} = \delta{_2} \approx 0.0478$

3. **计算隐藏层的误差项（δ₁）**

损失对隐藏层激活值的导数：

$\frac{\partial L}{\partial a_1} = \delta{_2} \cdot W_2 = 0.0478 \times [0.9, 0.1] \approx [0.0430, 0.0048]$

隐藏层激活对加权和的导数（sigmoid导数）：

$\frac{\partial a_1}{\partial z_1} = \sigma'(z_1) \approx [0.8808 \times 0.1192, \, 0.9084 \times 0.0916] \approx [0.1051, 0.0833]$

隐藏层误差项：

$\delta{_1} = \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \approx [0.0430 \times 0.1051, \, 0.0048 \times 0.0833] \approx [0.0045, 0.0004]$

4. **计算输入层到隐藏层的权重梯度（∂L/∂W₁）和偏置梯度（∂L/∂b₁）**

- 权重梯度：$\frac{\partial L}{\partial W_1} = \delta{_1}^T \cdot X = \begin{bmatrix}0.0045 \\ 0.0004\end{bmatrix} \cdot [1, 2] = \begin{bmatrix}0.0045 \times 1 & 0.0045 \times 2 \\ 0.0004 \times 1 & 0.0004 \times 2\end{bmatrix} \approx \begin{bmatrix}0.0045 & 0.0090 \\ 0.0004 & 0.0008\end{bmatrix}$
- 偏置梯度：$\frac{\partial L}{\partial b_1} = \delta{_1} \approx [0.0045, 0.0004]$

### 四、梯度汇总与参数更新

|     参数 |                                                   梯度值（近似） |
|:-------|:-----------------------------------------------------------------|
| $W_2$ |                                               $[0.0421, 0.0434]$ |
| $b_2$ |                                                         $0.0478$ |
| $W_1$ | $\begin{bmatrix}0.0045 & 0.0090 \\ 0.0004 & 0.0008\end{bmatrix}$ |
| $b_1$ |                                               $[0.0045, 0.0004]$ |

**参数更新（假设学习率η=0.1）**：

- $W_2 \leftarrow W_2 - \eta \cdot \frac{\partial L}{\partial W_2} \approx [0.9-0.0042, 0.1-0.0043] = [0.8958, 0.0957]$
- $b_2 \leftarrow b_2 - \eta \cdot \frac{\partial L}{\partial b_2} \approx 0.3 - 0.0048 = 0.2952$
- $W_1 \leftarrow W_1 - \eta \cdot \frac{\partial L}{\partial W_1} \approx \begin{bmatrix}0.5-0.0005 & 0.6-0.0009 \\ 0.7-0.00004 & 0.8-0.00008\end{bmatrix} \approx \begin{bmatrix}0.4995 & 0.5991 \\ 0.6996 & 0.7999\end{bmatrix}$
- $b_1 \leftarrow b_1 - \eta \cdot \frac{\partial L}{\partial b_1} \approx [0.1-0.0005, 0.2-0.00004] = [0.0995, 0.19996]$

### 五、反向传播核心逻辑总结

1. **前向传播**：从输入层到输出层计算各层激活值，得到预测值。
2. **误差反向传递**：从输出层开始，利用链式法则计算各层误差项（δ），其中：
- 输出层误差：$\delta{_2} = (\hat{y}-y) \cdot \sigma'(z_2)$
- 隐藏层误差：$\delta{_k} = \delta{_{k+1}} \cdot W_{k+1}^T \cdot \sigma'(z_k)$（k为隐藏层序号）
3. **梯度计算**：
- 权重梯度：$\frac{\partial L}{\partial W_k} = \delta{_k}^T \cdot a_{k-1}$
- 偏置梯度：$\frac{\partial L}{\partial b_k} = \delta{_k}$

可以看出，反向传播的本质是将输出误差逐层分解为各层参数的梯度，从而为梯度下降优化提供方向。

## 反向传播中的常见问题：梯度消失与梯度爆炸

在深度神经网络的训练过程中，尤其是在网络层数较深或使用某些特定的激活函数时，反向传播算法可能会遇到两个主要问题：**梯度消失（Vanishing Gradient）**和**梯度爆炸（Exploding Gradient）**。

### 梯度消失 (Vanishing Gradient)

- **现象**：在反向传播过程中，梯度值随着网络层数的增加而指数级减小，导致靠近输入层的网络层（即浅层）的参数更新变得非常缓慢甚至停滞。这意味着浅层网络几乎无法学习到有用的特征。
- **原因**：
    1.  **链式法则的乘法效应**：在反向传播中，梯度是通过链式法则逐层相乘得到的。如果每层的梯度（例如，激活函数的导数）都小于 1，那么经过多层相乘后，梯度会迅速趋近于 0。
    2.  **激活函数**：传统的激活函数如 Sigmoid 和 Tanh，在输入值过大或过小时，其导数会非常接近 0。这使得这些区域的梯度几乎为零，导致梯度无法有效传播。
- **影响**：深层网络难以训练，模型无法捕捉长程依赖（在 RNN 中尤为明显）。
- **解决方法**：
    -   使用 ReLU 及其变体（如 Leaky ReLU）作为激活函数。
    -   使用残差连接（如 ResNet）。
    -   使用批归一化（Batch Normalization）。
    -   使用 LSTM 和 GRU 等门控循环单元。

### 梯度爆炸 (Exploding Gradient)

- **现象**：与梯度消失相反，梯度值在反向传播过程中指数级增大，导致参数更新过大，模型权重剧烈震荡，甚至溢出（NaN），使得训练过程不稳定，模型无法收敛。
- **原因**：
    1.  **链式法则的乘法效应**：如果每层的梯度（例如，激活函数的导数或权重矩阵的范数）都大于 1，那么经过多层相乘后，梯度会迅速变得非常大。
    2.  **不合适的权重初始化**：如果初始权重过大，也容易导致梯度爆炸。
- **影响**：训练不稳定，损失函数值变为 NaN，模型无法学习。
- **解决方法**：
    -   **梯度裁剪 (Gradient Clipping)**：当梯度超过某个阈值时，将其限制在一个预设的最大值内。
    -   使用更小的学习率。
    -   使用批归一化（Batch Normalization）。
    -   使用合适的权重初始化策略。
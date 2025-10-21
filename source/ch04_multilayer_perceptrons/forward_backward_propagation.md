# 前向传播、反向传播和计算图

## 前向传播

前向传播（Forward Propagation）是指数据从输入层流向输出层的过程。在这个过程中，每一层的神经元接收来自前一层的输出，进行加权求和并通过激活函数，然后将结果传递给下一层。

这个过程最终在输出层产生模型的预测值。

## 反向传播

反向传播（Backpropagation）是神经网络训练的核心算法，其本质是利用**链式法则**高效计算损失函数对网络中每一个参数（权重和偏置）的梯度。

计算出梯度后，优化算法（如梯度下降）就可以利用这些梯度来更新参数，从而最小化损失函数。

### 计算过程示例

下面通过一个具体的两层神经网络实例，展示梯度计算的全过程。

**1. 网络结构与参数设定**
- **网络结构** ：输入层（2节点）→ 隐藏层（2节点）→ 输出层（1节点）
- **激活函数** ：隐藏层和输出层均使用sigmoid函数：
$\sigma(x)=1/(1+e^{-x}), \sigma'(x)=\sigma(x)(1-\sigma(x))$

- **输入数据** ：$X = [1, 2]$，真实标签：$y = 0.5$

- **初始参数** ：
  - 隐藏层权重：$W_1 = \begin{bmatrix}0.5 & 0.6 \\ 0.7 & 0.8\end{bmatrix}$，偏置：$b_1 = [0.1, 0.2]$
  - 输出层权重：$W_2 = [0.9, 0.1]$，偏置：$b_2 = 0.3$

**2. 前向传播计算**

- **隐藏层**
  - 加权和：$z_1 = X \cdot W_1 + b_1 = [2.0, 2.4]$
  - 激活值：$a_1 = \sigma(z_1) \approx [0.8808, 0.9084]$

- **输出层**
  - 加权和：$z_2 = a_1 \cdot W_2^T + b_2 \approx 1.1835$
  - 预测值：$\hat{y} = \sigma(z_2) \approx 0.7692$

**3. 反向传播计算**

- **损失函数** ：均方误差（MSE）
$L = \frac{1}{2}(\hat{y} - y)^2 \approx 0.035$

- **计算输出层的误差项（δ₂）**
  - 损失对输出的导数：$\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y) = 0.2692$
  - 输出对加权和的导数：$\frac{\partial \hat{y}}{\partial z_2} = \sigma'(z_2) \approx 0.1776$
  - 输出层误差项：$\delta{_2} = \frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \approx 0.0478$

- **计算隐藏层到输出层的梯度**
  - 权重梯度：$\frac{\partial L}{\partial W_2} = \delta{_2} \cdot a_1 \approx [0.0421, 0.0434]$
  - 偏置梯度：$\frac{\partial L}{\partial b_2} = \delta{_2} \approx 0.0478$

- **计算隐藏层的误差项（δ₁）**
  - 损失对隐藏层激活值的导数：$\frac{\partial L}{\partial a_1} = \delta{_2} \cdot W_2 \approx [0.0430, 0.0048]$
  - 隐藏层激活对加权和的导数：$\frac{\partial a_1}{\partial z_1} = \sigma'(z_1) \approx [0.1051, 0.0833]$
  - 隐藏层误差项：$\delta{_1} = \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \approx [0.0045, 0.0004]$

- **计算输入层到隐藏层的梯度**
  - 权重梯度：$\frac{\partial L}{\partial W_1} = \delta{_1}^T \cdot X = \begin{bmatrix}0.0045 \\ 0.0004\end{bmatrix} \cdot [1, 2] \approx \begin{bmatrix}0.0045 & 0.0090 \\ 0.0004 & 0.0008\end{bmatrix}$
  - 偏置梯度：$\frac{\partial L}{\partial b_1} = \delta{_1} \approx [0.0045, 0.0004]$

**4. 核心逻辑总结**
1. **前向传播**：计算预测值。
2. **误差反向传递**：从输出层开始，利用链式法则计算各层误差项（δ）。
3. **梯度计算**：利用误差项计算各层参数的梯度。

反向传播的本质是将输出误差逐层分解为各层参数的梯度，从而为梯度下降优化提供方向。

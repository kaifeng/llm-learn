# PyTorch：深度学习框架

PyTorch 是一个开源的机器学习框架，由 Facebook AI Research (FAIR) 开发。它以其灵活性、Pythonic 的接口和动态计算图而闻名，是研究和开发深度学习模型（包括大语言模型）的首选工具之一。

## 1. 核心概念

### 1.1 Tensor (张量)

-   **定义**：Tensor 是 PyTorch 中最基本的数据结构，类似于 NumPy 的 `ndarray`，但它支持 GPU 加速和自动微分。
-   **创建**：`torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`, `torch.randn()` 等。
-   **操作**：支持各种数学运算（加减乘除、矩阵乘法等）和切片、索引等操作，与 NumPy 类似。

### 1.2 Autograd (自动求导)

-   **定义**：PyTorch 的自动微分引擎，能够自动计算张量上的所有操作的梯度。这是训练神经网络（通过反向传播）的核心。
-   **`requires_grad`**：张量的属性，如果设置为 `True`，PyTorch 会跟踪其上的所有操作，以便后续计算梯度。
-   **`.backward()`**：在标量损失张量上调用此方法，PyTorch 会计算所有 `requires_grad=True` 的张量的梯度。
-   **`.grad`**：张量的属性，存储了该张量的梯度值。

### 1.3 `torch.nn.Module` (模块)

-   **定义**：所有神经网络模块（层、模型）的基类。通过继承 `nn.Module`，可以方便地组织网络结构、管理参数，并利用 PyTorch 的自动求导功能。
-   **`__init__`**：构造函数，用于定义网络层和子模块。
-   **`forward`**：定义数据在前向传播时如何通过网络层。

## 2. 常用模块与函数

### 2.1 数据处理 (`torch.utils.data`)

-   **`torch.utils.data.Dataset`**：一个抽象类，用于表示数据集。用户需要继承它并实现 `__len__` (返回数据集大小) 和 `__getitem__` (返回单个样本) 方法。
-   **`torch.utils.data.DataLoader`**：用于高效地加载数据。它支持批量处理、数据打乱、多进程加载等功能，是训练时数据输入管道的关键组件。

### 2.2 神经网络层 (`torch.nn`)

-   **`torch.nn.Linear`**：全连接层（或称线性层），实现 $y = xA^T + b$ 的线性变换。
-   **`torch.nn.Conv2d`**：二维卷积层，常用于图像处理。
-   **`torch.nn.ReLU`**, `torch.nn.Sigmoid`, `torch.nn.Tanh`：常用的激活函数。
-   **`torch.nn.Embedding`**：用于创建词嵌入层，将离散的整数索引映射到稠密的向量表示。
-   **`torch.nn.TransformerEncoderLayer`**, `torch.nn.TransformerDecoderLayer`, `torch.nn.TransformerEncoder`, `torch.nn.TransformerDecoder`, `torch.nn.Transformer`：PyTorch 提供了 Transformer 架构的各种组件，方便用户构建 Transformer 模型。

### 2.3 损失函数 (`torch.nn`)

-   **`torch.nn.MSELoss`**：**均方误差损失 (Mean Squared Error Loss)**，常用于回归问题。计算预测值与真实值之差的平方的平均值。
    ```python
    loss = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    output.backward() # 计算梯度
    ```
-   **`torch.nn.CrossEntropyLoss`**：**交叉熵损失**，常用于多分类问题。它结合了 `LogSoftmax` 和 `NLLLoss`，直接作用于模型的原始输出（logits）。
-   **`torch.nn.BCELoss`**：**二元交叉熵损失 (Binary Cross Entropy Loss)**，常用于二分类问题。需要模型的输出经过 Sigmoid 激活。
-   **`torch.nn.BCEWithLogitsLoss`**：`BCELoss` 的稳定版本，它将 Sigmoid 激活和 `BCELoss` 合并为一个操作，数值更稳定，推荐用于二分类。

### 2.4 优化器 (`torch.optim`)

-   **`torch.optim.SGD`**：**随机梯度下降 (Stochastic Gradient Descent)** 优化器，支持动量（momentum）、Nesterov 动量等。
-   **`torch.optim.Adam`**：**Adam 优化器**，一种自适应学习率优化算法，通常是深度学习任务的默认首选。
-   **`torch.optim.AdamW`**：**AdamW 优化器**，是 Adam 的一个变体，对权重衰减（Weight Decay）的处理更有效，在 Transformer 模型中常用。

### 2.5 模型保存与加载

-   **`torch.save(model.state_dict(), PATH)`**：保存模型参数（权重和偏置）。
-   **`model.load_state_dict(torch.load(PATH))`**：加载模型参数。
-   **`torch.save(model, PATH)`**：保存整个模型（包括结构和参数），但不推荐，因为加载时需要原始的模型类定义。

## 3. GPU 加速

-   **`torch.cuda.is_available()`**：检查是否有可用的 GPU。
-   **`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`**：定义设备。
-   **`.to(device)`**：将 Tensor 或 `nn.Module` 移动到指定设备（CPU 或 GPU）上进行计算。

PyTorch 的设计哲学是“易用性”和“灵活性”，这使得它成为研究人员快速迭代和实验的强大工具。
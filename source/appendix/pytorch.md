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

### 2.3 构建模型 (`torch.nn`)

-   **`nn.Parameter`**：这是一个特殊的张量（Tensor），当它被赋值为 `nn.Module` 的属性时，它会自动被注册为模型的一个参数。这意味着在训练过程中，PyTorch 的自动求导机制会追踪它的梯度，并且在优化器更新时，它的值会被更新。与普通张量的主要区别是，`nn.Parameter` 默认 `requires_grad=True`。

-   **`nn.Sequential`**：一个有序的模块容器。它按照模块在构造函数中传入的顺序，依次将输入数据传递给每个模块。这是一种快速构建简单线性堆叠网络（如 MLP）的便捷方式。
    ```python
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    ```

### 2.4 访问与管理模型参数

-   **`state_dict`**：一个 Python 字典，它将模型的每一个层映射到其对应的参数张量（权重和偏置）。`state_dict` 是保存和加载模型状态的核心，因为它只包含可训练的参数，不包含模型结构。

-   **`named_parameters()`**：一个返回模型所有参数的迭代器，每次迭代产生一个 `(参数名称, 参数本身)` 的元组。这对于需要按名称检查、修改或选择性地对某些参数进行操作（如只对某些层进行微调）的场景非常有用。

-   **`bias`, `bias.data`, `weight.grad`**：这些是访问层中具体参数和其属性的方式。
    -   `layer.weight` / `layer.bias`：访问一个层的权重或偏置，它们是 `nn.Parameter` 对象。
    -   `layer.bias.data`：直接访问参数底层的张量数据，对其的修改不会被 autograd 追踪。常用于权重初始化等场景。
    -   `layer.weight.grad`：在调用 `loss.backward()` 之后，这个属性会存储计算出的关于该权重的梯度值。

### 2.5 初始化与操作模型权重

-   **`net.apply(fn)`**：这是一个对 `nn.Module` 进行递归操作的方法。它会将函数 `fn` 应用到网络自身的每一个子模块上。最常见的用途是实现自定义的权重初始化策略。
    ```python
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)
    ```

-   **`torch.nn.init`**：这个模块包含了一系列常用的权重初始化函数。函数名以 `_` 结尾的通常表示这是一个**原地（in-place）**操作。
    -   `nn.init.normal_`：使用正态分布的值填充张量。
    -   `nn.init.zeros_`：使用 0 填充张量。
    -   `nn.init.constant_`：使用一个常数值填充张量。
    -   `nn.init.xavier_uniform_`：使用 Xavier（或称 Glorot）均匀分布初始化张量，适用于 Tanh/Sigmoid 激活函数。
    -   `nn.init.uniform_`：使用均匀分布的值填充张量。

### 2.6 损失函数 (`torch.nn`)

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


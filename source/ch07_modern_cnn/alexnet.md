# AlexNet

AlexNet 由 Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 在 2012 年提出，并以绝对优势赢得了当年的 ImageNet 图像识别挑战赛（ILSVRC）。它的成功被认为是引爆深度学习浪潮的关键事件，证明了深度卷积神经网络在复杂图像识别任务上的巨大潜力。

本质上，AlexNet 可以看作是一个更大、更深的 LeNet。它不仅增加了网络深度，还引入了一系列在当时非常创新的技术，这些技术后来成为了深度学习领域的标准实践。

## AlexNet 的主要贡献与改进

- **更深的网络结构**：AlexNet 包含 5 个卷积层和 3 个全连接层，比 LeNet-5 深得多，使其能够学习更复杂的特征。

- **ReLU 激活函数**：AlexNet 是首批成功将 ReLU（Rectified Linear Unit）激活函数应用于大型 CNN 的模型之一。ReLU 相比于传统的 Sigmoid 或 Tanh 函数，能够有效缓解梯度消失问题，从而加速了模型的训练过程。

- **暂退法 (Dropout)**：为了应对模型巨大的参数量（尤其是在全连接层）带来的过拟合风险，AlexNet 在全连接层中使用了 Dropout 技术。这极大地增强了模型的泛化能力。

- **数据增强 (Data Augmentation)**：AlexNet 在训练中广泛使用了数据增强技术，例如对图像进行随机裁剪、翻转和颜色变换。这相当于扩充了训练数据集，让模型学习到对位置、方向和光照等变化更具鲁棒性的特征。

- **局部响应归一化 (Local Response Normalization, LRN)**：虽然 LRN 后来被证明效果不如批量归一化（Batch Normalization）且已不再常用，但它在当时是 AlexNet 用于增强模型泛化能力的一种尝试。

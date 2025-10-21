# GoogLeNet (Inception v1)

GoogLeNet 是 2014 年 ImageNet 挑战赛的冠军。与 VGGNet 追求纯粹的深度不同，GoogLeNet 在加深网络的同时，也极其注重计算效率，其主要优点是**模型参数量小、计算复杂度低**。它一共使用了 9 个 Inception 块，是早期达到上百层（这里的“层”也计入了模块内的并行分支）的复杂网络。

## 核心创新：Inception 模块

Inception 模块是 GoogLeNet 的灵魂。它没有选择单一尺寸的卷积核，而是将不同尺度的处理路径并行化，其核心思想是：
1.  **并行多尺度处理**：在一个模块内，通过 4 个并行的路径（使用 1x1、3x3、5x5 的卷积核以及池化操作）从不同层面抽取信息。
2.  **1x1 卷积降维**：为了避免计算量爆炸，模块在 3x3 和 5x5 卷积之前，以及在池化之后，都巧妙地使用了一个 1x1 的卷积层来降低特征图的深度（通道数），这被称为“瓶颈层 (Bottleneck Layer)”。
3.  **结果拼接**：最后将 4 个路径的输出特征图在深度（通道）维度上拼接（concatenate）起来，形成一个信息更丰富的组合特征。

![Inception 模块](https://zh-v2.d2l.ai/_images/inception-full.svg)

## 意义与后续演进

GoogLeNet 证明了，通过精心设计的、非线性的网络拓扑结构（“网络中的网络”），可以在提升性能的同时，比 VGGNet 等模型更节省计算资源。Inception 架构也衍生出了一系列后续变种，例如：

-   **Inception-v2/v3**：引入了批量归一化（Batch Normalization）并进一步优化了模块结构。
-   **Inception-v4**：将 Inception 模块与 ResNet 的残差连接思想相结合。

# AI相关软件框架

## Hugging Face Transformers

它在 PyTorch、TensorFlow 等底层框架之上，提供了对上千种 Transformer 模型（如 BERT, GPT）的标准实现。开发者无需从零开始构建模型，可以直接使用 `transformers` 库加载 L2 层的预训练模型，并利用其 `Trainer` API 在 L1 框架上进行便捷的微调。LangChain 等应用框架通常会深度集成 `transformers` 库，通过它来调用和操作模型，以执行应用逻辑中的具体步骤。

可以将其理解为一个标准化的“模型引擎套件”，极大地降低了使用和训练 SOTA 模型的门槛，是事实上的模型生态标准。

## OpenCV

历史悠久且功能强大的**经典计算机视觉算法库**。在现代 AI 工作流中主要用于：
-   **数据预处理**：在模型训练前，使用 OpenCV 对图像进行读取、缩放、裁剪、颜色转换和数据增强。
-   **结果后处理**：在模型推理后，使用 OpenCV 对结果进行可视化，例如在图像上绘制检测框或分割掩码。

## torchvision

PyTorch 官方的计算机视觉库，是 PyTorch 生态的“标配”。它提供了一套基础、全面的工具，包括：
-   **`models`**：提供 ResNet, ViT 等经典、稳定的模型实现。
-   **`datasets`**：方便地加载 ImageNet, CIFAR-10 等标准数据集。
-   **`transforms`**：提供标准的图像变换和数据增强功能。

## timm (PyTorch Image Models)

如果说 `torchvision` 是官方标配，`timm` 就是 CV 领域的“专业级改装市场”和“模型军火库”。它的核心优势是提供了海量的、紧跟研究前沿的 SOTA 图像模型，是追求极致性能的研究者和开发者的首选。

## diffusers

由 Hugging Face 推出的，专注于**AI 生成（AIGC）领域扩散模型**的工具箱。它将复杂的扩散过程（如 Stable Diffusion）拆解为模型、调度器等模块化组件，让开发者可以灵活地构建和定制自己的文生图、图生图等生成流水线。

## PEFT (Parameter-Efficient Fine-Tuning)

由 Hugging Face 开发，专门用于**高效微调**大模型的“改装件”库。它将 LoRA, QLoRA 等参数高效微调技术封装成简单接口，使得在消费级硬件上微调巨型模型成为可能。它通常与 `transformers` 库配合使用。

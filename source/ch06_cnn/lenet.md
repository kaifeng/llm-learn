# LeNet

LeNet-5 由 Yann LeCun 在 20 世纪 80 年代末期提出，是现代卷积神经网络的“鼻祖”，被成功应用于银行支票的手写数字识别系统中，是 CNN 早期商业化应用的典范。

它的核心思想是**先用卷积层和池化层来学习图片的空间信息，然后使用全连接层来将提取到的特征转换到类别空间进行分类**。

![LeNet-5 架构](https://zh-v2.d2l.ai/_images/lenet.svg)

LeNet-5 的经典架构通常由 7 层构成（不含输入），其顺序为：

    输入 -> 卷积层 -> 池化层 -> 卷积层 -> 池化层 -> 全连接层 -> 全连接层 -> 输出层

这个“卷积/池化堆叠 + 全连接分类”的模式成为了后来几十年 CNN 设计的蓝图，并验证了通过参数共享的卷积核来提取全图特征的有效性。

下面的 PyTorch 代码片段展示了 LeNet 架构如何逐步改变数据形状：
```python
# 假设 net 是一个 LeNet-5 模型实例
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__:<15} output shape: 	{X.shape}')

# 可能的输出:
# Conv2d          output shape:   torch.Size([1, 6, 28, 28])
# Sigmoid         output shape:   torch.Size([1, 6, 28, 28])
# AvgPool2d       output shape:   torch.Size([1, 6, 14, 14])
# Conv2d          output shape:   torch.Size([1, 16, 10, 10])
# Sigmoid         output shape:   torch.Size([1, 16, 10, 10])
# AvgPool2d       output shape:   torch.Size([1, 16, 5, 5])
# Flatten         output shape:   torch.Size([1, 400])
# Linear          output shape:   torch.Size([1, 120])
# Sigmoid         output shape:   torch.Size([1, 120])
# Linear          output shape:   torch.Size([1, 84])
# Sigmoid         output shape:   torch.Size([1, 84])
# Linear          output shape:   torch.Size([1, 10])
```
> **可视化推荐**：想要直观地理解 CNN 每一层具体在做什么，可以访问这个交互式网站：[CNN Explainer](https://poloclub.github.io/cnn-explainer/)

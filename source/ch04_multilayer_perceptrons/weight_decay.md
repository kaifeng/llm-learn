# 权重衰退 (Weight Decay)

权重衰退是正则化技术的一种，旨在通过限制模型参数（特别是权重）的大小来防止过拟合。它的核心思想是在优化模型的损失函数时，增加一个对模型权重进行惩罚的惩罚项。

L2 正则化是最常见的权重衰退形式。它在原始损失函数 $l(w,b)$ 的基础上，增加所有权重平方和的 $\frac{\lambda}{2}$ 倍作为惩罚项。

$$\text{Loss} = l(w,b)+\frac{\lambda}{2}\|w\|^2$$

其中 $\lambda$ 是一个超参数，控制正则化的强度。

当使用梯度下降更新参数时，这个惩罚项对权重的梯度贡献为 $\lambda \mathbf{w}$。因此，权重的更新法则变为：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left( \frac{\partial \ell(\mathbf{w}_t, b_t)}{\partial \mathbf{w}_t} + \lambda \mathbf{w}_t \right) = (1 - \eta \lambda) \mathbf{w}_t - \eta \frac{\partial \ell(\mathbf{w}_t, b_t)}{\partial \mathbf{w}_t}$$

通常学习率 $\eta$ 和正则化强度 $\lambda$ 的乘积 $\eta\lambda$ 是一个小于 1 的正数。从上式可以看出，每次更新时，权重 $\mathbf{w}_t$ 都会先乘以一个小于 1 的系数 $(1 - \eta \lambda)$，这相当于对权重进行了一次“衰减”，然后再减去原始的梯度。这就是“权重衰退”名称的由来。

通过惩罚较大的权重值，权重衰退迫使模型学习到更小、更分散的权重，从而构建一个更简单、更不容易过拟合的模型。

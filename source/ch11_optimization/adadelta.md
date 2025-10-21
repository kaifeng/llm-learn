# Adadelta

Adadelta 是一种自适应学习率优化算法，它在 RMSProp 的基础上进一步改进，旨在完全消除学习率这个超参数。

## 核心思想

Adadelta 不仅使用梯度平方的指数加权移动平均来调整学习率，还使用参数更新量平方的指数加权移动平均。它通过维护两个状态变量来实现：

1.  **梯度平方的指数加权移动平均**：$E[g^2]_t$
2.  **参数更新量平方的指数加权移动平均**：$E[\Delta x^2]_t$

## 更新规则

Adadelta 的更新规则如下：

1.  **计算梯度平方的指数加权移动平均**：
    $$ E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) (\nabla L(\mathbf{w}_t))^2 $$
    其中 $\rho$ 是衰减率。

2.  **计算参数更新量**：
    $$ \Delta \mathbf{w}_t = - \frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \nabla L(\mathbf{w}_t) $$

3.  **累积参数更新量平方**：
    $$ E[\Delta x^2]_t = \rho E[\Delta x^2]_{t-1} + (1 - \rho) (\Delta \mathbf{w}_t)^2 $$

4.  **更新参数**：
    $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \Delta \mathbf{w}_t $$

## 优势

- **无需手动设置学习率**：Adadelta 的最大优势在于它不需要手动设置全局学习率，这使得它在实践中更易于使用。
- **适用于稀疏数据和非平稳目标**：与 Adagrad 和 RMSProp 类似，Adadelta 在处理稀疏数据和非平稳目标时表现良好。

## 劣势

- **计算复杂**：相比其他优化器，Adadelta 的计算过程相对复杂。

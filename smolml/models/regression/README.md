# SmolML - 回归：预测连续值

基于SmolML核心中解释的自动微分（Value）和N维数组（MLArray）的核心概念，我们现在可以实现各种机器学习模型。本节专注于**回归模型**，用于预测连续数值输出。例如根据输入特征预测房价、股票价值或温度。

尽管深度神经网络提供了巨大的能力，但像线性回归或其扩展多项式回归这样的简单模型往往是极好的起点，它们计算效率高且高度可解释。它们与复杂网络共享相同的基本学习原则：通过使用梯度下降调整参数来最小化损失函数，这一切都是由我们的自动微分引擎使用Value类完成的。
## 回归基础：从数据中学习

在回归中的目标是找到一个数学函数，该函数将输入特征（如房屋的平方英尺）映射到连续输出（如价格）。此函数具有内部参数（通常称为**权重**或系数，以及**偏置**或截距），这些参数决定了其确切形状。
<div align="center">
  <img src="https://github.com/user-attachments/assets/79874cec-8650-4628-af1f-ca6fdc4debe5" width="600">
</div>

那么，如何找到最佳参数呢？
- **预测**：我们以初始（通常是随机的）参数开始，并使用模型对训练数据进行预测。
- **损失计算**：我们使用**损失函数**（如均方误差-MSE）将这些预测与实际已知值进行比较。这个函数量化了模型当前的错误程度。较低的损失更好。
- **梯度计算**：就像在核心解释中一样，我们需要知道如何调整每个参数以减少损失。我们的Value对象和反向传播的概念自动计算损失相对于每个参数（权重和偏置）的**梯度**。记住，梯度指向损失最陡增加的方向。
- **参数更新**：我们使用**优化器**（如随机梯度下降-SGD）将参数沿着与其梯度相反的方向移动一小步，朝着降低损失的方向前进。
- **迭代**：我们重复步骤1-4多次（迭代或时期），逐渐改进模型的参数直到损失最小化或停止显著下降。


这种迭代过程使回归模型能够“学习”数据中输入和输出之间的基本关系。



## Regression基类：通用框架
为了简化不同回归算法的实现，在SmolML中我们创建了一个Regression基类（位于regression.py中）。这个类处理公共结构和训练循环逻辑。特定模型如LinearRegression继承自它。

以下是它的运作方式：
- 初始化(__init__)：
  - 接受input_size（预期输入特征的数量）、一个loss_function、一个optimizer实例和一个权重initializer。
  - 至关重要的是，它初始化了模型的可训练参数：
    - self.weights: 一个持有每个输入特征系数的MLArray。它的形状由input_size决定，值由initializer设置。
    - self.bias: 一个标量MLArray（初始化为1）表示截距项。
  - 由于weights和bias是MLArrays，它们本质上包含Value对象。这确保它们是计算图的一部分，可以在训练过程中自动计算它们的梯度。

- 训练(fit) 这个方法组织了前面描述的梯度下降循环。对于指定数量的iterations：
  - **前向传播**：调用self.predict(X)（必须由子类实现）获取预测y_pred。这为预测步骤构建了计算图。
  - **损失计算**：计算loss = self.loss_function(y, y_pred)。这个loss通常是包含单个Value的最终MLArray，代表这一迭代的整体错误。
  - **后向传播**：调用loss.backward()。这触发自动微分过程，计算涉及的所有Value对象的损失梯度，包括那些在self.weights和self.bias中的。
  - **参数更新**：使用self.optimizer.update(...)根据计算出的梯度(weights.grad()和bias.grad())和优化器的逻辑（例如学习率）调整self.weights和self.bias。
  - **梯度重置**：调用self.restart(X, y)将所有参数和数据的.grad属性重置为零，准备下一次迭代。

- **预测(predict)**： 在基类中定义但引发NotImplementedError。为什么？因为做出预测的具体逻辑因回归类型而异（例如线性vs多项式）。每个子类必须提供自己的predict方法，定义其具体的数学公式使用MLArray操作。
- **梯度重置(restart)**： 一个辅助函数，简单地在weights、bias、输入X和目标y MLArrays上调用.restart()方法。这有效地将所有底层Value对象的.grad属性重置为零。
- **表示(__repr__)**： 提供配置模型的格式良好的字符串摘要，包括其类型、参数形状、优化器、损失函数和估计内存使用情况。

## 具体实现的模型

<div align="center">
  <img src="https://github.com/user-attachments/assets/8b282ca1-7c17-460d-a64c-61b0624627f9" width="600">
</div>

### `LinearRegression`

这是最基本的回归模型。它假设输入特征X和输出y之间存在直接的线性关系。目标是找到最佳权重w和偏置b使得 $y \approx Xw + b$.

**实现(regression.py)**：
- 直接继承自Regression。
- 它的主要贡献是覆盖了predict方法。
**预测(predict)**：
- 实现了线性方程：return X @ self.weights + self.bias。
- 它接受输入X (MLArray)，执行矩阵乘法(@)与self.weights (MLArray)，并加上self.bias (MLArray)。由于X、weights和bias都是包含Value对象的MLArray，这条代码自动构建必要的计算图以便反向传播。
**训练**：
- 使用直接从Regression基类继承的fit方法无需修改。基类使用LinearRegression提供的predict逻辑处理整个训练循环。


### `PolynomialRegression`

如果关系不是一条直线怎么办？多项式回归通过拟合多项式曲线 (例如添加 $y \approx w_2 x^2 + w_1 x + b$) 扩展了线性回归以适应数据。

- **实现(regression.py)**： 同样继承自Regression。
- **核心思想**：不是直接将X拟合到y，而是首先将输入特征X转换为多项式特征 (例如添加 $X^2$, $X^3$, 等) 然后对这些新的，转换后的特征应用标准线性回归模型。
- **初始化(__init__)**：
    - 额外接受一个degree参数，指定特征转换中要包含的最高次幂（例如degree=2意味着包含额外接受一个degree参数，指定特征转换中要包含的最高次幂（例如degree=2意味着包含 (例如 `degree=2` 意味着包含 $X$ and $X^2$).
    - 调用基类__init__，但是传递给基类的input_size实际上是多项式特征的数量，而不是原始特征的数量。权重将对应于这些转换后的特征。
- **特征转换(transform_features)**：这个关键方法接收原始输入X并生成新的多项式特征。对于输入X和degree=d，它使用MLArray操作（如逐元素乘法*）计算 $X, X^2, \dots, X^d$ 并将它们连接成一个新的MLArray。这确保转换也是潜在计算图的一部分（尽管通常它是预先计算的）.
- **预测(predict)**：
  - 它首先调用X_poly = self.transform_features(X)获取多项式特征。
  - 然后，它使用这些转换后的特征执行标准线性预测：return X_poly @ self.weights + self.bias。这里的self.weights对应于多项式项的系数。
- 训练(fit)：
  - 它稍微覆盖了基类的fit方法。
  - 在主循环之前，它将整个训练输入X转换为X_poly = self.transform_features(X)。
  - 然后，它调用基类的fit方法（super().fit(...)），但传递X_poly（而不是X）作为输入数据。
  - 继承的fit方法然后照常进行，根据来自X_poly的预测计算损失，通过线性预测部分和特征转换步骤回传梯度，并更新与多项式项相关的权重。
  
## 示例用法

以下是如何使用LinearRegression的概念性示例：

```python
from smolml.models.regression import LinearRegression
from smolml.core.ml_array import MLArray
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses

# 样本数据（例如，2个特征，3个样本）
X_data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
# 目标值（连续）
y_data = [[3.5], [5.5], [7.5]]

# 转换为 MLArray
X = MLArray(X_data)
y = MLArray(y_data)

# 初始化模型
# 预期有2个输入特征
model = LinearRegression(input_size=2,
                         optimizer=optimizers.SGD(learning_rate=0.01),
                         loss_function=losses.mse_loss)

# 打印初始模型摘要
print(model)

# 训练模型
print("\n开始训练...")
losses_history = model.fit(X, y, iterations=100, verbose=True, print_every=10)
print("训练完成。")

# 打印最终模型摘要（权重和偏置会发生变化）
print(model)

# 对新数据进行预测
X_new = MLArray([[4.0, 5.0]])
prediction = model.predict(X_new)
print(f"\n对 {X_new.to_list()} 的预测: {prediction.to_list()}")
```

## Regression wrap-up

这些回归类展示了我们实现的基础Value和MLArray如何用于设计和训练经典的机器学习模型！只需几行代码！是不是很酷？
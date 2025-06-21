from smolml import MLArray, Value

def example():
    #  2个样本，每个样本2个特征
    input: MLArray = MLArray([[1.0, 2.0], [3.0, 4.0]])
    # 线性层的权重矩阵，将 2个特征 映射到 1个输出
    weight: MLArray = MLArray([[0.5], [-0.5]])

    # --- 构建计算图，实现向前传播 ---
    # 矩阵乘法：将输入 input 与权重 weight 相乘。得到线性输出
    result: MLArray = input @ weight
    # 对结果 result 应用激活函数
    result_active: MLArray = result.tanh()
    # 计算损失 loss_value，即所有激活值的总和
    loss_value: MLArray = result_active.sum()

    # 反向传播计算
    # 触发从损失值 loss_value 开始的梯度反向传播；
    # 自动计算图中所有参与前向传播的节点（如 input 和 weight）对 loss_value 的梯度；
    # 结果保存在各变量的 .grad() 属性中，例如 input.grad() 和 weight.grad()
    loss_value.backward()

    print("\n打印 输入 input 对损失 loss_value 的梯度值")
    print(input.grad())

    print("\n打印 权重 weight 对损失 loss_value 的梯度值")
    print(weight.grad())


if __name__ == "__main__":
    example()

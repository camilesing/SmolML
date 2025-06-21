from smolml.models.regression import LinearRegression
from smolml.core.ml_array import MLArray
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses


def example():
    # 样本数据（例如，2个特征，3个样本）
    example_data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    # 目标值（连续）
    target_data = [[3.5], [5.5], [7.5]]

    # 转换为 MLArray
    example = MLArray(example_data)
    target = MLArray(target_data)

    # 初始化模型
    # 预期有2个输入特征
    model = LinearRegression(input_size=2,
                             optimizer=optimizers.SGD(learning_rate=0.01),
                             loss_function=losses.mse_loss)

    # 打印初始模型摘要
    print(model)
    print(model)

    # 训练模型
    print("\n开始训练...")
    losses_history = model.fit(example, target, iterations=100, verbose=True, print_every=10)
    print("训练完成。")

    # 打印最终模型摘要（权重和偏置会发生变化）
    print(model)

    # 对新数据进行预测
    X_new = MLArray([[4.0, 5.0]])
    prediction = model.predict(X_new)
    print(f"\n对 {X_new.to_list()} 的预测: {prediction.to_list()}")


if __name__ == "__main__":
    example()


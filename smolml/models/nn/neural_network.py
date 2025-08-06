from smolml.core.ml_array import MLArray
from smolml.models.nn import DenseLayer
import smolml.utils.memory as memory
import smolml.utils.losses as losses
import smolml.utils.activation as activation
import smolml.utils.optimizers as optimizers

"""
//////////////////////
/// NEURAL NETWORK ///
//////////////////////
"""


class NeuralNetwork:
    """
    Implementation of a feedforward neural network with customizable layers and loss function.
    Supports training through backpropagation and gradient descent.
    """

    def __init__(self, layers: list, loss_function: callable,
                 optimizer: optimizers.Optimizer = optimizers.SGD()) -> None:
        """
        你可以通过传入一个包含多个层对象的列表（例如一系列 DenseLayer 实例）、
            一个损失函数（用于衡量网络预测结果的“错误程度”）以及一个优化器（用于定义如何更新各层参数），
            来创建一个 NeuralNetwork 实例。
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer if optimizer is not None else optimizers.SGD()

    def forward(self, input_data):
        """
        网络的前向传播过程非常直接：它接收输入数据，
            并将其按顺序依次传递给列表中的每一个层。第 i 层的输出会成为第 i+1 层的输入。
        """
        if not isinstance(input_data, MLArray):
            raise TypeError(f"Input data must be MLArray, not {type(input_data)}")
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def train(self, X, y, epochs, verbose=True, print_every=1):
        """
         用于训练神经网络模型，使用梯度下降优化算法进行参数更新
        """
        # 将输入数据 X 和标签 y 转换为统一格式
        X, y = MLArray.ensure_array(X, y)
        losses = []
        # 对指定的 epochs 次数进行循环训练
        for epoch in range(epochs):
            # 通过网络预测输出 y_pre
            y_pred = self.forward(X)

            # 使用损失函数计算当前预测误差
            loss = self.loss_function(y_pred, y)
            losses.append(loss.data.data)

            #
            loss.backward()

            for idx, layer in enumerate(self.layers):
                layer.update(self.optimizer, idx)

            # 防止梯度累积，为下一轮迭代做准备
            X.restart()
            y.restart()
            for layer in self.layers:
                layer.weights.restart()
                layer.biases.restart()

            if verbose:
                #  按设定频率打印训练进度（如损失值
                if (epoch + 1) % print_every == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")

        return losses

    def __repr__(self):
        """
        Returns a string representation of the neural network architecture.
        Displays layer information, loss function, optimizer details, and detailed memory usage.
        """
        # Get terminal width for formatting
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Create header
        header = "Neural Network Architecture"
        separator = "=" * terminal_width

        # Get size information
        size_info = memory.calculate_neural_network_size(self)

        # Format layers information
        layers_info = []
        for i, (layer, layer_size) in enumerate(zip(self.layers, size_info['layers'])):
            if isinstance(layer, DenseLayer):
                input_size = layer.weights.shape[0]
                output_size = layer.weights.shape[1]
                activation_name = layer.activation_function.__name__
                layer_info = [
                    f"Layer {i + 1}: Dense("
                    f"in={input_size}, "
                    f"out={output_size}, "
                    f"activation={activation_name})"
                ]

                # Parameters info
                params = input_size * output_size + output_size  # weights + biases
                layer_info.append(
                    f"    Parameters: {params:,} "
                    f"({input_size}×{output_size} weights + {output_size} biases)"
                )

                # Memory info
                layer_info.append(
                    f"    Memory: {memory.format_size(layer_size['total'])} "
                    f"(weights: {memory.format_size(layer_size['weights_size'])}, "
                    f"biases: {memory.format_size(layer_size['biases_size'])})"
                )

                layers_info.append("\n".join(layer_info))

        # Calculate total parameters
        total_params = sum(
            layer.weights.size() + layer.biases.size()
            for layer in self.layers
        )

        # Format optimizer information
        optimizer_info = [
            f"Optimizer: {self.optimizer.__class__.__name__}("
            f"learning_rate={self.optimizer.learning_rate})"
        ]

        # Add optimizer state information if it exists
        if size_info['optimizer']['state']:
            state_sizes = [
                f"    {key}: {memory.format_size(value)}"
                for key, value in size_info['optimizer']['state'].items()
            ]
            optimizer_info.extend(state_sizes)

        # Format loss function information
        loss_info = f"Loss Function: {self.loss_function.__name__}"

        # Detailed memory breakdown
        memory_info = ["Memory Usage:"]

        # Layer memory
        for i, layer_size in enumerate(size_info['layers']):
            memory_info.append(
                f"  Layer {i + 1}: {memory.format_size(layer_size['total'])} "
                f"(weights: {memory.format_size(layer_size['weights_size'])}, "
                f"biases: {memory.format_size(layer_size['biases_size'])})"
            )

        # Optimizer memory
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Optimizer State: {memory.format_size(opt_size)}")

        memory_info.append(f"  Base Objects: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")

        # Combine all parts
        return (
                f"\n{header}\n{separator}\n\n"
                f"Architecture:\n"
                + "\n".join(f"  {layer}" for layer in layers_info)
                + f"\n\n"
                + "\n".join(optimizer_info)
                + f"\n{loss_info}\n\n"
                  f"Total Parameters: {total_params:,}\n\n"
                + "\n".join(memory_info)
                + f"\n{separator}\n"
        )


def example_neural_network():
    # 输入层大小为 2（每个样本有两个特征）
    input_size = 2
    # 隐藏层大小为 32。就是中间的层数
    hidden_size = 32
    # 输出层大小为 1（二分类输出）
    output_size = 1

    # Create the neural network
    #   使用两个全连接层（DenseLayer）；
    #   第一层使用 ReLU 激活函数；
    #   第二层使用 tanh 激活函数；
    #   损失函数为均方误差（mse_loss）；
    #   优化器为 AdaGrad，学习率为 0.1；
    nn = NeuralNetwork([
        DenseLayer(input_size, hidden_size, activation.relu),
        DenseLayer(hidden_size, output_size, activation.tanh)
    ], losses.mse_loss, optimizers.AdaGrad(learning_rate=0.1))

    # 构造了一个经典的 XOR 逻辑门数据集，包含 4 个样本和对应标签；
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    # 触发 repr 方法输出详细模型摘要
    print("摘要输出")
    print(nn)
    # 使用 train 方法训练模型共 100 轮
    nn.train(X, y, epochs=100)
    # 使用训练好的模型对输入数据进行前向传播
    y_pred = nn.forward(MLArray(X))
    print("预测结果")
    print(y_pred)


if __name__ == '__main__':
    example_neural_network()

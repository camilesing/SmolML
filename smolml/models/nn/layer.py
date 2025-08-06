from smolml.core.ml_array import zeros
import smolml.utils.activation as activation
import smolml.utils.initializers as initializers


class DenseLayer:
    """
    Creates a standard Dense (linear) layer for a Neural Network.
    This layer performs the operation: output = activation(input @ weights + biases)
    """
    def __init__(self, input_size: int, output_size: int, activation_function: callable = activation.linear,
                 weight_initializer: initializers.WeightInitializer = initializers.XavierUniform) -> None:
        """
        根据输入输出维度和初始化方法设置权重和偏置
        """
        self.weights = weight_initializer.initialize(input_size, output_size)
        self.biases = zeros(1, output_size)  # Initialize biases with zeros
        self.activation_function = activation_function

    def forward(self, input_data):
        """
       执行 output = activation(input @ weights + biases
        """
        z = input_data @ self.weights + self.biases  # Linear transformation
        return self.activation_function(z)  # Apply activation function

    def update(self, optimizer, layer_idx):
        """通过优化器更新权重和偏置"""
        self.weights, self.biases = optimizer.update(self, layer_idx, param_names=("weights", "biases"))
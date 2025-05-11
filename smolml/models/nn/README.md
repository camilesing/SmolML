# SmolML - Neural Networks: Backpropagation to the limit

Welcome to the neural network section of SmolML! Having established our Value objects for automatic differentiation and MLArray for handling data (see 'core' section), we can now build models that learn. This guide will walk you through the fundamental concepts, from a single neuron to a fully trainable neural network, and how they're represented in SmolML.

> **IMPORTANT**: As our implementation is made fully in Python, handling the automatic differentiation of an entire neural network is very computationally expensive. If you plan on running an example, we recommend starting with a very small network and then escalate. Creating a too big neural network for your computer might make it freeze 🙂 

<div align="center">
  <img src="https://github.com/user-attachments/assets/e5315fca-5dd6-4c9c-9cf3-bf46edfbb40c" width="600">
</div>

## The Neuron: A Tiny Decision Maker

At the heart of a neural network is the neuron (or node). Think of it as a small computational unit that receives several inputs, processes them, and produces a single output.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2f95fdfe-1676-4a0b-9e10-95ecdf9155b6" width="600">
</div>

Here's what a neuron conceptually does:

- **Weighted Sum**: Each input connection to the neuron has an associated weight. The neuron multiplies each input value by its corresponding weight. These weights are crucial – they are what the network learns by adjusting during training, determining the influence of each input.
- **Bias**: The neuron then adds a bias term to this weighted sum. The bias allows the neuron to shift its output up or down, independent of its inputs. This helps the network fit data that doesn't necessarily pass through the origin.
- **Activation Function**: Finally, the result of the weighted sum + bias is passed through an activation function. This function introduces non-linearity, which is vital. Without non-linearity, a stack of multiple layers would behave just like a single layer, limiting the network's ability to learn complex patterns. Common activation functions include ReLU, Tanh, and Sigmoid.

While SmolML doesn't have a standalone Neuron class for this section (as it's often more efficient to work with layers directly), the logic of many such neurons operating in parallel is encapsulated within our DenseLayer. Each output feature of a DenseLayer can be thought of as the output of one such conceptual neuron.

## Layers: Organizing Neurons

A single neuron isn't very powerful on its own. Neural networks organize neurons into layers. The most common type is a Dense Layer (also known as a Fully Connected Layer).

What does a Dense Layer do?

In a dense layer, every neuron in the layer receives input from every neuron in the previous layer (or from the raw input data if it's the first layer).

Conceptually, a dense layer performs two main steps, building on the neuron's logic:

1. **Linear Transformation**: It takes an input vector (or a batch of input vectors) and performs a matrix multiplication with a weight matrix (`W`) and adds a bias vector (`b`).
   - Each row in the input vector connects to each column in the weight matrix. If you have input_size features and want output_size features from this layer (i.e., output_size conceptual neurons), the weight matrix `W` will have a shape of (input_size, output_size). Each element $W_ij$ is the weight connecting the i-th input feature to the j-th neuron in the layer.
   - The bias vector b will have output_size elements, one for each neuron.
   - Mathematically: $z=input×W+b$.
   - In SmolML, when you create a DenseLayer (from `layer.py`), you specify input_size and output_size. The layer then initializes self.weights (our `W`) and self.biases (our `b`) as `MLArray` objects. These are the learnable parameters of the layer.

```python
# From layer.py
class DenseLayer:
    def __init__(self, input_size: int, output_size: int, ...):
        self.weights = weight_initializer.initialize(input_size, output_size) # MLArray
        self.biases = zeros(1, output_size) # MLArray
        ...
```

2. **Activation Function**: The result (`z`) of this linear transformation is then passed element-wise through a chosen non-linear activation function (e.g., ReLU, Tanh).
   - This is applied to the output of each conceptual neuron in the layer.
   - In SmolML, you specify the activation_function when creating a DenseLayer, and it's applied in the forward method:

```python
# From layer.py
class DenseLayer:
    ...
    def forward(self, input_data):
        z = input_data @ self.weights + self.biases # Linear transformation
        return self.activation_function(z)      # Activation
```

The forward method essentially defines how data flows through the layer. Because `input_data`, `self.weights`, and `self.biases` are `MLArray`s (which use `Value` objects internally), all operations automatically build the computational graph needed for backpropagation.

## Neural Networks: Stacking Layers

The true power of neural networks comes from stacking multiple layers. The output of one layer becomes the input to the next. This allows the network to learn hierarchical features – earlier layers might learn simple patterns (like edges in an image), while later layers combine these to learn more complex concepts (like shapes or objects).

<div align="center">
  <img src="https://github.com/user-attachments/assets/3979a284-0b29-4110-b6c5-dfe1a13f50b9" width="600">
</div>

### The NeuralNetwork Class (neural_network.py)

In SmolML, the `NeuralNetwork` class manages this sequence of layers and orchestrates the entire training process.

- **Initialization (__init__)**:
  - You create a NeuralNetwork by providing it with a list of layer objects (e.g., a sequence of DenseLayer instances), a loss_function (to measure how "wrong" the network's predictions are), and an optimizer (which defines how to update the layer parameters).

```python
# From neural_network.py
class NeuralNetwork:
    def __init__(self, layers: list, loss_function: callable, optimizer: optimizers.Optimizer = optimizers.SGD()):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
```

- **Forward Pass (forward)**:
  - The network's forward pass is straightforward: it takes the input data and passes it sequentially through each layer in its list. The output of layer i becomes the input to layer i+1.

```python
# From neural_network.py
class NeuralNetwork:
    ...
    def forward(self, input_data):
        for layer in self.layers: # Pass data through each layer
            input_data = layer.forward(input_data)
        return input_data # Final output of the network
```

This chained forward pass, because each layer's forward method uses MLArray operations, builds one large computational graph from the initial input all the way to the network's final prediction.

## Teaching the Network: The Training Loop

"Learning" in a neural network means adjusting the weights and biases in all its layers to make better predictions. This is achieved through a process called training, which typically involves the following steps repeated over many epochs (passes through the entire dataset):

1. **Forward Pass**:
   - Feed the input data (`X`) through the network using `network.forward(X)` to get predictions (`y_pred`). As we've seen, this also builds the computational graph.

2. **Compute Loss**:
   - Compare the network's predictions (`y_pred`) with the actual target values (`y`) using the specified loss_function (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
   - The loss is a single Value (often wrapped in an `MLArray`) that quantifies how badly the network performed on this batch of data. This loss `Value` is the final node of our current computational graph.

3. **Backward Pass (Backpropagation)**:
   - This is where the magic of our `Value` objects (from the 'core' section) shines! We call `loss.backward()`.
   - This one command triggers the automatic differentiation process. It walks backwards through the entire computational graph (from the loss all the way back to every weight and bias in every `DenseLayer`, and even the input `X`) and calculates the gradient of the loss with respect to each of these `Value` objects. The `.grad` attribute of each `Value` (and thus each element in our `MLArray` parameters) is populated.
   - This tells us how much a small change in each `weight` or `bias` would affect the overall loss.

4. **Update Parameters**:
   - Now that we know the "direction of steepest ascent" for the loss (the gradients), the optimizer steps in. It uses these gradients (and its own internal logic, like a learning rate) to adjust the weights and biases in each layer. The goal is to nudge them in the opposite direction of their gradients to reduce the loss.
   - In SmolML, the `NeuralNetwork.train` method iterates through its layers and calls `layer.update(self.optimizer, ...)` for each. This method, in turn, uses the optimizer to modify layer.weights and layer.biases.

5. **Reset Gradients**:
   - The gradients calculated by `loss.backward()` are accumulated (added) to the `.grad` attribute of each `Value`. Before the next training iteration (the next forward/backward pass), it's absolutely crucial to reset these gradients back to zero.
   - This is done using the `.restart()` method on the relevant `MLArray`s (all weights and biases in every layer, and sometimes X and y if they are part of persistent graphs). If we didn't do this, gradients from previous iterations would incorrectly influence the updates in the current iteration.
   - You'll see this in `NeuralNetwork.train()`:

```python
# Inside NeuralNetwork.train() after parameter updates
X.restart()
y.restart()
for layer in self.layers:
    layer.weights.restart()
    layer.biases.restart()
```

By repeatedly cycling through these steps, the NeuralNetwork gradually tunes its DenseLayer parameters, leveraging the automatic differentiation power of Value and MLArray to minimize the loss and "learn" from the data.

# SmolML - Neural Networks and Layers

After defining our core components – `Value` for automatic differentiation and `MLArray` (see the 'core' section) for handling multi-dimensional data – we can start building actual models! This section covers how layers are defined and assembled into a trainable Neural Network.

## Building Blocks - The `DenseLayer`

Neural networks are often visualized as interconnected nodes arranged in layers. Data flows from one layer to the next, undergoing transformations at each step. The most fundamental type of layer is often called a "Dense" or "Fully Connected" layer.

**What does a Dense Layer do?**

Conceptually, a dense layer takes an input vector (or a batch of input vectors) and performs two main steps:
1.  **Linear Transformation:** It multiplies the input by a weight matrix (`W`) and adds a bias vector (`b`). Mathematically: $z = \text{input} \times W + b$. The weights and biases are the layer's **learnable parameters**. They start with initial values and are adjusted during training.
2.  **Activation Function:** It applies a non-linear function (like ReLU, Tanh, Sigmoid, or just a simple linear "pass-through") to the result ($z$) of the linear transformation. This non-linearity is crucial; without it, stacking multiple layers would be mathematically equivalent to just a single layer.

**The `DenseLayer` Class (`layer.py`)**

Our library implements this concept in the `DenseLayer` class.

* **Initialization (`__init__`)**:
    * When you create a `DenseLayer`, you specify the number of input features (`input_size`) and the number of output features (`output_size`).
    * It automatically creates the `weights` and `biases` needed for the linear transformation. These are initialized as `MLArray` objects, meaning they are ready for automatic differentiation!
        * `self.weights`: An `MLArray` of shape `(input_size, output_size)`. It's typically initialized using strategies like Xavier or He initialization (provided via `weight_initializer`, defaulting to `initializers.XavierUniform`) to help with training stability.
        * `self.biases`: An `MLArray` of shape `(1, output_size)`, usually initialized to zeros (`zeros(1, output_size)`). It gets broadcasted during addition.
    * You also specify the `activation_function` (like `activation.relu` or `activation.tanh`) to be applied after the linear step.

* **Forward Pass (`forward`)**:
    * The `.forward(input_data)` method defines the layer's core computation.
    * It takes an `input_data` (`MLArray`) and performs the calculation: `z = input_data @ self.weights + self.biases`. Notice the use of `@` for matrix multiplication and `+` for addition – these are `MLArray` operations that build our computational graph using the underlying `Value` objects.
    * Finally, it applies the chosen `self.activation_function` to `z` and returns the result (`MLArray`).

* **Parameter Update (`update`)**:
    * This method is a helper used during training. It doesn't perform calculations itself but interacts with an `optimizer` object (which we'll see in the `NeuralNetwork` class) to adjust the layer's `self.weights` and `self.biases` based on the computed gradients.

## Assembling Layers - The `NeuralNetwork`

A single layer isn't usually enough for complex tasks. A neural network typically consists of multiple layers stacked sequentially. The output of one layer becomes the input to the next.

**The `NeuralNetwork` Class (`neural_network.py`)**

This class manages a sequence of layers and orchestrates the training process.

* **Initialization (`__init__`)**:
    * You create a `NeuralNetwork` by providing a list of layer objects (e.g., `[DenseLayer(...), DenseLayer(...)]`), a `loss_function` (like `losses.mse_loss` for regression), and an `optimizer` (like `optimizers.SGD` or `optimizers.AdaGrad`).
    * The `optimizer` holds the logic for *how* to update the parameters based on gradients (e.g., simple gradient descent, or more advanced methods).

* **Forward Pass (`forward`)**:
    * The network's `.forward(input_data)` method implements the feedforward process.
    * It takes the initial `input_data` and passes it through the first layer in `self.layers`.
    * The output of that layer is then fed as input to the second layer, and so on, until the data has passed through all layers.
    * The final output of the last layer is returned. Because each layer's forward pass builds a computational graph, the network's forward pass chains these graphs together.

* **Training (`train`)**:
    * This is where the learning happens! The `.train(X, y, epochs, ...)` method implements the training loop. Here's a breakdown of one epoch (one pass through the entire dataset):
        1.  **Forward Pass:** Calculate the network's predictions (`y_pred`) for the input data `X` by calling `self.forward(X)`. This builds the complete computational graph from input `X` to `y_pred`.
        2.  **Compute Loss:** Use the specified `self.loss_function` to compare the predictions `y_pred` with the true target values `y`. This computes the loss, which is typically a single scalar `Value` (inside an `MLArray`). The loss function itself involves `Value`/`MLArray` operations, so it extends the computational graph. The `loss` variable now represents the final node of our graph for this iteration.
        3.  **Backward Pass:** Call `loss.backward()`. This triggers the automatic differentiation process, calculating the gradient of the loss with respect to *every* `Value` object involved in the computation, all the way back to the weights and biases in *each layer*, and even the input `X` (though we usually don't need the gradients for `X`).
        4.  **Update Parameters:** Iterate through each layer (`for idx, layer in enumerate(self.layers):`) and call `layer.update(self.optimizer, idx)`. The optimizer uses the gradients stored in `layer.weights.grad()` and `layer.biases.grad()` (which were populated by `loss.backward()`) along with its internal logic (e.g., learning rate) to compute updated values for the weights and biases.
        5.  **Reset Gradients:** **Crucially**, gradients computed by `.backward()` are *added* to the existing `.grad` attributes. Before the next training iteration, we must reset the gradients for all parameters (and potentially inputs/targets if they are part of subsequent graphs) back to zero. This is done using the `.restart()` method on the relevant `MLArray`s (weights, biases, and sometimes X/y if they persist). If we didn't do this, gradients from previous iterations would incorrectly influence the updates in the current iteration.
        6.  **Repeat:** Go back to step 1 for the next epoch.

* **Representation (`__repr__`)**:
    * Printing the `NeuralNetwork` object gives a nicely formatted summary including the layers, their shapes, activation functions, parameter counts, the chosen optimizer and loss function, and estimated memory usage.

## The Training Lifecycle

Putting it all together, training a neural network involves repeatedly cycling through these steps:

1.  **Feed Forward:** Pass the input data through the network layers to get a prediction. (`network.forward(X)`)
2.  **Calculate Loss:** Compare the prediction to the true values using a loss function. (`loss = loss_fn(y_pred, y)`)
3.  **Backpropagate:** Calculate the gradients of the loss with respect to all network parameters. (`loss.backward()`)
4.  **Update Weights:** Adjust the network parameters using the optimizer based on the calculated gradients. (`optimizer` acts via `layer.update()`)
5.  **Zero Gradients:** Reset all parameter gradients before the next iteration. (`params.restart()`)

By using `DenseLayer` to define the transformations and `NeuralNetwork` to manage the sequence and the training loop, we can leverage the underlying `MLArray` and `Value` objects to build and train sophisticated models effectively.
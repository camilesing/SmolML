# SmolML - The utility room!

Welcome to the util components of SmolML! This directory houses the supporting building blocks required for constructing, training, and analyzing machine learning models within our framework. Think of these modules as the assistance tools and ingredients you'll use repeatedly.

Everything here leverages the `MLArray` for handling numerical data and automatic differentiation (where applicable).

## Directory Structure

```
.
├── activation.py       # Non-linear functions for neural networks
├── initializers.py     # Strategies for setting initial model weights
├── losses.py           # Functions to measure model error
├── memory.py           # Utilities to calculate memory usage
└── optimizers.py       # Algorithms to update model weights during training
```

---

## Activation Functions (`activation.py`)

**Why do we need them?**

Imagine building a neural network. If you just stack linear operations (like matrix multiplications and additions), the entire network, no matter how deep, behaves like a single *linear* transformation. This severely limits the network's ability to learn complex, non-linear patterns often found in real-world data (like image recognition, language translation, etc.).

**Activation functions** introduce **non-linearity** into the network, typically applied element-wise after a linear transformation in a layer. This allows the network to approximate much more complicated functions.

**How they work (generally):**

Each activation function takes a numerical input (often the output of a linear layer) and applies a specific mathematical transformation to it. Most functions provided here operate element-by-element on an `MLArray`.

**Key Activation Functions Provided:**

* **`relu(x)` (Rectified Linear Unit):**
    * *Concept:* Outputs the input if it's positive, otherwise outputs zero ($f(x) = \max(0, x)$).
    * *Why:* Computationally very efficient, helps mitigate vanishing gradients, and is the most common choice for hidden layers in deep networks.
    * *Code:* Uses `_element_wise_activation` with `val.relu()`.

* **`leaky_relu(x, alpha=0.01)`:**
    * *Concept:* Like ReLU, but allows a small, non-zero gradient for negative inputs ($f(x) = x$ if $x > 0$, else $f(x) = \alpha x$).
    * *Why:* Attempts to fix the "dying ReLU" problem where neurons can become inactive if they consistently output negative values.
    * *Code:* Uses `_element_wise_activation` with a custom lambda checking the value.

* **`elu(x, alpha=1.0)` (Exponential Linear Unit):**
    * *Concept:* Similar to Leaky ReLU but uses an exponential curve for negative inputs ($f(x) = x$ if $x > 0$, else $f(x) = \alpha (e^x - 1)$).
    * *Why:* Aims to have negative outputs closer to -1 on average, potentially speeding up learning. Smoother than ReLU/Leaky ReLU.
    * *Code:* Uses `_element_wise_activation` with a custom lambda.

* **`sigmoid(x)`:**
    * *Concept:* Squashes input values into the range (0, 1) using the formula $f(x) = \frac{1}{1 + e^{-x}}$.
    * *Why:* Historically popular, often used in the output layer for **binary classification** problems to interpret the output as a probability. Can suffer from vanishing gradients in deep networks.
    * *Code:* Uses `_element_wise_activation` with the sigmoid formula.

* **`softmax(x, axis=-1)`:**
    * *Concept:* Transforms a vector of numbers into a probability distribution (values are non-negative and sum to 1). It exponentiates inputs and then normalizes them.
    * *Why:* Essential for the output layer in **multi-class classification** problems. Each output node represents the probability of the input belonging to a specific class. Note the `axis` argument determines *along which dimension* the normalization occurs.
    * *Code:* Handles scalars, 1D, and multi-dimensional `MLArray`s, applying the softmax logic recursively along the specified `axis`. Includes numerical stability improvements (subtracting the max value before exponentiation).

* **`tanh(x)` (Hyperbolic Tangent):**
    * *Concept:* Squashes input values into the range (-1, 1) ($f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$).
    * *Why:* Similar use cases to sigmoid but zero-centered output range can sometimes be beneficial. Also susceptible to vanishing gradients.
    * *Code:* Uses `_element_wise_activation` with `val.tanh()`.

* **`linear(x)`:**
    * *Concept:* Simply returns the input unchanged ($f(x) = x$).
    * *Why:* Used when no non-linearity is needed, for example, in the output layer of a regression model.

*(See `activation.py` for implementation details)*

---

## Weight Initializers (`initializers.py`)

**Why is initialization important?**

When you create a neural network layer, its weights and biases need starting values. Choosing these initial values poorly can drastically hinder training:
* **Too small:** Gradients might become tiny as they propagate backward (vanishing gradients), making learning extremely slow or impossible.
* **Too large:** Gradients might explode, leading to unstable training (NaN values).
* **Symmetry:** If all weights start the same, neurons in the same layer will learn the same thing, defeating the purpose of having multiple neurons.

**Weight initializers** provide strategies to set these starting weights intelligently, breaking symmetry and keeping signals/gradients in a reasonable range to promote stable and efficient learning.

**How they work (generally):**

They typically draw random numbers from specific probability distributions (like uniform or normal/Gaussian) whose parameters (like variance) are scaled based on the number of input units (`fan_in`) and/or output units (`fan_out`) of the layer. This scaling helps maintain signal variance as it passes through layers.

**Key Initializers Provided:**

* **`WeightInitializer` (Base Class):** Defines the interface and provides a helper `_create_array` for generating `MLArray`s.
* **`XavierUniform` / `XavierNormal` (Glorot Initialization):**
    * *Concept:* Scales the initialization variance based on both `fan_in` and `fan_out`. Aims to keep variance consistent forwards and backwards.
    * *Why:* Works well with activation functions like `sigmoid` and `tanh`. `XavierUniform` uses a uniform distribution, `XavierNormal` uses a normal distribution.
    * *Code:* Calculates limits/standard deviation based on $\sqrt{6 / (fan\_in + fan\_out)}$ (Uniform) or $\sqrt{2 / (fan\_in + fan\_out)}$ (Normal).

* **`HeInitialization` (Kaiming Initialization):**
    * *Concept:* Scales the initialization variance based primarily on `fan_in`.
    * *Why:* Specifically designed for and works well with `relu` and its variants, accounting for the fact that ReLU zeros out half the inputs. Uses a normal distribution.
    * *Code:* Calculates standard deviation based on $\sqrt{2 / fan\_in}$.

*(See `initializers.py` for implementation details)*

---

## Loss Functions (`losses.py`)

**What is a loss function?**

During training, we need a way to measure how "wrong" our model's predictions are compared to the actual target values (ground truth). This measure is the **loss** (or cost, or error). The goal of training is to adjust the model's parameters (weights/biases) to **minimize** this loss value.

**How they work:**

A loss function takes the model's predictions (`y_pred`) and the true target values (`y_true`) as input and outputs a single scalar value representing the average error across the samples. Different loss functions are suited for different types of problems (regression vs. classification) and have different properties (e.g., sensitivity to outliers).

**Key Loss Functions Provided:**

* **`mse_loss(y_pred, y_true)` (Mean Squared Error):**
    * *Concept:* Calculates the average of the squared differences between predictions and true values: $L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2$.
    * *Why:* Standard choice for **regression** problems. Penalizes larger errors more heavily due to the squaring. Sensitive to outliers.
    * *Code:* Implements the formula using `MLArray` operations.

* **`mae_loss(y_pred, y_true)` (Mean Absolute Error):**
    * *Concept:* Calculates the average of the absolute differences between predictions and true values: $L = \frac{1}{N} \sum_{i=1}^{N} |y_{pred, i} - y_{true, i}|$.
    * *Why:* Another common choice for **regression**. Less sensitive to outliers compared to MSE because errors are not squared.
    * *Code:* Implements the formula using `MLArray` operations.

* **`binary_cross_entropy(y_pred, y_true)`:**
    * *Concept:* Measures the difference between two probability distributions (the predicted probability and the true label 0 or 1).
    * *Why:* The standard loss function for **binary classification** problems where the model outputs a probability (usually via a `sigmoid` activation). Expects `y_pred` values between 0 and 1.
    * *Code:* Implements the formula, includes clipping (`epsilon`) to avoid `log(0)`.

* **`categorical_cross_entropy(y_pred, y_true)`:**
    * *Concept:* Extends binary cross-entropy to multiple classes. Compares the predicted probability distribution (output by `softmax`) to the true distribution (usually one-hot encoded).
    * *Why:* The standard loss function for **multi-class classification** problems. Expects `y_pred` to be a probability distribution across classes for each sample.
    * *Code:* Implements the formula, includes clipping (`epsilon`), sums across the class axis, then averages over samples.

* **`huber_loss(y_pred, y_true, delta=1.0)`:**
    * *Concept:* A hybrid loss function that behaves like MSE for small errors (quadratic) and like MAE for large errors (linear). The `delta` parameter controls the transition point.
    * *Why:* Useful for **regression** problems where you want robustness to outliers (like MAE) but also smoother gradients around the minimum (like MSE).
    * *Code:* Implements the conditional logic using `MLArray` operations.

*(See `losses.py` for implementation details)*

---

## Optimizers (`optimizers.py`)

**What do optimizers do?**

Once we have calculated the loss, we know how wrong the model is. We also use backpropagation (handled by `MLArray`'s automatic differentiation) to calculate the **gradients** – how the loss changes with respect to each weight and bias in the model.

The **optimizer** is the algorithm that uses these gradients to actually *update* the model's parameters (weights and biases) in a way that aims to decrease the loss over time.

**How they work (generally):**

They implement different update rules based on the gradients and often maintain internal "state" (like past gradients or momentum) to improve convergence speed and stability. The core idea is usually a variation of **gradient descent**: move the parameters slightly in the direction opposite to the gradient. The `learning_rate` controls the size of these steps.

**Key Optimizers Provided:**

* **`Optimizer` (Base Class):** Defines the interface, requiring an `update` method. Stores the `learning_rate`.
* **`SGD` (Stochastic Gradient Descent):**
    * *Concept:* The simplest optimizer. Updates parameters directly opposite to the gradient, scaled by the learning rate ($\theta = \theta - \alpha \nabla_\theta L$). "Stochastic" usually means the gradient is computed on a mini-batch of data, not the full dataset.
    * *Why:* Easy to understand, but can be slow, get stuck in local minima, or oscillate.
    * *Code:* Implements the basic update rule.

* **`SGDMomentum`:**
    * *Concept:* Adds a "momentum" term that accumulates an exponentially decaying average of past gradients. This helps accelerate descent in consistent directions and dampens oscillations ($v = \beta v + \alpha \nabla_\theta L$, $\theta = \theta - v$).
    * *Why:* Often converges faster and more reliably than basic SGD. Introduces `momentum_coefficient` ($\beta$) and maintains velocity (`self.velocities`) state per parameter.
    * *Code:* Implements the momentum update rule, storing velocities.

* **`AdaGrad` (Adaptive Gradient):**
    * *Concept:* Adapts the learning rate *per parameter*, using smaller updates for frequently changing parameters and larger updates for infrequent ones. It divides the learning rate by the square root of the sum of past squared gradients ($\theta = \theta - \frac{\alpha}{\sqrt{G + \epsilon}} \nabla_\theta L$).
    * *Why:* Good for sparse data (like in NLP). However, the learning rate monotonically decreases and can become too small. Maintains sum of squared gradients (`self.squared_gradients`) state.
    * *Code:* Implements the AdaGrad update rule, storing squared gradients.

* **`Adam` (Adaptive Moment Estimation):**
    * *Concept:* Combines the ideas of Momentum (using an exponentially decaying average of past gradients - 1st moment) and RMSProp/AdaGrad (using an exponentially decaying average of past *squared* gradients - 2nd moment). Includes bias correction terms to account for initialization.
    * *Why:* Often considered a robust, effective default optimizer for many problems. Requires tuning `learning_rate`, `exp_decay_gradients` ($\beta_1$), and `exp_decay_squared` ($\beta_2$). Maintains 1st and 2nd moment estimates (`self.gradients_momentum`, `self.squared_gradients_momentum`) and a `timestep`.
    * *Code:* Implements the Adam update rule with bias correction, storing moment estimates.

*(See `optimizers.py` for implementation details)*

---

## Memory Utilities (`memory.py`)

**Why measure memory?**

Machine learning models, especially large ones like deep neural networks or complex random forests, can consume significant amounts of memory (RAM). Understanding the memory footprint of your data structures (`Value`, `MLArray`) and models (`NeuralNetwork`, `DecisionTree`, etc.) is crucial for:
* **Resource Planning:** Ensuring your hardware can handle the model size.
* **Debugging:** Identifying memory bottlenecks or unexpected usage.
* **Optimization:** Comparing the memory efficiency of different model architectures or implementations.

**How they work:**

These utility functions use Python's built-in `sys.getsizeof` to estimate the memory usage of objects. For complex objects like `MLArray` or model structures (which contain nested objects or lists), they recursively traverse the components and sum their sizes.

**Key Utilities Provided:**

* **`format_size(size_bytes)`:** Converts raw byte counts into human-readable formats (KB, MB, GB).
* **`calculate_value_size(value)`:** Estimates the size of a single `smolml.core.value.Value` object (including its data, grad, etc.).
* **`calculate_mlarray_size(arr)`:** Estimates the size of an `MLArray`, including the nested lists/`Value` objects it contains.
* **`calculate_neural_network_size(model)`:** Estimates the total size of a `NeuralNetwork`, including its layers (weights, biases) and optimizer state.
* **(Other similar functions):** `calculate_decision_node_size`, `calculate_regression_size`, `calculate_decision_tree_size`, `calculate_random_forest_size` estimate memory for other (likely defined elsewhere in `smolml`) model types.

**Note:** These provide *estimates*. Actual memory usage can be influenced by Python's internal memory management, shared object references, etc.

*(See `memory.py` for implementation details)*

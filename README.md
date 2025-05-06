# SmolML: Machine Learning from Scratch, Made Clear! ✨

**A pure Python machine learning library built entirely from the ground up for educational purposes. Made to understand how ML really works!**

---

## What is SmolML? 🤔

Ever wondered what goes on *inside* those powerful machine learning libraries like Scikit-learn, PyTorch, or TensorFlow? How does a neural network *actually* learn? How is gradient descent implemented?

SmolML is our answer! It's a fully functional (though simplified) machine learning library built using **only pure Python** and its basic `collections`, `random`, and `math` modules. No NumPy, no SciPy, no C++ extensions – just Python, all the way down.

The goal isn't to compete with production-grade libraries on speed or features, but to provide a **transparent, understandable, and educational** implementation of core machine learning concepts.

## Walkthrough 📖

You can read these guides of the different sections of SmolML in any order, though this list presents the recommended order for learning.

- [SmolML - Core: Automatic Differentiation & N-Dimensional Arrays](smolml\core\README.md)
- [SmolML - Regression: Predicting Continuous Values](smolml\models\regression\README.md)
- [SmolML - Neural Networks: Backpropagation to the limit](smolml\models\nn\README.md)
- [SmolML - Tree Models: Decisions, Decisions!](smolml\models\tree\README.md)
- [SmolML - K-Means: Finding Groups in Your Data!](smolml\models\unsupervised\README.md)
- [SmolML - Preprocessing: Make your data meaningful](smolml\preprocessing\README.md)
- [SmolML - The utility room!](smolml\utils\README.md)
 
## Why SmolML? The Philosophy 🎓

We believe the best way to truly understand complex topics like machine learning is often to **build them yourself**. Production libraries are fantastic tools, but their internal complexity and optimizations can sometimes obscure the fundamental principles.

SmolML strips away these layers to focus on the core ideas:
* **Learning from First Principles:** Every major component is built from scratch, letting you trace the logic from basic operations to complex algorithms.
* **Demystifying the Magic:** See how concepts like automatic differentiation (autograd), optimization algorithms, and model architectures are implemented in code.
* **Minimal Dependencies:** Relying only on Python's standard library makes the codebase accessible and easy to explore without external setup hurdles.
* **Focus on Clarity:** Code is written with understanding, not peak performance, as the primary goal.

In order to learn as much as possible, we recommend reading through the guides, checking the code, and then trying to implement your own versions of these components.

## What's Inside? Features 🛠️

SmolML provides explains the essential building blocks for any ML library:

* **The Foundation: Custom Arrays & Autograd Engine:**
    * **Automatic Differentiation (`Value`):** A simple autograd engine that tracks operations and computes gradients automatically – the heart of training neural networks!
    * **N-dimensional Arrays (`MLArray`):** A custom array implementation inspired by NumPy, supporting common mathematical operations needed for ML. Extremely inefficient due to being written in Python, but ideal for understanding N-Dimensional Arrays.

* **Essential Preprocessing:**
    * **Scalers (`StandardScaler`, `MinMaxScaler`):** Fundamental tools to prepare your data, because algorithms tend to perform better when features are on a similar scale.

* **Build Your Own Neural Networks:**
    * **Activation Functions:** Non-linearities like `relu`, `sigmoid`, `softmax`, `tanh` that allow networks to learn complex patterns. (See `smolml/utils/activation.py`)
    * **Weight Initializers:** Smart strategies (`Xavier`, `He`) to set initial network weights for stable training. (See `smolml/utils/initializers.py`)
    * **Loss Functions:** Ways to measure model error (`mse_loss`, `binary_cross_entropy`, `categorical_cross_entropy`). (See `smolml/utils/losses.py`)
    * **Optimizers:** Algorithms like `SGD`, `Adam`, and `AdaGrad` that update model weights based on gradients to minimize loss. (See `smolml/utils/optimizers.py`)

* **Classic ML Models:**
    * **Regression:** Implementations of `Linear` and `Polynomial` regression.
    * **Neural Networks:** A flexible framework for building feed-forward neural networks.
    * **Tree-Based Models:** `Decision Tree` and `Random Forest` implementations for classification and regression.
    * **K-Means:** `KMeans` clustering algorithm for grouping similar data points together.

## Who is SmolML For? 🎯

* **Students:** Learning ML concepts for the first time.
* **Developers:** Curious about the internals of ML libraries they use daily.
* **Educators:** Looking for a simple, transparent codebase to demonstrate ML principles.
* **Anyone:** Who enjoys learning by building!

## Limitations! ⚠️

Let's be crystal clear: SmolML is built for **learning**, not for breaking speed records or handling massive datasets.
* **Performance:** Being pure Python, it's WAAAY slower than libraries using optimized C/C++/Fortran backends (like NumPy).
* **Scale:** It's best suited for small datasets and toy problems where understanding the mechanics is more important than computation time.
* **Production Use:** **Do not** use SmolML for production applications. Stick to battle-tested libraries like Scikit-learn, PyTorch, TensorFlow, JAX, etc., for real-world tasks.

Think of it as learning to build a go-kart engine from scratch before driving a Formula 1 car. It teaches you the fundamentals in a hands-on way!

## Getting Started

The best way to use SmolML is to clone this repository and explore the code and examples (if available).

```bash
git clone https://github.com/rodmarkun/SmolML
cd SmolML
# Explore the code in the smolml/ directory!
```

## Contributing

Contributions are always welcome! If you're interested in contributing to SmolML, please fork the repository and create a new branch for your changes. When you're done with your changes, submit a pull request to merge your changes into the main branch.

## Supporting Escordia

If you want to support SmolML, you can:
- **Star** :star: the project in Github!
- **Donate** :coin: to my [Ko-fi](https://ko-fi.com/rodmarkun) page!
- **Share** :heart: the project with your friends!
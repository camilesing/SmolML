from ml_array import zeros

class Optimizer:
    """Base optimizer class that defines the interface for all optimizers"""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, param, grad):
        """Update rule to be implemented by specific optimizers"""
        raise NotImplementedError

class SGD(Optimizer):
    """Standard Stochastic Gradient Descent optimizer"""
    def update(self, layer, layer_idx):
        """
        Update rule for standard SGD: θ = θ - α∇θ
        where α is the learning rate.
        
        This is the most basic form of gradient descent, which directly updates
        parameters in the opposite direction of the gradient, scaled by the learning rate.
        """
        new_weights = layer.weights - self.learning_rate * layer.weights.grad()
        new_biases = layer.biases - self.learning_rate * layer.biases.grad()
        return new_weights, new_biases


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    This optimizer accelerates SGD by accumulating a velocity vector in the direction of persistent gradients,
    helping to avoid local minima and speed up convergence.
    """
    def __init__(self, learning_rate: float = 0.01, momentum_coefficient: float = 0.9):
        super().__init__(learning_rate)
        self.momentum_coefficient = momentum_coefficient
        self.velocities = {}  # Dictionary to store velocities for each layer
        
    def update(self, layer, layer_idx):
        """
        Update rule for SGD with momentum: v = βv + α∇θ, θ = θ - v
        where β is the momentum coefficient and α is the learning rate.
        """
        # Initialize velocities for this layer if not exist
        if layer_idx not in self.velocities:
            self.velocities[layer_idx] = {"weights": zeros(*layer.weights.shape),
                                        "biases": zeros(*layer.biases.shape)}
        
        # Update velocity and parameters for weights
        v_w = self.velocities[layer_idx]["weights"]
        v_w = self.momentum_coefficient * v_w + self.learning_rate * layer.weights.grad()
        self.velocities[layer_idx]["weights"] = v_w
        
        # Update velocity and parameters for biases
        v_b = self.velocities[layer_idx]["biases"]
        v_b = self.momentum_coefficient * v_b + self.learning_rate * layer.biases.grad()
        self.velocities[layer_idx]["biases"] = v_b
        
        # Compute new parameters
        new_weights = layer.weights - v_w
        new_biases = layer.biases - v_b
        
        return new_weights, new_biases


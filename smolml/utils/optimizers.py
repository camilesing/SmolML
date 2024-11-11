from smolml.core.ml_array import zeros

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


class AdaGrad(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    This optimizer accelerates SGD by accumulating a velocity vector in the direction of persistent gradients,
    helping to avoid local minima and speed up convergence.
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.epsilon = 1e-8
        self.squared_gradients = {} 
        
    def update(self, layer, layer_idx):
        """
        Update rule for SGD with momentum: v = βv + α∇θ, θ = θ - v
        where β is the momentum coefficient and α is the learning rate.
        """
        # Initialize velocities for this layer if not exist
        if layer_idx not in self.squared_gradients:
            self.squared_gradients[layer_idx] = {"weights": zeros(*layer.weights.shape),
                                        "biases": zeros(*layer.biases.shape)}
        
        self.squared_gradients[layer_idx]["weights"] += layer.weights.grad()**2 
        self.squared_gradients[layer_idx]["biases"] += layer.biases.grad()**2 

        # parameter = parameter - (α / √(G + ε)) * gradient
        new_weights = layer.weights - (self.learning_rate / (self.squared_gradients[layer_idx]["weights"] + self.epsilon).sqrt()) * layer.weights.grad()
        new_biases = layer.biases - (self.learning_rate / (self.squared_gradients[layer_idx]["biases"] + self.epsilon).sqrt()) * layer.biases.grad()
        
        return new_weights, new_biases


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    Combines the benefits of:
    1. Momentum: By keeping track of exponentially decaying gradient average
    2. RMSprop: By keeping track of exponentially decaying squared gradients
    Also includes bias correction terms to handle initialization.
    """
    def __init__(self, learning_rate: float = 0.01, exp_decay_gradients: float = 0.9, exp_decay_squared: float = 0.999):
        super().__init__(learning_rate)
        self.exp_decay_gradients = exp_decay_gradients  # β₁: Decay rate for gradient momentum
        self.exp_decay_squared = exp_decay_squared      # β₂: Decay rate for squared gradient momentum
        self.gradients_momentum = {}                    # First moment estimates
        self.squared_gradients_momentum = {}            # Second moment estimates
        self.epsilon = 1e-8                            # Small constant for numerical stability
        self.timestep = 1                              # Timestep for bias correction
        
    def update(self, layer, layer_idx):
        """
        Update rule for Adam: θ = θ - α * m̂ / (√v̂ + ε)
        where:
        - m̂ is the bias-corrected first moment estimate
        - v̂ is the bias-corrected second moment estimate
        - α is the learning rate
        - ε is a small constant for numerical stability
        """
        # Initialize velocities for this layer if not exist
        if layer_idx not in self.gradients_momentum:
            self.gradients_momentum[layer_idx] = {"weights": zeros(*layer.weights.shape),
                                                "biases": zeros(*layer.biases.shape)}
        if layer_idx not in self.squared_gradients_momentum:
            self.squared_gradients_momentum[layer_idx] = {"weights": zeros(*layer.weights.shape),
                                                "biases": zeros(*layer.biases.shape)}
        
        self.gradients_momentum[layer_idx]['weights'] = self.exp_decay_gradients * self.gradients_momentum[layer_idx]['weights'] + (1 - self.exp_decay_gradients) * layer.weights.grad()
        self.gradients_momentum[layer_idx]['biases'] = self.exp_decay_gradients * self.gradients_momentum[layer_idx]['biases'] + (1 - self.exp_decay_gradients) * layer.biases.grad()

        self.squared_gradients_momentum[layer_idx]['weights'] = self.exp_decay_squared * self.squared_gradients_momentum[layer_idx]['weights'] + (1 - self.exp_decay_squared) * layer.weights.grad()**2
        self.squared_gradients_momentum[layer_idx]['biases'] = self.exp_decay_squared * self.squared_gradients_momentum[layer_idx]['biases'] + (1 - self.exp_decay_squared) * layer.biases.grad()**2

        m_w = self.gradients_momentum[layer_idx]['weights'] / (1 - self.exp_decay_gradients ** self.timestep)
        m_b = self.gradients_momentum[layer_idx]['biases'] / (1 - self.exp_decay_gradients ** self.timestep)

        v_w = self.squared_gradients_momentum[layer_idx]['weights'] / (1 - self.exp_decay_squared ** self.timestep)
        v_b = self.squared_gradients_momentum[layer_idx]['biases'] / (1 - self.exp_decay_squared ** self.timestep)

        # Compute new parameters
        new_weights = layer.weights - self.learning_rate * m_w / (v_w.sqrt() + self.epsilon)
        new_biases = layer.biases - self.learning_rate * m_b / (v_b.sqrt() + self.epsilon)
        
        return new_weights, new_biases
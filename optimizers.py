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
    def update(self, param, grad):
        return param - self.learning_rate * grad

""" 
    Not working yet:
    class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum_coefficient: float = 0.9):
        super().__init__(learning_rate)
        self.momentum_coefficient = momentum_coefficient
        self.velocities = {}
    
    def update(self, param, grad, param_identifier):
        if param_identifier not in self.velocities:
            self.velocities[param_identifier] = zeros(*param.shape)

        if self.velocities[param_identifier].shape != param.shape:
            print(self.velocities.keys())
        
        v = self.velocities[param_identifier]
        v = self.momentum_coefficient * v + self.learning_rate * grad
        self.velocities[param_identifier] = v
        return param - v """


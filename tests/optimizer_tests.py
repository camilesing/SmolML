import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray, zeros
import smolml.utils.losses as losses
import smolml.utils.activation as activation
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
import random
import numpy as np
from smolml.models.nn.neural_network import NeuralNetwork
from smolml.models.nn.layer import DenseLayer

def create_network(optimizer):
    """Helper function to create a network with specified optimizer"""
    input_size = 2
    hidden_size = 32
    output_size = 1
    
    return NeuralNetwork([
        DenseLayer(input_size, hidden_size, activation.relu),
        DenseLayer(hidden_size, output_size, activation.tanh)
    ], losses.mse_loss, optimizer)

def train_and_get_losses(network, X, y, epochs=100):
    """Train network and return list of losses"""
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = network.forward(X)
        loss = network.loss_function(y_pred, y)
        
        # Store loss
        losses.append(loss.data.data)
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss.data.data}")
        
        # Backward pass and update
        loss.backward()
        for idx, layer in enumerate(network.layers):
            layer.update(network.optimizer, idx)
        
        # Reset gradients
        X.restart()
        y.restart()
        for layer in network.layers:
            layer.weights.restart()
            layer.biases.restart()
    
    return losses

def compare_optimizers():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create data
    X = MLArray([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = MLArray([[0], [1], [1], [0]])

    # Create networks with different optimizers
    network_sgd = create_network(optimizers.SGD(learning_rate=0.1))
    network_momentum = create_network(optimizers.SGDMomentum(learning_rate=0.1, momentum_coefficient=0.9))
    network_adagrad = create_network(optimizers.AdaGrad(learning_rate=0.1))
    network_adam = create_network(optimizers.Adam(learning_rate=0.01)) 

    # Train networks
    losses_sgd = train_and_get_losses(network_sgd, X, y)
    losses_momentum = train_and_get_losses(network_momentum, X, y)
    losses_adagrad = train_and_get_losses(network_adagrad, X, y)
    losses_adam = train_and_get_losses(network_adam, X, y)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(losses_sgd, label='SGD')
    plt.plot(losses_momentum, label='SGD with Momentum')
    plt.plot(losses_adagrad, label='AdaGrad')
    plt.plot(losses_adam, label='Adam', linestyle='--')  # Different line style to distinguish
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale helps visualize convergence differences
    plt.show()

    # Print final losses
    print(f"Final loss SGD: {losses_sgd[-1]:.6f}")
    print(f"Final loss SGD with Momentum: {losses_momentum[-1]:.6f}")
    print(f"Final loss AdaGrad: {losses_adagrad[-1]:.6f}")
    print(f"Final loss Adam: {losses_adam[-1]:.6f}")

if __name__ == '__main__':
    compare_optimizers()
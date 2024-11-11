from smolml.core.ml_array import MLArray
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
    def __init__(self, layers: list, loss_function: callable, optimizer: optimizers.Optimizer = optimizers.SGD()) -> None:
        """
        Initializes the network with a list of layers and a loss function for training.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer if optimizer is not None else optimizers.SGD()

    def forward(self, input_data):
        """
        Performs forward pass by sequentially applying each layer's transformation.
        """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def train(self, X, y, epochs, print_every=None):
        """
        Trains the network using gradient descent for the specified number of epochs.
        Prints loss every 100 epochs to monitor training progress.
        """
        losses = []
        for epoch in range(epochs):
            # Forward pass through the network
            y_pred = self.forward(X)
            
            # Compute loss between predictions and targets
            loss = self.loss_function(y_pred, y)
            losses.append(loss.data.data)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update parameters in each layer
            for idx, layer in enumerate(self.layers):
                layer.update(self.optimizer, idx)
            
            # Reset gradients for next iteration
            X.restart()
            y.restart()
            for layer in self.layers:
                layer.weights.restart()
                layer.biases.restart()
                
            if print_every == None:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")
            else:
                # Print training progress
                if (epoch+1) % print_every == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")

            return losses

def example_neural_network():
    # Example usage
    input_size = 2
    hidden_size = 32
    output_size = 1

    # Create the neural network
    nn = NeuralNetwork([
        DenseLayer(input_size, hidden_size, activation.relu),
        DenseLayer(hidden_size, output_size, activation.tanh)
    ], losses.mse_loss, optimizers.AdaGrad(learning_rate=0.1))

    # Generate some dummy data
    X = MLArray([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = MLArray([[0], [1], [1], [0]])

    # Train the network
    nn.train(X, y, epochs=100)

    y_pred = nn.forward(X)
    print(y_pred)

if __name__ == '__main__':
    example_neural_network()
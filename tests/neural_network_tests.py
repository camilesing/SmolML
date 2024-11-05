import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ml_array import MLArray
from neural_network import NeuralNetwork, DenseLayer
import activation
import losses
import optimizers

class TestNeuralNetworkVsTensorflow(unittest.TestCase):
    """
    Compare custom neural network implementation against TensorFlow
    using the make_moons dataset
    """
    
    def setUp(self):
        """
        Set up dataset and models
        """
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Generate moon dataset
        X, y = make_moons(n_samples=200, noise=0.1)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert data for custom implementation
        self.X_train_ml = MLArray([[float(x) for x in row] for row in self.X_train])
        self.y_train_ml = MLArray([[float(y)] for y in self.y_train])
        self.X_test_ml = MLArray([[float(x) for x in row] for row in self.X_test])
        self.y_test_ml = MLArray([[float(y)] for y in self.y_test])
        
        # Model parameters
        self.input_size = 2
        self.hidden_size = 64
        self.output_size = 1
        self.epochs = 50
        self.learning_rate = 0.25
        
        # Initialize models
        self.custom_model = self._create_custom_model()
        self.tf_model = self._create_tf_model()

    def _create_custom_model(self):
        """
        Create custom neural network with same architecture
        """
        return NeuralNetwork([
            DenseLayer(self.input_size, self.hidden_size, activation.relu),
            DenseLayer(self.hidden_size, self.output_size, activation.sigmoid)
        ], losses.binary_cross_entropy, optimizer=optimizers.SGD(learning_rate=1))

    def _create_tf_model(self):
        """
        Create equivalent TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _plot_decision_boundary(self, model, is_tf=False):
        """
        Plot decision boundary for either model
        """
        x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
        y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh grid points
        if is_tf:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            X_mesh = MLArray([[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())])
            Z = model.forward(X_mesh).to_list()
        Z = np.array(Z).reshape(xx.shape)
        
        return xx, yy, Z

    def test_compare_models(self):
        """
        Train and compare both models
        """
        # Train custom model
        print("\nTraining custom model...")
        custom_history = []
        for epoch in range(self.epochs):
            y_pred = self.custom_model.forward(self.X_train_ml)
            loss = self.custom_model.loss_function(y_pred, self.y_train_ml)
            loss.backward()
            
            for layer in self.custom_model.layers:
                layer.update(self.custom_model.optimizer)
            
            # Reset computational graph
            self.X_train_ml = self.X_train_ml.restart()
            self.y_train_ml = self.y_train_ml.restart()
            for layer in self.custom_model.layers:
                layer.weights = layer.weights.restart()
                layer.biases = layer.biases.restart()
            
            custom_history.append(float(loss.data.data))
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.data.data}")
        
        # Train TensorFlow model
        print("\nTraining TensorFlow model...")
        tf_history = self.tf_model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=len(self.X_train),
            verbose=0
        )
        
        # Plot training curves
        plt.figure(figsize=(12, 4))

        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), custom_history, label='Custom NN')  # Changed this line
        plt.plot(range(self.epochs), tf_history.history['loss'], label='TensorFlow')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot 2: Decision Boundaries
        plt.subplot(1, 2, 2)
        
        # Plot decision boundaries
        xx, yy, Z_custom = self._plot_decision_boundary(self.custom_model)
        plt.contourf(xx, yy, Z_custom > 0.5, alpha=0.4)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, alpha=0.8)
        plt.title('Decision Boundaries (Custom NN)')
        
        plt.tight_layout()
        plt.savefig('neural_network_comparison.png')
        plt.close()
        
        # Compute and print accuracies
        custom_pred = np.array(self.custom_model.forward(self.X_test_ml).to_list()) > 0.5
        tf_pred = self.tf_model.predict(self.X_test) > 0.5
        
        custom_accuracy = np.mean(custom_pred.flatten() == self.y_test)
        tf_accuracy = np.mean(tf_pred.flatten() == self.y_test)
        
        print("\nTest Accuracies:")
        print(f"Custom NN: {custom_accuracy:.4f}")
        print(f"TensorFlow: {tf_accuracy:.4f}")
        
        # Assert models achieve reasonable accuracy
        self.assertGreater(custom_accuracy, 0.8, "Custom model accuracy should be > 80%")
        self.assertGreater(tf_accuracy, 0.8, "TensorFlow model accuracy should be > 80%")

if __name__ == '__main__':
    unittest.main()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from smolml.core.ml_array import MLArray, randn, ones
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import smolml.utils.initializers as initializers
import smolml.utils.losses as losses
from smolml.models.regression import LinearRegression, PolynomialRegression

class TestRegressionVisualization(unittest.TestCase):
    """
    Test and visualize linear and polynomial regression implementations
    with interactive epoch slider
    """
    
    def setUp(self):
        """
        Set up common parameters and styling for tests
        """
        np.random.seed(42)
        
        # Training parameters
        self.iterations = 100
        self.learning_rate = 0.1
        self.epochs_to_store = [0, 5, 10, 25, 50, 100]
        
        # Set plotting style
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.facecolor'] = '#f0f0f0'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        plt.rcParams['grid.color'] = '#ffffff'
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1

    def generate_linear_data(self, size=25):
        """Generate data with linear relationship plus noise"""
        X = randn(size, 1)
        y = X * 2 + 1 + randn(size, 1) * 0.1
        return X, y

    def generate_nonlinear_data(self, size=25):
        """Generate data with polynomial relationship plus noise"""
        X = randn(size, 1)
        y = X * 2 + X * X * 3 + 1 + randn(size, 1) * 0.1
        return X, y

    def train_and_visualize(self, model, X, y, title):
        """Train model and create interactive visualization"""
        # Store predictions history
        predictions_history = []
        
        # Training loop
        for i in range(self.iterations):
            y_pred = model.predict(X)
            loss = losses.mse_loss(y, y_pred)
            loss.backward()
            model.update_parameters()
            
            # Reset gradients
            X = X.restart()
            y = y.restart()
            model.weights = model.weights.restart()
            model.bias = model.bias.restart()
            
            if i in self.epochs_to_store:
                predictions_history.append(y_pred.to_list())
            
            if (i+1) % 10 == 0:
                print(f"Epoch {i + 1}/{self.iterations}, Loss: {loss.data}")
        
        # Convert to numpy for plotting
        X_np = np.array(X.to_list())
        y_np = np.array(y.to_list())
        
        # Sort for smooth curve plotting
        sort_idx = np.argsort(X_np.flatten())
        X_np = X_np[sort_idx]
        y_np = y_np[sort_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        scatter = ax.scatter(X_np, y_np, c='#1E88E5', alpha=0.6, label='Data')
        predictions_sorted = [np.array(pred)[sort_idx] for pred in predictions_history]
        line, = ax.plot(X_np, predictions_sorted[0], color='#D81B60', lw=2, label='Prediction')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#d3d3d3')
        slider = Slider(slider_ax, 'Epoch', 0, len(self.epochs_to_store) - 1,
                       valinit=0, valstep=1, color='#FFC107')
        
        def update(val):
            epoch_index = int(slider.val)
            line.set_ydata(predictions_sorted[epoch_index])
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Add epoch text
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold')
        
        def update_epoch_text(val):
            epoch_index = int(slider.val)
            epoch_text.set_text(f'Epoch: {self.epochs_to_store[epoch_index]}')
        
        slider.on_changed(update_epoch_text)
        update_epoch_text(0)  # Initialize text
        
        plt.show()
        
        # Print final parameters
        print("\nFinal Parameters:")
        print("Weights:", model.weights.data)
        print("Bias:", model.bias.data)
        
        return predictions_history, loss.data

    def test_linear_regression(self):
        """Test linear regression with visualization"""
        print("\nTesting Linear Regression...")
        X, y = self.generate_linear_data()
        
        model = LinearRegression(iterations=self.iterations, 
                               learning_rate=self.learning_rate)
        model.initialize_weights(1, initializers.XavierNormal())
        
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Linear Regression: Data vs Predictions'
        )
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Assuming convergence

    def test_polynomial_regression(self):
        """Test polynomial regression with visualization"""
        print("\nTesting Polynomial Regression...")
        X, y = self.generate_nonlinear_data()
        
        model = PolynomialRegression(degree=2, 
                                   iterations=self.iterations,
                                   learning_rate=self.learning_rate)
        model.initialize_weights(2, initializers.XavierNormal())
        
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Polynomial Regression: Data vs Predictions'
        )
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Assuming convergence

if __name__ == '__main__':
    unittest.main()
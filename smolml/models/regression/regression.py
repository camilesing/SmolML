import smolml.utils.initializers
from smolml.core.ml_array import MLArray
import smolml.core.ml_array as ml_array
import smolml.utils.losses as losses

"""
//////////////////
/// REGRESSION ///
//////////////////
"""

class LinearRegression:
   """
   Implements linear regression using gradient descent optimization.
   The model learns to fit: y = X @ weights + bias
   """
   def __init__(self, iterations, learning_rate) -> None:
       """
       Initializes regression model with training parameters.
       """
       self.iterations = iterations
       self.learning_rate = learning_rate 
       
   def initialize_weights(self, input_size, weight_initializer):
       """
       Initializes weights using specified initializer and sets bias to ones.
       """
       self.weights = weight_initializer.initialize((input_size,))
       self.bias = ml_array.ones((1))

   def fit(self, X, y):
       """
       Trains the model using gradient descent for specified iterations.
       Prints loss every 100 epochs to monitor convergence.
       """
       for i in range(self.iterations):
           # Make prediction 
           y_pred = self.predict(X)
           # Compute loss
           loss = losses.mse_loss(y, y_pred)
           # Backward pass
           loss.backward()

           # Update parameters
           self.weights = self.weights - self.learning_rate * self.weights.grad()
           self.bias = self.bias - self.learning_rate * self.bias.grad()

           # Reset gradients
           X, y = self.restart(X, y)

           if (i+1) % 100 == 0:
               print(f"Epoch {i + 1}/{self.iterations}, Loss: {loss.data}")

   def update_parameters(self):
       """
       Updates weights and bias using computed gradients.
       """
       self.weights = self.weights - self.learning_rate * self.weights.grad()
       self.bias = self.bias - self.learning_rate * self.bias.grad()

   def predict(self, X):
       """
       Makes predictions using linear model equation.
       """
       return X @ self.weights + self.bias
   
   def restart(self, X, y):
       """
       Resets gradients for all parameters and data for next iteration.
       """
       X = X.restart()
       y = y.restart()
       self.weights = self.weights.restart()
       self.bias = self.bias.restart()
       return X, y

class PolynomialRegression(LinearRegression):
   """
   Extends linear regression to fit polynomial relationships.
   Transforms features into polynomial terms before fitting.
   """
   def __init__(self, degree, iterations, learning_rate):
       """
       Initializes polynomial model with degree and training parameters.
       """
       super().__init__(iterations, learning_rate)
       self.degree = degree
       
   def transform_features(self, X):
       """
       Creates polynomial features up to specified degree.
       For input X and degree 2, outputs [X, X^2].
       """
       features = [X]
       for d in range(2, self.degree + 1):
           # Use element-wise multiplication for power
           power = X
           for _ in range(d-1):
               power = power * X
           features.append(power)
           
       # Concatenate features side by side
       result = features[0]
       for feature in features[1:]:
           # Assuming arrays are 2D with shape (n, 1)
           new_data = []
           for i in range(len(result.data)):
               new_data.append(result.data[i] + feature.data[i])
           result = MLArray(new_data)
           
       return result

   def initialize_weights(self, input_size, weight_initializer):
       """
       Initializes weights for each polynomial term.
       """
       self.weights = weight_initializer.initialize((self.degree,))
       self.bias = ml_array.ones((1))

   def predict(self, X):
       """
       Makes predictions after transforming features to polynomial form.
       """
       X_poly = self.transform_features(X)
       return X_poly @ self.weights + self.bias

   def fit(self, X, y):
       """
       Transforms features to polynomial form before training.
       """
       X_poly = self.transform_features(X)
       super().fit(X_poly, y)
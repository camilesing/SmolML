import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
import activation
from ml_array import MLArray

class TestActivationsVsTensorflow(unittest.TestCase):
    """
    Tests comparing custom activation functions against TensorFlow implementations
    """
    
    def setUp(self):
        """
        Set up test data in different shapes for thorough testing
        """
        # Random seed for reproducibility
        np.random.seed(42)
        
        # Generate test data
        self.scalar_np = np.random.randn()
        self.vector_np = np.random.randn(5)
        self.matrix_np = np.random.randn(3, 4)
        self.tensor_np = np.random.randn(2, 3, 4)
        
        # Convert to MLArray format
        self.scalar_ml = MLArray(float(self.scalar_np))
        self.vector_ml = MLArray([float(x) for x in self.vector_np])
        self.matrix_ml = MLArray([[float(x) for x in row] for row in self.matrix_np])
        self.tensor_ml = MLArray([[[float(x) for x in matrix] for matrix in tensor] for tensor in self.tensor_np])

    def _compare_outputs(self, ml_output, tf_output, places=5):
        """
        Helper method to compare MLArray output with TensorFlow output
        """
        # Convert MLArray output to regular Python list/float
        if len(ml_output.shape) == 0:  # scalar
            ml_result = ml_output.data.data
            tf_result = float(tf_output.numpy())
            self.assertAlmostEqual(ml_result, tf_result, places=places)
        else:
            ml_result = ml_output.to_list()
            tf_result = tf_output.numpy()
            # Recursively compare nested structures
            self._compare_nested(ml_result, tf_result, places)

    def _compare_nested(self, ml_list, tf_array, places):
        """
        Recursively compare nested lists with numpy arrays
        """
        if isinstance(ml_list, (int, float)):
            self.assertAlmostEqual(ml_list, float(tf_array), places=places)
        else:
            for ml_item, tf_item in zip(ml_list, tf_array):
                self._compare_nested(ml_item, tf_item, places)

    def _print_comparison(self, name, ml_result, tf_result, input_shape=""):
        """
        Print a formatted comparison of MLArray and TensorFlow results
        """
        print(f"\n{'-'*80}")
        print(f"{name} Comparison {input_shape}")
        print(f"{'-'*80}")
        print("MLArray output:")
        if isinstance(ml_result, (int, float)):
            print(f"{ml_result:.6f}")
        else:
            print(np.array(ml_result))
        print("\nTensorFlow output:")
        print(f"{tf_result.numpy()}")
        print(f"{'-'*80}\n")

    def test_relu(self):
        """
        Test ReLU activation against TensorFlow
        """
        # Test vector case (for example output)
        ml_result = activation.relu(self.vector_ml)
        tf_result = tf.nn.relu(self.vector_np)
        self._print_comparison("ReLU", ml_result.to_list(), tf_result, f"input shape={self.vector_np.shape}")
        self._compare_outputs(ml_result, tf_result)

    def test_sigmoid(self):
        """
        Test sigmoid activation against TensorFlow
        """
        # Test matrix case (for example output)
        ml_result = activation.sigmoid(self.matrix_ml)
        tf_result = tf.nn.sigmoid(self.matrix_np)
        self._print_comparison("Sigmoid", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)

    def test_tanh(self):
        """
        Test tanh activation against TensorFlow
        """
        # Test matrix case (for example output)
        ml_result = activation.tanh(self.matrix_ml)
        tf_result = tf.nn.tanh(self.matrix_np)
        self._print_comparison("Tanh", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)

    def test_softmax(self):
        """
        Test softmax activation against TensorFlow
        """
        # Test vector case
        ml_result = activation.softmax(self.vector_ml)
        tf_result = tf.nn.softmax(self.vector_np)
        self._print_comparison("Softmax (vector)", ml_result.to_list(), tf_result, f"input shape={self.vector_np.shape}")
        self._compare_outputs(ml_result, tf_result)
        
        # Test matrix case (last axis)
        ml_result = activation.softmax(self.matrix_ml)
        tf_result = tf.nn.softmax(self.matrix_np)
        self._print_comparison("Softmax (matrix, last axis)", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)
        
        # Test matrix case (axis=0)
        ml_result = activation.softmax(self.matrix_ml, axis=0)
        tf_result = tf.nn.softmax(self.matrix_np, axis=0)
        self._print_comparison("Softmax (matrix, axis=0)", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)

    def test_leaky_relu(self):
        """
        Test leaky ReLU activation against TensorFlow
        """
        alpha = 0.01
        # Test matrix case (for example output)
        ml_result = activation.leaky_relu(self.matrix_ml, alpha)
        tf_result = tf.nn.leaky_relu(self.matrix_np, alpha)
        self._print_comparison("Leaky ReLU", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)

    def test_elu(self):
        """
        Test ELU activation against TensorFlow
        """
        # Test matrix case (for example output)
        ml_result = activation.elu(self.matrix_ml)
        tf_result = tf.nn.elu(self.matrix_np)
        self._print_comparison("ELU", ml_result.to_list(), tf_result, f"input shape={self.matrix_np.shape}")
        self._compare_outputs(ml_result, tf_result)

if __name__ == '__main__':
    unittest.main()
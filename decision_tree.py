from ml_array import MLArray
import random
from collections import Counter
import math

class DecisionNode:
    """
    Node in decision tree that handles splitting logic.
    Can be either internal node (with split rule) or leaf (with prediction).
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Value to split feature on
        self.left = left               # Left subtree (feature <= threshold)
        self.right = right             # Right subtree (feature > threshold)
        self.value = value             # Prediction value (for leaf nodes)

class DecisionTree:
    """
    Decision Tree implementation supporting both classification and regression.
    Uses binary splitting based on feature thresholds.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, task="classification"):
        """
        Initialize decision tree with stopping criteria.
        
        max_depth: Maximum tree depth to prevent overfitting
        min_samples_split: Minimum samples required to split node
        min_samples_leaf: Minimum samples required in leaf nodes
        task: "classification" or "regression"
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.root = None

    def fit(self, X, y):
        """
        Build decision tree by recursively splitting data.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)
            
        self.n_classes = len(set(y.flatten(y.data))) if self.task == "classification" else None
        self.root = self._grow_tree(X.data, y.data)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows tree by finding best splits.
        """
        n_samples = len(X)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            self._is_pure(y)):
            return DecisionNode(value=self._leaf_value(y))

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No valid split found
            return DecisionNode(value=self._leaf_value(y))

        # Split data
        left_idxs, right_idxs = self._split_data(X, best_feature, best_threshold)
        
        # Check min_samples_leaf
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return DecisionNode(value=self._leaf_value(y))

        # Create child nodes
        left_X = [X[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        left_y = [y[i] for i in left_idxs]
        right_y = [y[i] for i in right_idxs]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)

        return DecisionNode(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        """
        Finds best feature and threshold for splitting data.
        Uses information gain for classification, MSE for regression.
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        n_features = len(X[0])
        
        for feature_idx in range(n_features):
            thresholds = sorted(set(row[feature_idx] for row in X))
            
            for threshold in thresholds:
                left_idxs, right_idxs = self._split_data(X, feature_idx, threshold)
                
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue

                left_y = [y[i] for i in left_idxs]
                right_y = [y[i] for i in right_idxs]
                
                gain = self._calculate_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split_data(self, X, feature_idx, threshold):
        """
        Splits data based on feature and threshold.
        """
        left_idxs = []
        right_idxs = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_idxs.append(i)
            else:
                right_idxs.append(i)
                
        return left_idxs, right_idxs

    def _calculate_gain(self, parent, left, right):
        """
        Calculates gain from split:
        - Information gain for classification
        - Reduction in MSE for regression
        """
        if self.task == "classification":
            return self._information_gain(parent, left, right)
        return self._mse_reduction(parent, left, right)

    def _information_gain(self, parent, left, right):
        """
        Calculates information gain using entropy.
        """
        def entropy(y):
            counts = Counter(y)
            probs = [count/len(y) for count in counts.values()]
            return -sum(p * math.log2(p) for p in probs)

        n = len(parent)
        entropy_parent = entropy(parent)
        entropy_children = (len(left)/n * entropy(left) + 
                          len(right)/n * entropy(right))
        return entropy_parent - entropy_children

    def _mse_reduction(self, parent, left, right):
        """
        Calculates reduction in MSE.
        """
        def mse(y):
            mean = sum(y)/len(y)
            return sum((val - mean)**2 for val in y)/len(y)

        n = len(parent)
        mse_parent = mse(parent)
        mse_children = (len(left)/n * mse(left) + 
                       len(right)/n * mse(right))
        return mse_parent - mse_children

    def _split_data(self, X, feature_idx, threshold):
        """
        Splits data based on feature and threshold.
        """
        left_idxs = [i for i, row in enumerate(X) if row[feature_idx] <= threshold]
        right_idxs = [i for i, row in enumerate(X) if row[feature_idx] > threshold]
        return left_idxs, right_idxs

    def _is_pure(self, y):
        """
        Checks if node is pure (all same class/value).
        """
        return len(set(y)) == 1

    def _leaf_value(self, y):
        """
        Determines prediction value for leaf node:
        - Most common class for classification
        - Mean value for regression
        """
        if self.task == "classification":
            return max(set(y), key=y.count)
        return sum(y)/len(y)

    def predict(self, X):
        """
        Makes predictions using trained tree.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
            
        return MLArray([self._traverse_tree(x, self.root) for x in X.data])

    def _traverse_tree(self, x, node):
        """
        Traverses tree to make prediction for single sample.
        """
        if node.value is not None:  # Leaf node
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def decision_tree_example():
    """
    Example of using DecisionTree for both classification and regression.
    """
    # Classification example - Simple XOR-like problem
    print("=== Classification Example ===")
    
    # Create synthetic classification data
    X_cls = MLArray([
        [0, 0],  # Class 0
        [0, 1],  # Class 1
        [1, 0],  # Class 1
        [1, 1],  # Class 0
        [0.1, 0.1],  # Class 0
        [0.9, 0.9],  # Class 0
        [0.1, 0.9],  # Class 1
        [0.9, 0.1],  # Class 1
    ])
    
    y_cls = MLArray([0, 1, 1, 0, 0, 0, 1, 1])
    
    # Create and train classification tree
    clf_tree = DecisionTree(
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        task="classification"
    )
    
    print("Training classification tree...")
    clf_tree.fit(X_cls, y_cls)
    
    # Make predictions
    y_pred = clf_tree.predict(X_cls)
    print("\nClassification Predictions:")
    print("True labels:", y_cls.to_list())
    print("Predictions:", y_pred.to_list())
    
    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(y_cls.data, y_pred.data) if true == pred) / len(y_cls.data)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Regression example - Simple quadratic relationship
    print("\n=== Regression Example ===")
    
    # Create synthetic regression data
    X_reg = MLArray([[x/10] for x in range(-10, 11)])  # Values from -1 to 1
    y_reg = MLArray([x[0]**2 for x in X_reg.data])  # Quadratic relationship: y = x²
    
    # Add some noise
    y_reg = y_reg + MLArray([random.uniform(-0.1, 0.1) for _ in range(len(y_reg.data))])
    
    # Create and train regression tree
    reg_tree = DecisionTree(
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        task="regression"
    )
    
    print("Training regression tree...")
    reg_tree.fit(X_reg, y_reg)
    
    # Make predictions
    y_pred = reg_tree.predict(X_reg)
    
    # Calculate MSE
    mse = sum((pred - true)**2 for pred, true in zip(y_pred.data, y_reg.data)) / len(y_reg.data)
    print(f"\nMean Squared Error: {mse}")
    
    print("\nSample regression predictions:")
    for x, y_true, y_p in zip(X_reg.data[:5], y_reg.data[:5], y_pred.data[:5]):
        print(f"x={x[0]}, True={y_true}, Predicted={y_p}")

# Run example
if __name__ == "__main__":
    decision_tree_example()
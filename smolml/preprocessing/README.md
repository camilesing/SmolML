# SmolML - Preprocessing: Make your data meaningful

Before we feed our precious data into many machine learning algorithms, there's often a crucial preprocessing step: **Feature Scaling**. Why? Because algorithms can be quite sensitive to the *scale* or *range* of our input features!

## Why Bother Scaling? Let's Talk Numbers!

Imagine you have a dataset for predicting house prices with features like:
* `size_sqft`: ranging from 500 to 5000 sq ft
* `num_bedrooms`: ranging from 1 to 5
* `distance_to_school_km`: ranging from 0.1 to 10 km

Now, consider algorithms that use distances (like K-Means) or rely on gradient descent (like Linear Regression or Neural Networks).
* **Distance-Based Algorithms:** If you calculate the distance between two houses, a difference of 1000 sq ft will numerically dwarf a difference of 2 bedrooms, just because the *numbers* are bigger. The algorithm might mistakenly think `size_sqft` is vastly more important, solely due to its larger range.
* **Gradient-Based Algorithms:** Features with vastly different scales can cause the optimization process (finding the best model weights) to be slow and unstable. Think of trying to find the bottom of a valley where one side is incredibly steep (large-range feature) and the other is very gentle (small-range feature) – it's tricky!

**The Goal:** Feature scaling brings all features onto a similar numerical playing field. This prevents features with larger values from dominating the learning process just because of their scale, often leading to faster training convergence and sometimes even better model performance.

SmolML provides two common scalers, built using our `MLArray`.

## `StandardScaler`: Zero Mean, Unit Variance

This is one of the most popular scaling techniques, often called **Z-score normalization**.

**The Concept:** It transforms the data for each feature so that it has:
* A **mean ($\mu$) of 0**.
* A **standard deviation ($\sigma$) of 1**.

**How it Works:** For each value $x$ in a feature, it applies the formula:
$$ z = \frac{x - \mu}{\sigma} $$
* **Subtracting the mean ($\mu$)**: This centers the data for that feature around zero.
* **Dividing by the standard deviation ($\sigma$)**: This scales the data so that it has a standard deviation of 1, meaning the "spread" of the data becomes consistent across features.

**Implementation (`StandardScaler` class in `scalers.py`):**

It's a two-step process:

1.  **`fit(self, X)`:** This method learns the necessary parameters *from your training data*.
    * It calculates the mean (`self.mean`) and standard deviation (`self.std`) for *each feature column* (using `X.mean(axis=0)` and `X.std(axis=0)` from our `MLArray`).
    * It stores these calculated `mean` and `std` values within the scaler object.
    * *Heads up:* It also includes logic to handle cases where a feature has zero standard deviation (i.e., all values are the same). In this case, it sets the standard deviation to 1 to avoid division by zero during the transform step.

2.  **`transform(self, X)`:** This method applies the scaling using the *previously learned* parameters.
    * It takes any dataset `X` (could be your training data again, or new test/prediction data) and applies the Z-score formula: `(X - self.mean) / self.std`.
    * The result is your scaled data, ready for your model!

## `MinMaxScaler`: Squeezing into [0, 1]

Another common technique, especially useful when you want features bounded within a specific range.

**The Concept:** It transforms the data for each feature so that all values fall neatly within the range **[0, 1]**.

**How it Works:** For each value $x$ in a feature, it applies the formula:
$$ x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} $$
* **Subtracting the minimum ($x_{min}$)**: This shifts the data so the minimum value becomes 0.
* **Dividing by the range ($x_{max} - x_{min}$)**: This scales the data proportionally so that the maximum value becomes 1. All other values fall somewhere in between.

**Implementation (`MinMaxScaler` class in `scalers.py`):**

Similar two-step process:

1.  **`fit(self, X)`:** Learns the parameters from the training data.
    * It finds the minimum (`self.min`) and maximum (`self.max`) value for *each feature column* (using `X.min(axis=0)` and `X.max(axis=0)` from `MLArray`).
    * It stores these `min` and `max` values.

2.  **`transform(self, X)`:** Applies the scaling using the learned parameters.
    * It takes any dataset `X` and applies the Min-Max formula: `(X - self.min) / (self.max - self.min)`.
    * Voila! Your data is now scaled between 0 and 1.

## The Golden Rule: Fit on Train, Transform Train & Test!

This is super important!

* You should **only** call the `fit()` or `fit_transform()` method on your **training data**. The scaler needs to learn the mean/std or min/max *only* from the data your model will train on.
* You then use the *same fitted scaler* (with the learned parameters) to call `transform()` on your **training data** AND your **test/validation/new prediction data**.

Why? You want to apply the *exact same transformation* to all your data, based only on what the model learned during training. Fitting the scaler on the test data would be a form of "data leakage" – letting your preprocessing step peek at data it shouldn't see yet!

**Convenience:** Both scalers have a `fit_transform(self, X)` method that simply calls `fit(X)` followed by `transform(X)` on the same data – perfect for applying scaling to your training set in one go.

## Example Usage

Let's scale some simple data using `StandardScaler`:

```python
from smolml.preprocessing import StandardScaler, MinMaxScaler
from smolml.core.ml_array import MLArray

# Sample data (e.g., 3 samples, 2 features with different scales)
X_train_data = [[10, 100],
                [20, 150],
                [30, 120]]

# New data to predict on later (must be scaled the same way!)
X_new_data = [[15, 110],
              [25, 160]]

# Convert to MLArray
X_train = MLArray(X_train_data)
X_new = MLArray(X_new_data)

# --- Using StandardScaler ---
print("--- Using StandardScaler ---")
scaler_std = StandardScaler()

# Fit ONCE on training data, then transform it
X_train_scaled_std = scaler_std.fit_transform(X_train)

# Transform the NEW data using the SAME scaler (already fitted)
X_new_scaled_std = scaler_std.transform(X_new)

print("Original Training Data:\n", X_train)
print("Scaled Training Data (StandardScaler):\n", X_train_scaled_std)
print("\nOriginal New Data:\n", X_new)
print("Scaled New Data (StandardScaler):\n", X_new_scaled_std)
print(f"Learned Mean: {scaler_std.mean}")
print(f"Learned Std Dev: {scaler_std.std}")


# --- Using MinMaxScaler ---
print("\n--- Using MinMaxScaler ---")
scaler_minmax = MinMaxScaler()

# Fit ONCE on training data, then transform it
X_train_scaled_mm = scaler_minmax.fit_transform(X_train)

# Transform the NEW data using the SAME scaler (already fitted)
X_new_scaled_mm = scaler_minmax.transform(X_new)

print("Original Training Data:\n", X_train)
print("Scaled Training Data (MinMaxScaler):\n", X_train_scaled_mm)
print("\nOriginal New Data:\n", X_new)
print("Scaled New Data (MinMaxScaler):\n", X_new_scaled_mm)
print(f"Learned Min: {scaler_minmax.min}")
print(f"Learned Max: {scaler_minmax.max}")
```

## Scaling Up!

Feature scaling is a fundamental step in the machine learning pipeline for many algorithms. By bringing features onto a common scale using techniques like standardization (``StandardScaler``) or normalization (``MinMaxScaler``), you can often improve your model's convergence speed and performance. SmolML provides these essential tools, leveraging the computational power of MLArray to make preprocessing straightforward! Don't forget the golden rule: fit only on train, transform everything!
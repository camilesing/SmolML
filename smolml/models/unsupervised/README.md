# SmolML - K-Means: Finding Groups in Your Data!

So far, we've looked at models that learn from labeled data (**supervised learning**): predicting house prices (*regression*) or classifying images (*classification*). But what if you just have a big pile of data and no labels? How can you find interesting structures or groups within it? Welcome to **Unsupervised Learning**, and one of its most popular tools: **K-Means Clustering**!

Imagine you have data points scattered on a graph. K-Means tries to automatically group these points into a specified number (`k`) of clusters, without any prior knowledge of what those groups should be. It's like trying to find natural groupings of customers based on their purchasing behavior, or grouping stars based on their brightness and temperature.

This part of SmolML implements the K-Means algorithm using our trusty `MLArray` class. Let's see how it works!

## The K-Means Algorithm: A Cluster Dance!

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d85e199-e5d1-4ff0-a70b-e1d8ab970e13" width="600">
</div>

K-Means aims to partition your data into `k` distinct, non-overlapping clusters. It does this by finding `k` central points, called **centroids**, and assigning each data point to the nearest centroid. The core idea is an iterative process, a sort of "dance" between assigning points and updating the centers:

1.  **Initialization - Pick Starting Points (`_initialize_centroids`):**
    * First, we need to decide how many clusters (`k`, or `n_clusters` in our code) we want to find. This is something *you* tell the algorithm.
    * Then, we need initial guesses for the location of the `k` cluster centroids. A common way (and the one used here) is to simply pick `k` random data points from your dataset and call them the initial centroids. Think of this as randomly dropping `k` pins on your data map.
    * **Code Connection:** The `_initialize_centroids` method handles this, randomly sampling `n_clusters` points from the input `X_train` to set the initial `self.centroids`. It also initializes `self.centroid_history` to track how these centers move.

2.  **Assignment Step - Find Your Closest Center (`_assign_clusters`):**
    * Now, for *every* data point, calculate its distance to *each* of the `k` centroids. The most common distance measure is the good old **Euclidean distance** (the straight-line distance).
    * Assign each data point to the cluster whose centroid it is closest to. Every point now belongs to one of the `k` clusters.
    * **Code Connection:** The `_compute_distances` method calculates all these distances efficiently. It cleverly uses `MLArray`'s broadcasting and vectorized operations (`reshape`, subtraction, element-wise multiplication, `sum`, `sqrt`) to avoid slow Python loops. The `_assign_clusters` method then iterates through the resulting distance matrix for each point, finds the index (`cluster_idx`) of the minimum distance, and stores these assignments in `self.labels_`.

3.  **Update Step - Move the Center (`_update_centroids`):**
    * The initial centroids were just random guesses. Now that we have points assigned to clusters, we can calculate *better* centroid locations.
    * For each cluster, find the new centroid by calculating the **mean** (average position) of all the data points assigned to that cluster in the previous step. Imagine finding the "center of gravity" for all points in a cluster – that's the new centroid position.
    * **Code Connection:** The `_update_centroids` method does this. It groups points based on `self.labels_`, calculates the mean for each dimension within each group, and updates `self.centroids`. It also handles cases where a cluster might become empty (in which case the centroid stays put).

4.  **Repeat!**
    * Now that the centroids have moved, the distances have changed! Go back to the **Assignment Step (2)** and re-assign all data points to their *new* nearest centroid.
    * Then, go back to the **Update Step (3)** and recalculate the centroid positions based on the new assignments.
    * Keep repeating this assign-and-update dance.

**When Does the Dance Stop? (Convergence)**

The algorithm keeps iterating until one of these happens:
* **Centroids Settle Down:** The centroids stop moving significantly between iterations. The total distance the centroids moved is less than a small threshold (`tol`). (Checked within `_update_centroids`).
* **Maximum Iterations Reached:** We hit the predefined maximum number of iterations (`max_iters`) to prevent the algorithm from running forever if it doesn't converge quickly.

**The Result:**

Once the algorithm stops (converges), you have:
* `self.centroids`: The final positions of the `k` cluster centers.
* `self.labels_`: An array indicating which cluster (0 to k-1) each of your input data points belongs to.

## Implementation in SmolML (`KMeans` class)

The `KMeans` class in `kmeans.py` wraps this entire process:
* **`__init__(self, n_clusters, max_iters, tol)`:** You initialize the model by telling it how many clusters to find (`n_clusters`), the maximum iterations (`max_iters`), and the convergence tolerance (`tol`).
* **`fit(self, X_train)`:** This is the main training method. It takes your data (`X_train`, expected as an `MLArray` or list-of-lists) and runs the iterative assign-and-update loop described above, calling the internal methods (`_initialize_centroids`, `_compute_distances`, `_assign_clusters`, `_update_centroids`) until convergence. It stores the final centroids and labels internally.
* **`predict(self, X)`:** After fitting, you can use this method to assign *new* data points (`X`) to the nearest learned centroid (`self.centroids`).
* **`fit_predict(self, X_train)`:** A handy shortcut that calls `fit` and then immediately `predict` on the same data, returning the cluster labels for the training data.

## Example Usage

Let's find 3 clusters in some simple 2D data:

```python
from smolml.cluster import KMeans
from smolml.core.ml_array import MLArray
import random

# Generate some synthetic 2D data around 3 centers
def generate_data(n_samples, centers):
    data = []
    for _ in range(n_samples):
        center = random.choice(centers)
        # Add some random noise around the center
        point = [center[0] + random.gauss(0, 0.5),
                 center[1] + random.gauss(0, 0.5)]
        data.append(point)
    return data

centers = [[2, 2], [8, 3], [5, 8]]
X_data = generate_data(150, centers)

# Convert to MLArray
X = MLArray(X_data)

# Initialize and fit K-Means
k = 3 # We want to find 3 clusters
kmeans = KMeans(n_clusters=k, max_iters=100, tol=1e-4)

print("Fitting K-Means...")
kmeans.fit(X)

# Get the results
final_centroids = kmeans.centroids
cluster_labels = kmeans.labels_

print("\nK-Means fitting complete!")
print(f"Final Centroid positions:\n{final_centroids}")
# print(f"Cluster labels for first 10 points: {cluster_labels.to_list()[:10]}")
print(f"Number of points in each cluster:")
labels_list = cluster_labels.to_list()
for i in range(k):
    print(f"  Cluster {i}: {labels_list.count(i)} points")

# You could now use these labels or centroids for further analysis or visualization!
# For example, predict the cluster for a new point:
new_point = MLArray([[6, 6]])
predicted_cluster = kmeans.predict(new_point)
print(f"\nNew point {new_point.to_list()} assigned to cluster: {predicted_cluster.to_list()[0]}")
```

> (Note: Because K-Means starts with random initial centroids, you might get slightly different clustering results each time you run it. Running it multiple times and choosing the best result based on some metric is a common practice, though not implemented here.)

## End of the dance

K-Means is a foundational unsupervised algorithm, fantastic for exploring unlabeled data and discovering potential groupings. It's intuitive, relatively simple to implement (especially with tools like our `MLArray` for efficient math!), and often provides valuable insights into the hidden structure of your data. It's a great first step into the world of unsupervised learning!

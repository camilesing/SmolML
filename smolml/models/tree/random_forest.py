from smolml.core.ml_array import MLArray
import random
import smolml.utils.memory as memory
from collections import Counter
from smolml.models.tree.decision_tree import DecisionTree

"""
/////////////////////
/// RANDOM FOREST ///
/////////////////////
"""


class RandomForest:
    """
    Random Forest implementation supporting both classification and regression.
    Uses bagging (bootstrap aggregating) and random feature selection.
    """

    def __init__(self, n_trees=100, max_features=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 task="classification"):
        """
    方法用于初始化随机森林模型，设置其关键参数：
        n_trees: 森林中决策树的数量，默认为100
        max_features: 每次分裂考虑的最大特征数，若为None则根据任务类型自动设定
        max_depth: 树的最大深度
        min_samples_split: 节点分裂所需的最小样本数
        min_samples_leaf: 叶子节点所需的最小样本数
        bootstrap: 是否使用自助法（有放回抽样）
        task: 任务类型，可为分类或回归
        self.trees: 用于存储生成的树对象的列表
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task if task in ["classification", "regression"] else None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
         实现自助采样（Bootstrap Sampling），即从数据集中有放回地随机抽取样本。具体功能如下：
            n_samples = len(X)：获取输入数据 X 的样本数量。
            indices = [...]：生成 n_samples 个随机索引，范围在 0 到 n_samples - 1，允许重复。
            bootstrap_X 和 bootstrap_y：根据随机索引从原始数据中抽取样本构建新的数据集。
            返回新抽取的样本对 (bootstrap_X, bootstrap_y)。
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]

        bootstrap_X = [X[i] for i in indices]
        bootstrap_y = [y[i] for i in indices]

        return bootstrap_X, bootstrap_y

    def _get_max_features(self, n_features):
        """
        用于确定随机森林中每个节点分裂时考虑的最大特征数量：
            如果 self.max_features 未指定（为 None）：
                分类任务：使用 sqrt(n_features)，至少取 1；
                回归任务：使用 n_features // 3，同样至少取 1。
            如果指定了 self.max_features，则取其与特征总数之间的较小值。
        """
        if self.max_features is None:
            # Use sqrt(n_features) for classification, n_features/3 for regression
            if self.task == "classification":
                return max(1, int(n_features ** 0.5))
            else:
                return max(1, n_features // 3)
        return min(self.max_features, n_features)

    def fit(self, X, y):
        """
         用于训练随机森林模型，其核心逻辑如下：
            1. 数据预处理：将输入 X 和 y 转换为 MLArray 格式，提取原始数据。
            2. 特征采样设置：根据特征总数计算每次分裂时随机选取的特征数 max_features。
            3. 构建多棵决策树：
                - 对每棵树，根据 bootstrap 参数决定是否使用自助采样生成训练集。
                - 创建一个 DecisionTree 实例。
            4. 修改分裂策略：重写树的 _find_best_split 方法，在每次分裂时只考虑 max_features 个随机选取的特征，以提升泛化能力。
            5. 训练与保存树模型：用修改后的分裂策略训练树模型，并将其加入森林中。
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)

        X_data, y_data = X.data, y.data
        n_features = len(X_data[0])
        max_features = self._get_max_features(n_features)

        # Create and train each tree
        for _ in range(self.n_trees):
            # Create bootstrap sample if enabled
            if self.bootstrap:
                sample_X, sample_y = self._bootstrap_sample(X_data, y_data)
            else:
                sample_X, sample_y = X_data, y_data

            # Create and train tree with random feature selection
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                task=self.task
            )

            # Modify tree's _find_best_split to use random feature subset
            original_find_best_split = tree._find_best_split

            def random_feature_find_best_split(self, X, y):
                n_features = len(X[0])
                feature_indices = random.sample(range(n_features), max_features)

                best_gain = -float('inf')
                best_feature = None
                best_threshold = None

                for feature_idx in feature_indices:
                    thresholds = sorted(set(row[feature_idx] for row in X))

                    for threshold in thresholds:
                        left_idxs, right_idxs = tree._split_data(X, feature_idx, threshold)

                        if len(left_idxs) < tree.min_samples_leaf or len(right_idxs) < tree.min_samples_leaf:
                            continue

                        left_y = [y[i] for i in left_idxs]
                        right_y = [y[i] for i in right_idxs]

                        gain = tree._calculate_gain(y, left_y, right_y)

                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold

                return best_feature, best_threshold

            # Replace tree's _find_best_split with our random feature version
            tree._find_best_split = random_feature_find_best_split.__get__(tree)

            # Train the tree
            tree.fit(MLArray(sample_X), MLArray(sample_y))
            self.trees.append(tree)

    def predict(self, X):
        """
   用于在随机森林模型中对输入数据 X 进行预测，根据任务类型（分类或回归）聚合所有决策树的预测结果：
        1. 输入处理：若 X 不是 MLArray 类型，则将其转换为该类型；
        2. 树预测收集：对随机森林中的每棵树进行预测，得到每个样本的预测结果；
        3. 结果聚合：
            - 若为分类任务（classification），对每个样本使用多数投票法决定预测类别；
            - 若为回归任务（regression），对每个样本取所有树预测结果的平均值；
        4. 输出结果：将最终预测结果封装为 MLArray 返回。
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)

        # Get predictions from all trees
        tree_predictions = [tree.predict(X) for tree in self.trees]

        # Aggregate predictions based on task
        if self.task == "classification":
            final_predictions = []
            for i in range(len(X)):
                # Get predictions for this sample from all trees
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Take majority vote
                vote = Counter(sample_predictions).most_common(1)[0][0]
                final_predictions.append(vote)
        elif self.task == "regression.py":
            final_predictions = []
            for i in range(len(X)):
                # Get predictions for this sample from all trees
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Take mean
                mean = sum(sample_predictions) / len(sample_predictions)
                final_predictions.append(mean)
        else:
            raise Exception(f"Task in Random Forest not assigned to either 'classification' or 'regression'")

        return MLArray(final_predictions)

    def __repr__(self):
        """
        Returns string representation of random forest with structure and memory information.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        header = f"Random Forest ({self.task.title()})"
        separator = "=" * terminal_width

        # Get size information
        size_info = memory.calculate_random_forest_size(self)

        # Model parameters
        params = [
            f"Number of Trees: {self.n_trees}",
            f"Max Features per Split: {self.max_features if self.max_features else 'auto'}",
            f"Bootstrap Sampling: {self.bootstrap}",
            f"Max Depth: {self.max_depth if self.max_depth else 'None'}",
            f"Min Samples Split: {self.min_samples_split}",
            f"Min Samples Leaf: {self.min_samples_leaf}",
            f"Task: {self.task}"
        ]

        # Forest structure information
        if self.trees:
            structure_info = [
                "Forest Structure:",
                f"  Trees Built: {len(self.trees)}",
                f"  Average Tree Depth: {size_info['forest_stats']['avg_tree_depth']:.1f}",
                f"  Average Nodes per Tree: {size_info['forest_stats']['avg_tree_nodes']:.1f}"
            ]

            # Add sample stats from first tree if available
            if self.trees:
                first_tree_size = size_info['trees']['individual'][0]
                structure_info.extend([
                    "\nSample Tree Structure (First Tree):",
                    f"  Internal Nodes: {first_tree_size['tree_structure']['internal_nodes']}",
                    f"  Leaf Nodes: {first_tree_size['tree_structure']['leaf_nodes']}",
                    f"  Max Depth: {first_tree_size['tree_structure']['max_depth']}"
                ])
        else:
            structure_info = ["Forest not yet trained"]

        # Memory usage
        memory_info = ["Memory Usage:"]
        memory_info.append(f"  Base Forest: {memory.format_size(size_info['base_size'])}")
        if self.trees:
            memory_info.extend([
                f"  All Trees: {memory.format_size(size_info['trees']['total'])}",
                f"  Average per Tree: {memory.format_size(size_info['trees']['total'] / len(self.trees))}"
            ])
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")

        return (
                f"\n{header}\n{separator}\n\n"
                + "Parameters:\n" + "\n".join(f"  {param}" for param in params)
                + "\n\n" + "\n".join(structure_info)
                + "\n\n" + "\n".join(memory_info)
                + f"\n{separator}\n"
        )


def example():
    # 定义了一个包含5个样本、4个特征的训练数据 X_data
    X_data = [
        [5.1, 3.5, 1.4, 0.2],  # 样本 0 ➜ 类别 0
        [4.9, 3.0, 1.4, 0.2],  # 样本 1 ➜ 类别 0
        [6.7, 3.1, 4.4, 1.4],  # 样本 2 ➜ 类别 1
        [6.0, 2.9, 4.5, 1.5],  # 样本 3 ➜ 类别 1
        [5.8, 2.7, 5.1, 1.9]  # 样本 4 ➜ 类别 2
    ]
    # 真实的标签 ，也就是我们的预期
    y_data = [0, 0, 1, 1, 2]

    # 将数据转换为 MLArray 类型（可能是自定义数组类型）；
    X = MLArray(X_data)
    y = MLArray(y_data)

    # 使用 DecisionTree 训练一个最大深度为3的决策树模型，并对训练数据进行预测；
    print("--- 训练决策树 ---")
    dt = DecisionTree(max_depth=3, task="classification")
    dt.fit(X, y)
    print(dt)  # Shows structure and stats
    dt_pred = dt.predict(X)
    print(f" 决策树模型在训练数据上的预测结果: {dt_pred.to_list()}")

    # --- Using a Random Forest ---
    print("\n--- 训练随机森林---")
    # Build a forest of 10 trees
    rf = RandomForest(n_trees=10, max_depth=3, task="classification")
    rf.fit(X, y)
    print(rf)  # Shows forest stats
    rf_pred = rf.predict(X)
    print(f"随机森林模型在训练数据上的预测结果: {rf_pred.to_list()}")

    # Predict on new data
    X_new = MLArray([[6.0, 3.0, 4.8, 1.8], [5.0, 3.4, 1.6, 0.4]])
    rf_new_pred = rf.predict(X_new)
    print(f"\n随机森林模型在新数据上的情况：{X_new.to_list()}: {rf_new_pred.to_list()}")


if __name__ == "__main__":
    example()

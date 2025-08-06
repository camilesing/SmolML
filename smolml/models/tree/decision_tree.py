from smolml.core.ml_array import MLArray
import smolml.utils.memory as memory
from collections import Counter
import math

"""
/////////////////////
/// DECISION TREE ///
/////////////////////
"""


class DecisionNode:
    """
    表示决策树中的节点，包含分裂规则或预测值。
    """

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold  # Value to split feature on
        self.left = left  # Left subtree (feature <= threshold)
        self.right = right  # Right subtree (feature > threshold)
        self.value = value  # Prediction value (for leaf nodes)

    def __repr__(self):
        if self.value is not None:
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature_idx}, threshold={self.threshold:.4f})"


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
        填充数据集开始构建
            数据类型转换：若输入 X 或 y 不是 MLArray 类型，则将其转换为 MLArray；
            类别数统计（仅分类任务）：
                若任务为分类（self.task == "classification"），
                则计算目标变量 y 的类别数；
            构建决策树：调用 _grow_tree 方法，
                使用输入特征 X.data 与标签 y.data 构建决策树的根节点
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)

        self.n_classes = len(set(y.flatten(y.data))) if self.task == "classification" else None
        self.root = self._grow_tree(X.data, y.data)

    def _grow_tree(self, X, y, depth=0):
        """
        递归构建树结构
            1. 判断是否停止分裂：若达到最大深度、样本数不足分裂要求或标签已纯，
                则生成叶节点
            2. 寻找最佳分裂特征与阈值，若无有效分裂则返回叶节点
            3. 划分数据集为左右子集，并检查是否满足最小叶节点样本数
            4. 递归构建左右子树，最终返回当前决策节点
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
        寻找最优特征和阈值进行分裂
            1. 遍历每个特征及其所有唯一取值作为候选阈值；
            2. 对每个候选阈值，将数据划分为左右两子集；
            3. 若任一子集样本数小于 min_samples_leaf 则跳过该阈值；
            4. 计算划分后的信息增益；
            5. 保留增益最大的特征和对应阈值；
            6. 最终返回最优特征索引和阈值。
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
       根据指定特征列和阈值将数据集划分为左右两个子集，返回满足条件的样本索引列表
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
         用于计算决策树中的信息增益，衡量一个特征划分数据集前后不确定性减少的程度。
          1. entropy(y)：计算给定标签集合 y 的熵，表示数据的纯度
          2.entropy_parent：计算父节点的熵
          3. entropy_children：计算划分后左右子节点的加权熵
          4. 返回 entropy_parent - entropy_children：即信息增益
        """
        if self.task == "classification":
            return self._information_gain(parent, left, right)
        return self._mse_reduction(parent, left, right)

    def _information_gain(self, parent, left, right):
        """
        用于计算信息增益，基于熵来评估数据划分的好坏：
            1. entropy(y)：计算标签 y 的熵，反映数据纯度
            2. 计算父节点的熵 entropy_parent
            3. 计算划分后左右子节点的加权熵 entropy_children
            4. 返回信息增益：entropy_parent - entropy_children，
                表示划分前后熵的减少量。
        """

        def entropy(y):
            counts = Counter(y)
            probs = [count / len(y) for count in counts.values()]
            return -sum(p * math.log2(p) for p in probs)

        n = len(parent)
        entropy_parent = entropy(parent)
        entropy_children = (len(left) / n * entropy(left) +
                            len(right) / n * entropy(right))
        return entropy_parent - entropy_children

    def _mse_reduction(self, parent, left, right):
        """
         计算决策树划分前后均方误差（MSE）的减少量，用于评估划分质量。具体步骤如下：
            1. 定义 mse 函数，计算给定标签列表的均方误差；
            2. 计算父节点的 MSE；
            3. 计算左右子节点的加权平均 MSE；
            4. 返回 MSE 的减少量（父节点 MSE 减去子节点加权 MSE）。
        """

        def mse(y):
            mean = sum(y) / len(y)
            return sum((val - mean) ** 2 for val in y) / len(y)

        n = len(parent)
        mse_parent = mse(parent)
        mse_children = (len(left) / n * mse(left) +
                        len(right) / n * mse(right))
        return mse_parent - mse_children

    def _split_data(self, X, feature_idx, threshold):
        """
         根据指定特征列和阈值将数据集划分为左右两部分：
            - left_idxs：保存满足条件（特征值 ≤ 阈值）的样本索引
            - right_idxs：保存不满足上述条件（特征值 > 阈值）的样本索引
                返回左右子集的索引列表，用于构建决策树的分支。
        """
        left_idxs = [i for i, row in enumerate(X) if row[feature_idx] <= threshold]
        right_idxs = [i for i, row in enumerate(X) if row[feature_idx] > threshold]
        return left_idxs, right_idxs

    def _is_pure(self, y):
        """
         判断当前节点是否为纯节点（即所有样本的目标值相同）。
            - y 是当前节点的样本目标值集合；
            - set(y) 会将重复的目标值去重；
            - 若去重后的长度为1，说明所有目标值相同，返回 True，表示该节点是纯节点
        """
        return len(set(y)) == 1

    def _leaf_value(self, y):
        """
        用于确定决策树叶子节点的预测值：
            - 若任务为分类（classification），则返回类别中出现次数最多的类；
            - 若任务为回归（regression），则返回目标值的平均值
        """
        if self.task == "classification":
            return max(set(y), key=y.count)
        return sum(y) / len(y)

    def predict(self, X):
        """
        用于对新数据 X 进行预测：
            - 如果 X 不是 MLArray 类型，则将其转换为 MLArray；
            - 对 X 中的每个样本 x，调用 _traverse_tree 方法从根节点 self.root 开始遍历决策树；
            - 将每个样本的预测结果收集到一个新的 MLArray 中并返回
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)

        return MLArray([self._traverse_tree(x, self.root) for x in X.data])

    def _traverse_tree(self, x, node):
        """
        用于在决策树中对新数据 x 进行预测：
            1. 如果当前节点是叶节点（node.value 不为 None），则返回该节点的预测值；
            2. 否则根据当前节点的特征索引和阈值，递归进入左子树或右子树继续预测
        """
        if node.value is not None:  # Leaf node
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def __repr__(self):
        """
        Returns string representation of decision tree with structure and memory information.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        header = f"Decision Tree ({self.task.title()})"
        separator = "=" * terminal_width

        # Get size information
        size_info = memory.calculate_decision_tree_size(self)

        # Model parameters
        params = [
            f"Max Depth: {self.max_depth if self.max_depth is not None else 'None'}",
            f"Min Samples Split: {self.min_samples_split}",
            f"Min Samples Leaf: {self.min_samples_leaf}",
            f"Task: {self.task}"
        ]

        # Tree structure information
        if self.root:
            structure_info = [
                "Tree Structure:",
                f"  Internal Nodes: {size_info['tree_structure']['internal_nodes']}",
                f"  Leaf Nodes: {size_info['tree_structure']['leaf_nodes']}",
                f"  Max Depth: {size_info['tree_structure']['max_depth']}",
                f"  Total Nodes: {size_info['tree_structure']['internal_nodes'] + size_info['tree_structure']['leaf_nodes']}"
            ]
        else:
            structure_info = ["Tree not yet trained"]

        # Memory usage
        memory_info = ["Memory Usage:"]
        memory_info.append(f"  Base Tree: {memory.format_size(size_info['base_size'])}")
        if self.root:
            memory_info.append(f"  Tree Structure: {memory.format_size(size_info['tree_structure']['total'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")

        return (
                f"\n{header}\n{separator}\n\n"
                + "Parameters:\n" + "\n".join(f"  {param}" for param in params)
                + "\n\n" + "\n".join(structure_info)
                + "\n\n" + "\n".join(memory_info)
                + f"\n{separator}\n"
        )
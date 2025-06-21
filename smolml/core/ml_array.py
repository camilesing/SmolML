from smolml.core.value import Value
import random
import math

"""
///////////////
/// MLARRAY ///
///////////////
"""


class MLArray:
    """
    一个用于机器学习的多维数组类 MLArray
    """

    """
    ///////////////
    /// General ///
    ///////////////
    """

    def __init__(self, data) -> None:
        """
        初始化时接收数据（标量、列表或嵌套列表），调用 _process_data 处理数据。
        """
        self.data = self._process_data(data)

    def _process_data(self, data):
        """
        递归处理数据，将所有数值封装为 Value 对象，支持自动微分；保留已有 Value 或 MLArray 对象
        """
        if isinstance(data, (int, float)):
            return Value(data)
        elif isinstance(data, list):
            return [self._process_data(item) for item in data]
        elif isinstance(data, (Value, MLArray)):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _infer_shape(self, data):
        """
       递归推断数据形状，返回表示维度的元组，如 () 表示标量，(3,) 表示一维数组，(2,3) 表示二维数组
        """
        if isinstance(data, Value):
            return ()
        elif isinstance(data, list):
            return (len(data),) + self._infer_shape(data[0])
        else:
            return ()

    """
    ///////////////////////////
    /// 标准操作 ///
    /// 各种数学运算符操作，使得该类可以支持类似NumPy数组的元素级运算
    ///////////////////////////
    """

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        return self._element_wise_operation(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._element_wise_operation(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return MLArray(other) - self

    def __mul__(self, other):
        return self._element_wise_operation(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._element_wise_operation(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return MLArray(other) / self

    def __pow__(self, other):
        return self._element_wise_operation(other, lambda x, y: x ** y)

    """
    ///////////////////////////
    /// Advanced Operations ///
    ///////////////////////////
    """

    def __T__(self):
        return self.transpose()

    '''   
    矩阵反转操作。
    什么叫反转？
        a = MLArray([[1, 2], 
             [3, 4]])
        反转后会变成
            [[1, 3],
            [2, 4]]
     常用于满足模型输入要求、提升计算效率等场景。   
    '''

    def transpose(self, axes=None):

        if len(self.shape) <= 1:  # Scalar or 1D array
            return self

        if axes is None:  # If no axes, reverse the current axes
            axes = list(range(len(self.shape)))[::-1]

        new_shape = tuple(self.shape[i] for i in axes)

        def _all_possible_indices(shape):
            """
            根据数组的形状，递归生成所有合法的索引组合，使得我们可以访问数组中每一个元素，并在转置时将它们放到新的位置上。
            """
            if len(shape) == 0:
                yield []
            else:
                for i in range(shape[0]):
                    for rest in _all_possible_indices(shape[1:]):
                        yield [i] + rest

        new_data = self._create_nested_list(new_shape)  # Create empty list with new transposed shape

        for indices in _all_possible_indices(self.shape):  # Add transposed elements
            new_indices = [indices[i] for i in axes]
            value = self._get_item(self.data, indices)
            self._set_item(new_data, new_indices, value)

        return MLArray(new_data)

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        """
        两个 MLArray 对象的矩阵乘法，支持一维和多维数组
        """
        # 若 other 不是 MLArray，则将其转换为 MLArray。
        if not isinstance(other, MLArray):
            other = MLArray(other)

        # 若任一操作数为标量（0维），则直接进行标量乘法
        if len(self.shape) == 0 or len(other.shape) == 0:
            return self * other

        # 若为一维数组，则扩展为二维以适配矩阵乘法规则
        if len(self.shape) == 1:
            a = MLArray([self.data])
        else:
            a = self

        if len(other.shape) == 1:
            b = MLArray([other.data]).transpose()
        else:
            b = other

        # 若为高维数组，则对输入进行重塑为二维矩阵形式
        a = a.reshape(-1, a.shape[-1]) if len(a.shape) > 2 else a
        b = b.reshape(b.shape[0], -1) if len(b.shape) > 2 else b

        # 确保最后两个维度满足矩阵乘法规则（a列 == b行）
        if a.shape[-1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")

        # 使用双重循环计算矩阵乘法结果
        index_b = 0 if len(b.shape) == 1 else 1
        result = self._create_nested_list((a.shape[0], b.shape[index_b]))
        for i in range(a.shape[0]):
            for j in range(b.shape[index_b]):
                result[i][j] = sum(a.data[i][k] * b.data[k][j] for k in range(a.shape[1]))

        # 根据输入维度重塑结果为合适的多维结构
        if len(self.shape) > 2 or len(other.shape) > 2:
            new_shape = self.shape[:-1] + other.shape[1:]
            return MLArray(result).reshape(new_shape)
        else:
            return MLArray(result)

    def sum(self, axis=None):
        """
         递归求和所有元素，若为标量则返回自身
            axis用于表达维度。
                在二维数组中：
                    axis=0 表示 列方向（垂直）
                    axis=1 表示 行方向（水平）
                在三维或更高维数组中：
                    axis=n 表示沿着第 n 个维度进行操作
                不指定则全部计算
         支持负数 axis——自动转换为正数轴索引
         指定 axis 求和：沿指定轴对数组进行逐层求和
         若结果为非列表（如标量），直接封装为 MLArray 返回
        """
        if axis is None:
            def recursive_sum(data):
                if isinstance(data, Value):
                    return data
                elif isinstance(data, list):
                    return sum(recursive_sum(x) for x in data)
                else:
                    return 0

            if len(self.shape) == 0:  # scalar
                return self
            else:
                return MLArray(recursive_sum(self.data))

        if axis < 0:
            axis += len(self.shape)

        if axis < 0 or axis >= len(self.shape):
            raise ValueError(f"Invalid axis {axis} for MLArray with shape {self.shape}")

        def sum_along_axis(data, current_depth):
            if current_depth == axis:
                if isinstance(data[0], list):
                    # Transpose the data at this level and sum
                    transposed = list(zip(*data))
                    return [sum(slice) for slice in transposed]
                else:
                    return sum(data)

            if isinstance(data[0], list):
                return [sum_along_axis(subdata, current_depth + 1) for subdata in data]
            return data

        result = sum_along_axis(self.data, 0)

        if not isinstance(result, list):
            return MLArray(result)

        return MLArray(result)

    def tanh(self):
        """
           计算 MLArray 对象中每个元素的双曲正切（tanh）值。
           支持标量和数组两种情况。
           """
        return self._element_wise_function(lambda val: val.tanh())

    def min(self, axis=None):
        """
        Returns the smallest element in the MLArray
        """
        return self.reduce_operation(min, axis)

    def max(self, axis=None):
        """
        Returns the biggest element in the MLArray
        """
        return self.reduce_operation(max, axis)

    def mean(self, axis=None):
        """
        Compute mean along specified axis or globally if axis=None
        """
        if axis is None:
            flat_data = self.flatten(self.data)
            return MLArray(sum(flat_data)) / len(flat_data)

        return self.reduce_operation(sum, axis=axis) / self.shape[axis]

    def std(self, axis=None):
        """
        计算标准差
        """
        mean = self.mean(axis=axis)

        # 将数据展平并计算每个元素与均值的差的平方；
        # 构造平方差的MLArray并求平均，最后开平方返回结果；
        if axis is None:
            # For global std, calculate flattened differences
            flat_diffs = [(Value(x) - Value(mean.data)) * (Value(x) - Value(mean.data))
                          for x in self.flatten(self.data)]
            squared_diff = MLArray([diff.data for diff in flat_diffs])
            return (squared_diff.sum() / squared_diff.size()).sqrt()

        # 按轴标准差
        # 1. 对均值结果进行广播对齐
        broadcast_shape = list(self.shape)
        broadcast_shape[axis] = 1
        mean_broadcast = mean.reshape(*broadcast_shape)

        # 2. 手动计算差的平方
        diff = self - mean_broadcast
        squared_diff = MLArray([[x * x for x in row] for row in diff.data])

        # 3. 沿指定轴求平均后开平方返回
        return (squared_diff.sum(axis=axis) / self.shape[axis]).sqrt()

    def sqrt(self):
        """
        计算 MLArray 对象中每个元素的平方根，支持标量和数组两种情况
        """
        if len(self.shape) == 0:  # scalar case
            return MLArray(math.sqrt(self.data.data))

        def sqrt_value(x):
            if isinstance(x, (int, float)):
                return math.sqrt(x)
            return Value(math.sqrt(x.data))

        flat_data = [sqrt_value(x) for x in self.flatten(self.data)]
        if len(self.shape) == 0:
            return MLArray(flat_data[0])
        return MLArray(flat_data).reshape(*self.shape)

    def exp(self):
        return self._element_wise_function(lambda val: val.exp())

    def log(self):
        return self._element_wise_function(lambda val: val.log())

    def abs(self):
        return self._element_wise_function(lambda val: abs(val))

    def __len__(self):
        return len(self.data)

    """
    /////////////////////////
    /// Utility Functions ///
    /////////////////////////
    """

    # 根据索引列表获取嵌套数据中的元素
    def _get_item(self, data, indices):
        for idx in indices:
            data = data[idx]
        return data

    # 根据索引列表设置嵌套数据中的值
    def _set_item(self, data, indices, value):
        for idx in indices[:-1]:
            data = data[idx]
        data[indices[-1]] = value

    # 将 MLArray 数据递归转换为普通 Python 列表
    def to_list(self):
        return self._to_list(self.data)

    def _to_list(self, data):
        """
        递归剥离 Value 类型，返回原始数据组成的列表
        """
        if isinstance(data, (Value)):
            return data.data
        elif isinstance(data, list):
            return [self._to_list(item) for item in data]

    def restart(self):
        """
         重置所有 Value 对象的梯度和计算图
        """
        self._restart_data(self.data)
        return self

    def _restart_data(self, data):
        """
        递归遍历并重置每个 Value 对象
        """
        if isinstance(data, Value):
            data.grad = 0
            data.prev = ()
        elif isinstance(data, list):
            for item in data:
                self._restart_data(item)

    def backward(self):
        """
        对所有 Value 执行反向传播
        """
        # Flatten MLArray to get all Value objects
        flat_data = self.flatten(self.data)

        # Find the output Value(assumed to be scalar or we take the sum)
        if len(flat_data) == 1:
            output_value = flat_data[0]
        else:
            output_value = sum(flat_data)

        # Call backward on output Value
        output_value.backward()

    def flatten(self, data):
        """
          递归展平多维结构为一维 Value 列表
        """
        if isinstance(data, Value):
            return [data]
        elif isinstance(data, list):
            return [item for sublist in data for item in self.flatten(sublist)]
        else:
            return []

    def unflatten(self, flat_list, shape):
        """
         将一维列表还原为指定形状的嵌套结构
        """
        if len(shape) == 1:
            return flat_list[:shape[0]]
        else:
            stride = len(flat_list) // shape[0]
            return [self.unflatten(flat_list[i * stride:(i + 1) * stride], shape[1:]) for i in range(shape[0])]

    def _create_nested_list(self, shape):
        """
         创建指定形状的空嵌套列表
        """
        if len(shape) == 1:
            return [None] * shape[0]
        return [self._create_nested_list(shape[1:]) for _ in range(shape[0])]

    def update_values(self, new_data):
        """
         更新 MLArray 中所有 Value 的数据，保持结构不变
        """

        def update_recursive(current_data, new_data):
            if isinstance(current_data, Value):
                current_data.data = new_data.data if isinstance(new_data, Value) else new_data
            elif isinstance(current_data, list):
                for i, (curr, new) in enumerate(zip(current_data, new_data)):
                    update_recursive(curr, new)

        update_recursive(self.data, new_data.data if isinstance(new_data, MLArray) else new_data)
        return self

    @staticmethod
    def ensure_array(*args):
        """
        将输入参数统一转换为MLArray对象，支持int、float、list及numpy数组。
        """

        def _convert_single_arg(arg):
            # If already MLArray, return as is
            if isinstance(arg, MLArray):
                return arg

            # If numpy array, convert to list first 
            if str(type(arg).__module__) == 'numpy':
                arg = arg.tolist()

            # Handle different input types
            if isinstance(arg, (int, float)):
                return MLArray([arg])
            elif isinstance(arg, list):
                # Check if the list contains only numbers
                def is_numeric_list(lst):
                    for item in lst:
                        if isinstance(item, list):
                            if not is_numeric_list(item):
                                return False
                        elif not isinstance(item, (int, float)):
                            return False
                    return True

                if is_numeric_list(arg):
                    return MLArray(arg)
                else:
                    raise TypeError(f"List contains non-numeric values: {type(arg)}")
            else:
                raise TypeError(f"Cannot convert type {type(arg)} to MLArray")

        # Convert each argument
        converted = []
        for i, arg in enumerate(args):
            try:
                converted.append(_convert_single_arg(arg))
            except Exception as e:
                raise TypeError(f"Error converting argument {i}: {str(e)}")

        # Return tuple of converted arrays
        return tuple(converted) if len(converted) > 1 else converted[0]

    @staticmethod
    def _broadcast_shapes(shape1, shape2):
        """
         根据广播规则计算两个shape的广播后形状
         比如: (3,4,5) | (4, 1) -> (3,4,5)
        """
        # Ensure shape1 is the longer shape
        if len(shape2) > len(shape1):
            shape1, shape2 = shape2, shape1

        # Pad the shorter shape with 1s
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2

        result = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == s2:
                result.append(s1)
            elif s1 == 1 or s2 == 1:
                result.append(max(s1, s2))
            else:
                raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
        return tuple(result)

    def _broadcast_and_apply(self, data1, data2, shape1, shape2, target_shape, op):
        """
        递归实现两个数组的广播与操作应用，支持标量与多维数组的广播运算。
        """
        if not shape1 and not shape2:  # Both scalars
            return op(data1, data2)
        elif not shape1:  # data1 is scalar -> Recursive call to apply the operation to scalar data1 and each element of data2
            return [self._broadcast_and_apply(data1, d2, (), shape2[1:], target_shape[1:], op) for d2 in data2]
        elif not shape2:  # data2 is scalar -> Recursive call to apply the operation to scalar data2 and each element of data1
            return [self._broadcast_and_apply(d1, data2, shape1[1:], (), target_shape[1:], op) for d1 in data1]
        else:  # Both arrays
            if len(shape1) > len(shape2):
                # Pad data2 with extra dimensions
                data2 = [data2] * target_shape[0]
                shape2 = (target_shape[0],) + shape2
            elif len(shape2) > len(shape1):
                # Pad data1 with extra dimensions
                data1 = [data1] * target_shape[0]
                shape1 = (target_shape[0],) + shape1

            if shape1[0] == target_shape[0] and shape2[0] == target_shape[
                0]:  # Both first dimensions match the target shape -> Recursive call to apply the operation to each element of data1 and data2
                return [self._broadcast_and_apply(d1, d2, shape1[1:], shape2[1:], target_shape[1:], op) for d1, d2 in
                        zip(data1, data2)]
            elif shape1[
                0] == 1:  # First dimension of shape1 is 1 (broadcasting needed) -> Recursive call to apply the operation to data1 and each element of data2
                return [self._broadcast_and_apply(data1[0], d2, shape1[1:], shape2[1:], target_shape[1:], op) for d2 in
                        data2]
            elif shape2[
                0] == 1:  # First dimension of shape2 is 1 (broadcasting needed) -> Recursive call to apply the operation to data2 and each element of data1
                return [self._broadcast_and_apply(d1, data2[0], shape1[1:], shape2[1:], target_shape[1:], op) for d1 in
                        data1]

    def _element_wise_operation(self, other, op):
        """
        对两个MLArray对象（或一个标量）执行逐元素的操作（如加减乘除），先广播形状再应用操作。
        """
        if isinstance(other, (int, float, Value)):
            other = MLArray(other)

        if not isinstance(other, MLArray):
            raise TypeError(f"Unsupported operand type: {type(other)}")

        target_shape = self._broadcast_shapes(self.shape, other.shape)
        result = self._broadcast_and_apply(self.data, other.data, self.shape, other.shape, target_shape, op)
        return MLArray(result)

    def _element_wise_function(self, fn):
        """
        对数组中的每个元素递归地应用一个函数（如sin、exp等），支持多维结构。
        """
        if len(self.shape) == 0:  # scalar
            return MLArray(fn(self.data))

        def apply_recursive(data):
            if isinstance(data, list):
                return [apply_recursive(d) for d in data]
            return fn(data)

        return MLArray(apply_recursive(self.data))

    def reduce_operation(self, op, axis=None):
        """
        沿指定轴进行归约操作（如sum、max），支持全局或特定轴归约。
        """
        # Case 1: Global reduction (reduce all elements)
        if axis is None:
            return op(self.flatten(self.data), key=lambda x: x)

        # Case 2: Reduce along specific axis
        if not isinstance(axis, int):
            raise TypeError("Axis must be None or an integer")

        # Handle negative axis
        if axis < 0:
            axis += len(self.shape)

        if axis < 0 or axis >= len(self.shape):
            raise ValueError(f"Invalid axis {axis} for MLArray with shape {self.shape}")

        def reduce_recursive(data, current_depth, target_axis, shape):
            # Base case: reached target axis
            if current_depth == target_axis:
                if isinstance(data[0], list):
                    # Transpose the data at this level
                    transposed = list(zip(*data))
                    # Apply reduction to each transposed slice
                    return [op(slice) for slice in transposed]
                else:
                    return op(data)

            # Recursive case: not at target axis yet
            return [reduce_recursive(subarray, current_depth + 1, target_axis, shape[1:])
                    for subarray in data]

        # Get new shape after reduction
        new_shape = list(self.shape)
        new_shape.pop(axis)

        # Perform reduction
        result = reduce_recursive(self.data, 0, axis, self.shape)

        # Handle the case where result is a scalar
        if not new_shape:
            return result

        return MLArray(result)

    def reshape(self, *new_shape):
        """
        改变数组形状，支持自动推导某一维度（用-1表示），确保总元素数量不变。
        """
        # Calculate the total size
        total_size = self.size()

        # Handle -1 in new_shape
        if -1 in new_shape:
            # Calculate the product of all dimensions except -1
            known_dim_product = 1
            unknown_dim_index = new_shape.index(-1)

            for i, dim in enumerate(new_shape):
                if i != unknown_dim_index:
                    known_dim_product *= dim

            # Calculate the missing dimension
            if total_size % known_dim_product != 0:
                raise ValueError(f"Cannot reshape array of size {total_size} into shape {new_shape}")

            missing_dim = total_size // known_dim_product
            new_shape = list(new_shape)
            new_shape[unknown_dim_index] = missing_dim
            new_shape = tuple(new_shape)

        # Calculate the product of new shape dimensions
        new_size = 1
        for dim in new_shape:
            new_size *= dim

        # Check if the new shape is valid
        if new_size != total_size:
            raise ValueError(f"Cannot reshape array of size {total_size} into shape {new_shape}")

        # Flatten the array and create new structure
        flat_data = self.flatten(self.data)
        new_data = self.unflatten(flat_data, new_shape)
        return MLArray(new_data)

    def size(self):
        """
        Returns the number of elements that compose the MLArray.
        """
        return len(self.flatten(self.data))

    def grad(self):
        """
        递归提取 MLArray 中所有 Value 对象的梯度，构建一个新的包含这些梯度的 MLArray。
        """

        def extract_grad(data):
            if isinstance(data, Value):
                return data.grad
            elif isinstance(data, list):
                return [extract_grad(item) for item in data]
            else:
                raise TypeError(f"Unexpected type in MLArray: {type(data)}")

        return MLArray(extract_grad(self.data))

    # 返回 MLArray 的字符串表示，美观地格式化多维数组数据
    def __repr__(self):
        def format_array(arr, indent=0):
            if not isinstance(arr, list):
                return str(arr)

            if not arr:
                return '[]'

            if isinstance(arr[0], list):
                # 2D or higher
                rows = [format_array(row, indent + 1) for row in arr]
                return '[\n' + ',\n'.join(' ' * (indent + 1) + row for row in rows) + '\n' + ' ' * indent + ']'
            else:
                # 1D
                return '[' + ', '.join(str(item) for item in arr) + ']'

        formatted_data = format_array(self.data)
        return f"MLArray(shape={self.shape},\ndata={formatted_data})"

    """
    ///////////////////////
    /// Suscriptability ///
    ///////////////////////
    """

    def __getitem__(self, index):
        """
        支持多维索引访问，通过递归获取指定位置的元素并封装为新的 MLArray。
        
        Examples:
            arr[0]     # get first element
            arr[1, 2]  # get element at row 1, column 2
        """
        if not isinstance(index, tuple):
            index = (index,)

        def get_item_recursive(data, index):
            if len(index) == 1:
                return data[index[0]]

            curr_index = index[0]
            return get_item_recursive(data[curr_index], index[1:])

        return MLArray(get_item_recursive(self.data, index))

    def __setitem__(self, index, value):
        """
        支持多维索引赋值，将指定位置的值替换为新的数据（自动转换为 MLArray）

        Examples:
            arr[0] = 1      # set first element
            arr[1, 2] = 3   # set element at row 1, column 2
        """
        if not isinstance(index, tuple):
            index = (index,)

        # Convert value to MLArray if it isn't already
        if not isinstance(value, MLArray):
            value = MLArray(value)

        def set_item_recursive(data, index, value):
            if len(index) == 1:
                data[index[0]] = value.data
            else:
                curr_index = index[0]
                set_item_recursive(data[curr_index], index[1:], value)

        set_item_recursive(self.data, index, value)

    """
    ////////////////////
    /// Classmethods ///
    Xavier 初始化（也叫 Glorot 初始化） 是一种在神经网络中初始化权重的策略
    ////////////////////
    """

    # 按Xavier均匀分布初始化权重，返回当前类的新实例
    @classmethod
    def xavier_uniform(cls, in_features, out_features):
        limit = math.sqrt(6. / (in_features + out_features))
        data = [[random.uniform(-limit, limit) for _ in range(out_features)] for _ in range(in_features)]
        return cls(data)

    # 按Xavier正态分布初始化权重，同样返回新实例
    @classmethod
    def xavier_normal(cls, in_features, out_features):
        std = math.sqrt(2. / (in_features + out_features))
        data = [[random.gauss(0, std) for _ in range(out_features)] for _ in range(in_features)]
        return cls(data)

    """
    //////////////////
    /// Properties ///
    //////////////////
    """

    # 通过_infer_shape方法推断并返回数据的形状
    @property
    def shape(self):
        return self._infer_shape(self.data)

"""
/////////////////////////
/// Pre-Made MLArrays ///
/////////////////////////
"""

def zeros(*shape):
    """
    创建一个指定形状的 MLArray，所有元素初始化为 0
    """

    def _zeros(shape):
        if len(shape) == 0:
            return Value(0.0)
        return [_zeros(shape[1:]) for _ in range(shape[0])]

    return MLArray(_zeros(shape))



def ones(*shape):
    """
    创建一个指定形状的 MLArray，所有元素初始化为 1
    """

    def _ones(shape):
        if len(shape) == 0:
            return Value(1.0)
        return [_ones(shape[1:]) for _ in range(shape[0])]

    return MLArray(_ones(shape))


def randn(*shape):
    """
    创建一个指定形状的 MLArray，元素为标准正态分布的随机数。
    """

    def _randn(shape):
        if len(shape) == 0:
            return Value(random.gauss(0, 1))
        return [_randn(shape[1:]) for _ in range(shape[0])]

    return MLArray(_randn(shape))
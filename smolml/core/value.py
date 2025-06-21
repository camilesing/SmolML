import math

"""
/////////////
/// VALUE ///
/////////////
    用于自动微分和梯度计算的类。
    基于karpathy的神奇代码(https://github.com/karpathy/micrograd).
"""


class Value:
    """
    存储单个标量值及其梯度以进行自动微分。
    通过Value对象的计算图实现反向传播。
    """

    def __init__(self, data, _children=(), _op=""):
        # 用数据和可选的梯度计算信息初始化值。  存储子节点和用于构建计算图的操作。
        self.data = data
        # 存储当前节点相对于最终输出值的梯度。默认初始化为 0，在调用 .backward() 时根据链式法则进行更新。
        self.grad = 0
        # 定义如何计算当前节点的输入值对应的梯度，是一个函数。在反向传播过程中被调用，将当前节点的梯度传递到其前置节点（_prev 中的节点）。
        self._backward = lambda: None
        # 记录当前节点所依赖的子节点（前驱节点），用于构建计算图。set类型是为了防止重复引用。
        self._prev = set(_children)
        # 记录生成当前节点的操作（operation），例如 "+", "*"。用于调试和可视化计算图中每个节点的来源操作。
        self._op = _op  # Operation that created this value

    def __add__(self, other):
        """
        实现了两个 Value 对象的加法运算，并构建计算图以支持反向传播
        """
        # 若 other 不是 Value 类型，则将其转换为 Value 实例
        other = other if isinstance(other, Value) else Value(other)
        # 创建一个新的 Value 实例 out，表示加法结果，记录运算符 "+" 和操作数 (self, other)
        out = Value(self.data + other.data, (self, other), "+")

        # 输出梯度 out.grad 传递给两个输入操作数；
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        # 绑定反向函数：将 _backward 赋值给 out._backward，为后续链式求导做准备
        # 后面代码类似，不再重复解释
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        实现两个 Value 对象的乘法运算，并构建了反向传播所需的梯度计算逻辑：
        """
        # 若 other 不是 Value 类型，则将其包装成 Value 对象；
        other = other if isinstance(other, Value) else Value(other)
        # 创建一个新的 Value 实例 out，表示乘法运算结果
        out = Value(self.data * other.data, (self, other), "*")

        # 定义 _backward 函数，实现乘法的导数规则（即：dx = y * dout, dy = x * dout）
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        实现了 Value 类的幂运算方法，支持int和float类型
        """
        # 若指数是 Value 类型，则提取其 .data 作为实际指数；
        other = other.data if isinstance(other, Value) else other
        assert isinstance(other, (int, float)), "only supporting int/float/value powers for now"
        # 创建新的 Value 实例表示运算结果，并记录梯度反传函数 _backward；
        out = Value(self.data ** other, (self,), f"**{other}")

        # 反向传播时根据幂函数求导规则更新梯度：d(out)/d(self) = other * self.data ** (other - 1)。
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __abs__(self):
        """
        对 Value 对象求绝对值，并支持反向传播计算梯度
        """
        # 创建新 Value 实例 out，记录绝对值结果和操作来源
        out = Value(abs(self.data), (self,), "abs")

        # 定义 _backward 函数，根据原值正负决定导数为 1 或 -1，并更新输入值的梯度；
        def _backward():
            self.grad += (1 if self.data >= 0 else -1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        实现了Value类的除法运算
        """
        # 若other不是Value类型，则将其转换为Value
        other = other if isinstance(other, Value) else Value(other)
        # 若除数绝对值极小（视为0），则返回值为0，并定义梯度为0的反向传播函数；
        if abs(other.data) < 1e-10:
            out = Value(0.0, (self, other), "/")

            def _backward():
                self.grad += 0.0
                other.grad += 0.0

            out._backward = _backward
            return out
        # 否则通过乘法与幂运算实现除法（self / other = self * other ^ -1）。
        return self * other ** -1

    def __rtruediv__(self, other):
        """
        Value类的反向除法运算（other / self），包含对除以零或极小数的处理
        """
        # 如果self.data接近零（绝对值小于1e-10），
        # 则返回一个值为0.0的Value对象，并定义其反向传播函数（梯度不传播）
        if abs(self.data) < 1e-10:  # Division by zero or very small number
            out = Value(0.0, (self,), "r/")

            def _backward():
                self.grad += 0.0

            # 将 _backward 赋给 out._backward，实现链式求导
            out._backward = _backward
            return out

        return other * self ** -1

    def exp(self):
        """
        指数运算（e^x）并支持反向传播计算梯度
        """
        # 计算 e^x 并创建输出 Value 对象。
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # 利用指数函数的导数仍为自身（即 d/dx e^x = e^x）进行梯度计算
            self.grad += math.exp(x) * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        计算自然对数并支持反向传播梯度计算
        """
        # 对数仅对正数有定义
        assert self.data > 0, "log is only defined for positive numbers"
        x = self.data
        # 使用 math.log 计算值并构建 Value 对象
        out = Value(math.log(x), (self,), "log")

        def _backward():
            # 依据导数公式 $ d(\ln x)/dx = 1/x $
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
         实现了 Rectified Linear Unit（修正线性单元），
         这是个激活函数。
         可以理解为只关注某个条件的单元。如果输入的对，就会输出一个高的值，
         不然输出一个低的值
        """
        # 创建新 Value 实例 out，若输入值小于0则为0，否则保留原值
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        # 根据 ReLU 的导数规则（正值为1，负值为0），将输出梯度乘对应导数并累加至输入的梯度中；
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        实现了双曲正切激活函数。
        本质就是把任意大小的输入“压扁”到 -1 到 1 的范围里，
        帮助 AI 模型更好地识别信号强弱
        """
        x = self.data
        # 前向传播：计算 tanh(x) 的值 t
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        # 结果封装为新的 Value 实例 out
        out = Value(t, (self,), "tanh")

        def _backward():
            # 导数规则
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        实现了反向传播算法，用于自动求导
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        # 从当前节点开始递归的向前构建拓扑结构，避免重复访问
        build_topo(self)
        # 初始化当前节点的梯度为 1
        self.grad = 1
        # 按照拓扑逆序调用每个节点的 _backward 方法，链式法则计算梯度。
        for v in reversed(topo):
            v._backward()

    # 义了运算符重载和比较方法，以支持对象间的数学运算与比较等操作
    def __neg__(self):
        return self * -1  # -self

    def __radd__(self, other):
        return self + other  # other + self

    def __sub__(self, other):
        return self + (-other)  # self - other

    def __rsub__(self, other):
        return other + (-self)  # other - self

    def __rmul__(self, other):
        return self * other  # other * self

    # Comparison and representation methods
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __eq__(self, other):
        if isinstance(other, Value): return self.data == other.data
        return self.data == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Value): return self.data < other.data
        return self.data < other

    def __le__(self, other):
        if isinstance(other, Value): return self.data <= other.data
        return self.data <= other

    def __gt__(self, other):
        if isinstance(other, Value): return self.data > other.data
        return self.data > other

    def __ge__(self, other):
        if isinstance(other, Value): return self.data >= other.data
        return self.data >= other

    def __hash__(self):
        return id(self)

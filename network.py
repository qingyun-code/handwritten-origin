import pickle
import gzip
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        功能：重写初始化函数，并初始化属性
        参数：
            sizes：存放网络各层神经元参数的列表，例如sizes=[2,3,1]
                   说明有三层网络，每层网络神经元个数分别为2个3个1个
        """

        # 初始化网络层数
        self.num_layers = len(sizes)

        # 初始化网络各参数的列表
        self.sizes = sizes

        """
        初始化偏执，初始值为使用高斯分布均值0，方差1的分布。
        测试程序：
        import numpy
        sizes = [2, 3, 1]
        biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        biases的结果：
        [array([[-1.39730573],
                [-0.84395433],
                [ 0.66160829]]), array([[1.03552743]])]
        """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        """
        初始化权重，初始值为使用高斯分布均值0，方差1的分布。
        设sizes = [2, 3, 1]
        则sizes[:-1] = [2, 3]
        sizes[1:] = [3, 1]
        zip(sizes[:-1], sizes[1:]) = [(2, 3), (3, 1)]
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        功能：进行前向传播操作
        参数：
            a：形状为(n,1)的输入值
        返回值：
            a：进行前向传播后的输出值
        """

        """
        前向传播计算
        设sizes = [2, 3, 1]
        biases在1~2和2~3层之间的行列数分别为(3,1)和(1,1)
        weights在1~2和2~3层之间的行列数分别为(3,2)和(1,3)
        因为是三层网络，所以进行两次循环，分别为：
        第二层输出的a=w*a+b=(3,2)dot(2,1)+(3,1)=(3,1)
        第三层输出的a=w*a+b=(1,3)dot(3,1)+(1,1)=(1,1)
        由上述假设所得到的a虽然只有一个值但是是二维的
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None):
        """
        功能：进行随机梯度下降，训练或者测试数据集并显示结果
        参数：
            training_data：例如(([[0x1,0x2]T],[[0y]]),......,([[9x1,9x2]T],[[9y]]))的训练集
            test_data：例如(([[0x1,0x2]T],[[0y]]),......,([[9x1,9x2]T],[[9y]]))的测试集
            epochs：迭代次数
            mini_batch_size：一个批次的训练数量
            eta：学习率
        """

        """
        将training_data的元组转换为列表并获取列表长度
        假设有10组训练数据，那么training_data列表化后的值为：
        [([[0x1,0x2]T],[[0y]]),......,([[9x1,9x2]T],[[9y]])]
        n的值为10
        """
        training_data = list(training_data)
        n = len(training_data)

        # 如果为测试模式的话，同训练数据所示
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # 进行迭代，每迭代一次输出一次结果
        for j in range(epochs):

            # 将训练数据打乱（每迭代一次打乱一次）
            random.shuffle(training_data)

            """
            将训练数据转换为小批量列表数据
            假设有10组训练数据，mini_batch_size=5则：
            training_data=[([[0x1,0x2]T],[[0y]]),........,([[9x1,9x2]T],[[9y]])]
            进行此操作后：
            mini_batches=[[([[0x1,0x2]T],[[0y]]),...,([[4x1,4x2]T],[[4y]])],
                          [([[5x1,5x2]T],[[5y]]),...,([[9x1,9x2]T],[[9y]])]]
            range(0, n, mini_batch_size)]这句话的意思就是在[0,n)中从
            0开始每隔mini_batch_size个数取一个数，所以最后取得的k为0，5
            """
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            """
            按照上述的假设我们可以得到：
            第一次：mini_batch=[([[0x1,0x2]T],[[0y]]),...,([[4x1,4x2]T],[[4y]])]
            第二次：mini_batch=[([[5x1,5x2]T],[[5y]]),...,([[9x1,9x2]T],[[9y]])]
            每循环一次便更新一次w和b
            """
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # 如果是测试数据则输出测试结果，反之输出训练结果
            if test_data:
                print("Epoch {} : {} / {}".format(j,
                    self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        功能：更新一个批次数据后的权重和偏执
        参数：
            mini_batch：一个批次的数据集
            eta：学习速率
        """

        # 将存放b和w的梯度的列表清零
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 对一个批次的数据进行反向传播，将所得梯度进行一个批次的求和
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b,
                delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,
                delta_nabla_w)]

        # 根据公式w-eta*(sum(dw)/m)和b-eta*(sum(db)/m)更新参数
        self.weights = [w-(eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        功能：进行反向传播
        参数：
            x：输入值列表
            y：准确值列表
        """

        # 设置要更新的b和w的相应列表并按照b和w的相应形状用零填充进行初始化
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # activation存放初始化的输入值x，只计算第一次
        activation = x

        # 列表存储所有激活，一层一层
        activations = [x]

        # 列表存储所有激活，一层一层
        zs = []

        """
        此步为前向传播操作并将相关数据进行存储
        假设sizes=[2,3,1]，此步操作后各数据行列值为：
        zs=[(3,1),(1,1)]
        activations=[(2,1),(3,1),(1,1)]
        """
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 反向传播的开始，从后开始获取第一个db值，delta=(1,1)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # 将db值传入db的更新列表中
        nabla_b[-1] = delta

        # 将dw值传入dw的更新列表中，nabla_w[-1]=(1,3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        """
        此操作为从后向前获取新的db和dw并放入其列表中
        若假设sizes=[2,3,1]，则此循环进行一次，得到最终行列为：
        sp=(3,1)
        nabla_b=[(3,1),(1,1)]
        nabla_w=[(3,2),(1,3)]
        最终返回的nabla_b和nabla_w与self.biases和self.weights形状相同
        """
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        功能：对测试数据进行测试
        参数：
            test_data：测试数据集
        返回值：返回正确的个数
        """

        # argmax返回列表中值最大的索引，相当于概率最大的值的索引
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        # 若预测结果等于真实结果将值记为1，并计算总和返回
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        功能：计算损失
        参数：
            output_activations：预测值
            y：实际值
        返回值：返回损失
        """
        return (output_activations - y)

def sigmoid(z):
    """
    功能：实现sigmoid函数
    参数：
        z：w*a+b的值的列表
    返回值：返回z的sigmoid函数值
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    功能：计算sigmoid函数的导数
    参数：
        z：w*a+b的值的列表
    返回值：返回z的sigmoid函数的导数值
    """
    return sigmoid(z) * (1 - sigmoid(z))

def load_data():
    """
    功能：打开压缩包并获取数据集。
    训练集中有50000个组数据，验证集和测试集中分别有10000组数据
    """

    # 打开压缩文件
    f = gzip.open('mnist.pkl.gz', 'rb')

    # 从文件中下载数据集存入变量中
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    # 关闭文件
    f.close()

    # 返回各类数据集
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    功能：将数据集整理为相应格式的数据集
    """

    # 从压缩文档中获取数据集
    tr_d, va_d, te_d = load_data()

    # 将训练集打包成行列数为((784,1),(10,1))的元组
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    # 将验证集打包成行列数为((784,1),1)的元组
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    # 将测试集打包成行列数为((784,1),1)的元组
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    # 返回各类数据集
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    功能：将正确值转换为索引号为对应值为1其余值为0的列表
    参数：
        j：正确值索引
    返回值：
        e：转换后的列表
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    # 加载数据集
    training_data, validation_data, test_data = 
        load_data_wrapper()

    # 对网络进行初始化
    net = Network([784, 30, 10])

    # 随机梯度下降显示测试结果
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
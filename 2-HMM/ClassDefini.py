import numpy as np
import pandas as pd

class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()
        
        #check
        assert len(states) == len(probs), "The probabilities must match the states."
        assert len(states) == len(set(states)), "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."
        
        self.states = sorted(probabilities) #dict key sorted
        self.values = np.array(list(map(lambda x:
            probabilities[x], self.states))).reshape(1, -1)
    
    #随机初始化概率，存储在rand中，概率和为1
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size #保证了基本没有很小很小的值，因为有1/size
        rand /= rand.sum(axis=0) #normalize
        return cls(dict(zip(states, rand))) #cls即类本身
    
    #使用特定的值
    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(state, list(array))))

    @property #将操作封装成一个属性
    def dict(self):
    #得到dict的key：value
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}
        #flatten()用户将数组转为1维度。
    
    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])
    #返回一个二维数组，列是states，行的值是values。

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)
    #输出当前的states和values

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        #判断other的类型是ProbabilityVector
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False
    #用于判断两个PV是否是相等的

    #得到state对应的值，结果为float。
    def __getitem__(self, state: str) -> float:
        #判断state有意义
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index]) #values[0]是因为array原本都是二维的，但是实际有意义的都在第一维。

    #
    def __mul__(self, other) -> np.ndarray:
        #如果other的类型是ProbabilityVector，即
        if isinstance(other, ProbabilityVector):
            return self.values * other.values #对于*，若是数组，是对应元素相乘；
        elif isinstance(other, (int, float)):
            #若other是数字，则直接相乘
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
    #若是矩阵，则按照矩阵的乘法进行运算。
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values #测验发现这个，如果values的值都是numpy.matrix，则会执行矩阵的乘法运算。

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)
    #将values的值都除以number。需要考虑number不为0情况。

    def argmax(self):
        index = self.values.argmax() #argmax()返回的是最大值的索引！是索引！！
        return self.states[index]

class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        #prob_vec_dict可以用二维字典来实现。
        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."
        #赋值
        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values \
                           for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        #根据用户输入的状态和观测序列，随机生成观测矩阵
        #初始化，假设states为list('abcd')，observables为list('xyz')
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
             / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        #根据自己设定的状态和观测状态，随机生成概率矩阵。
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
                  np.ndarray,
                  states: list,
                  observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) \
                  for x in array]
        return cls(dict(zip(states, p_vecs)))
        #根据用户给出的概率，生成转移观测矩阵

    @property
    def dict(self):
        return self.df.to_dict()
    #使用to_dict，转为dict格式

    @property
    def df(self):
        return pd.DataFrame(self.values,
               columns=self.observables, index=self.states)
    #返回df格式

    #输出显示
    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    #返回的是某一列的值
    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)

if __name__ == '__main__':
    a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
    a2 = ProbabilityVector({'rain': 0.6, 'sun': 0.4})
    A  = ProbabilityMatrix({'hot': a1, 'cold': a2})
    print(A)
    print(A.df)

    b1 = ProbabilityVector({'0S': 0.1, '1M': 0.4, '2L': 0.5})
    b2 = ProbabilityVector({'0S': 0.7, '1M': 0.2, '2L': 0.1})
    B =  ProbabilityMatrix({'0H': b1, '1C': b2})
    print(B)
    print(B.df)

    P = ProbabilityMatrix.initialize(list('abcd'), list('xyz'))

    print('Dot product:', a1 @ A)
    print('Initialization:', P)
    print(P.df)


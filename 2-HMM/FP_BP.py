import ClassDefini as CD
import numpy as np
from itertools import product
from functools import reduce

class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    #输出隐状态和观测状态的个数
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = CD.ProbabilityMatrix.initialize(states, states)
        E = CD.ProbabilityMatrix.initialize(states, observables)
        pi = CD.ProbabilityVector.initialize(states)
        return cls(T, E, pi)
    #如果调用这边，矩阵都随机初始化
    
    #生成所有可能的序列！！学到了太巧妙了
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
        
        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            #构造了隐状态的转移序列(B,A)，表示由A状态转为B状态。但是这一行代码把初始的隐状态定死了为T.states[0]
            
            expanded_obser = list(zip(observations, chain))
            #观测状态与对应的隐状态，其实是为了计算观测概率
            
            #查表，得到生成相应观测状态的概率
            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            
            #查表，得到生成这个隐状态序列的概率
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            #修正第一个状态的概率
            p_hidden_state[0] = self.pi[chain[0]]
            
            #reduce函数会对参数序列中的元素进行累积
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
            
        return score

class HiddenMarkovChain_FP(HiddenMarkovChain):

    def _alphas(self, observations: list) -> np.ndarray:
        #一个矩阵，用来记录前向概率的值。
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        #a0的前向概率的值为pi * 观测矩阵
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                         @ self.T.values) * self.E[observations[t]].T
        return alphas
        #alphas存储了所有节点的前向概率
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum()) #alphas[-1]得到的概率是 不同的隐状态生成相应的观测状态的概率。所以求和就是我们要求的前向概率。

class HiddenMarkovChain_BP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        #一个矩阵，用来记录后向概率的值。
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[-1, :] = 1.0
        #at的前向概率的值为1
        for t in range(len(observations) - 2, -1, -1):
            alphas[t, :] = ((alphas[t + 1, :].reshape(1, -1)
                         * self.T.values) * self.E[observations[t+1]].T).sum(axis = 1)
        return alphas
        #alphas存储了所有节点的后向概率
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float((alphas[0] * self.E[observations[0]].T * self.pi.values ).sum())
        #注意最开始的节点还需要乘上自身的观测概率以及初始的概率
         #alphas[-1]得到的概率是 不同的隐状态生成相应的观测状态的概率。所以求和就是我们要求的前向概率。

if __name__ == '__main__':
#test
    a1 = CD.ProbabilityVector({'1Sunny': 0.4, '2Cloud': 0.5, '3Rainy': 0.1})
    a2 = CD.ProbabilityVector({'1Sunny': 0.3, '2Cloud': 0.4, '3Rainy': 0.3})
    a3 = CD.ProbabilityVector({'1Sunny': 0.1, '2Cloud': 0.5, '3Rainy': 0.4})

    b1 = CD.ProbabilityVector({'1G': 0.8, '2C': 0.2})
    b2 = CD.ProbabilityVector({'1G': 0.6, '2C': 0.4})
    b3 = CD.ProbabilityVector({'1G': 0.3, '2C': 0.7})

    A = CD.ProbabilityMatrix({'1Sunny': a1, '2Cloud': a2, '3Rainy': a3})
    B = CD.ProbabilityMatrix({'1Sunny': b1, '2Cloud': b2, '3Rainy': b3})
    pi = CD.ProbabilityVector({'1Sunny': 0.4, '2Cloud': 0.3, '3Rainy': 0.3})

    hmc_fp = HiddenMarkovChain_BP(A, B, pi)
    hmc_bp = HiddenMarkovChain_BP(A, B, pi)


    observations = ['1G', '2C', '1G']


    print("FP Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))
    print("BP Score for {} is {:f}.".format(observations, hmc_bp.score(observations)))


'''
    #前向算法 验证概率之和为1
    all_possible_observations = {'1G', '2C'}
    chain_length = 20  # 速度很快
    all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
    all_possible_scores = list(map(lambda obs: hmc_fp.score(obs), all_observation_chains))
    print("FP check")
    print("chain_length is:", chain_length)
    print("All possible scores added: {}.".format(sum(all_possible_scores)))
'''

'''
#后向算法 验证概率之和为1
    all_possible_observations = {'1G', '2C'}
    chain_length = 10  # 速度很快
    all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
    all_possible_scores = list(map(lambda obs: hmc_bp.score(obs), all_observation_chains))
    print("BP check")
    print("chain_length is:", chain_length)
    print("All possible scores added: {}.".format(sum(all_possible_scores)))
'''

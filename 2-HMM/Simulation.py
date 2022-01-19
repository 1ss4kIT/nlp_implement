import ClassDefini as CD
from FP_BP import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)
        
        #prb存储初始概率
        prb = self.pi.values
        #obs存储初始的观测概率
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())
        
        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        return o_history, s_history

if __name__ == '__main__':
    a1 = CD.ProbabilityVector({'1Sunny': 0.4, '2Cloud': 0.5, '3Rainy': 0.1})
    a2 = CD.ProbabilityVector({'1Sunny': 0.3, '2Cloud': 0.4, '3Rainy': 0.3})
    a3 = CD.ProbabilityVector({'1Sunny': 0.1, '2Cloud': 0.5, '3Rainy': 0.4})
    b1 = CD.ProbabilityVector({'1G': 0.8, '2C': 0.2})
    b2 = CD.ProbabilityVector({'1G': 0.6, '2C': 0.4})
    b3 = CD.ProbabilityVector({'1G': 0.3, '2C': 0.7})
    A = CD.ProbabilityMatrix({'1Sunny': a1, '2Cloud': a2, '3Rainy': a3})
    B = CD.ProbabilityMatrix({'1Sunny': b1, '2Cloud': b2, '3Rainy': b3})
    pi = CD.ProbabilityVector({'1Sunny': 0.4, '2Cloud': 0.3, '3Rainy': 0.3})

    hmc_s = HiddenMarkovChain_Simulation(A, B, pi)
    '''
    observation_hist, states_hist = hmc_s.run(100)  # length = 100,除了最开始的初始状态0，还生成100个状态的序列

    #这行是用来画图可视化的
    stats = pd.DataFrame({
        'observations': observation_hist,
        'states': states_hist}).applymap(lambda x: int(x[0])).plot()
    '''
    stats = {}
#logspace用于创建等比数列
    for length in np.logspace(1, 5, 40).astype(int):
        observation_hist, states_hist = hmc_s.run(length)
        stats[length] = pd.DataFrame({
            'observations': observation_hist,
            'states': states_hist}).applymap(lambda x: int(x[0]))
    #以上的for循环，进行很多很多次独立的实验，用stats记录每次实验的观测结果和隐状态。记录的是第一个字符，所以需要首字母各不相同
    '''
    S = np.array(list(map(lambda x:
            x['states'].value_counts().to_numpy() / len(x), stats.values())))
#这里统计的是总体上每个隐状态的概率

    plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
    plt.xlabel('Chain length T')
    plt.ylabel('Probability')
    plt.title(' States Converging probabilities.')
    plt.legend(['1Sunny', '2Cloud', '3Rainy'])
    plt.show()
    '''
    S = np.array(list(map(lambda x:
            x['observations'].value_counts().to_numpy() / len(x), stats.values())))
#这里统计的是总体上每个隐状态的概率

    plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
    plt.xlabel('Chain length T')
    plt.ylabel('Probability')
    plt.title(' Observations Converging probabilities.')
    plt.legend(['1G', '2C'])
    plt.show()

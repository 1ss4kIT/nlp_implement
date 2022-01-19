import ClassDefini as CD
from FP_BP import *
from Simulation import HiddenMarkovChain_Simulation

import pandas as pd

class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    #前向算法
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                         * self.E[observations[t]].T
        #观测概率为：
        # np.sum(alphas[:,-1])
        return alphas
    
    #后向算法
    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas
    
    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        #此处的 alphas * betas是点乘，然后按行找出值最大的坐标，将坐标记录在maxargs矩阵中。
        return list(map(lambda x: self.states[x], maxargs))
        #return这行相当于按照坐标找出对应的状态序列。

    #自己实现滴！！
    def Viterbi(self, observations: list) -> list:
        path = np.zeros(len(observations), dtype=np.int32)
        alphas = np.zeros((len(observations), len(self.states)))
        phi = np.zeros((len(observations),len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        #这边改成max
        for t in range(1, len(observations)):
            alphas[t, :] = np.max(alphas[t - 1, :].reshape(-1, 1) * self.T.values, axis = 0) * self.E[observations[t]].T
            phi[t, :] = np.argmax(alphas[t - 1, :].reshape(-1, 1) * self.T.values, axis = 0)
            
        path[-1] = np.argmax(alphas[-1, :])
        for t in range(len(observations) - 2, -1, -1):
            path[t] = phi[t + 1, path[t + 1]]
        score = np.max(alphas[-1, :])
        return path, score

    #输出转换一下
    def Trans(self, paths):
        res = []
        for i in range(len(paths)):
            if paths[i] == 0:
                res.append('1Sunny')
            elif paths[i] == 1:
                res.append('2Cloud')
            elif paths[i] == 2:
                res.append('3Rainy')
            else:
                print("Translate wrong!Something happend.")
        return res
        
if __name__ == '__main__':
    np.random.seed(42)
    a1 = CD.ProbabilityVector({'1Sunny': 0.5, '2Cloud': 0.2, '3Rainy': 0.3})
    a2 = CD.ProbabilityVector({'1Sunny': 0.3, '2Cloud': 0.5, '3Rainy': 0.2})
    a3 = CD.ProbabilityVector({'1Sunny': 0.2, '2Cloud': 0.3, '3Rainy': 0.5})
    b1 = CD.ProbabilityVector({'1G': 0.5, '2C': 0.5})
    b2 = CD.ProbabilityVector({'1G': 0.4, '2C': 0.6})
    b3 = CD.ProbabilityVector({'1G': 0.7, '2C': 0.3})
    A = CD.ProbabilityMatrix({'1Sunny': a1, '2Cloud': a2, '3Rainy': a3})
    B = CD.ProbabilityMatrix({'1Sunny': b1, '2Cloud': b2, '3Rainy': b3})
    pi = CD.ProbabilityVector({'1Sunny': 0.2, '2Cloud': 0.4, '3Rainy': 0.4})
    hmc = HiddenMarkovChain_Uncover(A, B, pi)
    
    
    observed_sequence = ['1G', '2C', '1G']
    score = hmc.score(observed_sequence)
    print("Score of this observed_sequence is", score)
    ''' test one
    '''
    uncovered_sequence = hmc.uncover(observed_sequence)
#    observed_sequence, latent_sequence = hmc.run(3)
#    uncovered_sequence = hmc.uncover(observed_sequence)
    print("observed_sequence is: ",observed_sequence)
#    print("latent_sequence is: ", latent_sequence)
    print("uncovered_sequence is: ", uncovered_sequence)
#    hmc_score = HiddenMarkovChain(A, B, pi)
#    hmc_score.score(uncovered_sequence)

    Viterbi_sequence, score = hmc.Viterbi(observed_sequence)
    Viterbi_nomal = hmc.Trans(Viterbi_sequence)
    print("Viterbi_sequence is: ",Viterbi_nomal,"Score is:",score)

    '''观测概率的值相加，没什么实际意义
    all_possible_states = {'1Sunny', '2Cloud', '3Rainy'}
    chain_length = 3  # any int > 0
    all_states_chains = list(product(*(all_possible_states,) * chain_length))
    df = pd.DataFrame(all_states_chains)
    dfp = pd.DataFrame()
    for i in range(chain_length):
        dfp['p' + str(i)] = df.apply(lambda x:
            hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)

    #把这些值加起来了，实际上有什么物理意义吗？这里还没搞清楚
    scores = dfp.sum(axis=1).sort_values(ascending=False)
    df = df.iloc[scores.index]
    df['score'] = scores
    print(df.head(10).reset_index())
    '''

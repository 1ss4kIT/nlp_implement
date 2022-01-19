from Uncover import *
from ComputeScore import *
from Simulation import *
from ClassDefini import *

class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
#        可以看成是求解不同的转移时刻(共t-1个)，的r的值
        L, N = len(observations), len(self.states)
        #print("L is",L,"N is",N) ；L is 4 N is 3
        #N表示所有可能的状态的个数，L表示状态转移的次数
        digammas = np.zeros((L - 1, N, N))
        
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
#        print("1 alphas,betas,score")
#        print("alphas is\n", alphas)
#        print("betas is\n",betas)
#        print("score is\n",score)
        for t in range(L - 1):
#            print("t is",t)
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values) #P1可以看成是前向概率*转移概率
#            print("P1 is\n",P1)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1) #P2是下一个节点的观测概率*下一个节点的后向概率
#            print("P2 is\n", P2)
            digammas[t, :, :] = P1 * P2 / score
#            print("P1*P2\n",P1*P2)
#            print("digammas",digammas)
#            print("score is",score)
#        print("p1",P1)
#        print("p2",P2)
#        print(digammas)
        return digammas

class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        #clean
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)  #也是随机生成矩阵的参数
        return cls(layer)

    def update(self, observations: list) -> float:
        #函数的输入是observations,即观测序列
        print("Now update!!!")
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
#        print("alphas is :", alpha)
#        print("beta is :", beta)
#        print("digamma is :",digamma)
        score = alpha[-1].sum()
#        print("sum is:",alpha[-1], score)
        gamma = alpha * beta / score
        #rt(i)
#        print("gamma\n",gamma) #维度是4*3
        
        L = len(alpha)
#        print("L",L,self.layer.states)
#        for x in observations:
#            print(x)
        obs_idx = [self.layer.observables.index(x) \
                  for x in observations]
#        print("self.layer.observables",self.layer.observables)
#        print("obs_idx",obs_idx)
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables))) #维度为4*3*3
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0
#            print("t is, capture is\n",t,capture)
#            print("END!!!")
        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
#        分子可以看成是概率求和
#        分母对于gamma矩阵，先舍弃最后一行，再求按列求和
#        print("digamma\n",digamma.sum(axis=0))
#        print("\ngamma\n",gamma[:-1].sum(axis=0).reshape(-1, 1))
#        print("\nT is\n",T)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)
#        print("capture is\n",capture)
#        print("gamma is\n",gamma,np.newaxis,"\n",gamma[:, :, np.newaxis])
#        print("Multi\n",capture * gamma[:, :, np.newaxis],len(capture * gamma[:, :, np.newaxis]))
#        print("Sum",(capture * gamma[:, :, np.newaxis]).sum(axis=0))
#        print("gamma",gamma,"\n\n",gamma.sum(axis=0))
#        print("E is\n",E)
        
        #这样直接在self里更新参数了，不用再传回去了！
        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)
        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))
        #设定是否要提前结束。

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score
            
if __name__ == '__main__':
    '''
    np.random.seed(25)
    observations = ['1G', '2C', '2C','1G'] #真实观测到的状态
    states = ['1Sunny', '2Cloud', '3Rainy'] #所有可能的隐状态
    observables = ['1G', '2C'] #所有可能的观测状态
    
    #先随机生成矩阵的参数，可以调用hml.T hml.E hml.pi查看
    hml = HiddenMarkovLayer.initialize(states, observables)
    hmm = HiddenMarkovModel(hml)
    #hmm.layer就是原来的hml, hmm可以查看hmm._score_init，hmm.score_history
    
    hmm.train(observations, 8)
    #传入参数如下：(self, observations: list, epochs: int, tol=None)。若设定了tol，则训练会提前结束。
    '''
    
    np.random.seed(42)
    observations = ['3L', '2M', '1S', '3L', '3L', '3L']
    states = ['1H', '2C']
    observables = ['1S', '2M', '3L']
    hml = HiddenMarkovLayer.initialize(states, observables)
    hmm = HiddenMarkovModel(hml)
    #hmm.train(observations, 25)

    RUNS = 10000
    T = 5
    chains = RUNS * [0]
    for i in range(len(chains)):
        res = hmm.layer.run(T) #包含 观测状态、隐状态
        chain = res[0] #观测状态
#        print("res is",res)
#        print("chain is",chain)
        chains[i] = '-'.join(chain)
#        print("i is:",i,"chains is:",chains)
    df = pd.DataFrame(pd.Series(chains).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
    df = pd.merge(df, df['chain'].str.split('-', expand=True), left_index=True, right_index=True)

    s = []
    for i in range(T + 1):
        s.append(df.apply(lambda x: x[i] == observations[i], axis=1))

    df['matched'] = pd.concat(s, axis=1).sum(axis=1)
    df['counts'] = df['counts'] / RUNS * 100
    df = df.drop(columns=['chain'])
    print(df.head(30))


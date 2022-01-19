import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)
'''
start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

seen = np.array([[0,1,0]]).T

model.score(seen)

logprob, box = model.decode(seen, algorithm="viterbi")
#print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
print("The hidden box:", ", ".join(map(lambda x: states[x], box)))
'''

model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0],[1],[0],[1],[0],[0],[0],[1],[1],[0],[1],[1]])
model2.fit(X2,lengths=[4,4,4])
print(model2.startprob_)
print("\n")
print(model2.transmat_)
print("\n")
print(model2.emissionprob_)
print("\n")
print(model2.score(X2))
print("\n")

model2.fit(X2)
print(model2.startprob_)
print("\n")
print(model2.transmat_)
print("\n")
print(model2.emissionprob_)
print("\n")
print(model2.score(X2))
print("\n")

model2.fit(X2)
print(model2.startprob_)
print("\n")
print(model2.transmat_)
print("\n")
print(model2.emissionprob_)
print("\n")
print(model2.score(X2))
print("\n")

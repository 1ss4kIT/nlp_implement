import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline
plt.style.use('ggplot')
np.random.seed(1234)

np.set_printoptions(formatter={'all':lambda x: '%.3f' % x})

from IPython.display import Image
from numpy.core.umath_tests import matrix_multiply as mm

from scipy.optimize import minimize
from scipy.stats import bernoulli, binom





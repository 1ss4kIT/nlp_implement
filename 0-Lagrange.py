'''
有两种实现方法:minimize 和 sympy diff
method 1:
    不同的变量都用x来表示，如x y z分别表示为x[0] x[1] x[2]
    最小值的求解使用minimize函数。详细内容参考博客：https://blog.csdn.net/qq_38048756/article/details/103208834。method可选方法：BFGS / L-BFGS-B / SLSQP。

methos 2:
    运用diff，计算过程更接近人手工计算的过程。
    
注意：method 1不能得到所有的结果，method2能得到所有的结果，但是结果中有不可用的，需要排除一定的情况。
'''


'''
#method 1
from scipy.optimize import minimize
import numpy as np

def func(args):
    fun = lambda x: x[0]**2 + x[1]**2
    return fun
    
def con(args):
    cons = ({'type':'eq', 'fun': lambda x: x[0]*x[1]-3})
    return cons

if __name__ == "__main__":
    args = ()
    args1 = ()
    cons = con(args1)
    x0 = np.array((2.0, 100.0)) #initial
    res = minimize(func(args), x0, method = 'SLSQP', constraints = cons)
    print(res.success)
    print("x = ",res.x[0],";\ny = ", res.x[1])
    print("最优解的值为:", res.fun)

'''

#method 2
from sympy import *

x = symbols("x")
y = symbols("y")
alpha = symbols("alpha")

L = x**2 + y**2 + alpha*(x*y - 3)
difyL_x = diff(L,x)
difyL_y = diff(L,y)
difyL_alpha = diff(L, alpha)

aa = solve([difyL_x, difyL_y, difyL_alpha],[x, y, alpha])
print(aa)
#计算一下最后的结果，输出即可


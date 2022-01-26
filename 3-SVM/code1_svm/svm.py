# Mathieu Blondel, September 2010
# License: BSD 3 clause

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from DataGene import *

#linear_kernel,一维是求和相加，如果是矩阵，就是标准的矩阵乘法。此处是相加求和。
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
#        print("INIT\n...")
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
        print("C is",self.C)
#        print("...\nEND INIT.")
        
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
                #将每个点和其他的点进行点乘，求和，得到一个数值。
#        print("K is: ",K)
        
        P = cvxopt.matrix(np.outer(y,y) * K)
        #np.outer(y,y)会对y求外积，相当于y求外积，再和刚才求得的K点乘
        
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        #P q为求解其负值的最小值。
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            #如果没设置C，G就是斜线上值为-1的数组。
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            #还没看
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            #tmp1斜线上为-1，tmp2是斜线上为1的方阵。
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            #G把tmp1和tmp2连接起来
            tmp1 = np.zeros(n_samples) #置为0，长度为n_samples
            tmp2 = np.ones(n_samples) * self.C #值置为C，长度为n_samples
            h = cvxopt.matrix(np.hstack((tmp1, tmp2))) #横着拼接起来

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        #如果是支持向量，则a的值>0
        sv = a > 1e-5
        ind = np.arange(len(a))[sv] #找出支持向量的索引
        self.a = a[sv] #找出支持向量a,更新a，仅包含支持向量的a。
        self.sv = X[sv] #把X更新。X表示训练集中的支持向量。
        self.sv_y = y[sv] #支持向量的标签
        print("%d support vectors out of %d points" %(len(self.a), n_samples))

        # Intercept
        #求解b
        self.b = 0
        for n in range(len(self.a)):
#            print("n is",n)
            self.b += self.sv_y[n] #加上对应的标签
#            print("1---self.b:",self.b)
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
#            print("fenbie self.a, self.sv_y, K[ind[n],sv]",self.a,self.sv_y, K[ind[n],sv],"\n",self.b)
#            print("a*sv_y", self.a * self.sv_y,"\n")
            #减掉
        self.b /= len(self.a)
#        print("b is:", self.b)
        #上面的一个for循环就相当于把公式计算了一遍。现在循环了几次，就要再除以几求平均值。

        # Weight vector
        #此处线性才计算权重，如果不是线性就不计算了。
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
#                print("self.w:",self.w)
        else:
            self.w = None

    #预测函数
    def project(self, X):
        if self.w is not None:
            #正常的情况是前面已经得到了w和b，即是线性kernel。
            return np.dot(X, self.w) + self.b
        else:
            #非线性kernel
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl
    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" %(correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" %(correct, len(y_predict)))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)
        
    def test_soft():
        #前90个数据当训练集，之后的当测试集
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        #X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.5)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" %(correct, len(y_predict)))
        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
        
        
#    test_linear()
#    test_soft()
    test_non_linear()

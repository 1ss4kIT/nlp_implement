import numpy as np

#生成二维矩阵，带标签
def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 5)
    #multivariate_normal 从多元正态分布中随机抽取样本。mean表示均值，cov表示协方差，size此处是100，表示生成样本的数量。
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 5)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

#生成的数据线性不可分。
def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 3)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 2)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 3)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 2)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

#生成的数据有重叠，这是mean来控制的
def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 5)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 4)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2
    
#取前90个数据做训练集,并且将X1和X2的数据集合并在一起。
def split_train(X1, y1, X2, y2):
    X1_train = X1[:4]
    y1_train = y1[:4]
    X2_train = X2[:4]
    y2_train = y2[:4]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

#取90之后的数据当作测试集
def split_test(X1, y1, X2, y2):
    X1_test = X1[4:5]
    y1_test = y1[4:5]
    X2_test = X2[4:5]
    y2_test = y2[4:5]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

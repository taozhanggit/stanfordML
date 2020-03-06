import numpy as np
def normal_eq(X,y):
    '''
    :param X:  数据
    :param y:  结果
    :return: theta向量
    '''
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

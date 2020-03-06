import numpy as np
def normal_eq(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

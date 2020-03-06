import numpy as np
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    '''
    J = sum((X * theta - y).^2) / (2*m);
    :param X: 数据
    :param y: 结果
    :param theta: 梯度向量
    :return: cost value
    '''
    inner = np.power(np.dot(X, theta.T) - y, 2)
    # print(np.sum(inner) / (2 * len(X)))
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)  # 数据大小
    J_history = np.zeros(num_iters)  # 记录cost
    theta_temp = theta.copy()  # 临时theta，用于批量梯度下降

    print(X.shape, y.shape, theta.shape)
    for i in range(num_iters):
        error = np.dot(X, theta.T) - y
        # print(X.shape[1])
        for j in range(X.shape[1]):
            term = np.dot(X[:, j], error)
            theta_temp[0, j] = theta[0, j] - (alpha / m) * np.sum(term)
        theta = theta_temp
        print(theta_temp[0, 0], theta_temp[0, 1])

        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


ex1data1 = "ex1data1.txt"

data = np.genfromtxt(ex1data1, delimiter=",")
m = len(data)

# print(data[:,0])
# print(data[:,1])
# print(len(data))
# print(data.shape)
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of City in 10,000s")
plt.scatter(data[:, 0], data[:, 1])  # 散点图
plt.show()

# print(np.ones(m))
# print(data)
X = np.hstack((np.ones((m, 1)), data[:, 0].reshape(m, 1)))  # 构建X
y = data[:, 1].reshape(m, 1)  # 构建y
theta = np.zeros((1, 2))  # 全0列向量
# print(X.shape, y.shape, theta.shape)
# print(theta[0, 0])

iterations = 1500  # 迭代次数
alpha = 0.01  # 学习率

theta, J_his = gradient_descent(X, y, theta, alpha, iterations)
# print("theta:", theta, "CostHistory:", J_his)

plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of City in 10,000s")
plt.scatter(data[:, 0], data[:, 1])  # 散点图

plt.plot(data[:, 0], data[:, 0] * theta[:, 1] + theta[:, 0], label="Prediction", color='r')
plt.legend(loc=2)
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), J_his, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

'''
Next we conduct experiments on multiple variables. 
'''
ex1data2 = "ex1data2.txt"
data = np.genfromtxt(ex1data2, delimiter=",")
m = len(data)
print(len(data))
print(data.shape)

''' 预处理，feature standardization'''
def feature_normalize(dt):
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma


data2 = feature_normalize(data)
cols = data.shape[1]
X2 = np.hstack((np.ones((m, 1)), data2[:, 0:cols - 1].reshape(m, cols - 1)))  # 构建X
print(X.shape)
y2 = data2[:, 1].reshape(m, 1)  # 构建y
theta2 = np.zeros((1, cols))  # 全0列向量

g2, cost2 = gradient_descent(X2, y2, theta2, alpha, iterations)

print("Cost:", compute_cost(X2, y2, g2))

''' Next we use normal equation to get theta'''
data = np.genfromtxt(ex1data1, delimiter=",")

import normal_equation

theta3 = normal_equation.normal_eq(X, y)
print("theta3", theta3)
# print(theta3.shape)
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of City in 10,000s")
plt.scatter(data[:, 0], data[:, 1])  # 散点图

plt.plot(data[:, 0], data[:, 0] * theta3[1, :] + theta3[0, :], label="Prediction", color='r')
plt.legend(loc=2)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "ex1data1.txt"
data = pd.read_csv(data_path, names=['population', 'profit'])
print(data.head()) # 看看dataframe的前5行
print(data.describe())

data.plot(kind="scatter",x='population', y='profit')
plt.show()

#print(np.power(3,2))

#points是(x,y)的数据对
def compute_error(b, w, points):
    totalError = 0
    pointNum = len(points)
    for index, row in points.iterrows():
        x = row['population']
        #print("x", x)
        y = row['profit']
        #print("y", y)
        totalError += np.power((y-(w*x+b)), 2) # (y-(w*x+b))**2
    return totalError / float(2 * pointNum)

def batchGradientDescent(b, w, learningRate, points):
    b_gradient = 0
    w_gradient = 0
    pointNum = len(points)
    xs = points['population']
    ys = points['profit']
    for i in range(pointNum):
        x = xs[i]
        y = ys[i]
        b_gradient += (2/float(pointNum)) * ((w * x + b) - y)       #求导
        w_gradient += (2/float(pointNum)) * ((w * x + b) - y) * x
    new_b = b - (learningRate * b_gradient)
    new_w = w - (learningRate * w_gradient)
    return [new_b, new_w]

def gradientDescentCompute(points, startB, startW, learningRate, iters):
    b = startB
    w = startW
    for i in range(iters):
        b, w = batchGradientDescent(b, w, learningRate,points)
    return b, w

b, w = gradientDescentCompute(data, 0, 0, 0.001, 10000)


plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*w + b, label="Prediction")
plt.legend(loc=2)
plt.show()
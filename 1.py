import matplotlib.pyplot as plt
import pandas as pd 
import math 
import scipy as sc
import numpy as np


data = pd.read_csv('ex1data1.csv', header = None) 
x, y  = data[0], data[1]
fy = lambda x: 1.5*x - 10
x1 = sc.linspace(min(x),max(x),10)
y1 = list(map(fy,x1)) 
fig = plt.figure()  

"""
Алгоритма градиентного спуска
"""
def gradient_descent(X, Y, koef, n):
    l = len(x)
    theta0, theta1 = 0, 0
    for i in range(n):
        sum1 = 0
        for i in range(l):
            sum1 += theta0 + theta1 * x[i] - y[i]
        res1 = theta0 - koef * (1 / l) * sum1

        sum2 = 0
        for i in range(l):
            sum2 += (theta0 + theta1 * x[i] - y[i]) * x[i]
        res2 = theta1 - koef * (1 / l) * sum2

        theta0, theta1 = res1, res2

    return theta0, theta1



def sq_error(X,Y,f_x=None): 
    squared_error = []; 
    for i in range(len(X)): 
        squared_error.append((f_x(X[i])-Y[i])**2) 
    return sum(squared_error)

print(sq_error(x,y,fy)) 

pll = plt.subplot(111) 
pll.plot(x,y,'b*') 
pll.plot(x1,y1,'g-') 
plt.show()

"""
По градиентному спуску
"""
x2 = [1, 25]
y2 = [0, 0]
t0, t1 = gradient_descent(x, y, 0.01, len(x))
y2[0] = t0 + x2[0] * t1
y2[1] = t0 + x2[1] * t1
plt.plot(x2, y2, 'r')

"""
Метод polyfit
"""
x1,y1 = [0,22.5],[0,25]
numpy_x = np.array(x)
numpy_y = np.array(y)
numpy_t1, numpy_t0 = np.polyfit(numpy_x, numpy_y, 1)

num_y1 = [0, 0]
num_y1[0] = numpy_t0 + x1[0] * numpy_t1
num_y1[1] = numpy_t0 + x1[1] * numpy_t1
plt.plot(x1, num_y1, 'b')

print(numpy_t0, numpy_t1)


fig1 = plt.plot(x, y, 'g*') 
fig2 = plt.plot(x1, y1, 'y')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv


def sq_error(X,Y,f_x=None): 
    squared_error = []; 
    for i in range(len(X)): 
        squared_error.append((f_x(X[i])-Y[i])**2) 
    return sum(squared_error)

print(sq_error(x,y,fy)) 

data = pd.read_csv('web_traffic.tsv', sep='\t', header=None)

X, Y  = data[0], data[1]

x = list(X)
y = list(Y)

"""
   Фильтр
"""
for i in range(len(y)):
    if math.isnan(y[i]):
        y[i] = 0

"""
   Визуализация точек
"""
plt.plot(x, y, 'g*')
    
    
numpy_x = np.array(x)
numpy_y = np.array(y)

"""
    Подбор коэффициентов с помощью polyfit
"""
th0, th1 = np.polyfit(numpy_x, numpy_y, 1)
th2, th3, th4 = np.polyfit(numpy_x, numpy_y, 2)
th5, th6, th7, th8 = np.polyfit(numpy_x, numpy_y, 3)
th9, th10, th11, th12, th13 = np.polyfit(numpy_x, numpy_y, 4)
th14, th15, th16, th17, th18, th19 = np.polyfit(numpy_x, numpy_y, 5)

f1 = lambda x: th0*x + th1
f2 = lambda x: th2*x**2 + th3*x + th4
f3 = lambda x: th5*x**3 + th6*x**2 + th7*x + th8
f4 = lambda x: th9*x**4 + th10*x**3 + th11*x**2 + th12*x + th13
f5 = lambda x: th14*x**5 + th15*x**4 + th16*x**3 + th17*x**2 + th18*x + th19

result_one = sq_error(x, y, f1)
result_two = sq_error(x, y, f2)
result_three = sq_error(x, y, f3)
result_four = sq_error(x, y, f4)
result_five = sq_error(x, y, f5)

print("Среднее квадратичное отклонение 1: ", result_one)
print("Среднее квадратичное отклонение 2: ", result_two)
print("Среднее квадратичное отклонение 3: ", result_three)
print("Среднее квадратичное отклонение 4: ", result_four)
print("Среднее квадратичное отклонение 5: ", result_five)

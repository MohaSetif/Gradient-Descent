import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(x, y, w, b, learning_rate, N):
    dw = 0.0
    db = 0.0

    for xi, yi in zip(x, y):
        dw += -2 * (yi - (w*xi + b)) * xi
        db += -2 * (yi - (w*xi + b))
    
    w = w - learning_rate * (1/N) * dw
    b = b - learning_rate * (1/N) * db

    return w, b

data = pd.read_csv('Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
w = 0.0
b = 0.0
learning_rate = 0.01
N = x.shape[0]

for epoch in range(400):
    yhat = w*x + b
    w, b = gradientDescent(x, y, w, b, learning_rate, N)
    loss = np.sum((y - yhat)**2, axis=0) / N
    print(f'{epoch} loss is {loss}, parameters w: {w}, b: {b}')


plt.scatter(x, y, color='red')
plt.plot(x, yhat, color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Years of experience") 
plt.ylabel("Salaries") 
plt.show()

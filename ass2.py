import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

candy_data = pd.read_csv('candy-data.csv')
candy_data['winpercent'] = candy_data['winpercent'] / 100

traning_data = candy_data.sample(frac=0.9)
test_index = np.array(set(candy_data.index) - set(traning_data.index))
test_data = candy_data.loc[test_index, :]
x_columns = [
    'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer',
    'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent', 'winpercent'
]
y_columns = ['chocolate']
x_train = traning_data.loc[:, x_columns].values
y_train = traning_data.loc[:, y_columns].values
x_test = test_data.loc[:, x_columns].values
y_test = test_data.loc[:, y_columns].values
print(f'x_train shape = {x_train.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'x_test shape = {x_test.shape}')
print(f'y_test shape = {y_test.shape}')

g = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))
g_prime = np.vectorize(lambda z: g(z) * (1 - g(z)))


def cost(x, y, theta):
    Z = np.matmul(x, theta)
    h = g(Z)
    J = np.multiply(y, h) + np.multiply((1 - y), (1 - h))
    J = np.mean(J ** 2) / 2
    return J


def CostDerivative(X, Y, theta):
    t = X.shape[0]
    Z = np.matmul(X, theta)
    dJt_dh = 2 * Y - 1
    dh_dZ = g_prime(Z)
    # dZ_dTheta = X
    dJt_dZ = np.multiply(dJt_dh, dh_dZ)
    dJt_dTheta = np.matmul(np.tile(dJt_dZ.T, (t, 1)), X)
    dJ_dTheta = np.mean(dJt_dTheta, 0).T / 2
    return dJ_dTheta


def predict(X, theta):
    t = X.shape[0]
    X = np.c_[np.ones(t), X]  # Add a column of ones to x
    y = np.matmul(X, theta.T).T
    y = g(y)
    return np.round(1 - y)


def GradientDescent(X, Y, leraning_rate):
    error_tolerance = 0.01 * leraning_rate
    max_epoch = 1000
    t = X.shape[0]
    X = np.c_[np.ones(t), X]  # Add a column of ones to x
    m = X.shape[1]
    theta = np.zeros(shape=(m, 1))
    J_History = []
    while True:
        dJ_dTheta = CostDerivative(X, Y, theta)
        theta = theta - leraning_rate * dJ_dTheta.T.reshape(-1, 1)
        J_History.append(cost(X, Y, theta))
        if len(J_History) <= 1:
            continue
        if abs(J_History[-1] - J_History[-2]) > error_tolerance and J_History[-1] >= J_History[-2]:
            break
        if len(J_History) >= max_epoch:
            break

    theta = theta.T
    return theta, J_History


theta, JHistory = GradientDescent(x_train, y_train, 0.001)
print(theta)
plt.semilogy(JHistory)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

y_pred=predict(x_train, theta).reshape(-1,1)
cm = confusion_matrix(y_train,y_pred )
plt.imshow(cm)
y_pred=predict(x_test, theta).reshape(-1,1)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm)

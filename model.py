# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

# %%
# génération de dataset
X, y = make_blobs(n_samples=100, n_features=2, centers=2)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:,0], X[:, 1], c=y)
plt.show()

# %%
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)  # generation des poids W
    b = np.random.randn(1)  # generation du bias b
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b    # fonction d'aggregation
    A = 1 / (1 + np.exp(-Z))    # fonction d'activation
    return A

def log_loss(A, y): # cost function
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)    # variance des poids W
    db = 1 / len(y) * np.sum(A - y) # variance du bias b
    return (dW, db)
    
def update(dW, db, W, b, learning_rate):    # learning_rate : alpha
    W = W - learning_rate * dW  # nouveaux poids W
    b = b - learning_rate * db  # nouveau bias b
    return (W, b)


def neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)

    loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        l = log_loss(A, y)
        loss.append(l)
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    plt.plot(loss)
    plt.show()

# %%
neuron(X,y)
# %%

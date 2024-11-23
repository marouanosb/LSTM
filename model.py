# %%    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utilities import *

# %%
# load de dataset
X_train, y_train, X_test, y_test = load_data()
# show dataset
plt.figure(figsize=(16, 8))
for i in range (1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i])
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()

# %%
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)  # generation des poids W
    b = np.random.randn(1)  # generation du bias b
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b    # fonction d'aggregation
    A = 1 / (1 + np.exp(-Z))    # fonction d'activation sigmoide
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

def predict(X, W, b):
    A = model(X, W, b)
    print(A)
    return A >= 0.5 # si l'activation est supérieur à 0.5

def neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)
    loss = []

    # training
    for i in range(n_iter):
        A = model(X, W, b)
        l = log_loss(A, y)
        loss.append(l)
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    # prediction
    y_pred = predict(X, W, b)
    print("accuracy: {}", accuracy_score(y,y_pred))

    plt.plot(loss)
    plt.show()

    return (W, b) # pour sauvegarder les parametre optimaux

# %%
W, b = neuron(X,y)

# une entrée pour tester
new_item = np.array([np.random.rand(), np.random.rand()])

# tracer la droite qui sépare
x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x1 = (-W[0] * x0 - b) / W[1]
plt.plot(x0, x1, c="orange") 

plt.scatter(X[:,0], X[:, 1], c=y)
plt.scatter(new_item[0], new_item[1], c='r')
plt.show()

predict(new_item, W, b)

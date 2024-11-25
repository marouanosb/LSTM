# %%    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utilities import *
from tqdm import tqdm

# %%
# load de dataset
X_train, y_train, X_test, y_test = load_data()
# show
plt.figure(figsize=(16, 8))
for i in range (1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i])
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()

# pre processing
# flatten dataset
X_train_reshape = X_train.reshape(X_train.shape[0], -1) # or .reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_train_reshape = X_train_reshape / X_train.max()   # normaliser le dataset pour qu'il soit entre [0; 1]

X_test_reshape = X_test.reshape(X_test.shape[0], -1)
X_test_reshape = X_test_reshape / X_train.max()


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
    epsilon = 1e-15 # pour éviter d'avoir log(0)
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
    
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
    return A >= 0.5 # si l'activation est supérieur à 0.5

def neuron(X, y, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)
    loss = []
    acc = []
    loss_test = []
    acc_test = []

    # training
    for i in tqdm(range(n_iter)):
        A = model(X, W, b)

        if i % 10 == 0:
            # cost
            l = log_loss(A, y)
            loss.append(l)
            # acc prediction
            y_pred = predict(X, W, b)   
            acc.append(accuracy_score(y,y_pred))

            # cost
            A_test = model(X_test, W, b)
            l_test = log_loss(A_test, y_test)
            loss_test.append(l_test)
            # acc prediction
            y_pred = predict(X_test, W, b)   
            acc_test.append(accuracy_score(y_test,y_pred))

        # update weights
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='TRAIN LOSS')
    plt.plot(loss_test, label='TEST LOSS')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(acc, label='TRAIN ACC')
    plt.plot(acc_test, label='TEST ACC')
    plt.legend()
    plt.show()

    return (W, b) # pour sauvegarder les parametre optimaux

# %%
W, b = neuron(X_train_reshape, y_train, X_test_reshape, y_test, learning_rate=0.01, n_iter=10000)


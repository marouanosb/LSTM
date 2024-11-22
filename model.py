# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# %%
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

plt.scatter(X[:, 0],X[:,1], c=y)
plt.show()

# %%
#def Initialisation(X):

#def Model(X, W, b):

#def Cost(A, y)
    
#def Gradients(A, X, y)
    
#def Update(W, b, dW, db)

# %%

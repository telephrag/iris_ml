
import numpy as np
from random import seed

class AdalineSGD(object):
    eta: float # speed of learning
    n_iter: int # iteration over training dataset
    shuffle: bool = True # weather to shuffle dataset before fitting
    rng_seed: int = None # RNG's seed, used for shuffling
    w_initialized: bool = None

    w_ = [] # weights after fitting
    cost_ = [] # ???


    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)


    # train model by fiting data into categories
    # x -- contains pairs {sepals_length, petal_length} that needs to be "fitted" into category
    # y -- contains proofing data to check for correctness of fit
    def fit(self, X, y):
        self._initialize_weights(X.shape[1]) # X.shape[1] ~ amount of pairs of traits in X
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self


    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, target)
        
        return self


    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]


    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True


    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        # are -1 and 1 ids of classes?
        return np.where(self.activation(X) >= 0.0, 1, -1) 

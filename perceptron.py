import numpy as np

class Perceptron(object):
    eta : float # speed of learning [0.0..1.0]
    n_iter : int # amount of passes on training data

    w_ = [] # weight coefs after fitting
    errors_ = [] # amount of cases of false classification in each age

    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # fiting data into categories
    # x -- contains pairs {sepals_length, petal_length} that needs to be "fitted" into category
    # y -- contains proofing data to check for correctness of fit ("setosa" || "versicolor")
    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # xi -- pair of numbers representing 
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors      += int(update != 0.0)
            self.errors_.append(errors)
        
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)
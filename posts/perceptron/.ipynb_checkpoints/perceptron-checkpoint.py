import numpy as np
import random as rnd

class Perceptron:
    def fit(self, X, y, max_steps):
        # m = number of observations
        # n = number of features
        m, n = X.shape
        p = len(y)
        
        #weight tilda and X tilda that has one extra term
        w_tilda = np.random.rand(n+1)
        X_tilda = np.insert(X, n, 1, axis = 1)

        loop = 0
        history = []
        y_hat = 0
        accuracy = 0
        
        while loop < max_steps and accuracy != 1: 
            i = rnd.randint(0, p-1)
            y_tilda_i = 2*y[i] - 1
            w_tilda = w_tilda + (y_tilda_i * np.dot(w_tilda, X_tilda[i]) < 0) * y_tilda_i * X_tilda[i]
            y_hat = 1*(np.dot(X_tilda, w_tilda) >= 0)
            accuracy = np.average(y_hat == y)
            loop += 1
            history.append(accuracy)           
        
        self.w = w_tilda
        self.history = history
        
    def predict(self, X):
        m, n = X.shape
        X_tilda = np.insert(X, n, 1, axis = 1)
        return 1*(np.dot(X_tilda, self.w) >= 0)
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.average(y_hat == y)
import numpy as np

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)

class LinearRegression():
    def fit_analytic(self, X, y):
        '''
        X = feature matrix
        y = target vector
        alpha = learning rate, with default value 0.01
        max_epochs = the maximum number of loops, default 1000
        
        this function does not return anything, but assigns to the LinearRegression object
        an optimal weight via the analytic formula
        '''
        # n = number of observations
        # p = number of features
        n, p = X.shape
        
        X = pad(X)        
        self.w = np.linalg.inv(X.T@X)@X.T@y
    
    def fit_gradient(self, X, y, max_iter=5000, alpha=0.1):
        '''
        X = feature matrix
        y = target vector
        alpha = learning rate, with default value 0.1
        max_iter = the maximum number of loops, default 5000
        
        this function does not return anything, but assigns to the LinearRegression object
        a weight via gradient descent
        '''
        # n = number of observations
        # p = number of features
        n, p = X.shape
        
        w = np.random.rand(p+1).reshape(-1,1)
        X1 = pad(X)
        done = False
        i = 0
        prev_loss = np.inf
        loss_history = [] # to keep track of the loss history
        score_history = []
        y_bar = np.average(y)
        P = X1.T@X1
        q = X1.T@y
        
        while not done and i < max_iter:
            w -= 2*alpha*(P@w-q)/n
            i += 1
            new_loss = .5*np.sum((X1@w - y)**2)
            loss_history.append(new_loss) # update loss
            y_hat = X1@w
            score = 1 - np.sum((y_hat - y)**2)/np.sum((y - y_bar)**2)
            score_history.append(score)
            
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
        self.w = w
        self.loss_history = loss_history
        self.score_history = score_history

    def predict(self, X):
        X1 = pad(X)
        return X1@self.w
    
    def score(self, X, y):
        y_bar = np.average(y)
        y_hat = self.predict(X)
        return 1 - np.sum((y_hat - y)**2)/np.sum((y - y_bar)**2)
        
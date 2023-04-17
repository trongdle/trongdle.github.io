import numpy as np
import random as rnd

def sigmoid(z):
    # the sigmoid function
    return 1 / (1 + np.exp(-z))
    
def logistic_loss(y_hat, y):
    # the logistic loss function
    return -y*np.log(sigmoid(y_hat)) - (1-y)*np.log(1-sigmoid(y_hat))

class LogisticRegression():
    
    def fit(self, X, y, alpha=0.01, max_epochs=1000):
        '''
        X = feature matrix
        y = target vector
        alpha = learning rate, with default value 0.01
        max_epochs = the maximum number of loops, default 1000
        
        this function does not return anything, but assigns to the LogisticRegression object
        a weight, a loss history, and a score evolution
        '''
        # n = number of observations
        # p = number of features
        n, p = X.shape
         
        # weight tilda and X tilda that has one extra term
        w_tilda = np.random.rand(p+1)
        X_tilda = np.append(X, np.ones((X.shape[0], 1)), 1)

        done = False       # initialize for while loop
        prev_loss = np.inf # handy way to start off the loss

        loss_history = [] # to keep track of the loss history
        score_history = [] # to keep track of the score evolution
        
        epoch = 0 # to keep track of how many loops have been implemented
        
        # loop until we get the best weight or until we reach max number of loops
        while not done and epoch < max_epochs: 
            y_hat = np.dot(X_tilda, w_tilda) #inner product between X and w
            deltaL = 1/n*(sigmoid(y_hat) - y)@X_tilda #deltaL as given by the formula
            w_tilda -= alpha*deltaL #update the weight
            new_loss = logistic_loss(y_hat, y).mean() # compute loss
            y_pred = 1*(np.dot(X_tilda, w_tilda) >= 0) #prediction vector given by the weight
            score_history.append(np.average(y_pred == y)) # update score
            loss_history.append(new_loss) # update loss
            epoch += 1
            
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
        self.w = w_tilda
        self.loss_history = loss_history
        self.score_history = score_history
        
    def predict(self, X):
        '''
        X = feature matrix
        returns a prediction vector
        '''
        X_tilda = np.append(X, np.ones((X.shape[0], 1)), 1)
        return 1*(np.dot(X_tilda, self.w) >= 0)

    def loss(self, X, y):
        '''
        X = feature matrix
        y = target vector
        returns the overall loss (empirical risk) of the current weights on X and y
        '''
        y_hat = self.predict(X)
        return logistic_loss(y_hat, y).mean()
    
    def score(self, X, y):
        '''
        X = feature matrix
        y = target vector
        returns the accuracy of the predictions as a number between 0 and 1, with 1 corresponding to perfect classification
        '''
        y_hat = self.predict(X)
        return np.average(y_hat == y)
    
    def fit_stochastic(self, X, y, alpha=0.01, m_epochs=100, batch_size=1, momentum=False, beta=0):
        '''
        X = feature matrix
        y = target vector
        alpha = learning rate, default 0.01
        m_epochs = max number of loops, default 100
        batch_size = the size of each batch that X and y are divided into
        momentum = user decides whether to implement momentum, default False
        beta = momentum learning rate, default 0 if momentum == False, and 0.8 otherwise
        '''
        # n = number of samples; p = number of features
        n, p = X.shape
        
        # add 1 column to X and 1 component to w
        # w_tilda_prev, w_tilda and w_tilda_new correspond to w_{k-1}, w_{k} and w_{k+1} in the momentum stochastic gradient descent algorithm
        X_tilda = np.append(X, np.ones((X.shape[0], 1)), 1)
        w_tilda = np.random.rand(p+1)
        w_tilda_prev = np.random.rand(p+1)
        w_tilda_new = np.random.rand(p+1)
        
        # keep track of the loss and score
        prev_loss = np.inf # handy way to start off the loss
        loss_history = []
        score_history = []
        
        # if the user wants to use momentum, beta is set to 0.8
        if momentum:
            beta = 0.8
            
        # keep track of the loop implementation
        epoch = 0
        done = False

        while epoch < m_epochs and not done:
            #shuffle the order of X and y
            order = np.arange(n)
            np.random.shuffle(order)
            
            #divide X and y into batches corresponding to the shuffled order above
            for batch in np.array_split(order, n // batch_size + 1):
                X_batch = X_tilda[batch,:]
                y_batch = y[batch]
                y_hat = np.dot(X_batch, w_tilda)
                deltaL = 1/batch_size*(sigmoid(y_hat) - y_batch)@X_batch # compute delta L
                w_tilda_new = w_tilda - alpha * deltaL + beta*(w_tilda - w_tilda_prev) # this is w_{k+1} for this loop
                w_tilda_prev = w_tilda # this is w_{k-1} for next loop
                w_tilda = w_tilda_new # this is w_{k} for next loop
            
            y_pred = 1*(np.dot(X_tilda, w_tilda) >= 0) # prediction vector
            y_hat = np.dot(X_tilda, w_tilda) # inner product
            new_loss = np.nan_to_num(logistic_loss(y_hat, y)).mean() # compute loss
            loss_history.append(new_loss)
            score_history.append(np.average(y_pred == y))
            
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
            
            epoch += 1 #update loop
                
        self.w = w_tilda
        self.loss_history = loss_history
        self.score_history = score_history
    
    def fit_adam(self, X, y, batch_size=1, alpha=0.001, beta1=0.9, beta2=0.999, w_0 = None, epsilon=1e-8, m_epochs=10000):
        
        # n = number of samples; p = number of features
        n, p = X.shape
        
        X_tilda = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        if w_0 is None:
            w = np.random.rand(p+1)
        else:
            w = w_0 # w_0 is theta_0 from the paper; w is theta
              
        # keep track of the loss and score
        prev_loss = np.inf # handy way to start off the loss
        loss_history = []
        score_history = []
        
        done = False
        epoch = 0
            
        while epoch < m_epochs and not done:
            #shuffle the order of X and y
            order = np.arange(n)
            np.random.shuffle(order)
            m = 0 # Initialize 1st moment vector
            v = 0 # Initialize 2nd moment vector
            t = 0 # Initialize timestep
            
            #divide X and y into batches corresponding to the shuffled order above
            for batch in np.array_split(order, n // batch_size + 1):
                t += 1
                X_batch = X_tilda[batch,:]
                y_batch = y[batch]
                y_hat = np.dot(X_batch, w)
                deltaL = 1/batch_size*(sigmoid(y_hat) - y_batch)@X_batch # L(w) is equivalent to f(theta); this step is to get gt
                m = m*beta1 + (1 - beta1) * deltaL      # Update biased first moment estimate
                v = v*beta2 + (1 - beta2) * deltaL**2   # Update biased first moment estimate
                m_hat = m/(1 - beta1**t)            # Compute bias-corrected first moment estimate
                v_hat = v/(1 - beta2**t)            # Compute bias-corrected second raw moment estimate
                w -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)  # Update parameters
                
            y_pred = 1*(np.dot(X_tilda, w) >= 0) # prediction vector
            y_hat = np.dot(X_tilda, w) # inner product
            new_loss = logistic_loss(y_hat, y).mean() # compute loss
            loss_history.append(new_loss)
            score_history.append(np.average(y_pred == y))
            
            if np.isclose(new_loss, prev_loss):          
                done = True
            else:
                prev_loss = new_loss
            
            epoch += 1 #update loop
                    
        self.w = w
        self.loss_history = loss_history
        self.score_history = score_history
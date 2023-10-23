import numpy as np


class LogiReg:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=0, random_state=42):
        """
        :param learning_rate: learning rate value
        :param n_steps: number of epochs around gd
        :param n_features: number of features involved in regression
        :param lmd: regularization factor

        lmd_ is an array useful when is necessary compute theta's update with regularization factor
        """
        np.random.seed(random_state)
        self.theta = np.random.rand(n_features)
        print(self.theta.shape[0])
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.lmd = lmd

    def fit(self, x, y, x_val, y_val):
        """
        Perform fitting on training set and Regularization on validation set
        :param x: training samples with bias
        :param y: training target values
        :param x_val: validation samples with bias
        :param y_val: validation target values
        :param Regularization: if set true finds the best lambda for hyperparamenter tuning
        return: Theta Values for Hypothesis Function (h)
        """
        m=len(x)
        for i in range(0, self.n_steps):
            preds = 1/(1 + np.exp(np.dot(-self.theta, x.T)))  # è il prodotto interno tra vettori
            error = preds - y
            #h: 455 - 30
            """Performing Gradient Descent Algorithm"""
            self.theta = self.theta - (self.learning_rate * (1 / m) * np.dot(x.T, error))
        return self.theta


    def predict(self, x):
        """
            perform a complete prediction about X samples
            :param x: test sample with shape (m, n_features)
            :return: prediction wrt X sample. The shape of return array is (m,)
            """
        prediction = 1/(1 + np.exp(np.dot(-self.theta, x.T)))

        result = np.zeros(len(prediction))
        for i in range(0,len(prediction)):
            if prediction[i] > 0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result

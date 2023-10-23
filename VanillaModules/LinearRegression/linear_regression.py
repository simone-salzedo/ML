import numpy as np

np.random.seed(123)


class LinearRegression:
    """
    Class to perform learning for a linear regression. This class has all methods to be trained with different strategies
    and one method to produce a full prediction based on input samples. Moreover, this one is equipped by one method to
    measure performance and another method to build learning curves
    """
    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=0):
        """
        :param learning_rate: learning rate value
        :param n_steps: number of epochs around gd
        :param n_features: number of features involved in regression
        :param lmd: regularization factor

        lmd_ is an array useful when is necessary compute theta's update with regularization factor
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = lmd
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0

    def fit(self, X, y, X_val, y_val):
        """
        apply gradient descent in full batch mode, without regularization, to training samples and return evolution
        history of train and validation cost.
        :param X: training samples with bias
        :param y: training target values
        :param X_val: validation samples with bias
        :param y_val: validation target values
        :return: history of evolution about cost and theta during training steps and, cost during validation phase
        """
        m = len(X)
        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)
            preds_val = np.dot(X_val, self.theta)

            error = preds - y
            error_val = preds_val - y_val

            self.theta = self.theta - (self.learning_rate * (1/m) * np.dot(X.T, error)) - 2*self.lmd*self.theta
            theta_history[step, :] = self.theta.T
            cost_history[step] = 1/(2*m) * np.dot(error.T, error) + self.lmd*np.sum(np.square(self.theta))
            cost_history_val[step] = 1 / (2 * m) * np.dot(error_val.T, error_val) + self.lmd*np.sum(np.square(self.theta))

        return cost_history, cost_history_val, theta_history

    def predict(self, X):
        """
        perform a complete prediction about X samples
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        Xpred = np.c_[np.ones(X.shape[0]), X]

        return np.dot(Xpred, self.theta)

    def cost_grid(self, X, Y, A, B, first_dim, second_dim):
        """
        perform a complete prediction about X samples
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        result = np.zeros((100,100))

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                temp_theta = self.theta[:]
                temp_theta[first_dim] = A[r,c]
                temp_theta[second_dim] = B[r,c]
                result[r,c] = np.average((X @ temp_theta - Y)**2)*0.5

        return result

    def compute_performance(self, X, y):
        """
        compute performance for linear regression model
        :param X: test sample with shape (m, n_features)
        :param y: ground truth (correct) target values shape (m,)
        :return: a dictionary with name of specific metric as key and specific performance as value
        """
        preds = self.predict(X)

        mae = self._mean_absolute_error(preds, y)
        mape = self._mean_absolute_percentage_error(preds, y)
        mpe = self._mean_percentage_error(preds, y)
        mse = self._mean_squared_error(preds, y)
        rmse = self._root_mean_squared_error(preds, y)
        r2 = self._r_2(preds, y)
        return {'mae': mae, 'mape': mape, 'mpe': mpe, 'mse': mse, 'rmse': rmse, 'r2': r2}

    def _mean_absolute_error(self, pred, y):

        output_errors = np.abs(pred - y)
        return np.average(output_errors)

    def _mean_squared_error(self, pred, y):

        output_errors = (pred - y)**2
        return np.average(output_errors)

    def _root_mean_squared_error(self, pred, y):

        return np.sqrt(self._mean_squared_error(pred, y))

    def _mean_absolute_percentage_error(self, pred, y):

        output_errors = np.abs((pred - y)/y)
        return np.average(output_errors)

    def _mean_percentage_error(self, pred, y):

        output_errors = (pred - y)/y
        return np.average(output_errors)*100

    def _r_2(self, pred, y):

        sst = np.sum((y - y.mean()) ** 2)
        ssr = np.sum((pred - y) ** 2)

        r2 = 1 - (ssr / sst)
        return r2

    def learning_curves(self, X, y, X_val, y_val):

        m = len(X)
        cost_history = np.zeros(m)
        cost_history_val = np.zeros(m)

        for i in range(m):
            c_h, c_h_v, _ = self.fit(X[:i+1], y[:i+1], X_val, y_val)
            cost_history[i] = c_h[-1]
            cost_history_val[i] = c_h_v[-1]

        return cost_history, cost_history_val
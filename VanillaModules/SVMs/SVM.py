import numpy as np

class SupportVectorMachine:
    def __init__(self, kernel_trick=None, C=0, GaussianVariance=None, Gamma=None, b=None, c=None, d=None):
        self.d = d
        self.c = c
        self.b = b
        self.kernel_trick = kernel_trick
        self.Gamma = Gamma
        self.GaussianVariance = GaussianVariance
        self.C = C
        self.w = None

        match kernel_trick:
            case "linear":
                print('linear set, eventual kernel parameters ignored')

            case "polynomial":
                if self.d is None:
                    raise Exception("d must be not None, assign a value to it or change kernel trick")

            case "rbf":
                if self.Gamma is None:
                    raise Exception("Gamma must be not None, assign a value to it or change kernel trick")

            case "mlp":
                if self.b is None:
                    raise Exception("b must be not None, assign a value to it or change kernel trick")
                if self.c is None:
                    raise Exception("c must be not None, assign a value to it or change kernel trick")

            case "grbf":
                if self.GaussianVariance is None:
                    raise Exception("GaussianVariance must be not None, assign a value to it or change kernel trick")
            case _:
                raise Exception("Invalid Kernel Trick, you can choose between: linear, polynomial, rbf, mlp, grbf")

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X] #Adding Bias column to sample x features matrix
        self.w = np.zeros(X.shape[1] - 1) #row vector of weights
        print('Number of Features: ', X.shape[1] - 1)
        self.w =

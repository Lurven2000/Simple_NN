import numpy as np

class Dense(object):
    def __init__(self,n_inputs,n_outputs,debug = False):
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.debug = debug
        self.learning_rate = 0.05

        #choose random weights
        #Inputs match to rows. Outputs match to columns.
        #Add one to n_inputs to account for the bias term.
        self.initial_weight_scale = 1
        self.weights = self.initial_weight_scale * (np.random.sample(
            size = (self.n_inputs + 1, self.n_outputs + 1)) * 2 - 1)
        self.w_grad = np.zeros((n_inputs + 1, n_outputs))
        self.x = np.zeros((1, self.n_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))

    def forward_prop(self, inputs):
        bias = np.ones((1,1))
        self.x = np.concatenate((inputs,bias), axis = 1)
        self.y = self.x @ self.weights #matrix multiplication of x and weights
        return self.y
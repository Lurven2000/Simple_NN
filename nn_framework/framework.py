import numpy as np
import os
import matplotlib.pyplot as plt

class ANN(object):
    def __init__(self, model=None,expected_range = (-1,1)):
        self.layers = model
        self.n_iter_train = int(1e8)
        self.n_iter_eval = int(1e6)
        self.expected_range = expected_range
        self.error_history = []
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0

        self.report_path = "reports"
        self.report_name = "performance_history.png"


    def train(self, training_set):
        for i in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_prop(x)
            print(y)
    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_eval):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_prop(x)

    def normalize(self,data):
        return np.interp(data,[self.expected_range[0],self.expected_range[1]],[-.5,.5])

    def denormalize(self,data):
        return np.interp(data, [-.5, .5], [self.expected_range[0], self.expected_range[1]])

    def forward_prop(self,x):
        y = x.ravel()[np.newaxis,:]  # convert the inputs into a 2D array
        for layer in self.layers:
            y = layer.forward_prop(y)  # recursively call the forward prop function
        return y.ravel()
import data_loader_two_by_two as data_loader
import nn_framework.framework as framework
import nn_framework.layer as layer

training_set, test_set = data_loader.get_data_sets()

sample = next(training_set())
input_value_range = (0,1)
n_pixels = sample.shape[0] * sample.shape[1]

n_nodes = [n_pixels, n_pixels]
model = [layer.Dense(n_nodes[0], n_nodes[1])]

autoencoder = framework.ANN(model = model,
                            expected_range = input_value_range)
autoencoder.train(training_set)
autoencoder.evaluate(test_set)
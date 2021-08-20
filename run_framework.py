import data_loader_two_by_two as data_loader
import nn_framework.framework as framework
import nn_framework.layer as layer
import nn_framework.activation as activation

N_NODES = [7, 4, 6]  # List of numbers of nodes in the hidden layer

training_set, test_set = data_loader.get_data_sets()

sample = next(training_set())
input_value_range = (0,1)
n_pixels = sample.shape[0] * sample.shape[1]

n_nodes = [n_pixels] + N_NODES + [n_pixels]
model = []
for i_layer in range(len(n_nodes) - 1):
    model.append(layer.Dense(
        n_nodes[i_layer],
        n_nodes[i_layer],
        activation.TanH
    ))

autoencoder = framework.ANN(model = model,
                            expected_range = input_value_range)
autoencoder.train(training_set)
autoencoder.evaluate(test_set)
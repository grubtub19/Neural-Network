import math
import random
import statistics
from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    f = 1 / (1 + np.exp(-z))
    return f * (1 - f)


class Network:

    def __init__(self, training_data: pd.DataFrame, label_column: str, unique_labels: np.ndarray, alpha: float, layer_sizes: List[int]):
        # Training speed
        self.alpha = alpha
        self.num_epochs = 100

        # a list containing the number of nodes in each layer ( input, hidden, output )
        self.layer_sizes: List[Any] = layer_sizes

        # Number of input nodes is the number of features (- 1 to exclude the label column)
        self.layer_sizes.insert(0, len(training_data.columns) - 1)

        # Number of output nodes are the number of unique labels
        self.layer_sizes.append(len(unique_labels))
        print("Layer Sizes: " + str(layer_sizes))

        # Name of the column that contains the label
        self.label_column = label_column

        self.unique_labels: List[Any] = list(unique_labels)

        # Scale inputs and save the scaler to use when testing/classifying
        self.input_scaler = StandardScaler()
        labels = training_data[self.label_column].copy()
        features = training_data.drop(self.label_column, axis=1)

        scaled_features = pd.DataFrame(self.input_scaler.fit_transform(features),
                                          columns=features.columns, index=features.index)

        # A list of matrices where weights[0] is the weights of the connects between
        # rows are weights coming off each node directed in different directions
        self.weights: List[np.ndarray] = [np.random.rand(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.fit(scaled_features, labels)

    def fit(self, feature_data: pd.DataFrame, labels: pd.Series):

        for epoch in range(self.num_epochs):
            idx = np.random.permutation(feature_data.index)
            shuffled_feature_data = feature_data.reindex(idx)
            shuffled_labels = labels.reindex(idx)
            errors = list()
            for (_, features), (_, label) in zip(shuffled_feature_data.iterrows(), shuffled_labels.iteritems()):
                input_activations = features.values

                # Forward propagate and return lists of vectors for both inputs and activations in each layer (excluding the input layer)
                inputs, activations = self.forward_propagate(input_activations)

                # Identify the index of the output node corresponds with the answer
                answer_node = self.unique_labels.index(label)

                # Create a vector of zeros other than the answer node
                actual_answer_vector = np.zeros(len(self.unique_labels))
                actual_answer_vector[answer_node] = 1.0

                # Calculate the differences between the predicted and actual output vectors
                error = actual_answer_vector - activations[-1]
                errors.append(statistics.mean(error))

                # Calculate deltas for weights for the last hidden layer (between it and the output layer)
                deltas = list()
                #print("deltas[-1]: " + str(len(error * sigmoid_prime(inputs[-1]))))
                deltas.insert(0, error * sigmoid_prime(inputs[-1]))

                # Iterate backwards through hidden layers (len - 2 -> 1)
                for layer_i in reversed(range(0, len(self.layer_sizes) - 1)):
                    # Layer i is the previous layer (closer to input)
                    # Layer j is the next layer (closer to output)
                    # i the index for the weights matrix between i and j

                    # Feed the deltas from layer j backwards through the weights (deltas[0] because we are actively populating it)
                    # weighted_change is a vector corresponding to the ith weights matrix
                    weighted_change: np.ndarray = np.dot(self.weights[layer_i].transpose(), deltas[0])

                    # Calculate the sigmoid prime of the inputs of layer i
                    sigmoid_prime_inputs = sigmoid_prime(inputs[layer_i])
                    delta = np.multiply(sigmoid_prime_inputs, weighted_change)

                    # Add to beginning of deltas. Will eventually be at index layer_i
                    deltas.insert(0, delta)

                for layer_i in range(len(self.weights)):
                    #print("shape[" + str(layer_i) + "]: " + str(len(deltas[layer_i + 1])) + "," + str(len(activations[layer_i])))
                    thing = np.asarray(np.dot(np.asmatrix(deltas[layer_i + 1]).transpose(), np.asmatrix(activations[layer_i])))

                    self.weights[layer_i] = self.weights[layer_i] + self.alpha * thing

    def forward_propagate(self, inputs: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
        """
        Takes input activations, propagates forward, and returns output activations
        :param input_activations: List[float] Input Activations
        :return: List[float] Inputs for each layer including the input layer
        :return: List[float] Activations for each layer including the input layer
        """
        inputs_list = list()
        activations_list = list()

        # Input's activations are direct, so we don't need to sigmoid or anything
        inputs_list.append(inputs)
        activations_list.append(inputs)

        activations = inputs
        for weights in self.weights:

            input_vector = np.dot(weights, activations)

            inputs_list.append(input_vector)
            activations = sigmoid(input_vector)

            activations_list.append(activations)
        return inputs_list, activations_list

    def classify(self, data: pd.Series) -> Any:
        features = data.drop(self.label_column)
        scaled_features = pd.Series(self.input_scaler.transform([features])[0])
        activations = scaled_features
        for weights in self.weights:
            input_vector = np.dot(weights, activations)
            activations = sigmoid(input_vector)

        max_index = np.argmax(activations)
        return self.unique_labels[max_index]


##
# Runs a decision tree algorithm on a data file
# @param filename String The name of the file to read from that must be located in the same directory
# @param delimit String That is the delimiter for the items in a row
# @param label_col int Which column (starting from 0) is the one with the label
# @param remove_col array[int] The columns to remove
# @param header bool Whether or not there is a header
# @return float The accuracy of the k-NN prediction algorithm
#
def decide(filename, delimit, label_col, remove_col, header):
    # A Pandas DataFrame with all the data. Both training and testing
    data = pd.DataFrame()
    # Removes the first row if the columns are labeled
    if header:
        data = pd.read_csv(filename, delimiter=delimit, index_col=False)
    else:
        data = pd.read_csv(filename, delimiter=delimit, header=None, index_col=False)
    print(data)
    # Removes any rows that are not wanted
    if len(remove_col) > 0:
        data = data.drop(remove_col, axis=1)

        for col_num in remove_col:
            if label_col > col_num:
                label_col -= 1
            elif label_col == col_num:
                print("Cannot Delete the Label Column")

    # Split the data into train/test
    train, test = train_test_split(data, shuffle=False)

    print(data[label_col].unique())
    # Build the Tree
    net = Network(train, label_col, data[label_col].unique(), 0.1, [8, 4])
    accuracies = 0

    for _, entry in test.iterrows():
        prediction = net.classify(entry)
        if prediction == entry[label_col]:
            accuracies += 1
    accuracies /= len(test.index)
    return accuracies


if __name__ == "__main__":
    breast_accuracy = decide("breast-cancer-wisconsin.data", ",", 10, [0], False)
    print("Breast Accuracy: " + str(breast_accuracy))
    test_accuracy = decide("winequality-red.csv", ';', "quality", [], True)
    print("Wine Accuracy: " + str(test_accuracy))
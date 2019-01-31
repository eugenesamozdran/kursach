import numpy as np

class Perceptron:

    def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1) # creating array of zeros equal
                                                  # to the number of inputs + 1
           
    def predict(self, inputs): # activation function predicting the output
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            return 1
        else:
            return 0

    def train(self, training_inputs, labels): # weights update function
        for _ in range(self.epochs):
            # training_inputs is expected to be a list made up
            # of numpy vectors to be used as inputs by the predict method.
            # labels is expected to be a numpy array of
            # expected output values for each of the corresponding inputs
            # in the training_inputs list.
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Creating a list of 2500 zero arrays
training_inputs = [np.array([0]) for i in range(2500)]

with open("training_img_data.txt", "r") as f:
    # This is required to clean up the string from
    # unnecessary symbols
    labels = f.read().replace('[','').replace(']','').replace(' ','').split(',')

# Creating an array of expected outputs
labels = np.array([int(labels[i]) for i in range(0, len(labels))])

perceptron = Perceptron(2500)
perceptron.train(training_inputs, labels)

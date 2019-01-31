import numpy as np

class Perceptron:

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

training_inputs = [np.array([0]) for i in range(2500)]

with open("training_img_data.txt", "r") as f:
    labels = f.read().replace('[','').replace(']','').replace(' ','').split(',')

labels = np.array([int(labels[i]) for i in range(0, len(labels))])

perceptron = Perceptron(2500)
perceptron.train(training_inputs, labels)

##with open("testing_img_data.txt", "r") as f:
##    test_input = f.read().replace('[','').replace(']','').replace(' ','').split(',')
##
##test_input = np.array([int(test_input[i]) for i in range(len(test_input))])
##print(len(test_input))
##perceptron.predict(test_input)
#=> 1

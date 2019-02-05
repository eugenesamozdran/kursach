import numpy as np

class Perceptron:

    def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1) # creating array of zeros equal
                                                  # to the number of inputs + 1
           
    def predict(self, inputs): # activation function predicting the output
        summation = np.sum(np.dot(inputs, self.weights[1:])) + self.weights[0]
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

# Creating an array of 2500 zeroes
training_inputs = np.array([0 for i in range(2500)])

with open("training_img_data.txt", "r") as f:
    # This is required to clean up the string from
    # unnecessary symbols
    labels = f.read().replace('[','').replace(']','').replace(' ','').split(',')

# Creating an array of expected outputs
labels = np.array([int(labels[i]) for i in range(len(labels))])

# Creating Perceptron object and training it
perceptron = Perceptron(2500)
perceptron.train(training_inputs, labels)
print(len(perceptron.weights))

# Forming test image data for perceptron to try to predict it
with open("testing_img_data.txt", "r") as f:
    
    # Cleaning the string from the file again
    test_img_pixels = f.read().replace('[','').replace(']','').replace(' ','').split(',')

test_img_pixels = [int(test_img_pixels[i]) for i in range(len(test_img_pixels))]

# Creating an array that should be passed as an input to predict function for testing
test_input = np.array([test_img_pixels])

print(perceptron.predict(test_input))

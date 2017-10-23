
# coding: utf-8

# In[3]:

from numpy import exp, array, random, dot

random.seed(1)


# In[56]:

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2*random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def sigmoid_gradient(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
        
            error = training_set_outputs - output
        
            adjustment = dot(training_set_inputs.T, error*self.sigmoid_gradient(output))
        
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))


# In[67]:

neural_network = NeuralNetwork()
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
print "Random starting synaptic weights: "
print neural_network.synaptic_weights

neural_network.train(training_set_inputs,training_set_outputs,10000000)


# In[68]:

print "New synaptic weights after training: "
print neural_network.synaptic_weights


# In[66]:

print "Considering new situation [1, 0, 0] -> ?: "
print neural_network.think(array([1, 0, 0]))


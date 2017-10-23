
# coding: utf-8

# In[69]:

from numpy import exp, array, random, dot

random.seed(1)


# In[70]:

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


# In[75]:

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def sigmoid_gradient(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output_from_layer1, output_from_layer2 = self.think(training_set_inputs)
        
            layer2_error = training_set_outputs - output_from_layer2
            layer2_delta = layer2_error * self.sigmoid_gradient(output_from_layer2)
            
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.sigmoid_gradient(output_from_layer1)
            
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer1.T.dot(layer2_delta)
        
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def think(self, inputs):
        output_from_layer1 = self.sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2
    
    def print_weights(self):
        print "    Layer 1 (4 neurons, each with 3 inputs): "
        print self.layer1.synaptic_weights
        print "    Layer 2 (1 neuron, with 4 inputs):"
        print self.layer2.synaptic_weights


# In[79]:

random.seed(1)
layer1 = NeuronLayer(4, 3)
layer2 = NeuronLayer(1, 4)

neural_network = NeuralNetwork(layer1, layer2)
print "Stage 1) Random starting synaptic weights: "
neural_network.print_weights()


# In[81]:

training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T


# In[82]:

neural_network.train(training_set_inputs, training_set_outputs, 60000)
print "Stage 2) New synaptic weights after training: "
neural_network.print_weights()


# In[83]:

print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
hidden_state, output = neural_network.think(array([1, 1, 0]))
print output


#import vari
import numpy as np

#creation of the class NeuronLayer

class NeuronLayer:
  """
  Represents a layer in the neural network.
  """

  def __init__(self, n_input, n_neurons, activation, name):
    """
    :param int n_input: The input size for the considered layer
    :param int n_neurons: The number of neurons in the layer
    :param str activation: The activation function to use (it could be: sigmoid)
    :param str name: The name of the layer (it is useful for the print)

    by default, weigths and bias of the layer are random initialized 
    """
    np.random.seed(17)
    self.activation = activation
    self.bias = np.random.rand(n_neurons,1) 

    self.weights = np.random.rand(n_neurons, n_input)
    self.name = name if name else None


  def set_weights(self, weights):
    '''
    gives the possibility to change the weights of the considered layer.
    They were random initialized.
    '''

    self.weights = weights


  def set_bias(self, bias):
    '''
    gives the possibility to change the bias of the considered layer.
    It was random initialized.
    '''
    
    self.bias = bias  


  def check(self):
    '''
    shows the attributes of the layer.
    '''

    print("-------layer's name: "+  str(self.name) + "-------")
    print(" ")
    print(" Number of neurons: ", self.weights.shape[0])
    print(" Activation function: ", self.activation)
    print(" Weights: ", self.weights)
    print(" Bias: ", self.bias)
    print(" ")


  def apply_function(self, z):
    '''
    Applies the activation function
    :param z: the "total input" value for the layer to be activated.
    :return: the "activated" value according to the chosen function (i.e. the output of the layer)
    '''

    if self.activation == 'sigmoid':
      return 1 / (1 + np.exp(-z))
  
    
  def derivative_function(self, sigma):
    '''
    Calculate the derivative of the activation function
    :param sigma: the "total output" value to be derived.
    :return: the derivative of the activation function according to the chosen function
    '''

    if self.activation == 'sigmoid':
      return sigma * (1 - sigma)
    

  def calculate_output(self, inputs):
    self.inputs = inputs
    self.z =np.dot( self.weights, self.inputs.T)  + self.bias   #row = number of neurons layer, column = batch_size
                                                                #La colonna j-esima della matrice corrisponde alla zeta relativa al jesimo input.
                                                                #la iesima riga corrisponde alle zeta del iesimo neurone del layer.
    self.outputs = (self.apply_function(self.z)).T
    return self.outputs


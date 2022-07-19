import numpy as np
import matplotlib.pyplot as plt
from optimization import NeuronLayer as nl

class NeuralNetwork:
  """
  Represents the neural network.
  """

  def __init__(self):
    ''' 
    Creates the Neural Network object.
    It does not contain any layers.
    '''
    self.network = []


  def add(self, n_input, n_neurons, activation, name = None):
    '''
    Adds a new Layer
    :param layer: layer to add.
    '''
    self.network.append(nl.NeuronLayer(n_input, n_neurons, activation, name))


  def check(self):
    '''
    Shows the attributes of each layer of the considered neural network.
    '''
    print("Data input size: ", self.network[0].weights.shape[1])
    print(" ")
    for layer in self.network:
      layer.check()


  def train(self, X, y, learning_rate, alg_opt, batch_size, epochs, **kwargs):
    '''
    :param X: The data to use for the training
    :param y: The labels of the data to use for the training
    :param alg_opt: The optimization algorithm to use. (It could be GD )
    :param int batch_size: The batch size to use to compute the optimization algorithm.
    :param float learning_rate: The learning rate to use during the updating of the weights; a number between 0 and 1.
    :param int epochs: The number of epoches  
    '''
    #for each layer of the network create a tensor to store gradients of errors wrt weights and bias rispectively during the backpropagation 
    for layer in self.network:
      layer.batch_der_error_wrt_weights = np.zeros((batch_size, layer.weights.shape[0], layer.weights.shape[1]))
      layer.batch_der_error_wrt_bias = np.zeros((batch_size, layer.bias.shape[0], layer.bias.shape[1]))

    self.list_loss=[]
    self.list_acc=[]
    
    #iterate for each epoch
    for epoch in range(epochs):

      #shuffle dati
      X_shuf, y_shuf = self.shuffle_data(X,y)

      #divisione in batch
      X_list_batches, y_list_batches = self.create_batch(X_shuf, y_shuf, batch_size)

      #create tensor and matrix of zeros in case we use momentum algorithm. They will be useful during the upgrade of the weights
      if alg_opt == "momentum":
          for layer in self.network:
            layer.previous_velocity_weights = np.zeros(layer.weights.shape)
            layer.previous_velocity_bias = np.zeros(layer.bias.shape)

      elif alg_opt in ["RMSprop", "Adagrad"]:
         for layer in self.network:
            layer.cache_mean_weights = np.zeros(layer.weights.shape)
            layer.cache_mean_bias = np.zeros(layer.bias.shape)

      elif alg_opt == "Adam":
        self.iteration = epoch + 1 
        for layer in self.network:
          layer.previous_first_moment_weights = np.zeros(layer.weights.shape)
          layer.previous_second_moment_weights = np.zeros(layer.weights.shape)
          layer.previous_first_moment_bias = np.zeros(layer.bias.shape)
          layer.previous_second_moment_bias = np.zeros(layer.bias.shape)



      losses_batch=[]
      acc_batch=[]
      for X_batch, y_batch in zip(X_list_batches, y_list_batches):
        self.feedforward(X_batch)
        self.backprop(y_batch)
        self.update_weights(alg_opt, batch_size, learning_rate)
        predicted_label = self.feedforward(X_batch)
        loss = np.mean(np.square(y_batch - predicted_label), axis=0)
        losses_batch.append(loss[0])
        predicted_label = self.predict(X_batch)
        acc = self.accuracy(predicted_label, y_batch)
        acc_batch.append(acc)
      
      self.list_loss.append(losses_batch[-1])
      self.list_acc.append(acc_batch[-1])
      print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e, acc %f" % (batch_size, learning_rate, epoch, self.list_loss[-1], self.list_acc[-1]))


  def shuffle_data(self, a, b):
    '''
    Shuffles (randomizes the order of the elements in) the array a and the array b, mantaining the correct index.
    :param a: Array that needs to be shuffled
    :param b: Array that needs to be shuffled
    :return: The shuffled arrays.
    '''
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


  def create_batch(self, X, y, batch_size):
    '''
    Divides the data in batches, given the batch size.
    :param X: The data to be divided
    :param y: The labels to be divided
    :param int batch_size: The size of the batch
    :return: Two arrays that contains data organized in batch.
    '''
    n_batch = X.shape[0]//batch_size

    extra_data = X.shape[0] % batch_size
    
    if (extra_data != 0):
         X = X[:(-extra_data)]
         y = y[:(-extra_data)]
    X_batches = np.reshape(X,(n_batch, batch_size, X.shape[1]))
    y_batches = np.reshape(y,(n_batch, batch_size, y.shape[1]))

    return X_batches, y_batches

  
  def feedforward(self, X):
    '''
    Propagates one input sample throws the layers and gives the output of the whole network.
    :param X: Inputs data to be propagated.
    :return: the output of the last layer.
    '''
    for layer in self.network:
      X = layer.calculate_output(X)
    return X
  
  
  def backprop(self,y):
    '''
    Performs the backpropagation for a single sample.
    It computes the partial derivatives wrt weigths and wrt bias for each layer. (It does NOT update the weights)
    :param y: true label of the considered sample
    '''
    #backprop for output layer
    for l in range(len(self.network)-1, -1, -1):
      layer = self.network[l]
      

      if layer == self.network[-1]:
        delta_out = (y-layer.outputs) * layer.derivative_function(layer.outputs) #la riga iesima si riferisce ai delta del campione iesimo, per noi è matrice (batch_size,1)
                                                                                 #la colonna jesima si riferisce ai delta del jesimo neurone
                                                                                 #es. l'elemento 12 è il delta del secondo neurone relativo al primo campione
        
        for index, (x,delta) in enumerate(zip(layer.inputs,delta_out)):

          layer.batch_der_error_wrt_weights[index] =  x * delta[:, np.newaxis]      #broadcast, delta[:,np.newaxis] mette delta come vettore colonna

          layer.batch_der_error_wrt_bias[index] = 1 * delta[:, np.newaxis]

      else:
        
        delta_hid = np.dot(delta_out, self.network[l+1].weights) *layer.derivative_function(layer.outputs)

        for index, (x,delta) in enumerate(zip(layer.inputs,delta_hid)):

          layer.batch_der_error_wrt_weights[index] =  x * delta[:, np.newaxis]      #broadcast, delta[:,np.newaxis] mette delta come vettore colonna

          layer.batch_der_error_wrt_bias[index] = 1 * delta[:, np.newaxis]

        delta_out = delta_hid

  def update_weights(self, alg_opt, batch_size, learning_rate, **kwargs):
    '''
    Update weights and bias of the whole neural network according to the chosen optimization algorithm.
    :param alg_opt: The optimization algorithm to use. (It could be GD )
    :param int batch_size: The batch size to use to compute the optimization algorithm.
    :param float learning_rate: The learning rate to use during the updating of the weights; a number between 0 and 1.
    '''
    if alg_opt == 'GD':
      
      for layer in self.network:
        layer.weights += np.mean(layer.batch_der_error_wrt_weights, axis=0) * learning_rate
        layer.bias += np.mean(layer.batch_der_error_wrt_bias, axis=0) * learning_rate
    
    
    if alg_opt == "momentum":
      momentum_factor =  kwargs.get("momentum_factor", 0.9) 

      for layer in self.network:
        
        current_mean_der_error_wrt_weights = np.mean(layer.batch_der_error_wrt_weights, axis=0)
        weights_velocity = current_mean_der_error_wrt_weights * learning_rate + momentum_factor * layer.previous_velocity_weights
        layer.weights += weights_velocity
        layer.previous_velocity_weights = weights_velocity

        current_mean_der_error_wrt_bias = np.mean(layer.batch_der_error_wrt_bias, axis=0)
        bias_velocity = current_mean_der_error_wrt_bias * learning_rate + momentum_factor * layer.previous_velocity_bias
        layer.bias += bias_velocity
        layer.previous_velocity_bias = bias_velocity
        
        
    if alg_opt == "Adagrad":
      eps = kwargs.get("eps", 1e-8) 
     
      for layer in self.network:
        current_mean_der_error_wrt_weights = np.mean(layer.batch_der_error_wrt_weights, axis=0)
        layer.cache_mean_weights += np.square(current_mean_der_error_wrt_weights)
        layer.weights += learning_rate * current_mean_der_error_wrt_weights / (np.sqrt(layer.cache_mean_weights) + eps)

        current_mean_der_error_wrt_bias = np.mean(layer.batch_der_error_wrt_bias, axis=0)
        layer.cache_mean_bias += np.square(current_mean_der_error_wrt_bias)
        layer.bias += learning_rate * current_mean_der_error_wrt_bias / (np.sqrt(layer.cache_mean_bias) + eps)
        

    if alg_opt == "RMSprop":
      decay_rate = kwargs.get("decay_rate", 0.9)
      eps = kwargs.get("eps", 1e-8) 
     
      for layer in self.network:
        current_mean_der_error_wrt_weights = np.mean(layer.batch_der_error_wrt_weights, axis=0)
        layer.cache_mean_weights = decay_rate * layer.cache_mean_weights + (1- decay_rate) * np.square(current_mean_der_error_wrt_weights)
        layer.weights += learning_rate * current_mean_der_error_wrt_weights / (np.sqrt(layer.cache_mean_weights) + eps)

        current_mean_der_error_wrt_bias = np.mean(layer.batch_der_error_wrt_bias, axis=0)
        layer.cache_mean_bias = decay_rate * layer.cache_mean_bias + (1- decay_rate) * np.square(current_mean_der_error_wrt_bias)
        layer.bias += learning_rate * current_mean_der_error_wrt_bias / (np.sqrt(layer.cache_mean_bias) + eps)


    if alg_opt == "Adam":
      beta1= kwargs.get("beta1", 0.9)
      beta2= kwargs.get("beta2", 0.999)
      eps = kwargs.get("eps", 1e-8)
      
      for layer in self.network:
        current_mean_der_error_wrt_weights = np.mean(layer.batch_der_error_wrt_weights, axis=0)
        first_moment_weights = beta1 * layer.previous_first_moment_weights + (1 - beta1) * current_mean_der_error_wrt_weights
        first_moment_iteration_weights = first_moment_weights / (1 - (beta1 ** self.iteration))
        second_moment_weights = beta2 * layer.previous_second_moment_weights + (1-beta2) * np.square(current_mean_der_error_wrt_weights)  
        second_moment_iteration_weights = second_moment_weights / (1- (beta2 ** self.iteration))
        layer.weights += learning_rate * first_moment_iteration_weights / (np.sqrt(second_moment_iteration_weights) + eps)

        layer.previous_first_moment_iteration_weights = first_moment_iteration_weights 
        layer.previous_second_moment_iteration_weights = second_moment_iteration_weights 

        current_mean_der_error_wrt_bias = np.mean(layer.batch_der_error_wrt_bias, axis=0)
        first_moment_bias = beta1 * layer.previous_first_moment_bias + (1 - beta1) * current_mean_der_error_wrt_bias
        first_moment_iteration_bias = first_moment_bias / (1 - (beta1 ** self.iteration))
        second_moment_bias = beta2 * layer.previous_second_moment_bias + (1 - beta2) * np.square(current_mean_der_error_wrt_bias)  
        second_moment_iteration_bias = second_moment_bias / (1- (beta2 ** self.iteration))
        layer.bias += learning_rate * first_moment_iteration_bias / (np.sqrt(second_moment_iteration_bias) + eps)

        layer.previous_first_moment_iteration_bias = first_moment_iteration_bias 
        layer.previous_second_moment_iteration_bias = second_moment_iteration_bias 


  def predict(self, X):
        """
        Predicts a class.
        :param X: The input values.
        :return: The predictions.
        """
        
        return np.round(self.feedforward(X))
          

  def accuracy(self, predicted_labels, true_labels):
        """
        Calculates the accuracy between the predicted labels and the true labels.
        :param predicted_labels: The predicted labels.
        :param true_labels: The true labels.
        :return: The calculated accuracy.
        """

        return ((true_labels == predicted_labels)).mean()
  

  def show_result(self, epochs, list_loss, list_acc, title):
    '''
    Draws the result obtained during the training
    '''
    fig = plt.figure(figsize= [10,10])
    plt.plot(epochs, list_loss, '-b', label='loss')
    plt.plot(epochs, list_acc, '-r', label='accuracy')

    plt.xlabel("epochs")
    plt.legend(loc='upper left')
    plt.title(title)
    fig.savefig('out/' + str(title)+ '.png', dpi=fig.dpi)

  


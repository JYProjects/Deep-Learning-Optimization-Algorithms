
from optimization import NeuralNetwork as nn
import data_utils as du

def main():
    n_input = 3
    n_train_perc = 0.7
    
    learning_rate = 0.05
    alg_opt = 'GD'
    batch_size = 64
    epochs = 500
    
    #load data
    X_train, y_train, X_test, y_test = du.load_data(n_input = n_input, n_train_perc = n_train_perc)
    
    #riduco il dataset
    X_train = X_train[:, : int(X_train.shape[1]/10), : ]
    y_train = y_train[: int(y_train.shape[0]/10)]
    X_test = X_test[:, : int(X_test.shape[1]/10), :]
    y_test = y_test[: int(y_test.shape[0]/10)]
     
   
    models_trained = []
    
    
    for index in range(X_train.shape[0]): 
      #creation of our network
      model = nn.NeuralNetwork()
      model.add(n_input = X_train.shape[2], 
                n_neurons = 5, 
                activation = 'sigmoid', 
                name = "hidden1" )
      model.add(n_input = 5, 
                n_neurons = y_train.shape[1], 
                activation = 'sigmoid', 
                name = "output")
      model.check()
      
    
      model.train(X_train[index], 
                  y_train, 
                  learning_rate, 
                  alg_opt, 
                  batch_size, 
                  epochs)
      
      prediction_train = model.predict(X_train[index])
      print("train accuracy: ", model.accuracy(prediction_train, y_train))
      prediction_test = model.predict(X_test[index])
      print("test accuracy: ",model.accuracy(prediction_test, y_test))
      
      models_trained.append(model)
      title = str("model_") + str(index+1) + str("_") + str(alg_opt) 
      epochs_list = [i+1 for i in range(0, epochs)]
      model.show_result(epochs_list, model.list_loss, model.list_acc, title)
    
    #VALUTO LA BER
    BER_prediction = []
    BER_dataset, BER_bit = du.load_data_BER(n_input)
    
    for index, model in enumerate(models_trained):
      
      prediction_test = model.predict(BER_dataset[index])
      BER_prediction.append(1 - model.accuracy(prediction_test, BER_bit))
    
      
      
    du.plot_BER(BER_prediction)



main()

        

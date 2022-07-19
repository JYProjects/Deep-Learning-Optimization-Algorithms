import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
def load_data(n_input, n_train_perc):
    X = pd.read_excel('Data/train_04alfa_3sedicesimi.xlsx', index_col = None, header = None) 
    y = pd.read_excel('Data/bit_train_04alfa_3sedicesimi.xlsx', index_col = None, header = None) 
    
    n_train_perc = n_train_perc #0.7
    n = n_input #3
    
    discard = int((n-1)/2)
    
    yr = np.array(y.iloc[0])
    yr = yr[discard : yr.shape[0]-discard] 
    dataset_y = yr[:,np.newaxis]
    
    n_train = int(n_train_perc * dataset_y.shape[0])
    
    y_train = dataset_y[: n_train]
    y_test  = dataset_y[n_train:]
    
    X_train, X_test = create_X_dataset(X, n, n_train)
    
    return X_train, y_train, X_test, y_test

def load_data_BER(n_input): 
    data_BER1 = pd.read_excel('Data/inputBER1.xlsx', index_col = None, header = None)
    data_BER2 = pd.read_excel('Data/inputBER2.xlsx', index_col = None, header = None)
    data_BER3 = pd.read_excel('Data/inputBER3.xlsx', index_col = None, header = None)
    data_BER4 = pd.read_excel('Data/inputBER4.xlsx', index_col = None, header = None)
    data_BER5 = pd.read_excel('Data/inputBER5.xlsx', index_col = None, header = None)
    data_BER6 = pd.read_excel('Data/inputBER6.xlsx', index_col = None, header = None)
    data_BER7 = pd.read_excel('Data/inputBER7.xlsx', index_col = None, header = None)
    data_BER8 = pd.read_excel('Data/inputBER8.xlsx', index_col = None, header = None)
    data_BER9 = pd.read_excel('Data/inputBER9.xlsx', index_col = None, header = None)
    data_BER10 = pd.read_excel('Data/inputBER10.xlsx', index_col = None, header = None)
    
    data_stack = pd.concat([data_BER1, data_BER2, data_BER3, data_BER4, data_BER5,
                            data_BER6, data_BER7, data_BER8, data_BER9, data_BER10], axis=1)
    
    bit_BER1 = pd.read_excel('Data/bit_BER1.xlsx', index_col = None, header = None)
    bit_BER2 = pd.read_excel('Data/bit_BER2.xlsx', index_col = None, header = None)
    bit_BER3 = pd.read_excel('Data/bit_BER3.xlsx', index_col = None, header = None)
    bit_BER4 = pd.read_excel('Data/bit_BER4.xlsx', index_col = None, header = None)
    bit_BER5 = pd.read_excel('Data/bit_BER5.xlsx', index_col = None, header = None)
    bit_BER6 = pd.read_excel('Data/bit_BER6.xlsx', index_col = None, header = None)
    bit_BER7 = pd.read_excel('Data/bit_BER7.xlsx', index_col = None, header = None)
    bit_BER8 = pd.read_excel('Data/bit_BER8.xlsx', index_col = None, header = None)
    bit_BER9 = pd.read_excel('Data/bit_BER9.xlsx', index_col = None, header = None)
    bit_BER10 = pd.read_excel('Data/bit_BER10.xlsx', index_col = None, header = None)
    
    bit_stack = pd.concat([bit_BER1, bit_BER2, bit_BER3, bit_BER4, bit_BER5, 
                           bit_BER6, bit_BER7, bit_BER8, bit_BER9, bit_BER10], axis=1)

    BER_dataset, BER_bit = create_BER_dataset(data_stack, bit_stack, n = n_input)
    
    return BER_dataset, BER_bit




def create_X_dataset(X, n, n_train):
  
  X_dataset = np.zeros((X.shape[0],X.shape[1]-(n-1),n)) 
  print(X_dataset.shape)
  for el in range(X.shape[0]):
    tr = np.array(X.iloc[el])
    list_df = [tr[i:i+n] for i in range(0,tr.shape[0]-(n-1),1)]
    X_dataset[el] = np.array(list_df) 
    X_train = X_dataset[:, : n_train, :]
    X_test  = X_dataset[:, n_train : , :]
  return X_train, X_test




def create_BER_dataset(data_stack, bit_stack, n):
  
  BER_dataset = np.zeros((data_stack.shape[0],data_stack.shape[1]-(n-1),n)) 
  
  for el in range(data_stack.shape[0]):
    tr = np.array(data_stack.iloc[el])
    list_df = [tr[i:i+n] for i in range(0,tr.shape[0]-(n-1),1)]
    BER_dataset[el] = np.array(list_df) 
  
  discard = int((n-1)/2)

  BER_output = np.array(bit_stack.iloc[0])
  BER_output = BER_output[discard : BER_output.shape[0]-discard] 
  BER_bit = BER_output[:,np.newaxis]

  return BER_dataset, BER_bit


def plot_BER_total(**kwargs):
  BER_value_dB = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
  BER_NO_delay_alfa04 = [0.10281,	0.07752,	0.05503,	0.0367,	0.02257,	0.01275,	0.00579,	0.00219,	0.00077,	0.00022]
  BER_delay3_alfa04 = [0.12885,	0.10379,	0.08082,	0.06091,	0.04382,	0.03087,	0.02084,	0.01319,	0.00773,	0.00431] 
  
  BER_GD_bs64_epochs500_lr05 = [0.11365227304546088, 0.08817176343526867, 0.06553131062621254, 0.04552091041820838, 0.03034060681213624, 0.01783035660713217, 0.009780195603912079, 0.004600092001840084, 0.0018900378007560281, 0.0006400128002560512]
  BER_momentum_bs64_epochs100_lr05 = [0.11320226404528089, 0.08865177303546068, 0.06591131822636453, 0.045660913218264376, 0.030190603812076278, 0.01804036080721616, 0.009800196003920125, 0.004850097001940057, 0.00204004080081599, 0.000720014400288016] 
  BER_Adagrad_bs64_epochs50_lr01 = [0.1143022860457209, 0.08858177163543268, 0.06534130682613648, 0.04528090561811238, 0.02985059701194026, 0.01756035120702415, 0.009830196603932029, 0.004630092601851987, 0.0019500390007800572, 0.0005500110002200076]
  BER_RMSprop_bs64_epochs30_lr01 = [0.11485229704594091, 0.08887177743554875, 0.06580131602632056, 0.0455509110182204, 0.03012060241204828, 0.018000360007200178, 0.010110202204044128, 0.005010100202003986, 0.0018700374007479814, 0.0005700114002280543]
  BER_Adam_bs64_epochs10_lr05 = [0.11539230784615695, 0.09007180143602878, 0.06629132582651653, 0.04603092061841241, 0.02985059701194026, 0.01782035640712809, 0.00988019760395209, 0.004450089001780011, 0.0018400368007359669, 0.0005300106002119609]
  
  
  fig = plt.figure(figsize = [20,20])
  
  plt.semilogy(BER_value_dB, BER_NO_delay_alfa04, '--b', label = 'No_delay')
  plt.semilogy(BER_value_dB, BER_delay3_alfa04, 'bo-', label = 'delay3_teorica')
  plt.semilogy(BER_value_dB, BER_GD_bs64_epochs500_lr05, '-g', label = 'GD_epochs500')
  plt.semilogy(BER_value_dB, BER_momentum_bs64_epochs100_lr05, '-k', label = 'momentum_epochs100')
  plt.semilogy(BER_value_dB, BER_Adagrad_bs64_epochs50_lr01, '-r', label = 'Adagrad_epochs50')
  plt.semilogy(BER_value_dB, BER_RMSprop_bs64_epochs30_lr01, '-c', label = 'RMSprop_epochs30')
  plt.semilogy(BER_value_dB, BER_Adam_bs64_epochs10_lr05, '-m', label = 'Adam_epochs10')
  
  plt.legend(loc = 'upper right')
  plt.title('BER_algorithms')
  plt.grid()
  plt.xlabel("EsNodB")
  plt.ylabel("BER")

  fig.savefig('out/BER_total.png', dpi=fig.dpi)
    


def plot_BER(BER_prediction, **kwargs):
  BER_value_dB = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
  BER_NO_delay_alfa04 = [0.10281,	0.07752,	0.05503,	0.0367,	0.02257,	0.01275,	0.00579,	0.00219,	0.00077,	0.00022]
  BER_delay3_alfa04 = [0.12885,	0.10379,	0.08082,	0.06091,	0.04382,	0.03087,	0.02084,	0.01319,	0.00773,	0.00431] 
      
  fig = plt.figure(figsize = [15,15])
  plt.semilogy(BER_value_dB, BER_NO_delay_alfa04, '--b', label = 'No_delay')
  plt.semilogy(BER_value_dB, BER_delay3_alfa04, 'bo-', label = 'delay3_teorica')
  plt.semilogy(BER_value_dB, BER_prediction, '-m', label = 'prediction')
  plt.legend(loc = 'upper right')
  plt.title('BER_for_this_prediction')
  plt.grid()
  plt.xlabel("EsNodB")
  plt.ylabel("BER")

  fig.savefig('out/BER_prediction.png', dpi=fig.dpi)
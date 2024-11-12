import mat73
import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tf_cfc import CfcCell
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def get_flops(model):
    # Save the current file descriptor for stdout
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)

    # Create a temporary file to redirect stdout
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stdout_fd)
        
        # Execute profiling
        forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops

    # Restore the original stdout file descriptor so that printing to stdout works normally again
    os.dup2(saved_stdout_fd, stdout_fd)
    os.close(saved_stdout_fd)

    return flops

nSym = 20
nDSC = 48
pilots = 4
nUSC = nDSC + pilots
rate = 1
In = 2 * (nDSC / int(rate) + pilots)
Out = 2 * (nDSC + pilots)
LSTM_size = (nDSC / int(rate) + pilots)
NN_SIZE = 15
LNN_SIZE_WR = 54
backbone_units_WR = 190
LNN_SIZE = 20
backbone_units = 100

# Build LSTM model.
init = keras.initializers.glorot_uniform(seed=1)
modelLSTM = keras.models.Sequential([
    keras.layers.LSTM(units=int(LSTM_size), activation='relu',
                      kernel_initializer=init, bias_initializer=init,
                      return_sequences=True, input_shape=(nSym, int(In))),
    keras.layers.Dense(units=int(NN_SIZE), activation='relu',
                       kernel_initializer=init, bias_initializer=init),
    keras.layers.Dense(units=int(Out), kernel_initializer=init,
                       bias_initializer=init)
])
total_paramsLSTM = modelLSTM.count_params()
flopsLSTM = get_flops(modelLSTM)

# Build CfC model - Without complexity restriction
input_shape = (nSym, nUSC * 2)
num_classes = nUSC * 2
CFC_CONFIG_WR = {
    "backbone_activation": "gelu",
    "backbone_dr": 0.0,
    "forget_bias": 3.0,
    "backbone_units": backbone_units_WR,
    "backbone_layers": 1,
    "weight_decay": 0,
    "use_lstm": False,
    "no_gate": False,
    "minimal": False,
}
cfc_cell_WR = CfcCell(units=int(LNN_SIZE_WR), hparams=CFC_CONFIG_WR, name='RNN2')
modelCfC_WR = keras.models.Sequential()
modelCfC_WR.add(keras.layers.InputLayer(input_shape=input_shape))
modelCfC_WR.add(keras.layers.RNN(cfc_cell_WR, return_sequences=True, name='RNN2'))
modelCfC_WR.add(keras.layers.Dense(units=num_classes, name='Output'))
total_paramsCfC_WR = modelCfC_WR.count_params()
flopsCfC_WR = get_flops(modelCfC_WR)

# Build CfC model.
input_shape = (nSym, nUSC * 2)
num_classes = nUSC * 2
CFC_CONFIG = {
    "backbone_activation": "gelu",
    "backbone_dr": 0.0,
    "forget_bias": 3.0,
    "backbone_units": backbone_units,
    "backbone_layers": 1,
    "weight_decay": 0,
    "use_lstm": False,
    "no_gate": False,
    "minimal": False,
}
cfc_cell = CfcCell(units=int(LNN_SIZE), hparams=CFC_CONFIG, name='RNN1')
modelCfC = keras.models.Sequential()
modelCfC.add(keras.layers.InputLayer(input_shape=input_shape))
modelCfC.add(keras.layers.RNN(cfc_cell, return_sequences=True, name='RNN1'))
modelCfC.add(keras.layers.Dense(units=num_classes, name='Output'))
total_paramsCfC = modelCfC.count_params()
flopsCfC = get_flops(modelCfC)

# Build DNN model
hl1 = 40
hl2 = 20
hl3 = 40
modelDNN = keras.models.Sequential([
    keras.layers.Dense(units=int(hl1), activation='relu', input_dim=int(In),
                       kernel_initializer=init, bias_initializer=init),
    keras.layers.Dense(units=int(hl2), activation='relu',
                       kernel_initializer=init, bias_initializer=init),
    keras.layers.Dense(units=int(hl3), activation='relu',
                       kernel_initializer=init, bias_initializer=init),
    keras.layers.Dense(units=int(Out), kernel_initializer=init,
                       bias_initializer=init)
])
total_paramsDNN = modelDNN.count_params()
flopsDNN = get_flops(modelDNN)
print(f"DPA-DNN")
print(f"Params: {total_paramsDNN:,}")
print(f"FLOPs: {flopsDNN:,}")
print(f"DPA-LSTM-NN")
print(f"Params: {total_paramsLSTM:,}")
print(f"FLOPs: {flopsLSTM:,}")
print(f"DPA-CfC Without Restriction")
print(f"Params: {total_paramsCfC_WR:,}")
print(f"FLOPs: {flopsCfC_WR:,}")
print(f"DPA-CfC")
print(f"Params: {total_paramsCfC:,}")
print(f"FLOPs: {flopsCfC:,}")


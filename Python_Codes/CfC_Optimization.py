import optuna
import os
import sys
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow import keras
import mat73
from tf_cfc import CfcCell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import backend as K
import gc

# Load and preprocess data as before
scheme = 'DPA' 
nSym = 20
rate = 1
nDSC = 48
pilots = 4
rate = 1
nUSC = nDSC + pilots

mat = mat73.loadmat('data/{}_Less_{}_Dataset_7.mat'.format(scheme, rate))
Training_Dataset = mat['DNN_Datasets']
X = Training_Dataset['Train_X']
Y = Training_Dataset['Train_Y']

scalerx = StandardScaler()
scalerx.fit(X)
scalery = StandardScaler()
scalery.fit(Y)
XS = scalerx.transform(X)
YS = scalery.transform(Y)
XS = XS.transpose()
YS = YS.transpose()

XS = np.reshape(XS, (80000, nSym, nUSC * 2, 1))  # Adding a channel dimension for Conv2D
YS = np.reshape(YS, (80000, nSym, nUSC * 2))
input_shape = (nSym, nUSC * 2, 1)
num_classes = nUSC * 2

X_train, X_test, y_train, y_test = train_test_split(XS, YS, test_size=0.20)

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

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Nadam'])

    # CNN
    add_cnn = trial.suggest_categorical('add_cnn', [True, False])
    if add_cnn:
        num_filters = trial.suggest_int('num_filters', 16, 64)
        kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5)])
        pooling_type = trial.suggest_categorical('pooling_type', ['max', 'average'])

    # CfC
    backbone_units = trial.suggest_int('backbone_units', 1, 200,step=1)
    backbone_layers = trial.suggest_int('backbone_layers', 1, 3)
    backbone_activation = trial.suggest_categorical('backbone_activation', ['relu', 'gelu'])
    LNN_size = trial.suggest_int('LNN_size', 1, 100, step=1)
    add_second_cfc_layer = trial.suggest_categorical('add_second_cfc_layer', [True, False])
    
    # NN
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'gelu'])
    num_dense_layers = trial.suggest_int('num_dense_layers', 0, 2)  

    # Define CFC cell configuration
    CFC_CONFIG = {
        "backbone_activation": backbone_activation,
        "backbone_dr": 0.0,
        "forget_bias": 3.0,
        "backbone_units": backbone_units,
        "backbone_layers": backbone_layers,
        "weight_decay": 0,
        "use_lstm": False,
        "no_gate": False,
        "minimal": False,
    }

    # Build the model
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    if add_cnn:
        model.add(keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        if pooling_type == 'max':
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        elif pooling_type == 'average':
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())  

    model.add(keras.layers.Reshape((nSym, -1)))  
    cfc_cell = CfcCell(units=LNN_size, hparams=CFC_CONFIG, name='RNN1')
    model.add(keras.layers.RNN(cfc_cell, return_sequences=True))

    if add_second_cfc_layer:
        LNN_size_2 = trial.suggest_int('LNN_size_2', 1, 100, step = 1)
        cfc_cell_2 = CfcCell(units=LNN_size_2, hparams=CFC_CONFIG, name='RNN2')
        model.add(keras.layers.RNN(cfc_cell_2, return_sequences=True, name='RNN2'))

    for i in range(num_dense_layers):
        layer_units = trial.suggest_int(f'NN_size_layer_{i}', 1, 100, step = 1)
        model.add(keras.layers.Dense(units=layer_units, activation=activation_function))
        dropout_rate = trial.suggest_uniform(f'dropout_rate_{i}', 0.0, 0.6)
        model.add(keras.layers.Dropout(rate=dropout_rate, name=f'Dropout_{i}'))   

    model.add(keras.layers.Dense(units=num_classes, name='Output'))
   
    # If more complex then the LSTM-based model (35k params, 160k FLOPs) the study is pruned
    num_params = model.count_params()
    flops = get_flops(model)
    if num_params > 35*1e3 or flops > 160*1e3:
        raise optuna.TrialPruned()

    model.summary()
    print(f"FLOPs: {flops:,}")
    # Choose the optimizer
    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    # Callback for early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=0,
                        validation_data=(X_test, y_test),callbacks=early_stopping)
    
    
    val_loss = history.history['val_loss'][-1]
    complexity_penalty = num_params * 1e-8

    K.clear_session()
    gc.collect()

    objective = val_loss + complexity_penalty

    return objective

# Run optimization
study_name = "Study_LNN-architecture-optimization" 
storage_name = 'sqlite:///{}.db'.format(study_name)
study = optuna.create_study(direction='minimize',study_name=study_name, storage=storage_name,load_if_exists=True)

study.optimize(objective, n_trials=3)

# Get best parameters
best_params = study.best_params
print("Best params:", best_params)

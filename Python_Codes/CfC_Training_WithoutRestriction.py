from tensorflow import keras
from scipy.io import loadmat
from tf_cfc import CfcCell
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SNR_array = [0, 5, 10, 15, 20, 25, 30]

scheme = 'DPA' 
nSym = 20
snr = 7
nDSC = 48
pilots = 4
nUSC = nDSC + pilots
In = 2*(nDSC + pilots)
LNN_SIZE = 54
epoch = 500
batch_size = 32

# Training with the highest SNR
mat = loadmat('data/Dataset_{}.mat'.format(snr))
Training_Dataset = mat['Datasets']
Training_Dataset = Training_Dataset[0,0]
X = Training_Dataset['Train_X']
Y = Training_Dataset['Train_Y']
print('Loaded Dataset Inputs: ', X.shape)
print('Loaded Dataset Outputs: ', Y.shape)

# Normalizing Datasets
scalerx = StandardScaler()
scalerx.fit(X)
scalery = StandardScaler()
scalery.fit(Y)
XS = scalerx.transform(X)
YS = scalery.transform(Y)
XS = XS.transpose()
YS = YS.transpose()

# Reshape the input 
XS     = np.reshape(XS,(80000, nSym, nUSC*2))
YS     = np.reshape(YS,(80000, nSym,  nUSC*2))
print('Training shape', XS.shape)
input_shape = (nSym, nUSC*2)
num_classes = nUSC*2

X_train, X_test, y_train, y_test = train_test_split(XS, YS, test_size=0.10)

# Build the model.
CFC_CONFIG = {
    "backbone_activation": "gelu",
    "backbone_dr": 0.0,
    "forget_bias": 3.0,
    "backbone_units": 190,
    "backbone_layers": 1,
    "weight_decay": 0,
    "use_lstm": False,
    "no_gate": False,
    "minimal": False,
}
cfc_cell = CfcCell(units=int(LNN_SIZE), hparams=CFC_CONFIG,name='RNN1')

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))
model.add(keras.layers.RNN(cfc_cell, return_sequences=True, name='RNN1'))
model.add(keras.layers.Dense(units=num_classes, name='Output'))
model.summary()
# Compile the model.
opt = keras.optimizers.Nadam(learning_rate=0.001)
keras.backend.clear_session()
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# Train the models
seed(1)
model_path = 'data/models/{}_LNN_Opt_WithoutRestriction.h5.keras'.format(scheme)   
checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',verbose=1,mode='min',save_best_only=True,overwrite=True)
callbacks_list = [checkpoint, reduce_lr, early_stopping]
history = model.fit(X_train, y_train, epochs=epoch, batch_size=int(batch_size), verbose=2, validation_data=(X_test, y_test),  callbacks=callbacks_list)

# Plotting training history
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(14, 5))

# Plot loss
plt.plot(epochs, loss, 'k-o', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
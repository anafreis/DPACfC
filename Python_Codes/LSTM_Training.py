from tensorflow import keras
from scipy.io import loadmat
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
In = 2*(nDSC + pilots)
LSTM_size = (nDSC + pilots)
NN_SIZE = 15
Out =  2*(nDSC + pilots)
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

# To use LSTM networks, the input needs to be reshaped to be [samples, time steps, features]
XS     = np.reshape(XS,(80000, nSym, int(In)))
YS     = np.reshape(YS,(80000, nSym,  int(Out)))
print('Training shape', XS.shape)

X_train, X_test, y_train, y_test = train_test_split(XS, YS, test_size=0.10)

# Build the model.
# The weighs are initialized by the Glorot Initializer
      #http://proceedings.mlr.press/v9/glorot10a.html
init = keras.initializers.glorot_uniform(seed=1)
# Add a LSTM layer
model = keras.models.Sequential([
    keras.layers.LSTM(units=int(LSTM_size), activation='relu',
                      kernel_initializer=init, bias_initializer=init,
                      return_sequences=True, input_shape=(nSym, int(In))),  
    keras.layers.Dense(units=int(NN_SIZE), activation='relu',
                       kernel_initializer=init, bias_initializer=init),
    keras.layers.Dense(units=int(Out), kernel_initializer=init,
                       bias_initializer=init)
])
model.summary()
# Compile the model.
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# Train the models
seed(1)
model_path = 'data/models/{}_LSTM_{}{}_{}.h5.keras'.format(scheme,int(LSTM_size),NN_SIZE, SNR_array[int(snr)-1])
checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',verbose=1,mode='min',save_best_only=True)
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
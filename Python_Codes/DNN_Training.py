from tensorflow import keras
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.random import seed
import warnings
warnings.filterwarnings("ignore")

SNR_array = [0, 5, 10, 15, 20, 25, 30]

scheme = 'DPA' 
snr = 7
In = 104
hl1 = 40
hl2 = 20
hl3 = 40
Out = 104
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
print('Training shape', XS.shape)

X_train, X_test, y_train, y_test = train_test_split(XS, YS, test_size=0.10)

# Build the model.
# The weighs are initialized by the Glorot Initializer
      #http://proceedings.mlr.press/v9/glorot10a.html
init =  keras.initializers.glorot_uniform(seed=1)
model =  keras.models.Sequential([
      keras.layers.Dense(units= int(hl1), activation='relu', input_dim=int(In),
      kernel_initializer=init,
      bias_initializer=init),
      keras.layers.Dense(units= int(hl2), activation='relu',
      kernel_initializer=init,
      bias_initializer=init),
      keras.layers.Dense(units= int(hl3), activation='relu',
      kernel_initializer=init,
      bias_initializer=init),
      keras.layers.Dense(units=int(Out), kernel_initializer=init,
      bias_initializer=init)
])

# Compile the model.
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
print(model.summary())

# Callbacks for learning rate reduction and early stopping
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# Train the models
seed(1)
model_path = 'data/models/{}_DNN_{}{}{}_{}.h5.keras'.format(scheme,hl1,hl2,hl3, SNR_array[int(snr)-1])
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
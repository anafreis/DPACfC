from tensorflow import keras
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import scipy.io
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.backend import squeeze

DNN_Model = 30
scheme = 'DPA' 
nSym = 20
nDSC = 48
pilots = 4
In = 2*(nDSC + pilots)
LSTM_size = (nDSC + pilots)
Out =  2*(nDSC + pilots)
MLP_size = 15

SNR_index = np.arange(1, 8)

for j in SNR_index:
    mat = loadmat('data\Dataset_{}.mat'.format(j))
    Testing_Dataset = mat['Datasets']
    Testing_Dataset = Testing_Dataset[0,0]
    X = Testing_Dataset['Test_X']
    Y = Testing_Dataset['Test_Y']
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
    XS = np.reshape(XS,(20000, nSym, int(In)))
    print(XS.shape)

    # Loading trained model
    model = keras.models.load_model('data\models\{}_LSTM_{}{}_{}.h5.keras'.format(scheme, int(LSTM_size), MLP_size, DNN_Model))
    print('Model Loaded: ', DNN_Model)

    # Testing the model
    Y_pred = model.predict(XS)
   
    XS = np.reshape(XS,(20000*nSym, int(In)))
    Y_pred = np.reshape(Y_pred,(20000*nSym, int(Out)))

    XS = XS.transpose()
    YS = YS.transpose()
    Y_pred = Y_pred.transpose()

    # Calculation of Mean Squared Error (MSE)
    Original_Testing_X = scalerx.inverse_transform(XS)
    Original_Testing_Y = scalery.inverse_transform(YS)
    Prediction_Y = scalery.inverse_transform(Y_pred)

    Error = mean_squared_error(Original_Testing_Y, Prediction_Y)
    print('MSE: ', Error)
   
    # Saving the results and converting to .mat
    result_path = 'data\{}_LSTM_{}{}_Results_{}.pickle'.format(scheme, int(LSTM_size), MLP_size, j)
    with open(result_path, 'wb') as f:
        pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

    dest_name = 'data\{}_LSTM_{}{}_Results_{}.mat'.format(scheme, int(LSTM_size), MLP_size, j)
    a = pickle.load(open(result_path, "rb"))
    scipy.io.savemat(dest_name, {
        '{}_LSTM_{}{}_test_x_{}'.format(scheme, int(LSTM_size), MLP_size, j): a[0],
        '{}_LSTM_{}{}_test_y_{}'.format(scheme,  int(LSTM_size), MLP_size, j): a[1],
        '{}_LSTM_{}{}_corrected_y_{}'.format(scheme,  int(LSTM_size), MLP_size, j): a[2]
    })
    
    print("Data successfully converted to .mat file")

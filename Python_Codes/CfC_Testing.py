import tensorflow as tf
from scipy.io import loadmat
from tensorflow import keras
from tf_cfc import CfcCell
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import scipy.io
from sklearn.metrics import mean_squared_error

# Custom objects dictionary for loading the model
custom_objects = {'CfCCell': CfcCell}
DNN_Model = 30
scheme = 'DPA'
nSym = 20
nDSC = 48
pilots = 4
nUSC = nDSC + pilots
Out = 25

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

    # Reshape the input 
    XS     = np.reshape(XS,(20000, nSym, nUSC*2))
    print(XS.shape)

    # Loading trained model using custom_object_scope
    model_path = 'data/models/{}_LNN_Opt.h5.keras'.format(scheme)
    with keras.utils.custom_object_scope({'CfcCell': CfcCell}):
        model = keras.models.load_model(model_path)
    print('Model Loaded: ', DNN_Model)

    # Testing the model
    Y_pred = model.predict(XS)
   
    XS = np.reshape(XS,(20000*nSym, nUSC*2))
    Y_pred = np.reshape(Y_pred,(20000*nSym, nUSC*2))

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
    result_path = 'data\models\{}_LNN_Results_{}_Opt.pickle'.format(scheme, j)
    with open(result_path, 'wb') as f:
        pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

    dest_name = 'data\{}_LNN_Results_{}_Opt.mat'.format(scheme, j)
    a = pickle.load(open(result_path, "rb"))
    scipy.io.savemat(dest_name, {
        '{}_LNN_test_x_{}'.format(scheme, j): a[0],
        '{}_LNN_test_y_{}'.format(scheme, j): a[1],
        '{}_LNN_corrected_y_{}'.format(scheme, j): a[2]
    })
    print("Data successfully converted to .mat file ")

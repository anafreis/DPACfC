from tensorflow import keras
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import scipy.io

DNN_Model = 30
scheme = 'DPA'
hl1 = 40
hl2 = 20
hl3 = 40

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

    # Loading trained model
    model = keras.models.load_model('data\models\{}_DNN_{}{}{}_{}.h5.keras'.format(scheme, hl1, hl2, hl3, DNN_Model))

    # Testing the model
    Y_pred = model.predict(XS)

    XS = XS.transpose()
    YS = YS.transpose()
    Y_pred = Y_pred.transpose()

    Original_Testing_X = scalerx.inverse_transform(XS)
    Original_Testing_Y = scalery.inverse_transform(YS)
    Prediction_Y = scalery.inverse_transform(Y_pred)

    Error = mean_squared_error(Original_Testing_Y, Prediction_Y)
    print('MSE: ', Error)

    # Saving the results and converting to .mat
    result_path = 'data\{}_DNN_{}{}{}_Results_{}.pickle'.format(scheme, hl1, hl2, hl3, j)
    with open(result_path, 'wb') as f:
        pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

    dest_name = 'data\{}_DNN_{}{}{}_Results_{}.mat'.format(scheme, hl1, hl2, hl3, j)
    a = pickle.load(open(result_path, "rb"))
    scipy.io.savemat(dest_name, {
        '{}_DNN_{}{}{}_test_x_{}'.format(scheme, hl1, hl2, hl3, j): a[0],
        '{}_DNN_{}{}{}_test_y_{}'.format(scheme, hl1, hl2, hl3, j): a[1],
        '{}_DNN_{}{}{}_corrected_y_{}'.format(scheme, hl1, hl2, hl3, j): a[2]
    })
    
    print("Data successfully converted to .mat file ")

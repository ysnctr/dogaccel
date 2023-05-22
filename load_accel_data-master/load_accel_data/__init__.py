import pandas as pd
import scipy.io as scio

# function to load accelerometer data into a pandas Dataframe
def loadAccel (directory):
    mat = scio.loadmat(directory)
    x = pd.DataFrame.from_dict(mat['X'])
    x.columns = {'x'}
    y = pd.DataFrame.from_dict(mat['Y'])
    y.columns = {'y'}
    z = pd.DataFrame.from_dict(mat['Z'])
    z.columns = {'z'}
    #label for output data
    label = pd.DataFrame.from_dict(mat['Labels'])
    label.columns = {'label'}

    #join x, y, z into one dataframe
    accelData = pd.concat([x,y,z], axis = 1)

    return label, accelData;

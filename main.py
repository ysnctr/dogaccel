# import own packages
from dog_features_extraction import extractFeat
from dog_features_extraction import normalizeFeatures
from load_accel_data import loadAccel
from comparing_classifiers_ovo import classifier_comparison
# import system packages
import pandas as pd



dataDir = "data/all_dogs.mat"
label, accelData = loadAccel(dataDir)

#set number of data points grouped to extract features, n - i.e windowing
n = 5

#calculate spare data points so no null values are created in the features dataframe
remove = len(accelData)%n

for i in range(0, len(accelData)-remove, n):
    if (i==0): features = extractFeat(accelData[i:i+n], n)
    else: features = pd.concat([features,extractFeat(accelData[i:i+5], n)], axis = 0)
features = features.loc[0]
print(features.shape)

#select most common label for every n datapoints: y_features
for i in range(0, len(label)-remove, n):
    if(i==0):
        y_mode = label[i:i+n].mode()
        if(len(y_mode>1)): y_features = pd.DataFrame(data = y_mode.mean().round(0))
        else: y_features = pd.DataFrame(data = label[i:i+n].mode())
    else:
        y_mode = label[i:i+n].mode()
        if(len(y_mode>1)):
            y = pd.DataFrame(data = y_mode.mean().round(0))
            y_features = pd.concat([y_features, y], axis = 1)

        else:
            y_features = pd.concat([y_features, y_mode], axis = 1)
y_features = y_features.transpose()
y_features = y_features.astype('int32')
print(y_features.shape)

#normalize features
norm_features = normalizeFeatures(features, y_features)
y_features = y_features.values.ravel()
classifier_comparison(norm_features, y_features)

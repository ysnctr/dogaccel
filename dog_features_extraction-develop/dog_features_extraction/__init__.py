import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy import fftpack

# function to extract features from training data
def extractFeat (accelData, n):
    # mean
    x = accelData.x
    mean_x = pd.DataFrame(data = [x.mean()])
    mean_x.columns = {'mean_x'}
    # std dev
    std_x = pd.DataFrame(data = [x.std()])
    std_x.columns = {'std_x'}
    # skewness
    skew_x = pd.DataFrame(data = [x.skew()])
    skew_x.columns = {'skew_x'}
    # max
    max_x = pd.DataFrame(data = [x.max()])
    max_x.columns = {'max_x'}
    # min
    min_x = pd.DataFrame(data = [x.min()])
    min_x.columns = {'min_x'}

     # mean
    y = accelData.y
    mean_y = pd.DataFrame(data = [y.mean()])
    mean_y.columns = {'mean_y'}
    # std dev
    std_y = pd.DataFrame(data = [y.std()])
    std_y.columns = {'std_y'}
    # skewness
    skew_y = pd.DataFrame(data = [y.skew()])
    skew_y.columns = {'skew_y'}
    # max
    max_y = pd.DataFrame(data = [y.max()])
    max_y.columns = {'max_y'}
    # min
    min_y = pd.DataFrame(data = [y.min()])
    min_y.columns = {'min_y'}

     # mean
    z = accelData.z
    mean_z = pd.DataFrame(data = [z.mean()])
    mean_z.columns = {'mean_z'}
    # std dev
    std_z = pd.DataFrame(data = [z.std()])
    std_z.columns = {'std_z'}
    # skewness
    skew_z = pd.DataFrame(data = [z.skew()])
    skew_z.columns = {'skew_z'}
    # max
    max_z = pd.DataFrame(data = [z.max()])
    max_z.columns = {'max_z'}
    # min
    min_z = pd.DataFrame(data = [z.min()])
    min_z.columns = {'min_z'}

    # x/z
    try:
        x_z = pd.DataFrame(data = [x.mean()/z.mean()], dtype = 'float64')
        x_z.columns = {'x/z'}

    except ZeroDivisionError:
        x_z = pd.DataFrame(data = [0], dtype = 'float64')
        x_z.columns = {'x/z'}
    # y/z
    try:
        y_z = pd.DataFrame(data = [y.mean()/z.mean()], dtype = 'float64')
        y_z.columns = {'y/z'}

    except ZeroDivisionError:
        y_z = pd.DataFrame(data = [0], dtype = 'float64')
        y_z.columns = {'y/z'}
    # x/y
    try:
        x_y = pd.DataFrame(data = [x.mean()/y.mean()], dtype = 'float64')
        x_y.columns = {'x/y'}

    except ZeroDivisionError:
        x_y = pd.DataFrame(data = [0], dtype = 'float64')
        x_y.columns = {'x/y'}

    # FFT components
    fft_x = pd.DataFrame(data = fftpack.rfft(accelData.x))
    fft_mean_x = pd.DataFrame(data = fft_x.mean(axis = 0))
    fft_mean_x.columns = {'fft_mean_x'}
    fft_std_x = pd.DataFrame(data = fft_x.std(axis = 0))
    fft_std_x.columns = {'fft_std_x'}
    fft_skew_x = pd.DataFrame(data = fft_x.skew(axis = 0))
    fft_skew_x.columns = {'fft_skew_x'}
    fft_max_x = pd.DataFrame(data = fft_x.max(axis = 0))
    fft_max_x.columns = {'fft_max_x'}
    fft_2max_x = pd.DataFrame(data = [fft_x[0].nlargest(2).min(axis = 0)])
    fft_2max_x.columns = {'fft_2max_x'}
    fft_min_x = pd.DataFrame(data = fft_x.min(axis = 0))
    fft_min_x.columns = {'fft_min_x'}

    fft_y = pd.DataFrame(data = fftpack.rfft(accelData.y))
    fft_mean_y = pd.DataFrame(data = fft_y.mean(axis = 0))
    fft_mean_y.columns = {'fft_mean_y'}
    fft_std_y = pd.DataFrame(data = fft_y.std(axis = 0))
    fft_std_y.columns = {'fft_std_y'}
    fft_skew_y = pd.DataFrame(data = fft_y.skew(axis = 0))
    fft_skew_y.columns = {'fft_skew_y'}
    fft_max_y = pd.DataFrame(data = fft_y.max(axis = 0))
    fft_max_y.columns = {'fft_max_y'}
    fft_2max_y = pd.DataFrame(data = [fft_y[0].nlargest(2).min(axis = 0)])
    fft_2max_y.columns = {'fft_2max_y'}
    fft_min_y = pd.DataFrame(data = fft_y.min(axis = 0))
    fft_min_y.columns = {'fft_min_y'}

    fft_z = pd.DataFrame(data = fftpack.rfft(accelData.z))
    fft_mean_z = pd.DataFrame(data = fft_z.mean(axis = 0))
    fft_mean_z.columns = {'fft_mean_z'}
    fft_std_z = pd.DataFrame(data = fft_z.std(axis = 0))
    fft_std_z.columns = {'fft_std_z'}
    fft_skew_z = pd.DataFrame(data = fft_z.skew(axis = 0))
    fft_skew_z.columns = {'fft_skew_z'}
    fft_max_z = pd.DataFrame(data = fft_z.max(axis = 0))
    fft_max_z.columns = {'fft_max_z'}
    fft_2max_z = pd.DataFrame(data = [fft_z[0].nlargest(2).min(axis = 0)])
    fft_2max_z.columns = {'fft_2max_z'}
    fft_min_z = pd.DataFrame(data = fft_z.min(axis = 0))
    fft_min_z.columns = {'fft_min_z'}

    #power spectral density components
    psd_x = (fft_x*fft_x)/(n*n)
    psd_mean_x = pd.DataFrame(data = psd_x.mean(axis = 0))
    psd_mean_x.columns = {'psd_mean_x'}
    psd_max_x = pd.DataFrame(data = psd_x.max(axis = 0))
    psd_max_x.columns = {'psd_max_x'}
    psd_2max_x = pd.DataFrame(data = [psd_x[0].nlargest(2).min(axis = 0)])
    psd_2max_x.columns = {'psd_2max_x'}

    psd_y = (fft_y*fft_y)/(n*n)
    psd_mean_y = pd.DataFrame(data = psd_y.mean(axis = 0))
    psd_mean_y.columns = {'psd_mean_y'}
    psd_max_y = pd.DataFrame(data = psd_y.max(axis = 0))
    psd_max_y.columns = {'psd_max_y'}
    psd_2max_y = pd.DataFrame(data = [psd_y[0].nlargest(2).min(axis = 0)])
    psd_2max_y.columns = {'psd_2max_y'}

    psd_z = (fft_z*fft_z)/(n*n)
    psd_mean_z = pd.DataFrame(data = psd_z.mean(axis = 0))
    psd_mean_z.columns = {'psd_mean_z'}
    psd_max_z = pd.DataFrame(data = psd_z.max(axis = 0))
    psd_max_z.columns = {'psd_max_z'}
    psd_2max_z = pd.DataFrame(data = [psd_z[0].nlargest(2).min(axis = 0)])
    psd_2max_z.columns = {'psd_2max_z'}

    #create DataFrame with all features then return
    features = pd.concat([mean_x,std_x,skew_x,max_x,min_x,
                          mean_y,std_y,skew_y,max_y,min_y,
                          mean_z,std_z,skew_z,max_z,min_z,
                          x_z,y_z,x_y,
                          fft_mean_x,fft_std_x,fft_skew_x,fft_max_x,fft_2max_x,fft_min_x,
                          fft_mean_y,fft_std_y,fft_skew_y,fft_max_y,fft_2max_y,fft_min_y,
                          fft_mean_z,fft_std_z,fft_skew_z,fft_max_z,fft_2max_z,fft_min_z,
                          psd_mean_x,psd_max_x,psd_2max_x,
                          psd_mean_y,psd_max_y,psd_2max_y,
                          psd_mean_z,psd_max_z,psd_2max_z], axis = 1)



    features.dropna(how = 'all', inplace=True, axis = 0)
    return features;


def normalizeFeatures(features, y_features):
    normalizer = Normalizer()
    normalized_features = pd.DataFrame(data = normalizer.fit_transform(features, y_features), columns=features.columns, index=features.index)
    return normalized_features;

def scaleFeatures(features, y_features):
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(data = scaler.fit_transform(features, y_features), columns=features.columns, index=features.index)
    return scaled_features;

def featureSelection_n_variance_explained(features, n):
    pca = PCA(n_components= n)
    trans = pd.DataFrame(data = pca.fit_transform(features))
    print('Number of features for ',n,' variance explained:', trans.shape[1])
    return trans;

def featureSelection_pca(features, title):
    pca_trace = PCA().fit(features);
    fig, ax1 = plt.subplots(figsize = (10,6.5))
    ax1.semilogy(pca_trace.explained_variance_ratio_, '--o', label = 'explained variance ratio');
    color =  ax1.lines[0].get_color()
    ax1.set_xlabel('Principal Component', fontsize = 20);
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    plt.legend(loc=(0.4, 0.6) ,fontsize = 14);
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.semilogy(pca_trace.explained_variance_ratio_.cumsum(), '--go', label = 'cumulative explained variance ratio');
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    ax1.tick_params(axis='both', which='major', labelsize=18);
    ax1.tick_params(axis='both', which='minor', labelsize=12);
    ax2.tick_params(axis='both', which='major', labelsize=18);
    ax2.tick_params(axis='both', which='minor', labelsize=12);
    #plt.xlim([0, 29]);
    plt.legend(loc=(0.4,0.7),fontsize = 14);

    plt.savefig("featureSelection_pca_"+title, dpi=280)

def featureSelection_2d_visualise(features, y_features, title):
    pca = PCA(n_components = 2)
    trans = pca.fit_transform(features)
    # figure showing data in PC1 and PC2 axes
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    targets = [1, 2, 3, 4]
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = y_features == target
        ax.scatter(trans[indicesToKeep['label'], 0]
                   , trans[indicesToKeep['label'], 1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("2d_pca_"+title, dpi=280)
    return trans;

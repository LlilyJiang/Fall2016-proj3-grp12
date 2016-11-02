#filed feature extraction

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

def load_dataset():
    files = glob.glob("images/*")
    n = len(files)
    
    shape = (256,256)
    X = np.zeros((n,256,256))
    Y = []
    
    i = 0
    for singlefile in files:
        if singlefile[7] == 'c':
            Y.append(1)
        else:
            Y.append(0)
            
        im = Image.open(singlefile)
        im = im.convert('L')
        im = im.resize(shape)

        temp_array = np.asarray(im) 
        X[i] = temp_array
        i = i + 1
        print '%s has been loaded successfully.'%singlefile
    
    X = X.reshape(2000,1,256,256)
    Y = np.asarray(Y)
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='int32')
    return X, Y


X, Y = load_dataset()


plt.imshow(X[30][0], cmap=cm.binary) #show image testing
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 256, 256),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  

    # layer maxpool1
    maxpool1_pool_size=(4, 4),    
    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(4, 4),
    
    # dropout1
    dropout1_p=0.5,    

    
    # dense
    dense_num_units=512,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.05,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1
    )

# Train the network
nn = net1.fit(X, Y)

# Train the network
nn = net1.fit(X, Y)


#Visualize the feature
visualize.plot_conv_weights(net1.layers_['conv2d2'])

#Get weight
result = net1.get_all_params_values()
conv2d1 = result['conv2d1']
maxpool1 = result['maxpool1']
conv2d2 = result['conv2d2']
maxpool2 = result['maxpool2']




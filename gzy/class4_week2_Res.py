import numpy as np
from keras import layers
from keras.initializers import *
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Add
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from resnets_utils import *
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

def identity_block(X,f,filters,stage,block):
    '''

    :param X:
    :param f:
        integer, specifying the shape of the middle CONV's window for the main path
    :param filters:
        python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage:
        integer, used to name the layers, depending on their position in the network
    :param block:
        string/character, used to name the layers, depending on their position in the network
    :return:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    '''
    conv_name_base = "res"+str(stage)+block
    bn_name_base = "bn"+str(stage)+block

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    #main path
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2a",kernel_initializer=glorot_normal(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=glorot_normal(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2c",kernel_initializer=glorot_normal(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)

    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
#    conv_name_base = 'res' + str(stage) + block + '_branch'
#    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name_base = 'res'+str(stage)+block
    bn_name_base = 'bn'+str(stage)+block

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',
               kernel_initializer = glorot_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',
               padding = 'same', kernel_initializer = glorot_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c',
               kernel_initializer = glorot_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1',
                        kernel_initializer = glorot_normal(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(64,64,3),classes=6):
    '''
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    :return:
    model -- a Model() instance in Keras
    '''
    X_input = Input(input_shape)        #1

    X = ZeroPadding2D((3,3))(X_input)   #2

    #Stage1
    X = Conv2D(64,(7,7),strides=(2,2),name="conv1",kernel_initializer=
    glorot_normal(seed=0))(X)           #3
    X = BatchNormalization(axis=3,name="bn_conv1")(X) #4
    X = Activation("relu")(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)         #5

    #Stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)  #8
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')                      #11
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')                      #14

    #Stage3
    X = convolutional_block(X,3,filters=[128,128,512],stage=3,block="a",s=2)        #17
    X = identity_block(X,3,[128,128,512],stage=3,block="b")                         #20
    X = identity_block(X,3,[128,128,512],stage=3,block="c")                         #23
    X = identity_block(X,3,[128,128,512],stage=3,block="d")                         #26

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024],                  #29
                            stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')                 #32
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')                 #35
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')                 #38
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')                 #41
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')                 #44

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048],
                            stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_normal(seed=0))(X)

    model = Model(inputs=X_input,outputs=X,name='ReaNet50')

    return model

model = ResNet50(input_shape=(64,64,3),classes=6)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10,batch_size=32)

preds = model.evaluate(X_test,Y_test)
print("Loss = "+str(preds[0]))
print("Test Accuracy = "+str(preds[1]))

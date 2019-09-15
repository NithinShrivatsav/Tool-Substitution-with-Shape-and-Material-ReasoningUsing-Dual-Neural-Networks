from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Dropout, merge
import keras.backend as K
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model
import numpy as np
import pandas
import random
import os
import csv
import time

np.random.seed(11)

def sigmoid(z):
    return 1/(1+np.exp(-z))

if __name__ == "__main__":
    filename_embeddings = "/home/nithin/Desktop/Macgyver-Tool-Substitution/Test_Cases/final_test/Final_Datasets_ESF/Temp_SCOOP/embeddings_shape_scoop.npy"
    filename_scaler = "/home/nithin/Desktop/Macgyver-Tool-Substitution/Test_Cases/final_test/Final_Datasets_ESF/Temp_SCOOP/scaler_scoop.save"
    filename_weights = "/home/nithin/Desktop/Macgyver-Tool-Substitution/Test_Cases/final_test/Final_Datasets_ESF/Temp_SCOOP/model_weights_shape_scoop.h5"
    filename_test = "/home/nithin/Desktop/Macgyver-Tool-Substitution/Test_Cases/final_test/Final_Datasets_ESF/icra_2020_shape_test.csv"
    ## Read the test dataset
    data = pandas.read_csv(filename_test)
    # YY = data.Label
    # YY = np.array(YY)
    # YY_predict = []
    OO = data['Object']
    XX = data.drop(['Object'],axis=1)
    scaler = joblib.load(filename_scaler)
    XX = scaler.transform(XX)

    input_shape = XX.shape[1:]

    ## Build Model
    inputs = Input(input_shape)
    x = Dense(100,  activation='tanh', kernel_regularizer=regularizers.l2(0.001),name='Features1')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(100,  activation='tanh', kernel_regularizer=regularizers.l2(0.001),name='Features2')(x)
    x = Dropout(0.5)(x)
    x = Dense(25,  activation='tanh', kernel_regularizer=regularizers.l2(0.001),name='Features3')(x)
    x = Dropout(0.5)(x)

    ## Base Network
    base_network = Model(inputs=inputs, outputs=x)

    ## Create the inputs
    input_features_1 = Input(input_shape)
    input_features_2 = Input(input_shape)

    ## Tool Encodings
    tool_encoding_1 = base_network(input_features_1)
    tool_encoding_2 = base_network(input_features_2)

    ## Similarity Layer
    l1_distance_layer = Lambda(lambda tensors: K.square(tensors[0]-tensors[1]), name='L1_Distance')
    l1_distance = l1_distance_layer([tool_encoding_1, tool_encoding_2])
    prediction = Dense(1, activation='sigmoid', name='Final_Layer')(l1_distance)
    model = Model(inputs=[input_features_1, input_features_2], outputs=prediction)


    ## Compile and load the model
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['binary_accuracy'])
    model.load_weights(filename_weights)

    ## Load embeddings 
    new_tool_encoding = base_network.predict(XX)
    old_tool_encoding_full = np.load(filename_embeddings)
    old_tool_encoding = old_tool_encoding_full.sum(axis=0)/old_tool_encoding_full.shape[0]
    old_tool_encoding = old_tool_encoding.reshape(old_tool_encoding.shape[0],1)

    l1_distance_new_layer_new = np.square(new_tool_encoding.T-old_tool_encoding)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    weights_final_layer = layer_dict['Final_Layer'].get_weights()
    weights_final_layer = np.array(weights_final_layer)
    z2 = np.dot(l1_distance_new_layer_new.T,weights_final_layer[0]) + weights_final_layer[1]
    a2 = sigmoid(z2)
    for k in range(a2.shape[0]):
        print(OO[k])
        print(a2[k])

    # for i in a2:
    #     if i>0.5:
    #         YY_predict.append(1)
    #         num = num + 1
    #     else:
    #         YY_predict.append(0)
    
    # print(accuracy_score(YY, YY_predict))
    # print(num)

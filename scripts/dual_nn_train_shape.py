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

## Function to create data pairs
def data_pairs_creation(data, data_pairs, n_classes):
    pairs = []
    index_pairs = []
    labels = []
    count_pos = 0.0
    count_neg = 0.0
    n = [len(data_pairs[d]) for d in range(len(n_classes))]
    for i in range(int(n[0])):
    	for j in range(i+1,int(n[0])):
    		z1, z2 = data_pairs[1][i], data_pairs[1][j]
        	pairs.append([data[z1],data[z2]])
        	labels.append(1)
	        if j >= int(n[dn]):
	            continue
	        else:
	            z3, z4 = data_pairs[1][i], data_pairs[0][j]
	            pairs.append([data[z3],data[z4]])
	            labels.append(0)
	return np.array(pairs), np.array(labels)

if __name__ == "__main__":
    ## Read the dataset
    data = pandas.read_csv('esf_scoop_train.csv')
    Y = data.Label
    X = data.drop(['Object','Label'],axis=1)
    X, Y = shuffle(X, Y, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)

    ## Preprocessing
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # joblib.dump(scaler, 'scaler_scoop.save')

    input_shape = X_train.shape[1:]

    n_classes,_ = np.unique(Y_train, return_counts=True, axis=0)

    ## Create the training and testing pairs
    training_pairs = [np.where(Y_train==i)[0] for i in n_classes]
    x_train, y_train = data_pairs_creation(X_train, training_pairs, n_classes)
    x_train, y_train = shuffle(x_train, y_train,random_state=1)

    testing_pairs = [np.where(Y_test==i)[0] for i in n_classes]
    x_test, y_test = data_pairs_creation(X_test, testing_pairs, n_classes)

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
    # l2_distance_layer = Lambda(lambda tensors: K.square(tensors[0]-tensors[1]), name='L2_Distance')
    l1_distance = l1_distance_layer([tool_encoding_1, tool_encoding_2])
    # l2_distance = l2_distance_layer([tool_encoding_1, tool_encoding_2])

    ## Distance Fusion and Final Prediction Layer
    # concatenate_layer = merge.concatenate([l1_distance, l2_distance])
    # fusion_layer = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.001), name='Fusion')(l1_distance)
    prediction = Dense(1, activation='sigmoid', name='Final_Layer')(l1_distance)
    model = Model(inputs=[input_features_1, input_features_2], outputs=prediction)


    ## Compile and Fit the model
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['binary_accuracy'])
    model.fit([x_train[:,0], x_train[:,1]], y_train, validation_split=0.2, epochs=4000, batch_size=100)
    # model.save_weights('model_weights_shape_scoop.h5')
    results = model.predict([x_test[:,0], x_test[:,1]])
    for i in range(results.shape[0]):
    	if results[i]>=0.5:
    		results[i] = 1
    	else:
    		results[i] = 0
    results = results.flatten()
    # print(results)
    # print(y_test)
    print(classification_report(y_test,results))
    print(confusion_matrix(y_test,results))

    ## Create Embeddings
    embedding_data = pandas.read_csv('esf_scoop_embeddings.csv')
    embedding_inputs = embedding_data.drop(['Object'],axis=1)
    print(embedding_inputs.shape)
    embedding_inputs = scaler.transform(embedding_inputs)
    embedding_outputs = base_network.predict(embedding_inputs)
    # np.save('embeddings_shape_scoop.npy',embedding_outputs)
    

    ## Simple Test
    data_data = pandas.read_csv('esf_scoop_test.csv')
    XX = data_data.drop(['Object', 'Label'],axis=1)
    YY = data_data.Label
    YY = np.array(YY)
    YY_predict = []
    XX = scaler.transform(XX)
    OO = data_data['Object']
    num = 0

    new_tool_encoding = base_network.predict(XX)
    print(new_tool_encoding.T.shape)
    old_tool_encoding_full = embedding_outputs.copy()
    old_tool_encoding = old_tool_encoding_full.sum(axis=0)/old_tool_encoding_full.shape[0]
    old_tool_encoding = old_tool_encoding.reshape(old_tool_encoding.shape[0],1)
    print(old_tool_encoding.shape)

    l1_distance_new_layer_new = np.square(new_tool_encoding.T-old_tool_encoding)
    # l2_distance_new_layer_new = np.square(new_tool_encoding.T-old_tool_encoding)
    # concatenate_new_layer_new = np.concatenate([l1_distance_new_layer_new, l2_distance_new_layer_new], axis=0)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # weights_fusion_layer = layer_dict['Fusion'].get_weights()
    weights_final_layer = layer_dict['Final_Layer'].get_weights()

    # weights_fusion_layer = np.array(weights_fusion_layer)
    weights_final_layer = np.array(weights_final_layer)
    # z1 = np.dot(l1_distance_new_layer_new.T,weights_fusion_layer[0]) + weights_fusion_layer[1]
    # a1 = np.maximum(z1,0)
    # a1 = np.tanh(z1)
    # a1 = z1
    # print(a1.shape)
    z2 = np.dot(l1_distance_new_layer_new.T,weights_final_layer[0]) + weights_final_layer[1]
    a2 = sigmoid(z2)
    for k in range(a2.shape[0]):
        print(OO[k])
        print(a2[k])

    for i in a2:
        if i>0.5:
            YY_predict.append(1)
            num = num + 1
        else:
            YY_predict.append(0)
    
    print(accuracy_score(YY, YY_predict))
    print(num)

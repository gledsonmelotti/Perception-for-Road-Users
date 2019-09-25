    # -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:48:40 2018

@author: isr_deep_learning
"""

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from keras.models import load_model

import os

from tqdm import tqdm

import imageio

from skimage.transform import resize

import numpy as np

import scipy.io as sio


### Abrindo as imagens RGB
print('RGB')

train_Pedestre_dir_RGB='.../train/Pedestrian'
train_PedestreNao_dir_RGB='.../train/PedestrianNon'

test_Pedestre_dir_RGB='.../test/Pedestrian'
test_PedestreNao_dir_RGB='.../test/PedestrianNon'

validation_Pedestre_dir_RGB='.../validation/Pedestrian'
validation_PedestreNao_dir_RGB='.../validation/PedestrianNon'

train_dir_RGB='.../train'
test_dir_RGB='.../test'
validation_dir_RGB='.../validation'

print('total training Pedestrian images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)))
print('total training Non Pedestrian images-RGB:', len(os.listdir(train_PedestreNao_dir_RGB)))

print('total test Pedestrian images-RGB:', len(os.listdir(test_Pedestre_dir_RGB)))
print('total test Non Pedestrian images-RGB:', len(os.listdir(test_PedestreNao_dir_RGB)))

print('total validation Pedestrian images-RGB:', len(os.listdir(validation_Pedestre_dir_RGB)))
print('total validation Non Pedestrian images-RGB:', len(os.listdir(validation_PedestreNao_dir_RGB)))

print('total train images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)) + len(os.listdir(train_PedestreNao_dir_RGB)))
print('total validation images-RGB:', len(os.listdir(validation_Pedestre_dir_RGB)) + len(os.listdir(validation_PedestreNao_dir_RGB)))
print('total test images-RGB:', len(os.listdir(test_Pedestre_dir_RGB)) + len(os.listdir(test_PedestreNao_dir_RGB)))

print('################################')
print('total of images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)) + len(os.listdir(train_PedestreNao_dir_RGB)) + len(os.listdir(validation_Pedestre_dir_RGB)) + len(os.listdir(validation_PedestreNao_dir_RGB)) + len(os.listdir(test_Pedestre_dir_RGB)) + len(os.listdir(test_PedestreNao_dir_RGB)))
print('################################')
     

nb_train_samples = len(os.listdir(train_Pedestre_dir_RGB)) + len(os.listdir(train_PedestreNao_dir_RGB)) 
nb_validation_samples = len(os.listdir(validation_Pedestre_dir_RGB)) + len(os.listdir(validation_PedestreNao_dir_RGB))

############################## Tamanho da imagem ##############################

img_width, img_height, img_channels = 227, 227, 3

################################# TREINO  #####################################

trainP_ids = next(os.walk(train_Pedestre_dir_RGB))[2]
X_trainP = np.zeros((len(trainP_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(trainP_ids), total=len(trainP_ids)):
        path1 = train_Pedestre_dir_RGB + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        #img = np.expand_dims(img, axis=-1)
        X_trainP[n] = img
Y_trainP = np.ones((len(trainP_ids), 1), dtype=np.uint8)

trainPN_ids = next(os.walk(train_PedestreNao_dir_RGB))[2]
X_trainPN = np.zeros((len(trainPN_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(trainPN_ids), total=len(trainPN_ids)):
        path1 = train_PedestreNao_dir_RGB + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        #img = np.expand_dims(img, axis=-1)
        X_trainPN[n] = img
Y_trainPN = np.zeros((len(trainPN_ids), 1), dtype=np.uint8)

Y_train = np.concatenate((Y_trainP,Y_trainPN),axis=0)
num_classes = np.unique(Y_train).shape[0]
Y_train = np_utils.to_categorical(Y_train, num_classes) # One-hot encode the labels
    
X_train = np.concatenate((X_trainP,X_trainPN),axis=0)

del X_trainP, X_trainPN
del Y_trainP,Y_trainPN
##############################    Validação   #################################

valP_ids = next(os.walk(validation_Pedestre_dir_RGB))[2]
X_valP = np.zeros((len(valP_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(valP_ids), total=len(valP_ids)):
        path1 = validation_Pedestre_dir_RGB + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        #img = np.expand_dims(img, axis=-1)
        X_valP[n] = img
Y_valP = np.ones((len(valP_ids), 1), dtype=np.uint8)

valPN_ids = next(os.walk(validation_PedestreNao_dir_RGB))[2]
X_valPN = np.zeros((len(valPN_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(valPN_ids), total=len(valPN_ids)):
        path1 = validation_PedestreNao_dir_RGB + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        #img = np.expand_dims(img, axis=-1)
        X_valPN[n] = img
Y_valPN = np.zeros((len(valPN_ids), 1), dtype=np.uint8)

Y_val = np.concatenate((Y_valP,Y_valPN),axis=0)
num_classes = np.unique(Y_val).shape[0]
Y_val = np_utils.to_categorical(Y_val, num_classes) # One-hot encode the labels

X_val = np.concatenate((X_valP,X_valPN),axis=0)

del X_valP, X_valPN, 
del Y_valP, Y_valPN
###############################################################################

batch_size=64
num_epochs=30

DROPOUT = 0.5

model_input = Input(shape = (img_width, img_height, img_channels))
#s = Lambda(lambda x: x / 255.)(model_input)
#z = Conv2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = "relu")(s)

# First convolutional Layer
z = Convolution2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = "relu")(model_input)
z = BatchNormalization()(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)

# Second convolutional Layer
z = ZeroPadding2D(padding = (2,2))(z)
z = Convolution2D(filters = 256, kernel_size = (5,5), strides = (1,1), activation = "relu")(z)
z = BatchNormalization()(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)

# Rest 3 convolutional layers
z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
z = Flatten()(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

model_output = Dense(2, activation='softmax')(z)
model = Model(model_input, model_output)
model.summary()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

print('################################')
print('RGB')
print('################################')

from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
class LossHistory(Callback):
      def on_train_begin(self, logs={}):
          self.losses = []

      def on_batch_end(self, batch, logs={}):
          self.losses.append(logs.get('loss'))

path='.../Bilateral_13_13/CNN/Pedestre_PedestreNao_MP_1_channel.h5'
checkpointer = ModelCheckpoint(path, verbose=1, monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)

csv_logger = CSVLogger('.../Bilateral_13_13/CNN/Pedestre_PedestreNao_MP_1_channel.csv', append=True, separator=';')
History = LossHistory()

datagen_train = ImageDataGenerator(rescale=1./255)
datagen_val = ImageDataGenerator(rescale=1./255)

print('################################')
print("Fit_generator-Train the model using the training set")
print('################################')

Results_Train = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size = batch_size), 
                                    steps_per_epoch = nb_train_samples//batch_size, 
                                    epochs = num_epochs, 
                                    validation_data = datagen_val.flow(X_val,Y_val,batch_size = batch_size), #(X_val,Y_val),
                                    callbacks=[History, checkpointer, csv_logger],
                                    shuffle=True,
                                    verbose=1)

print(Results_Train.history)

# Salavndo o modelo

print("Saved model and weights finais")
model.save('.../Bilateral_13_13/CNN/Pedestre_PedestreNao_MP_1_channel_final.h5')
model.save_weights('.../Bilateral_13_13/CNN/Pedestre_PedestreNao_MP_1_channel_weights_final.h5')

############################## FAZENDO PREDICÕES ##############################
print('Prediction')

######################### Carregando o Modelo #################################
test_model = load_model(filepath='...//Bilateral_13_13/CNN/Pedestre_PedestreNao_MP_1_channel_final.h5')

DROPOUT = 0.5
img_width, img_height, img_channels = 227, 227, 1
model_input = Input(shape = (img_width, img_height, img_channels))

# First convolutional Layer (96x11x11)
z = Convolution2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = "relu")(model_input)
z = BatchNormalization()(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)

# Second convolutional Layer (256x5x5)
z = ZeroPadding2D(padding = (2,2))(z)
z = Convolution2D(filters = 256, kernel_size = (5,5), strides = (1,1), activation = "relu")(z)
z = BatchNormalization()(z)
z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)

# Rest 3 convolutional layers
z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = ZeroPadding2D(padding = (1,1))(z)
z = Convolution2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu")(z)

z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
z = Flatten()(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

z = Dense(4096, activation="relu")(z)
z = Dropout(DROPOUT)(z)

model_output = Dense(2, activation='softmax')(z)
model = Model(model_input, model_output)
model.summary()

######################### Observação ##########################################
# Considerando a primeira coluna sendo a classe de pedestres
# Considerando a segunda coluna sendo a classe de não pedestres
# Por isso 1-model.predict(x)

######################### Realizando as Predições #############################
print('Data of Test')

print('total test Pedestre images:', len(os.listdir(test_Pedestre_dir_RGB)))
print('total test PedestreNao images:', len(os.listdir(test_PedestreNao_dir_RGB)))

print('Predict and Probability to Pedestrian')
testP_ids = next(os.walk(test_Pedestre_dir_RGB))[2]
probsP= np.zeros((len(testP_ids),2),dtype=np.float32)
PredsP=np.zeros(len(testP_ids),dtype=np.float32)
def predictP(basedirP, model, testP):
    for n, id_ in tqdm(enumerate(testP), total=len(testP)):
        path1 = basedirP + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsP[n]=1-model.predict(x)
        PredsP[n]=np.argmax(probsP[n])   
        
basedirP = test_Pedestre_dir_RGB
testP=testP_ids
predictP(basedirP, test_model, testP)

print('Predict and Probability to Non Pedestrian')
testPN_ids = next(os.walk(test_PedestreNao_dir_RGB))[2]
probsPN= np.zeros((len(testPN_ids),2),dtype=np.float32)
PredsPN= np.zeros(len(testPN_ids),dtype=np.float32)
def predictPN(basedirPN, model, testPN):
    for n, id_ in tqdm(enumerate(testPN), total=len(testPN)):
        path1 = basedirPN + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsPN[n]=1-model.predict(x)
        PredsPN[n]=np.argmax(probsPN[n])   

basedirPN = test_PedestreNao_dir_RGB
testPN=testPN_ids
predictPN(basedirPN, test_model, testPN)

########################## Unindo Pedestres e Não Pedestres ###################
Test_predict = np.concatenate((PredsP,PredsPN),axis=0)
sio.savemat('.../Bilateral_13_13/CNN/Test/Test_predict.mat',{'Test_predict':Test_predict})


Probability = np.concatenate((probsP,probsPN),axis=0)
Probability_tes = Probability
sio.savemat('.../Bilateral_13_13/CNN/Test/Probability.mat',{'Probability':Probability})


############################## Predizendo o Treino ############################
print('Data of Train')

print('total Treino e Validação Pedestre imagens:', len(os.listdir(train_Pedestre_dir_RGB))+len(os.listdir(validation_Pedestre_dir_RGB)))
print('total Treino e Validação PedestreNao images:', len(os.listdir(train_PedestreNao_dir_RGB))+len(os.listdir(validation_PedestreNao_dir_RGB)))

print('Predict and Probability to Pedestrian')

trainP_ids = next(os.walk(train_Pedestre_dir_RGB))[2]
probsP1= np.zeros((len(trainP_ids),2),dtype=np.float32)
PredsP1=np.zeros(len(trainP_ids),dtype=np.float32)
def predictP1(basedirP, model, treinoP):
    for n, id_ in tqdm(enumerate(treinoP), total=len(treinoP)):
        path1 = basedirP + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsP1[n]=1-model.predict(x)
        PredsP1[n]=np.argmax(probsP1[n])   

basedirP = train_Pedestre_dir_RGB
treinoP=trainP_ids
predictP1(basedirP, test_model, treinoP)

valP_ids = next(os.walk(validation_Pedestre_dir_RGB))[2]
probsP2= np.zeros((len(valP_ids),2),dtype=np.float32)
PredsP2=np.zeros(len(valP_ids),dtype=np.float32)
def predictP2(basedirP, model, valP):
    for n, id_ in tqdm(enumerate(valP), total=len(valP)):
        path1 = basedirP + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsP2[n]=1-model.predict(x)
        PredsP2[n]=np.argmax(probsP2[n])   
        
basedirP = validation_Pedestre_dir_RGB
valP=valP_ids
predictP2(basedirP, test_model, valP)

Probabilidade_Treino_P = np.concatenate((probsP1,probsP2),axis=0)
Predicao_Treino_P = np.concatenate((PredsP1,PredsP2),axis=0)

print('Predict and Probability to Non Pedestrian')

trainPN_ids = next(os.walk(train_PedestreNao_dir_RGB))[2]
probsPN1= np.zeros((len(trainPN_ids),2),dtype=np.float32)
PredsPN1= np.zeros(len(trainPN_ids),dtype=np.float32)
def predictPN1(basedirPN, model, treinoPN):
    for n, id_ in tqdm(enumerate(treinoPN), total=len(treinoPN)):
        path1 = basedirPN + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsPN1[n]=1-model.predict(x)
        PredsPN1[n]=np.argmax(probsPN1[n])   

basedirPN = train_PedestreNao_dir_RGB
treinoPN=trainPN_ids
predictPN1(basedirPN, test_model, treinoPN)

valPN_ids = next(os.walk(validation_PedestreNao_dir_RGB))[2]
probsPN2= np.zeros((len(valPN_ids),2),dtype=np.float32)
PredsPN2=np.zeros(len(valPN_ids),dtype=np.float32)
def predictPN2(basedirPN, model, valPN):
    for n, id_ in tqdm(enumerate(valPN), total=len(valPN)):
        path1 = basedirPN + '/'+ id_
        img = imageio.imread(path1)#[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsPN2[n]=1-model.predict(x)
        PredsPN2[n]=np.argmax(probsPN2[n])   
        
basedirPN = validation_PedestreNao_dir_RGB
valPN=valPN_ids
predictPN2(basedirPN, test_model, valPN)

Probabilidade_Treino_PN = np.concatenate((probsPN1,probsPN2),axis=0)
Predicao_Treino_PN = np.concatenate((PredsPN1,PredsPN2),axis=0)

############### Unindo as probabilidades de Treino ############################
Probability_train = np.concatenate((Probabilidade_Treino_P,Probabilidade_Treino_PN),axis=0)
Probability = Probability_train 
sio.savemat('.../Bilateral_13_13/CNN/Train/Probability.mat',{'Probability':Probability})

Train_predict = np.concatenate((Predicao_Treino_P,Predicao_Treino_PN),axis=0)
sio.savemat('.../Bilateral_13_13/CNN/Train/Train_predict.mat',{'Train_predict':Train_predict})

Label_Pedestre_Verdadeiro = np.zeros((len(Probabilidade_Treino_P),1),dtype=np.float32)
Label_PedestreNao_Verdadeiro = np.ones((len(Probabilidade_Treino_PN),1),dtype=np.float32)
Train_labels = np.concatenate((Label_Pedestre_Verdadeiro,Label_PedestreNao_Verdadeiro),axis=0)
sio.savemat('.../Bilateral_13_13/CNN/Train/Train_labels.mat',{'Train_labels':Train_labels})

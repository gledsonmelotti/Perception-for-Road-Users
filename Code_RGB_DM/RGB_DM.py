from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D, Lambda
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

import matplotlib.pyplot as plt


### Open: RGB images
print('RGB')

train_Pedestre_dir_RGB='.../RGB/train/Pedestrian'
train_PedestreNao_dir_RGB='.../RGB/train/PedestrianNon'

test_Pedestre_dir_RGB='.../RGB/test/Pedestrian'
test_PedestreNao_dir_RGB='.../RGB/test/PedestrianNon'

validation_Pedestre_dir_RGB='.../RGB/validation/Pedestrian'
validation_PedestreNao_dir_RGB='.../RGB/validation/PedestrianNon'

train_dir_RGB='.../RGB/train'
test_dir_RGB='.../RGB/test'
validation_dir_RGB='.../RGB/validation'

print('total-training Pedestrian images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)))
print('total training Non Pedestrian images-RGB:', len(os.listdir(train_PedestreNao_dir_RGB)))

print('total-test Pedestrian images-RGB:', len(os.listdir(test_Pedestre_dir_RGB)))
print('total test Non Pedestrian images-RGB:', len(os.listdir(test_PedestreNao_dir_RGB)))

print('total-validation Pedestrian images-RGB:', len(os.listdir(validation_Pedestre_dir_RGB)))
print('total-validation Non Pedestrian images-RGB:', len(os.listdir(validation_PedestreNao_dir_RGB)))

print('total-train images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)) + len(os.listdir(train_PedestreNao_dir_RGB)))
print('total-validation images-RGB:', len(os.listdir(validation_Pedestre_dir_RGB)) + len(os.listdir(validation_PedestreNao_dir_RGB)))
print('total-test images-RGB:', len(os.listdir(test_Pedestre_dir_RGB)) + len(os.listdir(test_PedestreNao_dir_RGB)))

print('################################')
print('total of images-RGB:', len(os.listdir(train_Pedestre_dir_RGB)) + len(os.listdir(train_PedestreNao_dir_RGB)) + len(os.listdir(validation_Pedestre_dir_RGB)) + len(os.listdir(validation_PedestreNao_dir_RGB)) + len(os.listdir(test_Pedestre_dir_RGB)) + len(os.listdir(test_PedestreNao_dir_RGB)))
print('################################')

###########################
### Open: Bilateral Filter with 13 X 13 in size (Depth Map)
print('Depth Map-Filter Bilateral-13x13')
      
train_Pedestre_dir_DM='.../MP/train/Pedestrian'
train_PedestreNao_dir_DM='.../MP/train/PedestrianNon'

test_Pedestre_dir_DM='.../MP/test/Pedestrian'
test_PedestreNao_dir_DM='.../MP/test/PedestrianNon'

validation_Pedestre_dir_DM='.../MP/validation/Pedestrian'
validation_PedestreNao_dir_DM='.../MP/validation/PedestrianNon'

train_dir_DM='.../MP/train'
test_dir_DM='.../MP/test'
validation_dir_DM='.../MP/validation'

print('total-training Pedestrian images-DM:', len(os.listdir(train_Pedestre_dir_DM)))
print('total-training Non Pedestrian images-DM:', len(os.listdir(train_PedestreNao_dir_DM)))

print('total-test Pedestrian images-DM:', len(os.listdir(test_Pedestre_dir_DM))) 
print('total-test Non Pedestrian images-DM:', len(os.listdir(test_PedestreNao_dir_DM)))

print('total-validation Pedestrian images-DM:', len(os.listdir(validation_Pedestre_dir_DM)))
print('total-validation Non Pedestrian images-DM:', len(os.listdir(validation_PedestreNao_dir_DM)))

print('total-train images-DM:', len(os.listdir(train_Pedestre_dir_DM)) + len(os.listdir(train_PedestreNao_dir_DM)))
print('total-validation images-DM:', len(os.listdir(validation_Pedestre_dir_DM)) + len(os.listdir(validation_PedestreNao_dir_DM)))
print('total-test images-DM:', len(os.listdir(test_Pedestre_dir_DM)) + len(os.listdir(test_PedestreNao_dir_DM)))

print('################################')
print('total of images-DM:', len(os.listdir(train_Pedestre_dir_DM)) + len(os.listdir(train_PedestreNao_dir_DM)) + len(os.listdir(validation_Pedestre_dir_DM)) + len(os.listdir(validation_PedestreNao_dir_DM)) + len(os.listdir(test_Pedestre_dir_DM)) + len(os.listdir(test_PedestreNao_dir_DM)))
print('################################')


################### Número Total de Amostras de Treino e Validação
nb_train_samples = len(os.listdir(train_Pedestre_dir_DM)) + len(os.listdir(train_PedestreNao_dir_DM)) 
nb_validation_samples = len(os.listdir(validation_Pedestre_dir_DM)) + len(os.listdir(validation_PedestreNao_dir_DM))
nb_test_samples = len(os.listdir(test_Pedestre_dir_DM)) + len(os.listdir(test_PedestreNao_dir_DM)) 

print('################################')
print('total of images to Train:', nb_train_samples)
print('total of images to Validation:', nb_validation_samples)
print('total of images to Test:', nb_test_samples)
print('################################')

############################## Tamanho da imagem ##############################

img_width, img_height, img_channels = 227, 227, 4

################################# TREINO  #####################################
trainP_ids = next(os.walk(train_Pedestre_dir_DM))[2]
X_trainP = np.zeros((len(trainP_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(trainP_ids), total=len(trainP_ids)):
        path1 = train_Pedestre_dir_RGB + '/' + id_
        path2 = train_Pedestre_dir_DM + '/' + id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_trainP[n] = img
Y_trainP = np.ones((len(trainP_ids), 1), dtype=np.uint8)

trainPN_ids = next(os.walk(train_PedestreNao_dir_DM))[2]
X_trainPN = np.zeros((len(trainPN_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(trainPN_ids), total=len(trainPN_ids)):
        path1 = train_PedestreNao_dir_RGB + '/'+ id_
        path2 = train_PedestreNao_dir_DM + '/'+ id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_trainPN[n] = img
Y_trainPN = np.zeros((len(trainPN_ids), 1), dtype=np.uint8)

Y_train = np.concatenate((Y_trainP,Y_trainPN),axis=0)
num_classes = np.unique(Y_train).shape[0]
Y_train = np_utils.to_categorical(Y_train, num_classes) # One-hot encode the labels
    
X_train = np.concatenate((X_trainP,X_trainPN),axis=0)

del X_trainP, X_trainPN
del Y_trainP,Y_trainPN

##############################    Validação   #################################
valP_ids = next(os.walk(validation_Pedestre_dir_DM))[2]
X_valP = np.zeros((len(valP_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(valP_ids), total=len(valP_ids)):
        path1 = validation_Pedestre_dir_RGB + '/'+ id_
        path2 = validation_Pedestre_dir_DM + '/'+ id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_valP[n] = img
Y_valP = np.ones((len(valP_ids), 1), dtype=np.uint8)

valPN_ids = next(os.walk(validation_PedestreNao_dir_DM))[2]
X_valPN = np.zeros((len(valPN_ids), img_width, img_height, img_channels), dtype=np.uint8)
for n, id_ in tqdm(enumerate(valPN_ids), total=len(valPN_ids)):
        path1 = validation_PedestreNao_dir_RGB + '/'+ id_
        path2 = validation_PedestreNao_dir_DM + '/'+ id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
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
print('BF-DM_RGB-13')
print('################################')

from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
class LossHistory(Callback):
      def on_train_begin(self, logs={}):
          self.losses = []

      def on_batch_end(self, batch, logs={}):
          self.losses.append(logs.get('loss'))

path='.../RGB_MP/Bilateral_13_13/CNN/Model_RGB_DM_4_channel.h5'
checkpointer = ModelCheckpoint(path, verbose=1, monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)

csv_logger = CSVLogger('.../RGB_MP/Bilateral_13_13/CNN/Model_RGB_DM_4_channel.csv', append=True, separator=';')
History = LossHistory()

Gledson_datagen_train = ImageDataGenerator(rescale=1./255)
Gledson_datagen_val = ImageDataGenerator(rescale=1./255)

print('################################')
print("Fit_generator-Train the model using the training set")
print('################################')

Results_Train = model.fit_generator(Gledson_datagen_train.flow(X_train, Y_train, batch_size = batch_size), 
                                    steps_per_epoch = nb_train_samples//batch_size, 
                                    epochs = num_epochs, 
                                    validation_data = Gledson_datagen_val.flow(X_val,Y_val,batch_size = batch_size), #(X_val,Y_val),
                                    callbacks=[History, checkpointer, csv_logger],
                                    shuffle=True,
                                    verbose=1)

print(Results_Train.history)


# Salavndo o modelo

print("Saved model and weights finais")
model.save('.../RGB_MP/Bilateral_13_13/CNN/Model_RGB_DM_4_channel_final.h5')
model.save_weights('.../RGB_MP/Bilateral_13_13/CNN/Model_RGB_DM_4_channel_weights_final.h5')


print('#################################')
print('Prediction')
print('#################################')

######################### Carregando o Modelo #################################
test_model = load_model(filepath='.../RGB_MP/Bilateral_13_13/CNN/Model_RGB_DM_4_channel_final')

batch_size=64
DROPOUT = 0.5
img_width, img_height, img_channels = 227, 227, 4
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
# Considering the first column being the pedestrian class
# Considering the second column being the non-pedestrian class
# Because 1-model.predict(x)

######################### Predictions #############################
print('Data of Test')

print('total test Pedestre images:', len(os.listdir(test_Pedestre_dir_DM)))
print('total test PedestreNao images:', len(os.listdir(test_PedestreNao_dir_DM)))

print('Predict and Probability to Pedestrian')
testP_ids = next(os.walk(test_Pedestre_dir_DM))[2]
probsP= np.zeros((len(testP_ids),2),dtype=np.float32)
PredsP=np.zeros(len(testP_ids),dtype=np.float32)
def predictP(basedirP_RGB, basedirP_BFP, model, testP):
    for n, id_ in tqdm(enumerate(testP), total=len(testP)):
        path1 = basedirP_RGB + '/'+ id_
        path2 = basedirP_BFP + '/'+ id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsP[n]=1-model.predict(x)
        PredsP[n]=np.argmax(probsP[n])   
        
basedirP_RGB = test_Pedestre_dir_RGB
basedirP_BFP = test_Pedestre_dir_DM
testP=testP_ids
predictP(basedirP_RGB, basedirP_BFP, test_model, testP)

print('Predict and Probability to Non Pedestrian')
testPN_ids=next(os.walk(test_PedestreNao_dir_DM))[2]
probsPN= np.zeros((len(testPN_ids),2),dtype=np.float32)
PredsPN= np.zeros(len(testPN_ids),dtype=np.float32)
def predictPN(basedirPN_RGB, basedirPN_BFP, model, testPN):
    for n, id_ in tqdm(enumerate(testPN), total=len(testPN)):
        path1 = basedirPN_RGB + '/'+ id_
        path2 = basedirPN_BFP + '/'+ id_
        imgRGB = imageio.imread(path1)#[:,:,:img_channels]
        imgBFP = imageio.imread(path2)#[:,:,:img_channels]
        img = np.dstack((imgRGB,imgBFP))
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x = img_to_array(img)
        x= x / 255
        x = np.expand_dims(x, axis=0)
        probsPN[n]=1-model.predict(x)
        PredsPN[n]=np.argmax(probsPN[n])   

basedirPN_RGB = test_PedestreNao_dir_RGB
basedirPN_BFP = test_PedestreNao_dir_DM
testPN=testPN_ids
predictPN(basedirPN_RGB, basedirPN_BFP, test_model, testPN)

################## Concatenate Pedestrians and Non Pedestrians ################
Test_predict = np.concatenate((PredsP,PredsPN),axis=0)
sio.savemat('.../RGB_MP/Bilateral_13_13/CNN/Test/Test_predict.mat',{'Test_predict':Test_predict})

Probability = np.concatenate((probsP,probsPN),axis=0)
sio.savemat('.../RGB_MP/Bilateral_13_13/CNN/Test/Probability.mat',{'Probability':Probability})

Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
Label_Car_True = np.ones((len(PredsPN),1),dtype=np.float32)
Test_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True),axis=0)
sio.savemat('.../RGB_MP/Bilateral_13_13/CNN/Test/Test_labels.mat',{'Test_labels':Test_labels})

#To compute per-label precisions, recalls, F1-scores and supports instead of averaging:
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support=precision_recall_fscore_support(Test_labels, Test_predict, average=None, labels=[0, 1])

print('########################################################################')
print('The F1-scores of Pedestrians and Non Pedestrian are:')
print(fscore)
print('########################################################################')
print('F1-scores average is:')
print(np.sum(fscore)/3.)
print('########################################################################')
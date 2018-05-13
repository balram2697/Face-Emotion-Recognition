from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

from ResNet import resnetwork

batch_size = 32
num_epochs = 1000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
base_path = './models/emotion_models/'

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
						

						
model = resnetwork(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    trained_models_path = base_path + 'model_resnet'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)

    image_size=input_shape[:2]
    dataset_path = './fer2013/fer2013.csv'
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

 			
    faces = faces.astype('float32')
    faces = faces / 255.0
    faces = faces - 0.5
    faces = faces * 2.0

    num_samples, num_classes = emotions.shape
	
    num_samples = len(faces)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = faces[:num_train_samples]
    train_y = emotions[:num_train_samples]
    val_x = faces[num_train_samples:]
    val_y = emotions[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
	
    train_faces, train_emotions = train_data
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1,callbacks=None,
                        validation_data=val_data)







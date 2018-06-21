# -*- coding: utf-8 -*-
"""Leverage a pre-trained network (saved network previously trained on a large dataset)
in order to build an image recognition system and analyse traffic.

Transfer image representations from popular deep learning models.

[A] ConvNet as fixed feature extractor.`Feature extraction` will simply consist of taking the convolutional base
of a previously-trained network, running the new data through it, and training a new classifier on top of the output.
(i.e. train only the randomly initialized top layers while freezing all convolutional layers of the original model).

# References
- [https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2]
- [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]

"""

import os
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import matplotlib.pyplot as plt

from applications.vgg16_places_365 import VGG16_Places365

import datetime

# Preparation actions

now = datetime.datetime.now


epochs = 1

# Base directory of raw jpg/png images
base_dir = '/home/sandbox/GKalliatakis-GitHub Account/Traffic-Analysis/dataset/MotorwayTraffic'

# Base directory for saving the trained models
base_dir_trained_models = '/home/sandbox/GKalliatakis-GitHub Account/Traffic-Analysis/trained_models/'
bottleneck_features_dir = os.path.join(base_dir_trained_models, 'bottleneck_features/')
logs_dir = os.path.join(base_dir_trained_models, 'logs/')


train_dir = os.path.join(base_dir, 'train')
nb_train_samples = 360


val_dir = os.path.join(base_dir, 'val')
nb_val_samples = 40


classes = ['empty', 'fluid', 'heavy', 'jam']

# https://groups.google.com/forum/#!topic/keras-users/MUO6v3kRHUw
# To train unbalanced classes 'fairly', we want to increase the importance of the under-represented class(es).
# To do this, we need to chose a reference class. You can pick any class to serve as the reference, but conceptually,
# I like the majority class (the one with the most samples).
# Creating your class_weight dictionary:
# 1. determine the ratio of reference_class/other_class. If you choose class_0 as your reference,
# you'll have (1000/1000, 1000/500, 1000/100) = (1,2,10)
# 2. map the class label to the ratio: class_weight={0:1, 1:2, 2:10}
# class_weight = {0: 5.08, 1: 1, 2: 10.86, 3: 5.08, 4: 3.46, 5: 2.31, 6: 4.70, 7: 6.17, 8: 1.55}

# Augmentation configuration with only rescaling.
# Rescale is a value by which we will multiply the data before any other processing.
# Our original images consist in RGB coefficients in the 0-255, but such values would
# be too high for our models to process (given a typical learning rate),
# so we target values between 0 and 1 instead by scaling with a 1/255. factor.
train_datagen = ImageDataGenerator(rescale=1. / 255)

# This is the augmentation configuration we will use for training when data_augm_enabled argument is True
train_augmented_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')


val_datagen = ImageDataGenerator(rescale=1. / 255)

img_width, img_height = 224, 224

batch_size = 10


train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    classes=classes, class_mode='categorical',
                                                    batch_size=batch_size)

augmented_train_generator = train_augmented_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                                        classes=classes, class_mode='categorical',
                                                                        batch_size=batch_size)


val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
                                                    classes=classes, class_mode='categorical',
                                                    batch_size=batch_size)


steps_per_epoch = nb_train_samples // batch_size
validation_steps = nb_val_samples // batch_size




def retrain_classifier(pre_trained_model='VGG16',
                       pooling_mode='avg',
                       classes=4,
                       data_augm_enabled = False):
    """ConvNet as fixed feature extractor, consist of taking the convolutional base of a previously-trained network,
    running the new data through it, and training a new classifier on top of the output.
    (i.e. train only the randomly initialized top layers while freezing all convolutional layers of the original model).

    # Arguments
        pre_trained_model: one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`
        pooling_mode: Optional pooling_mode mode for feature extraction
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling_mode
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling_mode will
                be applied.
        classes: optional number of classes to classify images into.
        data_augm_enabled: whether to augment the samples during training

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `pre_trained_model`, `pooling_mode` or invalid input shape.
    """


    if not (pre_trained_model in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
        raise ValueError('The `pre_trained_model` argument should be either '
                         '`VGG16`, `VGG19`, `ResNet50`, '
                         'or `VGG16_Places365`. Other models will be supported in future releases. ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')


    # Define the name of the model and its weights
    if data_augm_enabled == True:
        filepath = bottleneck_features_dir+'augm_bottleneck_features_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'
        log_filepath = logs_dir + 'augm_' + pre_trained_model + '_' + pooling_mode + '_log.csv'
    else:
        filepath = bottleneck_features_dir+'bottleneck_features_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'
        log_filepath = logs_dir + pre_trained_model + '_' + pooling_mode + '_log.csv'


    # ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1,
                                   save_weights_only=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    csv_logger = CSVLogger(log_filepath, append=True, separator=',')

    callbacks_list = [checkpointer, early_stop, csv_logger]


    input_tensor = Input(shape=(224, 224, 3))

    # create the base pre-trained model for warm-up
    if pre_trained_model == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'VGG16_Places365':
        base_model = VGG16_Places365(weights='places', include_top=False, input_tensor=input_tensor)

    print ('\n \n')
    print('[INFO] Vanilla `' + pre_trained_model + '` pre-trained convnet was successfully initialised.\n')


    x = base_model.output

    # Now we set-up transfer learning process - freeze all but the penultimate layer
    # and re-train the last Dense layer with `classes` number of final outputs representing probabilities for the different classes.
    # Build a  randomly initialised classifier model to put on top of the convolutional model

    # both `avg`and `max`result in the same size of the Dense layer afterwards
    # Both Flatten and GlobalAveragePooling2D are valid options. So is GlobalMaxPooling2D.
    # Flatten will result in a larger Dense layer afterwards, which is more expensive
    # and may result in worse overfitting. But if you have lots of data, it might also perform better.
    # https://github.com/keras-team/keras/issues/8470
    if pooling_mode == 'avg':
        x = GlobalAveragePooling2D(name='GAP')(x)
    elif pooling_mode == 'max':
        x = GlobalMaxPooling2D(name='GMP')(x)
    elif pooling_mode == 'flatten':
        x = Flatten(name='FLATTEN')(x)


    x = Dense(256, activation='relu', name='FC1')(x)  # let's add a fully-connected layer

    # When random init is enabled, we want to include Dropout,
    # otherwise when loading a pre-trained HRA model we want to omit
    # Dropout layer so the visualisations are done properly (there is an issue if it is included)
    x = Dropout(0.5, name='DROPOUT')(x)
    # and a logistic layer with the number of classes defined by the `classes` argument
    predictions = Dense(classes, activation='softmax', name='PREDICTIONS')(x)  # new softmax layer

    # this is the transfer learning model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    print('[INFO] Randomly initialised classifier was successfully added on top of the original pre-trained conv. base. \n')

    print('[INFO] Number of trainable weights before freezing the conv. base of the original pre-trained convnet: '
          '' + str(len(model.trainable_weights)))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers of the preliminary base model
    for layer in base_model.layers:
        layer.trainable = False

    print('[INFO] Number of trainable weights after freezing the conv. base of the pre-trained convnet: '
          '' + str(len(model.trainable_weights)))

    print ('\n')

    # compile the warm_up_model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    # # The attribute model.metrics_names will give you the display labels for the scalar outputs.
    # print warm_up_model.metrics_names

    if data_augm_enabled:
        print('[INFO] Using augmented samples for training. This may take a while ! \n')

        t = now()

        history = model.fit_generator(augmented_train_generator,
                                      epochs= epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data = val_generator,
                                      validation_steps= validation_steps,
                                      callbacks=callbacks_list)

        print('[INFO] Training time for re-training the last dense layer using augmented samples: %s' % (now() - t))

        elapsed_time = now() - t



    else:
        t = now()
        history = model.fit_generator(train_generator,
                                      epochs= epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data = val_generator,
                                      validation_steps= validation_steps,
                                      callbacks=callbacks_list)

        print('[INFO] Training time for re-training the last dense layer: %s' % (now() - t))

        elapsed_time = now() - t

        print ('\n')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    elapsed_time_entry = pre_trained_model + '_' + pooling_mode + ': '+ str(elapsed_time)

    file = open('elapsed_time.txt', 'a+')

    file.write(elapsed_time_entry)

    file.close()

    return model, elapsed_time





if __name__ == "__main__":

    transfer_learning_model = retrain_classifier(pre_trained_model='VGG16',
                                                 pooling_mode='avg',
                                                 data_augm_enabled=False)





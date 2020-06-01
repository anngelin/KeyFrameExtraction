import keras
import math
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import mobilenet, mobilenet_v2, MobileNet
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from keras.regularizers import l2
from keras.losses import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
# from keras.utils import CustomObjectScope
from keras.utils.generic_utils import CustomObjectScope

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input

FIIQA_DATASET_PATH = '/home/datasets/fiiqa'
FIIQA_TRAIN = FIIQA_DATASET_PATH + '/train-faces'
FIIQA_TEST = FIIQA_DATASET_PATH + '/test-faces'
FIIQA_TRAIN_STANDARD = FIIQA_DATASET_PATH + '/train_standard.txt'
FIIQA_TEST_STANDARD = FIIQA_DATASET_PATH + '/test_standard.txt'

MODEL = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/vgg2_mobilenet.h5'

# MODEL = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\vgg2_mobilenet.h5'

input_height = input_width = 224  #192 #128 #192 #224 #200
activ_func, loss_func, class_mode, metric, monitor = 'softmax', categorical_crossentropy, 'categorical', \
                                                     'accuracy', 'val_acc'

epochs_num = 20

# NUM of quality scores
CLASSES_NUM = 3


def parse_dataset(train=True):
    standard_file = FIIQA_TRAIN_STANDARD if train else FIIQA_TEST_STANDARD
    dataset_path = FIIQA_TRAIN if train else FIIQA_TEST

    X = []
    y = []

    with open(standard_file, encoding='utf-8', mode='r') as fr:
        image_lines = fr.readlines()
        for line in image_lines:
            line_parts = line.strip().split(' ')
            quality_score = line_parts[1]
            img_path = dataset_path + '/' + str(line_parts[0])
            X.append(img_path)
            y.append(str(quality_score))

    return np.array(X), np.array(y)


def get_generator(imgs, labels, preprocess_input_func, target_size):

    for img, label in zip(imgs, labels):
        try:
            image = load_img(str(img), target_size=target_size)
        except Exception as e:
            print(e)
            break

        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input_func(image)

        yield image, label


def get_dataframe_generator(imgs, labels):
    df = pd.DataFrame({'filename': imgs, 'class': labels})

    print(df.head())

    return df


def get_generator_lenet(imgs, labels, target_size):
    for img, label in zip(imgs, labels):
        try:
            image = load_img(str(img), target_size=target_size)
        except Exception as e:
            print(e)
            break

        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        yield image, label


def get_net():
    # 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
    # model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
    # model.summary()
    # with CustomObjectScope({'relu6': keras.layers.ReLU(6.)}):
    #     model = load_model(MODEL)
    #     # model.layers.pop()
    #     # model.layers.pop()
    #
    #     print(model.summary())
    #
    #     return model

    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
    # Load saved weights: vgg2_mobilenet
    base_model.load_weights(MODEL, by_name=True)

    return base_model


def get_predictive_model(base_model, classes_num):
    last_model_layer = base_model.output
    x = GlobalAveragePooling2D(name='reshape_2')(last_model_layer)
    preds = Dense(classes_num, activation=activ_func, kernel_regularizer=l2(4e-5), name='predictions')(x)
    f_model = Model(base_model.input, preds)

    return f_model


def model_architecture(classes_num):
    base_model = get_net()
    return get_predictive_model(base_model, classes_num), base_model


def train_tune_model_mobilenet():
    model, base_model = model_architecture(CLASSES_NUM)

    train_batch_size = val_batch_size = 32  # 76 # 88 #104 #128

    num_layers_to_freeze = len(base_model.layers)
    #  Freeze layers
    for l in model.layers[:num_layers_to_freeze]:
        l.trainable = False
    for l in model.layers[num_layers_to_freeze:]:
        l.trainable = True

    train_imgs, train_labels = parse_dataset()
    train_features_len = len(train_imgs)

    test_imgs, test_labels = parse_dataset(train=False)
    test_features_len = len(test_imgs)

    target_size = (input_height, input_width)

    train_df = get_dataframe_generator(train_imgs, train_labels)
    test_df = get_dataframe_generator(test_imgs, test_labels)

    train_datagen = ImageDataGenerator(shear_range=0.3,  # 0.2
                                       rotation_range=10,
                                       zoom_range=0.2,  # 0.1
                                       width_shift_range=0.1, height_shift_range=0.1,
                                       horizontal_flip=True, preprocessing_function=preprocess_input)

    # train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    #
    #
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        target_size=target_size,
        batch_size=train_batch_size)
    #
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        target_size=target_size,
        batch_size=val_batch_size)
    #
    # nb_train_samples = train_generator.samples
    # classes_num = train_generator.num_classes
    # nb_validation_samples = test_generator.samples
    # print('after read', nb_train_samples, nb_validation_samples, classes_num)
    #
    # model_checkpoint = ModelCheckpoint('mobilenet_vgg_fiiqa_finetuned.h5', monitor=monitor, verbose=1,
    #                                    save_best_only=True)

    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])
    #
    filepath = 'mobilenet_vgg2_fiiqa'
    last_path = '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_acc', patience=2)
    callbacks = [checkpoint, es]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / train_batch_size),
        epochs=epochs_num,
        validation_data=test_generator,
        validation_steps=math.ceil(test_features_len / val_batch_size),
        callbacks=callbacks)

    model.save(filepath + '.hdf5')

    # Unfreeze layers and tune all

    # reset our data generators

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    for l in model.layers[:num_layers_to_freeze]:
        l.trainable = True
    for l in model.layers[num_layers_to_freeze:]:
        l.trainable = True

    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])

    filepath = 'mobilenet_vgg2_fiiqa_finetuned'
    last_path = '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_acc', patience=2)
    callbacks = [checkpoint, es]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / train_batch_size),
        epochs=epochs_num,
        validation_data=test_generator,
        validation_steps=math.ceil(test_features_len / val_batch_size),
        callbacks=callbacks)

    model.save(filepath + '.hdf5')







def train_tune_model():
    model, base_model = model_architecture(CLASSES_NUM)
    train_batch_size = val_batch_size = 32
    num_layers_to_freeze = len(base_model.layers)
    #  Freeze layers
    for l in model.layers[:num_layers_to_freeze]:
        l.trainable = False

    train_imgs, train_labels = parse_dataset()
    train_features_len = len(train_imgs)

    test_imgs, test_labels = parse_dataset(train=False)
    test_features_len = len(test_imgs)

    target_size = (input_height, input_width)

    train_df = get_dataframe_generator(train_imgs, train_labels)
    test_df = get_dataframe_generator(test_imgs, test_labels)

    train_datagen = ImageDataGenerator(shear_range=0.3,  # 0.2
                                       rotation_range=10,
                                       zoom_range=0.2,  # 0.1
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,
                                       preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        target_size=target_size,
        batch_size=train_batch_size)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        target_size=target_size,
        batch_size=val_batch_size)

    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])

    filepath = 'mobilenet_vgg2_fiiqa'
    last_path = '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_acc', patience=2)
    callbacks = [checkpoint, es]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / train_batch_size),
        epochs=epochs_num,
        validation_data=test_generator,
        validation_steps=math.ceil(test_features_len / val_batch_size),
        callbacks=callbacks)

    # Unfreeze layers and tune all
    for l in model.layers[:num_layers_to_freeze]:
        l.trainable = True
    for l in model.layers[num_layers_to_freeze:]:
        l.trainable = True

    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])

    filepath = 'mobilenet_vgg2_fiiqa_finetuned'
    last_path = '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_acc', patience=2)
    callbacks = [checkpoint, es]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / train_batch_size),
        epochs=epochs_num,
        validation_data=test_generator,
        validation_steps=math.ceil(test_features_len / val_batch_size),
        callbacks=callbacks)

    model.save(filepath + '.hdf5')


def preprocess_image_lenet(x):
    """
    image = [1, image.shape[0], image.shape[1], image.shape[2]]
    """
    pass


def create_model_lenet(classes_num):

    input_shape = (1, input_height, input_width, 3)

    model = keras.Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(GlobalAveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(classes_num, activation='softmax'))

    return model


# Simple lenet-5 architecture
def train_lenet():
    model = create_model_lenet(CLASSES_NUM)

    train_batch_size = val_batch_size = 32  # 76 # 88 #104 #128

    train_imgs, train_labels = parse_dataset()
    train_features_len = len(train_imgs)

    test_imgs, test_labels = parse_dataset(train=False)
    test_features_len = len(test_imgs)

    target_size = (input_height, input_width)

    train_generator = get_generator_lenet(train_imgs, train_labels, target_size)

    test_generator = get_generator_lenet(test_imgs, test_labels, target_size)

    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss=loss_func, optimizer=opt, metrics=[metric])
    #
    filepath = 'lenet_fiiqa'
    last_path = '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor=monitor, verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_acc', patience=2)
    callbacks = [checkpoint, es]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / train_batch_size),
        epochs=16,
        validation_data=test_generator,
        validation_steps=math.ceil(test_features_len / val_batch_size),
        callbacks=callbacks)

    model.save(filepath + '.hdf5')


if __name__ == '__main__':
    train_tune_model_mobilenet()

import keras
import cv2
import numpy as np
import pathlib
import pandas as pd
import math
from keras.models import load_model, Model
from typing import NamedTuple
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import mobilenet
from keras.losses import mean_squared_error as regr_logloss
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Lambda, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, AveragePooling2D, SeparableConv2D, Conv2D, MaxPooling2D
from keras import backend as K

TARGET_SIZE = (224, 224) #(250, 250)
BATCH_SIZE = 32

DATA_DIR = '/home/datasets/vggface2/'
ALL_DATA_DIR = DATA_DIR+'all/'
PREPARED_DATA_DIR = DATA_DIR+'faces_224/'
TRAIN_DATA_DIR = PREPARED_DATA_DIR+'train'
VAL_DATA_DIR = PREPARED_DATA_DIR+'val'
print(TRAIN_DATA_DIR, VAL_DATA_DIR)

METADATA_PATH = DATA_DIR + 'identity_meta.csv'
TEACHER_MODEL_PATH = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/FaceQnet.h5'

TEMPERATURE = 3.25


# class TeacherFeatures(NamedTuple):
#     img_id: str
#     feature_vec: np.array


def get_imgs_dirs(n=300):
    dirs = []
    with open(METADATA_PATH, mode='r', encoding='utf-8') as fr:
        data_lines = fr.readlines()

        for line in data_lines[1:]:
            line = line.strip().split(',')
            face_dir = line[0]
            face_dir = TRAIN_DATA_DIR + '/' + face_dir
            dirs.append(face_dir)

    return dirs[:n]


def get_imgs_path():
    dirs = get_imgs_dirs()
    imgs = []

    for dir in dirs:
        dir = pathlib.Path(dir)
        imgs_in_dir = list(dir.glob('*.jpg'))
        imgs_in_dir.extend(dir.glob('*.png'))
        for img in imgs_in_dir:
            imgs.append(img)

    return imgs


def get_teacher_outputs(pop_layers=True, output='layer_extra1'):
    basemodel = load_model(TEACHER_MODEL_PATH)

    if pop_layers:
        model = Model(inputs=basemodel.input, outputs=basemodel.get_layer(output).output)
        file_to_save = 'face_qnet_resnet_%s.csv'%(output)
    else:
        model = basemodel
        file_to_save = 'face_qnet_resnet_main_output.csv'

    train_imgs = get_imgs_path()

    with open(file_to_save, mode='w') as fr:
        for imgpath in train_imgs:
            imgpath = str(imgpath)
            print(imgpath)
            face = cv2.imread(imgpath)
            face = cv2.resize(face, TARGET_SIZE)
            frames = []
            frames.append(face)
            X = np.array(frames, dtype=np.float32)
            scores = model.predict(X, batch_size=1, verbose=1)

            str_scores = ','.join([str(score) for score in scores[0]])
            fr.write(imgpath + ',')
            fr.write(str_scores)
            fr.write('\n')


def get_teacher_soften_outputs():
    basemodel = load_model(TEACHER_MODEL_PATH)

    teacher_logits = Model(basemodel.input, outputs=basemodel.get_layer('flatten_1').output)

    T_layer = Lambda(lambda x: x / TEMPERATURE)(teacher_logits.output)

    # base_layer_extra1 = Dense(32, input_dim=512, kernel_initializer='normal', activation='relu')(T_layer)
    # base_main_output = Dense(1, kernel_initializer='normal')(base_layer_extra1)

    model = Model(inputs=teacher_logits.input, outputs=T_layer)

    file_to_save = 'face_qnet_resnet_main_soften_output.csv'

    train_imgs = get_imgs_path()

    with open(file_to_save, mode='w') as fr:
        for imgpath in train_imgs:
            imgpath = str(imgpath)
            print(imgpath)
            face = cv2.imread(imgpath)
            face = cv2.resize(face, TARGET_SIZE)
            frames = []
            frames.append(face)
            X = np.array(frames, dtype=np.float32)
            scores = model.predict(X, batch_size=1, verbose=1)

            str_scores = ','.join([str(score) for score in scores[0]])
            fr.write(imgpath + ',')
            fr.write(str_scores)
            fr.write('\n')


# def read_features(file_path):
#     features = []
#
#     with open(file_path, mode='r') as fr:
#         lines = fr.readlines()
#         for line in lines:
#             line = line.strip().split(',')
#             img_name = line[0]
#             img_features = line[1:]
#             img_features = np.array(img_features).astype(np.float32)
#             features.append(TeacherFeatures(img_name, img_features))
#
#     return features


def load_teacher_logits(filepath):
    logits = []

    with open(filepath, mode='r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(',')
            model_logits = np.array(line[1:]).astype(np.float32)
            logits.append(model_logits)

    return np.array(logits)


def load_teacher_preds(filepath):
    preds = []

    with open(filepath, mode='r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(',')
            model_preds = float(line[1])
            preds.append(model_preds)

    return preds


def load_teacher_train_dataset(filepath):
    imgs = []

    with open(filepath, mode='r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(',')
            imgs.append(line[0])

    return imgs


def create_train_generator(imgs, preds):

    df = pd.DataFrame({'filename': imgs, 'preds': preds})

    print(df.head())

    return df


def knowledge_distillation_loss(y_true, y_pred, lambda_const):
    # split in
    #    onehot hard true targets
    #    logits
    y_true, logits = y_true[:, :32], y_true[:, 32:]

    # convert logits to soft targets
    # y_soft = K.softmax(logits / TEMPERATURE)
    y_soft = K.relu(logits / TEMPERATURE)

    # split in
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :32], y_pred[:, 32:]

    return lambda_const * regr_logloss(y_true, y_pred) + (1 - lambda_const)*regr_logloss(y_soft, y_pred_soft)


# # logloss with only soft probabilities and targets
# def soft_logloss(y_true, y_pred):
#     logits = y_true[:, 256:]
#     y_soft = K.softmax(logits/TEMPERATURE)
#     y_pred_soft = y_pred[:, 256:]
#     return logloss(y_soft, y_pred_soft)


# TODO: IS IT RIGHT????????????????
def create_student_model():
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D(name='reshape_2')(base_model.output)
    base_layer_extra1 = Dense(32, input_dim=32, kernel_regularizer=l2(4e-5), activation='relu')(x)
    base_main_output = Dense(1, kernel_initializer='normal')(base_layer_extra1)
    model = Model(base_model.input, base_main_output)

    return model


def create_student_model_small():
    INPUT_SIZE = 224
    model = Sequential()
    model.add(Conv2D(112, (3, 3), activation='relu',
                     input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(1, kernel_initializer='normal'))

    return model


# def knowledge_distillation_loss_withBE(y_true, y_pred, beta=0.1):
#     # Extract the groundtruth from dataset and the prediction from teacher model
#     y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]
#
#     # Extract the prediction from student model
#     y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]
#
#     loss = beta * binary_crossentropy(y_true, y_pred) + (1 - beta) * binary_crossentropy(y_pred_teacher, y_pred_stu)
#
#     return loss

def create_student_model():
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D(name='reshape_2')(base_model.output)
    base_layer_extra1 = Dense(32, input_dim=32, kernel_regularizer=l2(4e-5), activation='relu')(x)
    base_main_output = Dense(1, kernel_initializer='normal')(base_layer_extra1)
    model = Model(base_model.input, base_main_output)

    return model

def train_student_teacher():
    teacher_logits_path_train = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_qnet_resnet_flatten_1.csv'
    teacher_logits_path_val = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_val_qnet_resnet_flatten_1.csv'

    teacher_logits_soften_path_train = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_qnet_resnet_soften.csv'
    teacher_logits_soften_path_val = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_val_qnet_resnet_soften.csv'

    teacher_preds_path_train = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_qnet_resnet_main_output.csv'
    teacher_preds_path_val = '/home/akharchevnikova/Face_Image_Quality_Assessment/face_val_qnet_resnet_main_output.csv'

    teacher_logits_train = load_teacher_logits(teacher_logits_path_train)
    teacher_logits_val = load_teacher_logits(teacher_logits_path_val)
    teacher_preds_train = load_teacher_preds(teacher_preds_path_train)
    teacher_preds_val = load_teacher_preds(teacher_preds_path_val)

    teacher_train_dataset = load_teacher_train_dataset(teacher_preds_path_train)
    teacher_val_dataset = load_teacher_train_dataset(teacher_preds_path_val)

    train_features_len = len(teacher_train_dataset)
    val_features_len = len(teacher_val_dataset)

    train_df = create_train_generator(teacher_train_dataset, teacher_preds_train)
    val_df = create_train_generator(teacher_val_dataset, teacher_preds_val)

    train_datagen = ImageDataGenerator(shear_range=0.3,  # 0.2
                                       rotation_range=10,
                                       zoom_range=0.2,  # 0.1
                                       width_shift_range=0.1, height_shift_range=0.1,
                                       horizontal_flip=True, preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="preds",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw')
    #
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="preds",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw')

    # student_model = create_student_model()
    student_model = create_student_model_small()

    opt = Adam(lr=1e-3, decay=1e-5)
    student_model.compile(loss=regr_logloss, optimizer=opt, metrics=['mse', 'mae'])

    # filepath = 'mobilenet_faceqnet'
    filepath = 'model_small'
    last_path = '-{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath + last_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_loss', patience=3)
    callbacks = [checkpoint, es]

    history = student_model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_features_len / BATCH_SIZE),
        epochs=50,
        validation_data=val_generator,
        validation_steps=math.ceil(val_features_len / BATCH_SIZE),
        callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist.to_csv('faceqnet_history_small.csv')

    student_model.save(filepath + '.hdf5')


if __name__ == '__main__':
    train_student_teacher()

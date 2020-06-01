import csv
import numpy as np
import math
import pandas as pd
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from keras.layers import Dense,GlobalAveragePooling2D

from keras.optimizers import SGD



class TrainGenerator(object):
    """Acts as an adapter of `Dataset` for Keras' `fit_generator` method."""
    def __init__(self,
                 batch_size,
                 photo_features_train,
                 photo_features_test,
                 tokenizer,
                 captions,
                 max_length,
                 vocab_size,
                 max_cache_size):

        self._batch_size = batch_size
        self._photo_features_train = photo_features_train
        self._photo_features_test = photo_features_test
        self._tokenizer = tokenizer
        self._captions = captions
        self._max_length = max_length
        self._vocab_size = vocab_size
        self._max_cache_size = max_cache_size

    def generate_data(self, train=True):
        X1, X2, y = list(), list(), list()

        n = 0

        photo_features = self._photo_features_train if train else self._photo_features_test

        while True:
            for img_id, features in photo_features:
                description = self._captions[img_id]

                seq = self._tokenizer.texts_to_sequences([description])[0]

                n += 1
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self._max_length)[0]

                    # encode output sequence

                    out_seq = to_categorical([out_seq], num_classes=self._vocab_size)[0]

                    # store
                    X1.append(features)
                    X2.append(in_seq)
                    y.append(out_seq)

                if n == self._batch_size:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    X1, X2, y = list(), list(), list()
                    n = 0

    def generate_by_cache(self, img_file, eof):
        X1, X2, y = list(), list(), list()

        img_cache = []

        while True:

            with img_file.open('r', encoding='utf-8') as fr:
                data = csv.reader(fr, delimiter=',')
                for i, row in enumerate(data):
                    img_cache.append(row)

                    if (len(img_cache) == self._max_cache_size) or (i == eof):

                        n = 0
                        for img in img_cache:
                            img_id = img[0]
                            feature = img[1:]
                            description = self._captions[img_id]

                            seq = self._tokenizer.texts_to_sequences([description])[0]
                            n+=1

                            for i in range(1, len(seq)):
                                in_seq, out_seq = seq[:i], seq[i]
                                in_seq = pad_sequences([in_seq], maxlen=self._max_length)[0]
                                out_seq = to_categorical([out_seq], num_classes=self._vocab_size)[0]

                                X1.append(feature)
                                X2.append(in_seq)
                                y.append(out_seq)

                            if n == self._batch_size:
                                yield [[np.array(X1), np.array(X2)], np.array(y)]
                                X1, X2, y = list(), list(), list()
                                n = 0
                        img_cache = []

    def generate_by_chunk(self, img_file):
        X1, X2, y = list(), list(), list()

        while True:
            for chunk in pd.read_csv(img_file, header=None, chunksize=self._max_cache_size, delimiter='\t'):
                iter = chunk.iterrows()
                img_cache = []

                for i in range(chunk.shape[0]):
                    x = next(iter)
                    img_cache.append(x[1][0])

                n = 0
                for img in img_cache:
                    img_data = img.split(',')
                    img_id, features = img_data[0], img_data[1:]

                    description = self._captions[img_id]
                    seq = self._tokenizer.texts_to_sequences([description])[0]
                    n += 1

                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=self._max_length)[0]

                        X1.append(features)
                        X2.append(in_seq)
                        y.append(out_seq)

                    if n == self._batch_size:
                        yield [[np.array(X1), np.array(X2)], np.array(y)]
                        X1, X2, y = list(), list(), list()
                        n = 0

    # define the captioning model
    def define_model(self):
        # feature extractor model
        inputs1 = Input(shape=(1280,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        # sequence model
        inputs2 = Input(shape=(self._max_length,))
        se1 = Embedding(self._vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self._vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        # compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        # summarize model
        model.summary()
        # plot_model(model, to_file='model.png', show_shapes=True)
        return model


# load clean descriptions into memory
def load_clean_descriptions(filename):
    descriptions = dict()
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter=',')
        for row in data:
            image_id = row[0]
            image_desc = row[1].split()
            descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
    return descriptions


# load photo features
def get_steps_by_features(filename):
    # load photo features from file
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        i = 0
        for i, _ in enumerate(data):
            pass
        return i


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    all_desc = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = list(descriptions.values())
    return max(len(d.split()) for d in lines)


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter=',')
        for row in data:
            yield row[0], row[1:]


def load_data(filepath):
    X = []
    y = []

    with filepath.open('r') as fr:
        lines = fr.readlines()

        for line in lines:
            # data = line.split(',')

            img = line.strip()
            img = Path(img).resolve()

            label = Path(img).parts[1]

            X.append(image)
            y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

    return X_train, X_test, y_train, y_test


def get_generator(data, prep_img, target_size):
    for img in data:
        try:
            image = load_img(str(img), target_size=target_size)
        except Exception as e:
            print(e)
            continue

        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = prep_img(image)
        yield image


def fine_tune(model_type, batch_size, epoch, X_train):
    prep_img = preprocess_input
    target_size = (224, 224)

    if model_type == 0:
        base_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), alpha=1.0, pooling='avg', include_top=False)
        target_size = (224, 224)

    elif model_type == 1:
        base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3), pooling='avg', include_top=False)
        prep_img = preprocess_input_inception_v3
        target_size = (299, 299)


    # Number of preferences for classifier
    num_class = 10

    x = base_model.output
    x = Dense(num_class, activation='softmax', use_bias=True)(x)
    model=Model(base_model.inputs, x)

    for l in base_model.layers:
        l.trainable = False
    
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mc = ModelCheckpoint('google_model_finetuned.h5', monitor='val_acc', verbose=1, save_best_only=True)

    train_generator = get_generator(X_train, prep_img, target_size)

    train_features_len = len(list(train_generator))

    history1 = model.fit_generator(
      train_generator,
      steps_per_epoch=math.ceil(train_features_len / batch_size),
      epochs=epoch,
      callbacks=[mc])

    return model


if __name__ == '__main__':

    dataset_path = Path('user_dataset_full.csv')

    X_train, X_test, y_train, y_test = load_data(dataset_path)

    model = fine_tune(0, 100, 20, X_train)

    evaluate_simple_vote(model, X_test, y_test)

    processed_docs_file = 'model_tuned'
    dump(processed_docs_file, open(processed_docs_file, 'wb'))


    # file_train_descriptions = Path('./train_clear_descr.csv')
    # file_train_features = Path('./google_train_features1.csv')
    #
    # file_test_features = Path('./google_test_features.csv')
    #
    # train_descriptions = load_clean_descriptions(file_train_descriptions)
    # print('Descriptions: train=%d' % len(train_descriptions))
    #
    #
    # # prepare tokenizer
    # tokenizer = create_tokenizer(train_descriptions)
    #
    # vocab_size = len(tokenizer.word_index) + 1
    # print('Vocabulary Size: %d' % vocab_size)
    #
    # # determine the maximum sequence length
    # max_length = max_length(train_descriptions)
    # print('Max Length: %d' % max_length)
    #
    # # features_generator_train = dataset_loading(file_train_features)
    # # features_generator_test = dataset_loading(file_test_features)
    #
    # features_generator_train = None
    # features_generator_test = None
    #
    # batch_size = 512
    # trainGenerator = TrainGenerator(batch_size=batch_size,
    #                                 photo_features_train=features_generator_train,
    #                                 photo_features_test=features_generator_test,
    #                                 tokenizer=tokenizer,
    #                                 captions=train_descriptions,
    #                                 max_length=max_length,
    #                                 vocab_size=vocab_size,
    #                                 max_cache_size=batch_size*500)
    #
    # # define the model
    # model = trainGenerator.define_model()
    #
    # epochs = 10
    #
    # train_features_len = get_steps_by_features(file_train_features)
    # test_features_len = get_steps_by_features(file_test_features)
    #
    # steps_train = math.ceil(train_features_len / batch_size)
    # steps_test = math.ceil(test_features_len / batch_size)
    #
    # filepath = 'new_google_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    #
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
    #
    # # model.fit_generator(trainGenerator.generate_data(),
    # #                     steps_per_epoch=steps_train,
    # #                     epochs=epochs,
    # #                     callbacks=[checkpoint],
    # #                     validation_data=trainGenerator.generate_data(train=False),
    # #                     validation_steps=steps_test,
    # #                     )
    #
    # model.fit_generator(trainGenerator.generate_by_chunk(file_train_features),
    #                     steps_per_epoch=steps_train,
    #                     epochs=epochs,
    #                     callbacks=[checkpoint],
    #                     validation_data=trainGenerator.generate_by_chunk(file_test_features),
    #                     validation_steps=steps_test,
    #                     )
    #
    # for i in range(epochs):
    #     # create the data generator
    #     features_generator = dataset_loading(file_train_features)
    #     generator = data_generator(train_descriptions, features_generator, tokenizer, max_length)
    #
    #     # fit for one epoch
    #     model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    #     # save model
    #     model.save('google_model_' + str(i) + '.h5')




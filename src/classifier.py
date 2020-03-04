import numpy as np
# from sklearn import knnClassifier
# from data_processing.data_loader import DataLoader
from data_loader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import preprocessing


FEATURE_V_PATH = '/home/student/akharchevnikova/face_features_mobilenet_vgg_all_ijbc.csv'


def aggregate_features_aver():
    # name_ids = DataLoader.get_subject_ids()
    template_to_ids = DataLoader.map_template_to_id()
    frames_to_templates = DataLoader.map_frame_to_template_id()

    y_frames, x = DataLoader.load_feature_vector(FEATURE_V_PATH)

    feature_map = {}

    for frame_label, feature in zip(y_frames, x):
        tempalte_id = frames_to_templates[frame_label]
        if tempalte_id not in feature_map:
            feature_map[tempalte_id] = []

        feature_map[tempalte_id].append(feature)

    labels = []
    x_agg = []

    j = 0

    for key, value in feature_map.items():
        # if int(template_to_ids[key]) in labels:
        #     j += 1
        #     labels.append(int(template_to_ids[key]))
        #     x_agg.append(np.mean(value, axis=0))
        #
        # else:
        n = 3
        new_values = np.split(np.array(value), [int(len(value)/4), int(len(value)/2), len(value)])
        for features in new_values:
            if features.size != 0:
                labels.append(int(template_to_ids[key]))
                x_agg.append(np.mean(features, axis=0))

    return x_agg, labels


def train_knn_classifier():
    print("START!!!")

    X, y = aggregate_features_aver()
    X_norm = preprocessing.normalize(X, norm='l2')

    # X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

    classifier = KNeighborsClassifier(1)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    print('train classes:', len(np.unique(y_train)))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)

    print("labels: ", y_train)

    acc = 100.0 * (np.sum(y_test == y_test_pred) / len(y_test))

    print('acc=', acc)

    classifier_results_file = '/home/student/akharchevnikova/results/KNeighborsClassifier_1_mobilenet_vgg2_results.npz'
    np.savez(classifier_results_file, x=X_test, y=y_test, z=y_test_pred)

    print('SUCCESS!!!')


if __name__ == '__main__':
    train_knn_classifier()



import numpy as np
# from sklearn import knnClassifier
# from data_processing.data_loader import DataLoader
import pickle
from data_processing.data_loader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans

from sklearn import preprocessing
from scipy.spatial.distance import cosine

from sklearn.preprocessing import MinMaxScaler, normalize


cluster_alg = {'kmeans': KMeans, 'minbatch': MiniBatchKMeans}


FEATURE_V_PATH_MXNET = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_arcface_resnet_100_ijbc_frames_cropped_all_prep.csv'
FEATURE_V_PATH_MOBILENET = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_mobilenet_vgg_all_ijbc.csv'
FEATURE_V_PATH_VGG_RESNET = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_vgg2_resnet_ijbc_frames.csv'

features_map = {
    'mobile_vgg': FEATURE_V_PATH_MOBILENET,
    'mxnet': FEATURE_V_PATH_MXNET,
    'resnet_vgg': FEATURE_V_PATH_VGG_RESNET

}


def aggregate_features_aver():
    # name_ids = DataLoader.get_subject_ids()
    IJBC_PROBE_MIXED = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(IJBC_PROBE_MIXED)
    frames_to_templates = DataLoader.map_frame_to_template_id(IJBC_PROBE_MIXED)

    y_frames, x = DataLoader.load_feature_vector(FEATURE_V_PATH_MXNET)
    # y_frames, x = DataLoader.load_feature_vector(FEATURE_V_PATH_MOBILENET)

    feature_map = {}

    for frame_label, feature in zip(y_frames, x):
        if 'img' in frame_label:
            continue

        tempalte_id = frames_to_templates[frame_label]
        if tempalte_id not in feature_map:
            feature_map[tempalte_id] = []

        feature_map[tempalte_id].append(feature)

    labels = []
    x_agg = []

    for key, value in feature_map.items():
        labels.append(int(template_to_ids[key]))
        x_agg.append(np.mean(value, axis=0))

    return x_agg, labels


def load_features(features_path):
    # metadata_path = {0: 'C:\\akharche\\MasterThesis\\ijbc_1N_gallery_G1.csv',
    #                  1: 'C:\\akharche\\MasterThesis\\ijbc_1N_gallery_G2.csv'}

    metadata_path = 'C:\\akharche\\MasterThesis\\ijbc_1N_all_imgs.csv'

    # template_to_ids = DataLoader.map_template_to_id(metadata_path.get(0))
    # template_to_ids.update(DataLoader.map_template_to_id(metadata_path.get(1)))

    template_to_ids = DataLoader.map_template_to_id(metadata_path)

    # imgs_to_templates = DataLoader.map_imgs_to_template_id(metadata_path.get(0))

    imgs_to_templates = DataLoader.map_imgs_to_template_id(metadata_path)


    # print("IMGS TO TEMPL 1: ", len(imgs_to_templates))
    # imgs_to_templates.update(DataLoader.map_imgs_to_template_id(metadata_path.get(1)))

    print("IMGS TO TEMPL 2: ", len(imgs_to_templates))

    y_frames, x = DataLoader.load_feature_vector(features_path)

    labels = []
    x_agg = []

    for img_label, feature in zip(y_frames, x):
        try:
            template_id = imgs_to_templates[img_label]

            labels.append(int(template_to_ids[template_id]))
            x_agg.append(np.array(feature))
        except:
            continue

    return x_agg, labels


# TEST on frames by average of all frames in a video
def test_classifier(model_path):
    # Load model
    classifier = pickle.load(open(model_path, 'rb'))

    X, y = aggregate_features_aver()

    y_test_pred = classifier.predict(X)
    print("labels: ", y)

    acc = 100.0 * (np.sum(y == y_test_pred) / len(y))

    print('ACCURACY on video: ', acc)

    # classifier_results_file = 'C:\\akharche\\MasterThesis\\Scores\\KNeighborsClassifier_1_mobilenet_vgg_frames.npz'
    # np.savez(classifier_results_file, x=X, y=y, z=y_test_pred)

    print('SUCCESS!!!')


#    By img gallery in IJB-C
def train_knn_classifier():
    print("START!!!")

    # features_path = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_arcface_resnet_100_ijbc_images_train.csv'
    # features_path = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_mobilenet_vgg_ijbc_images_all.csv'
    # features_path = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_vgg2_resnet_ijbc_images_train.csv'

    features_path = 'C:\\akharche\\MasterThesis\\ModelFeatures\\face_features_arcface_ijbc_imgs_cropped_all_prep.csv'

    X, y = load_features(features_path)
    # X_norm = preprocessing.normalize(X, norm='l2')
    X_norm = np.array(X)

    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.01, random_state=42, shuffle=True)

    classifier = KNeighborsClassifier(1, p=2)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    print('train classes:', len(np.unique(y_train)))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)

    print("labels: ", y_train)

    acc = 100.0 * (np.sum(y_test == y_test_pred) / len(y_test))

    print('acc=', acc)

    pickle.dump(classifier, open(
        'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_arcface', 'wb'))

    print('SUCCESS!!!')


def predict_knn(X, y, model_path):
    X_norm = preprocessing.normalize(X, norm='l2')

    # Load model
    classifier = pickle.load(open(model_path, 'rb'))

    y_test_pred = classifier.predict(X_norm)
    print("labels: ", y)

    acc = 100.0 * (np.sum(y == y_test_pred) / len(y))

    return acc


def classify_random(knn_model_path, model='mobile_vgg', k=2, all_k=False):
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)
    video_frames = DataLoader.get_videos(ijb_probe_mixed)

    features_file = features_map[model]
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD FRAMES")

    for video_id, frames in video_frames.items():
        n_frames = len(frames)
        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        random_frames_indx = np.random.randint(n_frames, size=n)
        frame_features = []

        for frame_i in random_frames_indx:
            frame = frames[frame_i]
            frame_feature_vec = frames_features_map[frame]
            frame_features.append(frame_feature_vec)

        person_id = template_to_ids[video_id]

        if len(frame_features) == 1:
            # print("ONE FEATURE!!!!")
            # print(video_id)
            mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            # print("NAAAN")
            # print(video_id)
            continue

        X.append(mean_arr)
        labels.append(int(person_id))
    #
    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY Random on video: ', acc)
    return acc


def classify_clustering(knn_model_path, model='mobile_vgg', k=4, alg='kmeans', all_k=False):
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    cl_alg = cluster_alg.get(alg, 'kmeans')

    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)
    video_frames = DataLoader.get_videos(ijb_probe_mixed)
    features_file = features_map[model]
    y_frames, x = DataLoader.load_feature_vector(features_file)
    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD FRAMES")

    for video_id, frames in video_frames.items():
        n_frames = len(frames)
        if all_k:
            n = k
        else:
            n = int(n_frames / k)
            if n == 0:
                n = 1

        frame_features = np.array([frames_features_map[frame] for frame in frames])

        if len(frame_features) > 1:

            cls = cl_alg(n_clusters=n, random_state=0).fit(frame_features)
            cluster_centers = cls.cluster_centers_
        else:
            cluster_centers = frame_features

        person_id = template_to_ids[video_id]

        if len(cluster_centers) == 1:
            # print("ONE FEATURE!!!!")
            # print(video_id)

            mean_arr = cluster_centers[0]
        else:
            mean_arr = np.mean(cluster_centers, axis=0)

        if np.isscalar(mean_arr):
            # print("NAAAN")
            # print(video_id)
            continue

        X.append(mean_arr)
        labels.append(int(person_id))
    #
    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print(f'ACCURACY Clustering {alg} on video: ', acc)
    return acc


def classify_faceqnet(scores_file, knn_model_path, model='mobile_vgg', k=2, all_k=False):
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)

    features_file = features_map[model]
    video_scores = DataLoader.get_video_scores(scores_file)
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD BEST FRAMES")

    for video, frames_scores in video_scores.items():
        n_frames = len(frames_scores)
        sorted_scores = sorted(frames_scores, key=lambda tup: tup.score, reverse=True)
        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        top_k_frames = sorted_scores[:n]
        frame_features = []

        i = 0
        for frame_data in top_k_frames:
            if frame_data.frame_id in frames_features_map:
                frame_features.append(frames_features_map[frame_data.frame_id])
            else:
                # print("NO such frame!")
                # print(frame_data.frame_id)
                i += 1
        # print(f'{i} frames from {n} are skipped')

        video_id = template_to_ids[video]

        if len(frame_features) == 1:
            mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            continue

        X.append(mean_arr)
        labels.append(int(video_id))

    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY FaceQNet on video: ', acc)
    return acc


def classify_brisque(scores_file, knn_model_path, model='mobile_vgg', k=2, all_k=False):
    # Less is better
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)

    features_file = features_map[model]
    video_scores = DataLoader.get_video_scores(scores_file)
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD BEST FRAMES")

    for video, frames_scores in video_scores.items():
        n_frames = len(frames_scores)
        sorted_scores = sorted(frames_scores, key=lambda tup: tup.score, reverse=False)

        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        top_k_frames = sorted_scores[:n]
        frame_features = []

        i = 0
        for frame_data in top_k_frames:
            if frame_data.frame_id in frames_features_map:
                frame_features.append(frames_features_map[frame_data.frame_id])
            else:
                # print("NO such frame!")
                # print(frame_data.frame_id)
                i += 1
        # print(f'{i} frames from {n} are skipped')

        video_id = template_to_ids[video]

        if len(frame_features) == 1:
            mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            continue

        X.append(mean_arr)
        labels.append(int(video_id))

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY Brisque on video: ', acc)
    return acc


def classify_brightness(scores_file, knn_model_path, model='mobile_vgg', k=2, all_k=False):
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)

    features_file = features_map[model]
    video_scores = DataLoader.get_video_scores(scores_file)
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD BEST FRAMES")

    for video, frames_scores in video_scores.items():
        n_frames = len(frames_scores)

        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        sorted_scores = sorted(frames_scores, key=lambda tup: tup.score, reverse=True)
        top_k_frames = sorted_scores[:n]
        frame_features = []

        i = 0
        for frame_data in top_k_frames:
            if frame_data.frame_id in frames_features_map:
                frame_features.append(frames_features_map[frame_data.frame_id])
            else:
                i += 1
        video_id = template_to_ids[video]

        if len(frame_features) == 1:
           mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            continue

        X.append(mean_arr)
        labels.append(int(video_id))

    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY Brightness on video: ', acc)
    return acc


def classify_contrast(scores_file, knn_model_path, model='mobile_vgg', k=2, all_k=False):
    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)

    features_file = features_map[model]
    video_scores = DataLoader.get_video_scores(scores_file)
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD BEST FRAMES")

    for video, frames_scores in video_scores.items():
        n_frames = len(frames_scores)
        sorted_scores = sorted(frames_scores, key=lambda tup: tup.score, reverse=True)

        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        top_k_frames = sorted_scores[:n]
        frame_features = []

        i = 0
        for frame_data in top_k_frames:
            if frame_data.frame_id in frames_features_map:
                frame_features.append(frames_features_map[frame_data.frame_id])
            else:
                i += 1

        video_id = template_to_ids[video]

        if len(frame_features) == 1:
            mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            # print("NAAAN")
            # print(video)
            continue

        X.append(mean_arr)
        labels.append(int(video_id))

    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY Contrast on video: ', acc)
    return acc


def classify_fiiqa(scores_file, knn_model_path, model='mobile_vgg', k=2, all_k=False):
    # Less better

    ijb_probe_mixed = 'C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv'
    template_to_ids = DataLoader.map_template_to_id(ijb_probe_mixed)

    features_file = features_map[model]
    video_scores = DataLoader.get_video_scores_fiiqa(scores_file)
    y_frames, x = DataLoader.load_feature_vector(features_file)

    frames_features_map = dict(zip(y_frames, x))

    X = []
    labels = []

    print("START LOAD BEST FRAMES")

    for video, frames_scores in video_scores.items():
        n_frames = len(frames_scores)
        sorted_scores = sorted(frames_scores, key=lambda tup: tup.score, reverse=True)

        if all_k:
            n = k
        else:
            n = int(n_frames / k)

        top_k_frames = sorted_scores[:n]
        frame_features = []

        i = 0
        for frame_data in top_k_frames:
            if frame_data.frame_id in frames_features_map:
                frame_features.append(frames_features_map[frame_data.frame_id])
            else:
                i += 1

        video_id = template_to_ids[video]

        if len(frame_features) == 1:
            mean_arr = frame_features[0]
        else:
            mean_arr = np.mean(frame_features, axis=0)

        if np.isscalar(mean_arr):
            continue

        X.append(mean_arr)
        labels.append(int(video_id))

    print("FINISH")

    X = np.array(X)

    acc = predict_knn(X, labels, knn_model_path)
    print('ACCURACY FIIQA on video: ', acc)
    return acc


def dump_results():
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet_imgs'
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_modelnet_vgg_imgs_all'

    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet'
    # model = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_arcface'

    model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_arcface'

    # model_cnn = 'resnet_vgg'
    # model_cnn ='mobile_vgg'

    model_cnn = 'mxnet'

    num_frames_experiments = [(2, False), (4, False), (8, False), (2, True), (1, True)]

    # num_frames_experiments = [(2, True), (1, True)]
    dump_file = 'experiments_k_arcface.csv'
    titles = ['k', 'rand', 'cluster_kmeans', 'cluster_minib', 'faceqnet', 'faceqnet_mobile', 'brisque', 'bright',
              'contr', 'fiiqa', 'faceqnet_small']

    with open(dump_file, mode='w', encoding='utf-8') as wr:
        title = ','.join(titles)
        wr.write(title)
        wr.write('\n')

        for num_k, allk in num_frames_experiments:
            print("EXPERIMENT: ", num_k)
            acc_all = []
            acc = classify_random(model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            acc = classify_clustering(model_path, model=model_cnn, k=num_k, all_k=allk, alg='kmeans')
            acc_all.append(acc)

            acc = classify_clustering(model_path, model=model_cnn, k=num_k, all_k=allk, alg='minbatch')
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_scores_frames.csv'
            acc = classify_faceqnet(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_1layer_scores_mobilenet_frames.csv'
            acc = classify_faceqnet(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_brisque_scores_frames.csv'
            acc = classify_brisque(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_brightness_scores_frames.csv'
            acc = classify_brightness(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_contrast_scores_frames.csv'
            acc = classify_contrast(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_fiiqa_vgg2_69_scores_frames.csv'
            acc = classify_fiiqa(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_scores_small_2.csv'
            acc = classify_faceqnet(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            acc_line = ','.join([str(a) for a in acc_all])
            if allk:
                num = f'{num_k},'
            else:
                num = f'n/{num_k},'

            wr.write(num + acc_line)
            wr.write('\n')


def dump_results_separate():
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet_imgs'
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'

    model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_modelnet_vgg_imgs_all'
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet'

    # model_cnn = 'resnet_vgg'
    # model_cnn ='mobile_vgg'
    model_cnn = 'mxnet'

    num_frames_experiments = [(2, False), (4, False), (8, False), (2, True), (1, True)]
    # num_frames_experiments = [(2, False), (4, False), (8, False)]
    dump_file = 'experiments_k_arcface.csv'
    titles = ['k', 'faceqnet']

    with open(dump_file, mode='w', encoding='utf-8') as wr:
        title = ','.join(titles)
        wr.write(title)
        wr.write('\n')

        for num_k, allk in num_frames_experiments:
            print("EXPERIMENT: ", num_k)
            acc_all = []

            # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_scores_small_2.csv'
            # acc = classify_faceqnet(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            # acc_all.append(acc)

            scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_mobile_vgg.csv'
            acc = classify_faceqnet(scores_file, model_path, model=model_cnn, k=num_k, all_k=allk)
            acc_all.append(acc)

            acc_line = ','.join([str(a) for a in acc_all])
            if allk:
                num = f'{num_k},'
            else:
                num = f'n/{num_k},'

            wr.write(num + acc_line)
            wr.write('\n')


if __name__ == '__main__':
    # train_knn_classifier()
    # # model = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_modelnet_vgg_imgs_all'
    #
    # model = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_arcface'
    # #
    # test_classifier(model)

    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet_imgs'

    dump_results()
    # dump_results_separate()

    ##################
    #Face QNet
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_vggface_resnet_imgs'
    #
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_scores_frames.csv'
    # # test_classifier(model_path)
    # classify_faceqnet(scores_file, model_path, model='resnet_vgg', k=4)

    ##################
    # Face QNet Light
    ##################
    #
    scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_1layer_scores_mobilenet_frames.csv'
    # # test_classifier(model_path)
    # classify_faceqnet(scores_file, model_path, model='resnet_vgg', k=2)
    # classify_faceqnet(scores_file, model_path, k=4)

    ##################
    # Brisque
    # ##################
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_brisque_scores_frames.csv'
    # test_classifier(model_path)

    # classify_brisque(scores_file, model_path, model='resnet_vgg', k=4)

    ##################
    # Brightness
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_brightness_scores_frames.csv'
    #
    # classify_brightness(scores_file, model_path, model='resnet_vgg', k=4)

    ##################
    # Contrast
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_contrast_scores_frames.csv'
    #
    # classify_contrast(scores_file, model_path, model='resnet_vgg', k=4)

    ##################
    # FIIQA
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_fiiqa_vgg2_69_scores_frames.csv'

    # classify_fiiqa(scores_file, model_path, model='resnet_vgg', k=4)


    ##################
    # Random
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    # scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_fiiqa_vgg2_69_scores_frames.csv'
    #
    # classify_random(model_path, model='resnet_vgg', k=8)

    ##################
    # Clustering k-means
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    #
    # classify_clustering(model_path, model='resnet_vgg', k=4, alg='kmeans')

    ##################
    # Clustering MiniBatch
    ##################
    # model_path = 'C:\\akharche\\MasterThesis\\KeyFrameExtraction\\models\\KNeighborsClassifier_1_mobilenet_vgg_imgs'
    #
    # classify_clustering(model_path, model='resnet_vgg', k=4, alg='minbatch')


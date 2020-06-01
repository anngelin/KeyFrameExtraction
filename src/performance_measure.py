import time
import cv2
import numpy as np
import pickle
from face_feature_extractor import TensorFlowInference

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn import preprocessing
from scipy.spatial.distance import cosine

from sklearn.preprocessing import MinMaxScaler, normalize

from image_quality_estimation import FaceQNetQualityEstimator, QualityEstimator
from estimate_brisque import calculate_video, measure_brisque

from keras.models import load_model
from keras.applications.mobilenet import preprocess_input

from keras import backend as K


VIDEO = 'video_performance_90_frames.csv'
DATASET_DIR = '/home/akharchevnikova/Face_Image_Quality_Assessment/video/'

baseline_model_vgg = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/vgg2_resnet.pb'
baseline_model_mobilenet = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/vgg2_mobilenet.pb'
knn_classifier = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/KNeighborsClassifier_1_vggface_resnet_imgs'

knn_classifier_mobilenet = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/KNeighborsClassifier_1_vgg_mobilenet_imgs'

cluster_alg = {'kmeans': KMeans, 'minbatch': MiniBatchKMeans}

models_map = {
    'mobile_vgg': baseline_model_mobilenet,
    'resnet_vgg': baseline_model_vgg

}

# To score fiiqa
def aggregate_expected_value(scores):
    res_score = 0
    for i, probab in enumerate(scores):
        res_score += (i+1) * probab

    res_score = res_score / len(scores)

    return res_score


def get_video_frames(filepath=VIDEO):
    frames = {}
    with open(filepath, mode='r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(',')
            groundtruth = line[4:]
            frames[DATASET_DIR+line[2]] = groundtruth

    return frames


def get_baseline_perf(model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                      output_tensor='reshape_1/Reshape:0',
                                      learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                      convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()


    frame_features = []

    start = time.time()
    #
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("BASELINE PERFORMANCE: ", end)
    print("BASELINE INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_random_vggface(k=2, model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()

    frame_features = []

    start = time.time()

    n_frames = len(video_frames)
    n = int(n_frames / k)
    random_frames_indx = np.random.randint(n_frames, size=n)
    video_frames_data = list(video_frames.keys())
    random_keys = [video_frames_data[i] for i in random_frames_indx]
    random_frames = {key: video_frames[key] for key in random_keys}

    for frame, gr_truth in random_frames.items():
        # print frame
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("RANDOM PERFORMANCE: ", end)
    print("RANDOM INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_clustering(k=4, alg='kmeans', model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()
    cl_alg = cluster_alg.get(alg, 'kmeans')

    frame_features = []

    start = time.time()

    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - start

    frame_features = np.array(frame_features)

    if len(frame_features) > 1:

        cls = cl_alg(n_clusters=n, random_state=0).fit(frame_features)
        cluster_centers = cls.cluster_centers_
    else:
        cluster_centers = frame_features

    if len(cluster_centers) == 1:
        print("ONE FEATURE!!!!")
        mean_arr = cluster_centers[0]
    else:
        mean_arr = np.mean(cluster_centers, axis=0)

    tmp = []
    tmp.append(mean_arr)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("CLUSTERING PERFORMANCE: ", end)
    print("CLUSTERING INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_brightness(k=2, model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()
    frame_features = []
    frames_labels = []
    brt_scores = []
    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    start = time.time()
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        brt_estimator = QualityEstimator(draw)

        brt_score = brt_estimator.estimate_brightness()
        brt_scores.append(brt_score)
        frames_labels.append(frame)

    frames_scores = zip(frames_labels, brt_scores)
    sorted_frames_scores = sorted(frames_scores, key=lambda x: x[1], reverse=False)[:n]

    t_inference_start = time.time()
    for frame, score in sorted_frames_scores:
        gr_truth = video_frames[frame]
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - t_inference_start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("BRIGHTNESS PERFORMANCE: ", end)
    print("BRIGHTNESS INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_contrast(k=2, model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()

    frame_features = []

    frames_labels = []
    brt_scores = []
    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    start = time.time()
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        brt_estimator = QualityEstimator(draw)

        brt_score = brt_estimator.get_image_contrast()
        brt_scores.append(brt_score)
        frames_labels.append(frame)

    frames_scores = zip(frames_labels, brt_scores)
    sorted_frames_scores = sorted(frames_scores, key=lambda x: x[1], reverse=False)[:n]

    t_inference_start = time.time()
    for frame, score in sorted_frames_scores:
        gr_truth = video_frames[frame]
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - t_inference_start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("CONTRAST PERFORMANCE: ", end)
    print("CONTRAST INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_brisque(k=2, model='mobilenet'):
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()
    frame_features = []

    frames_labels = []
    br_scores = []
    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    start = time.time()
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

        brt_score = measure_brisque(draw, load_image=False)
        br_scores.append(brt_score)
        frames_labels.append(frame)

    frames_scores = zip(frames_labels, br_scores)
    sorted_frames_scores = sorted(frames_scores, key=lambda x: x[1], reverse=True)[:n]

    t_inference_start = time.time()
    for frame, score in sorted_frames_scores:
        gr_truth = video_frames[frame]
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - t_inference_start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("BRISQUE PERFORMANCE: ", end)
    print("BRISQUE INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_faceqnet(k=2, model='mobilenet'):
    model_path_faceqnet = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/mobilenet_faceqnet_1layer-16-0.0010.hdf5'
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()
    model_faceqnet = load_model(model_path_faceqnet)

    frame_features = []

    frames_labels = []
    model_scores = []
    faces = []

    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    start = time.time()
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)
        x, y, w, h = gr_truth
        y = int(y)
        x = int(x)
        w = int(w)
        h = int(h)
        face = draw[y:y + h, x:x + w]

        faces.append(face)

        face = cv2.resize(face, (224, 224))

        image = face.reshape((1, face.shape[0], face.shape[1], face.shape[2]))
        # image = preprocess_input(image)

        score = model_faceqnet.predict(image, batch_size=1, verbose=1)[0]
        model_scores.append(score)

        frames_labels.append(frame)

    frames_scores = zip(frames_labels, model_scores, faces)

    sorted_frames_scores = sorted(frames_scores, key=lambda x: x[1], reverse=True)[:n]

    print(len(sorted_frames_scores))

    print("STOOOOP: ", time.time() - start)

    K.clear_session()

    t_inference_start = time.time()
    for frame, score, face in sorted_frames_scores:
        # gr_truth = video_frames[frame]
        # draw = cv2.imread(frame)
        #
        # y = int(gr_truth[1])
        # x = int(gr_truth[0])
        # w = int(gr_truth[2])
        # h = int(gr_truth[3])
        #
        # face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - t_inference_start

    tfInference.close_session()

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("FACEQNET PERFORMANCE: ", end)
    print("FACEQNET INFERENCE: ", t_inference)
    print(face_pred)


def get_perf_fiiqa(k=2, model='mobilenet'):
    model_path_fiiqa = '/home/akharchevnikova/Face_Image_Quality_Assessment/models/mobilenet_vgg_fiiqa_224_finetuned-09-0.69.hdf5'
    if model == 'mobilenet':
        model_path = baseline_model_mobilenet
        classifier = pickle.load(open(knn_classifier_mobilenet, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                          output_tensor='reshape_1/Reshape:0',
                                          learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    else:
        model_path = baseline_model_vgg
        classifier = pickle.load(open(knn_classifier, 'rb'))
        tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                          output_tensor='pool5_7x7_s1:0',
                                          convert2BGR=True, imageNetUtilsMean=False)
    video_frames = get_video_frames()
    model = load_model(model_path_fiiqa)

    frame_features = []
    frames_labels = []
    model_scores = []

    n_frames = len(video_frames)
    n = int(n_frames / k)
    if n == 0:
        n = 1

    start = time.time()
    for frame, gr_truth in video_frames.items():
        # print frame
        draw = cv2.imread(frame)
        face = cv2.resize(draw, (224, 224))

        image = face.reshape((1, face.shape[0], face.shape[1], face.shape[2]))
        image = preprocess_input(image)
        score = model.predict(image, verbose=1)[0]
        res = aggregate_expected_value(score)
        model_scores.append(res)

        frames_labels.append(frame)

    K.clear_session()

    frames_scores = zip(frames_labels, model_scores)

    sorted_frames_scores = sorted(frames_scores, key=lambda x: x[1], reverse=True)[:n]

    print(len(sorted_frames_scores))

    t_inference_start = time.time()
    for frame, score in sorted_frames_scores:
        gr_truth = video_frames[frame]
        draw = cv2.imread(frame)

        y = int(gr_truth[1])
        x = int(gr_truth[0])
        w = int(gr_truth[2])
        h = int(gr_truth[3])

        face = draw[y:y + h, x:x + w]
        features = tfInference.extract_features(face, crop_center=False, is_file=False)
        frame_features.append(features)

    t_inference = time.time() - t_inference_start

    average_vec = np.mean(frame_features, axis=0)
    print(average_vec)
    tmp = []
    tmp.append(average_vec)

    X = np.array(tmp)

    face_pred = classifier.predict(X)

    end = time.time() - start
    print("FACEQNET PERFORMANCE: ", end)
    print("FACEQNET INFERENCE: ", t_inference)
    print(face_pred)


if __name__ == '__main__':
    # get_baseline_perf()
    get_perf_random_vggface(k=2)
    # get_perf_clustering(k=4, alg='minbatch')
    # get_perf_brightness(k=4)
    # get_perf_contrast()
    # get_perf_brisque()
    # get_perf_faceqnet()
    # get_perf_fiiqa()

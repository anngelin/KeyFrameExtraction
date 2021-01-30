import numpy as np
import pickle
import cv2
import keras
import pathlib

from keras.models import Model, load_model
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array



from data_processing.data_loader import DataLoader
# from data_loader import DataLoader
from image_quality_assessment.image_quality_estimation import FaceQNetQualityEstimator
# from image_quality_estimation import FaceQNetQualityEstimator, QualityEstimator

from image_quality_assessment.estimate_brisque import calculate_video, measure_brisque

# from estimate_brisque import calculate_video, measure_brisque


# To score fiiqa
def aggregate_expected_value(scores):
    res_score = 0
    for i, probab in enumerate(scores):
        res_score += (i+1) * probab

    res_score = res_score / len(scores)

    return res_score


def classify_1nn(X, y):
    # Load model
    model_path = '/home/student/akharchevnikova/1nn_ijbc_images'
    classifier = pickle.load(open(model_path, 'rb'))

    y_test_pred = classifier.predict(X)
    acc = 100.0 * (np.sum(y == y_test_pred) / len(y))
    #TODO: save results to file
    return acc


def face_qnet_pipeline_knn(model_path, dataset_path):
    # Load with ground_truth
    data_loader = DataLoader()
    video_frames = DataLoader.load_video_frames()

    template_to_ids = DataLoader.map_template_to_id()
    frames_to_templates = DataLoader.map_frame_to_template_id()
    # feature_vectors = DataLoader.load_feature_vector_by_frames(feature_vector_path)

    best_frames = {}

    for video_id, frames_data in video_frames.items():
        cropped_frames = []
        frames_labels = []
        for frame_name, ground_truth in frames_data:
            frame_path = dataset_path + frame_name
            frames_labels.append(frame_name)

            draw = cv2.imread(frame_path)
            x, y, w, h = ground_truth
            y = int(y)
            x = int(x)
            w = int(w)
            h = int(h)
            face = draw[y:y + h, x:x + w]
            cropped_frames.append(face)

        face_qnet_estimator = FaceQNetQualityEstimator(cropped_frames, frames_labels, model_path)
        top_k_frames = face_qnet_estimator.get_best_frames(k=5)
        # if video_id not in best_frames:
        best_frames[video_id] = top_k_frames

    X, y = [], []

    for video_id, frames in best_frames:
        label = template_to_ids[video_id]
        y.append(label)

        feature_vectors = [data_loader.get_feature_vector(frame_id) for frame_id, score in frames]
        X.append(np.mean(feature_vectors))

    acc = classify_1nn(X, y)
    print('ACCURACY: ', acc)


def face_qnet_save_scores(model_path, dataset_path):
    scores_file = 'C:\\akharche\\MasterThesis\\Scores\\ytf_frames_qnet_scores.csv'
    metadata_path = 'C:\\akharche\\MasterThesis\\ytf_frames_meta_tmp2.csv'

    # Load with ground_truth
    face_qnet_estimator = FaceQNetQualityEstimator()

    videos = DataLoader.map_video_ytf(metadata_path)

    with open(scores_file, mode='a') as fr:

        for video_id, frames_data in videos.items():
            cropped_frames = []
            frames_labels = []
            for frame_name in frames_data:
                frame_path = frame_name
                frames_labels.append(frame_name)

                draw = cv2.imread(frame_path)
                cropped_frames.append(draw)

            frames_scores = face_qnet_estimator.estimate_quality_qnet(cropped_frames, frames_labels)

            for label, score in frames_scores:
                line = [video_id, label, str(score[0])]
                fr.write((',').join(line))
                fr.write('\n')


def face_qnet_light_save_scores(model_path, dataset_path):
    scores_file = 'results/ijb_c_frames_qnet_scores_mobilenet.csv'

    # Load with ground_truth
    video_frames = DataLoader.load_video_frames()
    model = load_model(model_path)

    with open(scores_file, mode='w') as fr:

        for video_id, frames_data in video_frames.items():

            for frame_name, ground_truth in frames_data:
                frame_path = frame_name

                draw = cv2.imread(frame_path)
                x, y, w, h = ground_truth
                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)
                face = draw[y:y + h, x:x + w]

                face = cv2.resize(face, (224, 224))

                image = face.reshape((1, face.shape[0], face.shape[1], face.shape[2]))
                image = preprocess_input(image)
                scores = model.predict(image, batch_size=1, verbose=1)[0]
                line = [video_id, frame_name, str(scores[0])]
                fr.write((',').join(line))
                fr.write('\n')


def brisque_pipeline_knn(dataset_path):
    # Load with ground_truth
    video_frames = DataLoader.load_video_frames()

    template_to_ids = DataLoader.map_template_to_id()
    frames_to_templates = DataLoader.map_frame_to_template_id()
    # feature_vectors = DataLoader.load_feature_vector_by_frames(feature_vector_path)

    best_frames = {}

    for video_id, frames_data in video_frames.items():
        cropped_frames = []
        frames_labels = []
        for frame_name, ground_truth in frames_data:
            frame_path = dataset_path + frame_name
            frames_labels.append(frame_name)

            draw = cv2.imread(frame_path)
            x, y, w, h = ground_truth
            y = int(y)
            x = int(x)
            w = int(w)
            h = int(h)
            face = draw[y:y + h, x:x + w]
            cropped_frames.append(face)

        frames_map = dict(zip(frames_labels, cropped_frames))

        top_k_frames = calculate_video(frames_map)
        # if video_id not in best_frames:
        best_frames[video_id] = top_k_frames

    print(best_frames)


def brisque_save_scores(dataset_path):
    scores_file = 'results/ijb_c_frames_brisque_scores.csv'

    # Load with ground_truth
    video_frames = DataLoader.load_video_frames()

    template_to_ids = DataLoader.map_template_to_id()
    frames_to_templates = DataLoader.map_frame_to_template_id()
    # feature_vectors = DataLoader.load_feature_vector_by_frames(feature_vector_path)

    best_frames = {}

    with open(scores_file, mode='w') as fr:
        for video_id, frames_data in video_frames.items():
            frames_labels = []

            for frame_name, ground_truth in frames_data:
                # frame_path = dataset_path + frame_name
                frame_path = frame_name
                frames_labels.append(frame_name)

                draw = cv2.imread(frame_path)

                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

                x, y, w, h = ground_truth
                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)
                face = draw[y:y + h, x:x + w]

                face_score = measure_brisque(face, load_image=False)

                line = [video_id, frame_name, str(face_score)]

                fr.write((',').join(line))
                fr.write('\n')


def fiiqa_save_scores():
    model_path = '/home/student/akharchevnikova/models/mobilenet_vgg_fiiqa_224_finetuned-09-0.69.hdf5'
    model = load_model(model_path)
    scores_file = 'results/ijb_c_frames_fiiqa_vgg2_69_scores.csv'
    target_size = (224, 224)

    video_frames = DataLoader.load_video_frames()

    i = 0

    with open(scores_file, mode='w') as fr:
        for video_id, frames_data in video_frames.items():
            frames_labels = []

            for frame_name, ground_truth in frames_data:
                # frame_path = dataset_path + frame_name
                frame_path = frame_name
                frames_labels.append(frame_name)

                # draw = cv2.imread(frame_path)
                #
                # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

                i += 1

                try:
                    image = load_img(str(frame_path), target_size=target_size)
                except Exception as e:
                    print(e)
                    continue

                print(i)
                image = img_to_array(image)

                # x, y, w, h = ground_truth
                # y = int(y)
                # x = int(x)
                # w = int(w)
                # h = int(h)
                # image = image[y:y + h, x:x + w]

                #
                # face = face.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # face = preprocess_input(face)

                image = image.reshape((1, 224, 224, image.shape[2]))
                image = preprocess_input(image)

                preds = model.predict(image)[0]
                res = aggregate_expected_value(preds)
                preds_line = [str(pred) for pred in preds]

                line = [video_id, frame_name]
                line.extend(preds_line)
                line.append(str(res))

                fr.write((',').join(line))
                fr.write('\n')


def brightness_save_scores():
    scores_file = 'results/ijb_c_frames_brightness_scores.csv'
    video_frames = DataLoader.load_video_frames()

    with open(scores_file, mode='w') as fr:
        for video_id, frames_data in video_frames.items():
            frames_labels = []

            for frame_name, ground_truth in frames_data:
                # frame_path = dataset_path + frame_name
                frame_path = frame_name
                frames_labels.append(frame_name)

                draw = cv2.imread(frame_path)

                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

                x, y, w, h = ground_truth
                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)
                face = draw[y:y + h, x:x + w]

                brt_estimator = QualityEstimator(face)

                brt_score = brt_estimator.estimate_brightness()

                line = [video_id, frame_name, str(brt_score)]

                fr.write((',').join(line))
                fr.write('\n')


def contrast_save_scores():
    scores_file = 'results/ijb_c_frames_contrast_scores.csv'
    video_frames = DataLoader.load_video_frames()

    with open(scores_file, mode='w') as fr:
        for video_id, frames_data in video_frames.items():
            frames_labels = []

            for frame_name, ground_truth in frames_data:
                # frame_path = dataset_path + frame_name
                frame_path = frame_name
                frames_labels.append(frame_name)

                draw = cv2.imread(frame_path)

                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

                x, y, w, h = ground_truth
                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)
                face = draw[y:y + h, x:x + w]

                brt_estimator = QualityEstimator(face)

                cr_score = brt_estimator.get_image_contrast()

                line = [video_id, frame_name, str(cr_score)]

                fr.write((',').join(line))
                fr.write('\n')


if __name__ == '__main__':
    dataset_path = 'C:\\akharche\\MasterThesis\\dataset\\aligned_images_DB\\'

    #################
    # QNet Pipeline
    #################
    model_qnet = '../models/FaceQnet.h5'

    face_qnet_save_scores(model_qnet, dataset_path)

    #################
    # QNet Light Pipeline
    #################
    # model_qnet = './models/mobilenet_faceqnet-02-0.0013.hdf5'

    # face_qnet_light_save_scores(model_qnet, dataset_path)

    #################
    # BRISQUE Pipeline
    #################
    # brisque_pipeline_knn(dataset_path)

    # brisque_save_scores(dataset_path)

    ################
    # FIIQA Pipeline
    ################
    # fiiqa_save_scores()

    #################
    # Brightness Pipeline
    #################

    # brightness_save_scores()

    #################
    # Contrast Pipeline
    #################

    # contrast_save_scores()

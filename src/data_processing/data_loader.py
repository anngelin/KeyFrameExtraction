import numpy as np
import pathlib
import os
from collections import namedtuple

import cv2




PATH_TO_METADATA = '/home/datasets/images/IJB/IJB-C/'
IJBC_NAMES = PATH_TO_METADATA + 'ijbc_subject_names.csv'
IJBC_METADATA = PATH_TO_METADATA + 'protocols/ijbc_metadata.csv'
IJBC_PROBE_MIXED = PATH_TO_METADATA + 'protocols/ijbc_1N_probe_mixed.csv'

PATH_TO_FEATURE_VECTOR_MOBILENET = 'face_features_mobilenet_vgg_ijbc_images_train.csv'

Quality_score = namedtuple('Quality_score', ['frame_id', 'score'])


class DataLoader:
    def __init__(self, metadata_path=PATH_TO_METADATA):
        self.metadata = pathlib.Path(metadata_path)

        labels, feature_v = self.load_feature_vector_by_frames(PATH_TO_FEATURE_VECTOR_MOBILENET)
        self.feature_vectors = dict(zip(labels, feature_v))

    def get_feature_vector(self, frame_id):
        return self.feature_vectors.get(frame_id)

    @staticmethod
    def get_subject_ids(subject_names=IJBC_NAMES):
        names_map = {}
        with open(subject_names, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                subject_id = data[0]
                subject_name = data[1]
                names_map[subject_id] = subject_name

        return names_map

    @staticmethod
    def map_frame_to_subject(metadata_path=IJBC_METADATA):
        frame_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                subject_id = data[0]
                frame_id = data[1]
                if frame_id not in frame_map and 'frame' in frame_id:
                    frame_map[frame_id] = subject_id

        return frame_map

    @staticmethod
    def map_frame_to_template_id(metadata_path=IJBC_PROBE_MIXED):
        frame_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                template_id = data[0]
                frame_id = data[2]
                if frame_id not in frame_map and 'frame' in frame_id:
                    frame_map[frame_id] = template_id

        return frame_map

    @staticmethod
    def map_imgs_to_template_id(metadata_path=IJBC_PROBE_MIXED):
        imgs_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                template_id = data[0]
                frame_id = data[2]
                if frame_id not in imgs_map:
                    imgs_map[frame_id] = template_id

        return imgs_map

    @staticmethod
    def map_template_to_id(metadata_path=IJBC_PROBE_MIXED):
        template_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                template_id = data[0]
                subject_id = data[1]
                if template_id not in template_map:
                    template_map[template_id] = subject_id

        return template_map

    @staticmethod
    def load_feature_vector(fv_path):
        labels = []
        feature_vectors = []

        with open(fv_path, 'r', encoding='utf-8') as fr:
            data = fr.readlines()
            for line in data:
                line = line.strip().split(',')
                frame_id, features = line[0], line[1:]
                labels.append(frame_id)
                features = np.array(features)
                feature_vectors.append(features.astype(np.float64))

        print("DONE!!!")

        return labels, np.array(feature_vectors)

    @staticmethod
    def load_feature_vector_by_frames(fv_path):
        labels = []
        feature_vectors = []

        with open(fv_path, 'r', encoding='utf-8') as fr:
            data = fr.readlines()
            for line in data:
                line = line.strip().split(',')
                frame_id, features = line[0], line[1:]
                labels.append(frame_id)
                features = np.array(features)
                feature_vectors.append(features.astype(np.float))

        print("DONE!!!")
        return labels, np.array(feature_vectors)

    @staticmethod
    def load_video_frames(data_path=IJBC_PROBE_MIXED, ground_truth=True):
        frames_map = {}
        frames_folder = PATH_TO_METADATA
        with open(data_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                template_id, frame_name = data[0], data[2]
                if template_id not in frames_map:
                    frames_map[template_id] = []
                if ground_truth:
                    ground_truth_values = data[4:]
                    frames_map[template_id].append((frames_folder+'images/'+frame_name, ground_truth_values))
                else:
                    frames_map[template_id].append(frames_folder+'images/'+frame_name)

        return frames_map

    @staticmethod
    def get_video_scores(scores_file_path):
        video_scores_map = {}

        with open(scores_file_path, mode='r', encoding='utf-8') as fr:
            all_data = fr.readlines()

            for line in all_data:
                video_id, frame_id, score = line.strip().split(',')
                frame_id = 'frames/' + str(pathlib.Path(frame_id).name)
                score = float(score)

                if video_id not in video_scores_map:
                    video_scores_map[video_id] = [Quality_score(frame_id, score)]
                else:
                    video_scores_map[video_id].append(Quality_score(frame_id, score))

        return video_scores_map

    @staticmethod
    def get_video_scores_fiiqa(scores_file_path):
        video_scores_map = {}

        with open(scores_file_path, mode='r', encoding='utf-8') as fr:
            all_data = fr.readlines()

            for line in all_data:
                line = line.strip().split(',')
                video_id, frame_id, score = line[0], line[1], line[5]
                frame_id = 'frames/' + str(pathlib.Path(frame_id).name)
                score = float(score)

                if video_id not in video_scores_map:
                    video_scores_map[video_id] = [Quality_score(frame_id, score)]
                else:
                    video_scores_map[video_id].append(Quality_score(frame_id, score))

        return video_scores_map

    @staticmethod
    def get_videos(metadata_path=IJBC_PROBE_MIXED):
        video_frames_map = {}

        with open(metadata_path, mode='r', encoding='utf-8') as fr:
            all_data = fr.readlines()

            for line in all_data[1:]:
                line = line.strip().split(',')
                video_id, frame_id = line[0], line[2]

                if 'img/' in frame_id:
                    continue

                if video_id not in video_frames_map:
                    video_frames_map[video_id] = []

                if 'frames' in frame_id:
                    video_frames_map[video_id].append(frame_id)

        return video_frames_map

    @staticmethod
    def calculate_unique_video(metadatafile_path):
        videos = DataLoader.map_template_to_id(metadatafile_path)
        return len(videos.keys())

    @staticmethod
    def get_data_topology(metadatafile_path, file_to_write):
        unique_videos = set()
        unique_faces = set()
        frames_video = {}

        aver_frame_num = []

        with open(file_to_write, mode='w', encoding='utf-8') as fw:
            with open(metadatafile_path, mode='r', encoding='utf-8') as fr:
                all_lines = fr.readlines()

                for line in all_lines[1:]:
                    line = line.strip().split(',')
                    video_id = line[0]
                    face_id = line[1]
                    frame_name = line[2]

                    # if video_id not in unique_videos:
                    unique_videos.add(video_id)

                    if video_id not in frames_video:
                        frames_video[video_id] = []

                    if 'frames/' in frame_name:
                        frames_video[video_id].append(frame_name)

                    else:
                        print('img or video')

                    unique_faces.add(face_id)

                fw.write('unique_videos, ' + str(len(unique_videos)) + '\n')
                fw.write('unique_faces, ' + str(len(unique_faces)) + '\n')

                print(unique_faces)

                for key, items in frames_video.items():
                    fw.write(str(key) + ',' + str(len(items)) + '\n')
                    aver_frame_num.append(len(items))

                fw.write('aver_frame_num, ' + str(np.sum(aver_frame_num) / len(aver_frame_num)) + '\n')

    def get_groundtruth(self, dataset):
        "{template_id: (sub_id, [frames...])}"
        template_map = {}
        with open(dataset, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                template_id, subject_id, frame_name = data[:3]
                if template_id not in template_map:
                    template_map[template_id] = {subject_id: []}
                if 'frames' in frame_name:
                    template_map[template_id][subject_id].append(frame_name)

        return template_map

    @staticmethod
    def get_meta():
        file_to_write = 'C:\\akharche\\MasterThesis\\ytf_train_meta.csv'
        dataset_train = pathlib.Path('C:\\akharche\\MasterThesis\\dataset\\faces\\')

        train_imgs = list(dataset_train.rglob('**/*.jpg'))
        train_imgs.extend(dataset_train.rglob('**/*.png'))

        with open(file_to_write, mode='w', encoding='utf-8') as fw:
            for img in train_imgs:
                line = str(img) + ','
                fw.write(line + '\n')

    @staticmethod
    def get_meta_frames():
        file_to_write = 'C:\\akharche\\MasterThesis\\ytf_frames_meta_cropped.csv'
        dataset_path = pathlib.Path('C:\\akharche\\MasterThesis\\dataset\\ytf_faces')
        target_classes_path = 'C:\\akharche\\MasterThesis\\lfw_ytf_classes.txt'

        target_classes = set()
        with open(target_classes_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                target_classes.add(line.strip())

        print(len(target_classes))

        data_imgs = list(dataset_path.rglob('**/*.jpg'))
        data_imgs.extend(dataset_path.rglob('**/*.png'))

        with open(file_to_write, mode='w', encoding='utf-8') as fw:
            i = 0
            for img in data_imgs:
                pts = img.parts
                img_class = pts[5]
                video_id = pts[6]

                if str(img_class) in target_classes:
                    print(img_class)
                    print(video_id)
                    line = str(img_class) + ',' + str(video_id) + ',' + str(img)
                    fw.write(line + '\n')
                    i += 1
            print(i)

    @staticmethod
    def get_diff():
        frames = 'C:\\akharche\\MasterThesis\\ytf_frames_meta.csv'
        imgs = 'C:\\akharche\\MasterThesis\\ytf_train_meta.csv'

        imgs_set = set()
        with open(imgs, mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split('\\')
                id = line[5]
                imgs_set.add(id)

        frames_set = set()
        with open(frames, mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split(',')
                id = line[0]
                frames_set.add(id)

        print(frames_set - imgs_set)

    @staticmethod
    def map_video_ytf(metadata_path):
        video_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                person = data[0]
                video_num = data[1]
                frame_path = data[2]
                frame_name = (pathlib.Path(frame_path)).name

                video_id = f'{person}_{video_num}'
                frame_id = f'{video_id}_{frame_name}'

                if video_id not in video_map:
                    video_map[video_id] = [frame_id]
                else:
                    video_map[video_id].append(frame_id)

        return video_map

    @staticmethod
    def load_feature_vector_ytf(fv_path):
        features_map = {}

        with open(fv_path, 'r', encoding='utf-8') as fr:
            data = fr.readlines()
            for line in data:
                line = line.strip().split(',')
                y, frame_path = line[0], line[1]

                frame_path = pathlib.Path(frame_path)
                frame_name = frame_path.name
                frame_parts = frame_path.parts
                frame_class, frame_video = frame_parts[6], frame_parts[7]

                frame_id = f'{frame_class}_{frame_video}_{frame_name}'

                features = line[2:]
                features_vec = np.array(features).astype(np.float64)

                features_map[frame_id] = (y, features_vec)

        print("DONE!!!")

        return features_map


if __name__ == '__main__':
    # data_loader = DataLoader(metadata_path='C:\\akharche\\MasterThesis\\ijbc_metadata.csv')
    # # frames = data_loader.map_frame_to_subject()
    # # print(len(frames['1']))
    # #
    # # d = data_loader.get_subject_ids('C:\\akharche\\MasterThesis\\ijbc_subject_names.csv')
    #
    # data = data_loader.get_groundtruth('C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv')

    # file_to_write = 'C:\\akharche\\MasterThesis\\ijbc_topology.csv'
    # DataLoader.get_data_topology('C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv', file_to_write)

    # DataLoader.get_meta()
    # DataLoader.get_diff()
    # DataLoader.crop_faces()
    DataLoader.get_meta_frames()

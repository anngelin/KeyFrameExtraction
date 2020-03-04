import pathlib
import numpy as np


PATH_TO_METADATA = '/home/datasets/images/IJB/IJB-C/'
IJBC_NAMES = PATH_TO_METADATA + 'ijbc_subject_names.csv'
IJBC_METADATA = PATH_TO_METADATA + 'protocols/ijbc_metadata.csv'
IJBC_PROBE_MIXED = PATH_TO_METADATA + 'protocols/ijbc_1N_probe_mixed.csv'


class DataLoader:
    def __init__(self, metadata_path=PATH_TO_METADATA):
        self.metadata = pathlib.Path(metadata_path)

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
                feature_vectors.append(features.astype(np.float))

        print("DONE!!!")
        return labels, np.array(feature_vectors)

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

    def map_frame_to_video(self):
        pass


# if __name__ == '__main__':
#     data_loader = DataLoader(metadata_path='C:\\akharche\\MasterThesis\\ijbc_metadata.csv')
#     # frames = data_loader.map_frame_to_subject()
#     # print(len(frames['1']))
#     #
#     # d = data_loader.get_subject_ids('C:\\akharche\\MasterThesis\\ijbc_subject_names.csv')
#
#     data = data_loader.get_groundtruth('C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv')
#     print(data['11065'])
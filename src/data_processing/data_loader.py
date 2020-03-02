import pathlib


class DataLoader:
    def __init__(self, metadata_path):
        self.metadata = pathlib.Path(metadata_path)

    def get_subject_ids(self, subject_names):
        names_map = {}
        with open(subject_names, 'r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                subject_id = data[0]
                subject_name = data[1]
                names_map[subject_id] = subject_name

        return names_map

    def map_frame_to_subject(self):
        frame_map = {}
        with self.metadata.open('r', encoding='utf-8') as csvreader:
            all_data = csvreader.readlines()
            for line in all_data[1:]:
                data = line.strip().split(',')
                subject_id = data[0]
                frame_id = data[1]
                if subject_id not in frame_map:
                    frame_map[subject_id] = []
                if 'frame' in frame_id:
                    frame_map[subject_id].append(frame_id)

        return frame_map

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
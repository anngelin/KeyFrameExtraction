import numpy as np


def clean_data(data_path, file_to_save):
    with open(file_to_save, mode='w', encoding='utf-8') as fw:
        new_lines = []
        with open(data_path, mode='r', encoding='utf-8') as fr:
            all_data = fr.readlines()
            for line in all_data:
                if 'img/' in line:
                    continue
                else:
                    new_lines.append(line)
        for line in new_lines:
            fw.write(line)


if __name__ == '__main__':
    data_path = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_1layer_scores_mobilenet.csv'
    file_to_save = 'C:\\akharche\\MasterThesis\\Scores\\ijb_c_frames_qnet_1layer_scores_mobilenet_frames.csv'
    clean_data(data_path, file_to_save)

import cv2
import pathlib

def get_groundtruth(dataset):
    " {frame_id: [template_id, x, y, w, h] "
    frame_map = {}
    # with open(dataset, 'r', encoding='utf-8') as csvreader:
    with open(dataset, 'r') as csvreader:

        all_data = csvreader.readlines()
        for line in all_data[1:]:
            data = line.strip().split(',')
            template_id, subject_id, frame_name = data[:3]
            x, y, w, h = data[4:]
            # if 'frames' in frame_name:
            if frame_name not in frame_map:
                frame_map[frame_name] = []
            frame_data = [x, y, w, h]
            frame_map[frame_name] = frame_data

    return frame_map


def save_faces():
    path_to_img = '/home/datasets/images/IJB/IJB-C/images/'
    # metadata_path = {0: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G1.csv',
    #                  1: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G2.csv'}

    metadata_path = '/home/student/akharchevnikova/ijbc_1N_all_imgs.csv'

    faces_dir = 'cropped_faces/img/'
    target_size = (224, 224)

    # for i in range(2):
    img_data = get_groundtruth(metadata_path)
    for frame_id, frame_data in img_data.items():
        print(frame_id)
        x, y, w, h = frame_data

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        frame_name = pathlib.Path(frame_id).name

        draw = cv2.imread(path_to_img + frame_id)
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        face_img = draw[y:y + h, x:x + w, :]
        face_img = cv2.resize(face_img, target_size)

        cv2.imwrite(faces_dir+frame_name, face_img)


if __name__ == '__main__':
    save_faces()

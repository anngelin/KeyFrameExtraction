import cv2
import numpy as np
import insightface


np.random.seed(123)  # for reproducibility


def get_groundtruth(dataset):
    "{frame_id: [template_id, x, y, w, h]"
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


def extract_mxnet_features(model, img_filepath):
    img = cv2.imread(img_filepath)
    embeddings = model.get_feature(img)
    if embeddings is None:
        print(img_filepath)
    return embeddings


def extract_facial_features_arcface_resnet_img():
    path_to_img = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = {0: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G1.csv',
                     1: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G2.csv'}

    features_file = 'face_features_arcface_resnet_50_ijbc_images_train.csv'

    model = insightface.model_zoo.get_model('arcface_r100_v1')
    model.prepare(ctx_id=-1)

    input_size = (112, 112)


    # with open(features_file, 'w', encoding='utf-8') as fw:
    with open(features_file, 'w', encoding='utf-8') as fw:
        # LOOP over 2 files
        for i in range(2):
            img_data = get_groundtruth(metadata_path.get(i))

            for frame_id, frame_data in img_data.items():
                print(frame_id)
                x, y, w, h = frame_data

                try:
                    draw = cv2.imread(path_to_img + frame_id)
                except Exception as e:
                    print(e)
                    continue

                y = int(y)
                x = int(x)
                w = int(w)
                h = int(h)

                face = draw[y:y + h, x:x + w]
                image = cv2.resize(face, input_size)

                emb = model.get_embedding(image)[0]

                feature_str = (',').join([str(f) for f in emb])
                res = (',').join([frame_id, feature_str])
                fw.write(res + '\n')

    print("SUCCESS!!!!!")


def extract_facial_features_arcface_resnet_frames():
    path_to_frames = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_probe_mixed.csv'

    features_file = 'face_features_arcface_resnet_100_ijbc_frames.csv'

    frames_data = get_groundtruth(metadata_path)

    model = insightface.model_zoo.get_model('arcface_r100_v1')
    model.prepare(ctx_id=-1)

    input_size = (112, 112)

    # with open(features_file, 'w', encoding='utf-8') as fw:
    with open(features_file, 'w') as fw:
        for frame_id, frame_data in frames_data.items():
            print(frame_id)
            x, y, w, h = frame_data

            try:
                draw = cv2.imread(path_to_frames + frame_id)
            except Exception as e:
                print(e)
                continue

            y = int(y)
            x = int(x)
            w = int(w)
            h = int(h)

            face = draw[y:y + h, x:x + w]
            image = cv2.resize(face, input_size)

            emb = model.get_embedding(image)[0]

            feature_str = (',').join([str(f) for f in emb])
            res = (',').join([frame_id, feature_str])
            fw.write(res + '\n')

    print("SUCCESS!!!!!")

if __name__ == '__main__':
    # extract_facial_features_mobilenet_img()

    # extract_facial_features_arcface_resnet_img()

    extract_facial_features_arcface_resnet_frames()


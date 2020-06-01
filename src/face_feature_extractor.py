import tensorflow as tf
import keras
import cv2
import numpy as np

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


def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph


class TensorFlowInference:
    def __init__(self, frozen_graph_filename, input_tensor, output_tensor, learning_phase_tensor=None, convert2BGR=True,
                 imageNetUtilsMean=True, additional_input_value=0):
        graph = load_graph(frozen_graph_filename, '')
        self.tf_sess = tf.Session(graph=graph)

        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=', self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=', self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None
        print('tf_learning_phase=', self.tf_learning_phase)
        if self.tf_input_image.shape.dims is None:
            w = h = 160
        else:
            _, w, h, _ = self.tf_input_image.shape
        self.w, self.h = int(w), int(h)

        self.convert2BGR = convert2BGR
        self.imageNetUtilsMean = imageNetUtilsMean
        self.additional_input_value = additional_input_value

    def preprocess_image(self, img, crop_center, is_file):
        if is_file:
            img = cv2.imread(img)  # In color by default

        if crop_center:
            orig_w, orig_h = 250, 250
            # img = misc.imread(img_filepath, mode='RGB')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = misc.imresize(img, (origimg = cv2.imread(img)_w, orig_h), interp='bilinear')
            cv2.resize(src=img, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            w1, h1 = 128, 128
            dw = (orig_w - w1) // 2
            dh = (orig_h - h1) // 2
            box = (dw, dh, orig_w - dw, orig_h - dh)
            img = img[dh:-dh, dw:-dw]

        # x = misc.imresize(img, (self.w, self.h), interp='bilinear').astype(float)
        x = cv2.resize(src=img, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32)

        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean:  # imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else:  # vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            # x=(x-127.5)/128.0
            x /= 127.5
            x -= 1.
            # x=x/128.0-1.0
        return x

    def extract_features(self, img, crop_center=False, is_file=False):
        x = self.preprocess_image(img, crop_center, is_file)
        x = np.expand_dims(x, axis=0)
        feed_dict = {self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase] = self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        # preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).mean(axis=(0,1,2)).reshape(-1)
        return preds

    def close_session(self):
        self.tf_sess.close()


def extract_facial_features_mobilenet_frames():

    path_to_frames = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_probe_mixed.csv'
    model_path = '/home/student/akharchevnikova/models/vgg2_mobilenet.pb'
    features_file = 'face_features_mobilenet_vgg_all_ijbc.csv'

    tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                      output_tensor='reshape_1/Reshape:0',
                                      learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                      convert2BGR=True, imageNetUtilsMean=False)

    frames_data = get_groundtruth(metadata_path)

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
            features = tfInference.extract_features(face, crop_center=False, is_file=False)

            feature_str = (',').join([str(f) for f in features])
            res = (',').join([frame_id, feature_str])
            fw.write(res + '\n')

    print("SUCCESS!!!!!")


def extract_facial_features_mobilenet_img():
    path_to_img = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = {0: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G1.csv',
                     1: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G2.csv'}

    model_path = '/home/student/akharchevnikova/models/vgg2_mobilenet.pb'
    features_file = 'face_features_mobilenet_vgg_ijbc_images_train.csv'

    tfInference = TensorFlowInference(model_path, input_tensor='input_1:0',
                                      output_tensor='reshape_1/Reshape:0',
                                      learning_phase_tensor='conv1_bn/keras_learning_phase:0',
                                      convert2BGR=True, imageNetUtilsMean=False)

    # with open(features_file, 'w', encoding='utf-8') as fw:
    with open(features_file, 'w') as fw:
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
                features = tfInference.extract_features(face, crop_center=False, is_file=False)

                feature_str = (',').join([str(f) for f in features])
                res = (',').join([frame_id, feature_str])
                fw.write(res + '\n')

    print("SUCCESS!!!!!")


def extract_facial_features_vggface_frames():

    path_to_frames = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_probe_mixed.csv'
    model_path = '/home/student/akharchevnikova/models/vgg2_resnet.pb'
    features_file = 'face_features_vgg2_resnet_ijbc_frames.csv'

    tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                      output_tensor='pool5_7x7_s1:0',
                                      convert2BGR=True, imageNetUtilsMean=False)

    frames_data = get_groundtruth(metadata_path)

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
            features = tfInference.extract_features(face, crop_center=False, is_file=False)

            feature_str = (',').join([str(f) for f in features])
            res = (',').join([frame_id, feature_str])
            fw.write(res + '\n')

    print("SUCCESS!!!!!")


def extract_facial_features_vggface_img():
    path_to_img = '/home/datasets/images/IJB/IJB-C/images/'
    # metadata_path = {0: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G1.csv',
    #                  1: '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_gallery_G2.csv'}

    metadata_path = '/home/student/akharchevnikova/ijbc_1N_all_imgs.csv'

    model_path = '/home/student/akharchevnikova/models/vgg2_resnet.pb'
    features_file = 'face_features_vgg2_resnet_ijbc_images_all.csv'

    tfInference = TensorFlowInference(model_path, input_tensor='input:0',
                                      output_tensor='pool5_7x7_s1:0',
                                      convert2BGR=True, imageNetUtilsMean=False)

    # with open(features_file, 'w', encoding='utf-8') as fw:
    with open(features_file, 'w') as fw:
        # LOOP over 2 files

        # for i in range(2):
        # img_data = get_groundtruth(metadata_path.get(i))

        img_data = get_groundtruth(metadata_path)

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
            features = tfInference.extract_features(face, crop_center=False, is_file=False)

            feature_str = (',').join([str(f) for f in features])
            res = (',').join([frame_id, feature_str])
            fw.write(res + '\n')

    print("SUCCESS!!!!!")

if __name__ == '__main__':
    extract_facial_features_vggface_img()
    #
    # extract_facial_features_vggface_frames()



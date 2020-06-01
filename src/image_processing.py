import pathlib
import cv2
import tensorflow as tf
import detect_face

from mtcnn import MTCNN

from image_quality_assessment.image_quality_estimation import QualityEstimator, FaceQNetQualityEstimator


class FacialImageProcessing:
    def __init__(self):
        self.detector = MTCNN()

    def detect_face(self, filename):
        draw = cv2.imread(filename)

        points = self.detector.detect_faces(draw)

        bounding_box = points[0]['box']

        x = int(bounding_box[0])
        y = int(bounding_box[1])
        width = int(bounding_box[2])
        height = int(bounding_box[3])
        face = draw[y:y + height, x:x + width]

        return face

    def crop_image(image):
        pass

    def process_image(self, filename):
        draw = cv2.imread(filename)

        height, width, channels = draw.shape
        if width > 640 or height > 480:
            draw = cv2.resize(draw, (min(width, 640), min(height, 480)))
        print(draw.shape)
        qualityEstimator = QualityEstimator(draw)
        br = qualityEstimator.get_image_brightness()
        c = qualityEstimator.get_image_contrast()

        print("Brightness: ", br)
        print("Contrast: ", c)

        QualityEstimator.varianceOfLaplacian(draw)

        return draw

    def process_all_images(self, args):
        for filename in args:
            draw = self.process_image(filename)
            # draw=cv2.resize(draw, (192,192))

            #TODO: detecet face
            # draw = imgProcessing.show_detection_results(draw)

            return draw


if __name__ == '__main__':
    filename = 'C:\\akharche\\MasterThesis\\TestImages\\3.jfif'
    # ip = FacialImageProcessing()
    # ip.process_image(filename)

    filenames = ['C:\\akharche\\MasterThesis\\TestImages\\3.jfif', 'C:\\akharche\\MasterThesis\\TestImages\\1.jfif']
    faces = []

    for f in filenames:
        face = FacialImageProcessing.detect_face(f)
        faces.append(face)

    qe = FaceQNetQualityEstimator(faces)
    qe.estimate_quality_qnet()

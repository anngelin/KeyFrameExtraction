import cv2
import numpy as np
import math
from enum import Enum
from keras.models import load_model


class Thresholds(Enum):
    good_lum = 0.2
    good_cont = 0.4
    good_sharp = 2700
    good_orient = 100


class QualityEstimator:
    r = 0
    g = 1
    b = 2

    def __init__(self, image):
        self.image = image
        self.img_height, self.img_width = image.shape[0], image.shape[1]

        self.avg_standard_lum = 0.0  # Luminocity
        self.avg_contrast = 0.0  # Contrast

    def _get_intensity(self):
        return ((0.2126 * self.image[..., self.r]) + (0.7152 * self.image[..., self.g]) +
                (0.0722 * self.image[..., self.b])) / 255

    def estimate_brightness(self):
        """Get brightness of image in RGB format """

        intensity = self._get_intensity()
        self.avg_standard_lum = np.sum(intensity) / (self.img_height * self.img_width)
        return self.avg_standard_lum

    def estimate_contrast(self):
        """Get contrast of image in RGB format """

        if self.avg_standard_lum == 0:
            self.estimate_brightness()

        intensity = self._get_intensity()

        self.avg_contrast = math.sqrt(
            (np.sum(intensity ** 2) / (self.img_height * self.img_width)) - (self.avg_standard_lum ** 2))

    def estimate_standard_luminosity(self):
        pass

    def get_image_stats(self):
        return self.avg_standard_lum, self.avg_contrast

    def get_image_brightness(self):
        self.estimate_brightness()
        return self.avg_standard_lum

    def get_image_contrast(self):
        self.estimate_contrast()
        return self.avg_contrast

    @staticmethod
    def varianceOfLaplacian(img):
        ''''LAPV' algorithm (Pech2000)'''
        lap = cv2.Laplacian(img, ddepth=-1)  # cv2.cv.CV_64F)
        stdev = cv2.meanStdDev(lap)[1]
        s = stdev[0] ** 2
        print(s[0])
        return s[0]


class DLQualityEstimator:
    #  No-Reference, end-to-end Quality Assessment (QA) based on deep learning
    model_qnet = '../models/FaceQnet.h5'

    def __init__(self, images, model='qnet'):
        self.model = load_model(self.model_qnet)
        self.images = images

    def estimate_quality_qnet(self):
        X = []
        for frame in self.images:
            img = cv2.resize(frame, (224, 224))
            X.append(img)

        X = np.array(X, dtype=np.float32)

        # Extract quality scores for the samples
        m = 0.7
        s = 0.5
        batch_size = 1
        scores = self.model.predict(X, batch_size=batch_size, verbose=1)

        print(scores)

        return scores

    def get_best_frames(self, k=5):
        """ k - the number of frames to return"""
        scores = self.estimate_quality_qnet()
        return np.sort(scores)[k:]




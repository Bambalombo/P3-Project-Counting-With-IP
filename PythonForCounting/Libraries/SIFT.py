import cv2 as cv
import numpy as np
import bordering


def differenceOfGaussian(image):
    images = []
    for k in range(0, 5)
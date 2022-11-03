import numpy as np
import cv2 as cv
from . import bordering as bd

def outlineFromBinary(img, kernelRadius):
    """
    :param img:
    :param kernelRadius:
    :return:

    Funktion der laver et eroted billede og trækker det fra det originale billede,
    for at få et billede med outlinen af objekter.

    """
    kernel = np.ones((kernelRadius * 2 + 1, kernelRadius * 2 + 1), dtype=np.uint8) * 255
    erodedImg = np.zeros((img.shape[0] - kernelRadius * 2, img.shape[1] - kernelRadius * 2), dtype=np.uint8)
    for y in range(erodedImg.shape[0]):
        for x in range(erodedImg.shape[1]):
            slice = img[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            if np.allclose(kernel, slice):
                erodedImg[y, x] = 255
            else:
                erodedImg[y, x] = 0

    paddedImage = bd.addPadding(erodedImg, img.shape[0], img.shape[1], np.uint8(0))
    output = cv.subtract(img, paddedImage)
    return output


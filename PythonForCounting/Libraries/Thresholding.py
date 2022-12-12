import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def getColorHistogramFromSlice(slice):
    outputHistograms = []
    outputBinEdges = []
    for channel in range(slice.shape[2]):
        histogram, binEdges = np.histogram(slice[:, :, channel], bins=256, range=(0, 255))
        outputHistograms.append(histogram)
        outputBinEdges.append(binEdges)
    return outputHistograms, outputBinEdges


def calculateIntensity(pixel):
    """
    Returnerer intensiteten af en enkelt pixel
    :param pixel:
    :return:
    """
    bgrMean = pixel[0] / 3 + pixel[1] / 3 + pixel[2] / 3
    intensity = bgrMean / 255
    return intensity


def makeImageBinaryIntensityThreshold(img, threshold):
    """
    :param img:
    :param threshold:
    :return black and white output:

    Funktion der gør pixels med en intensity værdi større end threshold sorte
    og pixels med en intensity værdi mindre end threshold hvide
    Skal tage imod et bgr image

    Linus er den bedste 😎
    -------
    """
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if calculateIntensity(img[y, x]) < threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0
    return output

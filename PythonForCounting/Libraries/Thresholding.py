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


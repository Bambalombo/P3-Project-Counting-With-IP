import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def calculateImageHistogramBinVector(image, bins: int, factor: int):
    # Hvis Grayscale / har kun en farvekanal
    if (len(image.shape) == 2):
        hist = np.histogram(image, bins, [0, 256])

    # Hvis billedet har farvekanaler/er BGR
    elif (len(image.shape) == 3):
        # Vi laver et histrogram til hver farvekanal
        B_hist = np.histogram(image[:, :, 0], bins, [0, 256])
        G_hist = np.histogram(image[:, :, 1], bins, [0, 256])
        R_hist = np.histogram(image[:, :, 2], bins, [0, 256])
        # Vi fyrer alle histogrammer ind i røven ad hinanden i et ny array 'hist' for at få det som en feature vektor
        hist = np.concatenate((B_hist[0], G_hist[0], R_hist[0]))

    # normaliserer histogrammet således at værdierne ligger mellem 0 og 1
    hist = hist.astype(np.float64)
    if (hist.max() != 0 or None):
        hist /= int(hist.max())
        hist *= factor

    return hist


def showHistogram(input, bins: int, factor):
    # Tager billedet og fyrer alle pixelværdierne ind i et lang array
    imageArray = createPixelArray(input)

    # Numpy laver et 2D array med 2 arrays i.
        # Første array: holder antallet af værdier inden for den nuværende bin
        # Andet array: holder værdierne for hvor hvert nyt bin starter
    np.histogram(imageArray, bins, [0, 256])
    plt.hist(imageArray, bins, [0, 256])
    plt.title("histogram")
    plt.show()


def createPixelArray(image):
    """
    Created a 1D numpy array of each pixel in the image provided.

    method does the same as
    image.revel()
    method
    """
    pixelArray = []

    if (len(image.shape) == 2):
        for y, row in enumerate(image):
            for x, pixel in enumerate(row):
                pixelArray.append(pixel)
    elif (len(image.shape) == 3):
        for y, row in enumerate(image):
            for x, column in enumerate(row):
                for c, pixel in enumerate(column):
                    pixelArray.append(pixel)

    numpy1DArray = np.array(pixelArray)
    numpy1DArray = numpy1DArray.astype(float)

    return numpy1DArray


def calculateEuclidianDistance(feature_vector1, feature_vector2):
    dist = np.sqrt(np.sum((feature_vector1-feature_vector2)**2))
    return dist
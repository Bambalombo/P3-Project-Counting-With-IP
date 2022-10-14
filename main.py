#Importer OpenCV og numpy
import cv2 as cv
import numpy as np
import even_lighting as el
import morphology as morph
from matplotlib import pyplot as plt

#Læser billedet
img_paper = cv.imread('Images/paper.jpg')
img_candles = cv.imread('Images/fyrfadslys.jpg')
img_scarf = cv.imread('Images/scarf.jpeg')

scale_percent = 50  # percent of original size
scarf_width = int(img_scarf.shape[1] * scale_percent / 100)
scarf_height = int(img_scarf.shape[0] * scale_percent / 100)
dim = (scarf_width, scarf_height)

# resize image
img_scarf_resized = cv.resize(img_scarf, dim, interpolation=cv.INTER_AREA)


def makeGrayscale(img):
    """
    Returnerer et grayscale image ud fra det som man har puttet ind i funktionen
    :param img:
    :return:
    """
    output = np.zeros((img.shape[0],img.shape[1]), dtype = np.uint8)
    output[:,:] = img[:,:,0]*0.114 + img[:,:,1]*0.587 + img[:,:,2]*0.299
    return output


def calculateIntensity(pixel):
    """
    Returnerer intensiteten af en enkelt pixel
    :param pixel:
    :return:
    """
    bgrMean = pixel[0] / 3 + pixel[1] / 3 + pixel[2] / 3
    intensity = bgrMean / 255
    return (intensity)


def makeImageBinary(img,threshold):
    """
    :param img:
    :param threshold:
    :return black and white output:

    Funktion der gør pixels med en intensity værdi mindre end threshold sorte
    og pixels med en intensity værdi større end threshold hvide
    Skal tage imod et bgr image

    Linus er den bedste 😎
    -------
    """
    output = np.zeros((img.shape[0], img.shape[1]), dtype= np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if calculateIntensity(img[y, x]) < threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0
    return output


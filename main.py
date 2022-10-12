#Importer OpenCV og numpy
import cv2 as cv
import numpy as np
import even_lighting as el
from matplotlib import pyplot as plt

#Læser billedet
img_paper = cv.imread('Images/paper.jpg')
img_candles = cv.imread('Images/fyrfadslys.jpg')


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


cv.imshow('input_image_p',img_paper)
lpf_paper = cv.blur(img_paper, (25,25))
#cv.imshow("low_pass_filter_p",lpf_paper)
cv.imshow('lpf_illum_correction_p ',el.low_pass_lighting(img_paper,50))
cv.imshow('input_image_c',img_candles)
lpf_candles = cv.blur(img_candles, (25,25))
#cv.imshow("low_pass_filter_c",lpf_candles)
cv.imshow('lpf_illum_correction_c ',el.low_pass_lighting(img_candles,50))
cv.waitKey(0)

"""
grayscaleImage = makeGrayscale(img)
binaryImage = makeImageBinary(img, 0.5)

cv.imshow('original',img)
cv.imshow('grayscale', grayscaleImage)
cv.imshow('binary', binaryImage)
cv.waitKey(0)
"""


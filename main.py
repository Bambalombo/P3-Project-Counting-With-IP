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


print('jeg er her 31')
mean31 = el.illumination_lpf_mean_color(img_scarf_resized,31)
mean31c = el.illumination_lpf_mean_color(img_candles,31)
print('jeg er her 61')
mean61 = el.illumination_lpf_mean_color(img_scarf_resized,61)
mean61c = el.illumination_lpf_mean_color(img_candles,61)
print('jeg er her 91')
mean91 = el.illumination_lpf_mean_color(img_scarf_resized,91)
mean91c = el.illumination_lpf_mean_color(img_candles,91)
print('jeg er her 121')
mean121 = el.illumination_lpf_mean_color(img_scarf_resized,121)
mean121c = el.illumination_lpf_mean_color(img_candles,121)
print('jeg er her 151')
mean151 = el.illumination_lpf_mean_color(img_scarf_resized,151)
mean151c = el.illumination_lpf_mean_color(img_candles,151)
print('jeg er her mean')
hsl = el.illumination_lpf_mean_hsl(img_scarf_resized,3)
print('jeg er klar')

cv.imshow("original",img_scarf_resized)
cv.imshow("mean31",mean31)
cv.imshow("mean61",mean61)
cv.imshow("mean91",mean91)
cv.imshow("mean121",mean121)
cv.imshow("mean151",mean151)

cv.imshow("originalc",img_scarf_resized)
cv.imshow("mean31c",mean31c)
cv.imshow("mean61c",mean61c)
cv.imshow("mean91c",mean91c)
cv.imshow("mean121c",mean121c)
cv.imshow("mean151c",mean151c)
cv.waitKey(0)

"""
grayscaleImage = makeGrayscale(img)
binaryImage = makeImageBinary(img, 0.5)

cv.imshow('original',img)
cv.imshow('grayscale', grayscaleImage)
cv.imshow('binary', binaryImage)
cv.waitKey(0)
"""


#Importer OpenCV og numpy
import cv2 as cv
import numpy as np

#Læser billedet
img = cv.imread('Images/coins_evenlyLit.png')

def evenLighting():
    # read image - gøres ovenover
    # img = cv.imread("Images/.jpg")
    h, w, c = img.shape

    # display it
    cv.imshow("IMAGE", img)
    #cv2.imshow("THRESHOLD", thresh)
    #cv2.imshow("RESULT1", result1)
    #cv2.imshow("RESULT2", result2)
    cv.waitKey(0)


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


evenLighting()
cv.destroyAllWindows()

"""
grayscaleImage = makeGrayscale(img)
binaryImage = makeImageBinary(img, 0.5)

cv.imshow('original',img)
cv.imshow('grayscale', grayscaleImage)
cv.imshow('binary', binaryImage)
cv.waitKey(0)
"""


#Importer OpenCV og numpy
import cv2 as cv
import numpy as np

#Læser billedet
coinPicture = cv.imread('Images/coins_evenlyLit.png')
fyrfadPicture = cv.imread('Images/fyrfadslys.jpg')
colorTest = cv.imread('Images/RGB_test_billede.png')

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


def makeImageBinaryIntensityThreshold(img, threshold):
    """
    :param img:
    :param threshold:
    :return black and white output:

    Funktion der gør pixels med en intensity værdi mindre end threshold hvide
    og pixels med en intensity værdi større end threshold sorte
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

def makeImageBinaryRGBThreshold(img, rThresholdMin, rThresholdMax, gThresholdMin, gThresholdMax, bThresholdMin, bThresholdMax):
    """

    :param img:
    :param rThresholdMin:
    :param rThresholdMax:
    :param gThresholdMin:
    :param gThresholdMax:
    :param bThresholdMin:
    :param bThresholdMax:
    :return black and white output image:

    Laver et binært billede ud fra farve thresholds hvor man har en min og en maks threshold for hvert farve i rgb colorspace
    """
    output = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if img[y, x, 0] >= bThresholdMin and img[y, x, 0] <= bThresholdMax and img[y, x, 1] >= gThresholdMin and img[y, x, 1] <= gThresholdMax and img[y, x, 2] >= rThresholdMin and img[y, x, 2] <= rThresholdMax:
                output[y, x] = 255
            else:
                output[y, x] = 0
    return output


#grayscaleImage = makeGrayscale(picture)
#binaryImage = makeImageBinaryIntensityThreshold(picture, 0.5)

#binaryFromColor = makeImageBinaryRGBThreshold(fyrfadPicture,25,50,60,80,30,55)


#cv.imshow('original',picture)
#cv.imshow('grayscale', grayscaleImage)
#cv.imshow('binary', binaryImage)
#cv.imshow('binary from color', binaryFromColor)
cv.waitKey(0)
cv.destroyAllWindows()
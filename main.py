#Importer OpenCV og numpy
import cv2 as cv
import numpy as np

#Læser billedet
inputPicture = cv.imread('Images/coins_evenlyLit.png')

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
    return intensity


def makeImageBinaryIntensityThreshold(img,threshold):
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


def morphClose(img):
    erodeKernel = np.ones((3, 3), dtype=np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    morphErotedImage = cv.erode(img, erodeKernel, iterations=1)
    morphClosedImage = cv.morphologyEx(morphErotedImage, cv.MORPH_CLOSE, kernel, iterations=2)
    return morphClosedImage

def findAndDrawContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Number of objects: "+ str(len(contours)))
    imageWithContours = cv.drawContours(inputPicture, contours,-1, (0,255,0), 3)
    return imageWithContours

#grayscaleImage = makeGrayscale(picture)
binaryImage = makeImageBinaryIntensityThreshold(inputPicture, 0.5)
processedPicture = morphClose(binaryImage)
imageWithContours = findAndDrawContours(processedPicture)

#cv.imshow('original',picture)
#cv.imshow('grayscale', grayscaleImage)
resizedImage = cv.resize(imageWithContours,(200,300))
cv.imshow('binary', resizedImage)
cv.waitKey(0)
cv.destroyAllWindows()
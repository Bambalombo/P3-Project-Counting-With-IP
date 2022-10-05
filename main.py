import cv2 as cv
import numpy as np

img = cv.imread('Images/coins_evenlyLit.png')
output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
intensityImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
whiteImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
whiteImg.fill(255)
kernel = np.ones((3, 3), np.uint8)


def calculateIntensity(img):
    bgrMean = img[0] / 3 + img[1] / 3 + img[2] / 3
    intensity = bgrMean / 255
    return (intensity)


def makeGrayscale(img):
    output = np.zeros((img.shape[0],img.shape[1]), dtype = np.uint8)
    output[:,:] = img[:,:,0]/3 + img[:,:,1]/3 + img[:,:,2]/3
    return output


def areSame(A, B):
    result = True
    for i, row in enumerate(A):
        for j, pixel in enumerate(A):
            if (A[i][j] != B[i][j]):
                result = False
    return result


def applyIntensity(img):
    #for y, row in enumerate(img):
    #    for x, pixel in enumerate(row):
    #        if calculateIntensity(img[y, x]) < 0.5:
    #            intensityImg[y, x] = 255
    #        else:
    #            intensityImg[y, x] = 0

    ret, thresh = cv.threshold(img, 127, 255, 0)
    image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=10)

    image, contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(closing, contours, -1, (0, 255, 0), 10)
    output = makeGrayscale(img)
    cv.imshow('input',img)
    cv.imshow('output', output)
    print("NUMBER OF CUNTS: " + str(len(contours)))
    cv.waitKey(0)
    cv.destroyAllWindows()


applyIntensity(img)
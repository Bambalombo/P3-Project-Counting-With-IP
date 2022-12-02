import cv2 as cv

def makeGrayscale(image):
    output = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return output

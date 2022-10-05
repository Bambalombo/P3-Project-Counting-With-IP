#Importer OpenCV og numpy
import cv2 as cv
import numpy as np

#Returnere et grayscale image ud fra det som man har puttet ind i funktionen
def makeGrayscale(img):
    output = np.zeros((img.shape[0],img.shape[1]), dtype = np.uint8)
    output[:,:] = img[:,:,0]*0.114 + img[:,:,1]*0.587 + img[:,:,2]*0.299
    return output

#Returnere intensiteten af en enkelt pixel
def calculateIntensity(pixel):
    bgrMean = pixel[0] / 3 + pixel[1] / 3 + pixel[2] / 3
    intensity = bgrMean / 255
    return (intensity)

import math

import cv2 as cv
import numpy as np
from . import bordering as bd


def differenceOfGaussian(image, kernelsize, SD, numberOfDoGs=5):

    borderimage = bd.addborder_reflect(image, kernelsize)
    blurredPictures = [convolve(borderimage,makeGuassianKernel(kernelsize, SD))]
    #blurredPictures = [cv.GaussianBlur(image,(0,0),sigmaX=SD,sigmaY=SD)]
    k = 2**(1./(numberOfDoGs-2))
    for i in range(1, numberOfDoGs+1):
        guassiankernel = makeGuassianKernel(kernelsize, SD * (k**i))
        blurredPictures.append(convolve(borderimage, guassiankernel))
        #blurredPictures.append(cv.GaussianBlur(image,(0,0),sigmaX=SD * (k**i),sigmaY=SD * (k**i)))

    DoG = []
    for (bottomPicture,topPicture) in zip(blurredPictures,blurredPictures[1:]):
        DoG.append(cv.subtract(bottomPicture, topPicture))
    return DoG


def makeGuassianKernel(kernelsize, SD):
    radius = int((kernelsize - 1) / 2)  # kernel radius
    guassian = [1/(math.sqrt(2*math.pi) * SD) * math.exp(-0.5 * (x/SD)**2) for x in range(-radius, radius+1)]
    guassianKernel = np.zeros((kernelsize, kernelsize))

    for y in range(guassianKernel.shape[0]):
        for x in range(guassianKernel.shape[1]):
            guassianKernel[y, x] = guassian[y] * guassian[x]
    guassianKernel *= 1/guassianKernel[0, 0]
    return guassianKernel

def convolve(image, kernel):
    kernelSize = kernel.shape[0]
    sumKernel = kernel/np.sum(kernel)
    output = np.zeros((image.shape[0] - kernelSize + 1, image.shape[1] - kernelSize + 1, image.shape[2]), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            for channel in range(output.shape[2]):
                slice = image[y: y+kernelSize, x: x+kernelSize, channel]
                output[y, x, channel] = np.sum(slice * sumKernel)
    return output


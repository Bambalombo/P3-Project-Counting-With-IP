import math

import cv2 as cv
import numpy as np
from . import bordering as bd


def differenceOfGaussian(image, SD, octave, numberOfDoGs=5):
    gaussianKernel = makeGuassianKernel(SD*octave)
    borderimage = bd.addborder_reflect(image, gaussianKernel.shape[0])
    blurredPictures = [convolve(borderimage,gaussianKernel)]
    #blurredPictures = [cv.GaussianBlur(image,(0,0),sigmaX=SD*octave,sigmaY=SD*octave)]
    k = (octave*2)**(1./(numberOfDoGs-2))
    for i in range(1, numberOfDoGs+1):
        guassiankernel = makeGuassianKernel(SD * (k**i))
        blurredPictures.append(convolve(borderimage, guassiankernel))
        #blurredPictures.append(cv.GaussianBlur(image,(0,0),sigmaX=(SD * (k**i)),sigmaY=(SD * (k**i))))

    DoG = []
    for (bottomPicture,topPicture) in zip(blurredPictures,blurredPictures[1:]):
        DoG.append(cv.subtract(topPicture,bottomPicture))
    return DoG


def makeGuassianKernel(SD):
    kernelsize = int(math.ceil(6*SD)) // 2 * 2 + 1
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
    borderimage = bd.addborder_reflect(image, kernelSize)
    sumKernel = kernel/np.sum(kernel)
    if len(borderimage.shape) == 3:
        output = np.zeros((borderimage.shape[0] - kernelSize + 1, borderimage.shape[1] - kernelSize + 1, borderimage.shape[2]))
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                for channel in range(output.shape[2]):
                    slice = borderimage[y: y+kernelSize, x: x+kernelSize, channel]
                    output[y, x, channel] = np.sum(slice * sumKernel)
        return output
    else:
        output = np.zeros((borderimage.shape[0] - kernelSize + 1, borderimage.shape[1] - kernelSize + 1))
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                slice = borderimage[y: y + kernelSize, x: x + kernelSize]
                output[y, x] = np.sum(slice * sumKernel)
        return output

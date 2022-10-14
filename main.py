#Importer OpenCV og numpy
import cv2 as cv
import numpy as np
from collections import deque
#Læser billedet
inputPicture = cv.imread('Images/DillerCoins.jpg')
inputPicture = cv.resize(inputPicture,(600,800))
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


def edgeWithSobel(img):
    kernelRadius = 1
    sobelVerKernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.uint8)
    sobelHorKernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype= np.uint8)
    sobelKernelSum = np.sum(sobelVerKernel)

    verticalApply = np.zeros(((img.shape[0]-(2*kernelRadius+1)),(img.shape[1]-(2*kernelRadius+1))), dtype=np.uint8)
    horizontalApply = verticalApply.copy()
    for y in range(verticalApply.shape[0]):
        for x in range(verticalApply.shape[1]):
            slice = img[y:y+sobelVerKernel.shape[0],x:x+sobelVerKernel.shape[1]]
            verticalApply[y,x] = (np.sum(slice*sobelVerKernel))

    for y in range(horizontalApply.shape[0]):
        for x in range(horizontalApply.shape[1]):
            slice = img[y:y+sobelHorKernel.shape[0],x:x+sobelHorKernel.shape[1]]
            horizontalApply[y,x] = (np.sum(slice*sobelHorKernel))

    output = cv.add(verticalApply,horizontalApply)
    return output
def addZeroPadding(img, sizey,sizex):
    """

    :param img:
    :param sizey:
    :param sizex:
    :return:

    rescales picture with zeropadding to size y,size x
    """
    paddedImage = np.zeros((sizey,sizex), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            paddedImage[y+int((sizey-img.shape[0])/2),x+int((sizex-img.shape[1])/2)] = img[y,x]
    return paddedImage

def outlineFromBinary(img, kernelRadius):
    """
    :param img:
    :param kernelRadius:
    :return:

    Funktion der laver et eroted billede og trækker det fra det originale billede,
    for at få et billede med outlinen af objekter.

    """
    kernel = np.ones((kernelRadius*2+1,kernelRadius*2+1),dtype=np.uint8)*255
    erodedImg = np.zeros((img.shape[0]-kernelRadius*2,img.shape[1]-kernelRadius*2), dtype=np.uint8)
    #erodedImg = cv.erode(img, kernel)
    for y in range(erodedImg.shape[0]):
        for x in range(erodedImg.shape[1]):
            slice = img[y:y+kernel.shape[0],x:x+kernel.shape[1]]
            if np.allclose(kernel,slice):
                erodedImg[y,x] = 255
            else:
                erodedImg[y,x] = 0

    paddedImage = addZeroPadding(erodedImg,img.shape[0],img.shape[1])
    output = cv.subtract(img,paddedImage)
    return output

def grassfire(img):
    burningImage = addZeroPadding(img.copy(),img.shape[0]+2,img.shape[1]+2)

    burnQueue = deque()
    blobs = []
    currentblob = []
    imageNextPos = []
    lastLoopingPixel = []
    for y in range(burningImage.shape[0]-2):
        for x in range(burningImage.shape[1]-2):
            if not imageNextPos:
                #print('i get next pixel')
                imageNextPos = [y+1,x+1]
                lastLoopingPixel = imageNextPos
            else:
                if lastLoopingPixel == imageNextPos:
                    imageNextPos = [y + 1, x + 1]
                    lastLoopingPixel = imageNextPos
            if burningImage[imageNextPos[0],imageNextPos[1]] == 255 and imageNextPos not in blobs[:] and imageNextPos not in currentblob:
                #print('i find white pixel')
                currentblob.append([imageNextPos[0],imageNextPos[1]])
                burningImage[imageNextPos[0],imageNextPos[1]] = 0
                for yPixel in range(-1,2):
                    for xPixel in range(-1,2):
                        #print('checking for nearby pixels')
                        if burningImage[imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] == 255 and [imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] not in currentblob and [imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] not in burnQueue:
                            #print('adding to burnqueue')
                            burnQueue.append([imageNextPos[0]+yPixel,imageNextPos[1]+xPixel])
                if len(burnQueue) != 0:
                    #print('i get here ')
                    imageNextPos = burnQueue.pop()
                else:
                    blobs.append(currentblob)
                    print(currentblob)
                    currentblob.clear()
                    imageNextPos.clear()
    print(str(len(blobs)))


print('making picture binary')
binaryImage = makeImageBinaryIntensityThreshold(inputPicture, 0.5)
print('morphing')
processedPicture = morphClose(binaryImage)
print('outlining')
outlineImage = outlineFromBinary(processedPicture,3)
print('counting blobs')
grassfire(outlineImage)

cv.imshow('original',inputPicture)
cv.imshow('binary',binaryImage)
cv.imshow('processed', processedPicture)
cv.imshow('outline', outlineImage)
cv.waitKey(0)
cv.destroyAllWindows()
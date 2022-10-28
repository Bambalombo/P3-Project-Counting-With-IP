#Importer OpenCV og numpy
import cv2 as cv
import numpy as np
from collections import deque
from Libraries import even_lighting as el
from Libraries import morphology as morph
from matplotlib import pyplot as plt
from Libraries import Thresholding as th
from Libraries import bordering as bd

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

    Funktion der gør pixels med en intensity værdi større end threshold sorte
    og pixels med en intensity værdi mindre end threshold hvide
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
    for y in range(erodedImg.shape[0]):
        for x in range(erodedImg.shape[1]):
            slice = img[y:y+kernel.shape[0],x:x+kernel.shape[1]]
            if np.allclose(kernel,slice):
                erodedImg[y,x] = 255
            else:
                erodedImg[y,x] = 0

    paddedImage = bd.addPadding(erodedImg,img.shape[0],img.shape[1],np.uint8(0))
    output = cv.subtract(img,paddedImage)
    return output

def grassfire(img, whitepixel = 255):
    """

    :param img:
    :return:
    """
    def startBurning(startpos, burningImage):
        eightConnectivityarray = [[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1]]
        burnQueue = deque()
        # den blob vi er i gang med at detekte lige nu
        currentblob = []
        burnQueue.append(startpos)
        # kontrollere om der er noget i burnqueue, hvis der er så tager man denne som nextpos og forsætter med at brænde
        while(len(burnQueue) > 0):
            nextpos = burnQueue.pop()
            #tilføjer den næste position til vores blob
            currentblob.append([nextpos[0]-1,nextpos[1]-1])
            #burningImage[nextpos[0],nextpos[1]] = 0
            #kontrollere rund om positionen om der der flere pixels
            for i in eightConnectivityarray:
                checkpos = [(nextpos[0] + i[0]), (nextpos[1] + i[1])]
                if burningImage[checkpos[0],checkpos[1]] == whitepixel and [checkpos[0]-1,checkpos[1]-1] not in currentblob and checkpos not in burnQueue:
                    burnQueue.append(checkpos)
        # hvis burnqueue er tom er blobben færdig så vi returner den
        return currentblob


    #laver en kant af nuller omkring det originale billede, for at kunne detekte blobs i kanten
    burningImage = bd.addPadding(img.copy(),img.shape[0]+2,img.shape[1]+2,np.uint8(0))
    #en liste over alle vores blobs, indeholder lister med koordinater for pixels
    blobs = []

    for y in range(burningImage.shape[0]-2):
        for x in range(burningImage.shape[1]-2):
                if burningImage[y+1,x+1] == whitepixel:
                    found = False
                    for blob in blobs:
                        if [y,x] in blob:
                            found = True
                            break
                    if found == False:
                        blobs.append(startBurning([y+1,x+1], burningImage))
    return blobs


inputPicture = cv.imread('Images/coins_evenlyLit.png')
cv.imshow('input', inputPicture)
binary = makeImageBinaryIntensityThreshold(inputPicture,0.5)
cv.imshow('binary', binary)
eroded = morph.erode(binary,3)
morphed = morph.close(eroded,11)
cv.imshow('morphed', morphed)
outlined = outlineFromBinary(morphed,1)

cv.imshow('outlined', outlined)

blobs = grassfire(outlined)

print(blobs)
print(len(blobs))

cv.waitKey(0)
cv.destroyAllWindows()


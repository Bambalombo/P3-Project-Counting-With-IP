# Importer OpenCV og numpy
import cv2 as cv
import numpy as np
from collections import deque
from Libraries import even_lighting as el
from Libraries import morphology as morph
from matplotlib import pyplot as plt
from Libraries import Thresholding as th
from Libraries import bordering as bd
from Libraries import Outlining as outl


def makeImagePyramide(startingImage, scale, minWidth):
    """
    Returnere en pyramidegenerator, det vil sige den returnere et objekt man kan loope over, som så returnerer hvert enkelt billede
    :param startingImage: startsbilledet
    :param scale: hvor meget mindre størrelsen skal blive pr. spring
    :param minWidth: hvor stort det mindste billede skal være
    """
    #yield gør så man kan loope over pyramiden, og få et objekt hver gang yield bliver kaldt
    yield startingImage
    currentImage = cv.resize(startingImage, (int(startingImage.shape[1] / scale), int(startingImage.shape[0] / scale)))
    while currentImage.shape[1] > minWidth:
        yield currentImage
        currentImage = cv.resize(currentImage, (int(currentImage.shape[1] / scale), int(currentImage.shape[0] / scale)))


def makeGrayscale(img):
    """
    Returnerer et grayscale image ud fra det som man har puttet ind i funktionen
    :param img:
    :return:
    """
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    output[:, :] = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    return output



def edgeWithSobel(img):
    kernelRadius = 1
    sobelVerKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint8)
    sobelHorKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.uint8)
    sobelKernelSum = np.sum(sobelVerKernel)

    verticalApply = np.zeros(((img.shape[0] - (2 * kernelRadius + 1)), (img.shape[1] - (2 * kernelRadius + 1))),
                             dtype=np.uint8)
    horizontalApply = verticalApply.copy()
    for y in range(verticalApply.shape[0]):
        for x in range(verticalApply.shape[1]):
            slice = img[y:y + sobelVerKernel.shape[0], x:x + sobelVerKernel.shape[1]]
            verticalApply[y, x] = (np.sum(slice * sobelVerKernel))

    for y in range(horizontalApply.shape[0]):
        for x in range(horizontalApply.shape[1]):
            slice = img[y:y + sobelHorKernel.shape[0], x:x + sobelHorKernel.shape[1]]
            horizontalApply[y, x] = (np.sum(slice * sobelHorKernel))

    output = cv.add(verticalApply, horizontalApply)
    return output



def grassfire(img, whitepixel=255):
    """

    :param img:
    :return:
    """

    def startBurning(startpos, burningImage):
        eightConnectivityarray = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
        burnQueue = deque()
        # den blob vi er i gang med at detekte lige nu
        currentblob = []
        burnQueue.append(startpos)
        # kontrollere om der er noget i burnqueue, hvis der er så tager man denne som nextpos og forsætter med at brænde
        while (len(burnQueue) > 0):
            nextpos = burnQueue.pop()
            # tilføjer den næste position til vores blob
            currentblob.append([nextpos[0] - 1, nextpos[1] - 1])
            # burningImage[nextpos[0],nextpos[1]] = 0
            # kontrollere rund om positionen om der der flere pixels
            for i in eightConnectivityarray:
                checkpos = [(nextpos[0] + i[0]), (nextpos[1] + i[1])]
                if burningImage[checkpos[0], checkpos[1]] == whitepixel and [checkpos[0] - 1, checkpos[
                                                                                                  1] - 1] not in currentblob and checkpos not in burnQueue:
                    burnQueue.append(checkpos)
        # hvis burnqueue er tom er blobben færdig så vi returner den
        return currentblob

    # laver en kant af nuller omkring det originale billede, for at kunne detekte blobs i kanten
    burningImage = bd.addPadding(img.copy(), img.shape[0] + 2, img.shape[1] + 2, np.uint8(0))
    # en liste over alle vores blobs, indeholder lister med koordinater for pixels
    blobs = []

    for y in range(burningImage.shape[0] - 2):
        for x in range(burningImage.shape[1] - 2):
            if burningImage[y + 1, x + 1] == whitepixel:
                found = False
                for blob in blobs:
                    if [y, x] in blob:
                        found = True
                        break
                if not found:
                    blobs.append(startBurning([y + 1, x + 1], burningImage))
    return blobs


inputPicture = cv.imread('Images/coins_evenlyLit.png')
imagePyramide  = makeImagePyramide(inputPicture,1.5,100)

for i, image in enumerate(imagePyramide):
    cv.imshow(f'billede nr.     {i+1}',image)

cv.waitKey(0)
cv.destroyAllWindows()

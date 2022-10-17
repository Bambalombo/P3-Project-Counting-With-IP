#Importer OpenCV og numpy
import cv2 as cv
import numpy as np
from collections import deque
#import even_lighting as el
#import morphology as morph
#from matplotlib import pyplot as plt

#Læser billedet
inputPicture = cv.imread('PythonForCounting/Images/coins_evenlyLit.png')
#inputPicture = cv.resize(inputPicture,(600,800))

#Læser billedet
#img_paper = cv.imread('Images/paper.jpg')
#img_candles = cv.imread('Images/fyrfadslys.jpg')
#img_scarf = cv.imread('Images/scarf.jpeg')

#scale_percent = 50  # percent of original size
#scarf_width = int(img_scarf.shape[1] * scale_percent / 100)
#scarf_height = int(img_scarf.shape[0] * scale_percent / 100)
#dim = (scarf_width, scarf_height)

# resize image
#img_scarf_resized = cv.resize(img_scarf, dim, interpolation=cv.INTER_AREA)

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
    """

    :param img:
    :return:
    """

    #laver en kant af nuller omkring det originale billede, for at kunne detekte blobs i kanten
    burningImage = addZeroPadding(img.copy(),img.shape[0]+2,img.shape[1]+2)

    burnQueue = deque()
    #en liste over alle vores blobs, indeholder lister med koordinater for pixels
    blobs = []
    #den blob vi er i gang med at detecte lige nu
    currentblob = []
    #Holder et koordinatsæt for den næste position der skal kontrolleres
    imageNextPos = []
    #holder værdien for hvor vi er nået til i vores gennemgang af billedet
    lastLoopingPixel = []
    #starten af genngemgang af billede
    for y in range(burningImage.shape[0]-2):
        for x in range(burningImage.shape[1]-2):
            if lastLoopingPixel == imageNextPos:
                imageNextPos = [y + 1, x + 1]
                lastLoopingPixel = imageNextPos
            # kontrollere hvornår vi når til en hvid pixel, som ikke er i blobs eller currentblob
            if burningImage[imageNextPos[0],imageNextPos[1]] == 255 and imageNextPos not in blobs[:] and imageNextPos not in currentblob:
                #print('i find white pixel')
                #tilføjer denne pixel til currentblob
                currentblob.append([imageNextPos[0],imageNextPos[1]])
                #brænder pixelen ved at gøre den sort
                burningImage[imageNextPos[0],imageNextPos[1]] = 0
                #kontrollere de omkringliggende pixels
                for yPixel in range(-1,2):
                    for xPixel in range(-1,2):
                        # hvis det er den originale pixel, så går man videre
                        if yPixel == 0 and xPixel == 0:
                            continue
                        #print('checking for nearby pixels')
                        #kontrollere om de omkringliggende pixels er hvide, og om de allerede er blevet brændt eller ligger i burnqueue hvis ikke, tilføjer den dem til burnqueue
                        if burningImage[imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] == 255 and [imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] not in blobs[:] and [imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] not in currentblob and [imageNextPos[0]+yPixel,imageNextPos[1]+xPixel] not in burnQueue:
                            #print('adding to burnqueue')
                            burnQueue.append([imageNextPos[0]+yPixel,imageNextPos[1]+xPixel])
                # kontrollere om der er noget i burnqueue, hvis der er så tager man denne som nextpos
                if len(burnQueue) != 0:
                    #print('i get here ')
                    imageNextPos = burnQueue.pop()
                else:
                    #hvis burnqueue er tom er blobben færdig
                    #derfor lægger vi vores currentblob ind i blobs
                    blobs.append(currentblob)
                    print(currentblob)
                    #fjerner alt i vores currentblob så den er klar til næste blob
                    currentblob.clear()
                    #sætter vores nexpos tilbage til udgangspunktet, så vi er klar til at loope igen
                    imageNextPos = lastLoopingPixel
    return blobs

print('making picture binary')
binaryImage = makeImageBinaryIntensityThreshold(inputPicture, 0.5)
print('morphing')
processedPicture = morphClose(binaryImage)
print('outlining')
outlineImage = outlineFromBinary(processedPicture,3)
print('counting blobs')
blobs = grassfire(outlineImage)
print(len(blobs))

cv.imshow('original',inputPicture)
cv.imshow('binary',binaryImage)
cv.imshow('processed', processedPicture)
cv.imshow('outline', outlineImage)
cv.waitKey(0)
cv.destroyAllWindows()


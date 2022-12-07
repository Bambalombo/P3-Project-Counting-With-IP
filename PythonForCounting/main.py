# Importer OpenCV og numpy
import time

import cv2 as cv
import numpy as np
from collections import deque
from Libraries import even_lighting as el
from Libraries import morphology as morph
from matplotlib import pyplot as plt
from Libraries import Thresholding as th
from Libraries import bordering as bd
from Libraries import Outlining as outl
from Libraries import FeatureMatching as fm
from Libraries import SIFT
from Libraries.OurKeyPoint import KeyPoint
import time


def makeImagePyramide(startingImage, scale, minWidth):
    """
    Returnere en pyramidegenerator, det vil sige den returnere et objekt man kan loope over, som så returnerer hvert enkelt billede
    :param startingImage: startsbilledet
    :param scale: hvor meget mindre størrelsen skal blive pr. spring
    :param minWidth: hvor stort det mindste billede skal være
    """
    #yield gør så man kan loope over pyramiden, og få et objekt hver gang yield bliver kaldt
    currentImage = startingImage
    while currentImage.shape[1] > minWidth:
        yield currentImage
        currentImage = cv.resize(currentImage, (int(currentImage.shape[1] / scale), int(currentImage.shape[0] / scale)))


def windowSlider(image, windowSize: tuple, stepSize):
    """
    Returnere en slicegenerator, som genererer et slice for hvert step igennem et billede, looper man over generatoren kan man så lave image processing på hvert slice.
    :param image: Billedet man vil loope henover
    :param windowSize: størrelesen på slicet (y,x)
    :param stepSize: hvor stort et skridt man skal tage mellem hvert slice
    """
    for y in range(0,image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (y,x, image[y:y+windowSize[0], x:x+windowSize[1]])


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


def nonMaximumSupression(outlines, threshold, scores = None):
    boxes = np.array(outlines).astype("float")
    if not outlines:
        return []
    #vores liste over alle hits der er tilbage efter supression
    realHits = []

    #vores bounding box koordinater
    xleft = boxes[:,0]
    xright = boxes[:,1]
    yleft = boxes[:,2]
    yright = boxes[:,3]

    # beregner arealet af alle vores boundingboxes og gemmer dem
    areaOfBoundingBoxes = (xright - xleft + 1) * (yright - yleft + 1)
    differentAreas = set(areaOfBoundingBoxes)
    # laver et midlertidigt array som vi sortere
    # enten efter nederste højre boundingbox, eller efter score
    # det er midlertidigt, så vi kan bruge det som kondition i vores whileloop,
    # mens vi fjerne alle boundingboxes

    tempSortingArray = yleft
    if scores is not None:
        tempSortingArray = scores

    #argsort giver os et array af indexer med den laveste som index på plads nr. 0,
    # men vi vil gerne have det omvendt så det er højeste index som 0, derfor omvender vi arrayet
    tempSortingArray = np.argsort(tempSortingArray, kind='stable')[::-1]

    #mens vi ikke har kontrolleret alle boundingboxes
    while len(tempSortingArray) > 0:
        lastTempIndex = len(tempSortingArray) - 1
        ndi = tempSortingArray[lastTempIndex]
        realHits.append(ndi)

        # Vi finder et array af de mindste og største x,y koordinator, for at finde alle vores overlap
        # af vores vinduer
        overlapLeftX = np.maximum(xleft[ndi], xleft[tempSortingArray[:lastTempIndex]])
        overlapRightX = np.minimum(xright[ndi], xright[tempSortingArray[:lastTempIndex]])
        overlapBottomY = np.maximum(yleft[ndi], yleft[tempSortingArray[:lastTempIndex]])
        overlapTopY = np.minimum(yright[ndi], yright[tempSortingArray[:lastTempIndex]])

        # så laver vi et array som holder alle bredder og højder på vores overlaps
        # der ligges 1 til for at få den rent faktiske bredde, da man trækker pixelpositioner fra hinanden
        overlapWidths = np.maximum(0, overlapRightX - overlapLeftX + 1)
        overlapHeights = np.maximum(0, overlapTopY - overlapBottomY + 1)

        #arealet af alle de overlappende områder beregnes og divideres med det oprindelige array af arealer
        overlapArea = overlapWidths * overlapHeights
        #for at få hvor meget areal der er overlap, mod hvor meget areal der rent faktisk var i boundingboxen
        overlapMatches = []
        for i in overlapArea:
            if i in differentAreas:
                overlapMatches.append(True)
            else:
                overlapMatches.append(False)
        overlapMatches = np.array(overlapMatches, dtype=bool)
        overlapAreaRatio = np.where(overlapMatches,1,overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]])
        #overlapAreaRatio = np.where(overlapArea in areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]], 1,overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]])
        #overlapAreaRatio = overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]]

        #slet alle indexes hvor overlapratio er større end threshold
        tempSortingArray = np.delete(tempSortingArray, np.concatenate(([lastTempIndex], np.where(overlapAreaRatio > threshold)[0])))

    #finder vores rigtigte outlines og scores or returnerer dem zippet
    realScores = np.array(scores)[realHits]
    realOutlines = boxes[realHits].astype("int")
    return realScores, realOutlines


def returnScoreAndImageWithOutlines(image, hits, nmsTreshhold = 0.3):
    if not hits:
        return 0,image
    scores = []
    outlines = []
    for (dist,outline) in hits:
        scores.append(dist)
        outlines.append(outline)
    outputImage = image.copy()
    hitScores, hitOutlines = nonMaximumSupression(outlines ,nmsTreshhold, scores)
    for i, (startx, endx, starty, endy) in enumerate(hitOutlines):
        cv.rectangle(outputImage, (startx,starty), (endx,endy), (0,255,0), 2)
        cv.putText(outputImage,f'Score: {int(scores[i])}', (startx,starty),cv.FONT_HERSHEY_PLAIN, 1, (255,255,255))

    return len(hitScores), outputImage


def main():
    inputPicture = cv.imread('Images/fyrfadslys.jpg')
    #inputPicture = cv.resize(inputPicture, (int(inputPicture.shape[1]/10), int(inputPicture.shape[0]/10)))
    userSlice = cv.imread('Images/red_candle_cutout.jpg')
    #userSlice = cv.resize(userSlice, (int(userSlice.shape[1]/10), int(userSlice.shape[0]/10)))

    sliceFeatureVector = fm.calculateImageHistogramBinVector(userSlice,16,500)
    scaleRatio = 1.5
    imagePyramid = makeImagePyramide(inputPicture, scaleRatio, 150)
    #definere vinduestørrelsen, tænker den skulle laves ud fra inputbilledet
    windowSize = (int(userSlice.shape[0]), int(userSlice.shape[1]))

    #vores liste over hits
    hits = []

    #looper over alle billeder i billedpyramiden, man behøver ikke at lave pyramiden først, den kan laves på samme linje hernede
    for i, image in enumerate(imagePyramid):
        #looper over alle vinduerne i billedet
        for (y,x,window) in windowSlider(image,windowSize,int(windowSize[0]/3)):
            #Vinduet kan godt blive lavet halvt uden for billedet, hvis dette ikke er ønsket kan vi skippe den beregning i loopet men det er lige en diskussion vi skal have i gruppen
            if(window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]):
                continue
            #Lav vores image processing her
            currentWindowVector = fm.calculateImageHistogramBinVector(window, 16, 500)
            euc_dist = fm.calculateEuclidianDistance(sliceFeatureVector, currentWindowVector)
            if euc_dist < 900:
                hits.append([euc_dist, [x*(scaleRatio**i), x*(scaleRatio**i) + (window.shape[1]*(scaleRatio**i)), y*(scaleRatio**i), y*(scaleRatio**i) + (window.shape[0]*(scaleRatio**i))]])

    score, doneImage = returnScoreAndImageWithOutlines(inputPicture,hits, 0.1)
    print(score)
    cv.imshow('input', inputPicture)
    cv.imshow('userSlice',userSlice)
    cv.imshow('output', doneImage)


def testSift():
    input_picture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    input_picture = cv.resize(input_picture,(0,0),fx=0.5,fy=0.5)
    greyscaleInput = makeGrayscale(input_picture.copy())
    sift = cv.SIFT_create()
    keypoints = sift.detect(greyscaleInput, None)
    print(f':OpenCV SIFT keypoints found: {len(keypoints)}')

    img = cv.drawKeypoints(greyscaleInput, keypoints, input_picture)
    cv.imshow('sift',img)


def testMaxima():
    # create pseudo image array
    img1 = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    img2 = np.array([[2, 2, 2, 2, 2, 2], [2, 69, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]])
    img3 = np.array([[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]])
    img_array = []
    img_array.append(img1)
    img_array.append(img2)
    img_array.append(img3)

    img4 = np.array([1,2])

    print(img_array)


def testGuassian():
    inputPicture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    greyscaleInput = makeGrayscale(inputPicture.copy())
    greyscaleInputAsFloat32 = greyscaleInput.astype("float32")

    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground.png')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    greyscaleInputAsFloat32_user = greyscaleInput_user.astype("float32")

    keypoints_picture = []
    keypoints_user_marked_area = []
    scalefactor = 2
    SD = 1.6
    for p, image in enumerate(makeImagePyramide(greyscaleInputAsFloat32_user,scalefactor,20)):

        print('Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, SD, scalefactor,5)

        print('Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images,DoG, p,SD, scalefactor)

        print(f'Creating feature descriptors ...')
        print('Checking for duplicate keypoints ...')

        print(f'\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints,keypoints_user_marked_area)
        print(f'\t - new keypoints found in octave {p} : {len(sorted_keypoints)}')

        keypoints_with_descriptors = SIFT.makeKeypointDescriptors(sorted_keypoints,Gaussian_images)
        print(f'\t - keypoints with descriptors found in {p} : {len(keypoints_with_descriptors)}')

        keypoints_user_marked_area.extend(keypoints_with_descriptors)

    for p, image in enumerate(makeImagePyramide(greyscaleInputAsFloat32,scalefactor,20)):

        print('Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, SD, scalefactor,5)

        print('Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images,DoG, p,SD, scalefactor)

        print(f'Creating feature descriptors ...')
        print('Checking for duplicate keypoints ...')

        print(f'\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints,keypoints_picture)
        print(f'\t - new keypoints found in octave {p} : {len(sorted_keypoints)}')

        keypoints_with_descriptors = SIFT.makeKeypointDescriptors(sorted_keypoints,Gaussian_images)
        print(f'\t - keypoints with descriptors found in {p} : {len(keypoints_with_descriptors)}')

        keypoints_picture.extend(keypoints_with_descriptors)

    match_keypoints = SIFT.matchDescriptors(keypoints_user_marked_area,keypoints_picture)
    print(f'Drawing keypoints ...')
    for keypoint in match_keypoints:
        # print(keypoint)
        cv.circle(inputPicture, (int(round(keypoint[1].coordinates[1])),int(round(keypoint[1].coordinates[0]))), 4, color=(0, 255, 0),thickness=-1)
    print(f'   Our SIFT keypoints found: {len(match_keypoints)}')
    cv.imshow('keypoints', inputPicture)


if __name__ == "__main__":
    print(f'~~~ STARTING TIMER ~~~')
    startTime = time.time()

    #main()
    testGuassian()
    #testMaxima()
    #testSift()

    print(f'~~~ TIMER ENDED: TOTAL TIME = {time.time() - startTime} s ~~~')
    cv.waitKey(0)
    cv.destroyAllWindows()



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

def windowSlider(image, windowSize: tuple, stepSize):
    """
    Returnere en slicegenerator, som genererer et slice for hvert step igennem et billede, looper man over generatoren kan man så lave image processing på hvert slice.
    :param image: Billedet man vil loope henover
    :param windowSize: størrelesen på slicet (y,x)
    :param stepSize: hvor stort et skridt man skal tage mellem hvert slice
    """
    for y in range(0,image.shape[0], stepSize):
        for x in range(0, image.shape[1],stepSize):
            yield (y,x, image[y:y+windowSize[0],x:x+windowSize[1]])

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


def temp_test():
    """
    En funktion der indeholder alle mine metodekald når jeg tester forskellige features undervejs.
    -------
    ### Kan slettes efter behov. ###
    """

    img1 = cv.imread('Images/fyrfadslys.jpg')
    img2 = cv.imread('Images/DillerCoins.jpg')
    img3 = cv.imread('Images/coins_evenlyLit.png')

    img1_vector = fm.calculateImageHistogramBinVector(img1, 16, 500)
    img2_vector = fm.calculateImageHistogramBinVector(img2, 16, 500)
    img3_vector = fm.calculateImageHistogramBinVector(img3, 16, 500)

    print(img1_vector.shape)
    print(img2_vector.shape)
    print(img3_vector.shape)

    print((img1_vector.astype(int)))
    print((img2_vector.astype(int)))
    print((img3_vector.astype(int)))

    print(f'1-2: {fm.calculateEuclidianDistance(img1_vector,img2_vector)}')
    print(f'1-3: {fm.calculateEuclidianDistance(img1_vector,img3_vector)}')
    print(f'2-3: {fm.calculateEuclidianDistance(img2_vector,img3_vector)}')

    cv.imshow("img1",img1)
    cv.imshow("img2",img2)
    cv.imshow("img3",img3)

    fm.showHistogram(img1,16,500)

    cv.waitKey(0)
    cv.destroyAllWindows()


def nonMaximumSuppression(outlines, threshold, scores=None):
    boxes = np.array(outlines).astype("float")
    if not outlines:
        return []
    # vores liste over alle hits der er tilbage efter supression
    real_hits = []

    # vores bounding box koordinater
    x_left = boxes[:, 0]
    x_right = boxes[:, 1]
    y_left = boxes[:, 2]
    y_right = boxes[:, 3]

    # beregner arealet af alle vores boundingboxes og gemmer dem
    area_of_bounding_boxes = (x_right - x_left + 1) * (y_right - y_left + 1)
    different_areas = set(area_of_bounding_boxes)
    # laver et midlertidigt array som vi sortere
    # enten efter nederste højre boundingbox, eller efter score
    # det er midlertidigt, så vi kan bruge det som kondition i vores whileloop,
    # mens vi fjerne alle boundingboxes

    temp_sorting_array = y_left
    if scores is not None:
        temp_sorting_array = scores

    # argsort giver os et array af indexer med den laveste som index på plads nr. 0,
    # men vi vil gerne have det omvendt så det er højeste index som 0, derfor omvender vi arrayet
    temp_sorting_array = np.argsort(temp_sorting_array, kind='stable')[::-1]

    # mens vi ikke har kontrolleret alle boundingboxes
    while len(temp_sorting_array) > 0:
        last_temp_index = len(temp_sorting_array) - 1
        ndi = temp_sorting_array[last_temp_index]
        real_hits.append(ndi)

        # Vi finder et array af de mindste og største x,y koordinator, for at finde alle vores overlap
        # af vores vinduer
        overlap_left_x = np.maximum(x_left[ndi], x_left[temp_sorting_array[:last_temp_index]])
        overlap_right_x = np.minimum(x_right[ndi], x_right[temp_sorting_array[:last_temp_index]])
        overlap_bottom_y = np.maximum(y_left[ndi], y_left[temp_sorting_array[:last_temp_index]])
        overlap_top_y = np.minimum(y_right[ndi], y_right[temp_sorting_array[:last_temp_index]])

        # så laver vi et array som holder alle bredder og højder på vores overlaps
        # der ligges 1 til for at få den rent faktiske bredde, da man trækker pixelpositioner fra hinanden
        overlap_widths = np.maximum(0, overlap_right_x - overlap_left_x + 1)
        overlap_heights = np.maximum(0, overlap_top_y - overlap_bottom_y + 1)

        # arealet af alle de overlappende områder beregnes og divideres med det oprindelige array af arealer
        overlap_area = overlap_widths * overlap_heights
        # for at få hvor meget areal der er overlap, mod hvor meget areal der rent faktisk var i boundingboxen
        overlap_matches = []
        for i in overlap_area:
            if i in different_areas:
                overlap_matches.append(True)
            else:
                overlap_matches.append(False)
        overlap_matches = np.array(overlap_matches, dtype=bool)
        overlap_area_ratio = np.where(overlap_matches, 1,
                                    overlap_area / area_of_bounding_boxes[temp_sorting_array[:last_temp_index]])
        # overlapAreaRatio = np.where(overlapArea in areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]], 1,overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]])
        # overlapAreaRatio = overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]]

        # slet alle indexes hvor overlapratio er større end threshold
        temp_sorting_array = np.delete(temp_sorting_array,
                                     np.concatenate(([last_temp_index], np.where(overlap_area_ratio > threshold)[0])))

    # finder vores rigtigte outlines og scores or returnerer dem zippet
    real_scores = np.array(scores)[real_hits]
    real_outlines = boxes[real_hits].astype("int")
    return real_scores, real_outlines


inputPicture = cv.imread('Images/TestImage6.jpeg')
userSlice = cv.imread('Images/TestImage6_slice.jpg')
#inputPicture = cv.resize(inputPicture, (0,0),fx=0.3, fy=0.3)
#userSlice = cv.resize(userSlice, (0,0),fx=0.3, fy=0.3)

sliceFeatureVector = fm.calculateImageHistogramBinVector(userSlice,16,500)

hit_coords = []
hit_scores = []

imagePyramid  = makeImagePyramide(inputPicture, 1.5, 150)
#definere vinduestørrelsen, tænker den skulle laves ud fra inputbilledet
windowSize = (int(userSlice.shape[0]), int(userSlice.shape[1]))
#looper over alle billeder i billedpyramiden, man behøver ikke at lave pyramiden først, den kan laves på samme linje hernede

for i, image in enumerate(imagePyramid):
    hit_count = 0
    #looper over alle vinduerne i billedet
    for (y,x,window) in windowSlider(image,windowSize,int(windowSize[0]/3)):
        #Vinduet kan godt blive lavet halvt uden for billedet, hvis dette ikke er ønsket kan vi skippe den beregning i loopet men det er lige en diskussion vi skal have i gruppen
        if(window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]):
            continue
        #Lav vores image processing her
        currentWindowVector = fm.calculateImageHistogramBinVector(window, 16, 500)
        euc_dist = fm.calculateEuclidianDistance(sliceFeatureVector, currentWindowVector)
        if (euc_dist < 900):
            hit_img = window

            x1 = int(round(x*(1.5**(i))))
            y1 = int(round(y*(1.5**(i))))
            x2 = int((x*(1.5**(i)))+hit_img.shape[1]*(1.5**(i)))
            y2 = int(y*(1.5**(i))+hit_img.shape[0]*(1.5**(i)))

            coords = [x1,x2,y1,y2]
            hit_coords.append(coords)
            hit_scores.append(euc_dist)

            hit_count += 1
            #print(f'{y,x} {euc_dist}')
            #print(x,y)
            #print(x+hit_img.shape[1], y+hit_img.shape[0])
            cv.rectangle(inputPicture, (x1, y1), (x2, y2), (0,255,50*i), 20)
            #cv.imshow(f'hit: {y,x}',window)

        #tegner en rektangel der går hen over billedet for illustrating purposes
        clone = image.copy()
        cv.rectangle(clone, (x, y), (x + windowSize[1], y + windowSize[0]), (50*i, 255, int(50*i)), 2)
        #cv.imshow("window", inputPicture)
        #cv.waitKey(1)
    print(f'found: {hit_count} hits')

#scores, outlines = nonMaximumSuppression(hit_coords,0.1, hit_scores)

#for hit in outlines:
#    x1, x2, y1, y2 = hit
#    cv.rectangle(inputPicture, (x1, y1), (x2, y2), (0, 255, 0), 20)

cv.imwrite('test_img6_no_nms.jpg',inputPicture)
cv.imshow('final image', inputPicture)
cv.waitKey(0)
cv.destroyAllWindows()

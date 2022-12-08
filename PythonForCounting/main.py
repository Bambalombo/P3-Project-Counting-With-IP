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
    # Picture
    input_picture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    input_picture2 = input_picture.copy()
    input_picture = cv.resize(input_picture,(0,0),fx=0.5,fy=0.5)
    greyscaleInput = makeGrayscale(input_picture.copy())

    # User area
    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground_large.jpg')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')

    # Picture keypoints
    sift = cv.SIFT_create()
    input_picture_keypoints, input_picture_descriptors = sift.detectAndCompute(greyscaleInput, None)
    print(f':OpenCV SIFT input keypoints found: {len(input_picture_keypoints)}')

    # User area keypoints
    sift_user = cv.SIFT_create()
    marked_area_keypoints, marked_area_descriptors = sift_user.detectAndCompute(greyscaleInput_user, None)

    print(f':OpenCV SIFT marked keypoints found: {len(marked_area_keypoints)}')

    #img = cv.drawKeypoints(greyscaleInput, input_picture_keypoints, input_picture)
    dist_array = []
    print(f'Drawing found picture keypoints ({len(input_picture_keypoints)})...')
    for user_descriptor, user_keypoint in zip(marked_area_descriptors,marked_area_keypoints):
        y1 = 80
        y2 = 130
        x1 = 60
        x2 = 110
        if y1 < user_keypoint.pt[1] < y2 and x1 < user_keypoint.pt[0] < x2:
            cv.circle(inputPicture_user,
                      (int(round(user_keypoint.pt[0])*2), int(round(user_keypoint.pt[1])*2)), 3,
                      color=(0, 255, 0), thickness=-1)

            for image_descriptor, image_keypoint in zip(input_picture_descriptors,input_picture_keypoints):
                dist = np.linalg.norm(user_descriptor - image_descriptor)
                dist_array.append(dist)
                if (dist) < 300:
                    cv.circle(input_picture2, (int(round(image_keypoint.pt[0])*2), int(round(image_keypoint.pt[1])*2)),
                              5, color=(255, 255, 0), thickness=-1)
                else:
                    # y1 = 430
                    # y2 = 480
                    # x1 = 610
                    # x2 = 660
                    # cv.rectangle(inputPicture2,(x1,y1),(x2,y2),(0,0,255),2)
                    cv.circle(input_picture2, (int(round(image_keypoint.pt[0])*2), int(round(image_keypoint.pt[1])*2)),
                              3, color=(0, 0, 255), thickness=-1)
                    # if y1 < keypoint.pt[0] < y2 and x1 < keypoint.pt[1] < x2:
                    #    cv.circle(inputPicture2, (int(round(keypoint.pt[1])),int(round(keypoint.pt[0]))), 3, color=(0, 255, 255),thickness=-1)
                    #    print(keypoint.descriptor)
                    #    picture_marked.append(keypoint.descriptor)

    print(f'average dist SIFT: {np.median(dist_array)}')

    cv.imshow('SIFT User keypoints', inputPicture_user)
    cv.imshow('SIFT Picture keypoints', input_picture2)
    #cv.imshow('sift',img)


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

def computeKeypointsWithDescriptorsFromImage(greyscale_input_image, scale_factor=2, SD=1.6):

    keypoints = []
    for p, image in enumerate(makeImagePyramide(greyscale_input_image.astype("float32"), scale_factor, 20)):
        print(f'{p}: Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, SD, scale_factor, 5)

        print(f'{p}: Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images, DoG, p, SD, scale_factor)

        print(f'{p}: Creating feature descriptors ...')
        print(f'{p}: Checking for duplicate keypoints ...')

        print(f'{p}:\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints, keypoints)
        print(f'{p}:\t - new keypoints found in octave {p} : {len(sorted_keypoints)}\n')

        keypoints.extend(SIFT.makeKeypointDescriptors(sorted_keypoints, Gaussian_images))

    return keypoints


def testGuassian():
    # Picture keypoints
    inputPicture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    print(inputPicture.shape)
    greyscaleInput = makeGrayscale(inputPicture.copy())
    print(f'Finding keypoints in full image:')
    input_picture_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput)

    # User area keypoints
    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground_large.jpg')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')
    marked_area_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput_user)

    # Find matches
    match_keypoints = SIFT.matchDescriptors(marked_area_keypoints,input_picture_keypoints)

    # Show results
    inputPicture2 = inputPicture.copy()
    print(f'Drawing found picture keypoints ({len(input_picture_keypoints)})...')
    picture_marked = []
    dist_array = []
    for user_keypoint in marked_area_keypoints:
        y1 = 90
        y2 = 100
        x1 = 70
        x2 = 90
        cv.rectangle(inputPicture_user,(x1,y1),(x2,y2),(0,0,255),1)
        if y1 < user_keypoint.coordinates[0] < y2 and x1 < user_keypoint.coordinates[1] < x2:
            cv.circle(inputPicture_user, (int(round(user_keypoint.coordinates[1])),int(round(user_keypoint.coordinates[0]))), 3, color=(0, 255, 0),thickness=-1)
            print('hep')
            for keypoint in input_picture_keypoints:
                dist = np.linalg.norm(user_keypoint.descriptor-keypoint.descriptor)
                dist_array.append(dist)
                if (dist) < 1300:
                    cv.circle(inputPicture2, (int(round(keypoint.coordinates[1])),int(round(keypoint.coordinates[0]))), 5, color=(255, 255, 0),thickness=-1)
                else:
                    cv.circle(inputPicture2, (int(round(keypoint.coordinates[1])),int(round(keypoint.coordinates[0]))), 3, color=(0, 0, 255),thickness=-1)


    """
    print(f'Drawing found user keypoints ({len(marked_area_keypoints)})...')
    user_marked = []
    for keypoint in marked_area_keypoints:
        y1 = 80
        y2 = 130
        x1 = 60
        x2 = 110
        cv.rectangle(inputPicture_user,(x1,y1),(x2,y2),(0,0,255),2)
        if y1 < keypoint.coordinates[0] < y2 and x1 < keypoint.coordinates[1] < x2:
            cv.circle(inputPicture_user, (int(round(keypoint.coordinates[1])),int(round(keypoint.coordinates[0]))), 3, color=(0, 255, 0),thickness=-1)
            print(keypoint.descriptor)
            user_marked.append(keypoint.descriptor)

    print(len(picture_marked),len(user_marked))

    for pic_kp in picture_marked:
        for user_kp in user_marked:
            dist = np.linalg.norm(pic_kp-user_kp)
            print(dist)
    """


    print(f'average dist OUR: {np.median(dist_array)}')

    print(f'Drawing found match picture keypoints ({len(match_keypoints)})...')
    for keypoint in match_keypoints:
        cv.circle(inputPicture, (int(round(keypoint[1].coordinates[1])),int(round(keypoint[1].coordinates[0]))), 3, color=(0, 255, 0),thickness=-1)

    print(f'Our SIFT keypoints found {len(match_keypoints)} matching keypoints')
    cv.imshow('OUR Marked keypoints', inputPicture_user)
    #cv.imshow('OUR Scene keypoints', inputPicture2)
    #cv.imshow('Match keypoints', inputPicture)

def expandMarkedArea(starting_coordinates, end_coordinates, input_picture):
    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    if 0 <= starting_coordinates[0] - height and end_coordinates[0] + height < input_picture.shape[0] \
            and 0 <= starting_coordinates[1] - width and end_coordinates[1] + width < input_picture.shape[1]:
        return input_picture[starting_coordinates[0] - height: end_coordinates[0] + height+1, starting_coordinates[1] - width: end_coordinates[1]+ width+1 ]
    else:
        return input_picture[starting_coordinates[0]:end_coordinates[0],starting_coordinates[1]:end_coordinates[1]]
def discardKeypointsOutsideMarkedArea(keypoints: [KeyPoint], starting_coordinates, end_coordinates):
    new_keypoints = []
    for keypoint in keypoints:
        if starting_coordinates[0] < keypoint.coordinates[0] < end_coordinates[0] and starting_coordinates[1] < keypoint.coordinates[1] < end_coordinates[1]:
            new_keypoints.append(keypoint)
    return new_keypoints


def testMatching(marked_area_start, marked_area_end):
    # Picture keypoints
    inputPicture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    print(inputPicture.shape)
    greyscaleInput = makeGrayscale(inputPicture.copy())
    print(f'Finding keypoints in full image:')
    input_picture_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput)

    # User area keypoints
    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground_large.jpg')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')
    marked_area_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput_user)

    for keypoint in marked_area_keypoints:
        keypoint.computeKeypointPointersInMarkedImage(marked_area_start, marked_area_end)


    matches = SIFT.matchDescriptors(marked_area_keypoints,input_picture_keypoints)
    for ref_keypoint_index, scene_matches in enumerate(matches):
        for keypointmatch in scene_matches:
            keypointmatch.computeKeypointPointersFromMatchingKeypoint(marked_area_keypoints[ref_keypoint_index])

    print(f'Our SIFT keypoints found {len(matches)} matching keypoints')
    cv.imshow('OUR Marked keypoints', inputPicture)
    cv.imshow('The marked Area', inputPicture_user)
def testMatchingOpenCV():
    # Picture
    input_picture = cv.imread('Images/DillerCoins.jpg')
    input_picture = cv.resize(input_picture,(0,0),fx=0.5,fy=0.5)
    greyscale_input = makeGrayscale(input_picture.copy())

    # User area
    inputPicture_user = cv.imread('Images/CoinCutout.png')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')

    # Picture keypoints
    sift = cv.SIFT_create()
    input_picture_keypoints, scene_descriptors = sift.detectAndCompute(greyscale_input, None)
    print(f':OpenCV SIFT input keypoints found: {len(input_picture_keypoints)}')

    # User area keypoints
    sift = cv.SIFT_create()
    marked_area_keypoints, marked_descriptors = sift.detectAndCompute(greyscaleInput_user, None)

    SIFT.matchKeypointsBetweenImages(marked_area_keypoints,input_picture_keypoints,marked_descriptors,scene_descriptors,greyscaleInput_user,greyscale_input)


def compareDescriptors():
    # User area keypoints
    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground_large.jpg')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')
    marked_area_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput_user)

    y1 = 90
    y2 = 100
    x1 = 70
    x2 = 90

    for keypoint in marked_area_keypoints:
        if y1 < keypoint.coordinates[0] < y2 and x1 < keypoint.coordinates[1] < x2:
            print('our descriptor')
            print(keypoint)

    sift = cv.SIFT_create()
    marked_area_keypoints, marked_descriptors = sift.detectAndCompute(greyscaleInput_user, None)
    for user_descriptor, user_keypoint in zip(marked_descriptors,marked_area_keypoints):
        if y1 < user_keypoint.pt[1] < y2 and x1 < user_keypoint.pt[0] < x2:
            print('sift descriptor')
            print(user_descriptor)



if __name__ == "__main__":
    print(f'~~~ STARTING TIMER ~~~')
    startTime = time.time()

    #main()
    #testGuassian()
    #testMaxima()
    #testSift()
    #testMatchingOpenCV()
    testMatching((12,12),(200,200))
    #compareDescriptors()

    print(f'~~~ TIMER ENDED: TOTAL TIME = {time.time() - startTime} s ~~~')
    cv.waitKey(0)
    cv.destroyAllWindows()



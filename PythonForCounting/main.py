# Importer OpenCV og numpy
import cv2 as cv
import numpy as np
from collections import deque
from Libraries import bordering as bd
from Libraries import FeatureMatching as fm
from Libraries import SIFT
from Libraries.OurKeyPoint import KeyPoint
import time
from Libraries import fastOpenCVVersion as openCVpipeline
import copy


def makeImagePyramide(starting_image, scale, min_width):
    """
    Returnere en pyramidegenerator, det vil sige den returnere et objekt man kan loope over, som så returnerer hvert
    enkelt billede :param starting_image: startsbilledet :param scale: hvor meget mindre størrelsen skal blive pr.
    spring :param min_width: hvor stort det mindste billede skal være
    """
    # yield gør så man kan loope over pyramiden, og få et objekt hver gang yield bliver kaldt
    current_image = starting_image
    while current_image.shape[1] > min_width:
        yield current_image
        current_image = cv.resize(current_image,
                                  (int(current_image.shape[1] / scale), int(current_image.shape[0] / scale)))


def windowSlider(image, windowSize: tuple, stepSize):
    """
    Returnere en slicegenerator, som genererer et slice for hvert step igennem et billede, looper man over
    generatoren kan man så lave image processing på hvert slice. :param image: Billedet man vil loope henover
    :param
    windowSize: størrelesen på slicet (y,x) :param stepSize: hvor stort et skridt man skal tage mellem hvert slice
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield y, x, image[y:y + windowSize[0], x:x + windowSize[1]]


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
    kernel_radius = 1
    sobel_ver_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint8)
    sobel_hor_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.uint8)
    sobel_kernel_sum = np.sum(sobel_ver_kernel)

    vertical_apply = np.zeros(((img.shape[0] - (2 * kernel_radius + 1)), (img.shape[1] - (2 * kernel_radius + 1))),
                              dtype=np.uint8)
    horizontal_apply = vertical_apply.copy()
    for y in range(vertical_apply.shape[0]):
        for x in range(vertical_apply.shape[1]):
            image_slice = img[y:y + sobel_ver_kernel.shape[0], x:x + sobel_ver_kernel.shape[1]]
            vertical_apply[y, x] = (np.sum(image_slice * sobel_ver_kernel))

    for y in range(horizontal_apply.shape[0]):
        for x in range(horizontal_apply.shape[1]):
            image_slice = img[y:y + sobel_hor_kernel.shape[0], x:x + sobel_hor_kernel.shape[1]]
            horizontal_apply[y, x] = (np.sum(image_slice * sobel_hor_kernel))

    output = cv.add(vertical_apply, horizontal_apply)
    return output


def grassfire(img, white_pixel=255):
    """

    :param img:
    :param white_pixel:
    :return:
    """

    def startBurning(startpos, burning_image):
        eight_connectivityarray = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
        burn_queue = deque()
        # den blob vi er i gang med at detekte lige nu
        currentblob = []
        burn_queue.append(startpos)
        # kontrollere om der er noget i burnqueue, hvis der er så tager man denne som nextpos og forsætter med at brænde
        while len(burn_queue) > 0:
            nextpos = burn_queue.pop()
            # tilføjer den næste position til vores blob
            currentblob.append([nextpos[0] - 1, nextpos[1] - 1])
            # burningImage[nextpos[0],nextpos[1]] = 0
            # kontrollere rund om positionen om der der flere pixels
            for i in eight_connectivityarray:
                checkpos = [(nextpos[0] + i[0]), (nextpos[1] + i[1])]
                if burning_image[checkpos[0], checkpos[1]] == white_pixel and \
                        [checkpos[0] - 1, checkpos[1] - 1] not in currentblob and checkpos not in burn_queue:
                    burn_queue.append(checkpos)
        # hvis burnqueue er tom er blobben færdig så vi returner den
        return currentblob

    # laver en kant af nuller omkring det originale billede, for at kunne detekte blobs i kanten
    burning_image = bd.addPadding(img.copy(), img.shape[0] + 2, img.shape[1] + 2, np.uint8(0))
    # en liste over alle vores blobs, indeholder lister med koordinater for pixels
    blobs = []

    for y in range(burning_image.shape[0] - 2):
        for x in range(burning_image.shape[1] - 2):
            if burning_image[y + 1, x + 1] == white_pixel:
                found = False
                for blob in blobs:
                    if [y, x] in blob:
                        found = True
                        break
                if not found:
                    blobs.append(startBurning([y + 1, x + 1], burning_image))
    return blobs


def nonMaximumSupression(outlines, threshold, scores=None):
    boxes = np.array(outlines).astype("float")
    if not outlines:
        return []
    # vores liste over alle hits der er tilbage efter supression
    realHits = []

    # vores bounding box koordinater
    xleft = boxes[:, 0]
    xright = boxes[:, 1]
    yleft = boxes[:, 2]
    yright = boxes[:, 3]

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

    # argsort giver os et array af indexer med den laveste som index på plads nr. 0,
    # men vi vil gerne have det omvendt så det er højeste index som 0, derfor omvender vi arrayet
    tempSortingArray = np.argsort(tempSortingArray, kind='stable')[::-1]

    # mens vi ikke har kontrolleret alle boundingboxes
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

        # arealet af alle de overlappende områder beregnes og divideres med det oprindelige array af arealer
        overlapArea = overlapWidths * overlapHeights
        # for at få hvor meget areal der er overlap, mod hvor meget areal der rent faktisk var i boundingboxen
        overlapMatches = []
        for i in overlapArea:
            if i in differentAreas:
                overlapMatches.append(True)
            else:
                overlapMatches.append(False)
        overlapMatches = np.array(overlapMatches, dtype=bool)
        overlapAreaRatio = np.where(overlapMatches, 1,
                                    overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]])
        # overlapAreaRatio = np.where(overlapArea in areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]], 1,overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]])
        # overlapAreaRatio = overlapArea / areaOfBoundingBoxes[tempSortingArray[:lastTempIndex]]

        # slet alle indexes hvor overlapratio er større end threshold
        tempSortingArray = np.delete(tempSortingArray,
                                     np.concatenate(([lastTempIndex], np.where(overlapAreaRatio > threshold)[0])))

    # finder vores rigtigte outlines og scores or returnerer dem zippet
    realScores = np.array(scores)[realHits]
    realOutlines = boxes[realHits].astype("int")
    return realScores, realOutlines


def returnScoreAndImageWithOutlines(image, hits, nmsTreshhold=0.3):
    if not hits:
        return 0, image
    scores = []
    outlines = []
    for (dist, outline) in hits:
        scores.append(dist)
        outlines.append(outline)
    outputImage = image.copy()
    hitScores, hitOutlines = nonMaximumSupression(outlines, nmsTreshhold, scores)
    for i, (startx, endx, starty, endy) in enumerate(hitOutlines):
        cv.rectangle(outputImage, (startx, starty), (endx, endy), (0, 255, 0), 2)
        cv.putText(outputImage, f'Score: {int(scores[i])}', (startx, starty), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    return len(hitScores), outputImage


def main(input_scene, slice_start, slice_end, scale_ratio=2, SD=1.6):
    user_slice = input_scene[slice_start[0]:slice_end[0],slice_start[1]: slice_end[1]]
    slice_feature_vector = fm.calculateImageHistogramBinVector(user_slice, 16, 500)
    windowSize = (int(user_slice.shape[0]), int(user_slice.shape[1]))

    greyscale_scene = makeGrayscale(input_scene.copy())
    keypoints_scene, keypoints_slice = computeKeypointsWithDescriptorsFromImage(greyscale_scene, slice_start, slice_end, scale_factor=scale_ratio, SD=SD)
    validated_slice_keypoints = SIFT.validateKeypoints(keypoints_slice, keypoints_scene)
    print(f'validated keypoints: {len(validated_slice_keypoints)}')

    best_keypoints_in_scene = SIFT.matchDescriptorsWithKeypointFromSlice(validated_slice_keypoints, keypoints_scene)
    image_pyramid = makeImagePyramide(input_scene, scale_ratio, windowSize[1])
    # definere vinduestørrelsen, tænker den skulle laves ud fra inputbilledet
    # vores liste over hits
    hits = []

    # looper over alle billeder i billedpyramiden, man behøver ikke at lave pyramiden først, den kan laves på samme linje hernede
    for i, image in enumerate(image_pyramid):
        # looper over alle vinduerne i billedet
        for (y, x, window) in windowSlider(image, windowSize, int(windowSize[0] / 3)):
            # Vinduet kan godt blive lavet halvt uden for billedet, hvis dette ikke er ønsket kan vi skippe den
            # beregning i loopet men det er lige en diskussion vi skal have i gruppen
            if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
                continue
            # Lav vores image processing her
            currentWindowVector = fm.calculateImageHistogramBinVector(window, 16, 500)
            hist_dist = fm.calculateEuclidianDistance(slice_feature_vector, currentWindowVector)
            keypoints_in_window = 0
            for array_of_keypoint_matches in best_keypoints_in_scene:
                for keypoint in array_of_keypoint_matches:
                    cv.circle(input_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 3, color=(0, 255, 0), thickness=-1)
                    key_y, key_x = keypoint.coordinates[0] * ((1/scale_ratio)**i), keypoint.coordinates[1] * ((1/scale_ratio)**i)
                    if x <= key_x <= x+windowSize[1] and y <= key_y <= y+windowSize[0] and keypoint.image_scale == ((1/scale_ratio)**i):
                        keypoints_in_window += 1

            if len(validated_slice_keypoints) != 0:
                if keypoints_in_window >= (0.25*len(validated_slice_keypoints)) and hist_dist < 900:
                    hits.append([hist_dist,
                             [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                              y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])
            else:
                if hist_dist < 900:
                    hits.append([hist_dist,
                             [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                              y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])


    score, doneImage = returnScoreAndImageWithOutlines(input_scene, hits, 0.1)
    print(score)
    cv.imshow('input', input_scene)
    cv.imshow('userSlice', user_slice)
    cv.imshow('output', doneImage)


def testSift():
    # Picture
    input_scene = cv.imread('Images/fyrfadslys.jpg')
    input_scene2 = input_scene.copy()
    greyscaleInput = makeGrayscale(input_scene.copy())

    # Picture keypoints
    sift = cv.SIFT_create()
    input_picture_keypoints, input_picture_descriptors = sift.detectAndCompute(greyscaleInput, None)
    print(f':OpenCV SIFT input keypoints found: {len(input_picture_keypoints)}')

    # img = cv.drawKeypoints(greyscaleInput, input_picture_keypoints, input_picture)
    dist_array = []
    print(f'Drawing found picture keypoints ({len(input_picture_keypoints)})...')
    # for user_descriptor, user_keypoint in zip(marked_area_descriptors, marked_area_keypoints):
    #     y1 = 80
    #     y2 = 130
    #     x1 = 60
    #     x2 = 110
    #     if y1 < user_keypoint.pt[1] < y2 and x1 < user_keypoint.pt[0] < x2:
    #         cv.circle(input_slice,
    #                   (int(round(user_keypoint.pt[0]) * 2), int(round(user_keypoint.pt[1]) * 2)), 3,
    #                   color=(0, 255, 0), thickness=-1)
    #
    #         for image_descriptor, image_keypoint in zip(input_picture_descriptors, input_picture_keypoints):
    #             dist = np.linalg.norm(user_descriptor - image_descriptor)
    #             dist_array.append(dist)
    #             if dist < 300:
    #                 cv.circle(input_scene2,
    #                           (int(round(image_keypoint.pt[0]) * 2), int(round(image_keypoint.pt[1]) * 2)),
    #                           5, color=(255, 255, 0), thickness=-1)
    #             else:
    #                 # y1 = 430
    #                 # y2 = 480
    #                 # x1 = 610
    #                 # x2 = 660
    #                 # cv.rectangle(inputPicture2,(x1,y1),(x2,y2),(0,0,255),2)
    #                 cv.circle(input_scene2,
    #                           (int(round(image_keypoint.pt[0]) * 2), int(round(image_keypoint.pt[1]) * 2)),
    #                           3, color=(0, 0, 255), thickness=-1)
    #                 # if y1 < keypoint.pt[0] < y2 and x1 < keypoint.pt[1] < x2: cv.circle(inputPicture2, (int(round(
    #                 # keypoint.pt[1])),int(round(keypoint.pt[0]))), 3, color=(0, 255, 255),thickness=-1) print(
    #                 # keypoint.descriptor) picture_marked.append(keypoint.descriptor)
    #
    # print(f'average dist SIFT: {np.median(dist_array)}')
    cv.drawKeypoints(input_scene2,input_picture_keypoints,input_scene2)
    cv.imshow('SIFT Picture keypoints', input_scene2)
    # cv.imshow('sift',img)


def computeKeypointsWithDescriptorsFromImage(greyscale_input_image, slice_start, slice_end, scale_factor=2.0, SD=1.6):
    keypoints = []
    keypoints_slice = []

    for p, image in enumerate(makeImagePyramide(greyscale_input_image.astype("float32"), scale_factor, 10)):
        print(f'{p}: Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, SD, scale_factor, 5)

        print(f'{p}: Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images, DoG, p, SD, scale_factor)
        print(f'{p}: Creating feature descriptors ...')
        print(f'{p}: Checking for duplicate keypoints ...')

        print(f'{p}:\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints, keypoints)
        print(f'{p}:\t - new keypoints found in octave {p} : {len(sorted_keypoints)}\n')
        #SIFT.resizeKeypoints(sorted_keypoints,scale_factor)
        keypoints.extend(SIFT.makeKeypointDescriptors(sorted_keypoints, Gaussian_images))

    for keypoint in keypoints:
        if slice_start[0] <= keypoint.coordinates[0] <= slice_end[0] and slice_start[1] <= keypoint.coordinates[1] <= slice_end[1]:
            keypoints_slice.append(copy.deepcopy(keypoint))

    return keypoints, keypoints_slice


def testGuassian():
    # Picture keypoints
    inputPicture = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    greyscaleInput = makeGrayscale(inputPicture.copy())
    print(f'Finding keypoints in scene image:')
    input_picture_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput)

    # User area keypoints
    inputPicture_user = cv.imread('Images/redCandleCutoutVaryingBackground_large.jpg')
    greyscaleInput_user = makeGrayscale(inputPicture_user.copy())
    print(f'Finding keypoints in marked area:')
    marked_area_keypoints = computeKeypointsWithDescriptorsFromImage(greyscaleInput_user)

    # Find matches
    match_keypoints = SIFT.matchDescriptorsWithKeypointFromSlice(marked_area_keypoints, input_picture_keypoints)

    # Show results
    inputPicture2 = inputPicture.copy()
    print(f'Drawing found picture keypoints ({len(input_picture_keypoints)})...')
    dist_array = []
    for user_keypoint in marked_area_keypoints:
        y1 = 90
        y2 = 100
        x1 = 70
        x2 = 90
        cv.rectangle(inputPicture_user, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if y1 < user_keypoint.coordinates[0] < y2 and x1 < user_keypoint.coordinates[1] < x2:
            cv.circle(inputPicture_user,
                      (int(round(user_keypoint.coordinates[1])), int(round(user_keypoint.coordinates[0]))), 3,
                      color=(0, 255, 0), thickness=-1)
            print('hep')
            for keypoint in input_picture_keypoints:
                dist = np.linalg.norm(user_keypoint.descriptor - keypoint.descriptor)
                dist_array.append(dist)
                if dist < 1300:
                    cv.circle(inputPicture2, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))),
                              5, color=(255, 255, 0), thickness=-1)
                else:
                    cv.circle(inputPicture2, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))),
                              3, color=(0, 0, 255), thickness=-1)

    print(f'average dist OUR: {np.median(dist_array)}')

    print(f'Drawing found match picture keypoints ({len(match_keypoints)})...')
    for keypoint in match_keypoints:
        cv.circle(inputPicture, (int(round(keypoint[1].coordinates[1])), int(round(keypoint[1].coordinates[0]))), 3,
                  color=(0, 255, 0), thickness=-1)

    print(f'Our SIFT keypoints found {len(match_keypoints)} matching keypoints')
    cv.imshow('OUR Marked keypoints', inputPicture_user)


def expandMarkedArea(starting_coordinates, end_coordinates, input_picture):
    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    if 0 <= starting_coordinates[0] - height and end_coordinates[0] + height < input_picture.shape[0] \
            and 0 <= starting_coordinates[1] - width and end_coordinates[1] + width < input_picture.shape[1]:
        return input_picture[starting_coordinates[0] - height: end_coordinates[0] + height + 1,
               starting_coordinates[1] - width: end_coordinates[1] + width + 1].copy()
    else:
        return input_picture[starting_coordinates[0]:end_coordinates[0],
               starting_coordinates[1]:end_coordinates[1]].copy()


def discardKeypointsOutsideMarkedArea(keypoints: [KeyPoint], starting_coordinates, end_coordinates):
    new_keypoints = []
    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    for keypoint in keypoints:
        if height < keypoint.coordinates[0] < height * 2 and width < keypoint.coordinates[1] < width * 2:
            new_keypoints.append(keypoint)

    return new_keypoints


def discardKeypointsOutsideMarkedAreaOpenCV(descriptors, keypoints: [KeyPoint], starting_coordinates, end_coordinates):
    new_keypoints = []
    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    for keypoint in keypoints:
        if height < keypoint.pt[0] < height * 2 and width < keypoint.pt[1] < width * 2:
            new_keypoints.append(keypoint)

    return new_keypoints


def testMatching(input_scene, marked_area_start, marked_area_end):
    # Picture keypoints
    input_scene = input_scene.copy()
    greyscale_scene = makeGrayscale(input_scene.copy())
    print(f'Finding keypoints in full image:')
    scene_keypoints, slice_keypoints = computeKeypointsWithDescriptorsFromImage(greyscale_scene, marked_area_start,marked_area_end)

    # Slice keypoints
    if len(slice_keypoints) == 0:
        print("no keypoints found in marked area")
        return None
    for keypoint in slice_keypoints:
        keypoint.computeKeypointPointersInMarkedImage((0, 0), (marked_area_end[0]-marked_area_start[0], marked_area_end[1] - marked_area_start[1]))

    matches = SIFT.matchDescriptors(slice_keypoints, scene_keypoints)
    for ref_keypoint_index, scene_matches in enumerate(matches):
        for keypointmatch in scene_matches:
            keypointmatch.computeKeypointPointersFromMatchingKeypoint(slice_keypoints[ref_keypoint_index])

    for scene_matches in matches:
        for keypoint in scene_matches:
            cv.circle(input_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 5,
                      color=(255, 0, 0), thickness=-1)
            cv.line(input_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))),
                    (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), (0, 0, 0),
                    thickness=3)
            cv.circle(input_scene, (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), 7,
                      color=(128, 178, 194), thickness=-1)
        break
    cv.imshow('OUR Marked keypoints', input_scene)


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
    for user_descriptor, user_keypoint in zip(marked_descriptors, marked_area_keypoints):
        if y1 < user_keypoint.pt[1] < y2 and x1 < user_keypoint.pt[0] < x2:
            print('sift descriptor')
            print(user_descriptor)


def testMatchingOpenCV(marked_area_start, marked_area_end):
    # Picture keypoints
    input_pic = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    input_pic = cv.resize(input_pic, (0, 0), fx=2, fy=2)
    greyscale_scene = makeGrayscale(input_pic.copy())

    print(f'Finding keypoints in full image:')
    sift = cv.SIFT_create()
    scene_keypoints, scene_descriptors = sift.detectAndCompute(greyscale_scene, None)

    # User area keypoints
    input_slice = expandMarkedArea(marked_area_start, marked_area_end, input_pic)
    greyscale_slice = makeGrayscale(input_slice.copy())
    print(f'Finding keypoints in marked area:')
    expanded_slice_keypoints, expanded_slice_descriptors = sift.detectAndCompute(greyscale_slice, None)
    marked_slice_keypoints = discardKeypointsOutsideMarkedArea(expanded_slice_keypoints, marked_area_start, marked_area_end)

    # Vi opretter nogle pointers for hvert keypoint i marked image
    keypoint_pointers = []
    for keypoint in marked_slice_keypoints:
        x, y = (keypoint.pt[0], keypoint.pt[1])

        center_y = int((keypoint.pt[1] - keypoint.pt[1]) / 2)
        center_x = int((keypoint.pt[0] - keypoint.pt[0]) / 2)

        pointing_point = (center_y, center_x)
        pointing_length = np.sqrt((center_y - y) ** 2 + (center_x - x) ** 2)# / self.image_scale
        pointing_angle = (np.rad2deg(np.arctan2(center_y - y, center_x - x)) - keypoint.orientation) % 360

        keypoint_pointers.append([pointing_point,pointing_length,pointing_angle])

    #matches = SIFT.matchDescriptors(marked_slice)
    """
    matches = SIFT.matchDescriptors(marked_slice_keypoints, scene_keypoints)
    for ref_keypoint_index, scene_matches in enumerate(matches):
        for keypointmatch in scene_matches:
            keypointmatch.computeKeypointPointersFromMatchingKeypoint(marked_slice_keypoints[ref_keypoint_index])

    for scene_matches in matches:
        for keypoint in scene_matches:
            cv.circle(input_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 5,
                      color=(255, 0, 0), thickness=-1)
            cv.line(input_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))),
                    (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), (0, 0, 0),
                    thickness=3)
            cv.circle(input_scene, (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), 7,
                      color=(128, 178, 194), thickness=-1)

    for keypoint in marked_slice_keypoints:
        cv.circle(input_slice, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 5,
                  color=(255, 0, 0), thickness=-1)
        cv.line(input_slice, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))),
                (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), (0, 0, 0),
                thickness=3)
        cv.circle(input_slice, (int(round(keypoint.pointing_point[1])), int(round(keypoint.pointing_point[0]))), 7,
                  color=(128, 178, 194), thickness=-1)

    cv.imshow('OUR Marked keypoints', input_scene)
    cv.imshow('The marked Area', input_slice)
    """

def openCVImplementation(image,slice_start,slice_end):
    openCVpipeline.findObjectsInImage(image,slice_start,slice_end)


if __name__ == "__main__":
    print(f'~~~ STARTING TIMER ~~~')
    startTime = time.time()

    # fyrfadslys.jpg
    # y1 = 217
    # y2 = 274
    # x1 = 196
    # x2 = 250
    hvid_lys1 = [(217, 196), (274, 250)]
    hvid_lys2 = [(174, 328), (234, 387)]

    # candlelightsOnVaryingBackground.jpg
    # vary_lys = [(y1,x1),(y2,x2)]
    vary_lys1 = [(250, 544), (357, 649)]
    vary_lys2 = [(296, 393), (404, 497)]
    vary_lys3 = [(395, 579), (499, 687)]
    vary_lys4 = [(521, 409), (624, 508)]
    vary_lys5 = [(552, 70), (655, 174)]

    # coins:
    coin1 = [(205, 135), (296, 231)]

    input_scene = cv.imread('Images/candlelightsOnVaryingBackground.jpg')
    #openCVImplementation(input_scene69, vary_lys2[0], vary_lys2[1])
    main(input_scene, vary_lys2[0], vary_lys2[1])
    # testGuassian()
    # testMaxima()
    # testSift()
    # testMatchingOpenCV()
    #testMatching(input_scene69, vary_lys2[0], vary_lys2[1])
    #testMatchingOpenCV(vary_lys2[0], vary_lys2[1])
    # compareDescriptors()


    print(f'~~~ TIMER ENDED: TOTAL TIME = {time.time() - startTime} s ~~~')
    cv.waitKey(0)
    cv.destroyAllWindows()

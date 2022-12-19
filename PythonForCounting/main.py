import random

import cv2 as cv
import numpy as np
from Libraries import FeatureMatching as fm
from Libraries import SIFT
import time
import copy
import os
import matplotlib.pyplot as plt

def makeImagePyramid(starting_image, scale, min_width):
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


def windowSlider(image, window_size: tuple, step_size):
    """
    Returnere en slicegenerator, som genererer et slice for hvert step igennem et billede, looper man over
    generatoren kan man så lave image processing på hvert slice. :param image: Billedet man vil loope henover
    :param
    windowSize: størrelesen på slicet (y,x) :param stepSize: hvor stort et skridt man skal tage mellem hvert slice
    """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield y, x, image[y:y + window_size[0], x:x + window_size[1]]


def makeGrayscale(img):
    """
    Returnerer et grayscale image ud fra det som man har puttet ind i funktionen
    :param img:
    :return:
    """
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    output[:, :] = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    return output


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


def returnScoreAndImageWithOutlines(image, hits, nms_threshold=0.3):
    if not hits:
        return 0, image
    scores = []
    outlines = []
    for (dist, outline) in hits:
        scores.append(dist)
        outlines.append(outline)
    output_image = image.copy()
    hit_scores, hit_outlines = nonMaximumSuppression(outlines, nms_threshold, scores)
    for i, (start_x, end_x, start_y, end_y) in enumerate(hit_outlines):
        cv.rectangle(output_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        #cv.putText(outputImage, f'Score: {int(scores[i])}', (startx, starty), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    return len(hit_scores), output_image


def main(input_scene, input_file_name, slice_start, slice_end, scale_ratio=2, standard_deviation=1.6, color_hist_threshold=1000):
    output_scene = input_scene.copy()
    start_time = time.time()
    user_slice = output_scene[slice_start[0]:slice_end[0],slice_start[1]: slice_end[1]]
    slice_feature_vector = fm.calculateImageHistogramBinVector(user_slice, 16, 500)
    window_size = (user_slice.shape[0], user_slice.shape[1])

    greyscale_scene = makeGrayscale(input_scene.copy())
    keypoints_scene, keypoints_slice = computeKeypointsWithDescriptorsFromImage(greyscale_scene, slice_start, slice_end, scale_factor=scale_ratio, standard_deviation=standard_deviation)
    array_to_print = [f'Keypoints in scene overall: {len(keypoints_scene)}\n']
    array_to_print.append(f'Keypoints in slice (not validated): {len(keypoints_slice)}\n')

    validated_slice_keypoints = SIFT.validateKeypoints(keypoints_slice, keypoints_scene)
    array_to_print.append(f'validated keypoints from slice: {len(validated_slice_keypoints)} \n')
    best_keypoints_in_scene = SIFT.matchDescriptorsWithKeypointFromSlice(validated_slice_keypoints, keypoints_scene)
    array_to_print.append(f'total keypoints mathching validated slice keypoints in image: {sum(len(l)for l in best_keypoints_in_scene)}\n')
    image_pyramid = makeImagePyramid(input_scene, scale_ratio, window_size[1])
    # definere vinduestørrelsen, tænker den skulle laves ud fra inputbilledet
    # vores liste over hits
    hits = []

    stepsize = min(window_size)/5
    # looper over alle billeder i billedpyramiden, man behøver ikke at lave pyramiden først, den kan laves på samme linje hernede
    for i, image in enumerate(image_pyramid):
        # looper over alle vinduerne i billedet
        for (y, x, window) in windowSlider(image, window_size, int(stepsize)):
            # Vinduet kan godt blive lavet halvt uden for billedet, hvis dette ikke er ønsket kan vi skippe den
            # beregning i loopet men det er lige en diskussion vi skal have i gruppen
            #if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
             #   continue
            # Lav vores image processing her
            current_window_vector = fm.calculateImageHistogramBinVector(window, 16, 500)
            hist_dist = fm.calculateEuclidianDistance(slice_feature_vector, current_window_vector)
            keypoints_in_window = 0
            image_scale = ((1/scale_ratio)**i)
            for array_of_keypoint_matches in best_keypoints_in_scene:
                for keypoint in array_of_keypoint_matches:
                    cv.circle(output_scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 3, color=(255, 0, 0), thickness=-1)
                    key_y, key_x = keypoint.coordinates[0] * image_scale, keypoint.coordinates[1] * image_scale
                    if x <= key_x <= x+window_size[1] and y <= key_y <= y+window_size[0]:
                        keypoints_in_window += 1

            if len(validated_slice_keypoints) != 0:
                if keypoints_in_window >= ((0.25/image_scale)*len(validated_slice_keypoints)) and hist_dist < color_hist_threshold:
                    hits.append([hist_dist,
                             [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                              y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])
            else:
                if hist_dist < color_hist_threshold:
                    hits.append([hist_dist,
                             [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                              y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])
    direct = input_file_name.split(".")[0]
    parent_direct = r"TestOutput\Our\\"
    path = os.path.join(parent_direct, direct)
    os.mkdir(path)
    path_for_image = path+r"\\"+input_file_name
    path_for_slice = path+r"\\SliceUsed.jpg"
    path_for_write = path + r"\\scores.txt"
    score, done_image = returnScoreAndImageWithOutlines(output_scene, hits, 0.1)
    cv.imwrite(path_for_image,done_image)
    cv.imwrite(path_for_slice, user_slice)
    array_to_print.append(f'Counted objects: {score}\n')
    array_to_print.append(f'Computing time: {time.time() - start_time} s\n')
    writer_object = open(path_for_write,"w")
    for l in array_to_print:
        writer_object.write(l)
    writer_object.close()


def mainCV(input_scene, input_file_name, slice_start, slice_end, scale_ratio=2, standard_deviation=1.6, color_hist_threshold=1000):
    output_scene = input_scene.copy()
    start_time = time.time()
    user_slice = output_scene[slice_start[0]:slice_end[0],slice_start[1]: slice_end[1]]
    slice_feature_vector = fm.calculateImageHistogramBinVector(user_slice, 16, 500)
    window_size = (user_slice.shape[0], user_slice.shape[1])

    greyscale_scene = makeGrayscale(input_scene.copy())
    sift = cv.SIFT_create()
    input_scene_keypoints, input_scene_descriptors = sift.detectAndCompute(greyscale_scene, None)
    array_to_print = [f'Keypoints in scene overall: {len(input_scene_keypoints)} \n']
    user_slice_keypoints = []
    user_slice_descriptors = []
    for keypoint,descriptor in zip(input_scene_keypoints, input_scene_descriptors):
        if slice_start[0] <= keypoint.pt[1] <= slice_end[0] and slice_start[1] <= keypoint.pt[0] <= slice_end[1]:
            user_slice_keypoints.append(cv.KeyPoint(*keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave))
            user_slice_descriptors.append(copy.deepcopy(descriptor))
    array_to_print.append(f'Keypoints in slice (not validated): {len(user_slice_keypoints)}\n')
    validated_slice_keypoints, validated_slice_descriptors = SIFT.validateOpenCVKeypoints(user_slice_keypoints, user_slice_descriptors, input_scene_keypoints, input_scene_descriptors)
    array_to_print.append(f'validated keypoints from slice: {len(validated_slice_keypoints)} \n')

    best_keypoints_in_scene, best_descriptors_in_scene = SIFT.matchOpenCVDescriptorsWithKeypointFromSlice(validated_slice_keypoints, validated_slice_descriptors, input_scene_keypoints, input_scene_descriptors)
    array_to_print.append(f'total keypoints mathching validated slice keypoints in image: {sum(len(l)for l in best_keypoints_in_scene)}\n')
    image_pyramid = makeImagePyramid(input_scene, scale_ratio, window_size[1])
    # definere vinduestørrelsen, tænker den skulle laves ud fra inputbilledet
    # vores liste over hits
    hits = []
    stepsize = min(window_size) / 5
    # looper over alle billeder i billedpyramiden, man behøver ikke at lave pyramiden først, den kan laves på samme linje hernede
    for i, image in enumerate(image_pyramid):
        # looper over alle vinduerne i billedet
        for (y, x, window) in windowSlider(image, window_size, int(stepsize)):
            # Vinduet kan godt blive lavet halvt uden for billedet, hvis dette ikke er ønsket kan vi skippe den
            # beregning i loopet men det er lige en diskussion vi skal have i gruppen
            #if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
             #   continue
            # Lav vores image processing her
            current_window_vector = fm.calculateImageHistogramBinVector(window, 16, 500)
            hist_dist = fm.calculateEuclidianDistance(slice_feature_vector, current_window_vector)
            keypoints_in_window = 0
            image_scale = ((1/scale_ratio)**i)
            for array_of_keypoint_matches in best_keypoints_in_scene:
                for keypoint in array_of_keypoint_matches:
                    cv.circle(output_scene, (int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))), 3, color=(255, 0, 0), thickness=-1)
                    key_y, key_x = keypoint.pt[1] * image_scale, keypoint.pt[0] * image_scale
                    if x <= key_x <= x+window_size[1] and y <= key_y <= y+window_size[0]:
                        keypoints_in_window += 1

            if len(validated_slice_keypoints) != 0:
                if keypoints_in_window >= ((0.25/image_scale)*len(validated_slice_keypoints)) and hist_dist < color_hist_threshold:
                    hits.append([hist_dist,
                                 [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                                  y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])
            else:
                if hist_dist < color_hist_threshold:
                    hits.append([hist_dist,
                                 [x * (scale_ratio ** i), x * (scale_ratio ** i) + (window.shape[1] * (scale_ratio ** i)),
                                  y * (scale_ratio ** i), y * (scale_ratio ** i) + (window.shape[0] * (scale_ratio ** i))]])
    direct = input_file_name.split(".")[0]
    parent_direct = r"TestOutput\OpenCV\\"
    path = os.path.join(parent_direct, direct)
    os.mkdir(path)
    path_for_image = path+r"\\"+input_file_name
    path_for_slice = path+r"\\SliceUsed.jpg"
    path_for_write = path + r"\\scores.txt"
    score, done_image = returnScoreAndImageWithOutlines(output_scene, hits, 0.1)
    cv.imwrite(path_for_image, done_image)
    cv.imwrite(path_for_slice, user_slice)
    array_to_print.append(f'Counted objects: {score}\n')
    array_to_print.append(f'Computing time: {time.time() - start_time} s\n')
    writer_object = open(path_for_write,"w")
    for l in array_to_print:
        writer_object.write(l)
    writer_object.close()

"""
def testSift():
    # Picture
    input_scene = cv.imread('TestInput/fyrfadslys.jpg')
    input_scene_copy = input_scene.copy()
    greyscale_input = makeGrayscale(input_scene.copy())

    # Picture keypoints
    sift = cv.SIFT_create()
    input_picture_keypoints, input_picture_descriptors = sift.detectAndCompute(greyscale_input, None)
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
    cv.drawKeypoints(input_scene_copy,input_picture_keypoints,input_scene_copy)
    cv.imshow('SIFT Picture keypoints', input_scene_copy)
    # cv.imshow('sift',img)
"""


def computeKeypointsWithDescriptorsFromImage(greyscale_input_image, slice_start, slice_end, scale_factor=2.0, standard_deviation=1.6):
    keypoints = []
    keypoints_slice = []

    for p, image in enumerate(makeImagePyramid(greyscale_input_image.astype("float32"), scale_factor, 10)):
        #print(f'{p}: Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, standard_deviation, scale_factor, 5)

       # print(f'{p}: Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images, DoG, p, standard_deviation, scale_factor)
       # print(f'{p}: Creating feature descriptors ...')
      #  print(f'{p}: Checking for duplicate keypoints ...')

      #  print(f'{p}:\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints, keypoints)
      #  print(f'{p}:\t - new keypoints found in octave {p} : {len(sorted_keypoints)}\n')
        #SIFT.resizeKeypoints(sorted_keypoints,scale_factor)
        keypoints.extend(SIFT.makeKeypointDescriptors(sorted_keypoints, Gaussian_images))

    for keypoint in keypoints:
        if slice_start[0] <= keypoint.coordinates[0] <= slice_end[0] and slice_start[1] <= keypoint.coordinates[1] <= slice_end[1]:
            keypoints_slice.append(copy.deepcopy(keypoint))

    return keypoints, keypoints_slice


"""
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
"""
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
        keypoint.computeKeypointPointersInMarkedImage(marked_area_start, marked_area_end)

    matches = SIFT.matchDescriptorsWithKeypointFromSlice(slice_keypoints, scene_keypoints)
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

    return input_scene

"""
def testMatchingOpenCV(marked_area_start, marked_area_end):
    # Picture keypoints
    input_pic = cv.imread('TestInput/candlelightsOnVaryingBackground.jpg')
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

def openCVImplementation(image,slice_start,slice_end):
    openCVpipeline.findObjectsInImage(image,slice_start,slice_end)
"""


def ensureInputPictureIsCorrectSize(scene, slice, max_size=1000):
    if max(scene.shape[:2]) > max_size:
        ratio = max_size/max(scene.shape[:2])
        output_image = cv.resize(scene, (0, 0), fx=ratio, fy=ratio)
        start_y, start_x = slice[0]
        end_y,end_x = slice[1]
        output_slice = [(int(start_y*ratio), int(start_x*ratio)), (int(end_y*ratio), int(end_x*ratio))]
        return output_image, output_slice
    else:
        return scene, slice

def compareDescriptors(scene):
    greyscale_input_image = makeGrayscale(scene.copy())

    keypoints = []

    for p, image in enumerate(makeImagePyramid(greyscale_input_image.astype("float32"), 2, 10)):
        # print(f'{p}: Creating DoG array ...')
        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, 1.6, 2, 5)

        # print(f'{p}: Creating keypoints ...')
        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images, DoG, p, 1.6,
                                                               2)
        # print(f'{p}: Creating feature descriptors ...')
        #  print(f'{p}: Checking for duplicate keypoints ...')

        #  print(f'{p}:\t - keypoints found in octave {p} : {len(found_keypoints)}')
        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints, keypoints)
        #  print(f'{p}:\t - new keypoints found in octave {p} : {len(sorted_keypoints)}\n')
        # SIFT.resizeKeypoints(sorted_keypoints,scale_factor)
        keypoints.extend(SIFT.makeKeypointDescriptors(sorted_keypoints, Gaussian_images))

    sift = cv.SIFT_create()
    cv_keypoints, cv_descriptors = sift.detectAndCompute(greyscale_input_image, None)

    our_descriptors_for_plot = []
    for i, keypoint in enumerate(keypoints):
        cv.circle(scene, (int(round(keypoint.coordinates[1])), int(round(keypoint.coordinates[0]))), 5,
                  color=(255, 0, 0), thickness=-1)
        if 692 < keypoint.coordinates[0] < 694 and 31 < keypoint.coordinates[1] < 33:
            our_descriptors_for_plot.append(keypoint.descriptor)
            print(keypoint.orientation)
            print(keypoint.coordinates)
            print(keypoint.descriptor)
    cv_descriptors_for_plot =[]
    for cvk, cvd in zip(cv_keypoints,cv_descriptors):
        if 692 < cvk.pt[1] < 694 and 31 < cvk.pt[0] < 33:
            cv_descriptors_for_plot.append(cvd)
            print(cvk.angle)
            print(cvk.pt)
            print(cvd)
    fig = plt.figure()

    cv.drawKeypoints(scene,cv_keypoints,scene)
    plt.subplot(4, 1, 1)
    plt.stairs(our_descriptors_for_plot[0])

    plt.subplot(4, 1, 2)
    plt.stairs(cv_descriptors_for_plot[0])


    plt.show()
    cv.imshow('scene',scene)

if __name__ == "__main__":
    print(f'~~~ STARTING TIMER ~~~')
    startTime = time.time()
    # input_scene = cv.imread("TestInput/candlelightsOnVaryingBackground.jpg")
    # greyscaleimage = makeGrayscale(input_scene.copy())
    # keypoints =[]
    # for p, image in enumerate(makeImagePyramid(greyscaleimage.astype("float32"), 2, 10)):
    #     blurred, DoG = SIFT.differenceOfGaussian(image,1.6,2)
    #     keypoints.extend(SIFT.defineKeyPointsFromPixelExtrema(blurred, DoG, p, 1.6,
    #                                                            2))
    # print(len(keypoints))
    input_directory = r"TestInput"
    input_images = []
    input_names = []
    input_slices =[[(581,477),(642,950)],[(296, 393), (404, 497)], [(234, 285), (328, 384)],[(317,827),(640,1006)],[(589,379),(864,578)],[(329,497),(405,577)], [(174, 328), (234, 387)],[(505,441),(580,524)],[(271,305),(297,328)],[(698,348),(735,378)],[(224,310),(311,366)],[(390,216),(474,355)],[(195,227),(344,396)],[(650,297),(677,328)],[(238,214),(275,260)],[(435,530),(527,872)],[(554,26),(712,120)],[(494,176),(558,339)],[(480,340),(540,407)],[(280,597),(636,790)],[(416,202),(559,344)],[(402,323),(479,450)],[(257,521),(552,700)],[(260,550),(525,874)],[(397,278),(422,311)],[(322,346),(392,420)],[(497,272),(589,333)],[(451,616),(659,856)],[(295,196),(381,280)],[(371,178),(509,323)],[(335,93),(522,282)], [(953, 1089), (1426, 1410)],[(421,742),(650,878)],[(336,290),(583,317)],[(1342,1831),(1627,2127)],[(562,2518),(845,2821)],[(1360,1681),(1951,2294)]]
    image = cv.imread("TestInput/candlelightsOnVaryingBackground.jpg")
    compareDescriptors(image)
    # images =[]
    # for file in os.listdir(input_directory):
    #     if file.endswith(".jpg"):
    #         input_images.append(cv.imread(input_directory + r"\\" + file))
    #         input_names.append(file)
    #
    # for i, (in_image, in_slice) in enumerate(zip(input_images, input_slices)):
    #     input_images[i], input_slices[i] = ensureInputPictureIsCorrectSize(in_image, in_slice)
    #
    #
    # cv.imshow("dik", output)
    # for i, (input_scene, input_name, input_slice) in enumerate(zip(input_images, input_names, input_slices)):
    #     jule_hygge = ['🎅', '🎄', '🤶', '💝', '🎇']
    #     random_index = random.randrange(0,len(jule_hygge))
    #     print(f'Computing image: {i+1} of {len(input_images)} {jule_hygge[random_index]}')
    #     images.append(testMatching(input_scene,input_slice[0],input_slice[1]))
    #     main(input_scene, input_name, input_slice[0], input_slice[1], color_hist_threshold=950)
    #     mainCV(input_scene, input_name, input_slice[0], input_slice[1], color_hist_threshold=950)
    # for i, image in enumerate(images):
    #     cv.imshow(f'{i}', image)
    print(f'~~~ TIMER ENDED: TOTAL TIME = {time.time() - startTime} s ~~~')
    cv.waitKey(0)
    cv.destroyAllWindows()

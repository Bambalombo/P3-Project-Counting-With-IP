import random

import cv2 as cv
import numpy as np
from Libraries import FeatureMatching as fm
from Libraries import SIFT
import time
import copy
import os


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



def computeKeypointsWithDescriptorsFromImage(greyscale_input_image, slice_start, slice_end, scale_factor=2.0, standard_deviation=1.6):
    keypoints = []
    keypoints_slice = []

    for p, image in enumerate(makeImagePyramid(greyscale_input_image.astype("float32"), scale_factor, 10)):

        Gaussian_images, DoG = SIFT.differenceOfGaussian(image, standard_deviation, scale_factor, 5)

        found_keypoints = SIFT.defineKeyPointsFromPixelExtrema(Gaussian_images, DoG, p, standard_deviation, scale_factor)

        sorted_keypoints = SIFT.checkForDuplicateKeypoints(found_keypoints, keypoints)

        keypoints.extend(SIFT.makeKeypointDescriptors(sorted_keypoints, Gaussian_images))

    for keypoint in keypoints:
        if slice_start[0] <= keypoint.coordinates[0] <= slice_end[0] and slice_start[1] <= keypoint.coordinates[1] <= slice_end[1]:
            keypoints_slice.append(copy.deepcopy(keypoint))

    return keypoints, keypoints_slice




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


if __name__ == "__main__":
    print(f'~~~ STARTING TIMER ~~~')
    startTime = time.time()
    parent_dir = r"TestOutput\\"
    os.mkdir(parent_dir)
    child_dir1 = r"TestOutput\OpenCV\\"
    child_dir2 = r"TestOutput\Our\\"
    os.mkdir(child_dir1)
    os.mkdir(child_dir2)
    input_directory = r"TestInput"
    input_images = []
    input_names = []
    input_slices =[[(581,477),(642,950)],[(296, 393), (404, 497)], [(234, 285), (328, 384)],[(317,827),(640,1006)],[(589,379),(864,578)],[(329,497),(405,577)], [(174, 328), (234, 387)],[(505,441),(580,524)],[(271,305),(297,328)],[(698,348),(735,378)],[(224,310),(311,366)],[(390,216),(474,355)],[(195,227),(344,396)],[(650,297),(677,328)],[(238,214),(275,260)],[(435,530),(527,872)],[(554,26),(712,120)],[(494,176),(558,339)],[(480,340),(540,407)],[(280,597),(636,790)],[(416,202),(559,344)],[(402,323),(479,450)],[(257,521),(552,700)],[(260,550),(525,874)],[(397,278),(422,311)],[(322,346),(392,420)],[(497,272),(589,333)],[(451,616),(659,856)],[(295,196),(381,280)],[(371,178),(509,323)],[(335,93),(522,282)], [(953, 1089), (1426, 1410)],[(421,742),(650,878)],[(336,290),(583,317)],[(1342,1831),(1627,2127)],[(562,2518),(845,2821)],[(1360,1681),(1951,2294)]]

    for file in os.listdir(input_directory):
        if file.endswith(".jpg"):
            input_images.append(cv.imread(input_directory + r"\\" + file))
            input_names.append(file)

    for i, (in_image, in_slice) in enumerate(zip(input_images, input_slices)):
        input_images[i], input_slices[i] = ensureInputPictureIsCorrectSize(in_image, in_slice)

    for i, (input_scene, input_name, input_slice) in enumerate(zip(input_images, input_names, input_slices)):
        jule_hygge = ['🎅', '🎄', '🤶', '💝', '🎇']
        random_index = random.randrange(0,len(jule_hygge))
        print(f'Computing image: {i+1} of {len(input_images)} {jule_hygge[random_index]}')
        main(input_scene, input_name, input_slice[0], input_slice[1], color_hist_threshold=950)
        mainCV(input_scene, input_name, input_slice[0], input_slice[1], color_hist_threshold=950)

    print(f'~~~ TIMER ENDED: TOTAL TIME = {time.time() - startTime} s ~~~')

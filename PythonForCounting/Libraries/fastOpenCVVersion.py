import cv2 as cv
import numpy as np
from cv2 import KeyPoint
from . import SIFT


def findObjectsInImage(input_scene, marked_area_start, marked_area_end):
    # GRAYSCALE SCENE IMAGE
    input_scene = input_scene.copy()
    greyscale_scene = cv.cvtColor(input_scene, cv.COLOR_BGR2GRAY)

    # CREATE SIFT OBJECT
    sift = cv.SIFT_create()

    # GENERATE SCENE KEYPOINTS, DESCRIPTORS
    scene_keypoints, scene_descriptors = sift.detectAndCompute(greyscale_scene, None)

    # DEFINE SLICE, ADD BORDERS, MAKE GREYSCALE
    input_slice = expandMarkedAreaOpenCV(marked_area_start, marked_area_end, input_scene)
    greyscale_slice = cv.cvtColor(input_slice, cv.COLOR_BGR2GRAY)

    # FIND SLICE KEYPOINTS
    expanded_slice_keypoints, expanded_slice_descriptors = sift.detectAndCompute(greyscale_slice, None)

    # REMOVE KEYPOINTS OUTSIDE MARKED AREA
    slice_keypoints, slice_descriptors = discardKeypointsOutsideMarkedAreaOpenCV(expanded_slice_descriptors, expanded_slice_keypoints, marked_area_start, marked_area_end)

    # CALCULATE POINTERS FOR KEYPOINTS
    slice_pointers = []
    for keypoint in slice_keypoints:
        x, y = keypoint.pt

        center_y = int((marked_area_end[0] - marked_area_start[0])) + int((marked_area_end[0] - marked_area_start[0]) / 2)
        center_x = int((marked_area_end[1] - marked_area_start[1]))+ int((marked_area_end[1] - marked_area_start[1]) / 2)

        #print(keypoint.size)

        pointing_point = (center_y, center_x)
        pointing_length = np.sqrt((center_y - y) ** 2 + (center_x - x) ** 2) / keypoint.size
        pointing_angle = (np.rad2deg(np.arctan2(center_y - y, center_x - x)) + keypoint.angle) % 360

        slice_pointers.append([pointing_point, pointing_length, pointing_angle])

    scene_best_matching_keypoints, scene_matching_descriptors = matchDescriptorsOpenCV(slice_keypoints, slice_descriptors, scene_keypoints, scene_descriptors)

    # CALCULATE POINTERS FOR REMAINING SCENE KEYPOINTS
    best_scene_pointers = []
    for match_index, keypoint_matches in enumerate(scene_best_matching_keypoints):
        for keypoint in keypoint_matches:
            x, y = (keypoint.pt[0], keypoint.pt[1])

            pointing_length = slice_pointers[match_index][1] * keypoint.size
            pointing_angle = (slice_pointers[match_index][2] - keypoint.angle) % 360
            pointing_point = ((pointing_length * np.sin(np.deg2rad(pointing_angle))) + keypoint.pt[1],
                (pointing_length * np.cos(np.deg2rad(pointing_angle))) + keypoint.pt[0])

            best_scene_pointers.append([pointing_point, pointing_length, pointing_angle])

    best_keypoints_list = []

    for match_index, keypoint_matches in enumerate(scene_best_matching_keypoints):
        for keypoint in keypoint_matches:
            best_keypoints_list.append(keypoint)

    # DRAW SORTED KEYPOINTS
    output_image1 = input_scene.copy()
    cv.drawKeypoints(input_scene, best_keypoints_list, output_image1)

    # DRAW UNSORTED KEYPOINTS
    output_image2 = input_scene.copy()
    cv.drawKeypoints(input_scene, scene_keypoints, output_image2)

    print(len(best_keypoints_list), len(best_scene_pointers))

    cv.drawKeypoints(input_slice,slice_keypoints,input_slice)
    cv.drawKeypoints(input_scene,best_keypoints_list,input_scene)

    #for keypoint, (pointing_point, pointing_length, pointing_angle) in zip(best_keypoints_list,best_scene_pointers):
        #cv.circle(output_image1, (int((keypoint.pt[0])), int((keypoint.pt[1]))), 5, color=(255, 0, 0), thickness=-1)
        #cv.line(output_image1, (int((keypoint.pt[0])), int((keypoint.pt[1]))), (int((pointing_point[1])), int((pointing_point[0]))), (0, 0, 0), thickness=3)
        #cv.circle(output_image1, (int((pointing_point[1])), int((pointing_point[0]))), 7, color=(128, 178, 194), thickness=-1

    #for keypoint, (pointing_point, pointing_length, pointing_angle) in zip(slice_keypoints,slice_pointers):
        #cv.circle(input_slice, (int((keypoint.pt[0])), int((keypoint.pt[1]))), 5, color=(255, 0, 0), thickness=-1)
        # cv.line(input_slice, (int((keypoint.pt[0])), int((keypoint.pt[1]))), (int((pointing_point[1])), int((pointing_point[0]))), (0, 0, 0), thickness=3)
        # cv.circle(input_slice, (int((pointing_point[1])), int((pointing_point[0]))), 7, color=(128, 178, 194), thickness=-1)

    cv.rectangle(output_image1,(marked_area_start[1],marked_area_start[0]),(marked_area_end[1],marked_area_end[0]),(255,255,0),thickness=2)

    cv.imshow('Slice area',input_slice)
    cv.imshow('grayscale', greyscale_scene)
    cv.imshow('Best keypoints', output_image1)
    #cv.imshow('All keypoints', output_image2)


def matchDescriptorsOpenCV(slice_keypoints: [KeyPoint], slice_descriptors, scene_keypoints: [KeyPoint], scene_descriptors, distance_ratio_treshold=1.5):
    best_matching_keypoints = [[] for _ in range(len(slice_keypoints))]
    best_matching_descriptors = [[] for _ in range(len(slice_keypoints))]
    best_matching_dist = [[] for _ in range(len(slice_keypoints))]

    for scene_keypoint, scene_descriptor in zip(scene_keypoints, scene_descriptors):
        dist_list = []
        for slice_keypoint, slice_descriptor in zip(slice_keypoints, slice_descriptors):
            #dist = np.dot(slice_descriptor, scene_descriptor)
            dist = np.linalg.norm(slice_descriptor - scene_descriptor)
            dist_list.append(dist)
        best_matching_keypoints[dist_list.index((min(dist_list)))].append(scene_keypoint)
        best_matching_descriptors[dist_list.index((min(dist_list)))].append(scene_descriptor)
        best_matching_dist[dist_list.index((min(dist_list)))].append(min(dist_list))

    output_match_keypoints = []
    output_match_descriptors = []

    for slice_keypoint_dists, slice_keypoint_matches, slice_descriptor_matches in zip(best_matching_dist, best_matching_keypoints, best_matching_descriptors):
        if len(slice_keypoint_dists) == 0:
            continue
        min_value = min(slice_keypoint_dists)
        indices = np.where(slice_keypoint_dists >= min_value * distance_ratio_treshold)[0]
        output_match_keypoints.append(np.array(slice_keypoint_matches)[indices])
        output_match_descriptors.append(np.array(slice_descriptor_matches)[indices])

    return output_match_keypoints, output_match_descriptors


def expandMarkedAreaOpenCV(starting_coordinates, end_coordinates, input_picture):
    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    if 0 <= starting_coordinates[0] - height and end_coordinates[0] + height < input_picture.shape[0] \
            and 0 <= starting_coordinates[1] - width and end_coordinates[1] + width < input_picture.shape[1]:
        return input_picture[starting_coordinates[0] - height: end_coordinates[0] + height + 1,
               starting_coordinates[1] - width: end_coordinates[1] + width + 1].copy()
    else:
        return input_picture[starting_coordinates[0]:end_coordinates[0],
               starting_coordinates[1]:end_coordinates[1]].copy()


def discardKeypointsOutsideMarkedAreaOpenCV(descriptors, keypoints, starting_coordinates, end_coordinates):
    new_keypoints = []
    new_descriptors = []

    height = end_coordinates[0] - starting_coordinates[0]
    width = end_coordinates[1] - starting_coordinates[1]
    for keypoint, descriptor in zip(keypoints, descriptors):
        if height < keypoint.pt[0] < height * 2 and width < keypoint.pt[1] < width * 2:
            new_keypoints.append(keypoint)
            new_descriptors.append(descriptor)

    return new_keypoints, new_descriptors

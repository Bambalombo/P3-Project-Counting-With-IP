import numpy as np
class KeyPoint:
    def __init__(self, coordinates: tuple, strength=0., octave=0, scale_space=0, image_scale=0., size_sigma=0., orientation=0, descriptor=[], pointing_angle=0, pointing_length=0, pointing_point=[0,0]):
        self.coordinates = coordinates
        self.strength = strength
        self.octave = octave
        self.scale_space = scale_space
        self.image_scale = image_scale
        self.size_sigma = size_sigma
        self.orientation = orientation
        self.descriptor = descriptor
        self.pointing_angle = pointing_angle
        self.pointing_length = pointing_length
        self.pointing_point = pointing_point

    def __repr__(self):
        return f'Keypoint: \n Coordinates = {self.coordinates} \n Strenght = {self.strength} \n Octave = {self.octave} \n Scale Space = {self.scale_space} \n Image Scale = {self.image_scale} \n Size (sigma) = {self.size_sigma} \n Orientation = {self.orientation}, \n Pointing angle = {self.pointing_angle}, \n Pointing length = {self.pointing_length}, \n Pointing point = {self.pointing_point}'
    def computeKeypointPointersInMarkedImage(self, starting_coordinates, end_coordinates):
        center_y = end_coordinates[0] - int(starting_coordinates[0] / 2)
        center_x = end_coordinates[1] - int(starting_coordinates[1] / 2)
        self.pointing_point = (center_y,center_x)
        self.pointing_length = np.sqrt((self.coordinates[0]-center_y)**2 + (self.coordinates[1]-center_x)**2) / self.octave
        self.pointing_angle = np.rad2deg(np.arctan2(center_y-self.coordinates[0], center_x-self.coordinates[1])) - self.orientation

    def computeKeypointPointersFromMatchingKeypoint(self, keypoint):
       self.pointing_length = keypoint.pointing_length * self.octave
       self.pointing_angle = keypoint.pointing_angle + self.orientation
       self.pointing_point = [self.pointing_length * np.sin(np.deg2rad(self.pointing_angle)) + self.coordinates[0], self.pointing_length * np.cos(np.deg2rad(self.pointing_angle)) + self.coordinates[1]]
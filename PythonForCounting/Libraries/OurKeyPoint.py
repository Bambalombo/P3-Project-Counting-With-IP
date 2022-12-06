class KeyPoint:
    def __init__(self, coordinates: tuple, strength=0., octave=0, scale_space=0, image_scale=0., size_sigma=0., orientation=0, descriptor = []):
        self.coordinates = coordinates
        self.strength = strength
        self.octave = octave
        self.scale_space = scale_space
        self.image_scale = image_scale
        self.size_sigma = size_sigma
        self.orientation = orientation
        self.descriptor = descriptor
    def __repr__(self):
        return f'Keypoint: \n Coordinates = {self.coordinates} \n Strenght = {self.strength} \n Octave = {self.octave} \n Scale Space = {self.scale_space} \n Image Scale = {self.image_scale} \n Size (sigma) = {self.size_sigma} \n Orientation = {self.orientation}'
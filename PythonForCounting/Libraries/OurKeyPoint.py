class KeyPoint:
    def __init__(self, koordinates: tuple, strength=0., octave=0, scale_space=0, image_scale=0., size_radius=0., orientation=0):
        self.koordinates = koordinates
        self.strength = strength
        self.octave = octave
        self.scaleSpace = scale_space
        self.image_scale = image_scale
        self.size_radius = size_radius
        self.orientation = orientation

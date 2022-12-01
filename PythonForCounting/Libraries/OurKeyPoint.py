class KeyPoint:
    def __init__(self, koordinates: tuple, strength, octave, scaleSpace, scale, orientation, sigma):
        self.koordinates = koordinates
        self.strength = strength
        self.octave = octave
        self.scaleSpace = scaleSpace
        self.scale = scale
        self.orientation = orientation
        self.sigma = sigma

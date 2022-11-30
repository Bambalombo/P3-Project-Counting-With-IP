import math

import cv2 as cv
import numpy as np
from . import bordering as bd


def makeGuassianKernel(SD):
    kernelsize = int(math.ceil(6 * SD)) // 2 * 2 + 1
    radius = int((kernelsize - 1) / 2)  # kernel radius
    guassian = [1 / (math.sqrt(2 * math.pi) * SD) * math.exp(-0.5 * (x / SD) ** 2) for x in range(-radius, radius + 1)]
    guassianKernel = np.zeros((kernelsize, kernelsize))

    for y in range(guassianKernel.shape[0]):
        for x in range(guassianKernel.shape[1]):
            guassianKernel[y, x] = guassian[y] * guassian[x]
    guassianKernel *= 1 / guassianKernel[0, 0]
    return guassianKernel


def convolve(image, kernel):
    kernelSize = kernel.shape[0]
    borderimage = bd.addborder_reflect(image, kernelSize)
    sumKernel = kernel / np.sum(kernel)
    if len(borderimage.shape) == 3:
        output = np.zeros(
            (borderimage.shape[0] - kernelSize + 1, borderimage.shape[1] - kernelSize + 1, borderimage.shape[2]))
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                for channel in range(output.shape[2]):
                    slice = borderimage[y: y + kernelSize, x: x + kernelSize, channel]
                    output[y, x, channel] = np.sum(slice * sumKernel)
        return output
    else:
        output = np.zeros((borderimage.shape[0] - kernelSize + 1, borderimage.shape[1] - kernelSize + 1))
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                slice = borderimage[y: y + kernelSize, x: x + kernelSize]
                output[y, x] = np.sum(slice * sumKernel)
        return output


def differenceOfGaussian(image, SD, octave, numberOfDoGs=5):
    gaussianKernel = makeGuassianKernel(SD * octave)
    borderimage = bd.addborder_reflect(image, gaussianKernel.shape[0])
    blurredPictures = [convolve(borderimage, gaussianKernel)]
    # blurredPictures = [cv.GaussianBlur(image,(0,0),sigmaX=SD*octave,sigmaY=SD*octave)]
    k = (octave * 2) ** (1. / (numberOfDoGs - 2))
    for i in range(1, numberOfDoGs + 1):
        guassiankernel = makeGuassianKernel(SD * (k ** i))
        blurredPictures.append(convolve(borderimage, guassiankernel))
        # blurredPictures.append(cv.GaussianBlur(image,(0,0),sigmaX=(SD * (k**i)),sigmaY=(SD * (k**i))))

    DoG = []
    for (bottomPicture, topPicture) in zip(blurredPictures, blurredPictures[1:]):
        DoG.append(cv.subtract(topPicture, bottomPicture))
    return DoG


def defineKeyPointsFromPixelExtrema(DoG_array):
    """
    Vi vil finde ekstrema (vores keypoints). Processen overordnet:
    - Hvert scalespace består af 3 DoG billeder.
    - Vi looper igennem alle pixels og laver en 3x3x3 cube fra midterste billedes pixels.
    - For hver cube finder vi ud af om den midterste pixel er et extremum (minimum eller maximum værdien i cuben).
    """
    def centerPixelIsExtrema(pixel_cube):
        """
        METODE-DEFINITION: KALDES NEDENUNDER
        -----
        Her gemmer vi værdien for center pixelen og så ændrer vi efterfølgende centerpixelens værdi i cuben til 0, så
        den efterfølgende ikke tælles med i tjekket for om den er et extremum. Så kan vi nemlig bruge .all()
        """
        center_pixel = image_mid[y, x].copy()
        pixel_cube[1, 1, 1] = 0  # set center pixel to 0
        # Vi tjekker her om center_pixel er større end alle andre pixels
        # --
        if (center_pixel > pixel_cube).all() or (center_pixel < pixel_cube).all():
            return True
        return False

    def specifyExtremumLocation(y,x,octave_images,pixel_cube):
        """
        METODE-DEFINITION: KALDES NEDENUNDER
        -----
        Metode for at beregne den præcise placering af extremum i en 3x3x3 cube af pixels.
        """
        # Vi pakker billederne ud som vi passerede fra før
        image_top, image_mid, image_bot = octave_images

        # Vi definerer et for-loop, der sætter et max for hvor mange gange vi max vil forsøge at tilnærme os placeringen
        for attemt in range(5):
            pixel_cube = np.array([image_top[y-1:y+2, x-1:x+2],
                                   image_mid[y-1:y+2, x-1:x+2],
                                   image_bot[y-1:y+2, x-1:x+2]]).astype('float32') / 255.0




    # Vi opretter et array til at holde vores extrema pixels til senerehen
    # --
    keypoints = []

    # Vi vil gerne loope igennem hver oktav (tre sammenhængende DoG billeder)
    # -- Her zipper vi elementer sammen fra den passerede DoG array, således at index 0 bliver grupperet med 1 og 2:
    # -- (0,1,2). Index 1 bliver ligeledes sat sammen med (1,2,3) og så får vi til sidst (2,3,4). Det svarer til de DoG
    # -- billeder der udgør vores tre oktaver.
    for scale_space_index, (image_top, image_mid, image_bot) in enumerate(zip(DoG_array[:],
                                                                              DoG_array[1:],
                                                                              DoG_array[2:])):
        # Her looper vi nu over alle pixels men starter ved pixel 1 og til den andensidste pixel, da vi kræver at have
        # -- pixels hele vejen rundt om vores center-pixel. Det gælder ikke ved kanterne så derfor starter vi ved [1,1]
        for y in range(1, image_mid.shape[0] - 1):
            for x in range(1, image_mid.shape[1] - 1):
                # Her vil vi udføre et tjek om den givne pixel er et extremum i vores oktav. Altså vi vil tjekke om
                # -- denne pixel er ENTEN den største ELLER den mindste i sin pixel-cube. Først opretter vi cuben. Det
                # -- gøres ved at slice de tre billeder som udgør den nuværende oktav på en måde så y,x er vores midte.
                current_octave = (image_top,image_mid,image_bot)
                current_pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                                               image_mid[y - 1:y + 2, x - 1:x + 2],
                                               image_bot[y - 1:y + 2, x - 1:x + 2]])
                # Hvis pixelen er et extremum skal vi finde ud af extremumets præcice placering. Dette er fordi at den
                # -- i teorien kan lægge imellem pixels, forskudt af alle akser. Det er der noget matematik der siger.
                # -- Vi skal bare vide at afhængig af de andre pixels værdier i cuben, så er det IKKE sikkert at selve
                # -- toppunktet vi netop har fundet, ligger indenfor den samme pixel celle.
                if centerPixelIsExtrema(current_pixel_cube):
                    result = specifyExtremumLocation(y,x,current_octave,current_pixel_cube)


def calculateGradient():
    pass

def calculateHovedKrumning():
    pass
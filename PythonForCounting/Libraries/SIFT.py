import math
from .OurKeyPoint import KeyPoint
import cv2 as cv
import numpy as np
from . import bordering as bd


def makeGaussianKernel(SD):
    kernelsize = int(math.ceil(6 * SD)) // 2 * 2 + 1
    radius = int((kernelsize - 1) / 2)  # kernel radius
    gaussian = [1 / (math.sqrt(2 * math.pi) * SD) * math.exp(-0.5 * (x / SD) ** 2) for x in range(-radius, radius + 1)]
    gaussianKernel = np.zeros((kernelsize, kernelsize))

    for y in range(gaussianKernel.shape[0]):
        for x in range(gaussianKernel.shape[1]):
            gaussianKernel[y, x] = gaussian[y] * gaussian[x]
    gaussianKernel *= 1 / gaussianKernel[0, 0]
    return gaussianKernel


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
    gaussianKernel = makeGaussianKernel(SD * octave)
    borderimage = bd.addborder_reflect(image, gaussianKernel.shape[0])
    blurredPictures = [convolve(borderimage, gaussianKernel)]
    # blurredPictures = [cv.GaussianBlur(image,(0,0),sigmaX=SD*octave,sigmaY=SD*octave)]
    k = (octave * 2) ** (1. / (numberOfDoGs - 2))
    for i in range(1, numberOfDoGs + 1):
        guassiankernel = makeGaussianKernel(SD * (k ** i))
        blurredPictures.append(convolve(borderimage, gaussianKernel))
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

    def specifyExtremumLocation(y,x,current_scale_space,pixel_cube,current_DoG_stack,image_index, number_of_attempts = 5):
        """
        Metode for at beregne den præcise placering af extremum i en 3x3x3 cube af pixels.
        """
        def calculateGradient(pixel_cube):
            """
            Den her metode er til for at beregne gradient (retningen for en border i et billede).
            -----
            Gradient beregnes ved følgende formel (husk vi skal helst bare forstå hvorfor vi bruger matematikken og ikke
            nødvendigvis hvordan den virker):
            f'(x) = (f(x + s) - f(x - s)) / (2 * s)
            I formlen står x for koordinat og s for step, altså størrelsen af skridtet mellem hver værdi. Da vi her har
            at gøre med pixels i et grid er s (afstanden mellem hver pixel) lig med 1, så vi kan i stedet skrive:
            f'(x) = (f(x + 1) - f(x - 1)) / 2.
            Her tager vi udgangspunkt i centerpixelen. Vi husker at pixel_cube har holder pixel_cube[top,mid,bot], hvor
            top, mid og bot hver er et 3x3 slice af et billede.
            """

            # Først regner vi x. Vi vil gerne finde forskellen over x-aksen. Derfor skriver vi 1 i første indgang (altså
            # -- midterste billede), med 1 i anden indgang (i midten af y aksen). Og så tager vi x=2 og x=0 og trækker
            # -- fra hinanden. Så
            dx = (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0]) / 2

            # Samme for y. Første indgag (top/mid/bot) er 1 igen. Nu er det y ændres mellem 2 og 0, og x er 1 konstant.
            dy = (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1]) / 2

            # For den sidste er det forskellen over de tre billeder i laget, så her ændres første indgang mellem 0 og 2.
            # -- y og x er konstante.
            ds = (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1]) / 2

            return np.array([dx, dy, ds])

        def calculateHessian(pixel_cube):
            """
            Hovedkrumning
            """
            center_pixel_value = pixel_cube[1, 1, 1]

            dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
            dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
            dss = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]

            dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
            dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
            dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])

            return np.array([[dxx, dxy, dxs],[dxy, dyy, dys],[dxs, dys, dss]])

        ### --- specifyExtremumLocation --- ###

        # Vi pakker billederne ud som vi passerede fra før
        image_top, image_mid, image_bot = current_scale_space
        print(image_top, image_mid, image_bot)

        # Vi definerer et for-loop, der sætter et max for hvor mange gange, vi vil forsøge at tilnærme os placeringen
        for attemt in range(number_of_attempts):
            pixel_cube = np.array([image_top[y-1:y+2, x-1:x+2],
                                   image_mid[y-1:y+2, x-1:x+2],
                                   image_bot[y-1:y+2, x-1:x+2]]).astype('float32') / 255.0
            # Gradient beregnes. Læs metode for uddybelse.
            gradient = calculateGradient(pixel_cube)
            # Hessian beregnes. Læs metode for uddybelse.
            hessian = calculateHessian(pixel_cube)
            #
            offset = -np.lstsq(hessian,gradient, rcond=None)[0]

            if all(abs(offset) < 0.5):
                break
            y += int(round(offset[0]))
            x += int(round(offset[1]))
            image_index = int(round(offset[2]))
            if y < 3 or y > image_mid.shape[0] - 3 or x < 3 or x > image_mid.shape[1] - 3 or image_index < 1 or image_index > len(current_DoG_stack)-2:
                # Det beregnede punkt er endten for tæt på kanten, eller uden for billedet, derfor er keypointet her ikke stabilt
                return None
            if attemt >= number_of_attempts-1:
                return None
            image_top, image_mid, image_bot = current_DoG_stack[image_index-1: image_index+2]
            print(image_top,image_mid,image_bot)
        maximum_strength = image_mid[1,1] + (0.5 * np.dot(gradient, offset))

    ### --- defineKeyPointsFromPixelExtrema --- ###

    # Vi opretter et array til at holde vores extrema pixels til senere hen https://cdn.discordapp.com/attachments/938032088756674652/1044169017155387511/image.png

    keypoints = []

    # Vi vil gerne loope igennem hver oktav (tre sammenhængende DoG billeder)
    # -- Her zipper vi elementer sammen fra det passerede DoG array, således at index 0 bliver grupperet med 1 og 2:
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
                current_scale_space_DoG_images = (image_top,image_mid,image_bot)
                current_pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                                               image_mid[y - 1:y + 2, x - 1:x + 2],
                                               image_bot[y - 1:y + 2, x - 1:x + 2]])
                # Hvis pixelen er et extremum skal vi finde ud af extremumets præcice placering. Dette er fordi at den
                # -- i teorien kan lægge imellem pixels, forskudt af alle akser. Det er der noget matematik der siger.
                # -- Vi skal bare vide at afhængig af de andre pixels værdier i cuben, så er det IKKE sikkert at selve
                # -- toppunktet vi netop har fundet, ligger indenfor den samme pixel celle.
                if centerPixelIsExtrema(current_pixel_cube):
                    result = specifyExtremumLocation(y,x,current_scale_space_DoG_images,current_pixel_cube, DoG_array,image_index=(scale_space_index+1) )

import math

from matplotlib import pyplot as plt

from .OurKeyPoint import KeyPoint
import cv2 as cv
import numpy as np
from . import bordering as bd


def makeGaussianKernel(SD, image):
    kernelsize = int(math.ceil(6 * SD)) // 2 * 2 + 1
    if kernelsize > image.shape[0] or kernelsize > image.shape[1]:
        kernelsize = min(image.shape) - 1
        if kernelsize % 2 == 0:
            kernelsize -= 1

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


def differenceOfGaussian(image, SD, scale_ratio, numberOfDoGs=5):
    # gaussianKernel = makeGaussianKernel(SD, image)
    # borderimage = bd.addborder_reflect(image, gaussianKernel.shape[0])
    # blurredPictures = [convolve(borderimage, gaussianKernel)]
    blurredPictures = [cv.GaussianBlur(image, (0, 0), sigmaX=SD, sigmaY=SD)]
    k = scale_ratio ** (1. / (numberOfDoGs - 2))
    for i in range(1, numberOfDoGs + 1):
        # guassiankernel = makeGaussianKernel(SD * (k ** i), image)
        # blurredPictures.append(convolve(borderimage, gaussianKernel))
        blurred_image = cv.GaussianBlur(image, (0, 0), sigmaX=(SD * (k ** i)), sigmaY=(SD * (k ** i)))
        blurredPictures.append(blurred_image)

    DoG = []
    for (bottomPicture, topPicture) in zip(blurredPictures, blurredPictures[1:]):
        DoG.append(cv.subtract(topPicture, bottomPicture))

    return blurredPictures, DoG


def defineKeyPointsFromPixelExtrema(gaussian_images, DoG_array, octave_index, SD, scale_ratio):
    """
    Vi vil finde ekstrema (vores keypoints). Processen overordnet:
    - Hvert scalespace består af 3 DoG billeder.
    - Vi looper igennem alle pixels og laver en 3x3x3 cube fra midterste billedes pixels.
    - For hver cube finder vi ud af om den midterste pixel er et extremum (minimum eller maximum værdien i cuben).
    """
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
                current_scale_space_DoG_images = (image_top, image_mid, image_bot)
                current_pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                                               image_mid[y - 1:y + 2, x - 1:x + 2],
                                               image_bot[y - 1:y + 2, x - 1:x + 2]])
                # Hvis pixelen er et extremum skal vi finde ud af extremumets præcice placering. Dette er fordi at den
                # -- i teorien kan lægge imellem pixels, forskudt af alle akser. Det er der noget matematik der siger.
                # -- Vi skal bare vide at afhængig af de andre pixels værdier i cuben, så er det IKKE sikkert at selve
                # -- toppunktet vi netop har fundet, ligger indenfor den samme pixel celle.
                if centerPixelIsExtrema(current_pixel_cube, image_mid, y, x):
                    result = specifyExtremumLocation(y, x, current_scale_space_DoG_images, DoG_array, octave_index,
                                                     (scale_space_index + 1), SD, scale_ratio)
                    if result is not None:
                        keypoint_without_orientation, keypoint_image_index = result
                        keypoints_with_orientation = computeKeypointOrientations(keypoint_without_orientation,
                                                                                 octave_index,
                                                                                 gaussian_images[keypoint_image_index])

                        # Før vi tilføjer keypointsne til vores samlede keypoint array vil vi lige være sikre på at de
                        # -- ikke allerede findes i arrayet. Dette kan være tilfældet hvis de allerede er fundet i en
                        # -- anden oktav.
                        keypoints.extend(keypoints_with_orientation)
    return keypoints


def checkForDuplicateKeypoints(new_keypoints, keypoints_array):
    """

    """
    # For hvert keypoints i vores nye keypoints array
    for new_keypoint in new_keypoints:
        for existing_keypoint in keypoints_array:
            # Vi tjekker om keypointet allerede eksiterer
            if new_keypoint.coordinates[0] == existing_keypoint.coordinates[0] and \
                    new_keypoint.coordinates[1] == existing_keypoint.coordinates[1] and \
                    new_keypoint.size_sigma == existing_keypoint.size_sigma and \
                    new_keypoint.strength == existing_keypoint.strength and \
                    new_keypoint.orientation == existing_keypoint.orientation:  # and \
                # new_keypoint.octave != existing_keypoint.octave and \
                # new_keypoint.scale_space != existing_keypoint.scale_space:

                # Hvis det allerede eksiterer så fjerner vi det fra det nye array
                new_keypoints.remove(new_keypoint)
                print(f'\t\t(REMOVED DUPLICATE KEYPOINT)')

    # Til sidst returnerer vi det sorterede array
    return new_keypoints


def centerPixelIsExtrema(pixel_cube, image_mid, y, x):
    """
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


def specifyExtremumLocation(y, x, current_scale_space, current_DoG_stack, current_octave, image_index, SD, scale_ratio,
                            number_of_attempts=5, strenght_threshold=0.03, eigenvalue_ratio_threshold=10):
    """
    Metode for at beregne den præcise placering af extremum i en 3x3x3 cube af pixels.
    """
    ### --- specifyExtremumLocation --- ###

    # Vi pakker billederne ud som vi passerede fra før
    image_top, image_mid, image_bot = current_scale_space

    # Vi definerer et for-loop, der sætter et max for hvor mange gange, vi vil forsøge at tilnærme os placeringen
    for attemt in range(number_of_attempts):
        pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                               image_mid[y - 1:y + 2, x - 1:x + 2],
                               image_bot[y - 1:y + 2, x - 1:x + 2]]).astype('float32') / 255.0
        # Gradient beregnes. Læs metode for uddybelse.
        gradient = calculateGradient(pixel_cube)
        # Hessian beregnes. Læs metode for uddybelse.
        hessian = calculateHessian(pixel_cube)
        #
        offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if all(abs(offset) < 0.5):
            break
        y += int(round(offset[0]))
        x += int(round(offset[1]))
        image_index = int(round(offset[2]))
        if y < 3 or y > image_mid.shape[0] - 3 or x < 3 or x > image_mid.shape[
            1] - 3 or image_index < 1 or image_index > len(current_DoG_stack) - 2:
            # Det beregnede punkt er enten for tæt på kanten eller uden for billedet, derfor er keypointet her ikke stabilt
            return None
        if attemt >= number_of_attempts - 1:
            return None
        image_top, image_mid, image_bot = current_DoG_stack[image_index - 1: image_index + 2]

    extremum_strength = pixel_cube[1, 1, 1] + (0.5 * np.dot(gradient, offset))
    if abs(extremum_strength) >= strenght_threshold:
        one_image_hessian = hessian[:2, :2]
        hessian_trace = np.trace(one_image_hessian)
        hessian_determinant = np.linalg.det(one_image_hessian)
        if hessian_determinant > 0 and (hessian_trace ** 2) / hessian_determinant < (
                (eigenvalue_ratio_threshold + 1) ** 2) / eigenvalue_ratio_threshold:
            keypoint = KeyPoint(((y + offset[0]) * (scale_ratio ** current_octave),
                                 (x + offset[1]) * (scale_ratio ** current_octave)),
                                abs(extremum_strength), current_octave,
                                image_index, 1 / (scale_ratio ** current_octave),
                                SD * ((scale_ratio ** (1 / (len(current_DoG_stack) - 2))) ** image_index) * (
                                            scale_ratio ** (current_octave)))
            return keypoint, image_index
    return None


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

    # Vi vil gerne beregne vores gradient, det vil sige den retning som vores pixel har. Til at beregne denne bruges
    # -- hældningen for vores nuværende pixel. Hældningen approximeres igen ved at tage værdien for de to pixels på hver
    # -- side trukket fra hinanden. Først regner vi x. Vi vil gerne finde forskellen over x-aksen. Derfor skriver vi 1 i
    # -- første indgang (altså midterste billede), med 1 i anden indgang (i midten af y aksen). Og så tager vi x = 2 og
    # -- x = 0 og trækker fra hinanden. Hældningen er normal fundet og beskrevet vha differentialregning så vi kalder
    # -- vores variable for dx, dy og ds (ds er hældningen over vores scalespace, aksen igennem de tre lag billeder).
    dx = (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0]) / 2

    # Samme for y. Første indgag (top/mid/bot) er 1 igen. Nu er det y ændres mellem 2 og 0, og x er 1 konstant.
    dy = (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1]) / 2

    # For den sidste er det forskellen over de tre billeder i laget, så her ændres første indgang mellem 0 og 2.
    # -- y og x er konstante.
    ds = (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1]) / 2

    return np.array([dx, dy, ds])


def calculateHessian(pixel_cube):
    """
    Hessian regnes ud når vi har fundet et ekstremum. Værdierne i en hessian kan bruges til at se hvor et
    keypoint ligger henne. Vi er interesserede i at se om det ligger på en linje eller en i et hjørne. Essensen
    er: Hvis det ligger på en linje er vi rimelig ligeglade med keypointet da det kan være svært at sammenligne
    om to keypoints ligger det samme sted langs en linje (Dette er tilfældet hvis et keypoint hovedsagligt har
    en høj værdi i en enkelt retning). Hvis det derimod ligger i et hjørne er det lettere at sammenligne om de
    to keypoints er placeret tæt på hinanden og om de er ens (Vi ser et keypoint som bedre og mere beskrivende
    hvis det har høje værdier i flere retninger. Det betyder at det ligger hvor flere linjer krydser/ligger tæt
    som fx et hjørne eller et knudepunkt. Disse er mere interessante da de er mere karakteristiske end bare
    punkter langs en linje).

    Hessian bruges i bund og grund til at kigge på om et keypoint er "godt nok" til at kunne bruges senere til
    at finde ligheder imellem keypoints.
    """
    center_pixel_value = pixel_cube[1, 1, 1]

    dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
    dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
    dss = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]

    dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
    dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
    dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])

    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


def computeKeypointOrientations(keypoint, current_octave, image, SD_scale_factor=1.5, times_SD_covered_by_radius=3,
                                num_bins=36, peak_threshold_ratio=0.8):
    """
    Denne metode går ud på at beregne keypointets orientation. Det vil altså sige hvilken retning har det keypoint vi
    har med at gøre lige nu. Ideem er, at vi i selve keypointet og ud fra en radius fra keypointet ser på hvilke
    "retninger" der findes i de pixels der ligger rundt omkring vores keypoint. Ud fra alle de radier vil vi finde
    keypointets hoved-orientation. Så den dominerende retning beregnes ud fra alle de retninger der findes i en om vores
    pixel. Jo længere væk pixelen ligger fra vores keypoint position, jo mindre vægtes den i beregningen af hoved-
    retningen.
    """
    # Vi opretter et array til at holde vores keypoints efter vi har beregnet deres orientation
    keypoints_with_orientation = []

    y_coord, x_coord = int(round(keypoint.coordinates[0] * keypoint.image_scale)), int(
        round(keypoint.coordinates[1] * keypoint.image_scale))

    new_SD = keypoint.size_sigma * SD_scale_factor
    radius = int(round(new_SD * times_SD_covered_by_radius))
    weight_factor = -0.5 / (new_SD ** 2)

    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    # Vi vil gerne loope i en radius rundt om vores keypoint. Det gør vi ved at lave to arrays y og x, der looper over
    # -- et areal fra minus radius til radius. Vi har rad_y og rad_x der repræsenterer vores position i en radius ift
    # -- nuværende keypoint koordinat.
    for rad_y in range(-radius, radius + 1):
        # Vi tjekker om vores position plus vores radius index er indenfor billedet. Hvis ikke kan vi ikke bruge det i
        # -- vores beregning.
        if (y_coord + rad_y > 0 and y_coord + rad_y < image.shape[0] - 1):
            for rad_x in range(-radius, radius + 1):
                # Vi tjekker om nuværende pixel er indenfor x-aksen
                if (x_coord + rad_x > 0 and x_coord + rad_x < image.shape[1] - 1):
                    # Til beregningen af vores gradient, retningen for vores pixels kant, bruges hældningen på både x og
                    # -- y aksen af billedet. Hældningen approximeres igen ved at tage værdien for de to pixels på hver
                    # -- side trukket fra hinanden. Dette gøres for både x og y.

                    # For z beholder vi samme y-koordinat og tager pixelsne x + 1 og x -1 og trækker fra hinanden.
                    difference_x = image[y_coord + rad_y, x_coord + rad_x + 1] - image[
                        y_coord + rad_y, x_coord + rad_x - 1]
                    difference_y = image[y_coord + rad_y + 1, x_coord + rad_x] - image[
                        y_coord + rad_y - 1, x_coord + rad_x]

                    # For at beregne længden af gradienten bruges den euklidiske distance (basically pythagoras).
                    # -- P.S. tror ikke det hedder euklidisk distance på dansk men who cares
                    magnitude = np.sqrt(difference_x ** 2 + difference_y ** 2)

                    # Vi vil gerne beregne retningen af den fundne hældning. Det gørse vha funktionen arctan2. Den gives
                    # -- et punkt og spytter en vinkel ud i radianer. Den regner vinklen ud for en vektor som den
                    # -- tegner der går fra origo og et punkt P (som vi giver den) og en vektor der går
                    # -- fra origo til punktet (0, 1) (givet:(x,y)). Så bascially vinklen mellem OP og x-aksen i positiv
                    # -- retning. For senere at kunne opdele vores vinkler i bins med 10 graders mellemrum så bruger vi
                    # -- funktionen rad2deg til at omdanne resultatet fra radianer til grader.
                    orientation = np.rad2deg(np.arctan2(difference_y, difference_x))

                    # Jo længere væk fra keypointet vi befinder os, jo mindre skal denne pixels retning vægte i den
                    # -- endelige hovedretning, så nu bestemmer vi en vægt-faktor. Til beregningen bruges vores weight-
                    # -- factor fra før og hvor langt væk vi befinder os fra koordinatet, rad_y og rad_x. Vi vil som
                    # -- sagt gerne at jo længere væk vi er, jo mindre skal den vægte. Så omvendt, jo tættere vores
                    # -- rad_x og rad_y er på 0 jo højere vil vi gerne vægte værdien. Til det bruger vi at finde den
                    # -- exponentielle værdi af weight_factor ganget (rad_y ** 2 + rad_x ** 2). (De sættes i 2. for at
                    # -- undgå at de går ud med hinanden tilfældigt). Det vil sige når vi er
                    # -- ovenpå vores centerpixel for vores keypoint bliver resultatet af den eksponentielle funktion 1,
                    # -- så den vægtes 100%. Og jo større rad_x og rad_x er jo mindre bliver resultatet:
                    keypoint_weight = np.exp(weight_factor * (rad_y ** 2 + rad_x ** 2))

                    # Nu har vi vinklen for vores keypoint gradient. Den lægger vi i et histogram med 36 bins der svarer
                    # -- til alle 360 grader. Så første bin hedder 0-10 grader, næste hedder 11-20 grader, osv til 360.
                    # -- Vi regner det ud ved at tage num_bins (36) og dele med 360. Det giver 1/10. Når de ganges,
                    # -- svarer det til at dele orientation med 10; vi finder den af de 36 bins hvor vinklen hører til.
                    hist_index = int(round(orientation * num_bins / 360.0))

                    # Til sidst placerer vi vinklen i den bin hvor den hører til. Vi bruger et lille trick idet vi siger
                    # -- [hist_index % num_bins]. Det gør vi fordi vores nuværende måde at beregne hist_index på godt
                    # -- kan resultere i at give et index 36, hvis graden er tæt på 360. Så giver det jo 360/10 = 36. Da
                    # -- vi har oprettet et array med 36 indgange er det højeste index jo 35 (0-35), så det vi gør for
                    # -- at undgå en out of bounds er at vi finder modulus af hist_index. Hvis hist_index er 36 får vi
                    # -- 36 % 36 = 0, og så kommer vinklen bare i bin med 0, og en vinkel på 0 og 360 grader ligger jo
                    # -- også samme sted så det er bare fjong.
                    # -- Værdien vi lægger ind i histogram arrayet svarer til længden af gradienten ganget med weight.
                    raw_histogram[hist_index % num_bins] += keypoint_weight * magnitude

    # Nu har vi loopet igennem alle pixels i en radius omkring vores keypoints koordinater og fundet ud af hvor meget
    # -- hver retning vægter for det givne keypoints. Nu vil vi gerne finde ud af hvor i det histogram der er peaks for
    # -- at kunne bestemme vores keypoints hoved-orientation. Vi starter med at loope igennem alle bins i raw_histogram
    for n in range(num_bins):
        # På samme måde som vi vægtede hver keypoint før med omkringliggende pixels, vil vi gerne vægte hver bin med
        # -- omkringliggende bins. Vi laver en gaussian weighting hvor den midterste bin tæller mest (6) de to ved siden
        # -- af tæller næsten ligeså meget (4) og de to bins der ligger to ved siden af tæller kun for 1. Til sammen
        # -- giver det (6+4+4+1+1=) 16, så der deles med 16 for at normalisere summen af værdierne.
        # Den midtserte bin
        weighted_center_bin = 6 * raw_histogram[n]
        # De to bins der ligger lige til højre og venstre for
        weighted_adjacent_bins = 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins])
        # De to bins der ligger to til højre og venstre for
        weighted_next_adjacent_bins = raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]
        # Den fundne normaliserede værdi ligges ind i et nyt histogram kaldet smooth histogram på tilsvarende plads, n
        smooth_histogram[n] = (weighted_center_bin + weighted_adjacent_bins + weighted_next_adjacent_bins) / 16

    # np.where returnere alle pladser (indexer) i arrayet hvor den givede condition er sand
    # np.logical_and returnere et array med true på alle de pladser hvor begge conditions er sande
    # np.roll forskyder alle pladser i arrayet x antal pladser (i dette tilfælde 1 til højre og venstre)
    # Vi definerer et peak som værende der, hvor den givne bin er større end hver af sine naboer.
    orientation_peaks_indexes = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                        smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    # Vi starter med at finde den største peak vha np.max
    biggest_orientation = np.max(smooth_histogram)
    # Nu vil vi gerne loope over hver peak for at finde ud af om de ligger indenfor 80% af max peak. Hvis de gør, så
    # -- tæller vi de peaks med og gemmer dem som et nyt keypoint med den givne orientation
    for peak_index in orientation_peaks_indexes:
        orientation_peak = smooth_histogram[peak_index]
        # Vi tjekker om nuværende peak er over threshold (80% af max peak)
        if orientation_peak > biggest_orientation * peak_threshold_ratio:
            # her fitter vi en parabel til alle vores peaks som er store nok, for at beregne den deres "sub-bin"
            # -- position, så man kan beregne en præcis orientation "imellem" bins. Vi starter med at beregne værdierne
            # -- for binsne til højre og venstre for vores peak
            left_peak_value = smooth_histogram[peak_index - 1]
            right_peak_value = smooth_histogram[(peak_index + 1) % num_bins]

            # Ud fra nabo-værdierne kan vi vha formlen for kvadratisk interpolation:
            # -- (https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html)
            subbin_peak_index = (peak_index + (0.5 * ((left_peak_value - right_peak_value) / left_peak_value -
                                                      (2 * orientation_peak) + right_peak_value))) % num_bins

            # Ved at kende sub-bin indekset kan vi nu omregne dette til grader.
            keypoint_orientation = 360 - (subbin_peak_index * (360 / num_bins))
            if abs(keypoint_orientation - 360) < 0.001:
                keypoint_orientation = 0
            keypoint_with_orientation = KeyPoint(keypoint.coordinates, keypoint.strength, keypoint.octave,
                                                 keypoint.scale_space, keypoint.image_scale, keypoint.size_sigma,
                                                 keypoint_orientation)
            keypoints_with_orientation.append(keypoint_with_orientation)

    return keypoints_with_orientation


def makeKeypointDescriptors(keypoints, Gaussian_images, num_bins=8, num_windows=4, magnitude_threshold = 0.2):
    """

    """
    # Vi vil gerne finde hver pixel og rotere hver pixel modsat hovedrotationen for keypointet. På den måde udligner vi
    # -- for rotationen på keypointet før vi beregner dets desciptor. Det gør at når vi senere skal sammenligne alle
    # -- vores keypoints, så er de rotation invariant, og vi kan derfor sammenligne dem uden at vi risikerer at de ikke
    # -- matcher grundet et roteret objekt.

    # Vi sætter en distance for hvor langt rundt omkring vores keypoint vi vil tjekke for features der skal indgå i
    # -- vores descriptor. Her definerer vi distancen som værende 6 gange sigma. Hvis vi har en gaussian fordeling og
    # -- man tager en bredde på 3 gange sigma så har man dækket noget ala 97% eller 99% af hele fordelingen. Det gør vi
    # -- vi til hver side.
    keypoints_with_descriptors = []
    keypoints_outside_image = 0

    if len(keypoints) != 0:
        print(f'image: {Gaussian_images[1].shape}, sigma: {keypoints[0].size_sigma}, radius: {np.sqrt(2 * (int(keypoints[0].size_sigma*2)**2))}')

    for keypoint in keypoints:
        # Til at holde vores gradients opretter vi et 3d array. De første to pladser svarer til indicerne på vores 16
        # -- pladser i vores array.
        descriptor_histogram = np.zeros((num_windows, num_windows, num_bins))
        sigma_dist = int(2 * keypoint.size_sigma)
        rotation_angle = 360.0 - keypoint.orientation
        weight_factor = -0.5/(sigma_dist**2)

        degrees_pr_bin = 360/num_bins
        cos_angle = np.cos(np.deg2rad(rotation_angle))
        sin_angle = np.sin(np.deg2rad(rotation_angle))

        keypoint_y = keypoint.coordinates[0] * keypoint.image_scale
        keypoint_x = keypoint.coordinates[1] * keypoint.image_scale

        if keypoint_y + np.sqrt(2*(sigma_dist**2)) > Gaussian_images[keypoint.scale_space].shape[0]-1 or \
                keypoint_y - np.sqrt(2 * (sigma_dist**2)) < 1 or \
                keypoint_x - np.sqrt(2 * (sigma_dist**2)) < 1 or \
                keypoint_x + np.sqrt(2 * (sigma_dist**2)) > Gaussian_images[keypoint.scale_space].shape[1]-1:
            #print('Keypoint to close to image border')
            keypoints_outside_image += 1
            continue
        # Vi laver et loop over hver i et areal ud for vores keypoint center.
        for y in range(-sigma_dist, sigma_dist + 1):
            for x in range(-sigma_dist, sigma_dist + 1):
                ### AT FORDELE GRADIENT BLANDT DE FIRE NÆRMESTE BINS

                # Det næste vi skal til er at beregne hvor meget af den gradient vi finder midt i vores grid, der skal
                # -- lægges i hvert af de omkringliggende bins. Til det gør vi brug af noget matematik der hedder
                # -- trilinear interpolation. Det går ud på at man for et punkt der ligger midt i et kvadrat, finder ud
                # -- af hvor tæt på, den ligger på hvert af hjørnepunkterne, og derigennem finder ud af hvor meget af
                # -- dens værdi der skal puttes i hvert af punkterne. Jo tættere på et punkt, jo mere bidrager punktets
                # -- værdi til hvert hjørne. I vores tilfælde er hvert hjørne en bin et histogram, der hvert indeholder
                # -- 8 bins.
                # Vi lægger et grid med en længde og bredde lig med 2 gange sigma_dist ind over vores keypoint. Det gør
                # -- vi for at dække over et område, der ændrer sig alt efter hvor stort et keypoint vi har med at gøre.
                # -- Vi vil gerne ende med et koordinatsystem der går fra 0 til 4. Vi starter med at lægge sigma_dist
                # -- til både x og y værdierne for at de forskydes op så laveste x (x = -sigma_dist) bliver til x = 0.
                # -- Og fordi vi kigger på centrum af hver tile i vores grid, og de lige nu ligger i midten af hvert til
                # -- skal vi forskyde det med 1/4 sigma_dist, da det svarer til en halv tile-længde.
                x_pos = x + sigma_dist - ((1/4) * sigma_dist)
                y_pos = y + sigma_dist - ((1/4) * sigma_dist)

                # Lige nu har vi at der loopes over 0 til 2*sigma_dist. For at få det til at blive mellem 0 og 4, så
                # -- normaliserer vi med 0,5 * sigma dist for at få det mellem 0 og 4. (At dele med sigma_dist gør så
                # -- det bliver mellem 0 og 2, og ved så at dele med 0,5 også svarer det til at gange med 2.
                x_pos /= (0.5 * sigma_dist)
                y_pos /= (0.5 * sigma_dist)

                # For at finde ud af hvilken bin som vores nuværende x- og y-koordinat ligger tættest på, kan vi finde
                # -- den mindste x-koordinat ved at runde vores x-position ned til nærmeste heltal. For at finde største
                # -- x koordinat runder vi op til nærmeste heltal. Dette gøres for både x og y.
                min_bin_y = int(np.floor(y_pos))
                min_bin_x = int(np.floor(x_pos))
                max_bin_y = int(np.ceil(y_pos))
                max_bin_x = int(np.ceil(x_pos))

                UL_weight = (1 - (x_pos - min_bin_x)) * (1 - (y_pos - min_bin_y))
                UR_weight = (1 - (max_bin_x - x_pos)) * (1 - (y_pos - min_bin_y))
                LL_weight = (1 - (x_pos - min_bin_x)) * (1 - (max_bin_y - y_pos))
                LR_weight = (1 - (max_bin_x - x_pos)) * (1 - (max_bin_y - y_pos))

                ### AT ROTERE VORES GRID SÅ VI KIGGER PÅ DE RIGTIGE PIXELS I FORHOLD TIL VORES KEYPOINTS HOVEDROTATION

                # Nu ved vi hvilke y,x bins vores gradient skal placeres i. Nu mangler vi bare at finde ud af hvilke
                # -- bins de skal ende i. Så..!
                # ..nu skal der ganges vores yndlingsmatrix med en vektor 😎        | cos -sin |     | y |
                # -- Vi har rotationsmatricen ganget med en vektor [y,x],           | sin  cos |  *  | x |
                # -- hvor [y,x] svarer til vores nuværende koordinater i billedet. Ud fra den beregning kan vi finde
                # -- den nye værdi for y, y_rotated og den nye værdi for x, x_rotated. Det kan vi da resultatet af
                # -- rotationen giver en ny vektor [y_rotated, x_rotated] svarende til de nye beregnede værdier.
                y_rotated = int(round(y * cos_angle - x * sin_angle))
                x_rotated = int(round(y * sin_angle + x * cos_angle))

                # Nu har vi pixel-koordinaterne for der, hvor vores x og y værdi for vores roterede keypoint ligger. For
                # -- at finde beregne gradient og magnitude i den rigtige pixel, skal vi ind og have fat i den pxeil i
                # -- det nuværende billede. Det gør vi ved at kigge ind i billedet på vores x_rotated og y_rotated index
                current_image = Gaussian_images[keypoint.scale_space]

                # Til at beskrive vores keypoint via dets 16 histogrammer, så vægter vi de histogrammer der ligger tæt
                # -- på centrum af vores keypoint højere. Derfor beregner vi en Gaussian vægt-fordeling, der gør, at jo
                # -- længere pixelen ligger fra keypoint centrum, jo mindre bidrager dens værdier til dets histogram.
                pixel_weight = np.exp(weight_factor*(y**2 + x**2))

                # Vores nuværende pixel finder vi på det nuværende billede. Vores keypoint centrum har de koordinater
                # -- som vi har gemt i keypoint.coordinates. Og det er ud fra det centrum vi har beregnet vores roterede
                # -- x og y koordinater. Så vores position i billedet svarer til vores keypoint koordinater lagt sammen
                # -- med vores koordinatpar for y roteret og x roteret. Den pixel gemmer vi i en variabel current pixel.
                distance_x = current_image[int(y_rotated + keypoint_y), int((x_rotated + keypoint_x) + 1)]/255 - current_image[int(y_rotated + keypoint_y), int((x_rotated + keypoint_x) - 1)]/255
                distance_y = current_image[int((y_rotated + keypoint_y) - 1), int(x_rotated + keypoint_x)]/255 - current_image[int((y_rotated + keypoint_y) + 1), int(x_rotated + keypoint_x)]/255
                gradient_mag = np.sqrt(distance_x**2 + distance_y**2)
                gradient_ori = (np.rad2deg(np.arctan2(distance_y,distance_x))-rotation_angle) % 360
                pixel_contribution = gradient_mag * pixel_weight

                # Nu har vi de to vinkel-bins som vores vinkel ligger imellem. Nu skal vi finde ud af hvor tæt den er                
                min_bin_index = int(np.floor(gradient_ori/degrees_pr_bin)) % num_bins
                max_bin_index = int(np.ceil(gradient_ori/degrees_pr_bin)) % num_bins

                # Her har vi hvor meget den contributor til hver
                min_bin_angle_contribution = 1-((gradient_ori/degrees_pr_bin) - min_bin_index)

                if min_bin_y >= 0 and min_bin_x >= 0:
                    descriptor_histogram[min_bin_y, min_bin_x, min_bin_index] += pixel_contribution * UL_weight * min_bin_angle_contribution
                if min_bin_y >= 0 and max_bin_x <= 3:
                    descriptor_histogram[min_bin_y, max_bin_x, min_bin_index] += pixel_contribution * UR_weight * min_bin_angle_contribution
                if max_bin_y <= 3 and min_bin_x >= 0:
                    descriptor_histogram[max_bin_y, min_bin_x, min_bin_index] += pixel_contribution * LL_weight * min_bin_angle_contribution
                if max_bin_y <= 3 and max_bin_x <= 3:
                    descriptor_histogram[max_bin_y, max_bin_x, min_bin_index] += pixel_contribution * LR_weight * min_bin_angle_contribution

                if min_bin_index != max_bin_index:
                    max_bin_angle_contribution = 1-(max_bin_index - (gradient_ori/degrees_pr_bin))

                    if min_bin_y >= 0 and min_bin_x >= 0:
                        descriptor_histogram[min_bin_y, min_bin_x, max_bin_index] += pixel_contribution * UL_weight * max_bin_angle_contribution
                    if min_bin_y >= 0 and max_bin_x <= 3:
                        descriptor_histogram[min_bin_y, max_bin_x, max_bin_index] += pixel_contribution * UR_weight * max_bin_angle_contribution
                    if max_bin_y <= 3 and min_bin_x >= 0:
                        descriptor_histogram[max_bin_y, min_bin_x, max_bin_index] += pixel_contribution * LL_weight * max_bin_angle_contribution
                    if max_bin_y <= 3 and max_bin_x <= 3:
                        descriptor_histogram[max_bin_y, max_bin_x, max_bin_index] += pixel_contribution * LR_weight * max_bin_angle_contribution

        descriptor_vector = []

        for row in descriptor_histogram:
            for hist in row:
                descriptor_vector.extend(hist)

        descriptor_mag = np.sqrt(np.dot(descriptor_vector,descriptor_vector))
        descriptor_value_threshold = descriptor_mag * magnitude_threshold
        descriptor_vector = np.where(descriptor_vector > descriptor_value_threshold, descriptor_value_threshold, descriptor_vector)
        new_descriptor_mag = np.sqrt(np.dot(descriptor_vector,descriptor_vector))
        for i, vec_bin in enumerate(descriptor_vector):
            descriptor_vector[i] = int(round((vec_bin * 512)/new_descriptor_mag)) if (vec_bin * 512)/new_descriptor_mag < 255 else 255

        keypoint.descriptor = descriptor_vector.astype("uint8")
        keypoints_with_descriptors.append(keypoint)

    return keypoints_with_descriptors


def matchDescriptors(object_keypoints: [KeyPoint], data_keypoints: [KeyPoint], distance_ratio_treshold=1.5):
    """
    ER IKKE SIKKER PÅ DENNE BESKRIVELSE GÆLDER LÆNGERE
    Den her metode er til for at sammenligne to arrays af descriptors. Metoden skal bruge to arrays af descriptors. For
    at finde de nærmeste naboer gøres brug af k-nearest neighbor algoritmen. Det går kort sagt ud op at hvis vi har to
    lister af desriptors list_a og list_b, så for hver descriptor i list_a looper vi over og måler distancen mellem
    de descriptors. For hver sammenligning gemmer vi de to descriptors der har den korteste afstand til hinandenn.

    Lad os sige vi har descriptor 1 fra liste af: D1_a. Den sammenlignes nu med alle descriptors i list_b. Den første
    målte afstand mellem D1_a og D1_b vil selfølgelig være den korteste, da vi ikke har målt andre indtil nu. Hvis den
    anden afstand (D1_a til D2_b) er kortere vil vi gemme den som den korteste. Hvis vi så finder en afstand der er
    endnu kortere, fx D1_a til D5_b vil vi gemme den som den korteste afstand. Og på måde looper vi igennem begge lister
    af descriptors og for hver descriptor i list_a finder vi den nærmeste descriptor i list_b.

    Hvis vi tager det skridtet videre kan vi på samme måde gemme en liste af afstande til descriptors, over de x antal
    descriptors der ligger tættest på. I det her tilfælde vil vi gerne gemme de nærmeste to descriptors. I så fald
    har vi et array med to pladser, hvor vi først tjekker om den nye afstand er kortere end først index 0. Hvis ja, så
    fyrer vi den ind på index 0. I så fald skal vi så rykke hele arrayet en tak og sætte den nye værdi ind på første
    plads. Hvis ikke den er større så tjekker vi om den nye værdi er større end index 1. Hvis ja så rykker vi den ind på
    den plads og rykker resten af arrayet en tak. Hvis ikke så hvis vi gemmer flere kan vi tjekke flere pladser igennem.
    I vores tilfælde beholder vi kun de tætteste to.
    """
    match_list = []
    for object_keypoint in object_keypoints:
        keypoint_dists = []
        # Vi finder neraest neighbours
        for data_keypoint in data_keypoints:
            dist = np.linalg.norm(object_keypoint.descriptor - data_keypoint.descriptor)
            keypoint_dists.append(dist)

        min_dist = np.min(keypoint_dists)
        indexes_of_close_keypoints = np.where(keypoint_dists < min_dist*distance_ratio_treshold)[0]
        keypoint_match_list = np.array(data_keypoints)[indexes_of_close_keypoints]
        match_list.append(keypoint_match_list)
    return match_list

def matchKeypointsBetweenImages(marked_keypoints, scene_keypoints, marked_descriptors, scene_descriptors, marked_image, scene_image):
    MIN_MATCH_COUNT = 10

    # Passer billederne ind i stedet

        #img1 = cv2.imread('box.png', 0)  # queryImage
        #img2 = cv2.imread('box_in_scene.png', 0)  # trainImage


    # Vores descriptors er opbevaret i hver keypoint. Hmmm....

        #kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
        #kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(marked_descriptors, scene_descriptors, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([marked_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = marked_image.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        scene_image = cv.polylines(scene_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        h1, w1 = marked_image.shape
        h2, w2 = scene_image.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = marked_image
            newimg[:h2, w1:w1 + w2, i] = scene_image

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(marked_keypoints[m.queryIdx].pt[0]), int(marked_keypoints[m.queryIdx].pt[1] + hdif))
            pt2 = (int(scene_keypoints[m.trainIdx].pt[0] + w1), int(scene_keypoints[m.trainIdx].pt[1]))
            cv.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
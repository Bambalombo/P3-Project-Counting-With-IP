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
    - Hvert scalespace best??r af 3 DoG billeder.
    - Vi looper igennem alle pixels og laver en 3x3x3 cube fra midterste billedes pixels.
    - For hver cube finder vi ud af om den midterste pixel er et extremum (minimum eller maximum v??rdien i cuben).
    """
    # Vi opretter et array til at holde vores extrema pixels til senere hen https://cdn.discordapp.com/attachments/938032088756674652/1044169017155387511/image.png

    keypoints = []

    # Vi vil gerne loope igennem hver oktav (tre sammenh??ngende DoG billeder)
    # -- Her zipper vi elementer sammen fra det passerede DoG array, s??ledes at index 0 bliver grupperet med 1 og 2:
    # -- (0,1,2). Index 1 bliver ligeledes sat sammen med (1,2,3) og s?? f??r vi til sidst (2,3,4). Det svarer til de DoG
    # -- billeder der udg??r vores tre oktaver.
    for scale_space_index, (image_top, image_mid, image_bot) in enumerate(zip(DoG_array[:],
                                                                              DoG_array[1:],
                                                                              DoG_array[2:])):
        # Her looper vi nu over alle pixels men starter ved pixel 1 og til den andensidste pixel, da vi kr??ver at have
        # -- pixels hele vejen rundt om vores center-pixel. Det g??lder ikke ved kanterne s?? derfor starter vi ved [1,1]
        for y in range(1, image_mid.shape[0] - 1):
            for x in range(1, image_mid.shape[1] - 1):
                # Her vil vi udf??re et tjek om den givne pixel er et extremum i vores oktav. Alts?? vi vil tjekke om
                # -- denne pixel er ENTEN den st??rste ELLER den mindste i sin pixel-cube. F??rst opretter vi cuben. Det
                # -- g??res ved at slice de tre billeder som udg??r den nuv??rende oktav p?? en m??de s?? y,x er vores midte.
                current_scale_space_DoG_images = (image_top, image_mid, image_bot)
                current_pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                                               image_mid[y - 1:y + 2, x - 1:x + 2],
                                               image_bot[y - 1:y + 2, x - 1:x + 2]])
                # Hvis pixelen er et extremum skal vi finde ud af extremumets pr??cice placering. Dette er fordi at den
                # -- i teorien kan l??gge imellem pixels, forskudt af alle akser. Det er der noget matematik der siger.
                # -- Vi skal bare vide at afh??ngig af de andre pixels v??rdier i cuben, s?? er det IKKE sikkert at selve
                # -- toppunktet vi netop har fundet, ligger indenfor den samme pixel celle.
                if centerPixelIsExtrema(current_pixel_cube, image_mid, y, x):
                    result = specifyExtremumLocation(y, x, current_scale_space_DoG_images, DoG_array, octave_index,
                                                     (scale_space_index + 1), SD, scale_ratio)
                    if result is not None:
                        keypoint_without_orientation, keypoint_image_index = result
                        keypoints_with_orientation = computeKeypointOrientations(keypoint_without_orientation,
                                                                                 octave_index,
                                                                                 gaussian_images[keypoint_image_index])

                        # F??r vi tilf??jer keypointsne til vores samlede keypoint array vil vi lige v??re sikre p?? at de
                        # -- ikke allerede findes i arrayet. Dette kan v??re tilf??ldet hvis de allerede er fundet i en
                        # -- anden oktav.
                        keypoints.extend(keypoints_with_orientation)
    return keypoints

def resizeKeypoints(keypoints: [KeyPoint], scale_factor):
    for keypoint in keypoints:
        keypoint.coordinates = (np.array(keypoint.coordinates) / scale_factor)


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

                # Hvis det allerede eksiterer s?? fjerner vi det fra det nye array
                new_keypoints.remove(new_keypoint)
                print(f'\t\t(REMOVED DUPLICATE KEYPOINT)')

    # Til sidst returnerer vi det sorterede array
    return new_keypoints


def centerPixelIsExtrema(pixel_cube, image_mid, y, x):
    """
    Her gemmer vi v??rdien for center pixelen og s?? ??ndrer vi efterf??lgende centerpixelens v??rdi i cuben til 0, s??
    den efterf??lgende ikke t??lles med i tjekket for om den er et extremum. S?? kan vi nemlig bruge .all()
    """
    center_pixel = image_mid[y, x].copy()
    pixel_cube[1, 1, 1] = 0  # set center pixel to 0
    # Vi tjekker her om center_pixel er st??rre end alle andre pixels
    # --
    if (center_pixel > pixel_cube).all() or (center_pixel < pixel_cube).all():
        return True
    return False


def specifyExtremumLocation(y, x, current_scale_space, current_DoG_stack, current_octave, image_index, SD, scale_ratio,
                            number_of_attempts=5, strenght_threshold=0.03, eigenvalue_ratio_threshold=10):
    """
    Metode for at beregne den pr??cise placering af extremum i en 3x3x3 cube af pixels.
    """
    ### --- specifyExtremumLocation --- ###

    # Vi pakker billederne ud som vi passerede fra f??r
    image_top, image_mid, image_bot = current_scale_space

    # Vi definerer et for-loop, der s??tter et max for hvor mange gange, vi vil fors??ge at tiln??rme os placeringen
    for attemt in range(number_of_attempts):
        pixel_cube = np.array([image_top[y - 1:y + 2, x - 1:x + 2],
                               image_mid[y - 1:y + 2, x - 1:x + 2],
                               image_bot[y - 1:y + 2, x - 1:x + 2]]).astype('float32') / 255.0
        # Gradient beregnes. L??s metode for uddybelse.
        gradient = calculateGradient(pixel_cube)
        # Hessian beregnes. L??s metode for uddybelse.
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
            # Det beregnede punkt er enten for t??t p?? kanten eller uden for billedet, derfor er keypointet her ikke stabilt
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
    Gradient beregnes ved f??lgende formel (husk vi skal helst bare forst?? hvorfor vi bruger matematikken og ikke
    n??dvendigvis hvordan den virker):
    f'(x) = (f(x + s) - f(x - s)) / (2 * s)
    I formlen st??r x for koordinat og s for step, alts?? st??rrelsen af skridtet mellem hver v??rdi. Da vi her har
    at g??re med pixels i et grid er s (afstanden mellem hver pixel) lig med 1, s?? vi kan i stedet skrive:
    f'(x) = (f(x + 1) - f(x - 1)) / 2.
    Her tager vi udgangspunkt i centerpixelen. Vi husker at pixel_cube har holder pixel_cube[top,mid,bot], hvor
    top, mid og bot hver er et 3x3 slice af et billede.
    """

    # Vi vil gerne beregne vores gradient, det vil sige den retning som vores pixel har. Til at beregne denne bruges
    # -- h??ldningen for vores nuv??rende pixel. H??ldningen approximeres igen ved at tage v??rdien for de to pixels p?? hver
    # -- side trukket fra hinanden. F??rst regner vi x. Vi vil gerne finde forskellen over x-aksen. Derfor skriver vi 1 i
    # -- f??rste indgang (alts?? midterste billede), med 1 i anden indgang (i midten af y aksen). Og s?? tager vi x = 2 og
    # -- x = 0 og tr??kker fra hinanden. H??ldningen er normal fundet og beskrevet vha differentialregning s?? vi kalder
    # -- vores variable for dx, dy og ds (ds er h??ldningen over vores scalespace, aksen igennem de tre lag billeder).
    dx = (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0]) / 2

    # Samme for y. F??rste indgag (top/mid/bot) er 1 igen. Nu er det y ??ndres mellem 2 og 0, og x er 1 konstant.
    dy = (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1]) / 2

    # For den sidste er det forskellen over de tre billeder i laget, s?? her ??ndres f??rste indgang mellem 0 og 2.
    # -- y og x er konstante.
    ds = (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1]) / 2

    return np.array([dx, dy, ds])


def calculateHessian(pixel_cube):
    """
    Hessian regnes ud n??r vi har fundet et ekstremum. V??rdierne i en hessian kan bruges til at se hvor et
    keypoint ligger henne. Vi er interesserede i at se om det ligger p?? en linje eller en i et hj??rne. Essensen
    er: Hvis det ligger p?? en linje er vi rimelig ligeglade med keypointet da det kan v??re sv??rt at sammenligne
    om to keypoints ligger det samme sted langs en linje (Dette er tilf??ldet hvis et keypoint hovedsagligt har
    en h??j v??rdi i en enkelt retning). Hvis det derimod ligger i et hj??rne er det lettere at sammenligne om de
    to keypoints er placeret t??t p?? hinanden og om de er ens (Vi ser et keypoint som bedre og mere beskrivende
    hvis det har h??je v??rdier i flere retninger. Det betyder at det ligger hvor flere linjer krydser/ligger t??t
    som fx et hj??rne eller et knudepunkt. Disse er mere interessante da de er mere karakteristiske end bare
    punkter langs en linje).

    Hessian bruges i bund og grund til at kigge p?? om et keypoint er "godt nok" til at kunne bruges senere til
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
    Denne metode g??r ud p?? at beregne keypointets orientation. Det vil alts?? sige hvilken retning har det keypoint vi
    har med at g??re lige nu. Ideem er, at vi i selve keypointet og ud fra en radius fra keypointet ser p?? hvilke
    "retninger" der findes i de pixels der ligger rundt omkring vores keypoint. Ud fra alle de radier vil vi finde
    keypointets hoved-orientation. S?? den dominerende retning beregnes ud fra alle de retninger der findes i en om vores
    pixel. Jo l??ngere v??k pixelen ligger fra vores keypoint position, jo mindre v??gtes den i beregningen af hoved-
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

    # Vi vil gerne loope i en radius rundt om vores keypoint. Det g??r vi ved at lave to arrays y og x, der looper over
    # -- et areal fra minus radius til radius. Vi har rad_y og rad_x der repr??senterer vores position i en radius ift
    # -- nuv??rende keypoint koordinat.
    for rad_y in range(-radius, radius + 1):
        # Vi tjekker om vores position plus vores radius index er indenfor billedet. Hvis ikke kan vi ikke bruge det i
        # -- vores beregning.
        if (y_coord + rad_y > 0 and y_coord + rad_y < image.shape[0] - 1):
            for rad_x in range(-radius, radius + 1):
                # Vi tjekker om nuv??rende pixel er indenfor x-aksen
                if (x_coord + rad_x > 0 and x_coord + rad_x < image.shape[1] - 1):
                    # Til beregningen af vores gradient, retningen for vores pixels kant, bruges h??ldningen p?? b??de x og
                    # -- y aksen af billedet. H??ldningen approximeres igen ved at tage v??rdien for de to pixels p?? hver
                    # -- side trukket fra hinanden. Dette g??res for b??de x og y.

                    # For z beholder vi samme y-koordinat og tager pixelsne x + 1 og x -1 og tr??kker fra hinanden.
                    difference_x = image[y_coord + rad_y, x_coord + rad_x + 1] - image[
                        y_coord + rad_y, x_coord + rad_x - 1]
                    difference_y = image[y_coord + rad_y + 1, x_coord + rad_x] - image[
                        y_coord + rad_y - 1, x_coord + rad_x]

                    # For at beregne l??ngden af gradienten bruges den euklidiske distance (basically pythagoras).
                    # -- P.S. tror ikke det hedder euklidisk distance p?? dansk men who cares
                    magnitude = np.sqrt(difference_x ** 2 + difference_y ** 2)

                    # Vi vil gerne beregne retningen af den fundne h??ldning. Det g??rse vha funktionen arctan2. Den gives
                    # -- et punkt og spytter en vinkel ud i radianer. Den regner vinklen ud for en vektor som den
                    # -- tegner der g??r fra origo og et punkt P (som vi giver den) og en vektor der g??r
                    # -- fra origo til punktet (0, 1) (givet:(x,y)). S?? bascially vinklen mellem OP og x-aksen i positiv
                    # -- retning. For senere at kunne opdele vores vinkler i bins med 10 graders mellemrum s?? bruger vi
                    # -- funktionen rad2deg til at omdanne resultatet fra radianer til grader.
                    orientation = np.rad2deg(np.arctan2(difference_y, difference_x))
                    # Jo l??ngere v??k fra keypointet vi befinder os, jo mindre skal denne pixels retning v??gte i den
                    # -- endelige hovedretning, s?? nu bestemmer vi en v??gt-faktor. Til beregningen bruges vores weight-
                    # -- factor fra f??r og hvor langt v??k vi befinder os fra koordinatet, rad_y og rad_x. Vi vil som
                    # -- sagt gerne at jo l??ngere v??k vi er, jo mindre skal den v??gte. S?? omvendt, jo t??ttere vores
                    # -- rad_x og rad_y er p?? 0 jo h??jere vil vi gerne v??gte v??rdien. Til det bruger vi at finde den
                    # -- exponentielle v??rdi af weight_factor ganget (rad_y ** 2 + rad_x ** 2). (De s??ttes i 2. for at
                    # -- undg?? at de g??r ud med hinanden tilf??ldigt). Det vil sige n??r vi er
                    # -- ovenp?? vores centerpixel for vores keypoint bliver resultatet af den eksponentielle funktion 1,
                    # -- s?? den v??gtes 100%. Og jo st??rre rad_x og rad_x er jo mindre bliver resultatet:
                    keypoint_weight = np.exp(weight_factor * (rad_y ** 2 + rad_x ** 2))

                    # Nu har vi vinklen for vores keypoint gradient. Den l??gger vi i et histogram med 36 bins der svarer
                    # -- til alle 360 grader. S?? f??rste bin hedder 0-10 grader, n??ste hedder 11-20 grader, osv til 360.
                    # -- Vi regner det ud ved at tage num_bins (36) og dele med 360. Det giver 1/10. N??r de ganges,
                    # -- svarer det til at dele orientation med 10; vi finder den af de 36 bins hvor vinklen h??rer til.
                    hist_index = int(round(orientation * num_bins / 360.0))
                    # Til sidst placerer vi vinklen i den bin hvor den h??rer til. Vi bruger et lille trick idet vi siger
                    # -- [hist_index % num_bins]. Det g??r vi fordi vores nuv??rende m??de at beregne hist_index p?? godt
                    # -- kan resultere i at give et index 36, hvis graden er t??t p?? 360. S?? giver det jo 360/10 = 36. Da
                    # -- vi har oprettet et array med 36 indgange er det h??jeste index jo 35 (0-35), s?? det vi g??r for
                    # -- at undg?? en out of bounds er at vi finder modulus af hist_index. Hvis hist_index er 36 f??r vi
                    # -- 36 % 36 = 0, og s?? kommer vinklen bare i bin med 0, og en vinkel p?? 0 og 360 grader ligger jo
                    # -- ogs?? samme sted s?? det er bare fjong.
                    # -- V??rdien vi l??gger ind i histogram arrayet svarer til l??ngden af gradienten ganget med weight.
                    raw_histogram[hist_index % num_bins] += keypoint_weight * magnitude

    # Nu har vi loopet igennem alle pixels i en radius omkring vores keypoints koordinater og fundet ud af hvor meget
    # -- hver retning v??gter for det givne keypoints. Nu vil vi gerne finde ud af hvor i det histogram der er peaks for
    # -- at kunne bestemme vores keypoints hoved-orientation. Vi starter med at loope igennem alle bins i raw_histogram
    for n in range(num_bins):
        # P?? samme m??de som vi v??gtede hver keypoint f??r med omkringliggende pixels, vil vi gerne v??gte hver bin med
        # -- omkringliggende bins. Vi laver en gaussian weighting hvor den midterste bin t??ller mest (6) de to ved siden
        # -- af t??ller n??sten liges?? meget (4) og de to bins der ligger to ved siden af t??ller kun for 1. Til sammen
        # -- giver det (6+4+4+1+1=) 16, s?? der deles med 16 for at normalisere summen af v??rdierne.
        # Den midtserte bin
        weighted_center_bin = 6 * raw_histogram[n]
        # De to bins der ligger lige til h??jre og venstre for
        weighted_adjacent_bins = 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins])
        # De to bins der ligger to til h??jre og venstre for
        weighted_next_adjacent_bins = raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]
        # Den fundne normaliserede v??rdi ligges ind i et nyt histogram kaldet smooth histogram p?? tilsvarende plads, n
        smooth_histogram[n] = (weighted_center_bin + weighted_adjacent_bins + weighted_next_adjacent_bins) / 16

    # np.where returnere alle pladser (indexer) i arrayet hvor den givede condition er sand
    # np.logical_and returnere et array med true p?? alle de pladser hvor begge conditions er sande
    # np.roll forskyder alle pladser i arrayet x antal pladser (i dette tilf??lde 1 til h??jre og venstre)
    # Vi definerer et peak som v??rende der, hvor den givne bin er st??rre end hver af sine naboer.
    orientation_peaks_indexes = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                        smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    # Vi starter med at finde den st??rste peak vha np.max
    biggest_orientation = np.max(smooth_histogram)
    # Nu vil vi gerne loope over hver peak for at finde ud af om de ligger indenfor 80% af max peak. Hvis de g??r, s??
    # -- t??ller vi de peaks med og gemmer dem som et nyt keypoint med den givne orientation
    for peak_index in orientation_peaks_indexes:
        orientation_peak = smooth_histogram[peak_index]
        # Vi tjekker om nuv??rende peak er over threshold (80% af max peak)
        if orientation_peak > biggest_orientation * peak_threshold_ratio:
            # her fitter vi en parabel til alle vores peaks som er store nok, for at beregne den deres "sub-bin"
            # -- position, s?? man kan beregne en pr??cis orientation "imellem" bins. Vi starter med at beregne v??rdierne
            # -- for binsne til h??jre og venstre for vores peak
            left_peak_value = smooth_histogram[peak_index - 1]
            right_peak_value = smooth_histogram[(peak_index + 1) % num_bins]

            # Ud fra nabo-v??rdierne kan vi vha formlen for kvadratisk interpolation:
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
    # Vi vil gerne finde hver pixel og rotere hver pixel modsat hovedrotationen for keypointet. P?? den m??de udligner vi
    # -- for rotationen p?? keypointet f??r vi beregner dets desciptor. Det g??r at n??r vi senere skal sammenligne alle
    # -- vores keypoints, s?? er de rotation invariant, og vi kan derfor sammenligne dem uden at vi risikerer at de ikke
    # -- matcher grundet et roteret objekt.

    # Vi s??tter en distance for hvor langt rundt omkring vores keypoint vi vil tjekke for features der skal indg?? i
    # -- vores descriptor. Her definerer vi distancen som v??rende 6 gange sigma. Hvis vi har en gaussian fordeling og
    # -- man tager en bredde p?? 3 gange sigma s?? har man d??kket noget ala 97% eller 99% af hele fordelingen. Det g??r vi
    # -- vi til hver side.
    keypoints_with_descriptors = []
    keypoints_outside_image = 0

    for keypoint in keypoints:
        # Til at holde vores gradients opretter vi et 3d array. De f??rste to pladser svarer til indicerne p?? vores 16
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
            keypoints_outside_image += 1
            continue
        # Vi laver et loop over hver i et areal ud for vores keypoint center.
        for y in range(-sigma_dist, sigma_dist + 1):
            for x in range(-sigma_dist, sigma_dist + 1):
                ### AT FORDELE GRADIENT BLANDT DE FIRE N??RMESTE BINS

                # Det n??ste vi skal til er at beregne hvor meget af den gradient vi finder midt i vores grid, der skal
                # -- l??gges i hvert af de omkringliggende bins. Til det g??r vi brug af noget matematik der hedder
                # -- trilinear interpolation. Det g??r ud p?? at man for et punkt der ligger midt i et kvadrat, finder ud
                # -- af hvor t??t p??, den ligger p?? hvert af hj??rnepunkterne, og derigennem finder ud af hvor meget af
                # -- dens v??rdi der skal puttes i hvert af punkterne. Jo t??ttere p?? et punkt, jo mere bidrager punktets
                # -- v??rdi til hvert hj??rne. I vores tilf??lde er hvert hj??rne en bin et histogram, der hvert indeholder
                # -- 8 bins.
                # Vi l??gger et grid med en l??ngde og bredde lig med 2 gange sigma_dist ind over vores keypoint. Det g??r
                # -- vi for at d??kke over et omr??de, der ??ndrer sig alt efter hvor stort et keypoint vi har med at g??re.
                # -- Vi vil gerne ende med et koordinatsystem der g??r fra 0 til 4. Vi starter med at l??gge sigma_dist
                # -- til b??de x og y v??rdierne for at de forskydes op s?? laveste x (x = -sigma_dist) bliver til x = 0.
                # -- Og fordi vi kigger p?? centrum af hver tile i vores grid, og de lige nu ligger i midten af hvert til
                # -- skal vi forskyde det med 1/4 sigma_dist, da det svarer til en halv tile-l??ngde.
                x_pos = x + sigma_dist - ((1/4) * sigma_dist)
                y_pos = y + sigma_dist - ((1/4) * sigma_dist)

                # Lige nu har vi at der loopes over 0 til 2*sigma_dist. For at f?? det til at blive mellem 0 og 4, s??
                # -- normaliserer vi med 0,5 * sigma dist for at f?? det mellem 0 og 4. (At dele med sigma_dist g??r s??
                # -- det bliver mellem 0 og 2, og ved s?? at dele med 0,5 ogs?? svarer det til at gange med 2.
                x_pos /= (0.5 * sigma_dist)
                y_pos /= (0.5 * sigma_dist)

                # For at finde ud af hvilken bin som vores nuv??rende x- og y-koordinat ligger t??ttest p??, kan vi finde
                # -- den mindste x-koordinat ved at runde vores x-position ned til n??rmeste heltal. For at finde st??rste
                # -- x koordinat runder vi op til n??rmeste heltal. Dette g??res for b??de x og y.
                min_bin_y = int(np.floor(y_pos))
                min_bin_x = int(np.floor(x_pos))
                max_bin_y = int(np.ceil(y_pos))
                max_bin_x = int(np.ceil(x_pos))

                UL_weight = (1 - (x_pos - min_bin_x)) * (1 - (y_pos - min_bin_y))
                UR_weight = (1 - (max_bin_x - x_pos)) * (1 - (y_pos - min_bin_y))
                LL_weight = (1 - (x_pos - min_bin_x)) * (1 - (max_bin_y - y_pos))
                LR_weight = (1 - (max_bin_x - x_pos)) * (1 - (max_bin_y - y_pos))

                ### AT ROTERE VORES GRID S?? VI KIGGER P?? DE RIGTIGE PIXELS I FORHOLD TIL VORES KEYPOINTS HOVEDROTATION

                # Nu ved vi hvilke y,x bins vores gradient skal placeres i. Nu mangler vi bare at finde ud af hvilke
                # -- bins de skal ende i. S??..!
                # ..nu skal der ganges vores yndlingsmatrix med en vektor ????        | cos -sin |     | y |
                # -- Vi har rotationsmatricen ganget med en vektor [y,x],           | sin  cos |  *  | x |
                # -- hvor [y,x] svarer til vores nuv??rende koordinater i billedet. Ud fra den beregning kan vi finde
                # -- den nye v??rdi for y, y_rotated og den nye v??rdi for x, x_rotated. Det kan vi da resultatet af
                # -- rotationen giver en ny vektor [y_rotated, x_rotated] svarende til de nye beregnede v??rdier.
                y_rotated = int(round(y * cos_angle - x * sin_angle))
                x_rotated = int(round(y * sin_angle + x * cos_angle))

                # Nu har vi pixel-koordinaterne for der, hvor vores x og y v??rdi for vores roterede keypoint ligger. For
                # -- at finde beregne gradient og magnitude i den rigtige pixel, skal vi ind og have fat i den pxeil i
                # -- det nuv??rende billede. Det g??r vi ved at kigge ind i billedet p?? vores x_rotated og y_rotated index
                current_image = Gaussian_images[keypoint.scale_space]

                # Til at beskrive vores keypoint via dets 16 histogrammer, s?? v??gter vi de histogrammer der ligger t??t
                # -- p?? centrum af vores keypoint h??jere. Derfor beregner vi en Gaussian v??gt-fordeling, der g??r, at jo
                # -- l??ngere pixelen ligger fra keypoint centrum, jo mindre bidrager dens v??rdier til dets histogram.
                pixel_weight = np.exp(weight_factor*(y**2 + x**2))

                # Vores nuv??rende pixel finder vi p?? det nuv??rende billede. Vores keypoint centrum har de koordinater
                # -- som vi har gemt i keypoint.coordinates. Og det er ud fra det centrum vi har beregnet vores roterede
                # -- x og y koordinater. S?? vores position i billedet svarer til vores keypoint koordinater lagt sammen
                # -- med vores koordinatpar for y roteret og x roteret. Den pixel gemmer vi i en variabel current pixel.
                distance_x = current_image[int(y_rotated + keypoint_y), int((x_rotated + keypoint_x) + 1)]/255 - current_image[int(y_rotated + keypoint_y), int((x_rotated + keypoint_x) - 1)]/255
                distance_y = current_image[int((y_rotated + keypoint_y) - 1), int(x_rotated + keypoint_x)]/255 - current_image[int((y_rotated + keypoint_y) + 1), int(x_rotated + keypoint_x)]/255
                gradient_mag = np.sqrt(distance_x**2 + distance_y**2)
                gradient_ori = (np.rad2deg(np.arctan2(distance_y,distance_x))-rotation_angle) % 360
                pixel_contribution = gradient_mag * pixel_weight

                # Nu har vi de to vinkel-bins som vores vinkel ligger imellem. Nu skal vi finde ud af hvor t??t den er                
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


def matchDescriptorsWithKeypointFromSlice(object_keypoints: [KeyPoint], data_keypoints: [KeyPoint], distance_ratio_treshold=1.4):
    """
    Description
    """
    best_match_list = [[]for _ in range(len(object_keypoints))]
    best_match_dist = [[]for _ in range(len(object_keypoints))]
    for data_keypoint in data_keypoints:
        dist_list = []
        for object_keypoint in object_keypoints:
            dist = np.linalg.norm(object_keypoint.descriptor - data_keypoint.descriptor)
            dist_list.append(dist)
        if len(dist_list) == 0:
            continue
        best_match_list[dist_list.index((min(dist_list)))].append(data_keypoint)
        best_match_dist[dist_list.index((min(dist_list)))].append(min(dist_list))

    output_match_list = []
    for object_keypoint_dists, object_keypoint_matches in zip(best_match_dist, best_match_list):
        if len(object_keypoint_dists) == 0:
            continue
        elif len(object_keypoint_dists) == 1:
            min_value = object_keypoint_dists[0]
        else:
            min_value = min(i for i in object_keypoint_dists if i > 0)
        indices = np.where(object_keypoint_dists < min_value*distance_ratio_treshold)[0]
        output_match_list.append(np.array(object_keypoint_matches)[indices])

    return output_match_list

def matchOpenCVDescriptorsWithKeypointFromSlice(object_keypoints, object_descriptors, data_keypoints, data_descriptors, distance_ratio_treshold=1.4):
    """
    Description
    """
    best_match_keypoints = [[] for _ in range(len(object_keypoints))]
    best_match_descriptors = [[] for _ in range(len(object_keypoints))]
    best_match_dist = [[]for _ in range(len(object_keypoints))]
    for data_keypoint, data_descriptor in zip(data_keypoints,data_descriptors):
        dist_list = []
        for object_keypoint, object_descriptor in zip(object_keypoints, object_descriptors):
            dist = np.linalg.norm(object_descriptor - data_descriptor)
            dist_list.append(dist)
        if len(dist_list) == 0:
            continue
        best_match_keypoints[dist_list.index((min(dist_list)))].append(data_keypoint)
        best_match_descriptors[dist_list.index((min(dist_list)))].append(data_descriptor)
        best_match_dist[dist_list.index((min(dist_list)))].append(min(dist_list))

    output_match_keypoints = []
    output_match_descriptors = []
    for object_keypoint_dists, object_keypoint_matches, object_descriptor_matches in zip(best_match_dist, best_match_keypoints, best_match_descriptors):
        if len(object_keypoint_dists) == 0:
            continue
        elif len(object_keypoint_dists) == 1:
            min_value = object_keypoint_dists[0]
        else:
            min_value = min(i for i in object_keypoint_dists if i > 0)
        indices = np.where(object_keypoint_dists < min_value*distance_ratio_treshold)[0]
        output_match_keypoints.append(np.array(object_keypoint_matches)[indices])
        output_match_descriptors.append(np.array(object_descriptor_matches)[indices])
    return output_match_keypoints, output_match_descriptors
def validateKeypoints(slice_keypoints: [KeyPoint], scene_keypoints: [KeyPoint]):
    validated_keypoints = []
    for slice_keypoint in slice_keypoints:
        if not kNearestNeighbor(slice_keypoint,scene_keypoints) == None:
            validated_keypoints.append(slice_keypoint)

    return validated_keypoints
def validateOpenCVKeypoints(slice_keypoints, slice_descriptors, scene_keypoints, scene_descriptors):
    validated_keypoints = []
    validated_descriptors = []
    for slice_keypoint, slice_descriptor in zip(slice_keypoints,slice_descriptors):
        if not kNearestNeighborOpenCV(slice_descriptors,scene_descriptors) == None:
            validated_keypoints.append(slice_keypoint)
            validated_descriptors.append(slice_descriptor)
    return validated_keypoints, validated_descriptors

def kNearestNeighbor(keypoint: KeyPoint, data: [KeyPoint], kNN_treshold=0.8):
    nearest_neighbors = []
    nearest_neighbors_dist = []

    for data_keypoint in data:
        dist = np.linalg.norm(data_keypoint.descriptor - keypoint.descriptor)
        if len(nearest_neighbors_dist) != 2:
            nearest_neighbors_dist.append(dist)
            nearest_neighbors.append(keypoint)
        elif max(nearest_neighbors_dist) > dist:
            nearest_neighbors_dist[nearest_neighbors_dist.index(max(nearest_neighbors_dist))] = dist
            nearest_neighbors[nearest_neighbors_dist.index(max(nearest_neighbors_dist))] = keypoint

    if min(nearest_neighbors_dist) == 0:
        return nearest_neighbors[nearest_neighbors_dist.index(min(nearest_neighbors_dist))]
    elif min(nearest_neighbors_dist) >= max(nearest_neighbors_dist) * kNN_treshold:
        return nearest_neighbors[nearest_neighbors_dist.index(min(nearest_neighbors_dist))]
    else:
        return None

def kNearestNeighborOpenCV(descriptor, scene_descriptors, kNN_treshold=0.8):
    nearest_neighbors = []
    nearest_neighbors_dist = []

    for data_keypoint in scene_descriptors:
        dist = np.linalg.norm(data_keypoint - descriptor)
        if len(nearest_neighbors_dist) != 2:
            nearest_neighbors_dist.append(dist)
            nearest_neighbors.append(descriptor)
        elif max(nearest_neighbors_dist) > dist:
            nearest_neighbors_dist[nearest_neighbors_dist.index(max(nearest_neighbors_dist))] = dist
            nearest_neighbors[nearest_neighbors_dist.index(max(nearest_neighbors_dist))] = descriptor

    if min(nearest_neighbors_dist) == 0:
        return nearest_neighbors[nearest_neighbors_dist.index(min(nearest_neighbors_dist))]
    elif min(nearest_neighbors_dist) >= max(nearest_neighbors_dist) * kNN_treshold:
        return nearest_neighbors[nearest_neighbors_dist.index(min(nearest_neighbors_dist))]
    else:
        return None
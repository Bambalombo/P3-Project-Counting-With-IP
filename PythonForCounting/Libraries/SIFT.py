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


def differenceOfGaussian(image, SD, octave, scale_ratio, numberOfDoGs=5):
    gaussianKernel = makeGaussianKernel(SD * octave)
    borderimage = bd.addborder_reflect(image, gaussianKernel.shape[0])
    # blurredPictures = [convolve(borderimage, gaussianKernel)]
    blurredPictures = [cv.GaussianBlur(image, (0, 0), sigmaX=SD * octave, sigmaY=SD * octave)]
    k = (octave * scale_ratio) ** (1. / (numberOfDoGs - 2))
    for i in range(1, numberOfDoGs + 1):
        guassiankernel = makeGaussianKernel(SD * (k ** i))
        # blurredPictures.append(convolve(borderimage, gaussianKernel))
        blurredPictures.append(cv.GaussianBlur(image, (0, 0), sigmaX=(SD * (k ** i)), sigmaY=(SD * (k ** i))))

    DoG = []
    for (bottomPicture, topPicture) in zip(blurredPictures, blurredPictures[1:]):
        DoG.append(cv.subtract(topPicture, bottomPicture))

    return DoG


def defineKeyPointsFromPixelExtrema(DoG_array, octave_index, SD, scale_ratio):
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
                                                                                 DoG_array[keypoint_image_index])
                        keypoints.extend(keypoints_with_orientation)
    return keypoints


def centerPixelIsExtrema(pixel_cube, image_mid, y, x):
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
            # Det beregnede punkt er endten for tæt på kanten, eller uden for billedet, derfor er keypointet her ikke stabilt
            return None
        if attemt >= number_of_attempts - 1:
            return None
        image_top, image_mid, image_bot = current_DoG_stack[image_index - 1: image_index + 2]

    extremum_strength = image_mid[1, 1] + (0.5 * np.dot(gradient, offset))
    if abs(extremum_strength) >= strenght_threshold:
        one_image_hessian = hessian[:2, :2]
        hessian_trace = np.trace(one_image_hessian)
        hessian_determinant = np.linalg.det(one_image_hessian)
        if hessian_determinant > 0 and (hessian_trace ** 2) / hessian_determinant < (
                (eigenvalue_ratio_threshold + 1) ** 2) / eigenvalue_ratio_threshold:
            keypoint = KeyPoint((y + offset[0], x + offset[1]), abs(extremum_strength), current_octave, image_index + 1,
                                1 / current_octave,
                                SD * ((scale_ratio ** (1 / (len(current_DoG_stack) - 2))) ** image_index) * (
                                        scale_ratio ** (current_octave - 1)))
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

    y_coord, x_coord = int(round(keypoint.coordinates[0])), int(round(keypoint.coordinates[1]))

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

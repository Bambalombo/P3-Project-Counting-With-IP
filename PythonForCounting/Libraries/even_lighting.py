import cv2 as cv
import numpy as np
import math

def convolve_2D(image, kernel_size=3, filter_type=0, standard_deviation = 1):
    """
    - This method is for applying a kernel on an image. The kernel is convoluted over the image and applied to every pixel. 
    - The standard kernel is a mean-kernel of ones which blurs the image based on the provided kernel-size.
    - inspireret af andreas møgelmose live coding
    - lægger et mean filter af given kernel størrelse over det angivne billede
    """

    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    kernel = np.ones((kernel_size, kernel_size))
    kernel /= np.sum(kernel)

    output = np.zeros((image.shape[0] - kernel_size + 1, image.shape[1] - kernel_size + 1), dtype=np.uint8)

    print('der blurres')
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = image[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.sum(slice * kernel)


    return output


def convolve_3D(input_image, kernel_size=3, filter_type=0, standard_deviation = 1):
    """
    - This method is for applying a kernel on an image. The kernel is convoluted over every **channel** the image (*for a BGR this blurs B, G, and R separately, then assembled them into a new image*)
    - Blurs each channel of the input image with a kernel of one's of the specified size.
    - (ikke implementeret) filter_type = 0: mean-filter
    - (ikke implementeret) filter_type = 1: gaussian-filter
    """

    if len(input_image.shape) != 3:
        print(f"even_lighting.convolve_3d: input is missing channels attribute. Requires = 3. (Is image grayscale?)")
        return

    height, width, channels = input_image.shape
    output = np.zeros((height - kernel_size + 1, width - kernel_size + 1, channels), dtype=np.uint8)


    print('kalder at blurre kanal 1')
    output[:, :, 0] = convolve_2D(input_image[:, :, 0], kernel_size)
    print('kalder at blurre kanal 2')
    output[:, :, 1] = convolve_2D(input_image[:, :, 1], kernel_size)
    print('kalder at blurre kanal 3')
    output[:, :, 2] = convolve_2D(input_image[:, :, 2], kernel_size)

    return output


def illumination_mean_filter_2D(input_image, kernel_size=1):
    """
    https://www.sciencedirect.com/science/article/pii/S0030402619302359
    https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
    corrected_image[y,x] = original_image[y,x] - mean_filter_img[y,x] + mean(mean_filter_image)
    """
    height, width = input_image.shape
    #output = np.zeros((height, width, channels), dtype=np.uint8) # for cv2, since image doesn't get smaller
    output = np.zeros((height-kernel_size+1, width-kernel_size+1), dtype=np.float64)

    #lpf_img = cv.blur(input_image, (kernel_size, kernel_size))
    lpf_img = convolve_2D(input_image, kernel_size)
    lpf_mean = lpf_img.mean()

    for y in range(height-kernel_size+1):
        for x in range(width-kernel_size+1):
            value = float(input_image[int(y+(kernel_size/2)), int(x+(kernel_size/2))]) - float(lpf_img[y, x]) + lpf_mean
            output[y, x] = value

    print(output)

    # Normaliser array mellem 0 og 255. Formel fra: https://www.statology.org/normalize-data-between-0-and-100/
    # np.ptp betyder PeakToPeak. Det er forskellen mellem min og max værdien i et array
    output = ((output - np.min(output))/np.ptp(output))*255
    # Datatypen ændres til uint8 så cv kan læse den som et billede med pixelværdier.
    output = output.astype(np.uint8)

    print(output)

    return output


def illumination_mean_filter_BGR(input_image, kernel_size=1):
    """
    https://www.sciencedirect.com/science/article/pii/S0030402619302359
    https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
    corrected_image[y,x] = original_image[y,x] - mean_filter_img[y,x] + mean(mean_filter_image)

    Performs the 2D illumination correction for each color channel.
    """
    height, width, channels = input_image.shape
    #output = np.zeros((height, width, channels), dtype=np.uint8) # for cv2, since image doesn't get smaller
    # Opretter et nyt tomt billede. Her bruges dtype=float64 for beregne værdier over 255 og under 0 undervejs
    output = np.zeros((height-kernel_size+1, width-kernel_size+1, channels), dtype=np.float64)

    #lpf_img = cv.blur(input_image, (kernel_size, kernel_size))
    lpf_img = convolve_3D(input_image, kernel_size)
    lpf_mean = lpf_img.mean()

    print(f'jeg fik et billede. Jeg fik en kernel size på {kernel_size}')
    for channel in range(channels):
        print(f'jeg fixer lige lighting på channel {channel+1} af {channels}')
        for y in range(height-kernel_size+1):
            for x in range(width-kernel_size+1):
                new_pixel_value = float(input_image[int(y+(kernel_size/2)), int(x+(kernel_size/2)), channel]) - float(lpf_img[y, x, channel]) + lpf_mean
                output[y, x, channel] = new_pixel_value

    # Normaliser array mellem 0 og 255. Formel fra: https://www.statology.org/normalize-data-between-0-and-100/
    # np.ptp betyder PeakToPeak. Det er forskellen mellem min og max værdien i et array
    output = ((output - np.min(output))/np.ptp(output))*255
    # Datatypen ændres til uint8 så cv kan læse den som et billede med pixelværdier.
    output = output.astype(np.uint8)

    return output


def generate_gaussian_kernel(radius, standard_deviation):
    """
    Gaussian kernel inspireret af møgelmanden
    """
    gaussian = [(1 / (standard_deviation * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x / standard_deviation) ** 2) for x in range(-radius, radius + 1)]

    gaussian_kernel = np.zeros((radius * 2 + 1, radius * 2 + 1))

    for y in range(gaussian_kernel.shape[0]):
        for x in range(gaussian_kernel.shape[1]):
            gaussian_kernel[y, x] = gaussian[y] * gaussian[x]

    gaussian_kernel *= 1 / gaussian_kernel[0, 0]

    return gaussian_kernel


def clamp(n, minn, maxn):
    """
    short method for clamping a value between a minimum and a maximum value
    """
    return max(min(maxn, n), minn)

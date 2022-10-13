import cv2 as cv
import numpy as np
import math

def mean_filter_2d(input_image, kernel_size):
    """
    inspireret af andreas møgelmose live coding
    lægger et mean filter af given kernel størrelse over det angivne billede

    :param input_image:
    :param kernel_size:
    :return:
    """
    image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size))
    output = np.zeros((image.shape[0] - kernel_size + 1, image.shape[1] - kernel_size + 1), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = image[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.sum(slice * kernel) / np.sum(kernel)
    return output


def mean_filter_3d_VIRKER_IKKE(input_image, kernel_size=1):
    """
    Blurs the input image with a kernel of one's of the specified size.

    :param input_image: The image you wish to blur.
    :param kernel_size: The size (intensity) of the blur.
    :return: Outputs the blurred image.
    """
    height, width, channels = input_image.shape
    kernel = np.ones((kernel_size, kernel_size)) / np.sum(kernel_size)

    output = np.zeros((height - kernel_size + 1, width - kernel_size + 1, channels), dtype=np.uint8)

    for channel in range(channels):
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                slice = input_image[y:y + kernel_size, x:x + kernel_size, channel]
                output[y, x, channel] = np.sum(slice * kernel)
        cv.imshow("Har færddiggjort " + str(channel), output)

    return output


def illumination_low_pass_mean(input_image, kernel_size=1):
    """
    https://www.sciencedirect.com/science/article/pii/S0030402619302359
    corrected image = original image - LPF_img + mean(LPF_img)
    , where LFP_img a low pass filter of the input
    :param input_image:
    :param kernel_size:
    :return:
    """
    height, width, channels = input_image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)

    lpf_img = cv.blur(input_image, (kernel_size, kernel_size))
    lpf_mean = lpf_img.mean()

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                value = clamp(int(input_image[y, x, channel]) - int(lpf_img[y, x, channel]) + int(round(lpf_mean)), 0,
                              255)
                output[y, x, channel] = value

    return output

def generate_gaussian_kernel(radius, standard_deviation):
    gaussian = [(1/(standard_deviation*math.sqrt(2*math.pi)))*math.exp(-0.5*(x/standard_deviation)**2) for x in range(-radius, radius+1)]
    gaussian_kernel = np.zeros((radius*2+1,radius*2+1))
    for y in range(gaussian_kernel.shape[0]):
        for x in range(gaussian_kernel.shape[1]):
            gaussian_kernel[y, x] = gaussian[y] * gaussian[x]
    gaussian_kernel *= 1/gaussian_kernel[0,0]
    return gaussian_kernel

def illumination_low_pass_homomorphic(input_image, kernel_size):
    """
    https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
    :param input_image:
    :return:
    """
    height, width, channels = input_image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)

    logarithmic = np.zeros((height, width, channels), dtype=np.float64)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                logarithmic[y,x,channel] = math.log(input_image[y,x,channel])

    lpf_img = cv.blur(logarithmic, (kernel_size, kernel_size))
    exponential = np.zeros((height, width, channels), dtype=np.float64)
    print(logarithmic)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                exponential[y,x,channel] = math.exp(lpf_img[y,x,channel])

    fraction_footer = np.zeros((height, width, channels), dtype=np.float64)
    print(exponential)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                fraction_footer = input_image[y,x,channel]/exponential[y,x,channel]

    normalization_coefficient = input_image.mean() / fraction_footer.mean()
    print(normalization_coefficient)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                output[y,x,channel] = exponential[y,x,channel]*normalization_coefficient

    return output


def black_top_hat_filter(input_image):
    print('')



def linear_regression_lighting(input_image):
    print('')


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

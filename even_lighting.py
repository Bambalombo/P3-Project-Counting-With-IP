import cv2 as cv
import numpy as np


def convolve(input_image, kernel_size):
    image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size))
    kernel_size = kernel.shape[0]
    output = np.zeros((image.shape[0] - kernel_size + 1, image.shape[1] - kernel_size + 1), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = image[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.sum(slice * kernel) / np.sum(kernel)
    return output


def blur_image(input_image, kernel_size=1):
    """
    Blurs the input image with a kernel of one's of the specified size.

    :param input_image: The image you wish to blur.
    :param kernel_size: The size (intensity) of the blur.
    :return: Outputs the blurred image.
    """

    height, width, channels = input_image.shape
    kernel = np.ones((kernel_size, kernel_size)) / np.sum(kernel_size)

    output = np.zeros((height - kernel_size + 1, width - kernel_size + 1, channels),dtype=np.uint8)

    for channel in range(channels):
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                slice = input_image[y:y + kernel_size, x:x + kernel_size, channel]
                output[y, x, channel] = np.sum(slice * kernel)
        cv.imshow("Har færddiggjort "+str(channel), output)

    return output


def low_pass_lighting(input_image, kernel_size=1):
    """
    corrected image = original image - LPF_img + mean(LPF_img)
    , where LFP_img a low pass filter of the input
    :param input_image:
    :param kernel_size:
    :return:
    """
    height, width, channels = input_image.shape
    output = np.zeros((height, width, channels),dtype=np.uint8)

    lpf_img = cv.blur(input_image, (kernel_size,kernel_size))
    lpf_mean = lpf_img.mean()

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                value = clamp(int(input_image[y,x,channel]) - int(lpf_img[y,x,channel]) + int(round(lpf_mean)),0,255)
                output[y,x,channel] = value

    return output


def morphological_lighting(input_image):
    print('')


def homomorphic_lighting(input_image):
    print('')


def linear_regression_lighting(input_image):
    print('')


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
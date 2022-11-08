import cv2 as cv
import numpy as np
import even_lighting as el
import math

def illumination_lpf_mean_hsl(input_image, kernel_size=1):
    """
    https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
    https://www.sciencedirect.com/science/article/pii/S0030402619302359
    """
    height, width, channels = input_image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)

    input_image_HLS = cv.cvtColor(input_image, cv.COLOR_BGR2HLS)

    Lchannel = input_image_HLS[:, :, 1]

    Lchannel_bth = black_top_hat_filter(Lchannel,kernel_size)
    Lchannel_bth_mean = Lchannel_bth.mean()

    input_image_HLS[:,:,1] = input_image[:,:,1] - Lchannel_bth + Lchannel_bth_mean

    output = cv.cvtColor(input_image_HLS, cv.COLOR_HLS2BGR)

    return output


def illumination_low_pass_homomorphic(input_image, kernel_size, kernel=None):
    """
    https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html
    https://www.sciencedirect.com/science/article/pii/S0030402619302359
    """
    height, width, channels = input_image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)

    logarithmic = np.zeros((height, width, channels), dtype=np.float64)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                logarithmic[y, x, channel] = math.log(input_image[y, x, channel])

    if kernel == None:
        lpf_img = cv.blur(logarithmic, (kernel_size, kernel_size))
    else:
        lpf_img = el.convolve_2D(input_image, kernel)

    exponential = np.zeros((height, width, channels), dtype=np.float64)
    print(logarithmic)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                exponential[y, x, channel] = math.exp(lpf_img[y, x, channel])

    fraction_footer = np.zeros((height, width, channels), dtype=np.float64)
    print(exponential)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                fraction_footer = input_image[y, x, channel] / exponential[y, x, channel]

    normalization_coefficient = input_image.mean() / fraction_footer.mean()
    print(normalization_coefficient)

    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                output[y, x, channel] = exponential[y, x, channel] * normalization_coefficient

    return output


def black_top_hat_filter(input_image, kernel_size):
    """
    https://www.geeksforgeeks.org/top-hat-and-black-hat-transform-using-python-opencv/
    """
    filter_size = (kernel_size, kernel_size)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filter_size)

    # Reading the image named 'input.jpg'
    #input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # Applying the Top-Hat operation
    tophat_img = cv.morphologyEx(input_image, cv.MORPH_BLACKHAT, kernel)

    cv.imshow("original", input_image)
    cv.imshow("tophat", tophat_img)

    return tophat_img
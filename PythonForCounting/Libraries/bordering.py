import cv2 as cv
import numpy as np

def addPadding(img, sizey,sizex,padding: np.uint8):
    """

    :param img:
    :param sizey:
    :param sizex:
    :return:

    rescales picture with padding to size y,size x with intensity provided in the padding variable
    """
    paddedImage = np.ones((sizey,sizex), dtype=np.uint8)*padding
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            paddedImage[y+int((sizey-img.shape[0])/2),x+int((sizex-img.shape[1])/2)] = img[y,x]
    return paddedImage


def addborder_reflect(input_image, kernel_size):
    """
    This is a nice ass method
    """
    radius = int((kernel_size - 1) / 2) # kernel radius

    # Check to see if the image is 2D or 3D and creates shape attributes accordingly
    # channel is never used but this prevents an error
    if len(input_image.shape) == 3:
        input_height, input_width, input_channel = input_image.shape

        output = np.zeros((input_height + (radius * 2), input_width + (radius * 2), input_channel),dtype=np.uint8)
        output_height, output_width, output_channel = output.shape

    elif len(input_image.shape) == 2:
        input_height, input_width = input_image.shape

        output = np.ones((input_height + (radius * 2), input_width + (radius * 2)), dtype=np.uint8)*255
        output_height, output_width = output.shape

    cv.imwrite("input_image.png",input_image)

    if radius == 0:
        output = input_image

    elif radius <= input_height and input_width:

       
        for y in range(input_height):

            # Fill middle of output image
            for x in range(input_width):
                output[y + radius, x + radius] = input_image[y, x]

            # Sidernes kanter
            for x in range(radius):
                # Højre
                output[(radius) + y, (output_width - 1) - x] = input_image[y,(input_width - radius) + x]
                # Venstre
                output[(radius) + y, (radius) - x] = input_image[y, x]

        # Top og bund kanter           # Bemærk vi gør det i omvendt rækkefølge her (x før y), fordi begge gange tjekker
        for x in range(output_width):  # vi over hele bredden, så derfor kan vi ligeså godt gøre det i samme loop.
            for y in range(radius):
                # Top
                output[y, x] = output[(2 * radius) - y, x]
                # Bund
                output[(output_height - 1) - y, x] = output[(input_height - 1) + y, x]

    else:
        print("Kernel too big for image, idiot")

    return output
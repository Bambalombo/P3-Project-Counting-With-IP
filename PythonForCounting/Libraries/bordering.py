import cv2 as cv
import numpy as np

def addZeroPadding(img, sizey,sizex):
    """

    :param img:
    :param sizey:
    :param sizex:
    :return:

    rescales picture with zeropadding to size y,size x
    """
    paddedImage = np.zeros((sizey,sizex), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            paddedImage[y+int((sizey-img.shape[0])/2),x+int((sizex-img.shape[1])/2)] = img[y,x]
    return paddedImage

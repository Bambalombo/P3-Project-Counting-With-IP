import cv2 as cv
import numpy as np
from . import bordering as bd

def dilate(input_image,kernel_size):
    whitePixel = np.max(input_image)
    structuring_element = np.ones((kernel_size, kernel_size))*whitePixel
    borderedImg = bd.addZeroPadding(input_image, input_image.shape[0]+(kernel_size//2),input_image.shape[1]+(kernel_size//2))
    output = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = borderedImg[y:y + kernel_size, x:x + kernel_size]
            if np.any(slice == structuring_element):
                output[y,x] = 255
    return output

def erode(input_image,kernel_size):
    whitePixel = np.max(input_image)
    structuring_element = np.ones((kernel_size, kernel_size))*whitePixel
    borderedImg = bd.addZeroPadding(input_image, input_image.shape[0]+(kernel_size//2),input_image.shape[1]+(kernel_size//2))
    output = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = borderedImg[y:y + kernel_size, x:x + kernel_size]
            if np.all(slice == structuring_element):
                output[y,x] = 255
    return output

def close(input_image, kernel_size):
    dilateOutput = dilate(input_image, kernel_size)
    output = erode(dilateOutput,kernel_size)
    return output

def open(input_image, kernel_size):
    erodeOutput = erode(input_image, kernel_size)
    output = erode(erodeOutput,kernel_size)
    return output
import cv2 as cv
import numpy as np

def dilate(input_image,kernel_size,binary_threshold = 130):
    #if image ikke binary
    #binary = main.makeImageBinary(input_image,binary_threshold)

    structuring_element = np.ones((kernel_size, kernel_size))
    output = np.zeros((input_image.shape[0] - kernel_size + 1, input_image.shape[1] - kernel_size + 1), dtype=np.uint8)

    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            slice = input_image[y:y + kernel_size, x:x + kernel_size]
            if np.any(slice == structuring_element):
                output[y,x] = 255
            else:
                output[y,x] = 0

    return output

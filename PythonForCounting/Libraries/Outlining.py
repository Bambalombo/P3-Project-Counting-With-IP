import numpy as np
import cv2 as cv
from . import bordering as bd


def outlineFromBinary(img, kernelRadius):
    """
    :param img:
    :param kernelRadius:
    :return:

    Funktion der laver et eroted billede og trækker det fra det originale billede,
    for at få et billede med outlinen af objekter.

    """
    kernel = np.ones((kernelRadius * 2 + 1, kernelRadius * 2 + 1), dtype=np.uint8) * 255
    erodedImg = np.zeros((img.shape[0] - kernelRadius * 2, img.shape[1] - kernelRadius * 2), dtype=np.uint8)
    for y in range(erodedImg.shape[0]):
        for x in range(erodedImg.shape[1]):
            slice = img[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            if np.allclose(kernel, slice):
                erodedImg[y, x] = 255
            else:
                erodedImg[y, x] = 0

    paddedImage = bd.addPadding(erodedImg, img.shape[0], img.shape[1], np.uint8(0))
    output = cv.subtract(img, paddedImage)
    return output


def edgeWithSobel(img):
    kernel_radius = 1
    sobel_ver_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint8)
    sobel_hor_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.uint8)
    sobel_kernel_sum = np.sum(sobel_ver_kernel)

    vertical_apply = np.zeros(((img.shape[0] - (2 * kernel_radius + 1)), (img.shape[1] - (2 * kernel_radius + 1))),
                              dtype=np.uint8)
    horizontal_apply = vertical_apply.copy()
    for y in range(vertical_apply.shape[0]):
        for x in range(vertical_apply.shape[1]):
            image_slice = img[y:y + sobel_ver_kernel.shape[0], x:x + sobel_ver_kernel.shape[1]]
            vertical_apply[y, x] = (np.sum(image_slice * sobel_ver_kernel))

    for y in range(horizontal_apply.shape[0]):
        for x in range(horizontal_apply.shape[1]):
            image_slice = img[y:y + sobel_hor_kernel.shape[0], x:x + sobel_hor_kernel.shape[1]]
            horizontal_apply[y, x] = (np.sum(image_slice * sobel_hor_kernel))

    output = cv.add(vertical_apply, horizontal_apply)
    return output


def grassfire(img, white_pixel=255):
    """

    :param img:
    :param white_pixel:
    :return:
    """

    def startBurning(start_pos, current_burning_image):
        eight_connectivity_array = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
        burn_queue = deque()
        # den blob vi er i gang med at detekte lige nu
        current_blob = []
        burn_queue.append(start_pos)
        # kontrollere om der er noget i burnqueue, hvis der er så tager man denne som nextpos og forsætter med at brænde
        while len(burn_queue) > 0:
            next_pos = burn_queue.pop()
            # tilføjer den næste position til vores blob
            current_blob.append([next_pos[0] - 1, next_pos[1] - 1])
            # burningImage[nextpos[0],nextpos[1]] = 0
            # kontrollere rund om positionen om der der flere pixels
            for i in eight_connectivity_array:
                checkpos = [(next_pos[0] + i[0]), (next_pos[1] + i[1])]
                if current_burning_image[checkpos[0], checkpos[1]] == white_pixel and \
                        [checkpos[0] - 1, checkpos[1] - 1] not in current_blob and checkpos not in burn_queue:
                    burn_queue.append(checkpos)
        # hvis burnqueue er tom er blobben færdig så vi returner den
        return current_blob

    # laver en kant af nuller omkring det originale billede, for at kunne detekte blobs i kanten
    burning_image = bd.addPadding(img.copy(), img.shape[0] + 2, img.shape[1] + 2, np.uint8(0))
    # en liste over alle vores blobs, indeholder lister med koordinater for pixels
    blobs = []

    for y in range(burning_image.shape[0] - 2):
        for x in range(burning_image.shape[1] - 2):
            if burning_image[y + 1, x + 1] == white_pixel:
                found = False
                for blob in blobs:
                    if [y, x] in blob:
                        found = True
                        break
                if not found:
                    blobs.append(startBurning([y + 1, x + 1], burning_image))
    return blobs

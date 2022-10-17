import cv2
import numpy as np

#Reading the image
img = cv2.imread('EvenlightCoins.jpg')
#convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#lower bound and upper bound for color we want to find, BGR
#Green
lower_boundG = np.array([50, 20, 20])
upper_boundG = np.array([100, 255, 255])

#Yellow
#lower_boundY = np.array([20, 80, 80])
#upper_boundY = np.array([30, 255, 255])

#Red
lower_boundY = np.array([120, 40, 10])
upper_boundY = np.array([200, 255, 255])
#Blue

#Silver
lower_boundS = np.array([20, 20, 20])
upper_boundS = np.array([40, 255, 255])

#find the colors within the boundaries
maskG = cv2.inRange(hsv, lower_boundG, upper_boundG)
maskY = cv2.inRange(hsv, lower_boundY, upper_boundY)
maskS = cv2.inRange(hsv, lower_boundS, upper_boundS)

#define kernel size
kernel = np.ones((7,7),np.uint8)
# Remove unnecessary noise from mask
maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)
maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)
maskY = cv2.morphologyEx(maskY, cv2.MORPH_OPEN, kernel)
maskY = cv2.morphologyEx(maskY, cv2.MORPH_CLOSE, kernel)
maskS = cv2.morphologyEx(maskS, cv2.MORPH_OPEN, kernel)
maskS = cv2.morphologyEx(maskS, cv2.MORPH_CLOSE, kernel)

# Segment only the detected region
segmented_imgG = cv2.bitwise_and(img, img, mask=maskG)
segmented_imgY = cv2.bitwise_and(img, img, mask=maskY)
segmented_imgS = cv2.bitwise_and(img, img, mask=maskS)

#Find contours from the mask
contoursG, hierarchy = cv2.findContours(maskG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursY, hierarchy = cv2.findContours(maskY.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursS, hierarchy = cv2.findContours(maskS.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Draw contours
outputG = cv2.drawContours(segmented_imgG, contoursG, -1, (0, 255, 0), 3)
outputY = cv2.drawContours(segmented_imgY, contoursY, -2, (0, 0, 255), 3)
outputS = cv2.drawContours(segmented_imgS, contoursS, -2, (192, 192, 192), 3)

number_of_objects_in_imageG= len(contoursG)
number_of_objects_in_imageY= len(contoursY)
number_of_objects_in_imageS= len(contoursS)

print ("The number of GREEN objects in this image: ", str(number_of_objects_in_imageG))
print ("The number of RED objects in this image: ", str(number_of_objects_in_imageY))
print ("The number of SILVER COINS in this image: ", str(number_of_objects_in_imageS))

#Scaling the image
scale_percent = 60  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (300, 400)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
rHSV = cv2.resize(hsv, dim, interpolation=cv2.INTER_AREA)
reM = cv2.resize(maskG, dim, interpolation=cv2.INTER_AREA)
reM1 = cv2.resize(maskY, dim, interpolation=cv2.INTER_AREA)
reMS = cv2.resize(maskS, dim, interpolation=cv2.INTER_AREA)
reS = cv2.resize(segmented_imgG, dim, interpolation=cv2.INTER_AREA)
reOutG = cv2.resize(outputG, dim, interpolation=cv2.INTER_AREA)
reOutY = cv2.resize(outputY, dim, interpolation=cv2.INTER_AREA)
reOutS = cv2.resize(outputS, dim, interpolation=cv2.INTER_AREA)

# Showing the output
cv2.imshow("Image", resized)
cv2.imshow("HSV", rHSV)
#cv2.imshow('maskG', reM)
#cv2.imshow('maskY', reM1)
cv2.imshow('maskS', reMS)
#cv2.imshow("OutputG", reOutG)
#cv2.imshow("OutputY", reOutY)
cv2.imshow("OutputS", reOutS)


cv2.waitKey(0)
cv2.destroyAllWindows()
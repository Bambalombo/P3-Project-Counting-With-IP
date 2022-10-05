### det her er noget som vi har leget med men ikke fik os til at komme
#    ret, thresh = cv.threshold(img, 127, 255, 0)
 #   image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#
 #   opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
  #  closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=10)
#
 #   image, contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  #  cv.drawContours(closing, contours, -1, (0, 255, 0), 10)
   # output = makeGrayscale(img)
    #cv.imshow('input',img)
  #  cv.imshow('output', output)
   # print("NUMBER OF CUNTS: " + str(len(contours)))
   # cv.waitKey(0)
   # cv.destroyAllWindows()


# applyIntensity(img)
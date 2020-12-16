# --width 0.955   =>2.292.cm genislige sahip referans nesnemiz

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt



# x ve y koordinatları belli olan iki noktanın ortasını buluyor
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread('dogan.jpeg', cv2.IMREAD_COLOR)

#Resize The image
if img2.shape[1] >900:
     img2 = imutils.resize(img2, width=800,height=800)

# Reading same image in another
# variable and converting to gray scale.
img = cv2.imread('dogan.jpeg', cv2.IMREAD_GRAYSCALE)
if img.shape[1] > 900:
     img = imutils.resize(img, width=800,height=800)

_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

pixelsPerMetric = None

# loop over the contours individually
for c in contours:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 1000:
        continue
    elif cv2.contourArea(c) >150000:
        continue

    # compute the rotated bounding box of the contour
    orig = img2.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 100, 0), 3)


    list1 = box.tolist()  #nump arrayimi listeye cevirdim
    print(f'Contour coordinates: {list1}')
    print(len(list1))
    list2 = []

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the Eu clidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, thenl
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 0.955 #0.955 pixel_per_metric

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", orig)
    plt.imshow(orig)

    plt.show()

    # Exiting the window if 'q' is pressed on the keyboard.
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
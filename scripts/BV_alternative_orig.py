# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:20:31 2022

@author: ekinc
"""

# import opencv and numpy
import cv2
import numpy as np


#trackbar callback fucntion to update HSV value
def callback(x):
	global H_low,H_high,S_low,S_high,V_low,V_high
	#assign trackbar position value to H,S,V High and low variable
	H_low = cv2.getTrackbarPos('low H','controls')
	H_high = cv2.getTrackbarPos('high H','controls')
	S_low = cv2.getTrackbarPos('low S','controls')
	S_high = cv2.getTrackbarPos('high S','controls')
	V_low = cv2.getTrackbarPos('low V','controls')
	V_high = cv2.getTrackbarPos('high V','controls')


#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);


#global variable
H_low = 0
H_high = 179
S_low= 0
S_high = 255
V_low= 0
V_high = 255

#create trackbars for high,low H,S,V
cv2.createTrackbar('low H','controls',0,179,callback)
cv2.createTrackbar('high H','controls',179,179,callback)

cv2.createTrackbar('low S','controls',0,255,callback)
cv2.createTrackbar('high S','controls',255,255,callback)

cv2.createTrackbar('low V','controls',100,255,callback) 		# normally -> 0, 255, callback)
cv2.createTrackbar('high V','controls',255,255,callback)


name = ("U")
path1 = ("../images")



contours = []


def hsv():
    while (1):
        #
        # read source image
        img = cv2.imread("scan_Color." + name + ".png")
        # img = cv2.imread(path1 + "/illum20/scan_Color." + name + ".png")

        # convert sourece image to HSC color mode
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #
        hsv_low = np.array([H_low, S_low, V_low], np.uint8)
        hsv_high = np.array([H_high, S_high, V_high], np.uint8)

        # making mask for hsv range
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        #print(mask)
        # masking HSV value selected color becomes black
        res = cv2.bitwise_and(img, img, mask=mask)

        # show image
        cv2.imshow(name + ' bin', mask)
        cv2.imshow(name + ' rgb', res)

        # Filename

        # waitfor the user to press escape and break the while loop
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    filename2 = ('binary-' + name + '.jpg')
    cv2.imwrite("./saved/" + filename2, mask)
    cv2.imwrite("./saved/" + "res.jpg", res)


    k = cv2.waitKey(0)

    img_morph_in = cv2.imread("./saved/binary-" + name + ".jpg", 0)
    cv2.imshow("windowname", img_morph_in)
    k = cv2.waitKey(0)

    # Taking a matrix of size 5 as the kernel
    kernel_size = 20

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_erosion = cv2.erode(img_morph_in, kernel, iterations=1)
    cv2.imshow("EROSION", img_erosion)
    k = cv2.waitKey(0)
    # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    # cv2.imshow("EROSION", img_dilation)
    k = cv2.waitKey(0)

    cv2.imwrite("./saved/erosion.jpg", img_erosion)

    contours, hierarchy = cv2.findContours(img_erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print type(contours)
    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
        area = w * h
        if w > h:
            width = w
            height = h
        else:
            width = h
            height = w
        aspect_ratio = width / height
        if 100 < area < 3600 and  0.70 < aspect_ratio < 1.3:              # if 100 and area < 3600 and aspect_ratio > 0.70 and aspect_ratio < 1.3:
            print(x,y)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow("Show",img)
    cv2.imwrite("extracted.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    cv2.waitKey()




def main():
    hsv()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

cv2.destroyAllWindows()




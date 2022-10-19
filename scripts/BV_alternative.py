# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:20:31 2022

@author: ekinc
"""

# import opencv and numpy
import cv2
import numpy as np
import rospkg
from twophase_solver.enums import Color

# rospack = rospkg.RosPack()
fpath = '/home/student/catkin_ws/src/twophase_solver_ros/images/'
customname = 'scan_'


class Zeug():
    def __init__(self):

        # global variable
        self.H_low = 0
        self.H_high = 179
        self.S_low = 0
        self.S_high = 255
        self.V_low = 0
        self.V_high = 255
        # create a seperate window named 'controls' for trackbar
        cv2.namedWindow('controls', 2)
        cv2.resizeWindow("controls", 550, 10);
        # create trackbars for high,low H,S,V
        cv2.createTrackbar('low H', 'controls', 0, 179, self.callback)
        cv2.createTrackbar('high H', 'controls', 179, 179, self.callback)
        cv2.createTrackbar('low S', 'controls', 0, 255, self.callback)
        cv2.createTrackbar('high S', 'controls', 255, 255, self.callback)
        cv2.createTrackbar('low V', 'controls', 89, 255, self.callback)  # normally -> 0, 255, self.callback)
        cv2.createTrackbar('high V', 'controls', 255, 255, self.callback)

    # trackbar callback fucntion to update HSV value
    def callback(self, x):
        # assign trackbar position value to H,S,V High and low variable
        self.H_low = cv2.getTrackbarPos('low H', 'controls')
        self.H_high = cv2.getTrackbarPos('high H', 'controls')
        self.S_low = cv2.getTrackbarPos('low S', 'controls')
        self.S_high = cv2.getTrackbarPos('high S', 'controls')
        self.V_low = cv2.getTrackbarPos('low V', 'controls')
        self.V_high = cv2.getTrackbarPos('high V', 'controls')
        

    def load_image(self, i):
        filepath = fpath + customname + str(Color(i)) + '.png'
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        print "image loaded: " + filepath
        return img

    def hsv(self, img, i):
        # convert sourece image to HSC color mode
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_low = np.array([self.H_low, self.S_low, self.V_low], np.uint8)
        hsv_high = np.array([self.H_high, self.S_high, self.V_high], np.uint8)

        # making mask for hsv range
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        cv2.imshow('bin', mask)
        # print(mask)
        # masking HSV value selected color becomes black
        res = img.copy()
        res[:, :, 0] = img[:, :, 0] & mask
        res[:, :, 1] = img[:, :, 1] & mask
        res[:, :, 2] = img[:, :, 2] & mask

        # show image

        cv2.imshow('rgb', res)

        # Filename

        filename2 = ('binary-' + str(i) + '.jpg')
        # cv2.imwrite("./saved/" + filename2, mask)
        # cv2.imwrite("./saved/" + "res.jpg", res)

        # img_morph_in = cv2.imread("./saved/binary-" + str(i) + ".jpg", 0)
        cv2.imshow("windowname", mask)

        # Taking a matrix of size 5 as the kernel
        kernel_size = 20

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        img_erosion = cv2.erode(mask, kernel, iterations=1)
        cv2.imshow("EROSION", img_erosion)

        # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        # cv2.imshow("EROSION", img_dilation)

        cv2.imwrite("./saved/erosion.jpg", img_erosion)

        contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print type(contours)
        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w > h:
                width = w
                height = h
            else:
                width = h
                height = w
            aspect_ratio = width / height
            if 100 < area < 3600 and 0.70 < aspect_ratio < 1.3:
                print(x, y)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Show", img)
        # cv2.imwrite("extracted.jpg", img)
        cv2.waitKey(1)

    def loop(self):
        for i in range(6):
            img = self.load_image(i)
            self.hsv(img, i)
            k = cv2.waitKey(0)
            if k == 27:  # esc
                break
        cv2.destroyAllWindows()


def main():
    o = Zeug()
    o.loop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

cv2.destroyAllWindows()

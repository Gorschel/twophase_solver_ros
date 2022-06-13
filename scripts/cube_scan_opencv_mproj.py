#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from twophase_solver.enums import Color  # U=0 R=1 F=2 D=3 L=4 B=5
import rospkg
import math as m

# config
rospack = rospkg.RosPack()
fpath = rospack.get_path('twophase_solver_ros') + '/images/'
customname = 'scan_'


def save_image(img, i):
    filepath = fpath + customname + str(Color(i)) + '.png'
    cv2.imwrite(filepath, img)
    print "image saved: " + filepath


def load_image(i):
    filepath = fpath + customname + str(Color(i)) + '.png'
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    print "image loaded: " + filepath
    return img


def generateDefParams(h, w):
    'create default parameter set'
    params = cv2.SimpleBlobDetector_Params()
    maxArea = w * h
    params.minDistBetweenBlobs = 9.0
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256
    params.thresholdStep = 37
    # Filter by Area.
    params.filterByArea = True
    params.minArea = maxArea / 60
    params.maxArea = maxArea / 10
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.maxConvexity = 1.0
    # mglw verbugt
    params.filterByColor = False
    params.blobColor = 0  # 0..255
    return params


def detect_blobs(img):
    h, w = img.shape
    params = generateDefParams(h, w)  # create default parameter set
    detector = cv2.SimpleBlobDetector_create(params)  # create detector
    keypoints = detector.detect(img)  # detect
    img_pts = cv2.drawKeypoints(img, keypoints, np.array([]), (
        0, 0, 255))  # , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #flags für entsprechende kreisgröße
    if np.size(keypoints) == 9: pass  # prüfen ob farbzahl stimmt (9 von jeder farbe)
    return keypoints, img_pts


def find_nearest(array, value):
    """finds nearest array-value to a given scalar"""
    array = np.asarray(array)
    dif = np.abs(np.subtract(array, value))
    dev = np.sqrt(
        np.power(dif[:, 0], 2) + np.power(dif[:, 1], 2) + np.power(dif[:, 2], 2))  # abweichung zwischen centerfarben
    idx = dev.argmin()  # minimum
    return idx, array[idx]


def sort_data(data):
    """sort data_arr for rows, cols wrt corresponding coordinates"""
    # x, y, L, a, b
    # sort rows (y)
    for r in range(9):
        for n in range(9):
            if data[r, 1] < data[n, 1] and not r == n:
                save = np.copy(data[r])
                data[r] = np.copy(data[n])
                data[n] = np.copy(save)
    # sort cols (x) (3 felder)
    for b in range(3):
        for r in range(3):
            for n in range(3):
                if data[r + b * 3, 0] < data[n + b * 3, 0] and not r == n:
                    save = np.copy(data[r + b * 3])
                    data[r + b * 3] = np.copy(data[n + b * 3])
                    data[n + b * 3] = np.copy(save)
    return data


def get_chars(idx):
    chars = {
        0: "U",
        1: "R",
        2: "F",
        3: "D",
        4: "L",
        5: "B",
    }
    return chars.get(idx, "_")


def imshow(i, x, winname, img):
    if winname is not None:
        try:
            winname = str(i) + winname
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 20+(x*350), 20+(i*350))  # Move it to (40,30)
            cv2.imshow(winname, img)
        except:
            raise Exception("empty image")


def scan_cube():
    """get CubeDefStr from actual Rubics cube, using opencv"""
    ### init
    data_arr = np.zeros(shape=(6, 9, 5), dtype=np.int16)
    centers = np.zeros(shape=(6, 3), dtype='int')
    plot = range(0, 5)

    ### Cube-faces loop
    for i in range(6):

        img_raw = load_image(i)
        img = img_raw

        """
        # find circular sticker on face
        output = img_raw.copy()
        circles, n = None, 0

        img_raw_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        #img_raw_gray = cv2.medianBlur(img_raw_gray, 5)  # Reduce the noise to avoid false circle detection
        while circles is None and n < 10:
            circles = cv2.HoughCircles(image=img_raw_gray,
                                       method=cv2.HOUGH_GRADIENT,
                                       dp=1.2, minDist=500000,
                                       param1=100,  # Upper threshold for the internal Canny edge detector.
                                       param2=35,  # Threshold for center detection.
                                       minRadius=5, maxRadius=35)
            n += 1
        if circles is not None:  # ensure at least some circles were found
            circs = np.round(circles[0, :]).astype("int")  # convert the (x, y) coordinates and radius of the circles to integers
            for (x, y, r) in circs:  # loop over the (x, y) coordinates and radius of the circles
                # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
                cube_width = int(m.ceil(r*3.6))
                cv2.rectangle(output, (x - cube_width, y - cube_width), (x + cube_width, y + cube_width), (128, 0, 255), 0)
                img = img_raw[y-cube_width:y+cube_width, x-cube_width:x+cube_width, :]  # y_min:y_max, x_min:x_max
            # show the output image
            if 0 in plot:
                imshow(i, 0, "_0_circle", np.hstack([img_raw, output]))
                imshow(i, 3, "_0_img", img)
        else:
            raise Exception("[ResultError] no matching circles found in face " + str(Color(i)) + " image")
        """

        # farbraum trafos
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if 1 in plot:
            imshow(i, 4, "_1_GRAY", img_gray)
            imshow(i, 5, "_1_HSV", img_hsv[:, :, 2])

        # TODO: for 2nd cube (carbon): HSV.V[195..205..255] -> black stickers
        img_bin = np.zeros(img.shape, np.uint8)
        # 220..270° = blue -> 255*220/360..225*270/360 -> 156..191

        # threshold for hsv
        """
        lower = np.array([0, 0, 0], np.uint8)
        upper = np.array([255, 230, 100], np.uint8)
        cv2.inRange(img_hsv, lower, upper, img_bin)
        """
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 10, 200)
        if 2 in plot:
            imshow(i, 6, "_2_bin", canny)

        # img_bin_adapt = cv2.adaptiveThreshold(src=img_lab[:, :, 0], maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=81,C=0)

        # morphologische filter; iterations > 1 not working
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8), iterations=1,
                                   borderType=cv2.MORPH_RECT)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1,
                                   borderType=cv2.MORPH_CROSS)
        if 3 in plot:
            imshow(i, 7, "_3_bin_morph", img_bin)

        # TODO: find contours



        # TODO: find centers of mass (floodfill/fillPoly first)



        # TODO: find circular sticker



        # TODO: find other stickers around this



        # TODO: only look at sticker if in one of the 8 directions:



        # TODO: > iterate over centers of mass in ascending distance to face center.



        # TODO: > if nearest 4 stickers found find next 4 nearest



        # TODO: > if dist to any mass center > 2*min(dist(center0, center_nearest)) : too far away



        # TODO: if missing sticker or too far: new thresh and repeat cube face



        # TODO: allgemeines daten-array bereitstellen (koordinaten,farbdaten)


        ############################################################################################## i-Schleife vorbei

    # sticker- nach center-farben zuordnen und wertebereich ändern (U..B anstatt 0..5))
    CubeDefStr = ""
    for i in range(6):
        for r in range(9):
            value = data_arr[i][r][2:]  # current sticker color info
            idx, color = find_nearest(centers, value)
            CubeDefStr += get_chars(idx)

    # deinit & return
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    retval = 0  # platzhalter

    # TODO (main): validate solution

    return retval, CubeDefStr


class Scanner(object):
    # TODO
    def __init__(self):
        pass

    def sth(self):
        pass

def get_camera_settings(cam):
    print "########## current camera settings ##### (unset not shown)"
    res = {}
    vcap_properties = {'cv::CAP_PROP_POS_MSEC': 0,
                       "cv::CAP_PROP_POS_FRAMES": 1,
                       "cv::CAP_PROP_POS_AVI_RATIO": 2,
                       "cv::CAP_PROP_FRAME_WIDTH": 3,
                       "cv::CAP_PROP_FRAME_HEIGHT": 4,
                       "cv::CAP_PROP_FPS": 5,
                       "cv::CAP_PROP_FOURCC": 6,
                       "cv::CAP_PROP_FRAME_COUNT": 7,
                       "cv::CAP_PROP_FORMAT": 8,
                       "cv::CAP_PROP_MODE": 9,
                       "cv::CAP_PROP_BRIGHTNESS": 10,
                       "cv::CAP_PROP_CONTRAST": 11,
                       "cv::CAP_PROP_SATURATION": 12,
                       "cv::CAP_PROP_HUE": 13,
                       "cv::CAP_PROP_GAIN": 14,
                       "cv::CAP_PROP_EXPOSURE": 15,
                       "cv::CAP_PROP_CONVERT_RGB": 16,
                       "cv::CAP_PROP_WHITE_BALANCE_BLUE_U": 17,
                       "cv::CAP_PROP_RECTIFICATION": 18,
                       "cv::CAP_PROP_MONOCHROME": 19,
                       "cv::CAP_PROP_SHARPNESS": 20,
                       "cv::CAP_PROP_AUTO_EXPOSURE": 21,
                       "cv::CAP_PROP_GAMMA": 22,
                       "cv::CAP_PROP_TEMPERATURE": 23,
                       "cv::CAP_PROP_TRIGGER": 24,
                       "cv::CAP_PROP_TRIGGER_DELAY": 25,
                       "cv::CAP_PROP_WHITE_BALANCE_RED_V": 26,
                       "cv::CAP_PROP_ZOOM": 27,
                       "cv::CAP_PROP_FOCUS": 28,
                       "cv::CAP_PROP_GUID": 29,
                       "cv::CAP_PROP_ISO_SPEED": 30,
                       "cv::CAP_PROP_BACKLIGHT": 32,
                       "cv::CAP_PROP_PAN": 33,
                       "cv::CAP_PROP_TILT": 34,
                       "cv::CAP_PROP_ROLL": 35,
                       "cv::CAP_PROP_IRIS": 36,
                       "cv::CAP_PROP_SETTINGS": 37,
                       "cv::CAP_PROP_BUFFERSIZE": 38,
                       "cv::CAP_PROP_AUTOFOCUS": 39,
                       "cv::CAP_PROP_SAR_NUM": 40,
                       "cv::CAP_PROP_SAR_DEN": 41,
                       "cv::CAP_PROP_BACKEND": 42,
                       "cv::CAP_PROP_CHANNEL": 43,
                       "cv::CAP_PROP_AUTO_WB": 44,
                       "cv::CAP_PROP_WB_TEMPERATURE": 45,
                       "cv::CAP_PROP_CODEC_PIXEL_FORMAT": 46,
                       "cv::CAP_PROP_BITRATE": 47,
                       "cv::CAP_PROP_ORIENTATION_META": 48,
                       "cv::CAP_PROP_ORIENTATION_AUTO": 49,
                       "cv::CAP_PROP_OPEN_TIMEOUT_MSEC": 53,
                       "cv::CAP_PROP_READ_TIMEOUT_MSEC": 54
                       }
    for key in vcap_properties:
        value = vcap_properties[key]
        res[key] = cam.get(value)
        if res[key] != -1:
            print "{}:\t{}".format(key[4:], res[key]).expandtabs(13)

def set_camera(cam):
    pass

def set_sliders():
    cv2.namedWindow('marking')
    cv2.createTrackbar('H Lower', 'marking', 0, 179, nothing)
    cv2.createTrackbar('H Higher', 'marking', 179, 179, nothing)
    cv2.createTrackbar('S Lower', 'marking', 0, 255, nothing)
    cv2.createTrackbar('S Higher', 'marking', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'marking', 0, 255, nothing)
    cv2.createTrackbar('V Higher', 'marking', 255, 255, nothing)

def nothing(x):
    pass

def slider_test():
    """
    not working
    """
    # print "manual cube scan started."
    # retval, cube = scan_cube()
    # print "scan result: %s (%ss)" % (cube, len(cube))
    vcap = cv2.VideoCapture(0)
    # set_sliders()

    while True:
        _, img = vcap.read()
        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hL = cv2.getTrackbarPos('H Lower', 'marking')
        hH = cv2.getTrackbarPos('H Higher', 'marking')
        sL = cv2.getTrackbarPos('S Lower', 'marking')
        sH = cv2.getTrackbarPos('S Higher', 'marking')
        vL = cv2.getTrackbarPos('V Lower', 'marking')
        vH = cv2.getTrackbarPos('V Higher', 'marking')

        LowerRegion = np.array([hL, sL, vL], np.uint8)
        upperRegion = np.array([hH, sH, vH], np.uint8)

        img_bin = cv2.inRange(hsv, LowerRegion, upperRegion)

        kernel = np.ones((3, 3), "uint8")

        mask = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Masking ", mask)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            vcap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    scan_cube()

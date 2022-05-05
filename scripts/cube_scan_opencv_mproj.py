#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from twophase_solver.enums import Color  # U=0 R=1 F=2 D=3 L=4 B=5
import rospkg

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


def scan_cube():
    """get CubeDefStr from actual Rubics cube, using opencv"""

    # init
    ret = 0
    win_name = "init"  # vlt named window für menü hier erstellen
    data_arr = np.zeros(shape=(6, 9, 5), dtype=np.int16)
    centers = np.zeros(shape=(6, 3), dtype='int')

    # Cube-faces loop
    for i in range(6):
        # get cube face images

        img_raw = load_image(i)
        # bilder zuschneiden
        img = img_raw[150:480, 150:430, :]  # y_min:y_max, x_min:x_max, :

        # Save/Plot cfg
        plot = range(0, 9)

        # farbraum trafo / split
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_lab = img
        """if i == var:
            cv2.imwrite(path + "awbsp/img" + ".png", img)
            cv2.imwrite(path + "awbsp/hsv" + ".png", img_hsv)
            cv2.imwrite(path + "awbsp/lab" + ".png", img_lab)"""

        # binary image "schwarzes Würfelgitter"
        # adaptive threshhold 
        if 0 in plot:
            cv2.imshow("0_L" + str(i), img_lab[:, :, 0])
            cv2.imshow("0_H" + str(i), img_hsv[:, :, 0])

        # TODO (main): nach sättigung binarisieren
        #  -> schwarzes würfelgitter finden
        if 0 in plot: cv2.imshow("0_saturation" + str(i), img_hsv[:, :, 1])
        img_bin_gripper = img_gray.copy()
        # 220..270° = blue -> 255*220/360..225*270/360 -> 156..191
        cv2.inRange(img_hsv[:, :, 0], 150, 160, img_bin_gripper)
        # cv2.threshold(img_hsv[:, :, 0], 80, 255, cv2.THRESH_BINARY, img_bin_gripper)

        cv2.morphologyEx(img_bin_gripper, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8), iterations=1,
                         dst=img_bin_gripper)
        cv2.morphologyEx(img_bin_gripper, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=1,
                         dst=img_bin_gripper)
        if 1 in plot: cv2.imshow("1_thresh_blue" + str(i), img_bin_gripper)
        img_bin = cv2.adaptiveThreshold(src=img_lab[:, :, 0], maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=81,
                                        C=0)

        cv2.bitwise_or(img_bin, img_bin_gripper, dst=img_bin)

        if 2 in plot: cv2.imshow("2_thresh" + str(i), img_bin)

        # if i == var: cv2.imwrite(path + "awbsp/adaptivethresh" + ".png", img_bin)
        # morphologische filter; iterations > 1 not working
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8), iterations=1,
                                   borderType=cv2.MORPH_RECT)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1,
                                   borderType=cv2.MORPH_CROSS)
        # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1,
        #                           borderType=cv2.MORPH_CROSS)
        # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1,
        #                           borderType=cv2.MORPH_CROSS)
        if 3 in plot: cv2.imshow("3_nach_morph" + str(i), img_bin)
        # if i == var: cv2.imwrite(path + "awbsp/morph" + ".png", img_bin)

        # individuelle ROI finden
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(img_bin), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]  # unnötige listenstruktur loswerden

        # TODO: neuer ansatz hierachie-bedingungen:
        for nr, (cnt, hie) in enumerate(zip(contours, hierarchy)):
            if hie[2] >= 0 and hie[3] < 0:  # contour has child(s) but no parent
                # maybe_cube = cnt
                cube_cnt_id = nr
                frame = img.copy()
                min_y, min_x, _ = frame.shape
                max_y = max_x = 0
                for contour, hier in zip(contours, hierarchy):
                    if hier[3] == cube_cnt_id:
                        (x, y, w, h) = cv2.boundingRect(contour)
                        min_x, max_x = min(x, min_x), max(x + w, max_x)
                        min_y, max_y = min(y, min_y), max(y + h, max_y)
                        if w > 80 and h > 80:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if max_x - min_x > 0 and max_y - min_y > 0:
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

                if 4 in plot: cv2.imshow("konturbedingungen" + str(i), frame)
                # 9 subcontours
                # iterate over all child contours

                # get bounding rects for all subcontours -> array

                # array -> min/max x/y -> new rect

                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # if cv2.contourArea(approx) >= 500:
                x, y, w, h = cv2.boundingRect(cnt)
                mask = img_bin.copy()  # np.zeros((h,w)) #! wert wird auch später sonst immer auf 0 gesetzt
                mask[:, :] = 0  # maske leeren
                cv2.fillPoly(mask, [cnt], 255)
                img_bin_roi = cv2.bitwise_and(img_bin[y:y + h, x:x + w],
                                              mask[y:y + h, x:x + w])  # remove any other objects in roi
                img_lab_roi = img_lab[y:y + h, x:x + w]  # für pixelzugriff mit schwerpunktkordinaten nötig
                if 5 in plot: cv2.imshow('3_mask' + str(i), mask)
                """if i == var:        
                    cv2.imwrite(path + "awbsp/bin-roi" + ".png", img_bin_roi)
                    cv2.imwrite(path + "awbsp/mask_cnt" + ".png", mask)
                    cv2.imwrite(path + "awbsp/lab-roi" + ".png", img_lab_roi)"""
            else:
                continue  # skip contour

        # schwerpunkte der sticker finden (blob detector)
        try:
            if 6 in plot:
                cv2.imshow("bin_roi" + str(i), img_bin_roi)
        except UnboundLocalError:
            print "threshold failed"
        keypoints, img_pts = detect_blobs(img_bin_roi)
        if 7 in plot: cv2.imshow('4_keypoints' + str(i), img_pts)
        # if i == var: cv2.imwrite(path + "awbsp/keypoints" + ".png", img_pts)

        # allgemeines daten-array bereitstellen (koordinaten,farbdaten) 
        img_probe = cv2.medianBlur(img_lab_roi, 3)
        if len(keypoints) == 9:
            for r in range(9):
                pt = keypoints[r].pt
                x = int(pt[0])
                y = int(pt[1])
                data_arr[i][r][0] = x  # x-Pos
                data_arr[i][r][1] = y  # y-Pos
                data_arr[i][r][2:] = img_probe[y][x]  # L a b
        else:
            pass

        # Blob-schwerpunkte nach Bildkoordinaten sortieren
        data_arr[i] = sort_data(np.copy(data_arr[i]))

        # center farben zwischenspeichern
        centers[i] = data_arr[i][4][2:]

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


if __name__ == "__main__":
    print "manual cube scan started."
    retval, cube = scan_cube()
    print "scan result: %s (%ss)" % (cube, len(cube))

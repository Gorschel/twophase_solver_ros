#!/usr/bin/env python
# coding: utf-8

import rospy
import numpy as np
import cv2
import rospkg
from enums import Color # U=0 R=1 F=2 D=3 L=4 B=5
from matplotlib import pyplot as plt # histograms
import rospkg


### config

modus = 1  # 0 = record # vlt über menü auswahl steuern
rospack = rospkg.RosPack()
fpath = rospack.get_path('twophase_solver_ros') + '/images/'
customname = 'Anwendungsbeispiel'


def save_image(img, i):
    filepath = fpath + customname + str(Color(i)) + '.png'
    cv2.imwrite(filepath, img)
    print "image saved: " + filepath

def load_image(i):
    filepath = fpath + customname + str(Color(i)) + '.png'
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    print "image loaded: " + filepath
    return img

def generateDefParams(h,w):
    'create default parameter set'
    params = cv2.SimpleBlobDetector_Params()
    maxArea = w*h
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256
    params.thresholdStep = 37
    # Filter by Area.
    params.filterByArea = True
    params.minArea = maxArea/50
    params.maxArea = maxArea/9
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.maxConvexity = 1.0
    # mglw verbugt
    params.filterByColor = False
    params.blobColor = 0 #0..255
    return params

def detect_blobs(img):
    h,w = img.shape
    params = generateDefParams(h,w) # create default parameter set
    detector = cv2.SimpleBlobDetector_create(params) # create detector
    keypoints = detector.detect(img) # detect
    img_pts = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255)) #, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #flags für entsprechende kreisgröße
    if np.size(keypoints) == 9: # prüfen ob farbzahl stimmt (9 von jeder farbe) 
    return keypoints, img_pts

def find_nearest(array, value):
    """finds nearest array-value to a given scalar"""
    array = np.asarray(array)
    dif = np.abs(np.subtract(array, value))
    dev = np.sqrt( np.power(dif[:,0], 2) + np.power(dif[:,1], 2) + np.power(dif[:,2], 2))   # abweichung zwischen centerfarben
    idx = dev.argmin() # minimum
    return idx, array[idx]

def sort_data(data):
    """sort data_arr for rows, cols wrt corresponding coordinates"""
    # x, y, L, a, b
    #sort rows (y)
    for r in range(9):
        for n in range(9):
            if data[r,1] < data[n,1] and not r == n:
                save = np.copy(data[r])
                data[r] = np.copy(data[n])
                data[n] = np.copy(save)
    # sort cols (x) (3 felder)
    for b in range(3):
        for r in range(3):
            for n in range(3):
                if data[r+b*3,0] < data[n+b*3,0] and not r == n:
                    save = np.copy(data[r+b*3])
                    data[r+b*3] = np.copy(data[n+b*3])
                    data[n+b*3] = np.copy(save)
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
    return chars.get(idx,"_")

def scan_cube():
    """get CubeDefStr from actual Rubics cube, using opencv"""

    ### init
    ret = 0
    win_name = "init"  # vlt named window für menü hier erstellen
    data_arr = np.zeros(shape=(6,9,5), dtype = np.int16)
    centers = np.zeros(shape=(6,3), dtype = 'int')

    ### vlt menü
    # plot text in blank image
    # actions like save imgs, start scan, etc

    ### Cube-faces loop
    for i in range(6):
        ### get cube face images
        if modus == 0:      # bilder aufnehmen
            cam = cv2.VideoCapture(2)
            #cam.set(cv2.CAP_PROP_EXPOSURE, 0.1)
            #cam.set(cv2.CAP_PROP)
            width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            channels = 3
            cv2.namedWindow(win_name)
            # update window name
            win_name_new = "cam preview for: " + str(Color(i)) + " . [ESC:abort, SPACE:save]"
            cv2.setWindowTitle(win_name, win_name_new)   
            # preview camera feed
            while True:
                ret, frame = cam.read()
                cv2.imshow(win_name, frame)
                if not ret: 
                    retval = 0 
                    break
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    cv2.destroyAllWindows() #disable preview window
                    print("aborted")
                    retval = 1
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img = frame
                    retval = 2
                    cv2.setWindowTitle(win_name, "[saved]" + str(Color(i)))
                    break
            if retval < 2: break    
            win_name = win_name_new
            cv2.destroyAllWindows
            cam.release()
            save_image(img, i)
        elif modus == 1:    # bilder laden
            img = load_image(i)

        ##Save/Plot cfg
        var = -1
        plot = False

        ## farbraum trafo / split
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        #img_lab = img
        if i == var:
            cv2.imwrite(path + "awbsp/img" + ".png", img)
            cv2.imwrite(path + "awbsp/hsv" + ".png", img_hsv)
            cv2.imwrite(path + "awbsp/lab" + ".png", img_lab)
    
        ## binary image "schwarzes Würfelgitter"
        # adaptive threshhold 
        if plot: cv2.imshow("L"+str(i), img_lab[:,:,0])
        img_bin = cv2.adaptiveThreshold(img_lab[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 5) # nach value binarisieren
        if plot: cv2.imshow("adaptive"+str(i), img_bin)
        if i == var: cv2.imwrite(path + "awbsp/adaptivethresh" + ".png", img_bin)
        # morphologische filter; iterations > 1 not working
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1, borderType=cv2.MORPH_CROSS)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1, borderType=cv2.MORPH_CROSS)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1, borderType=cv2.MORPH_CROSS)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1, borderType=cv2.MORPH_CROSS)
        if plot: cv2.imshow("nach morph"+str(i), img_bin) 
        if i == var: cv2.imwrite(path + "awbsp/morph" + ".png", img_bin)

        ## individuelle ROI finden
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(img_bin), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] # unnötige listenstruktur loswerden
        for (cnt, hie) in zip(contours, hierarchy):      
            if hie[2] >= 0 and hie[3] < 0: # contour has child(s) but no parent
                x,y,w,h = cv2.boundingRect(cnt)
                mask = img_bin.copy() #np.zeros((h,w)) #! wert wird auch später immer auf 0 gesetzt
                mask[:,:] = 0 # maske leeren 
                cv2.fillPoly(mask, [cnt], 255)
                img_bin_roi = cv2.bitwise_and(img_bin[y:y+h, x:x+w], mask[y:y+h, x:x+w]) # remove any other objects in roi  
                img_lab_roi = img_lab[y:y+h, x:x+w] # für pixelzugriff mit schwerpunktkordinaten nötig
                if plot: cv2.imshow('mask'+str(i), mask)   
                if i == var:        
                    cv2.imwrite(path + "awbsp/bin-roi" + ".png", img_bin_roi)
                    cv2.imwrite(path + "awbsp/mask_cnt" + ".png", mask)
                    cv2.imwrite(path + "awbsp/lab-roi" + ".png", img_lab_roi)
            else:
                continue

        ## schwerpunkte der sticker finden (blob detector)
        keypoints, img_pts = detect_blobs(img_bin_roi)
        if plot: cv2.imshow('keypoints'+str(i), img_pts)
        if i == var: cv2.imwrite(path + "awbsp/keypoints" + ".png", img_pts)

        ## allgemeines daten-array bereitstellen (koordinaten,farbdaten) 
        img_probe = cv2.medianBlur(img_lab_roi, 3)
        if len(keypoints) == 9:
            for r in range(9):
                pt = keypoints[r].pt
                x = int(pt[0])
                y = int(pt[1])
                data_arr[i][r][0] = x # x-Pos
                data_arr[i][r][1] = y # y-Pos
                data_arr[i][r][2:] = img_probe[y][x] # L a b
        else:
            pass

        ## Blob-schwerpunkte nach Bildkoordinaten sortieren
        data_arr[i] = sort_data(np.copy(data_arr[i]))

        ## center farben zwischenspeichern
        centers[i] = data_arr[i][4][2:]
    # i-Schleife vorbei

    ## sticker- nach center-farben zuordnen und wertebereich ändern (U..B anstatt 0..5))
    CubeDefStr = ""
    for i in range(6):
        for r in range(9):
            value = data_arr[i][r][2:] # current sticker color info
            idx, color = find_nearest(centers, value)
            CubeDefStr += get_chars(idx)

    ### deinit & return
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    retval = 0  # platzhalter
    return retval, CubeDefStr

if __name__ == "__main__":
    print "manual cube scan started."
    retval, cube = scan_cube()
    print "scan result: %s (%ss)" %(cube, len(cube))
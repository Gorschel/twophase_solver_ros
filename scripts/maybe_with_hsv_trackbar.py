import cv2
import cv2 as cv
import nothing as nothing
import numpy as np

file_path = ".\\images"
custom_name = '\scan_Color.'
c_color = ['U', 'R', 'F', 'D', 'L', 'B']

# trackbar callback function to update HSV value
def nothing(x):
    pass


def image_processing():
    data_arr = np.zeros(shape=(6, 9, 5), dtype=np.int16)
    centers = np.zeros(shape=(6, 3), dtype='int')

    # Cube-faces loop
    for i in range(6):
        img = load_image(i)

        img_input = img.copy()
        img_hsv = hsv(img_input, i)

        img_morph = morphological(img_hsv, i)

        img_Contours = img.copy()
        img_lab = cv.cvtColor(img, cv2.COLOR_BGR2LAB)

        contours, hierarchy = cv.findContours(img_morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        no_sticker = 0
        for cnt in contours:
            channel_0_mean = 0
            channel_1_mean = 0
            channel_2_mean = 0

            # compute the center of the contour
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            if w > h:
                width = w
                height = h
            else:
                width = h
                height = w
            aspect_ratio = width / height
            # conditions for a sticker, if true then find a sticker
            if 500 < area < 5000 and 0.70 < aspect_ratio < 1.3 and no_sticker != 9:
                cv.rectangle(img_Contours, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw bounding box
                cv.circle(img_Contours, (cX, cY), 3, (0, 0, 0), -1)  # draw circle (center)

                pX = x
                pY = y
                # Sum of the pixel colour values for the 3 channels (for the stickers)
                for j in range(w):
                    for k in range(h):
                        channel_0_mean = channel_0_mean + img_lab[pY, pX][0]
                        channel_1_mean = channel_1_mean + img_lab[pY, pX][1]
                        channel_2_mean = channel_2_mean + img_lab[pY, pX][2]
                        # go to next pixel with increase the pixel value
                        pX += 1
                    pY += 1
                    pX = x  # reset x-pixel value, because new row

                # Mean value of the pixel colour values for the 3 channels (for the stickers)
                no_pixel = w * h
                channel_0_mean = int(round(channel_0_mean / no_pixel, 0)) * (100/255)
                channel_1_mean = int(round(channel_1_mean / no_pixel, 0)) - 128
                channel_2_mean = int(round(channel_2_mean / no_pixel, 0)) - 128
                print(x, y, channel_0_mean, channel_1_mean, channel_2_mean)

                # write the centroid-coordinates and the mean color-value of the 3 channels in an array
                if no_sticker < 10:
                    data_arr[i][no_sticker][0] = cX  # x-Pos        # data_arr[i][no_sticker][0] = cX  # x-Pos
                    data_arr[i][no_sticker][1] = cY  # y-Pos
                    data_arr[i][no_sticker][2] = channel_0_mean
                    data_arr[i][no_sticker][3] = channel_1_mean
                    data_arr[i][no_sticker][4] = channel_2_mean

                no_sticker += 1
        print(" ")
        cv.imshow("Contours: Face " + c_color[i], img_Contours)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # Sort contour centroids according to image coordinates
        data_arr[i] = sort_data(np.copy(data_arr[i]))

        # centre colours store temporarily
        centers[i] = data_arr[i][4][2:]

        if c_color[i] == 'B':
            print(data_arr)
    # i-Loop End

    # assign sticker to centre colours and change value range (U..B instead of 0..5))
    CubeDefStr = ""
    for i in range(6):
        for r in range(9):
            value = data_arr[i][r][2:]  # current sticker color info
            idx, color = find_nearest(centers, value)
            CubeDefStr += get_chars(idx)

    # return
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return_val = 0  # placeholder
    return return_val, CubeDefStr


def load_image(i):
    fspath = file_path + custom_name + c_color[i] + ".png"
    img = cv.imread(fspath, 1)
    # print(fspath)
    return img


def hsv(img_in, i):
    # create a separate window named 'controls' for trackbar
    cv2.namedWindow("marking", cv2.WINDOW_NORMAL);
    cv2.createTrackbar('H Lower', 'marking', 0, 179, nothing)
    cv2.createTrackbar('H Higher', 'marking', 179, 179, nothing)
    cv2.createTrackbar('S Lower', 'marking', 0, 255, nothing)
    cv2.createTrackbar('S Higher', 'marking', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'marking', 100, 255, nothing)
    cv2.createTrackbar('V Higher', 'marking', 255, 255, nothing)
    cv2.resizeWindow("marking", 300, 100);
    # function: looking for cube with the help of HSV colour space
    while True:
        hMin = 0
        hMax = 179
        sMin = 0
        sMax = 255
        vMin = 0
        vMax = 255

        hMin = cv2.getTrackbarPos('H Lower', 'marking')
        hMax = cv2.getTrackbarPos('H Higher', 'marking')
        sMin = cv2.getTrackbarPos('S Lower', 'marking')
        sMax = cv2.getTrackbarPos('S Higher', 'marking')
        vMin = cv2.getTrackbarPos('V Lower', 'marking')
        vMax = cv2.getTrackbarPos('V Higher', 'marking')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv_img = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_img, lower, upper)

        cv.imshow("Mask: Face " + c_color[i], mask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv.waitKey(0)
            return mask
    cv2.destroyAllWindows()


# function: morphological filter
def morphological(img_in, i):
    # perform erosion on the image
    kernel_size = 20
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erosion = cv.erode(img_in, kernel, iterations=1)
    cv.imshow("Erosion: Face " + c_color[i], img_erosion)

    # perform dilation on the image
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv.dilate(img_erosion, kernel, iterations=1)
    cv.imshow("Dilation: Face " + c_color[i], img_dilation)
    return img_dilation


# sort data_arr for rows, cols with corresponding coordinates
def sort_data(data):
    # x, y, L, a, b
    # sort rows (y)
    for r in range(9):
        for n in range(9):
            if data[r, 1] < data[n, 1] and not r == n:
                save = np.copy(data[r])
                data[r] = np.copy(data[n])
                data[n] = np.copy(save)
    # sort cols (x) (3 stickers)
    for b in range(3):
        for r in range(3):
            for n in range(3):
                if data[r + b * 3, 0] < data[n + b * 3, 0] and not r == n:
                    save = np.copy(data[r + b * 3])
                    data[r + b * 3] = np.copy(data[n + b * 3])
                    data[n + b * 3] = np.copy(save)
    return data


def find_nearest(array, value):
    """finds nearest array-value to a given scalar"""
    array = np.asarray(array)
    dif = np.abs(np.subtract(array, value))
    dev = np.sqrt(
        np.power(dif[:, 0], 2) + np.power(dif[:, 1], 2) + np.power(dif[:, 2], 2))  # deviation between centre colours
    idx = dev.argmin()  # minimum
    return idx, array[idx]


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


def main():
    ret_val, cube = image_processing()
    print("\nSan result: %s (%ss)" % (cube, len(cube)))

    # Condition for: Checking the Scan Result
    condition_1 = len(cube) == 6 * 9
    condition_2 = cube.count('U') and cube.count('R') and cube.count('F') and cube.count('D') and cube.count('L') and cube.count('B')
    if not condition_1:
        print("\nscan result has not enough characters")
    elif condition_1 and condition_2:
        print("\nCongratulation! Scan Result has exact 54 characters and all characters occur exactly 9 times.")
    elif condition_1 and not condition_2:
        print("\nScan Result has exact 54 characters but not all characters occur exactly 9 times.")


if __name__ == "__main__":
    main()
    cv.waitKey(0)

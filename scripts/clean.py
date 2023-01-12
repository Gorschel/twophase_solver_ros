import cv2, os
import numpy as np

import pathlib

from twophase_solver.enums import Color

# show images
show_image_processing = False
show_hsv = False
show_morphological = False
show_roi_cube = False

try:
    import rospkg

    rospack = rospkg.RosPack()
    fpath = rospack.get_path('twophase_solver_ros') + '/images/'
except Exception as e:
    print(e)
    fpath = '/home/student/catkin_ws/src/twophase_solver_ros/images/'

customname = 'scan_'

verbosity = False


def image_processing(i, img):
    if show_image_processing:
        if verbosity:
            cv2.imshow("Input " + str(Color(i)), img)
            
        cv2.destroyAllWindows()

    ''' Stickers (Colour) search and binarisation via HSV channel '''
    hsv_trackbar = False  # hsv trackbar activation flag
    img_input = img.copy()
    img_hsv_coloured = hsv(img_input, 'coloured ' + str(Color(i)), hsv_trackbar, 0, 179, 99, 255, 50, 255)
    img_hsv_white = hsv(img_input, "white and yellow " + str(Color(i)), hsv_trackbar, 0, 179, 0, 255, 107, 255)

    ''' bitwise OR of two images'''
    bitwiseOr = cv2.bitwise_or(img_hsv_coloured, img_hsv_white)
    if show_image_processing:
        if verbosity:
            cv2.imshow("OR: " + str(Color(i)), bitwiseOr)
            
        cv2.destroyAllWindows()

    ''' perform morphological filter (Erosion / Dilation) '''
    img_morph = morphological(bitwiseOr, i)

    return img_morph


def load_image(i):
    filepath = fpath + customname + str(Color(i)) + '.png'
    if os.path.isfile(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    else:
        raise Exception('file {} not a file'.format(filepath))
    print "image loaded: " + filepath
    return img


# function: looking for cube with the help of HSV colour space
def hsv(img_in, string, hsv_trackbar, hMin, hMax, sMin, sMax, vMin, vMax):
    """ Convert from RGB Image to HSV"""
    hsv_coloured = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    if show_hsv:
        if verbosity:cv2.imshow("HSV Image: " + string, hsv_coloured)
        # 
        cv2.destroyAllWindows()

    if hsv_trackbar:
        # create a separate window named 'controls' for trackbar
        cv2.namedWindow("marking", cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H Lower', 'marking', hMin, 179, lambda *args: None)
        cv2.createTrackbar('H Higher', 'marking', hMax, 179, lambda *args: None)
        cv2.createTrackbar('S Lower', 'marking', sMin, 255, lambda *args: None)
        cv2.createTrackbar('S Higher', 'marking', sMax, 255, lambda *args: None)
        cv2.createTrackbar('V Lower', 'marking', vMin, 255, lambda *args: None)
        cv2.createTrackbar('V Higher', 'marking', vMax, 255, lambda *args: None)
        cv2.resizeWindow("marking", 300, 100)

    cnt = 0
    # function: looking for cube with the help of HSV colour space
    while True:
        cnt += 1
        if hsv_trackbar:
            hMin = cv2.getTrackbarPos('H Lower', 'marking')
            hMax = cv2.getTrackbarPos('H Higher', 'marking')
            sMin = cv2.getTrackbarPos('S Lower', 'marking')
            sMax = cv2.getTrackbarPos('S Higher', 'marking')
            vMin = cv2.getTrackbarPos('V Lower', 'marking')
            vMax = cv2.getTrackbarPos('V Higher', 'marking')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        mask = cv2.inRange(hsv_coloured, lower, upper)
        # mask1 = cv2.bitwise_not(mask1)
        if show_hsv:
            if verbosity:cv2.imshow("Mask. " + string, mask)

        # Defining the kernel size
        kernelSize = 1
        # Performing Median Blurring and store it in numpy array "medianBlurred"
        medianBlurred_mask = cv2.medianBlur(mask, kernelSize)
        if show_hsv:
            cv2.imshow("Median blurred Mask. " + string, medianBlurred_mask)
        # 
        if cnt == 10:
            return medianBlurred_mask
        if hsv_trackbar:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                
                return medianBlurred_mask
        else:
            return medianBlurred_mask
    # cv2.destroyAllWindows()


def morphological(img_bitwise, i):
    # perform erosion on the image
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erosion = cv2.erode(img_bitwise, kernel, iterations=1)
    if show_morphological:
        if verbosity:
            cv2.imshow("Erosion: " + str(Color(i)), img_erosion)

    # perform dilation on the image
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    if show_morphological:
        if verbosity:cv2.imshow("Dilation: " + str(Color(i)), img_dilation)
        # 
        cv2.destroyAllWindows()
    return img_erosion


def find_cube_pos(i, img_input, img_cont):
    all_cont = []
    img_draw = img_input.copy()
    img_drawing = img_input.copy()

    contours, hierarchy = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_values = []
    cX, cY = 0, 0
    aspect_ratio_min = 0.9
    aspect_ratio_max = 1.4

    # this for loop is to get the area of the founded contours
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        # compute the center of the contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        if w > h:
            width = float(w)
            height = float(h)
        else:
            width = float(h)
            height = float(w)

        aspect_ratio = float(width / height)
        if aspect_ratio_min < aspect_ratio < aspect_ratio_max and len(approx) == 4:
            area_values.append(area)
            if verbosity: print(
                    "Area " + str(area) + "  All cx: " + str(cX) + " cY: " + str(cY) + " Aspect Ratio " + str(
                aspect_ratio))

    # calculate the median of the founded contours
    area_median = np.median(area_values)
    if verbosity:
        print("Area Median", area_median)

    cx_values = []
    cy_values = []

    height_values = []
    width_values = []
    no_sticker = 0
    conturen = []

    # this for loop calculate all square elements with the array size like a sticker
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        # compute the center of the contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        if w > h:
            width = float(w)
            height = float(h)
        else:
            width = float(h)
            height = float(w)

        aspect_ratio = float(width / height)
        area_c1 = area_median - (area_median * 0.25)
        area_c2 = area_median + (area_median * 0.25)
        if area_c1 < area < area_median + area_c2 and aspect_ratio_min < aspect_ratio < aspect_ratio_max and len(
                approx) == 4:  # if 500 < area < 6000 and 0.90 < aspect_ratio < 1.4 and len(approx) == 4:
            cx_values.append(cX)  # list with center x-values
            cy_values.append(cY)  # list with center y-values

            conturen.append([cX, cY, no_sticker])

            all_cont.append([x, y, w, h, cX, cY, no_sticker])

            height_values.append(h)  # list with height values
            width_values.append(w)  # list with width values

            cv2.rectangle(img_drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw bounding box
            cv2.circle(img_drawing, (cX, cY), 3, (0, 0, 0), -1)  # draw circle (center)
            cv2.putText(img_drawing, str(no_sticker), (cX - 20, cY - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (250, 250, 100), 1)
            if verbosity: print("No Sticker: " + str(no_sticker) + " cx: " + str(cX) + " cY: " + str(
                cY) + " aspectratio " + str(
                aspect_ratio))
            no_sticker += 1

    if no_sticker != 0:
        cv2.imshow("Contours with good conditions", img_drawing)

        # Sorting the x-coordinates of the contours
        result_x = sorted(conturen, key=lambda l: l[0])
        if verbosity: print("Sorted center-X coordinates: ", result_x)

        # Sorting the x-coordinates of the contours
        result_y = sorted(conturen, key=lambda l: l[1])
        if verbosity: print("Sorted center-Y coordinates: ", result_y)

        # calculate the Median of the center coordinates, width and height of the contours
        cX_median = np.median(cx_values)
        cY_median = np.median(cy_values)
        width_median = np.median(width_values)
        height_median = np.median(height_values)

        # check if the contours have neighbours in the x-coordinate
        index_contour_x = nearest_contours(result_x, cX_median, width_median, 0)
        if verbosity: print("index", index_contour_x)

        # check if the contours have neighbours in the y-coordinate
        index_contour_y = nearest_contours(result_y, cY_median, height_median, 1)
        if verbosity: print("index y", index_contour_y)

        # write the neighbour contours in an array
        result = set(index_contour_x + index_contour_y)
        if verbosity: print("Result", result)

        image_stickers_binary = np.zeros((img_draw.shape[0], img_draw.shape[1], 1), dtype=np.uint8)

        rect_sticker = []
        cx_values = []
        cy_values = []
        height_values = []
        width_values = []

        # draw the neighbour contours and get the position and size of the contours
        for lo in range(no_sticker):
            if all_cont[lo][6] in result:
                x = all_cont[lo][0]
                y = all_cont[lo][1]
                w = all_cont[lo][2]
                h = all_cont[lo][3]
                cX = all_cont[lo][4]
                cY = all_cont[lo][5]
                no_sticker = all_cont[lo][6]

                cx_values.append(cX)  # list with center x-values
                cy_values.append(cY)  # list with center y-values
                width_values.append(w)
                height_values.append(h)

                rect_sticker.append([x, y])

                # area = w * h
                if w > h:
                    width = float(w)
                    height = float(h)
                else:
                    width = float(h)
                    height = float(w)

                aspect_ratio = float(width / height)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw bounding box
                cv2.circle(img_draw, (cX, cY), 3, (0, 0, 0), -1)  # draw circle (center)
                cv2.putText(img_draw, str(no_sticker), (cX - 20, cY - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            (250, 250, 100), 1)
                if verbosity: print("No Sticker: " + str(no_sticker) + " cx: " + str(cX) + " cY: " + str(
                    cY) + " Aspect Ratio " + str(
                    aspect_ratio))

                cv2.rectangle(image_stickers_binary, (x, y), (x + w, y + h), 255, 2)  # draw bounding box

        if verbosity:cv2.imshow("Neighbour Contours Image", img_draw)
        # cv2.imshow("Neighbour Contours Binary Image", image_stickers_binary)

        cx_median = np.median(cx_values)
        cy_median = np.median(cy_values)

        if verbosity:
            print("cx median", cx_median)
            print("cy median", cy_median)
            
        # cv2.destroyAllWindows()
        return cx_values, cy_values, width_values, height_values, image_stickers_binary, rect_sticker


def nearest_contours(result_i, c_m, wh_m, array_element):
    index = []
    length_result = len(result_i)
    c_plus = c_m + (2.5 * wh_m)
    c_minus = c_m - (2.5 * wh_m)
    for index_image in range(length_result):
        # looking for distance between current and next contour
        if length_result - 1 != index_image:
            contour_current = result_i[index_image][array_element]
            contour_next = result_i[index_image + 1][array_element]
            diff_next_current = contour_next - contour_current
            current_index = int(result_i[index_image][2])

            condition_1 = diff_next_current < wh_m * 2.0
            condition_2 = c_minus < contour_current < c_plus
        # looking for distance between current and previous contour, for example for the last sticker
        else:
            contour_current = result_i[index_image][array_element]
            contour_prev = result_i[index_image - 1][array_element]
            current_index = int(result_i[index_image][2])
            diff_last_prev = contour_current - contour_prev

            condition_1 = diff_last_prev < wh_m * 2.0
            condition_2 = c_minus < contour_current < c_plus

        # if conditions ok, then append the index of current contour
        if condition_1 and condition_2:
            index.append(current_index)
    return index


def roi_cube(i, img_in_bgr, image_stickers_binary, cX_values, cY_values, width_values, height_values):
    # Median of all center values
    cX_Median = np.median(cX_values)
    cY_Median = np.median(cY_values)
    if verbosity: print("Cx_median", cX_Median)
    if verbosity: print("Cy_median", cY_Median)

    # deviation of the width and height values of all contours
    cX_deviation = np.median(width_values)
    cY_deviation = np.median(height_values)
    if verbosity: print("Cx_deviation", cX_deviation)
    if verbosity: print("Cy_deviation", cY_deviation)

    # calculate the start- and end-x values of the roi cube
    factor = 2.8
    start_cube_x = int(cX_Median - (cX_deviation * factor))
    end_cube_x = int(cX_Median + (cX_deviation * factor))

    # calculate the start- and end-y values of the roi cube
    start_cube_y = int(cY_Median - (cY_deviation * factor))
    end_cube_y = int(cY_Median + (cY_deviation * factor))

    if start_cube_x < 0:
        start_cube_x = 0
    if start_cube_y < 0:
        start_cube_y = 0

    # exception handling: if the end-x or end-y values are bigger than the image coordinates
    img_in_height_end, img_in_width_end, _ = img_in_bgr.shape
    if end_cube_x > img_in_width_end:
        end_cube_x = img_in_width_end
    if end_cube_y > img_in_height_end:
        end_cube_y = img_in_height_end

    # define the roi cube
    start_point = (start_cube_x, start_cube_y)
    end_point = (end_cube_x, end_cube_y)
    color = (0, 255, 0)
    cv2.rectangle(
        img=img_in_bgr,
        pt1=start_point,
        pt2=end_point,
        color=color,
        thickness=3
    )

    if show_roi_cube and verbosity:
        cv2.imshow("Rectangle around ROI " + str(Color(i)), img_in_bgr)  # Image with drawn contours

        img_roi = img_in_bgr[start_cube_y:end_cube_y, start_cube_x:end_cube_x]
        roi = image_stickers_binary[start_cube_y:end_cube_y, start_cube_x:end_cube_x]
        cv2.imshow("ROI " + str(Color(i)), roi)
        cv2.imshow("ROI BGR: " + str(Color(i)), img_roi)


def min_max_coordinates(i, copy_img, cX_values, cY_values, width_values, height_values, rect_sticker):
    img_in_bgr = copy_img.copy()
    # calculate mean width and height values
    mean_width = np.median(width_values)
    mean_height = np.median(height_values)

    # min contour center values
    min_cX = float(min(cX_values))
    min_cY = float(min(cY_values))

    # max contour center values
    max_cX = float(max(cX_values))
    max_cY = float(max(cY_values))

    # calculate the stickers center position above min-cx, min-cy, max-cx, max-cy
    step_x = mean_width * 1.8
    step_y = mean_height * 1.8
    point = []
    stickers_rectangle = []
    for number in range(0, 9):
        if number == 0:
            point.append([min_cX, min_cY])
        if number == 1:
            point.append([min_cX + step_x, min_cY])
        if number == 2:
            point.append([max_cX, min_cY])
        if number == 3:
            point.append([min_cX, min_cY + step_y])
        if number == 4:
            point.append([min_cX + step_x, min_cY + step_y])
        if number == 5:
            point.append([max_cX, min_cY + step_y])
        if number == 6:
            point.append([min_cX, max_cY])
        if number == 7:
            point.append([min_cX + step_x, max_cY])
        if number == 8:
            point.append([max_cX, max_cY])

    # calculate the contours of the stickers
    fact_width = mean_width / 4.5
    fact_height = mean_height / 4.5
    for number in range(0, 9):
        stickers_rectangle.append([round(point[number][0] - fact_width), round(point[number][1] - fact_height),
                                   round(point[number][0] + fact_width), round(point[number][1] + fact_height)])

    # draw the stickers in the image
    for stickers in stickers_rectangle:
        x1, y1, x2, y2 = stickers
        cv2.rectangle(img_in_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (180, 95, 180), 2)
        #if verbosity:
        cv2.imshow("Region for color samples: " + str(Color(i)), img_in_bgr)
    cv2.waitKey(0)
            
    return stickers_rectangle


def color_samples(i, copy_img, stickers, data_arr):
    # copy input image
    src_img = copy_img.copy()
    # convert image from bgr to lab
    img_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab)
    for no_sticker in range(9):
        # calculate the average color
        start_x = int(stickers[no_sticker][0])  # start x-position
        start_y = int(stickers[no_sticker][1])  # start y-position
        end_x = int(stickers[no_sticker][2])  # end x-position
        end_y = int(stickers[no_sticker][3])  # end y-position

        probe = img_lab[start_y:end_y, start_x:end_x]
        average_color_row = np.average(probe, axis=0)
        average_color = np.average(average_color_row, axis=0)

        data_arr[i][no_sticker][0] = start_x + (end_x - start_x) / 2  # x-Pos
        data_arr[i][no_sticker][1] = start_y + (end_y - start_y) / 2  # y-Pos
        data_arr[i][no_sticker][2] = average_color[0] * 100 / 255
        data_arr[i][no_sticker][3] = average_color[1] - 128
        data_arr[i][no_sticker][4] = average_color[2] - 128
    return data_arr


def sticker_colours(data_arr):
    centers = np.zeros(shape=(6, 3), dtype='int')
    # assign sticker to centre colours and change value range (U..B instead of 0..5))
    CubeDefStr = ""
    for i in range(6):
        centers[i] = data_arr[i][4][2:]
    for i in range(6):
        for r in range(9):
            value = data_arr[i][r][2:]  # current sticker color info
            idx, color = find_nearest(centers, value)
            CubeDefStr += get_chars(idx)
    
    cv2.destroyAllWindows()
    return_val = 0  # placeholder
    return return_val, CubeDefStr


def find_nearest(centers, value):
    """finds nearest array-value to a given scalar"""
    centers = np.asarray(centers)
    dif = np.abs(np.subtract(centers, value))
    dev = np.sqrt(
        np.power(dif[:, 0], 2) + np.power(dif[:, 1], 2) + np.power(dif[:, 2], 2))  # deviation between centre colours
    idx = dev.argmin()  # minimum
    return idx, centers[idx]


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
    # declaration of an empty array --> to store the Lab colour values of the stickers
    data_arr = np.zeros(shape=(6, 9, 5), dtype=np.int16)
    # Cube-faces loop
    for i in range(0, 6):
        # load current face image
        img_in = load_image(i)
        copy_img = img_in.copy()
        # perform binarisation via HSV and morphological filter
        img_morph = image_processing(i, img_in)
        # find cube position and stickers
        cX, cY, width_values, height_values, image_stickers_binary, rect_sticker = find_cube_pos(i, img_in, img_morph)
        # draw roi cube
        roi_cube(i, img_in, image_stickers_binary, cX, cY, width_values, height_values)
        # if not all stickers could be recognised --> find min and max x and y- coordinates of the stickers
        stickers = min_max_coordinates(i, copy_img, cX, cY, width_values, height_values, rect_sticker)
        cv2.destroyAllWindows()
        # taking colour samples from stickers
        data_arr = color_samples(i, copy_img, stickers, data_arr)
    # determining the colours
    return_val, CubeDefStr = sticker_colours(data_arr)
    return None, CubeDefStr


if __name__ == "__main__":
    verbosity = True
    print "manual cube scan started."
    _, cube = scan_cube()
    print(cube)
    print "scan result: %s (%ss)" % (cube, len(cube))
    

# coding=utf-8
import os
import numpy as np  # Library for numerical calculations (with alias)
import math
from collections import Counter  # Counts the number of occurrences of elements in a list
import cv2  # Module for image processing
from twophase_solver.enums import Color

try:
    import rospkg

    # Get the path to the images folder in the twophase_solver_ros package (if available)
    rospack = rospkg.RosPack()  # Initialize the rospack object
    fpath = rospack.get_path('twophase_solver_ros') + '/images/'  # Get the path to the package's images folder
except Exception as e:
    print(e)

    # If the twophase_solver_ros package is not available, use a default folder path
    fpath = '/home/student/catkin_ws/src/twophase_solver_ros/images/'
    if not os.path.exists(fpath):
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        fpath = os.path.join(parent_dir, "images")  # Get the path to the images folder


#  START Parameters ####################################################################################################

# need for input image
custom_name = 'scan_'

# default HSV-Value for Color Stickers
h_min_color = 0
h_max_color = 179
s_min_color = 99
s_max_color = 255
v_min_color = 70 #55
v_max_color = 255

# default HSV-Value for White and Yellow Stickers
h_min_white_yellow = 0  # 0
h_max_white_yellow = 179
s_min_white_yellow = 0
s_max_white_yellow = 255
v_min_white_yellow = 107
v_max_white_yellow = 255

# show images
verbosity = 0  # only if verbosity is true, the show images can be activated
show_input_image = 0
show_hsv_image = 0
show_hsv_color_detection = 0
show_blurred_image = 0
show_binary_or_image = 0
show_morphological_image = 0

destroy_windows = 0


#  END Parameters ------------------------------------------------------------------------------------------------------

def load_image(i):
    filepath = fpath + custom_name + str(Color(i)) + '.png'
    if os.path.isfile(filepath) and cv2.haveImageReader(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    else:
        raise Exception('{} is not a file'.format(filepath))
    print("image loaded: " + filepath)
    return img


def pre_processing(i, img):
    if verbosity and show_input_image:
        cv2.imshow("Input " + str(Color(i)), img)
        cv2.waitKey(0)
        if destroy_windows:
            cv2.destroyAllWindows()

    # create a copy of the input image for hsv thresholding
    img_input_hsv = img.copy()

    # Detect colored stickers in an image using HSV color space
    img_hsv_coloured = hsv(img_in=img_input_hsv,
                           string='coloured ' + str(Color(i)),
                           hsv_trackbar=show_hsv_color_detection,
                           h_min=h_min_color,
                           h_max=h_max_color,
                           s_min=s_min_color,
                           s_max=s_max_color,
                           v_min=v_min_color,
                           v_max=v_max_color)

    # Detect white and yellow stickers in an image using HSV color space
    img_hsv_white = hsv(img_in=img_input_hsv,
                        string="white and yellow" + str(Color(i)),
                        hsv_trackbar=show_hsv_color_detection,
                        h_min=h_min_white_yellow,
                        h_max=h_max_white_yellow,
                        s_min=s_min_white_yellow,
                        s_max=s_max_white_yellow,
                        v_min=v_min_white_yellow,
                        v_max=v_max_white_yellow)

    # apply median blurring on coloured stickers
    median_blurred_mask_color = blurred(5, img_hsv_coloured, 'coloured')

    # apply median blurring on white and yellow stickers
    median_blurred_mask_white_yellow = blurred(5, img_hsv_white, 'white and yellow')

    # bitwise OR of two images
    bitwise_or = cv2.bitwise_or(median_blurred_mask_color, median_blurred_mask_white_yellow)
    if verbosity:
        if show_binary_or_image:
            cv2.imshow("OR: " + str(Color(i)), bitwise_or)
            cv2.waitKey(0)
            if destroy_windows:
                cv2.destroyAllWindows()

    # perform morphological filter (Erosion / Dilation)
    img_morph = morphological(bitwise_or, i)

    return img_morph


# function: looking for cube with the help of HSV colour space
def hsv(img_in, string, hsv_trackbar, h_min, h_max, s_min, s_max, v_min, v_max):
    """ Convert from RGB Image to HSV"""
    hsv_coloured = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    if verbosity and show_hsv_image:
        cv2.imshow("HSV Image: " + string, hsv_coloured)
        cv2.waitKey(0)
        if destroy_windows:
            cv2.destroyAllWindows()

    if verbosity and hsv_trackbar:
        # create a separate window named 'controls' for trackbar
        cv2.namedWindow("marking", cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('H Lower', 'marking', h_min, 179, lambda *args: None)
        cv2.createTrackbar('H Higher', 'marking', h_max, 179, lambda *args: None)
        cv2.createTrackbar('S Lower', 'marking', s_min, 255, lambda *args: None)
        cv2.createTrackbar('S Higher', 'marking', s_max, 255, lambda *args: None)
        cv2.createTrackbar('V Lower', 'marking', v_min, 255, lambda *args: None)
        cv2.createTrackbar('V Higher', 'marking', v_max, 255, lambda *args: None)
        cv2.resizeWindow("marking", 300, 100)

    # function: looking for cube with the help of HSV colour space
    while True:
        if verbosity and hsv_trackbar:
            h_min = cv2.getTrackbarPos('H Lower', 'marking')
            h_max = cv2.getTrackbarPos('H Higher', 'marking')
            s_min = cv2.getTrackbarPos('S Lower', 'marking')
            s_max = cv2.getTrackbarPos('S Higher', 'marking')
            v_min = cv2.getTrackbarPos('V Lower', 'marking')
            v_max = cv2.getTrackbarPos('V Higher', 'marking')

        # Set minimum and maximum HSV values to display
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv_coloured, lower, upper)
        # mask1 = cv2.bitwise_not(mask1)
        if verbosity and hsv_trackbar:
            cv2.imshow("Mask. " + string, mask)
        if verbosity and hsv_trackbar:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                if destroy_windows:
                    cv2.destroyAllWindows()
                cv2.destroyWindow('marking')
                return mask
        else:
            return mask


def blurred(kernel, input_img, str_blurrred):
    # Performing Median Blurring and store it in numpy array "medianBlurred"
    median_blurred = cv2.medianBlur(input_img, kernel)
    if verbosity and show_blurred_image:
        cv2.imshow("Med. blur. Mask " + str_blurrred, median_blurred)
        cv2.waitKey(0)
        return median_blurred
    else:
        return median_blurred


def morphological(img_bitwise, i):
    # perform erosion on the image
    kernel_size = 13
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erosion = cv2.erode(img_bitwise, kernel, iterations=1)
    if verbosity:
        if show_morphological_image:
            cv2.imshow("Erosion: " + str(Color(i)), img_erosion)

    # perform dilation on the image
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    if verbosity:
        if show_morphological_image:
            cv2.imshow("Dilation: " + str(Color(i)), img_dilation)
            cv2.waitKey(0)
            if destroy_windows:
                cv2.destroyAllWindows()
    return img_erosion


def find_sticker_pos(i, img_input, img_cont):
    """
    Get sticker positions. called two times. first to get sticker location for rectifying the image,
    second to get positions again (no warping)
    """
    all_cont = []
    img_draw = img_input.copy()
    img_drawing = img_input.copy()

    contours, hierarchy = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_values = []
    cX, cY = 0, 0
    aspect_ratio_min = 0.9
    aspect_ratio_max = 1.4

    # START 1 st FOR LOOP ##############################################################################################
    for cnt in contours:
        """
            ° 1 st for loop: determines the area of the contours found
            ° with this information the median value of the areas is calculated
            
            ° Condition for a contour is:
                - aspect ratio of the contour falls within the range (aspect_ratio_min and aspect_ratio_max)
                - if the contour has exactly four points
        """
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

    # calculate the median of the founded contours
    area_median = np.median(area_values)
    if verbosity:
        print("Area Median", area_median)
    # END 1st FOR LOOP  ------------------------------------------------------------------------------------------------

    # initializing variables for 2nd for loop
    cx_values = []
    cy_values = []

    height_values = []
    width_values = []
    no_sticker = 0
    conturen = []

    # Calculates the range of the area of the contours based on the median area value
    area_min = area_median - (area_median * 0.3)  # value from 0.25 to 0.3
    area_max = area_median + (area_median * 0.3)  # value from 0.25 to 0.3

    contours2, hierarchy = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # START 2nd for loop ###############################################################################################
    for cnt in contours2:
        """
           ° 2nd for loop: determines all contours which are potential stickers

           ° Condition for a contour is:
               - aspect ratio of the contour falls within the range (aspect_ratio_min and aspect_ratio_max)
               - if the contour has exactly four points
               - NEW CONDITION for 2nd for loop:
                 ° if the area of the contour falls within the range defined by area_min and area_max.
       """
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

        if area_min < area < area_max and aspect_ratio_min < aspect_ratio < aspect_ratio_max and len(
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

            no_sticker += 1
    # END 2nd for loop -------------------------------------------------------------------------------------------------

    # check if enough stickers can be detected
    if no_sticker >= 5:
        if verbosity:
            print("\n{} stickers found. sufficient: ".format(no_sticker))
        # cv2.imshow("Contours with good conditions", img_drawing)

        # Sorting the x-coordinates of the contours
        result_x = sorted(conturen, key=lambda l: l[0])
        if verbosity:
            print("Sorted center-X coordinates: ", result_x)
        # Sorting the x-coordinates of the contours
        result_y = sorted(conturen, key=lambda l: l[1])
        if verbosity:
            print("Sorted center-Y coordinates: ", result_y)

        # calculate the Median of the center coordinates, width and height of the contours
        cX_median = np.median(cx_values)
        cY_median = np.median(cy_values)
        width_median = np.median(width_values)
        height_median = np.median(height_values)

        # check if the contours have neighbours in the x-coordinate
        index_contour_x = nearest_contours(result_x, cX_median, width_median, 0)
        if verbosity:
            print("index", index_contour_x)

        # check if the contours have neighbours in the y-coordinate
        index_contour_y = nearest_contours(result_y, cY_median, height_median, 1)
        if verbosity:
            print("index y", index_contour_y)

        # write the neighbour contours in an array
        result = set(index_contour_x + index_contour_y)
        if verbosity:
            print("Result", result)

        image_stickers_binary = np.zeros((img_draw.shape[0], img_draw.shape[1], 1), dtype=np.uint8)

        rect_sticker = []
        cx_values = []
        cy_values = []
        height_values = []
        width_values = []

        #no_sticker = no_sticker

        # draw the neighbour contours and get the position and size of the contours
        for sticker_id in range(no_sticker):
            if all_cont[sticker_id][6] in result:
                x = all_cont[sticker_id][0]
                y = all_cont[sticker_id][1]
                w = all_cont[sticker_id][2]
                h = all_cont[sticker_id][3]
                cX = all_cont[sticker_id][4]
                cY = all_cont[sticker_id][5]
                no_sticker = all_cont[sticker_id][6]

                cx_values.append(cX)  # list with center x-values
                cy_values.append(cY)  # list with center y-values
                width_values.append(w)
                height_values.append(h)

                rect_sticker.append([x, y])

            # draw found stickers
            width = float(h)
            height = float(w)

            aspect_ratio = float(width / height)
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw bounding box
            cv2.circle(img_draw, (cX, cY), 3, (0, 0, 0), -1)  # draw circle (center)
            cv2.putText(img_draw, str(no_sticker), (cX - 5, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)

            if verbosity:
                print("No Sticker: " + str(no_sticker) + " cx: " + str(cX) + " cY: " + str(
                    cY) + " Aspect Ratio " + str(aspect_ratio))

            cv2.rectangle(image_stickers_binary, (x, y), (x + w, y + h), 255, 2)  # draw bounding box

            # sticker_rectangle = (x, y), (x + w, y + h) is not used


        cx_median = np.median(cx_values)
        cy_median = np.median(cy_values)

        if verbosity:
            print("cx median", cx_median)
            print("cy median", cy_median)

        # if no_sticker >= 5, check if minimum 3 stickers in rows and 3 stickers in columns is available
        min_cx_val = np.min(cx_values)
        max_cx_val = np.max(cx_values)
        min_cy_val = np.min(cy_values)
        max_cy_val = np.max(cy_values)

        mean_height = np.mean(height_values)
        mean_width = np.mean(width_values)

        diff_cx = max_cx_val - min_cx_val
        diff_cy = max_cy_val - min_cy_val

        enough_cols = min_cx_val + (2.2 * mean_width) < max_cx_val
        enough_rows = min_cy_val + (2.2 * mean_height) < max_cy_val
        if verbosity:
            if enough_cols:
                print("- Enough columns available")
            else:
                print("- Not enough columns available")

            if enough_rows:
                print("- Enough rows available\n")
            else:
                print("- Not enough rows available\n")

        if enough_cols and enough_rows:
            return True, cx_values, cy_values, width_values, height_values
        else:
            return False, cx_values, cy_values, width_values, height_values
    else:
        print("Not Enough sticker {}".format(no_sticker))
        return False, cx_values, cy_values, width_values, height_values


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


def get_tilt(cY, cX, w_val, h_val):
    """
    sums smallest point distances in x, y directions for all points
    returns tilt-per-sticker in pixels
    """
    tilt_y, tilt_x, step = 0, 0, 0

    # get point list
    pts = []
    for i in range(len(cY)):
        x, y = cX[i], cY[i]
        pts.append([x, y])

    # iterate every point distance
    dx1, dy1, sx1, sy1 = [], [], [], []
    for x1, y1 in pts:
        dx2, dy2, sx2, sy2 = [], [], [], []
        for x2, y2 in pts:
            if (x1, y1) != (x2, y2):
                diff_x = x1 - x2 if y2 > y1 else x2 - x1
                diff_y = y1 - y2 if x2 > x1 else y2 - y1
                dx2.append(diff_x)
                dy2.append(diff_y)
        mdx = np.mean([d for d in dx2 if abs(d) <= w_val / 2])
        mdy = np.mean([d for d in dy2 if abs(d) <= h_val / 2])
        sx = np.median([d for d in dx2 if w_val / 2 < abs(d) < w_val * 2.5])
        sy = np.median([d for d in dy2 if abs(d) > h_val / 2])
        dx1.append(mdx)
        dy1.append(mdy)
        sx1.append(abs(sx))
        sy1.append(abs(sy))
    tilt_x = np.mean(dx1)
    tilt_y = np.mean(dy1)
    step_x = np.mean([d for d in sx1 if w_val / 2 < abs(d) < w_val * 3])
    step_y = np.mean([d for d in sy1 if h_val / 2 < abs(d) < h_val * 3])

    return tilt_y, tilt_x, step_y, step_x


def reconstruct(i, copy_img, cX_values, cY_values, width_values, height_values, step_x,
                delta_y, step_y):
    img_contours_all = copy_img.copy()
    img_contours_sample = copy_img.copy()
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
    # step_x = mean_width * 2
    # step_y = mean_height * 2
    points = []
    stickers_all = []
    stickers_sample = []

    # tilt_y, tilt_x, _, _ = get_tilt(cY_values, cX_values, mean_width, mean_height)
    # tilt_x = step_x
    tilt_y = delta_y
    for y in range(0, 3):
        for x in range(0, 3):
            # TODO: use actual center distance, not size of sticker as step size
            a = int(float(min_cX) + float(step_x) * x)
            b = int(float(min_cY) + float(step_y) * y)
            points.append([a, b])

    color_val = 0
    new_centers = []
    for coordinate in points:
        index = 0
        x_res = 0
        y_res = 0
        for index in range(len(cX_values)):
            # define a point
            point = (coordinate[0], coordinate[1])

            w = width_values[index]
            h = height_values[index]
            cx = cX_values[index]
            cy = cY_values[index]

            # Check if the point is inside the rectangle
            if cx - (w / 2) <= point[0] <= cx + (w / 2) and cy - (h / 2) <= point[1] <= cy + (h / 2):
                if verbosity:
                    print("Point is inside the rectangle")
                x_res = cx
                y_res = cy
                color_val = 0
                break
            else:
                x_res = point[0]
                y_res = point[1]
                color_val = 1

            index = index + 1
            # result = cv2.pointPolygonTest(contour, point, True)
        new_centers.append([x_res, y_res, color_val])

        # calculate the bounding rectangle of the stickers
        fact_width1 = mean_width / 2.0
        fact_height1 = mean_height / 2.0

        # calculate the bounding rectangle of the samples
        fact_width2 = mean_width / 4.5
        fact_height2 = mean_height / 4.5

    for number in range(0, 9):
        stickers_all.append(
            [round(new_centers[number][0] - fact_width1), round(new_centers[number][1] - fact_height1),
             round(new_centers[number][0] + fact_width1), round(new_centers[number][1] + fact_height1)])

        stickers_sample.append(
            [round(new_centers[number][0] - fact_width2), round(new_centers[number][1] - fact_height2),
             round(new_centers[number][0] + fact_width2), round(new_centers[number][1] + fact_height2)])

    index = 0
    for coordinates in new_centers:
        # Bounding Boxes for all contours
        if new_centers[index][2] == 0:
            color = (255, 0, 255)
        else:
            color = (110, 110, 120)

        x1, y1, x2, y2 = stickers_all[index]
        cv2.rectangle(img_contours_all, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img_contours_all, str(index), (coordinates[0], coordinates[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)
        cv2.circle(img_contours_all, (coordinates[0], coordinates[1]), 3, (0, 0, 0), -1)  # draw circle (center)
        if verbosity:
            cv2.imshow("All Stickers: " + str(Color(i)), img_contours_all)
            # cv2.waitKey(0)  # LOSCHEN
        # cv2.imshow("All Stickers: " + str(Color(i)), img_contours_all)
        # Bounding Boxes for samples
        x1, y1, x2, y2 = stickers_sample[index]
        cv2.rectangle(img_contours_sample, (int(x1), int(y1)), (int(x2), int(y2)), (238, 28, 247), 2)
        if verbosity:
            cv2.imshow("Stickers Sample: " + str(Color(i)), img_contours_sample)
            # cv2.waitKey(0)  # LOSCHEN
        # cv2.imshow("Stickers Sample: " + str(Color(i)), img_contours_sample)
        index += 1
    cv2.waitKey(0)
    return stickers_sample, img_contours_all, img_contours_sample


def colour_samples(i, copy_img, stickers, data_arr):
    # convert image from bgr to lab
    img_lab = cv2.cvtColor(copy_img, cv2.COLOR_BGR2Lab)
    zeile = 0
    for no_sticker in range(9):
        # calculate the average color
        start_x = int(stickers[no_sticker][0])  # start x-position
        start_y = int(stickers[no_sticker][1])  # start y-position
        end_x = int(stickers[no_sticker][2])  # end x-position
        end_y = int(stickers[no_sticker][3])  # end y-position

        probe = img_lab[start_y:end_y, start_x:end_x]
        average_colors = np.average(probe, axis=(0, 1))

        data_arr[i][no_sticker][0] = start_x + (end_x - start_x) / 2  # x-Pos
        data_arr[i][no_sticker][1] = start_y + (end_y - start_y) / 2  # y-Pos
        data_arr[i][no_sticker][2] = average_colors[0]  # * 100 / 255  # L
        data_arr[i][no_sticker][3] = average_colors[1] - 128  # a
        data_arr[i][no_sticker][4] = average_colors[2] - 128  # b
        zeile += 1
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
    return_val = 0  # placeholder
    return return_val, CubeDefStr


def find_nearest(centers, value):
    """finds nearest array-value to a given scalar"""
    centers = np.asarray(centers)
    dif = np.abs(np.subtract(centers, value))
    dev = np.sqrt(
        # np.power(dif[:, 0], 2) + np.power(dif[:, 1], 2) + np.power(dif[:, 2], 2))  # deviation between all centre colors
        np.power(dif[:, 1], 2) + np.power(dif[:, 2], 2))  # deviation between AB centre colors
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


def slope_contours(cX, cY, h_val, w_val):
    """
    calculate cube rotation angle.
    returns: angle, slope_x, delta_y, slope_y
    """
    h_median = np.median(h_val)
    w_median = np.median(w_val)

    # get point list
    pts = []
    for k in range(len(cY)):
        x, y = cX[k], cY[k]
        pts.append([x, y])

    angle_arr = []

    dy_arr = []
    sx_arr = []
    sy_arr = []
    # iterate every point distance
    for x1, y1 in pts:
        buffer_dy = []
        buffer_sx = []
        buffer_sy = []
        no_neighbour_x = True
        no_neighbour_y = True
        for x2, y2 in pts:
            if (x1, y1) != (x2, y2):
                if abs(y2 - y1) < int(h_median):
                    if x2 - x1 > 0:
                        no_neighbour_x = False  # if no neighbour in increasing x-direction
                        # Calculate angle
                        delta_y = abs(y2 - y1)
                        step_x = abs(x2 - x1)

                        buffer_dy.append(delta_y)
                        buffer_sx.append(step_x)

                        angle = math.atan2(delta_y, step_x)
                        angle_arr.append(-1 * math.degrees(angle))

                if abs(x2 - x1) < int(w_median):
                    if y2 - y1 > 0:
                        no_neighbour_y = False  # if no neighbour in increasing y-direction
                        step_y = abs(y2 - y1)
                        buffer_sy.append(step_y)

        if not no_neighbour_x:
            minindex = np.argmin(buffer_sx)  # get the index of the nearest neighbour difference in x-coordinate
            sx_arr.append(buffer_sx[minindex])
            dy_arr.append(buffer_dy[minindex])

        if not no_neighbour_y:
            minindex = np.argmin(buffer_sy)  # get the index of the nearest neighbour difference in x-coordinate
            sy_arr.append(buffer_sy[minindex])

    sx = np.median(sx_arr)  # Median of the step x
    dy = np.median(dy_arr)  # Median of the delta y

    sy = np.median(sy_arr)
    angle = np.median(angle_arr)
    if verbosity:
        print(angle)

    if math.isnan(sx):
        sx = np.max(cX) - np.min(cX)

    if math.isnan(sy):
        sy = np.max(cY) - np.min(cY)

    if math.isnan(dy):
        dy = 0

    if math.isnan(angle):
        angle = 0.0

    if sx > w_median * 2.3:
        sx = sx * 0.5

    if sy > h_median * 2.3:
        sy = sy * 0.5

    return angle, sx, dy, sy


def affine_transformation(angle_sticker, i, img_in, img_morph):
    (h, w) = img_in.shape[:2]
    center = (w / 2, h / 2)
    angle = angle_sticker# * 0.9
    scale = 1
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_bgr = cv2.warpAffine(img_in, M, (w, h))
    rotated_morph = cv2.warpAffine(img_morph, M, (w, h))
    return rotated_bgr, rotated_morph


def check_def_str(defstr, img_arr):
    if not all(defstr.count(letter.name) == 9 for letter in Color):
        for i, (img_reconstruct, img_samples) in enumerate(img_arr):
            c = Color(i).name
            # img_reconstruct, img_samples = img_tuple
            n1 = "Reconstructed centers for {} side.".format(c)
            n2 = "Sample Regions for {} side.".format(c)
            cv2.imshow(n1, img_reconstruct)
            cv2.imshow(n2, img_samples)
            cv2.moveWindow(n1, i * 500, 0)
            cv2.moveWindow(n2, i * 500, 200)
            if i == 5:
                print('Error. letters not evenly distributed')
                cv2.destroyAllWindows()
                return False
    return True


def scan_cube():
    """
    returns: retval, CubeDefStr
    """
    data_arr = np.zeros(shape=(6, 9, 5), dtype=np.int16)
    # Cube-faces loop
    final_imgs = []
    for i in range(0, 6):
        # load current face image
        img_in = load_image(i)

        # perform binarisation via HSV and morphological filter
        img_morph = pre_processing(i, img_in)

        '''
        1st call of find_sticker_pos: 
            - find cube position and stickers
            - needed to determine the rotation of the cube
        '''
        enough_stickers, cX, cY, width_values, height_values = find_sticker_pos(i, img_in, img_morph)

        if enough_stickers:
            if verbosity:
                print("Sticker amount ok")
            # find the slope of the cube with the calculation of the slope sticker
            angle_sticker, _, _, _ = slope_contours(cX, cY, height_values, width_values)

            img_rotated, img_morph_rotated = affine_transformation(angle_sticker, i, img_in, img_morph)

            '''
            2nd call of find_sticker_pos: 
                - find cube position and stickers
            '''
            enough_stickers, cX, cY, width_values, height_values = find_sticker_pos(
                i,
                img_rotated,
                img_morph_rotated)

            angle_sticker, step_x, delta_y, step_y = slope_contours(cX, cY, height_values, width_values)

            stickers, img_cnts_all, img_cnts_sample = reconstruct(i, img_rotated, cX, cY, width_values, height_values,
                                                                  step_x, delta_y, step_y)

            cv2.destroyAllWindows()

            final_imgs.append((img_cnts_all, img_cnts_sample))
            if destroy_windows:
                cv2.destroyAllWindows()
            # taking colour samples from stickers
            data_arr = colour_samples(i, img_rotated, stickers, data_arr)
        else:
            break

    # determining the sticker colours
    _, CubeDefStr = sticker_colours(data_arr)
    retval = check_def_str(CubeDefStr, final_imgs)

    cv2.waitKey(0)
    if destroy_windows:
        cv2.destroyAllWindows()
    return retval, CubeDefStr


if __name__ == "__main__":
    # verbosity = True
    print("manual cube scan started.")

    # image processing
    retval, cube = scan_cube()

    if retval:
        print("scan result: %s (%ss)" % (cube, len(cube)))
    else:
        print("False string {}".format(cube))

    # Outputs the count of each letter in the "cube" string
    printstr = ''
    for letter, count in Counter(cube).items():
        printstr += "{}: {}".format(letter, count)
        printstr += '  '
    print printstr

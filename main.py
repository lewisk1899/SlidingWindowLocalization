import argparse
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def estimate_sliding_window_size(distance, focal_length):
    # we want to estimate the screen size based off of the claimed distance, the real screen size is
    actual_screen_dimensions = [.1524, .1143]  # in the form width and height
    image_dimensions = [2532, 1170]  # in pixels width and height
    # distance to object = focal length * real object height * image height
    # image width or height = distance to object/(focal length * real object width or height)
    # https://www.scantips.com/lights/subjectdistance.html
    object_width_on_sensor = (focal_length * actual_screen_dimensions[0]) / distance
    object_height_on_sensor = (focal_length * actual_screen_dimensions[1]) / distance

    # object width or height on sensor = sensor height*object height or width/sensor height

    return int(object_width_on_sensor * image_dimensions[0]), int(object_height_on_sensor * image_dimensions[1])


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


# def difference(sliding_window, image):
#     np.subtract(sliding_window, image)


def main():
    print(os.getcwd() + '/screendata/')
    file_name = os.path.join(os.path.dirname(os.getcwd() + '/screendata/'), 'white.jpg')
    assert os.path.exists(file_name)

    object_image_width, object_image_height = estimate_sliding_window_size(1, .004)

    image = cv2.imread(file_name)
    cv2.imshow("Original image", image)
    cv2.waitKey(0)
    print('Resized Dimensions:', str(object_image_width), str(object_image_height))
    resized_image = cv2.resize(image, (int(object_image_width), int(object_image_height)))
    cv2.imshow("Resized image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# main()

def test():
    (winW, winH) = (500, 320)

    image = cv2.cvtColor(cv2.imread("C:/Users/lwk18/PycharmProjects/SlidingWindow/SlidingWindowLocalization/screendata/test.PNG"),cv2.COLOR_BGR2GRAY)
    white = cv2.cvtColor(cv2.imread("C:/Users/lwk18/PycharmProjects/SlidingWindow/SlidingWindowLocalization/screendata/white.jpg"),cv2.COLOR_BGR2GRAY)
    # resize the images
    white_resized = cv2.resize(white, (winW, winH))
    max_white = 0
    white_coord = [0, 0]

    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # save the x and y that maximizes the ssim between the sliding window and the image we are scanning through
        crop_img = image[y:y + winH, x:x + winW]
        # red green blue white
        ssim_value = ssim(crop_img, white_resized)

        if ssim_value > max_white:
            max_white = ssim_value
            white_coord = [y, x]

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)

    if max_white > .70:
        cv2.rectangle(clone, (white_coord[1], white_coord[0]), (white_coord[1] + winW, white_coord[0] + winH), (0, 255, 0), 2)


    cv2.imshow("Window", clone)
    cv2.waitKey(0)

test()

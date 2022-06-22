import argparse
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import os


def estimate_sliding_window_size(distance, focal_length):
    # we want to estimate the screen size based off of the claimed distance, the real screen size is
    actual_screen_dimensions = [.1524, .1143]  # in the form width and height
    image_dimensions = [2532, 1170]  # in pixels width and height
    # distance to object = focal length * real object height * image height
    # image width or height = distance to object/(focal length * real object width or height)
    #https://www.scantips.com/lights/subjectdistance.html
    object_width_on_sensor = (focal_length * actual_screen_dimensions[0]) / distance
    object_height_on_sensor = (focal_length * actual_screen_dimensions[1]) / distance

    #object width or height on sensor = sensor height*object height or width/sensor height

    return int(object_width_on_sensor * image_dimensions[0]), int(object_height_on_sensor * image_dimensions[1])


def sliding_window(image, stride, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


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

main()
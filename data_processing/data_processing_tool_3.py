from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Summary of the file
This tool is used to do image argumentation, basically just rotate the
pictures. In this project, only image rotation is used as a image argumentation.
"""
IMAGE_READING_PATH = '../dataset/new_training_image/'
IMAGE_SAVING_PATH = '../dataset/new_training_label/'
LABEL_READING_PATH = '../dataset/new_training_image/'
LABEL_SAVING_PATH = '../dataset/new_training_label/'


def rotate_image(saving_path, file_full_name):
    img = Image.open(IMAGE_READING_PATH + file_full_name)
    file_name=file_full_name.split('.')[0]
    # rotate image 90 degrees counterclockwise
    img_rotate_90 = img.rotate(90)
    img_rotate_90.save(saving_path + file_name+'_1.png', 'PNG')
    img_rotate_180 = img.rotate(180)
    img_rotate_180.save(saving_path + file_name + '_2.png', 'PNG')
    img_rotate_270 = img.rotate(270)
    img_rotate_270.save(saving_path + file_name + '_3.png', 'PNG')


if __name__ == '__main__':
    for file in os.listdir(IMAGE_READING_PATH):
        rotate_image(IMAGE_SAVING_PATH,file)
    for file in os.listdir(LABEL_READING_PATH):
        rotate_image(LABEL_SAVING_PATH, file)
    print("Save successfully!")

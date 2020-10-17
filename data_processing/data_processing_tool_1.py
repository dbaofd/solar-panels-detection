from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Summary of this file here.
This tool is used to crop pictures in batch.
For example, there are 30 300x300 pictures, using this tool can 
efficiently crop these pictures into 256x256 pictures.
"""
READING_PATH = '../dataset/fairfield_1_image/'
SAVING_PATH = 'dataset/test/image/saving_folder/'
CROP_WIDTH = 256
CROP_HEIGHT = 256


def crop_image(img):
    """
    Crop image into CROP_WIDTH by CROP_WIDTH picture.
    Args:
        img: numpy array.
    Return:
        cropped_image: cropped image.
    Raises:
        Wrong type exception: if img is not float32 format, bug may occur.
    """
    # When using image.fromarray(image), need to make sure image is unit8 format
    # convert image from float32 to unit8
    if str(img.dtype) != 'uint8':
        img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    width, height = img.size
    if width <= CROP_WIDTH and height <= CROP_HEIGHT:
        return img
    box_list = []
    box = (0, 0, CROP_WIDTH, CROP_HEIGHT)  # (left, top, right, bottom)
    box_list.append(box)
    cropped_image = img.crop(box_list[0])
    return cropped_image


def save_image(img, name):
    """
    Save CROP_WIDTH by CROP_WIDTH picture.
    Args:
        img: PIL.image.image format.
        name: original name of file.
    Return:
        None.
    Raises:
        None.
    """
    img.save(SAVING_PATH + name, 'PNG')


if __name__ == '__main__':
    for file in os.listdir(READING_PATH):
        img_path = READING_PATH + file
        image = plt.imread(img_path)
        image = crop_image(image)
        save_image(image, file)
    print("Save successfully!")
